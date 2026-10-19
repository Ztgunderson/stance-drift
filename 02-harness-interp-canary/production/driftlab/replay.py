"""Replay harness-v1 transcripts through HF weights; cache residuals and
logit-based self-reports (P0 of PLAN-9B-WEEK).

Port of microscope/replay_cache.py to the tutorbench (harness-v1) schema.
For every trial in an Inspect .eval sweep:

1. Re-forward the EXACT message list the tutor saw (sample.messages is
   ground truth) and save the residual-stream vector at the END-OF-TURN
   TOKEN of every assistant turn, all layers (Gilg et al. design decision,
   PROJECT-SCOPE §3). The replay model must be the SAME weights the
   behavior came from (bf16, not a different quant).

2. Logit-based self-report readout (Martorell & Bianchi 2603.18893 style):
   the per-round state notes live in a PRIVATE channel that is not part of
   sample.messages, so each round's note context is rebuilt exactly as
   task.tutoring_rounds built it (same NOTE_SYS / _render / state_prompt),
   with the stored note as the assistant message, and the next-token
   distribution is read at each stored digit's position.

   Two documented approximations:
   - RoundRecord.note stores the PARSED dict, not the raw generation, so
     the assistant message is a canonical json.dumps re-rendering
     (`canonical_note_render` in meta). The digits and key order are the
     stored ones; byte-level spacing may differ from the original sample.
   - 1-vs-10 on a 0-10 scale: the first-digit distribution conflates 1
     and 10. P(10) = P("1")·P("0" | ...digit "1") is only readable off the
     stored path when the stored value is 1 or 10; otherwise P(10) is set
     to 0 and the (round, item) is listed in meta["ten_ambiguous"].

Output per trial: <out>/<trial-stem>.npz
  resid        float16 [n_assistant_turns, n_layers+1, d_model]
  ends         int     [n_assistant_turns]   token index of each end-of-turn
  resid_rounds int     [n_assistant_turns]   round number of each resid row
  resid_user   float16 [n_user_turns, n_layers+1, d_model]  PRE-DECISION node
                       states: end of each student turn — the state round r's
                       reply is conditioned on (Amendment 4 node regression)
  user_ends    int     [n_user_turns]
  user_rounds  int     [n_user_turns]
                       (a round whose reply was empty — pure end_chat call —
                       has no resid row, so align on this, not on position)
  report_ev    float32 [n_rounds, n_items]   E[value] per STATE item; nan
                       where the note was unparsed or a digit unreadable
                       (n_items = len(notes.STATE_ITEMS); 7 as of the
                       calculation-axis restore, NOT the 6 older docs say)
  report_probs float32 [n_rounds, n_items, 11]  value distribution 0..10
plus <trial-stem>.json meta: persona, item_id, outcome, leak_round,
leave_round, trait_before/after, per-round note dicts, ten_ambiguous list.

Usage (bench venv):
  .venv/bin/python production/driftlab/replay.py \
      --model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<x> \
      --log-dir results-v1/qwen3.5-9b \
      --out microscope/cache/qwen35-9b-v1 [--device cpu] [--limit 1]
      [--only <trial-stem>]
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tutorbench import notes as nb                       # noqa: E402
from tutorbench.task import TrialState, _render          # noqa: E402

NOTE_SYS = ("You are the tutor from the conversation. This channel "
            "is your private notebook.")
DIGITS = [str(d) for d in range(10)]


# -- loading ----------------------------------------------------------------

def eval_files(log_dir):
    fs = sorted(glob.glob(os.path.join(log_dir, "**", "*.eval"),
                          recursive=True))
    if not fs:
        sys.exit(f"no .eval files under {log_dir}")
    return fs


def load_trial(path):
    """One sample per file (sweep.py invariant). Returns (msgs, st, meta)."""
    from inspect_ai.log import read_eval_log
    log = read_eval_log(path)
    sample = (log.samples or [None])[0]
    if sample is None:
        return None, None, None
    msgs = [{"role": m.role, "content": m.text or ""}
            for m in sample.messages]
    st = sample.store_as(TrialState)
    meta = {
        "trial": os.path.splitext(os.path.basename(path))[0],
        "persona": st.persona, "item_id": st.item_id,
        "outcome": st.outcome,
        "leak_round": st.leak_round, "leave_round": st.leave_round,
        "trait_before": st.trait_before, "trait_after": st.trait_after,
        "rounds": [{"round": r.round, "note": r.note,
                    "called_end_chat": r.called_end_chat}
                   for r in st.rounds],
        "canonical_note_render": True,
        "ten_ambiguous": [],   # filled by score_reports
    }
    return msgs, st, meta


# -- residual cache ---------------------------------------------------------

def assistant_end_indices(tokenizer, msgs):
    """Token index of the LAST CONTENT TOKEN of each assistant turn, plus
    the round number each index belongs to.

    Template-agnostic: render the whole conversation once, locate each
    message's content span in the rendered string (sequential find, so
    repeated text can't mismatch), and map the last character of each
    assistant reply to its token via the offset mapping. Prefix-rendering
    is NOT used — Qwen's template renders middle assistant turns
    differently from last ones, so prefix lengths lie. Caveat inherited
    from microscope/replay_cache.py: an injected message that duplicates
    earlier text verbatim could shift the find cursor.

    Rounds are counted by USER messages (the scripted student turns);
    system messages — the tutor system prompt and any Arm-1 reminders —
    do not advance the round. An assistant turn with empty content (pure
    end_chat tool call) gets no resid row; align via resid_rounds.
    """
    text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=False)
    enc = tokenizer(text, add_special_tokens=False,
                    return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    ends, rounds_of, pos, rnd = [], [], 0, 0
    u_ends, u_rounds = [], []          # last content token of each USER turn:
    for i, m in enumerate(msgs):       # the PRE-DECISION node state for round r
        if m["role"] == "user":
            rnd += 1
        content = (m["content"] or "").strip()
        if not content:
            continue
        j = text.find(content, pos)
        if j < 0:
            raise RuntimeError(
                f"message {i} ({m['role']}) content not found in rendered "
                "template — template transforms content, needs a look")
        pos = j + len(content)
        if m["role"] in ("assistant", "user"):
            char_end = pos - 1
            tok_i = next(k for k, (a, b) in enumerate(offs)
                         if a <= char_end < b)
            if m["role"] == "assistant":
                ends.append(tok_i)
                rounds_of.append(rnd)
            else:
                u_ends.append(tok_i)
                u_rounds.append(rnd)
    if not ends:
        raise RuntimeError("no assistant turns located")
    return ids, ends, rounds_of, u_ends, u_rounds


# -- logit self-report readout ----------------------------------------------

def canonical_note(note):
    """Stored parsed dict -> canonical assistant JSON (STATE key order)."""
    body = {k: int(note[k]) for k in nb.STATE_ITEMS}
    body["note"] = str(note.get("note", ""))
    return json.dumps(body)


def note_context(rounds, upto):
    """(system, user, assistant) messages for round `upto`'s note channel,
    mirroring task.tutoring_rounds exactly."""
    convo = []
    for r in rounds:
        if r.round > upto:
            break
        reply = r.reply or f"[ended the chat: {r.end_chat_reason}]"
        convo += [("them", r.student), ("me", reply)]
    target = next(r for r in rounds if r.round == upto)
    return [{"role": "system", "content": NOTE_SYS},
            {"role": "user",
             "content": "The conversation so far:\n\n" + _render(convo)
                        + "\n\n" + nb.state_prompt()},
            {"role": "assistant", "content": canonical_note(target.note)}]


def render_chat(tokenizer, msgs):
    """Chat-template render; enable_thinking=False to match the harness's
    generation kwargs where the template supports it."""
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)


def digit_positions(tokenizer, text, note):
    """Locate each STATE item's stored value digit in the rendered text.

    Returns (enc, items) where items[k] = dict(tok=first-digit token index,
    value=stored int, candidates=[10 single-token ids for '0'..'9' in this
    token's local form], second_tok=token index of the '0' of a stored 10,
    cont_id=token id that '0' takes right after the first digit) — any item
    that can't be resolved to clean single-token candidates maps to None.
    """
    enc = tokenizer(text, add_special_tokens=False,
                    return_offsets_mapping=True)
    offs = enc["offset_mapping"]

    def tok_at(char_pos):
        return next(k for k, (a, b) in enumerate(offs)
                    if a <= char_pos < b)

    items, pos = {}, 0
    for key in nb.STATE_ITEMS:
        v = int(note[key])
        anchor = f'"{key}": '
        j = text.find(anchor + str(v), pos)
        if j < 0:
            items[key] = None
            continue
        pos = j + len(anchor)
        t = tok_at(pos)
        a, b = offs[t]
        tok_txt = text[a:b]
        rel = pos - a
        cands = []
        for d in DIGITS:
            cand_ids = tokenizer(tok_txt[:rel] + d + tok_txt[rel + 1:],
                                 add_special_tokens=False)["input_ids"]
            if len(cand_ids) != 1:
                cands = None
                break
            cands.append(cand_ids[0])
        if cands is None:
            items[key] = None
            continue
        second_tok = None
        if v == 10:
            t2 = tok_at(pos + 1)
            second_tok = t2 if t2 != t else None
            if second_tok is None:      # fused "10" token — bail to nan
                items[key] = None
                continue
        cont = tokenizer("0", add_special_tokens=False)["input_ids"]
        items[key] = {"tok": t, "value": v, "candidates": cands,
                      "second_tok": second_tok,
                      "cont_id": cont[0] if len(cont) == 1 else None}
    return enc, items


def score_reports(model, tokenizer, st, meta, device):
    """report_ev [n_rounds, 6], report_probs [n_rounds, 6, 11]."""
    import torch
    keys = list(nb.STATE_ITEMS)
    n = max(r.round for r in st.rounds)
    ev = np.full((n, len(keys)), np.nan, dtype=np.float32)
    probs = np.full((n, len(keys), 11), np.nan, dtype=np.float32)

    for r in st.rounds:
        if r.note.get("_unparsed"):
            continue
        msgs = note_context(st.rounds, r.round)
        text = render_chat(tokenizer, msgs)
        enc, items = digit_positions(tokenizer, text, r.note)
        with torch.no_grad():
            out = model(torch.tensor([enc["input_ids"]], device=device),
                        use_cache=False)
            logits = out.logits[0].float()
        for ki, key in enumerate(keys):
            it = items.get(key)
            if it is None:
                continue
            p_first = torch.softmax(logits[it["tok"] - 1], dim=-1)
            p10 = p_first[it["candidates"]]
            p10 = (p10 / p10.sum()).cpu().numpy()      # renormalized 0..9
            vec = np.zeros(11, dtype=np.float32)
            vec[:10] = p10
            v = it["value"]
            if v in (1, 10) and it["cont_id"] is not None:
                # P('0' | ...'1') is on the stored path: logits at the
                # first-digit position predict the following token.
                p_cont = float(torch.softmax(
                    logits[it["tok"]], dim=-1)[it["cont_id"]])
                vec[10] = vec[1] * p_cont
                vec[1] = vec[1] * (1.0 - p_cont)
            else:
                meta["ten_ambiguous"].append([r.round, key])
            probs[r.round - 1, ki] = vec
            ev[r.round - 1, ki] = float((vec * np.arange(11)).sum())
        del out, logits
    return ev, probs


# -- CLI --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0 = all trials")
    ap.add_argument("--only", default=None, help="single trial stem")
    ap.add_argument("--no-reports", action="store_true",
                    help="skip the logit self-report pass")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    print(f"model loaded in {time.time()-t0:.0f}s on {args.device}",
          flush=True)

    files = eval_files(args.log_dir)
    if args.only:
        files = [f for f in files if args.only in os.path.basename(f)]
        if not files:
            sys.exit(f"--only {args.only}: no match")
    if args.limit:
        files = files[: args.limit]

    for nfile, f in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(f))[0]
        dst = os.path.join(args.out, stem + ".npz")
        if os.path.exists(dst):
            print(f"[{nfile}/{len(files)}] {stem} cached — skip", flush=True)
            continue
        msgs, st, meta = load_trial(f)
        if not msgs:
            print(f"[{nfile}/{len(files)}] {stem} EMPTY — skip", flush=True)
            continue
        ids, ends, rounds_of, u_ends, u_rounds = \
            assistant_end_indices(tok, msgs)
        t0 = time.time()
        with torch.no_grad():
            out = model(torch.tensor([ids], device=args.device),
                        output_hidden_states=True, use_cache=False)
        resid = np.stack(
            [torch.stack([h[0, e] for h in out.hidden_states])
                  .to(torch.float16).cpu().numpy() for e in ends])
        # pre-decision node states: end of each USER (student) turn — the
        # state the round-r reply decision is conditioned on (Amendment 4)
        resid_user = np.stack(
            [torch.stack([h[0, e] for h in out.hidden_states])
                  .to(torch.float16).cpu().numpy() for e in u_ends])
        del out

        if args.no_reports:
            n = max(r.round for r in st.rounds)
            ev = np.full((n, len(nb.STATE_ITEMS)), np.nan, np.float32)
            probs = np.full((n, len(nb.STATE_ITEMS), 11), np.nan, np.float32)
        else:
            ev, probs = score_reports(model, tok, st, meta, args.device)

        np.savez_compressed(dst, resid=resid, ends=np.array(ends),
                            resid_rounds=np.array(rounds_of),
                            resid_user=resid_user,
                            user_ends=np.array(u_ends),
                            user_rounds=np.array(u_rounds),
                            report_ev=ev, report_probs=probs)
        with open(os.path.join(args.out, stem + ".json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        print(f"[{nfile}/{len(files)}] {stem}: {len(ids)} tok, "
              f"resid {resid.shape}, reports {ev.shape} "
              f"in {time.time()-t0:.0f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
