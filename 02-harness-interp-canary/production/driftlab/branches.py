"""Commitment-point probing over resampled branches (chain-of-thought B).

Question: WITHIN a sampled round-r reply, at which token does the action
(leak / leave / continue) become linearly decodable from the residual stream?
All k branches of a node share an identical pre-branch state; their states
diverge token-by-token once generation begins. Position-wise probes over the
early reply tokens measure when the decision has formed — strictly BEFORE the
textual evidence of the action (the anti-circularity rule).

Precedent: Sycophantic Anchors (arXiv 2601.21183) sentence-level commitment;
Thought Anchors (arXiv 2506.19143). Design doc:
refs/commitment-probing-design.md. Preregistration: draft Amendment 5 in the
design doc — NOT yet approved; everything here is exploratory machinery.

Pipeline:
  load_branches(node_json)      node_resample output -> branch records
  branch_replay_plan(...)       messages + eligible probe token positions
  run_branch_replay(...)        GPU-GATED: forward passes, cache states
  commitment_curve(...)         per-position AUROC with bootstrap CIs

Anti-circularity invariant (tested): no probe position sits at or after the
evidence position — the first strict-root match or bare root-pair (leak), or
the tool-call rendering, which begins only after the content tokens (leave).

KNOWN LIMIT of the smoke node JSON: samples store `reply_head` truncated to
300 chars. A leak whose first root evidence lies beyond the truncation cannot
be planned (valid=False, reason="evidence_missing"). Full-batch node runs
must store complete replies.
"""

import json
import re

import numpy as np

from tutorbench.items import _PAIR_JOIN, _root_pat


# -- branch records ---------------------------------------------------------

def load_branches(node_json_path):
    """node_resample output -> flat branch records.

    Returns list of dicts: node_key, trial, round, persona, item_id, action,
    reply (NOTE: truncated reply_head in smoke-era files), reason.
    """
    d = json.load(open(node_json_path))
    out = []
    for node in d["nodes"]:
        for s in node["samples"]:
            out.append({
                "node_key": f'{node["trial"]}#r{node["round"]}',
                "trial": node["trial"], "round": node["round"],
                "persona": node["persona"], "item_id": node["item_id"],
                "action": s["action"],
                "reply": s.get("reply", s.get("reply_head", "")),
                "reason": s.get("reason"),
            })
    return out


# -- evidence location ------------------------------------------------------

def evidence_char(reply, roots):
    """Char offset of the FIRST textual evidence of a leak in `reply`.

    Conservative: the earliest start among either root's strict pattern and
    the loose bare-pair pattern — a probe at or after ANY of these reads
    partial evidence, so eligibility ends at the earliest. None if no
    pattern matches (a continue reply, or a truncated leak).
    """
    r1, r2 = roots
    pats = [_root_pat(r1), _root_pat(r2),
            re.compile(r"(?<![\d.-])" + str(r1) + _PAIR_JOIN + str(r2)
                       + r"(?!\d)|(?<![\d.-])" + str(r2) + _PAIR_JOIN
                       + str(r1) + r"(?!\d)")]
    starts = [m.start() for p in pats if (m := p.search(reply))]
    return min(starts) if starts else None


# -- replay planning --------------------------------------------------------

def _find_spans(text, msgs):
    """Sequential-find char span of each message's content in the rendered
    template (same discipline as driftlab.replay.assistant_end_indices)."""
    spans, pos = [], 0
    for m in msgs:
        content = (m["content"] or "").strip()
        if not content:
            spans.append(None)
            continue
        j = text.find(content, pos)
        if j < 0:
            raise RuntimeError(
                f"content of a {m['role']} message not found in rendered "
                "template")
        spans.append((j, j + len(content)))
        pos = j + len(content)
    return spans


def branch_replay_plan(prefix_msgs, reply, tokenizer, roots, action):
    """Messages + probe positions for one branch.

    The assistant message is rendered with content only: activations at
    content tokens are causal in the prefix+content alone, so the tool-call
    rendering (a leave's evidence, which follows the content) need not be
    reconstructed to probe pre-evidence positions.

    Returns dict:
      msgs             prefix + assistant reply message
      ids              token ids of the rendered conversation
      reply_span       (first, last) token index of reply content, inclusive
      evidence_tok     absolute token index of first leak evidence, or None
      probe_positions  absolute token indices, reply start .. evidence-1
                       (whole reply for leave/continue)
      valid, reason    False when unplannable (empty reply; leak whose
                       evidence is missing from a truncated reply)
    """
    reply = (reply or "").strip()
    msgs = list(prefix_msgs) + [{"role": "assistant", "content": reply}]
    plan = {"msgs": msgs, "ids": None, "reply_span": None,
            "evidence_tok": None, "probe_positions": [],
            "valid": True, "reason": ""}
    if not reply:
        plan["valid"], plan["reason"] = False, "empty_reply"
        return plan

    text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=False)
    enc = tokenizer(text, add_special_tokens=False,
                    return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    r_lo, r_hi = _find_spans(text, msgs)[-1]

    def tok_at(char):
        return next(k for k, (a, b) in enumerate(offs) if a <= char < b)

    t0, t1 = tok_at(r_lo), tok_at(r_hi - 1)
    plan["ids"], plan["reply_span"] = ids, (t0, t1)

    ev = evidence_char(reply, roots)
    if ev is not None:
        e_tok = tok_at(r_lo + ev)
        plan["evidence_tok"] = e_tok
        plan["probe_positions"] = list(range(t0, e_tok))
    else:
        if action == "leak":
            plan["valid"], plan["reason"] = False, "evidence_missing"
            return plan
        plan["probe_positions"] = list(range(t0, t1 + 1))
    if not plan["probe_positions"]:
        plan["valid"], plan["reason"] = False, "no_pre_evidence_positions"
    return plan


# -- GPU-gated runner -------------------------------------------------------

def run_branch_replay(model_dir, node_json, prefix_msgs_by_node, out_dir,
                      device="cuda", layers=(20,), stride=1, max_pos=256):
    """GATED: do not run while vLLM holds the GPU; launch only inside an
    approved queue window (PLAN-9B-WEEK GPU budget).

    Saves, per branch, resid at `layers` for every `stride`-th probe
    position (cap `max_pos`): <out>/<node_key>#<i>.npz with
    resid [n_pos, n_layers, d] float16, positions, action, valid.
    Cost ~ one forward pass of prefix+reply per branch (same order as a
    replay.py trial). Storage at one layer, 600-token reply: ~5 MB/branch.
    """
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=device)
    model.eval()
    for i, br in enumerate(load_branches(node_json)):
        prefix = prefix_msgs_by_node[br["node_key"]]
        plan = branch_replay_plan(prefix, br["reply"], tok,
                                  roots_for(br), br["action"])
        dst = os.path.join(out_dir, f'{br["node_key"]}#{i}.npz'
                           .replace("/", "_"))
        if os.path.exists(dst):
            continue
        if not plan["valid"]:
            np.savez(dst, valid=False, reason=plan["reason"],
                     action=br["action"])
            continue
        pos = plan["probe_positions"][::stride][:max_pos]
        with torch.no_grad():
            out = model(torch.tensor([plan["ids"]], device=device),
                        output_hidden_states=True, use_cache=False)
        resid = np.stack(
            [torch.stack([out.hidden_states[layer][0, p] for layer in layers])
                  .to(torch.float16).cpu().numpy() for p in pos])
        del out
        np.savez_compressed(dst, resid=resid, positions=np.array(pos),
                            action=br["action"], valid=True)


def roots_for(branch):
    """Roots for a branch's item — resolved via the items bundle."""
    from tutorbench.items import make_items
    table = {it["id"]: it["roots"] for it in make_items()}
    return table[branch["item_id"]]


# -- analysis ---------------------------------------------------------------

def commitment_curve(pos_scores, labels, min_per_class=4, n_boot=1000,
                     seed=0):
    """Decodability-vs-position curve.

    pos_scores: per branch, a 1-D array of probe scores at RELATIVE eligible
    positions 0..L_b-1 (position 0 = first reply token). labels: per-branch
    bool (e.g. leak=True vs leave=False). At each relative position p, AUROC
    over the branches still eligible at p, with a bootstrap CI over branches;
    NaN where either class has fewer than `min_per_class` branches.

    Returns dict: positions, auroc, lo, hi, n_pos, n_neg,
    first_committed = smallest p where the CI lower bound exceeds 0.5 at p
    AND p+1 (two consecutive positions — a persistence requirement, because
    with dozens of positions tested a single CI clearing 0.5 happens by
    chance on pure noise; caught by test_commitment_curve_noise_never_commits).
    None if the curve never separates.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, bool)
    max_len = max((len(s) for s in pos_scores), default=0)

    def auroc(s, y):
        pos, neg = s[y], s[~y]
        gt = (pos[:, None] > neg[None, :]).sum()
        eq = (pos[:, None] == neg[None, :]).sum()
        return (gt + 0.5 * eq) / (len(pos) * len(neg))

    out = {k: [] for k in ("positions", "auroc", "lo", "hi",
                           "n_pos", "n_neg")}
    first = None
    for p in range(max_len):
        idx = [i for i, s in enumerate(pos_scores) if len(s) > p]
        y = labels[idx]
        s = np.array([pos_scores[i][p] for i in idx])
        out["positions"].append(p)
        out["n_pos"].append(int(y.sum()))
        out["n_neg"].append(int((~y).sum()))
        if y.sum() < min_per_class or (~y).sum() < min_per_class:
            out["auroc"].append(np.nan)
            out["lo"].append(np.nan)
            out["hi"].append(np.nan)
            continue
        a = auroc(s, y)
        boots = []
        for _ in range(n_boot):
            b = rng.integers(0, len(y), len(y))
            if y[b].sum() and (~y[b]).sum():
                boots.append(auroc(s[b], y[b]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["auroc"].append(a)
        out["lo"].append(lo)
        out["hi"].append(hi)
    out = {k: np.array(v) for k, v in out.items()}
    lo_arr = out["lo"]
    for p in range(len(lo_arr) - 1):
        if lo_arr[p] > 0.5 and lo_arr[p + 1] > 0.5:
            first = p
            break
    out["first_committed"] = first
    return out
