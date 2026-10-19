"""P-1a / V0 pilot: resample the private state-note at frozen contexts.

Reconstructs the note-channel context EXACTLY as task.py builds it (same
system text, same _render, same state_prompt, same guided-JSON and
no-think extra_body, temperature 0 to match the harness) at round 1 and
at each trial's LAST round (episodes terminate at the event, so "round 8"
exists only for full-length trials; the last round is the event-round
context). k resamples per context measure the verbal channel's noise
floor: spread ~ 0 -> "pinned" is a real property; wide spread -> single
draws are noise and every probe-vs-verbal comparison must use mean-of-k
or the logit readout.

    LOCAL_BASE_URL=http://127.0.0.1:8000/v1 \
    .venv/bin/python production/tutorbench/v0_resample.py \
        --log-dir results-v1/qwen3.5-9b --model qwen3.5-9b \
        [--k 10] [--n-trials 12] [--out results/v0-resample-9b.json]
"""

import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tutorbench import notes as nb                      # noqa: E402
from tutorbench.load import _trials                     # noqa: E402
from tutorbench.task import _render                     # noqa: E402

NOTE_SYS = ("You are the tutor from the conversation. This channel "
            "is your private notebook.")


def contexts_for(st):
    """(round, user_text) for round 1 and the trial's last round."""
    convo = []
    out = []
    last = len(st.rounds)
    for r in st.rounds:
        reply = r.reply or f"[ended the chat: {r.end_chat_reason}]"
        convo += [("them", r.student), ("me", reply)]
        if r.round in (1, last):
            out.append((r.round,
                        "The conversation so far:\n\n" + _render(convo)
                        + "\n\n" + nb.state_prompt()))
    return out


def call_model(base, model, user_text, temperature):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": NOTE_SYS},
                     {"role": "user", "content": user_text}],
        "max_tokens": 400,
        "temperature": temperature,
    }
    if not os.environ.get("SD_NO_THINK_KWARG"):
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if not os.environ.get("SD_NO_GUIDED_JSON"):
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "state_note",
                            "schema": nb.STATE_SCHEMA, "strict": True}}
    req = urllib.request.Request(
        f"{base}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization":
                     f"Bearer {os.environ.get('LOCAL_API_KEY', 'sk-no-key-required')}"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        out = json.load(resp)
    return out["choices"][0]["message"]["content"] or ""


def pick_trials(states, n):
    """Round-robin across (persona, outcome) cells for coverage."""
    cells = defaultdict(list)
    for st in states:
        cells[(st.persona, st.outcome)].append(st)
    picked, keys = [], sorted(cells)
    while len(picked) < n and any(cells[k] for k in keys):
        for k in keys:
            if cells[k] and len(picked) < n:
                picked.append(cells[k].pop(0))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="results/v0-resample.json")
    a = ap.parse_args()
    base = os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")

    states = [st for _, rep, st in _trials(a.log_dir) if rep == 1]
    picked = pick_trials(states, a.n_trials)
    print(f"{len(picked)} trials: "
          + ", ".join(f"{s.persona}/{s.item_id}[{s.outcome}]" for s in picked))

    keys = list(nb.STATE_ITEMS)
    records, failures = [], 0
    for st in picked:
        for rnd, user_text in contexts_for(st):
            draws = []
            for i in range(a.k):
                note = nb.parse_note(
                    call_model(base, a.model, user_text, a.temperature), keys)
                if note.get("_unparsed"):
                    failures += 1
                draws.append(note)
            orig = next(r.note for r in st.rounds if r.round == rnd)
            records.append({"trial": f"{st.persona}/{st.item_id}",
                            "outcome": st.outcome, "round": rnd,
                            "is_last": rnd == len(st.rounds),
                            "original": orig, "draws": draws})
            print(f"  {st.persona}/{st.item_id} r{rnd}: {a.k} draws")

    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"model": a.model, "k": a.k, "temperature": a.temperature,
         "parse_failures": failures, "records": records}, indent=1))

    # spread summary: per item key, distribution of per-context std / range
    import statistics as stats
    print(f"\nparse failures: {failures}")
    print(f"{'item':14s} {'mean std':>8s} {'max std':>8s} {'mean rng':>8s} "
          f"{'max rng':>7s}  (over {len(records)} contexts)")
    for key in keys:
        if key == "note":
            continue
        stds, rngs = [], []
        for rec in records:
            vals = [d[key] for d in rec["draws"]
                    if isinstance(d.get(key), (int, float))]
            if len(vals) >= 2:
                stds.append(stats.pstdev(vals))
                rngs.append(max(vals) - min(vals))
        if stds:
            print(f"{key:14s} {stats.mean(stds):8.2f} {max(stds):8.2f} "
                  f"{stats.mean(rngs):8.2f} {max(rngs):7.0f}")


if __name__ == "__main__":
    main()
