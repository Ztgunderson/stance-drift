"""Dump the visible transcript of every cached trial to one JSON sidecar so
CPU analyses (text baseline, random example draws) never need inspect_ai.

Output: {trial_stem: {"persona", "item_id", "messages": [{"role", "content"}]}}
Messages are sample.messages verbatim (system + student/tutor turns + any
tool messages); the private note channel is NOT included.
"""

import glob
import json
import os
import sys


def dump(log_dirs, cache_dir, out_path):
    from driftlab.replay import load_trial
    stems = {os.path.splitext(os.path.basename(f))[0]
             for f in glob.glob(os.path.join(cache_dir, "*.json"))}
    have = {}
    if os.path.exists(out_path):
        have = json.load(open(out_path))
    n_new = 0
    for d in log_dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "*.eval"), recursive=True)):
            stem = os.path.splitext(os.path.basename(f))[0]
            if stem not in stems or stem in have:
                continue
            msgs, st, meta = load_trial(f)
            if msgs is None:
                continue
            have[stem] = {"persona": meta["persona"], "item_id": meta["item_id"],
                          "messages": msgs}
            n_new += 1
    json.dump(have, open(out_path, "w"))
    print(f"{len(have)} transcripts ({n_new} new) -> {out_path}", flush=True)
    return have


def conversation_text(msgs, upto):
    """Student + tutor text through round `upto` (rounds = user messages),
    system prompt excluded; plus cumulative tutor reply length."""
    parts, rnd, cum = [], 0, 0
    for m in msgs:
        if m["role"] == "user":
            rnd += 1
            if rnd > upto:
                break
        if m["role"] in ("user", "assistant") and (m.get("content") or "").strip():
            parts.append(m["content"].strip())
            if m["role"] == "assistant":
                cum += len(m["content"])
    return "\n".join(parts), cum


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dirs", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dump(a.log_dirs.split(","), a.cache, a.out)
