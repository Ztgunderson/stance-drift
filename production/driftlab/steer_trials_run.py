"""Driver: three tiers x negations, priority-ordered cells, one model load.
  python -m driftlab.steer_trials_run --plan smoke   (3 items, 2 cells)
  python -m driftlab.steer_trials_run --plan p1-4    (priorities 1-4)
Writes results/steer/<persona>__<tier>__<negation>.json and appends to
results/MORNING-STATUS.md."""

import argparse
import json
import os
import time

from driftlab.steer_trials import (TIERS, build_direction, load_model, run_cell)

SNAP = ("/home/jetson/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/"
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
STATUS = "results/MORNING-STATUS.md"

# (persona, tier, negation) in priority order
PLANS = {
    "smoke": [("aggressor", "base", "none"), ("aggressor", "base", "N1")],
    "p1": [(p, "base", "none") for p in ("aggressor", "supportive", "neutral")],
    "p2": [(p, t, "none") for p in ("aggressor", "supportive") for t in ("noleak", "noleak_noleave")],
    "p3": [("aggressor", "base", "N1"), ("aggressor", "base", "random")],
    "p4": [("supportive", "base", "N1"), ("supportive", "base", "random")],
    "p5": [(p, "base", n) for n in ("N3", "N2") for p in ("aggressor", "supportive")],
    "p6": [("neutral", t, "none") for t in ("noleak", "noleak_noleave")] + [("neutral", "base", "N3")],
}
PLANS["p1-4"] = PLANS["p1"] + PLANS["p2"] + PLANS["p3"] + PLANS["p4"]
PLANS["p5-6"] = PLANS["p5"] + PLANS["p6"]
PLANS["all"] = PLANS["p1-4"] + PLANS["p5-6"]
PLANS["headline"] = [(p, "base", n) for p in ("aggressor", "supportive") for n in ("none", "N1", "random")]
PLANS["n3"] = [("aggressor", "base", "N3"), ("supportive", "base", "N3")]


def note(msg):
    line = f"- {time.strftime('%H:%M')} STEER: {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="smoke")
    ap.add_argument("--items", type=int, default=0, help="first N items (smoke)")
    ap.add_argument("--cache", default="microscope/cache/qwen35-9b-v1")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("--out-dir", default="results/steer")
    ap.add_argument("--model", default=SNAP)
    ap.add_argument("--no-notes", action="store_true")
    ap.add_argument("--rep", type=int, default=1, help="replicate index (>1 adds __repN to the cell file)")
    ap.add_argument("--axis-persona", default="", help="falsification: compute the persona axis from THIS persona instead of the run persona (cross-persona control)")
    ap.add_argument("--random-seed", type=int, default=0, help="seed for the random-direction control")
    ap.add_argument("--cells", default="", help="explicit cells persona/tier/neg,... (overrides --plan)")
    a = ap.parse_args()
    items = json.load(open("production/tutorbench/items-v1.json"))["items"]
    if a.items:
        items = items[: a.items]
    cells = ([tuple(c.split("/")) for c in a.cells.split(",")] if a.cells else PLANS[a.plan])
    os.makedirs(a.out_dir, exist_ok=True)
    note(f"plan={a.plan} cells={len(cells)} items={len(items)} layer={a.layer} dose={a.dose} (HF generate)")
    model, tok = load_model(a.model)
    t0 = time.time()
    for persona, tier, neg in cells:
        tag = (f"{persona}__{tier}__{neg}" + (f"__d{a.dose:g}" if neg not in ("none",) and a.dose != 1.0 else "")
               + (f"__axis{a.axis_persona}" if a.axis_persona and neg != "none" else "")
               + (f"__L{a.layer}" if a.layer != 20 and neg != "none" else "")
               + (f"__seed{a.random_seed}" if neg == "random" and a.random_seed else "")
               + (f"__rep{a.rep}" if a.rep > 1 else ""))
        out = os.path.join(a.out_dir, tag + ".json")
        direction, info = (None, None) if neg == "none" else build_direction(
            a.cache, a.layer, neg, a.axis_persona or persona, seed=a.random_seed)
        if info:
            print(f"direction {neg}/{persona}: {json.dumps(info)[:300]}", flush=True)
        neg_label = (neg + (f"@axis={a.axis_persona}" if a.axis_persona and neg != "none" else "")
                     + (f"@L{a.layer}" if a.layer != 20 and neg != "none" else "")
                     + (f"@seed{a.random_seed}" if neg == "random" and a.random_seed else ""))
        rec = run_cell(model, tok, items, persona, tier, neg_label, direction, out, dose=a.dose,
                       layer=a.layer, with_notes=not a.no_notes, log=lambda s: print(s, flush=True))
        n = rec["n_items"]
        note(f"{tag}: leak {sum(t['leak_round'] is not None for t in rec['trials'])}/{n} "
             f"leave {sum(t['leave_round'] is not None for t in rec['trials'])}/{n} "
             f"({rec['elapsed_s']}s; total {time.time()-t0:.0f}s)")
    note(f"plan {a.plan} done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
