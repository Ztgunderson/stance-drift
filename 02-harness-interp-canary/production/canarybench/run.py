"""Run canary cells through Inspect with both roles on one set of weights.

  .venv/bin/python production/canarybench/run.py --tones neutral,supportive,aggressor \
      --tiers base,rule --bundle train --log-dir results/db [--vector data/x.npz --dose 1]

Each tone x tier is one Inspect task; eval_set makes the grid resume-safe.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import canarybench.provider  # noqa: E402,F401  (registers steerhf)
from canarybench.task import canary  # noqa: E402
from inspect_ai import eval_set  # noqa: E402
from inspect_ai.model import get_model  # noqa: E402

SNAP = ("/home/jetson/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/"
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SNAP)
    ap.add_argument("--tones", default="neutral,supportive,aggressor")
    ap.add_argument("--tiers", default="base")
    ap.add_argument("--bundle", default="test")
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--vector", default=None)
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    common = dict(batch_size=a.batch_size, device=a.device)
    attacker = get_model(f"steerhf/{a.model}", **common)
    defender = get_model(f"steerhf/{a.model}", **common, **(
        dict(vector=a.vector, dose=a.dose, layer=a.layer) if a.vector else {}))
    tasks = [canary(tone=t, tier=r, rounds=a.rounds, bundle=a.bundle, limit=a.limit)
             for t in a.tones.split(",") for r in a.tiers.split(",")]
    ok, _ = eval_set(tasks, log_dir=a.log_dir, model=defender,
                     model_roles={"defender": defender, "attacker": attacker},
                     max_tasks=len(tasks), max_samples=a.batch_size * len(tasks),
                     retry_attempts=1, log_level="warning")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
