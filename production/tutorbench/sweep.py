"""Sweep driver: 3 personas x 24 items, rep-major, resumable, deadlined.

    LOCAL_BASE_URL=http://127.0.0.1:8000/v1 LOCAL_API_KEY=... \
    <sd-venv>/bin/python production/tutorbench/sweep.py \
        --model openai-api/local/qwen3.5-4b \
        --log-dir results-v1/qwen3.5-4b [--items q00,q01] [--personas neutral]
        [--reps 1] [--deadline 7200] [--max-tasks 6]

Rep-major like stance-drift's sweep_plan: a deadline stop leaves equal n
in every persona x item cell of the completed passes.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tutorbench.task import ITEMS, tb_trial  # noqa: E402
from tutorbench.script import PERSONAS       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--items", default=None, help="comma list, default all")
    ap.add_argument("--personas", default=None, help="comma list, default all")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--deadline", type=int, default=None, help="seconds")
    ap.add_argument("--max-tasks", type=int, default=6)
    a = ap.parse_args()

    from inspect_ai import eval_set

    items = a.items.split(",") if a.items else sorted(ITEMS)
    personas = a.personas.split(",") if a.personas else sorted(PERSONAS)
    for i in items:
        assert i in ITEMS, f"unknown item {i}"
    for p in personas:
        assert p in PERSONAS, f"unknown persona {p}"

    start = time.monotonic()
    for rep in range(1, a.reps + 1):
        if a.deadline and (time.monotonic() - start) > a.deadline:
            print(f">> deadline before pass {rep} — stopping with equal n",
                  flush=True)
            break
        tasks = [tb_trial(persona=p, item_id=i, rep=rep)
                 for p in personas for i in items]
        t0 = time.monotonic()
        print(f"[pass {rep}] {len(tasks)} trials "
              f"({len(personas)} personas x {len(items)} items)", flush=True)
        success, logs = eval_set(
            tasks, log_dir=f"{a.log_dir}/pass{rep:02d}", model=a.model,
            temperature=0.0, max_tasks=a.max_tasks, display="plain")
        took = time.monotonic() - t0
        print(f"[pass {rep}] done in {took/60:.1f}min "
              f"({took/max(len(tasks),1):.0f}s/trial) ok={success}", flush=True)


if __name__ == "__main__":
    main()
