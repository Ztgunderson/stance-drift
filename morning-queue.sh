#!/usr/bin/env bash
# Morning queue — 2026-08-26. Ministral retry after root-causing the overnight
# boot failure (gpu-mem-util 0.8 needed 49GB free on unified memory; a
# concurrent HF prefetch ate the margin). Registry now pins 0.7 for ministral.
# Sequence: ministral smoke -> (if clean) ministral full 12 passes -> status.
set -uo pipefail
SD=~/lab/existing/jetson-llm/stance-drift
BENCH=~/lab/benches/mats-nanda
exec >>"$BENCH/results/overnight.log" 2>&1
note(){ echo "[$(date +%H:%M)] $*"; }

check_clean(){ # $1 = results subdir; pass if >=6 evals and 0 unparsed
  local d="$1"
  [ "$(find "$SD/results/$d" -name '*.eval' 2>/dev/null | wc -l)" -ge 6 ] || return 1
  (cd "$SD" && RD="$d" .venv/bin/python - <<'PY'
import sys, os; sys.path.insert(0, ".")
from stancedrift import analysis
try:
    df = analysis.load_sweep("results/" + os.environ["RD"])
    bad = int(df.stance.isna().sum())
    print(f"{os.environ['RD']}: {df.trial.nunique()} trials, {bad} unparsed")
    sys.exit(0 if bad == 0 else 1)
except SystemExit: raise
except Exception as e:
    print("check failed:", e); sys.exit(1)
PY
  )
}

note "=== morning queue (ministral retry, util 0.7) start ==="
note "ministral-3-14b smoke (1 pass)"
(cd "$SD" && TUTOR_REPS=1 DEADLINE=3000 ./runners/run_ministral3-14b_tutor8.sh)
if check_clean ministral-3-14b-tutor8; then
  note "ministral smoke CLEAN -> full run (12 passes)"
  (cd "$SD" && TUTOR_REPS=12 DEADLINE=12600 ./runners/run_ministral3-14b_tutor8.sh)
else
  note "ministral smoke NOT clean — review needed"
fi
docker stop pleasing-ministral3 >/dev/null 2>&1

{
  echo "# Status — $(date '+%F %H:%M') (morning queue)"
  echo
  echo "Ministral boot failure root-caused: unified-memory contention at util 0.8; registry now 0.7."
  for d in qwen3.5-9b-tutor8 ministral-3-14b-tutor8 qwen3.5-4b-tutor8; do
    echo "- $d: $(find "$SD/results/$d" -name '*.eval' 2>/dev/null | wc -l) trials"
  done
} > "$BENCH/results/OVERNIGHT-STATUS.md"
note "morning queue complete"
