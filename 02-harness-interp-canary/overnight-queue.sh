#!/usr/bin/env bash
# Overnight queue v2 — 2026-08-25 18:05. Gemma-4 rungs BLOCKED (arch unknown
# to local vLLM image). Sequence: ministral smoke -> (if clean) ministral full
# -> qwen3.5-4b floor smoke -> morning status. Fully detached.
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

note "=== overnight v2 start ==="
note "waiting for 4B prefetch to finish (ministral needs GPU only, ok to overlap)"

note "ministral-3-14b smoke (1 pass)"
(cd "$SD" && TUTOR_REPS=1 DEADLINE=3000 ./runners/run_ministral3-14b_tutor8.sh)
if check_clean ministral-3-14b-tutor8; then
  note "ministral smoke CLEAN -> full run (12 passes)"
  (cd "$SD" && TUTOR_REPS=12 DEADLINE=12600 ./runners/run_ministral3-14b_tutor8.sh)
else
  note "ministral smoke NOT clean — skipping full run (review in morning)"
fi

while pgrep -f '[d]ownload Qwen/Qwen3.5-4B' >/dev/null; do sleep 60; done
note "qwen3.5-4b floor smoke (2 passes)"
(cd "$SD" && TUTOR_REPS=2 DEADLINE=3600 ./runners/run_qwen35-4b_tutor8.sh)

{
  echo "# Overnight status — $(date '+%F %H:%M')"
  echo
  echo "Gemma-4 rungs BLOCKED: arch gemma4_unified unknown to local vLLM image."
  for d in qwen3.5-9b-tutor8 ministral-3-14b-tutor8 qwen3.5-4b-tutor8; do
    echo "- $d: $(find "$SD/results/$d" -name '*.eval' 2>/dev/null | wc -l) trials"
  done
  echo
  echo "Morning move: read ministral transcripts — the 14B threshold verdict."
} > "$BENCH/results/OVERNIGHT-STATUS.md"
note "overnight v2 complete"
