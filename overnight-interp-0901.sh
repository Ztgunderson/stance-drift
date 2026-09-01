#!/usr/bin/env bash
# Q0 — 6 pm queue, 2026-09-01 (PLAN-9B-WEEK §6a2).
#   stop vLLM -> GPU replay-cache all banked 9B trials (+ logit readouts)
#   -> execute review/04-l1-preview.ipynb -> results/MORNING-STATUS.md
# Launch:  nohup ./overnight-interp-0901.sh >> results/overnight-interp-0901.log 2>&1 &
# Resume-safe: replay skips existing .npz; nbconvert re-runs cheaply.
set -uo pipefail
BENCH="$(cd "$(dirname "$0")" && pwd)"
cd "$BENCH"
STATUS=results/MORNING-STATUS.md
LOG=results/overnight-interp-0901.log
SNAP=/home/jetson/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
CACHE=microscope/cache/qwen35-9b-v1
JUPYTER=/home/jetson/lab/existing/jetson-llm/stance-drift/.venv/bin/jupyter

note() { echo "- $(date '+%H:%M') $1" >> "$STATUS"; echo "[q0] $1"; }

echo "# MORNING-STATUS — Q0 $(date '+%F %H:%M')" > "$STATUS"
echo "" >> "$STATUS"

# 1. stop vLLM (replay needs the memory)
if timeout 90 docker stop pleasing-qwen35-9b; then
  note "vLLM stopped"
else
  note "WARN: docker stop failed/timed out — checking if replay can proceed anyway"
fi
sleep 5

# 2. GPU replay over all banked 9B trials
note "replay start (72 trials, GPU)"
if timeout 14400 env HF_HUB_OFFLINE=1 .venv/bin/python production/driftlab/replay.py \
    --model "$SNAP" --log-dir results-v1/qwen3.5-9b --out "$CACHE" --device cuda; then
  note "replay done: $(ls "$CACHE"/*.npz 2>/dev/null | wc -l) npz cached"
else
  note "ERROR: replay exited nonzero — $(ls "$CACHE"/*.npz 2>/dev/null | wc -l) npz so far; see $LOG"
fi

# 3. execute the preview notebook against whatever cache exists
if [ -f review/04-l1-preview.ipynb ]; then
  if timeout 3600 "$JUPYTER" nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name=mats-bench \
      --ExecutePreprocessor.timeout=3000 review/04-l1-preview.ipynb; then
    note "preview notebook executed: http://100.76.200.13:8890/lab/tree/review/04-l1-preview.ipynb"
  else
    note "ERROR: preview notebook execution failed — open it anyway, partial outputs may exist"
  fi
fi

# 4. wrap up
{
  echo ""
  echo "## Review links"
  echo "- Inbox: http://100.76.200.13:8890/lab/tree/review/inbox/INDEX.md"
  echo "- Preview notebook: http://100.76.200.13:8890/lab/tree/review/04-l1-preview.ipynb"
  echo "- Pilots: results/2026-09-01-pilots.md"
  echo "- Cache: $CACHE ($(ls "$CACHE"/*.npz 2>/dev/null | wc -l) trials)"
  echo ""
  echo "O0 (expansion) is NOT auto-launched — run ./overnight-O0-expand.sh after the 9 pm review."
} >> "$STATUS"
note "Q0 complete"
