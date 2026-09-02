#!/usr/bin/env bash
# Nightcap 09-01: after the evening chain signs off, re-execute the analysis
# notebooks against the 264-trial cache so morning curves are full-n.
# Launch: nohup ./nightcap-0901.sh >> results/nightcap-0901.log 2>&1 &
set -uo pipefail
BENCH="$(cd "$(dirname "$0")" && pwd)"; cd "$BENCH"
STATUS=results/MORNING-STATUS.md
JUPYTER=/home/jetson/lab/existing/jetson-llm/stance-drift/.venv/bin/jupyter
note() { echo "- $(date '+%H:%M') NIGHTCAP: $1" >> "$STATUS"; echo "[nightcap] $1"; }

for i in $(seq 1 360); do
  grep -q 'EVENING CHAIN complete' "$STATUS" 2>/dev/null && break; sleep 60
done
grep -q 'EVENING CHAIN complete' "$STATUS" || { note "chain never signed off — running anyway (CPU only)"; }

for nb in review/04-l1-preview.ipynb review/07-layer-analysis-leave-vs-leak.ipynb; do
  if timeout 10000 "$JUPYTER" nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name=mats-bench \
      --ExecutePreprocessor.timeout=9000 "$nb"; then
    note "re-executed $nb on the 264-trial cache"
  else
    note "ERROR: $nb re-execution failed (previous outputs may be partially overwritten)"
  fi
done
note "NIGHTCAP complete — morning notebooks are full-n"
