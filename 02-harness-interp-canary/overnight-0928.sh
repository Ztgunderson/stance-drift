#!/usr/bin/env bash
# v3 night 09-28: 9B smoke (4 rounds, 2 codes/arm) -> gate -> night plan (16 rounds).
# Detached; every step timeout-wrapped; status in results/OVERNIGHT-STATUS.md.
set -u
cd "$(dirname "$0")/production"
PY=/home/jetson/lab/benches/mats-nanda/.venv/bin/python
ST=../results/OVERNIGHT-STATUS.md
echo "## night 09-28 ($(date +%H:%M))" >> $ST
timeout 1h $PY canarybench/overnight.py --plan smoke --rounds 4 --limit 2 \
  || { echo "- $(date +%H:%M) SMOKE GATE FAILED or timed out: night plan NOT started" >> $ST; exit 2; }
timeout 9h $PY canarybench/overnight.py --plan night --rounds 16 --limit 16 \
  || echo "- $(date +%H:%M) night plan exited nonzero or hit the 9h timeout (eval_set resumes on relaunch)" >> $ST
echo "- $(date +%H:%M) queue finished" >> $ST
