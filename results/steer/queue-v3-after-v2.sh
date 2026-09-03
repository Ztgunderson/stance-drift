#!/usr/bin/env bash
cd /home/jetson/lab/benches/mats-nanda
R() { PYTHONPATH=production HF_HUB_OFFLINE=1 timeout 7200 .venv/bin/python -m driftlab.steer_trials_run --out-dir results/steer "$@"; }
until grep -q "queue-v2 done" results/MORNING-STATUS.md; do sleep 60; done
# generalisation: one generic pressure axis (mean of aggressor and supportive axes) on both personas
R --cells aggressor/base/N2,supportive/base/N2 > results/steer/N2-run.log 2>&1
echo "- $(date '+%H:%M') STEER: queue-v3 (N2 both personas) done" >> results/MORNING-STATUS.md
