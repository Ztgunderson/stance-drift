#!/usr/bin/env bash
cd /home/jetson/lab/benches/mats-nanda
until grep -q "plan p1-4 done" results/steer/p1-4-run.log; do sleep 30; done
PYTHONPATH=production HF_HUB_OFFLINE=1 timeout 7200 .venv/bin/python -m driftlab.steer_trials_run --cells neutral/noleak/none,neutral/noleak_noleave/none --out-dir results/steer > results/steer/neutral-tiers-run.log 2>&1
PYTHONPATH=production HF_HUB_OFFLINE=1 timeout 7200 .venv/bin/python -m driftlab.steer_trials_run --plan headline --rep 2 --out-dir results/steer > results/steer/headline-rep2-run.log 2>&1
