#!/usr/bin/env bash
# gemma-4-12b — tutor only, 8 rounds — cross-model replication run #2.
# (MATS sprint 2026-08-25; see ~/lab/benches/mats-nanda/SPRINT-PLAN.md §2.)
#
#   TUTOR_REPS=1 ./runners/run_gemma4-12b_tutor8.sh   # smoke: one balanced pass
#   ./runners/run_gemma4-12b_tutor8.sh                # full run
#
# IDENTICAL protocol to run_qwen36_tutor8.sh — only the model varies.
# Run AFTER run_qwen35-9b_tutor8.sh finishes (one model owns the GPU at a time;
# `make pleasing` handles the swap).
#
# ⚠️ Gemma-4-12B-it is an Any-to-Any arch. Pre-declared fallback (SPRINT-PLAN
# §1): if serving or preflight fights for >30 min, swap the registry entry to
# google/gemma-3-12b-it (gated repo — needs HF login) and rerun. Do not sink
# sprint hours into "newest Gemma".
TUTOR_REPS=${TUTOR_REPS:-12}
CONTRACT_REPS=0
ROUNDS=${ROUNDS:-8}
DEADLINE=${DEADLINE:-5400}

SD_MAX_TASKS=${SD_MAX_TASKS:-6}

export RESULT_DIR=gemma-4-12b-tutor8
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_model "gemma-4-12b" "8000" "yes"
