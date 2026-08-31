#!/usr/bin/env bash
# qwen3.5-9b — tutor only, 8 rounds — cross-model replication run #1.
# (MATS sprint 2026-08-25; see ~/lab/benches/mats-nanda/SPRINT-PLAN.md §2.)
#
#   TUTOR_REPS=1 ./runners/run_qwen35-9b_tutor8.sh    # smoke: one balanced pass
#   ./runners/run_qwen35-9b_tutor8.sh                 # full run
#
# IDENTICAL protocol to run_qwen36_tutor8.sh — same scenes, scratchpad, rounds,
# cells (3 agents x 2 arms) — different model. The point is replication, so
# nothing about the design is allowed to vary except the model.
#
# Differences from the 35B runner, all serving-side:
#   * port 8000 (direct vLLM via pleasing/serve-model.sh), not LiteLLM :4000
#   * needs_swap=yes: boots pleasing-qwen35-9b via `make pleasing`, which downs
#     serve/ + sprint/ first (single 64GB pool). Cold start ~5-15 min.
#   * dense bf16 9B, not AWQ MoE: per-trial wall-clock unknown — measure on the
#     smoke pass before trusting DEADLINE arithmetic.
TUTOR_REPS=${TUTOR_REPS:-12}      # ceiling; DEADLINE decides what actually lands
CONTRACT_REPS=0                   # refusal floor already banked on the 35B
ROUNDS=${ROUNDS:-8}
DEADLINE=${DEADLINE:-5400}

SD_MAX_TASKS=${SD_MAX_TASKS:-6}   # start conservative on an unmeasured model

export RESULT_DIR=qwen3.5-9b-tutor8
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_model "qwen3.5-9b" "8000" "yes"
