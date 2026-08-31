#!/usr/bin/env bash
# ministral-3-14b — tutor only, 8 rounds — FALLBACK for the 12B-class slot.
# (MATS sprint 2026-08-25; runs ONLY if gemma-4-12b fails its smoke test.
#  See ~/lab/benches/mats-nanda/SPRINT-PLAN.md §1 and registry.py.)
#
#   TUTOR_REPS=1 ./runners/run_ministral3-14b_tutor8.sh   # smoke first, always
#   ./runners/run_ministral3-14b_tutor8.sh                # full run
#
# IDENTICAL protocol to run_qwen36_tutor8.sh — only the model varies.
TUTOR_REPS=${TUTOR_REPS:-12}
CONTRACT_REPS=0
ROUNDS=${ROUNDS:-8}
DEADLINE=${DEADLINE:-5400}

SD_MAX_TASKS=${SD_MAX_TASKS:-6}

# Ministral 3's chat template 400s on Qwen's enable_thinking kwarg (it has no
# thinking mode) — verified against the live endpoint 2026-08-26.
export SD_NO_THINK_KWARG=1
export RESULT_DIR=ministral-3-14b-tutor8
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_model "ministral-3-14b" "8000" "yes"
