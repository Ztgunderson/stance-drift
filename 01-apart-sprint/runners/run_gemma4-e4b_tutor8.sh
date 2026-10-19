#!/usr/bin/env bash
# gemma-4-e4b — tutor only, 8 rounds — capability-ladder FLOOR rung.
# (MATS sprint 2026-08-25c; see ~/lab/benches/mats-nanda/SPRINT-PLAN.md.)
#
# Expected result: precondition failure (no stance forms), like the 9B.
# That expectation is the point — this rung anchors the floor of the
# scale-threshold curve. Budget: SMOKE-LEVEL ONLY (1-2 passes). Do not run
# a full sweep here without a surprising smoke result.
#
#   TUTOR_REPS=1 ./runners/run_gemma4-e4b_tutor8.sh
TUTOR_REPS=${TUTOR_REPS:-2}
CONTRACT_REPS=0
ROUNDS=${ROUNDS:-8}
DEADLINE=${DEADLINE:-3000}

SD_MAX_TASKS=${SD_MAX_TASKS:-6}

export RESULT_DIR=gemma-4-e4b-tutor8
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_model "gemma-4-e4b" "8000" "yes"
