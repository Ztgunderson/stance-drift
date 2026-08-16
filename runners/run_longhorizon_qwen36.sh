#!/usr/bin/env bash
# Qwen3.6-35B, tutor only, 16 rounds — the long-horizon probe.
#
#   ./runners/run_longhorizon_qwen36.sh
#
# Separate from the main runner because it answers a different question: does
# stance drift keep accumulating over a long interaction, or plateau? That is the
# question that matters for long-running agentic systems, and it is not
# answerable at 8 rounds. Tutor only, because that is the arena whose outcome
# varies; the contract arenas sit at a refusal floor.
export RESULT_DIR=qwen3.6-35b-16round
# 3 reps (18 trials), not 4. At 16 rounds a pass costs ~2x an 8-round pass
# (~8min), and the window has to also verify two llama.cpp models. DEADLINE is
# the real guard: sweep_plan checks it only at pass boundaries, so this stops
# clean with equal n rather than truncating a pass.
DEADLINE=${DEADLINE:-2100}
TUTOR_REPS=${TUTOR_REPS:-3}
CONTRACT_REPS=0
ROUNDS=${ROUNDS:-16}
SD_MAX_TASKS=${SD_MAX_TASKS:-6}
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_model "qwen3.6-35b" "4000" "serve"
