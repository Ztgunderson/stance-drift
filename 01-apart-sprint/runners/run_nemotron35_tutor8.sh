#!/usr/bin/env bash
# nemotron-3.5-30b — tutor, 8 rounds. The replication run for the second model.
#
#   ./runners/serve_nemotron35.sh          # first: get it serving on :8000
#   ./runners/run_nemotron35_tutor8.sh
#
# Talks to vLLM directly on :8000 (the container uses --network host), so
# LiteLLM on :4000 is not involved and no API key is needed.
#
# Identical design to run_qwen36_tutor8.sh — same scene, personas, arms, rounds —
# so the only difference between the two datasets is the model. That is the whole
# point: anything that differs is attributable to the model, not the protocol.
export RESULT_DIR=nemotron35-tutor8
export LOCAL_BASE_URL=${LOCAL_BASE_URL:-http://127.0.0.1:8000/v1}
export LOCAL_API_KEY=${LOCAL_API_KEY:-EMPTY}

TUTOR_REPS=${TUTOR_REPS:-12}
CONTRACT_REPS=0
ROUNDS=${ROUNDS:-8}
DEADLINE=${DEADLINE:-2400}
SD_MAX_TASKS=${SD_MAX_TASKS:-6}

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_model "nemotron35" "8000" "no"
