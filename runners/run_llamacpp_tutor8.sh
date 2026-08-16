#!/usr/bin/env bash
# Breadth run for a llama.cpp model — tutor only, 8 rounds, low n.
#
#   ./runners/serve_llamacpp.sh qwen38 && ./runners/run_llamacpp_tutor8.sh qwen3.8-27b
#   ./runners/serve_llamacpp.sh muse   && ./runners/run_llamacpp_tutor8.sh muse-glimmer-30b
#
# The point of the slow stack is EXTERNAL VALIDITY, NOT STATISTICAL POWER. A
# second model showing the same shape is worth more than a tighter interval on
# the first — but only the fast stack can afford the trial counts that make a
# rate meaningful. So: 3 reps here (18 trials), 12 on qwen3.6. Do not "fix" this
# by raising reps; raise them on vLLM instead, where they are 6x cheaper.
#
# needs_swap="no": serve_llamacpp.sh already started the container and asserted
# the GPU is live. This runner will refuse rather than silently start something.
ALIAS="${1:?usage: $0 <alias>   e.g. qwen3.8-27b | muse-glimmer-30b}"

TUTOR_REPS=${TUTOR_REPS:-3}
CONTRACT_REPS=0
ROUNDS=${ROUNDS:-8}
DEADLINE=${DEADLINE:-2400}
SD_MAX_TASKS=${SD_MAX_TASKS:-6}    # matches --parallel 6 on llama-server

export RESULT_DIR="${ALIAS}-tutor8"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_model "$ALIAS" "8080" "no"
