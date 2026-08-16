#!/usr/bin/env bash
# qwen3.6-35b — priority 4 — the model we know best; breadth
#
#   ./runners/run_qwen36-35b.sh
#
# Idempotent: starts the container if it is not already serving, preflights it,
# then sweeps until DEADLINE. Re-running adds more passes (each pass gets its own
# log directory), so this is also how you top up a model later.
DEADLINE=${DEADLINE:-2700}
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_model "qwen3.6-35b" "4000" "serve"
