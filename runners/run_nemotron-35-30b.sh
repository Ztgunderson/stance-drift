#!/usr/bin/env bash
# nemotron-3.5-30b — priority 3 — NVIDIA, breadth; NVFP4 may not load on Ampere
#
#   ./runners/run_nemotron-35-30b.sh
#
# Idempotent: starts the container if it is not already serving, preflights it,
# then sweeps until DEADLINE. Re-running adds more passes (each pass gets its own
# log directory), so this is also how you top up a model later.
DEADLINE=${DEADLINE:-2400}
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_model "nemotron-3.5-30b" "8000" "yes"
