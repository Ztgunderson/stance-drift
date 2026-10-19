#!/usr/bin/env bash
# qwen3.6-35b — tutor only, 8 rounds — the baseline arm-comparison run.
#
#   ./runners/run_qwen36_tutor8.sh
#
# Scope narrowed 2026-08-16 15:30. The contract arenas are dropped from new
# collection: all three sit at a hard refusal floor (0 hand-overs across
# predatory/fair/generous, including the generous one), so more reps there buy
# no variance. That result is already banked in results/qwen3.6-35b/ and is
# reported as a finding, not re-run.
#
# What is left is the arena whose outcome actually varies (tutor) and the one
# contrast the study rests on: arm=in_context (the scratchpad is fed back into
# the conversation) vs arm=scratchpad (it is never shown again).
#
# One pass = 3 agents x 2 arms = 6 trials, balanced. sweep_plan is rep-major and
# checks the deadline only at pass boundaries, so a truncated run still has
# equal n in every cell — never a half-filled design. TUTOR_REPS is therefore
# set high deliberately: it is a ceiling, not a target. The run banks as many
# balanced passes as DEADLINE allows and stops clean.
TUTOR_REPS=${TUTOR_REPS:-12}      # ceiling; DEADLINE decides what actually lands
CONTRACT_REPS=0                   # stashed at the refusal floor, see above
ROUNDS=${ROUNDS:-8}
DEADLINE=${DEADLINE:-3600}

# 6 -> 12. vLLM here runs --max-num-seqs 24, so 12 in flight stays well inside
# the batch and roughly halves wall-clock per pass. Measured at 6: ~30-35s/trial.
SD_MAX_TASKS=${SD_MAX_TASKS:-12}

export RESULT_DIR=qwen3.6-35b-tutor8
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_model "qwen3.6-35b" "4000" "serve"
