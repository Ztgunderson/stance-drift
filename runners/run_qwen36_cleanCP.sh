#!/usr/bin/env bash
# qwen3.6-35b — tutor, 8 rounds — with the counterparty forbidden to state the
# answer itself. The confound-fix condition.
#
#   ./runners/run_qwen36_cleanCP.sh
#
# WHY. In the 72-trial baseline the self-play student solved its own homework and
# said the answer out loud — 4% of convincer trials but **75% of supportive**
# ones. Every later mention by the tutor then scored as a hand-over, so the
# outcome measure was contaminated in lockstep with the persona being compared,
# and the persona ordering inverted once corrected (NOTES-FOR-PAPER §4e).
#
# SD_CP_NO_ANSWER=1 tells the counterparty it has not solved the problem and must
# never state or work toward the answer. Everything else is identical to
# run_qwen36_tutor8.sh, so this is a clean A/B **on the measurement fix itself**:
# if leakage drops sharply and the persona ordering changes, the confound is
# confirmed as the cause rather than inferred from a re-analysis.
#
# Results land in their own directory — do NOT pool these with the baseline.
export SD_CP_NO_ANSWER=1
export RESULT_DIR=qwen3.6-35b-cleanCP

TUTOR_REPS=${TUTOR_REPS:-3}
CONTRACT_REPS=0
ROUNDS=${ROUNDS:-8}
DEADLINE=${DEADLINE:-1500}
SD_MAX_TASKS=${SD_MAX_TASKS:-6}

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_model "qwen3.6-35b" "4000" "serve"
