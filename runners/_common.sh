#!/usr/bin/env bash
# Shared body for the per-model runners. Not run directly.
#
# One runner per model, on purpose. The automated multi-model orchestrator was
# deleted after it broke the run twice: once by polling for a process that had
# no matchable command line (it declared model 1 finished 2s after launch and
# tore down its container), and once by feeding eval_set a log directory it
# refused as "not fresh" (silently capping every model at one pass). Each model
# is now a single command you run and watch.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
JETSON="$(dirname "$(pwd)")"
set -a; source "$JETSON/.env"; set +a
export LOCAL_API_KEY="${LITELLM_MASTER_KEY:-sk-no-key-required}"

# 8 rounds, not 6. The self-report trajectories flatten after ~6, which argued
# for 6 — but the OUTCOME is a different measure, and 6 rounds produced 0/6 while
# 8 rounds produced 7/24. A rate with no variance cannot be made informative by
# adding reps. 8 also matches the earlier 48-trial run, keeping it comparable.
#
# Defaults; every runner overrides DEADLINE to its share of the window.
TUTOR_REPS=${TUTOR_REPS:-16}
CONTRACT_REPS=${CONTRACT_REPS:-6}
ROUNDS=${ROUNDS:-8}
SD_MAX_TASKS=${SD_MAX_TASKS:-6}
export SD_MAX_TASKS

run_model() {                 # alias  port  [needs_swap]
  local alias="$1" port="$2" needs_swap="${3:-yes}"
  local url="http://127.0.0.1:${port}/v1"
  export LOCAL_BASE_URL="$url" SD_MODEL="openai-api/local/${alias}"

  echo "=========================================================="
  echo " $alias   deadline $((DEADLINE/60))min   started $(date +%H:%M)"
  echo "=========================================================="

  # The Authorization header is required, not optional: :4000 is LiteLLM, which
  # answers 401 to an unauthenticated /v1/models. Without it `curl -sf` fails on
  # a perfectly healthy endpoint and the probe reports "not serving" — which is
  # exactly what happened at 15:34 on 2026-08-16. The direct vLLM ports do not
  # care about the header, so sending it always is safe.
  if curl -sf --max-time 5 "$url/models" \
       -H "Authorization: Bearer ${LOCAL_API_KEY}" 2>/dev/null | grep -q "$alias"; then
    echo "[$(date +%H:%M)] already serving — no swap needed"
  elif [[ "$needs_swap" == "yes" ]]; then
    echo "[$(date +%H:%M)] freeing memory and starting $alias"
    docker container prune -f >/dev/null 2>&1 || true
    free -g | awk '/^Mem:/{printf "  %sG available before start\n", $7}'
    make -C "$JETSON" pleasing MODEL="$alias" || {
      echo ">> $alias FAILED TO BOOT"; return 1; }
  else
    echo ">> $alias is not serving on :$port and no swap was requested"; return 1
  fi

  echo "[$(date +%H:%M)] preflight"
  .venv/bin/python stancedrift/preflight.py "$alias"
  local rc=$?
  [[ $rc -eq 1 ]] && { echo ">> preflight FAILED — not sweeping $alias"; return 1; }
  [[ $rc -eq 2 ]] && export SD_NO_GUIDED_JSON=1 || unset SD_NO_GUIDED_JSON

  .venv/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from stancedrift import analysis
analysis.setup_env()
analysis.sweep_plan(tutor_reps=$TUTOR_REPS, contract_reps=$CONTRACT_REPS,
                    rounds=$ROUNDS, log_dir="results/${RESULT_DIR:-$alias}",
                    deadline_s=$DEADLINE)
PYEOF

  echo
  echo "[$(date +%H:%M)] $alias done — $(find results/${RESULT_DIR:-$alias} -name '*.eval' 2>/dev/null | wc -l) trials"
  .venv/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from stancedrift import analysis
try:
    df = analysis.load_sweep("results/${RESULT_DIR:-$alias}")
    t = df.drop_duplicates("trial")
    print(f"  {t.trial.nunique()} trials, {df.stance.isna().sum()} unparsed scratchpads")
    print("  gave in:", t.groupby("scene")["gave_in"].agg(["sum","count"]).to_dict("index"))
except SystemExit as e:
    print(" ", e)
PYEOF
}
