#!/usr/bin/env bash
# tutorbench overnight queue v2 — 2026-08-28 (Friday night).
# Large-model night: qwen3.6-35b (boot proven earlier today, daylight rule).
# Gemma stages stay commented until their daylight boot test passes; nemotron
# stays behind its runtime gate. Same hardened skeleton as v1: pre-night
# checklist, timeout-wrapped docker, preflight rc=2 fallback, smoke gate.
# Detach with:
#   nohup ./tutorbench-queue2.sh >/dev/null 2>&1 &
set -uo pipefail
SD=~/lab/existing/jetson-llm/stance-drift
JETSON=~/lab/existing/jetson-llm
BENCH=~/lab/benches/mats-nanda
PY=$SD/.venv/bin/python
exec >>"$BENCH/results/overnight2.log" 2>&1
note(){ echo "[$(date +%H:%M)] $*"; }

set -a; source "$JETSON/.env"; set +a
export LOCAL_API_KEY="${LITELLM_MASTER_KEY:-sk-no-key-required}"
# qwen3.8-27b is llama.cpp-served on 8080 (vLLM entries use 8000; serve-model's
# own readiness wait already knew this — preflight/sweep learn it here).
export LOCAL_BASE_URL="http://127.0.0.1:8080/v1"

note "=== tutorbench overnight v2 start ==="

# ---- pre-night checklist (every line is a past root cause) -----------------
FREE_G=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "$FREE_G" -ge 15 ] || { note "ABORT: ${FREE_G}G free < 15G"; exit 1; }
pgrep -f "huggingface_hub\|hf_transfer\|docker pull" >/dev/null \
  && { note "ABORT: a download is running (unified-memory lesson)"; exit 1; }
docker ps --format '{{.Names}}' | grep -q . \
  && { note "note: containers up at start:"; docker ps --format ' {{.Names}}'; }
note "checklist ok (${FREE_G}G free)"

check_clean(){ # $1 = results-v1 subdir; $2 = min trials
  RD="$1" MIN="$2" $PY - <<'PYEOF'
import os, sys
sys.path.insert(0, os.path.expanduser("~/lab/benches/mats-nanda/production"))
from tutorbench.load import load_trials, load_rounds
rd = os.path.expanduser("~/lab/benches/mats-nanda/results-v1/") + os.environ["RD"]
try:
    t = load_trials(rd); r = load_rounds(rd)
except SystemExit as e:
    print(e); sys.exit(1)
bad = int(r.unparsed.sum())
print(f"{os.environ['RD']}: {len(t)} trials, {bad} unparsed notes, "
      f"outcomes: {t.outcome.value_counts().to_dict()}")
sys.exit(0 if len(t) >= int(os.environ["MIN"]) and bad == 0 else 1)
PYEOF
}

run_model(){ # alias deadline_s
  local alias="$1" deadline="$2"
  note "----- $alias (deadline $((deadline/60))min) -----"
  timeout 60 docker container prune -f >/dev/null 2>&1 || true
  free -g | awk '/^Mem:/{printf "  %sG available before start\n", $7}'
  if ! curl -sf --max-time 5 "$LOCAL_BASE_URL/models" \
        -H "Authorization: Bearer $LOCAL_API_KEY" 2>/dev/null | grep -q "$alias"; then
    timeout 2400 make -C "$JETSON" pleasing MODEL="$alias" || {
      note ">> $alias FAILED TO BOOT — engine log tail:"
      timeout 30 docker ps -a --format '{{.Names}} {{.Status}}' | sed 's/^/    /'
      timeout 30 docker logs --tail 15 \
        "$(timeout 15 docker ps -aq --filter name=pleasing- | head -1)" 2>&1 \
        | sed 's/^/    /' || true
      timeout 60 docker rm -f \
        $(timeout 15 docker ps -aq --filter name=pleasing-) >/dev/null 2>&1 || true
      return 1; }
  else
    note "already serving"
  fi

  note "preflight"
  (cd "$SD" && .venv/bin/python stancedrift/preflight.py "$alias"); local rc=$?
  [ $rc -eq 1 ] && { note ">> preflight FAILED — skipping $alias"; return 1; }
  [ $rc -eq 2 ] && export SD_NO_GUIDED_JSON=1 || unset SD_NO_GUIDED_JSON

  note "smoke: 1 item x 3 personas"
  $PY "$BENCH/production/tutorbench/sweep.py" \
      --model "openai-api/local/$alias" \
      --log-dir "$BENCH/results-v1/$alias-smoke" \
      --items q00 --reps 1 --deadline 1800 --max-tasks 3
  check_clean "$alias-smoke" 3 || { note ">> $alias smoke NOT clean — skipping full"; return 1; }

  note "full: 24 items x 3 personas"
  $PY "$BENCH/production/tutorbench/sweep.py" \
      --model "openai-api/local/$alias" \
      --log-dir "$BENCH/results-v1/$alias" \
      --reps 1 --deadline "$deadline" --max-tasks "${TB_MAX_TASKS:-4}"
  check_clean "$alias" 60 || note ">> $alias full run has issues — review"

  note "teardown $alias"
  timeout 120 docker ps --format '{{.Names}}' | grep '^pleasing-' \
    | xargs -r -n1 timeout 90 docker rm -f
}

# 35B MoE (3B active) at max_tasks 4: budget generously — smaller models ran
# 17s/trial at max_tasks 6; give the 72 trials up to 5h before eval_set stops.
run_model qwen3.8-27b 18000 || true

# --- Friday-gate stages: uncomment only after a daylight boot test passes ---
# gemma-4-e4b BANKED 08-28 16:44: 72 trials, 0 unparsed
# (left_after_leak 45 / leaked 21 / left 6). Stage disabled to avoid a
# duplicate pass; re-enable only for reps>1.

# gemma-4-12b CUT 08-28 ~09:40: checkpoint model_type gemma4_unified is not
# known to the gemma4-jetson-orin image's transformers 5.5.0 (E4B is plain
# gemma4 and boots fine). Its pre-armed 12B-class fallback is ministral-3-14b,
# which already banked 72/72 clean trials on 08-28 — slot covered, no action.

note "teardown (final)"
timeout 120 docker ps --format '{{.Names}}' | grep '^pleasing-' \
  | xargs -r -n1 timeout 90 docker rm -f

note "bundling traces for review"
"$SD/.venv/bin/inspect" view bundle --log-dir "$BENCH/results-v1" \
    --output-dir "$BENCH/review/traces-v1" --overwrite \
  && note "bundle ok: review/traces-v1" || note ">> bundle FAILED"

{
  echo "# Status — $(date '+%F %H:%M') (tutorbench overnight v2)"
  echo
  for d in "$BENCH"/results-v1/*/; do
    n=$(find "$d" -name '*.eval' 2>/dev/null | wc -l)
    echo "- $(basename "$d"): $n trials"
  done
  echo
  echo "Review: notebooks in review/ (kernel mats-bench); traces at"
  echo "review/traces-v1/index.html via the Jupyter file server."
} > "$BENCH/results/OVERNIGHT-STATUS.md"
note "tutorbench overnight v2 complete"
