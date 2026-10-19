#!/usr/bin/env bash
# Overnight queue v3 — 2026-08-26 evening. Goal: 9B behavioral N 36 -> 72
# (TUTOR_REPS=12; eval_set resumes past banked passes 1-6, runs 7-12), then
# tear down serving, replay-cache the new trials, and re-run the 9B probes.
#
# Sequencing note: give_in_markers were widened tonight (scenes.json — reversed
# order, factored form, worded numbers) BEFORE this run, so tonight's trials are
# scored on the new set at generation time. All banked trials were re-scored
# offline against the same set: exactly 1 flip (9B reversed-order trial,
# held->gave), cache sidecar patched, probes re-run. Labels are consistent
# across old and new data.
set -uo pipefail
SD=~/lab/existing/jetson-llm/stance-drift
BENCH=~/lab/benches/mats-nanda
MS=$BENCH/microscope
PY=$BENCH/.venv/bin/python
exec >>"$BENCH/results/overnight.log" 2>&1
note(){ echo "[$(date +%H:%M)] $*"; }

note "=== overnight v3 start (9B 36->72 + recache + probes) ==="

note "9B tutor8 full run (TUTOR_REPS=12; banked passes 1-6 skip via eval_set resume)"
(cd "$SD" && TUTOR_REPS=12 DEADLINE=14400 ./runners/run_qwen35-9b_tutor8.sh)
note "9B trials on disk: $(find "$SD/results/qwen3.5-9b-tutor8" -name '*.eval' 2>/dev/null | wc -l)"

note "tearing down pleasing-* containers to free unified memory for HF replay"
docker ps --format '{{.Names}}' | grep '^pleasing-' | xargs -r docker stop
sleep 30

snap(){ ls -d ~/.cache/huggingface/hub/models--$1/snapshots/*/ | head -1; }
note "replay-caching 9B trials (existing npz skip — only new ones run)"
$PY "$MS/replay_cache.py" --model "$(snap Qwen--Qwen3.5-9B)" \
    --log-dir "$SD/results/qwen3.5-9b-tutor8" \
    --out "$MS/cache/qwen3.5-9b-tutor8" --device cuda \
  || note "9B recache FAILED"

note "probes on refreshed 9B cache"
$PY "$MS/probes.py" --cache "$MS/cache/qwen3.5-9b-tutor8" --label turn    > "$MS/probe-9b-turn.txt"   2>&1 || note "9B turn probe FAILED"
$PY "$MS/probes.py" --cache "$MS/cache/qwen3.5-9b-tutor8" --label gave_in > "$MS/probe-9b-gavein.txt" 2>&1 || note "9B gave_in probe FAILED"

{
  echo "# Status — $(date '+%F %H:%M') (overnight v3: 9B expansion)"
  echo
  echo "Marker set widened before the run; banked labels re-scored (1 flip, patched)."
  for d in qwen3.5-4b-tutor8 qwen3.5-9b-tutor8 ministral-3-14b-tutor8; do
    echo "- trials $d: $(find "$SD/results/$d" -name '*.eval' 2>/dev/null | wc -l)"
  done
  echo "- 9B cache npz: $(ls "$MS/cache/qwen3.5-9b-tutor8"/*.npz 2>/dev/null | wc -l)"
  echo "- 9B gave_in probe: $(grep 'best' "$MS/probe-9b-gavein.txt" 2>/dev/null || echo 'n/a')"
  echo "- 9B turn probe:    $(grep 'best' "$MS/probe-9b-turn.txt" 2>/dev/null || echo 'n/a')"
} > "$BENCH/results/OVERNIGHT-STATUS.md"
note "overnight v3 complete"
