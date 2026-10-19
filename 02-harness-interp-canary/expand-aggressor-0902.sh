#!/usr/bin/env bash
# Aggressor expansion — user-approved 2026-09-02 for the 1–3 pm away window.
#   boot vLLM -> aggressor reps 1-4 (adds 96 trials -> 120/cell, matching the
#   supportive/neutral cells; Arm 0, no reminder) -> stop vLLM -> replay-cache
#   the new trials into the shared cache -> append to MORNING-STATUS.md.
# Own log dir (not results-v1/qwen3.5-9b-expand): inspect eval_set validates a
# log dir against its task set, so mixing personas into last night's pass dirs
# is not safe. Replay skips already-cached stems, so the shared cache is fine.
# Launch:  nohup ./expand-aggressor-0902.sh >> results/expand-aggressor-0902.log 2>&1 &
set -uo pipefail
BENCH="$(cd "$(dirname "$0")" && pwd)"
cd "$BENCH"
STATUS=results/MORNING-STATUS.md
SNAP=/home/jetson/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
CACHE=microscope/cache/qwen35-9b-v1
LOGDIR=results-v1/qwen3.5-9b-expand-aggressor
SWEEP_PY=/home/jetson/lab/existing/jetson-llm/stance-drift/.venv/bin/python
note() { echo "- $(date '+%H:%M') AGG-EXPAND: $1" >> "$STATUS"; echo "[agg] $1"; }

note "starting aggressor expansion (4 reps x 24 items) — user-approved 1-3 pm window"

# 1. boot vLLM (compile cache disabled for the week; watch for the silent-wedge signature)
timeout 60 docker rm -f pleasing-qwen35-9b >/dev/null 2>&1 || true
timeout 120 docker run -d --rm --init --name pleasing-qwen35-9b --runtime nvidia --network host \
  -v /home/jetson/.cache/huggingface:/root/.cache/huggingface \
  -v /home/jetson/.cache/vllm:/root/.cache/vllm \
  -e HF_HOME=/root/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
  -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  vllm serve Qwen/Qwen3.5-9B --served-model-name qwen3.5-9b --max-model-len 32768 \
  --gpu-memory-utilization 0.6 --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder || { note "ERROR: docker run failed"; exit 1; }

for i in $(seq 1 80); do
  sleep 15
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null)
  [ "$code" = "200" ] && break
done
if [ "${code:-}" != "200" ]; then
  note "ERROR: boot not healthy in 20 min — stopping container early (anti-zombie) and aborting"
  timeout 90 docker stop pleasing-qwen35-9b || note "WARN: stop also failed — tell the user, may need reboot"
  exit 1
fi
note "vLLM healthy"

# 2. aggressor reps (own pass dirs; Arm 0 baseline)
if timeout 7200 env LOCAL_BASE_URL=http://127.0.0.1:8000/v1 LOCAL_API_KEY=sk-no-key-required \
    "$SWEEP_PY" production/tutorbench/sweep.py \
    --model openai-api/local/qwen3.5-9b \
    --log-dir "$LOGDIR" \
    --personas aggressor --reps 4; then
  note "aggressor sweep done: $(find "$LOGDIR" -name '*.eval' | wc -l) eval files"
else
  note "ERROR: aggressor sweep exited nonzero — partial passes are resume-safe"
fi

# 3. stop vLLM, replay-cache the new trials into the shared cache dir
timeout 90 docker stop pleasing-qwen35-9b || note "WARN: docker stop failed post-sweep"
sleep 5
if timeout 7200 env HF_HUB_OFFLINE=1 .venv/bin/python production/driftlab/replay.py \
    --model "$SNAP" --log-dir "$LOGDIR" --out "$CACHE" --device cuda; then
  note "aggressor replay done: cache now $(ls "$CACHE"/*.npz 2>/dev/null | wc -l) npz"
else
  note "ERROR: aggressor replay exited nonzero"
fi
note "AGG-EXPAND complete — server DOWN; cache should read 360 npz (120/cell x 3)"
