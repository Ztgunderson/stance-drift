#!/usr/bin/env bash
# O0 — overnight expansion, launched ONLY after the 9 pm review green-light.
#   boot vLLM -> supportive+neutral reps 2-4 (adds 3 reps -> 48/cell within
#   Amendment-1 primary cells; Arm 0, no reminder) -> stop vLLM -> replay-cache
#   the new trials -> append to MORNING-STATUS.md
# Launch:  nohup ./overnight-O0-expand.sh >> results/overnight-O0.log 2>&1 &
set -uo pipefail
BENCH="$(cd "$(dirname "$0")" && pwd)"
cd "$BENCH"
STATUS=results/MORNING-STATUS.md
SNAP=/home/jetson/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
CACHE=microscope/cache/qwen35-9b-v1
SWEEP_PY=/home/jetson/lab/existing/jetson-llm/stance-drift/.venv/bin/python
note() { echo "- $(date '+%H:%M') O0: $1" >> "$STATUS"; echo "[o0] $1"; }

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

# boot watchdog: healthy within 20 min or stop WHILE the process is alive
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

# 2. expansion reps (resume-safe pass dirs; Arm 0 baseline, primary cells)
if timeout 21600 env LOCAL_BASE_URL=http://127.0.0.1:8000/v1 LOCAL_API_KEY=sk-no-key-required \
    "$SWEEP_PY" production/tutorbench/sweep.py \
    --model openai-api/local/qwen3.5-9b \
    --log-dir results-v1/qwen3.5-9b-expand \
    --personas supportive,neutral --reps 4; then
  note "expansion sweep done"
else
  note "ERROR: expansion sweep exited nonzero — partial passes are resume-safe"
fi

# 3. stop vLLM, replay-cache the new trials into the same cache dir
timeout 90 docker stop pleasing-qwen35-9b || note "WARN: docker stop failed post-sweep"
sleep 5
if timeout 14400 env HF_HUB_OFFLINE=1 .venv/bin/python production/driftlab/replay.py \
    --model "$SNAP" --log-dir results-v1/qwen3.5-9b-expand --out "$CACHE" --device cuda; then
  note "expansion replay done: cache now $(ls "$CACHE"/*.npz 2>/dev/null | wc -l) npz"
else
  note "ERROR: expansion replay exited nonzero"
fi
note "O0 complete — morning review: re-execute 04-l1-preview against the fuller cache"
