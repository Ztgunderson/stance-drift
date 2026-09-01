#!/usr/bin/env bash
# Post-Q0 sequence, 2026-09-01 evening (user-approved):
#   wait for Q0 -> boot vLLM -> Amendment-4 node batch -> stop vLLM ->
#   reminder-arm replay -> re-execute notebook 08 (H1'/H2 test, both arms)
# Launch: nohup ./post-q0-0901.sh >> results/post-q0-0901.log 2>&1 &
set -uo pipefail
BENCH="$(cd "$(dirname "$0")" && pwd)"; cd "$BENCH"
STATUS=results/MORNING-STATUS.md
SNAP=/home/jetson/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
JUPYTER=/home/jetson/lab/existing/jetson-llm/stance-drift/.venv/bin/jupyter
note() { echo "- $(date '+%H:%M') POST-Q0: $1" >> "$STATUS"; echo "[post-q0] $1"; }

# 0. wait for Q0 (max 90 min)
for i in $(seq 1 90); do
  grep -q 'Q0 complete' "$STATUS" 2>/dev/null && break; sleep 60
done
grep -q 'Q0 complete' "$STATUS" || { note "Q0 never completed — aborting"; exit 1; }

# 1. boot vLLM with watchdog
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
for i in $(seq 1 80); do sleep 15
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null)
  [ "$code" = "200" ] && break
done
if [ "${code:-}" != "200" ]; then
  note "ERROR: boot unhealthy in 20 min — stopping early (anti-zombie)"
  timeout 90 docker stop pleasing-qwen35-9b || note "WARN: stop failed too"
  exit 1
fi
note "vLLM healthy — node batch starting"

# 2. Amendment-4 node batch (supportive r2+r3, k=25)
if timeout 7200 env LOCAL_BASE_URL=http://127.0.0.1:8000/v1 \
    .venv/bin/python production/tutorbench/node_resample.py \
    --log-dir results-v1/qwen3.5-9b --model qwen3.5-9b \
    --personas supportive --rounds 2,3 --k 25 --workers 6 \
    --out results/nodes-supportive-k25.json; then
  note "node batch done: results/nodes-supportive-k25.json"
else
  note "ERROR: node batch exited nonzero (resume-safe)"
fi

# 3. stop vLLM, reminder-arm replay
timeout 90 docker stop pleasing-qwen35-9b || note "WARN: docker stop failed"
sleep 5
if timeout 7200 env HF_HUB_OFFLINE=1 .venv/bin/python production/driftlab/replay.py \
    --model "$SNAP" --log-dir results/pilot-reminder-v2 \
    --out microscope/cache/qwen35-9b-reminder-v1 --device cuda; then
  note "reminder replay done: $(ls microscope/cache/qwen35-9b-reminder-v1/*.npz 2>/dev/null | wc -l) npz"
else
  note "ERROR: reminder replay exited nonzero"
fi

# 4. re-execute notebook 08 (now sees both caches)
if timeout 3600 "$JUPYTER" nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=mats-bench \
    --ExecutePreprocessor.timeout=3000 review/08-reminder-traces.ipynb; then
  note "notebook 08 re-executed (H1'/H2 both arms)"
else
  note "ERROR: notebook 08 re-execution failed"
fi
note "POST-Q0 complete — server left DOWN; O0 remains user-gated"
