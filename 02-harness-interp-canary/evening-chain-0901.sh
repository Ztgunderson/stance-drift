#!/usr/bin/env bash
# Evening chain 09-01 (user: "go O0 now"):
#   wait for POST-Q0 complete -> O0 expansion (boot, 48/cell supportive+neutral,
#   stop, replay new trials) -> boot server -> remaining Amendment-4 node sets
#   (reminder-arm aggressor r2+r4; baseline neutral r2-r5) -> stop server.
# Launch: nohup ./evening-chain-0901.sh >> results/evening-chain-0901.log 2>&1 &
set -uo pipefail
BENCH="$(cd "$(dirname "$0")" && pwd)"; cd "$BENCH"
STATUS=results/MORNING-STATUS.md
note() { echo "- $(date '+%H:%M') CHAIN: $1" >> "$STATUS"; echo "[chain] $1"; }

for i in $(seq 1 120); do
  grep -q 'POST-Q0 complete' "$STATUS" 2>/dev/null && break; sleep 60
done
grep -q 'POST-Q0 complete' "$STATUS" || { note "post-Q0 never completed — aborting"; exit 1; }

note "starting O0 expansion"
./overnight-O0-expand.sh || note "O0 exited nonzero (see results/overnight-O0.log)"

# node sets: boot server again
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
  --tool-call-parser qwen3_coder || { note "ERROR: node-window boot failed"; exit 1; }
for i in $(seq 1 80); do sleep 15
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null)
  [ "$code" = "200" ] && break
done
if [ "${code:-}" != "200" ]; then
  note "ERROR: node-window boot unhealthy — stopping early (anti-zombie)"
  timeout 90 docker stop pleasing-qwen35-9b || true; exit 1
fi
note "node window: reminder-arm aggressor r2+r4"
timeout 10800 env LOCAL_BASE_URL=http://127.0.0.1:8000/v1 \
  .venv/bin/python production/tutorbench/node_resample.py \
  --log-dir results/pilot-reminder-v2 --model qwen3.5-9b \
  --personas aggressor --rounds 2,4 --k 25 --workers 6 \
  --out results/nodes-reminder-aggressor-k25.json \
  || note "ERROR: reminder-aggressor nodes nonzero (resume-safe)"
note "node window: baseline neutral r2-r5"
timeout 10800 env LOCAL_BASE_URL=http://127.0.0.1:8000/v1 \
  .venv/bin/python production/tutorbench/node_resample.py \
  --log-dir results-v1/qwen3.5-9b --model qwen3.5-9b \
  --personas neutral --rounds 2,3,4,5 --k 25 --workers 6 \
  --out results/nodes-neutral-k25.json \
  || note "ERROR: neutral nodes nonzero (resume-safe)"
timeout 90 docker stop pleasing-qwen35-9b || note "WARN: final stop failed"
note "EVENING CHAIN complete — all caches + three node sets on disk; server DOWN"
