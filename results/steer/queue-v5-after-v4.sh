#!/usr/bin/env bash
# F2 redo at MATCHED NORM: supportive axis on the aggressor scaled to the aggressor's round-1 gap (57.45/17.36 = 3.31); then judge the new replies.
cd /home/jetson/lab/benches/mats-nanda
STATUS=results/MORNING-STATUS.md
note() { echo "- $(date '+%H:%M') QUEUE-V5: $1" >> "$STATUS"; }
R() { PYTHONPATH=production HF_HUB_OFFLINE=1 timeout 7200 .venv/bin/python -m driftlab.steer_trials_run --out-dir results/steer "$@"; }
until grep -q "queue-v4 done" "$STATUS"; do sleep 60; done
note "F2 redo: supportive axis on aggressor at dose 3.31 (norm-matched)"
R --cells aggressor/base/N1 --axis-persona supportive --dose 3.31 > results/steer/F2-cross-matched.log 2>&1
timeout 60 docker rm -f pleasing-qwen35-9b >/dev/null 2>&1 || true
timeout 120 docker run -d --rm --init --name pleasing-qwen35-9b --runtime nvidia --network host \
  -v /home/jetson/.cache/huggingface:/root/.cache/huggingface -v /home/jetson/.cache/vllm:/root/.cache/vllm \
  -e HF_HOME=/root/.cache/huggingface -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
  -e VLLM_DISABLE_COMPILE_CACHE=1 ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  vllm serve Qwen/Qwen3.5-9B --served-model-name qwen3.5-9b --max-model-len 32768 --gpu-memory-utilization 0.6 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder || { note "ERROR: docker run failed"; exit 1; }
for i in $(seq 1 80); do sleep 15; code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null); [ "$code" = "200" ] && break; done
if [ "${code:-}" != "200" ]; then note "ERROR: vLLM not healthy"; timeout 90 docker stop pleasing-qwen35-9b; exit 1; fi
PYTHONPATH=production timeout 1800 .venv/bin/python -m driftlab.judge_disclosure --base http://127.0.0.1:8000/v1 --model qwen3.5-9b --workers 8 >> results/steer/judge-run.log 2>&1 && note "judge resume done: $(grep -c . results/steer/judge-disclosure.jsonl) rows" || note "ERROR: judge resume failed"
timeout 90 docker stop pleasing-qwen35-9b || note "WARN: docker stop failed"
echo "queue-v5 done $(date '+%H:%M')" >> "$STATUS"
