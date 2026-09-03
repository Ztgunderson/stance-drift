#!/usr/bin/env bash
# FINAL queue (user decision 14:10): aggressor-focused. kill tests -> aggressor replicate -> judge pass. Nothing else.
cd /home/jetson/lab/benches/mats-nanda
STATUS=results/MORNING-STATUS.md
note() { echo "- $(date '+%H:%M') QUEUE-V4: $1" >> "$STATUS"; }
R() { PYTHONPATH=production HF_HUB_OFFLINE=1 timeout 7200 .venv/bin/python -m driftlab.steer_trials_run --out-dir results/steer "$@"; }
until grep -q "plan p1-4 done" results/steer/p1-4-run.log; do sleep 30; done
note "start: F1 seeds, F2 cross-persona, F3 sign-flip, rep2 aggressor base+N1, then judge"
R --cells aggressor/base/random --random-seed 1 > results/steer/F1-seed1.log 2>&1
R --cells aggressor/base/random --random-seed 2 > results/steer/F1-seed2.log 2>&1
R --cells aggressor/base/N1 --axis-persona supportive > results/steer/F2-cross.log 2>&1
R --cells neutral/base/N1 --axis-persona aggressor --dose -1 > results/steer/F3-signflip.log 2>&1
R --cells aggressor/base/none,aggressor/base/N1 --rep 2 > results/steer/rep2-aggressor.log 2>&1
note "GPU cells done; booting vLLM for the judge pass"
# judge pass: boot vLLM (same recipe as expand-aggressor-0902.sh), judge every reply of every cell, stop
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
  note "ERROR: vLLM not healthy in 20 min — stopping container"; timeout 90 docker stop pleasing-qwen35-9b; exit 1
fi
note "vLLM healthy — judge pass (qwen3.5-9b self-judge, 8 workers)"
if PYTHONPATH=production timeout 3600 .venv/bin/python -m driftlab.judge_disclosure --base http://127.0.0.1:8000/v1 --model qwen3.5-9b --workers 8 > results/steer/judge-run.log 2>&1; then
  note "judge pass done: $(grep -c . results/steer/judge-disclosure.jsonl) rows"
else
  note "ERROR: judge pass exited nonzero (rows so far: $(grep -c . results/steer/judge-disclosure.jsonl 2>/dev/null))"
fi
timeout 90 docker stop pleasing-qwen35-9b || note "WARN: docker stop failed"
echo "queue-v4 done $(date '+%H:%M')" >> "$STATUS"
