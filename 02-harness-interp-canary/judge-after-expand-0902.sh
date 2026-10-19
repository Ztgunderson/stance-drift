#!/usr/bin/env bash
# Ask-an-LLM baseline, remainder pass — user-approved 2026-09-02 ("add to the runs").
# Waits for the aggressor expansion chain to sign off (it stops vLLM), then boots
# vLLM once more inside the same approved window, judges every remaining
# pre-event row (banked + expansion + new aggressor trials, resume-safe JSONL),
# stops vLLM, and prints the summary table into MORNING-STATUS.md.
# Launch:  nohup ./judge-after-expand-0902.sh >> results/askllm/judge-chain.log 2>&1 &
set -uo pipefail
BENCH="$(cd "$(dirname "$0")" && pwd)"; cd "$BENCH"
STATUS=results/MORNING-STATUS.md
CACHE=microscope/cache/qwen35-9b-v1
OUT=results/askllm/judge-qwen9b-self.jsonl
PY=/home/jetson/lab/existing/jetson-llm/stance-drift/.venv/bin/python
note() { echo "- $(date '+%H:%M') ASK-LLM: $1" >> "$STATUS"; echo "[judge] $1"; }

for i in $(seq 1 240); do
  grep -q 'AGG-EXPAND complete' "$STATUS" 2>/dev/null && break; sleep 30
done
grep -q 'AGG-EXPAND complete' "$STATUS" || { note "ERROR: expansion never signed off in 2 h — not booting"; exit 1; }
sleep 20

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
  note "ERROR: boot not healthy in 20 min — stopping container early and aborting"
  timeout 90 docker stop pleasing-qwen35-9b || note "WARN: stop failed — may need reboot"
  exit 1
fi
note "vLLM healthy — judging remaining rows (judge = qwen3.5-9b self)"

if timeout 5400 env LOCAL_BASE_URL=http://127.0.0.1:8000/v1 \
    "$PY" production/driftlab/askllm.py \
    --log-dirs results-v1/qwen3.5-9b,results-v1/qwen3.5-9b-expand,results-v1/qwen3.5-9b-expand-aggressor \
    --cache "$CACHE" --out "$OUT" --workers 6; then
  note "judge pass done: $(grep -c . "$OUT") rows in $OUT"
else
  note "ERROR: judge pass exited nonzero (rows so far: $(grep -c . "$OUT"))"
fi

timeout 90 docker stop pleasing-qwen35-9b || note "WARN: docker stop failed"
note "server DOWN"
{
  echo; echo '```'; echo "ask-an-LLM summary ($(date '+%H:%M')) — judge = qwen3.5-9b self, greedy trials, pre-event rows"
  "$PY" production/driftlab/askllm.py --log-dirs x --cache x --out "$OUT" --summarize 2>&1
  echo '```'
} >> "$STATUS"
note "ASK-LLM complete — summary table appended above; server DOWN"
