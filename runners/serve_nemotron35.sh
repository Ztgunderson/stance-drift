#!/usr/bin/env bash
# Serve NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 on :8000 as the second model.
#
#   ./runners/serve_nemotron35.sh            # base attempt
#   SD_NEMO_SPEC=1 ./runners/serve_nemotron35.sh   # add speculative decoding
#
# Based on the vendor serve command, with two deliberate departures:
#
# 1. SPECULATIVE DECODING IS OFF BY DEFAULT. The vendor command carries
#    `--speculative_config.method dspark` plus a companion `...-DSpark` draft
#    model. "dspark" is DGX Spark; the method and the draft weights are both
#    Blackwell-oriented, and the draft model is a second multi-GB download. It is
#    a throughput optimisation, not a requirement, and it is the part most likely
#    to fail on this hardware — so the first attempt isolates the real question:
#    does NVFP4 load on Ampere at all. Set SD_NEMO_SPEC=1 to add it back.
#
# 2. --max-model-len 128000 -> 32768. The experiment never exceeds 32k, and on a
#    62 GiB shared pool the 128k KV reservation is the difference between fitting
#    and crash-looping. (serve/ learned this the hard way: at 0.85 utilisation
#    vLLM demanded 52.15 GiB of a 61.36 GiB pool and crash-looped 38 times.)
#
# KNOWN RISK: this GPU is sm_87 (Orin, Ampere). NVFP4 is an sm_100+ (Blackwell)
# format. vLLM may refuse outright, or dequantize at a speed cost. If it refuses,
# that is a real platform finding, not a misconfiguration — record it and move to
# one of the AWQ candidates instead.
set -euo pipefail

IMAGE=${SD_NEMO_IMAGE:-vllm/vllm-openai:v0.27.1}
NAME=stance-nemotron
PORT=8000

docker rm -f "$NAME" >/dev/null 2>&1 || true

ARGS=(
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
  --served-model-name nemotron35
  --host 0.0.0.0 --port "$PORT"
  --reasoning-parser nemotron_v3
  --trust-remote-code
  --max-model-len 32768
  --kv-cache-dtype bfloat16
  --gpu-memory-utilization 0.70
  --max-num-batched-tokens 16384
  --enable-prefix-caching
  --mamba-backend flashinfer
  --mamba-ssm-cache-dtype float16
  --enable-mamba-cache-stochastic-rounding
  --mamba-cache-philox-rounds 5
  --mamba-cache-mode align
)

if [[ -n "${SD_NEMO_SPEC:-}" ]]; then
  echo ">> speculative decoding ON (needs the DSpark draft model — extra download)"
  ARGS+=(
    --speculative_config.method dspark
    --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
    --speculative_config.num_speculative_tokens 5
    --speculative_config.kv_cache_dtype bfloat16
  )
fi

echo "[$(date +%H:%M)] starting nemotron35 on :$PORT ($IMAGE)"
docker run -d --name "$NAME" --runtime=nvidia --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  "$IMAGE" "${ARGS[@]}" >/dev/null

echo "[$(date +%H:%M)] waiting for load (NVFP4 on sm_87 is the open question)..."
for i in $(seq 1 120); do
  if curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q nemotron35; then
    echo "[$(date +%H:%M)] nemotron35 is serving on :$PORT"; exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "[$(date +%H:%M)] container exited — last 25 lines:"
    docker logs --tail 25 "$NAME" 2>&1
    exit 1
  fi
  sleep 15
done
echo "[$(date +%H:%M)] timed out; last 25 lines:"; docker logs --tail 25 "$NAME" 2>&1; exit 1
