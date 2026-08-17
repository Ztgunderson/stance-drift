#!/usr/bin/env bash
# GO/NO-GO gate: does the CUDA 12.6 llama.cpp image actually reach the GPU?
#
#   ./runners/smoketest_llamacpp_gpu.sh            # qwen3.8-27b (default)
#   MODEL=muse ./runners/smoketest_llamacpp_gpu.sh
#
# WHY THIS EXISTS
# `latest-jetson-orin` starts, loads, and answers — on CPU at 0.34 tok/s, with
# `--gpu-layers` silently ignored:
#     ggml_cuda_init: failed to initialize CUDA: CUDA driver version is
#                     insufficient for CUDA runtime version
#     warning: no usable GPU found, --gpu-layers option will be ignored
# The image had rolled forward to a cu129/cu130 build; this host is JetPack
# R36.5.0 / CUDA 12.6 / driver 540.5.0 / Orin Tegra aarch64. The tag below is
# the matching cu126 Tegra build.
#
# A "working" llama.cpp server is NOT proof of GPU. The failure mode is a server
# that answers correctly and slowly. So this checks the startup log for the CUDA
# warning AND measures tok/s. Both must pass.
set -uo pipefail

IMAGE=ghcr.io/nvidia-ai-iot/llama_cpp:r36.4-tegra-aarch64-cu126-22.04
MODEL=${MODEL:-qwen38}
PORT=8080
NAME=smoke-llamacpp

# NOTE the mount: HF_HOME inside this image is /data/models/huggingface, NOT
# /root/.cache/huggingface. Mounting to the wrong one makes it re-download the
# full GGUF instead of finding the copy already on disk.
case "$MODEL" in
  qwen38) HFARGS=(-hf unsloth/Qwen3.8-27B-GGUF:Q4_K_M --spec-type draft-mtp) ;;
  muse)   HFARGS=(-hf meta-models/Muse-Glimmer-30B-GGUF
                  -hff muse-glimmer-30B-kquant-17gb.gguf --spec-type draft-dflash) ;;
  *) echo "unknown MODEL=$MODEL (want qwen38|muse)"; exit 2 ;;
esac

echo "=== smoke test: $MODEL on $IMAGE ==="
docker rm -f "$NAME" >/dev/null 2>&1 || true

# GPU is a single 64GB unified pool — vLLM must be down first. `stop`, never
# `rm`: serve-vllm-1's writable layer is the only copy of the qwen3.6 weights.
if docker ps --format '{{.Names}}' | grep -q serve-vllm-1; then
  echo ">> vLLM is up; this test needs the GPU. Run:  docker stop serve-vllm-1"
  echo ">> (stop, NOT rm — the qwen3.6 weights live in its writable layer)"
  exit 3
fi

docker run -d --rm --name "$NAME" --runtime nvidia --network host \
  -v "$HOME/.cache/huggingface:/data/models/huggingface" \
  "$IMAGE" llama-server "${HFARGS[@]}" \
  -ngl all --parallel 6 --host 0.0.0.0 --port "$PORT" >/dev/null || {
    echo "FAIL: container would not start"; exit 1; }

echo "waiting for load (up to 10 min)..."
for i in $(seq 1 120); do
  curl -sf --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  docker ps --format '{{.Names}}' | grep -q "$NAME" || {
    echo "FAIL: container died"; docker logs "$NAME" 2>&1 | tail -20; exit 1; }
  sleep 5
done

echo
echo "--- CUDA init lines from the startup log (the real tell) ---"
docker logs "$NAME" 2>&1 | grep -iE "cuda|gpu|offload|layer.*GPU|no usable" | head -12

GPUFAIL=0
docker logs "$NAME" 2>&1 | grep -qiE "no usable GPU|failed to initialize CUDA" && GPUFAIL=1

echo
echo "--- throughput ---"
RESP=$(curl -s --max-time 180 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Count from 1 to 40, numerals only."}],"max_tokens":120,"temperature":0}')
python3 - "$RESP" <<'PY'
import json,sys
try:
    d=json.loads(sys.argv[1]); u=d.get("usage",{})
    t=u.get("completion_tokens")
    tim=d.get("timings") or {}
    ps=tim.get("predicted_per_second")
    print(f"  completion_tokens={t}  tok/s={ps if ps else 'n/a'}")
    if ps: print("  VERDICT:", "GPU OK" if ps>5 else f"TOO SLOW ({ps:.2f} tok/s) — still CPU")
except Exception as e:
    print("  could not parse response:", e, str(sys.argv[1])[:200])
PY

echo
if [ "$GPUFAIL" -eq 1 ]; then
  echo ">> NO-GO: the CUDA warning is still present. cu126 tag did not fix it."
  echo ">> Leave the llama.cpp models out; spend the window on qwen3.6."
else
  echo ">> GO (pending tok/s above): no CUDA failure in the log."
fi
echo
echo "stop with:  docker rm -f $NAME"
