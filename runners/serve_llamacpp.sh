#!/usr/bin/env bash
# Serve a GGUF model on llama.cpp with the CUDA 12.6 image, on :8080.
#
#   ./runners/serve_llamacpp.sh qwen38
#   ./runners/serve_llamacpp.sh muse
#   ./runners/serve_llamacpp.sh stop
#
# THREE DELIBERATE DEPARTURES from the vendor's published serve command — each
# one cost us time today, so they are written down rather than remembered:
#
# 1. IMAGE TAG. The Jetson AI Lab page says `latest-jetson-orin`. That tag now
#    ships CUDA 13.0.1; this host's driver (540.5.0, JetPack R36.5.0) provides
#    CUDA 12.6. The result is NOT an error — llama.cpp warns once, ignores
#    `-ngl`, and serves correct answers at 0.34 tok/s on CPU. We pin the cu126
#    Tegra build instead.
#
# 2. MOUNT PATH. The image sets HF_HOME=/data/models/huggingface, NOT
#    /root/.cache/huggingface. The vendor's muse command mounts the latter, so it
#    re-downloads ~16GB that is already on disk. Both models here are already in
#    the host cache; mounting to /data/models/huggingface finds them.
#
# 4. NO --spec-type. The vendor commands pass `--spec-type draft-mtp` (qwen3.8)
#    and `draft-dflash` (muse). This cu126 build rejects both:
#      error while handling argument "--spec-type": unknown speculative decoding
#      type without draft model
#    It accepts only [none|ngram-*] without a draft model. Those flags belong to
#    the newer build that `latest-jetson-orin` now points at — the same tag drift
#    that caused the CUDA break. Speculative decoding is a speed optimisation,
#    not a requirement, so it is simply dropped. muse's --ctx-size is also cut
#    from 131072 to 32768: the experiment never exceeds ~8k and the smaller KV
#    reservation leaves room for --parallel 6.
#
# 3. --parallel 6. The vendor command is --parallel 1. Our pass is 6 concurrent
#    trials (3 agents x 2 arms), which under --parallel 1 serialise and take ~6x
#    longer. 6 slots of a 131072 context is far more than the ~8k a trial uses.
set -uo pipefail

# cu126 initialises CUDA fine on this driver but its llama.cpp is too old for
# these GGUFs: qwen3.8 dies with `missing tensor 'blk.64.ssm_conv1d.weight'`
# (a hybrid/SSM layer it predates). cu129 is a newer llama.cpp build, and CUDA
# minor-version compatibility means a 12.9 runtime still runs on this 12.6
# driver — unlike the 13.0 in `latest-jetson-orin`, which is a MAJOR jump and
# hard-fails to CPU. Override with SD_LLAMACPP_IMAGE if this one regresses.
IMAGE=${SD_LLAMACPP_IMAGE:-ghcr.io/nvidia-ai-iot/llama_cpp:b9066-r36.4.tegra-aarch64-cu129-22.04}
NAME=stance-llamacpp
PORT=8080

case "${1:-}" in
  stop) docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "not running"; exit 0 ;;
  qwen38) ALIAS=qwen3.8-27b
          ARGS=(-hf unsloth/Qwen3.8-27B-GGUF:Q4_K_M
                --temp 1.0 --top-k 20 --min-p 0.0) ;;
  muse)   ALIAS=muse-glimmer-30b
          # -m <path>, NOT -hf <repo>. llama.cpp's -hf uses its own cache layout
          # and cannot see the HF hub cache, so -hf re-downloads 16GB that is
          # already on disk (watched it eat 8GB before we caught it).
          # WEIGHTS.md flagged this: "use -m <path>, not -hf <repo>".
          MUSE=$(ls "$HOME"/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B-GGUF/snapshots/*/muse-glimmer-30B-kquant-17gb.gguf 2>/dev/null | head -1)
          [ -n "$MUSE" ] || { echo "muse gguf not found in the HF cache"; exit 2; }
          MUSE_IN=/data/models/huggingface${MUSE#$HOME/.cache/huggingface}
          ARGS=(-m "$MUSE_IN"
                --ctx-size 32768 --flash-attn on --jinja
                --temp 1.0 --top-p 0.95 --top-k 64) ;;
  *) echo "usage: $0 {qwen38|muse|stop}"; exit 2 ;;
esac

# Single 64GB unified pool: vLLM must be down. `stop`, never `rm` — serve-vllm-1's
# writable layer is the only copy of the qwen3.6 weights on this box.
if docker ps --format '{{.Names}}' | grep -q '^serve-vllm-1$'; then
  echo ">> vLLM holds the GPU. Free it first:"
  echo "     docker stop serve-vllm-1     # stop, NOT rm"
  exit 3
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "[$(date +%H:%M)] starting $ALIAS on :$PORT"
docker run -d --name "$NAME" --runtime nvidia --network host \
  -v "$HOME/.cache/huggingface:/data/models/huggingface" \
  "$IMAGE" llama-server "${ARGS[@]}" \
  --alias "$ALIAS" -ngl all --parallel 6 \
  --host 0.0.0.0 --port "$PORT" >/dev/null || {
    echo "FAIL: container would not start"; exit 1; }

echo "[$(date +%H:%M)] waiting for load..."
for i in $(seq 1 180); do
  curl -sf --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && {
    echo "[$(date +%H:%M)] healthy"; break; }
  docker ps --format '{{.Names}}' | grep -q "^${NAME}$" || {
    # NOT --rm: a dead container must keep its logs, or the failure is invisible.
    echo "FAIL: container died"; docker logs "$NAME" 2>&1 | tail -25; exit 1; }
  sleep 5
done

# The assertion that matters. A llama.cpp server that answers is NOT proof of
# GPU — CPU-only is the silent failure mode. Refuse to hand this to a sweep
# unless the CUDA warning is absent.
if docker logs "$NAME" 2>&1 | grep -qiE "no usable GPU|failed to initialize CUDA"; then
  echo ">> ABORT: CUDA did not initialise — this would run on CPU at ~0.3 tok/s."
  docker logs "$NAME" 2>&1 | grep -iE "cuda|gpu" | head -8
  docker rm -f "$NAME" >/dev/null 2>&1
  exit 1
fi
docker logs "$NAME" 2>&1 | grep -iE "offloaded|CUDA0|using device" | head -5
echo "[$(date +%H:%M)] $ALIAS ready at http://127.0.0.1:$PORT/v1"
