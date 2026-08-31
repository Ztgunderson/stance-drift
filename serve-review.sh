#!/usr/bin/env bash
# Serve the review notebooks over Tailscale.
#
#   ./serve-review.sh          start (idempotent: kills a previous instance)
#   ./serve-review.sh stop
#
# Binds to the Tailscale IP only — nothing is exposed on LAN/WAN. The kernel
# is the bench venv (registered as "mats-bench"); the server comes from the
# stance-drift venv, which has jupyterlab.

set -euo pipefail
BENCH="$(cd "$(dirname "$0")" && pwd)"
JUPYTER=/home/jetson/lab/existing/jetson-llm/stance-drift/.venv/bin/jupyter-lab
IP="$(tailscale ip -4)"
PORT=8890
TOKEN_FILE="$BENCH/.jupyter-token"
PIDFILE="$BENCH/.jupyter-review.pid"
LOG="$BENCH/results/jupyter-review.log"

if [[ "${1:-}" == "stop" ]]; then
  [[ -f "$PIDFILE" ]] && kill "$(cat "$PIDFILE")" 2>/dev/null && rm -f "$PIDFILE" \
    && echo "stopped" || echo "not running"
  exit 0
fi

[[ -f "$PIDFILE" ]] && kill "$(cat "$PIDFILE")" 2>/dev/null || true
[[ -f "$TOKEN_FILE" ]] || head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$TOKEN_FILE"
TOKEN="$(cat "$TOKEN_FILE")"

nohup "$JUPYTER" --no-browser --ip "$IP" --port "$PORT" \
  --ServerApp.root_dir="$BENCH" \
  --IdentityProvider.token="$TOKEN" \
  >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
sleep 3
kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "FAILED — see $LOG"; exit 1; }
echo "http://$IP:$PORT/lab?token=$TOKEN"
