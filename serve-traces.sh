#!/usr/bin/env bash
# Rebuild the static inspect-view bundle(s) served through the review JupyterLab.
#
#   ./serve-traces.sh            rebundle banked 9B logs -> review/traces-9b
#   ./serve-traces.sh <log-dir> <name>   bundle any log dir -> review/traces-<name>
#
# Viewer URL (static server on the tailnet, port 7676 — Jupyter's /files/ CSP
# sandbox breaks the SPA, so bundles get their own server):
#   http://100.76.200.13:7676/traces-<name>/index.html
# Server (idempotent-ish; check first):
#   pgrep -f 'http.server 7676' || nohup python3 -m http.server 7676 \
#     --bind 100.76.200.13 --directory review >> results/traces-server.log 2>&1 &
set -euo pipefail
BENCH="$(cd "$(dirname "$0")" && pwd)"
cd "$BENCH"
LOG_DIR="${1:-results-v1/qwen3.5-9b}"
NAME="${2:-9b}"
.venv/bin/inspect view bundle --log-dir "$LOG_DIR" \
  --output-dir "review/traces-$NAME" --overwrite
pgrep -f "http.server 7676" >/dev/null || nohup python3 -m http.server 7676 \
  --bind 100.76.200.13 --directory "$BENCH/review" >> results/traces-server.log 2>&1 &
echo "http://100.76.200.13:7676/traces-$NAME/index.html"
