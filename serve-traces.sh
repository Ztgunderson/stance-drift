#!/usr/bin/env bash
# Rebuild the static inspect-view bundle(s) served through the review JupyterLab.
#
#   ./serve-traces.sh            rebundle banked 9B logs -> review/traces-9b
#   ./serve-traces.sh <log-dir> <name>   bundle any log dir -> review/traces-<name>
#
# Viewer URL (token in .jupyter-token):
#   http://100.76.200.13:8890/files/review/traces-<name>/index.html?token=<token>
# No extra server: JupyterLab (tailnet-bound, token-auth) serves the static bundle.
set -euo pipefail
BENCH="$(cd "$(dirname "$0")" && pwd)"
cd "$BENCH"
LOG_DIR="${1:-results-v1/qwen3.5-9b}"
NAME="${2:-9b}"
.venv/bin/inspect view bundle --log-dir "$LOG_DIR" \
  --output-dir "review/traces-$NAME" --overwrite
TOKEN="$(cat .jupyter-token 2>/dev/null || echo '<token>')"
echo "http://100.76.200.13:8890/files/review/traces-$NAME/index.html?token=$TOKEN"
