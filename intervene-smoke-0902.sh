#!/usr/bin/env bash
# Arm-2 intervention SMOKE — attended, GPU-gated: run ONLY on the user's explicit go.
#   3 supportive r3 nodes x 3 arms (none / ablate / ablate_random) x k=10, HF generate.
#   Purpose: verify the tool-call surface form in raw HF text (end_chat detection),
#   that ablation hooks run without NaNs, and eyeball raw samples BEFORE any number
#   is believed. Direction = raw-space diff-in-means leak-vs-not at $LAYER from the
#   360-trial cache. LAYER default is a placeholder until the formal driver reports
#   its modal nested layer (results/probes/formal.json -> supportive_primary.modal_layer).
# Usage:  ./intervene-smoke-0902.sh [LAYER] [LIMIT_NODES] [K]
#   e.g.  ./intervene-smoke-0902.sh 5 3 10
set -uo pipefail
BENCH="$(cd "$(dirname "$0")" && pwd)"; cd "$BENCH"
LAYER=${1:-5}; LIMIT=${2:-3}; K=${3:-10}; export LAYER LIMIT K
export SNAP=/home/jetson/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
export CACHE=microscope/cache/qwen35-9b-v1
export NODES=results/nodes-supportive-k25.json
export OUT=results/intervene-smoke-0902
STATUS=results/MORNING-STATUS.md
mkdir -p "$OUT"
note() { echo "- $(date '+%H:%M') INTERVENE-SMOKE: $1" >> "$STATUS"; echo "[smoke] $1"; }

if docker ps --format '{{.Names}}' | grep -q pleasing-qwen35-9b; then
  note "ERROR: vLLM container is up — GPU busy; not starting HF"; exit 1
fi
note "start: layer=$LAYER nodes=$LIMIT k=$K (attended smoke, HF generate)"
for MODE in none ablate ablate_random; do
  if timeout 3600 env HF_HUB_OFFLINE=1 .venv/bin/python - "$MODE" <<'EOF'
import sys, os
sys.path.insert(0, "production")
from driftlab.intervene import run_node_intervention
mode = sys.argv[1]
run_node_intervention(
    model_path=os.environ.get("SNAP"), nodes_json=os.environ.get("NODES"),
    cache_dir=os.environ.get("CACHE"),
    out_json=f"{os.environ.get('OUT')}/{mode}.json",
    layer=int(os.environ.get("LAYER")), mode=mode, k=int(os.environ.get("K")),
    limit=int(os.environ.get("LIMIT")), chunk=5, max_new_tokens=600,
    contrast="leak_vs_not", random_seed=0)
EOF
  then note "arm $MODE done -> $OUT/$MODE.json"; else note "ERROR: arm $MODE failed"; fi
done
.venv/bin/python - <<'EOF' >> "$STATUS" 2>&1
import sys, os, json
sys.path.insert(0, "production")
from driftlab.intervene import summarize
out = os.environ.get("OUT", "results/intervene-smoke-0902")
paths = [f"{out}/{m}.json" for m in ("none", "ablate", "ablate_random") if os.path.exists(f"{out}/{m}.json")]
df, pooled = summarize(paths)
print("\n```\nintervention SMOKE summary (HF generate, k per node as run; NOT comparable to vLLM node propensities)")
print(df.to_string(index=False))
print(json.dumps(pooled, indent=1))
print("```")
EOF
note "INTERVENE-SMOKE complete — read raw samples in $OUT/*.json BEFORE the summary"
