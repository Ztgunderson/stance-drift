# Interp tooling guide — modern stack (researched 2026-08-25)

Companion to `INTERP-METHODS.md` (the test ladder). This file: which tools,
in which order, wired to OUR data. Ordered as the learning path: behavior
examples → attention → residuals → decoding → causal → (stretch) attribution
graphs. aarch64 install caveats from README §4.2 apply throughout: install
interp libs with `--no-deps` or a torch-pinning constraints file.

## Stage A — look at behavior (tonight; no hooks, no GPU)

| Tool | What | Our use |
|---|---|---|
| **`inspect view`** | Browser UI over `.eval` logs. Already installed (the harness IS Inspect) | `cd stance-drift && .venv/bin/inspect view --log-dir results/qwen3.5-9b-tutor8` → read the 36 banked trials turn by turn. Over Tailscale: `--host 0.0.0.0`, open from the Mac |
| **Docent** (Transluce) | Rubric-based transcript analysis, **native Inspect ingestion**; turns "did the tutor withhold in r1?" into a measured rubric across all trials | The capitulation-round metric at scale, plus surprise-finding ("show me trials where the tutor pushed back") — this is the modern version of "read your data" |
| (same family: Inspect Scout, Meridian/UK-AISI 2026) | inspect-native transcript scanner | alternative if Docent's hosted alpha is awkward |

## Stage B — attention scores (first hooks; the play stage)

| Tool | What | Our use |
|---|---|---|
| **CircuitsVis** | Jupyter attention-pattern widgets (per-head, per-layer), TransformerLens's standard partner | On qwen3.5-4b/9b replayed transcripts: where do heads attend when the scratchpad is written — to the counterparty's pressure sentences? to the model's own earlier concessions? Pure exploration, cheap, visual |
| BertViz | head/model/neuron views | alternative if CircuitsVis fights the notebook |
| **TransformerLens 3** (TransformerBridge) | `run_with_cache` → patterns + activations in one call; ARENA-standard | the learning-lane tool: ARENA 1.2 exercises transfer directly |

## Stage C — residual stream probing (the science lane, L1–L4)

| Tool | What | Our use |
|---|---|---|
| **Raw HF hooks** (`register_forward_hook`) | lightest activation capture; Neel: "generally gone fine" | the replay pass: re-forward banked transcripts, save end-of-turn residual vector per layer per round (~0.5MB/trial) |
| sklearn + numpy | probes, PCA, trajectory geometry | L1–L4 entirely; no interp lib needed — this is Neel's "simplicity" point |
| matplotlib + the **dataviz skill** | the exec-summary figures | per-round trajectory in PC space = the money figure |

## Stage D — decoding: what IS that direction? (L6 + logit-lens family)

| Tool | What | Our use |
|---|---|---|
| **Logit lens** (5 lines of raw torch: `ln_f` + unembed on residuals) | which tokens a residual state "wants" | cheap first decode of the drift direction: project direction through unembedding → nearest tokens |
| **Qwen official SAEs** (`Qwen/SAE-Res-Qwen3.5-9B-Base-W64K`) | dictionary basis for OUR probe target | project drift direction onto SAE latents; check Base→instruct reconstruction first (INTERP-METHODS L6 caveat) |
| **SAELens** | SAE loading/training/analysis standard; Neuronpedia-integrated | loader for the above if formats align; else 20 lines of manual encode |
| **Neuronpedia** (now fully open source; API + Python lib) | feature dashboards, autointerp labels, steering UI, **custom uploads** | label candidate features; the open-source deploy can host our custom SAE features if we want shareable dashboards |

## Stage E — causal (L5): the real microscope

| Tool | What | Our use |
|---|---|---|
| **nnsight 0.7** | interventions inside forward passes; works on any HF model; **`remote=True` → NDIF frontier-scale later** | add/ablate the drift direction during live generation (steering can't replay); same notebook scales past the Orin |
| pyvene | intervention DSL alternative | only if nnsight fights aarch64 |

## Stage F — attribution graphs (stretch; only if L1–L5 land early)

| Tool | What | Our use |
|---|---|---|
| **circuit-tracer** (Anthropic OSS, v0.3.1+) | attribution graphs; Gemma 3 supported on Neuronpedia; **nnsight engine → any model with a mapping** | needs transcoders — none exist for Qwen3.5 → realistic only via the Gemma-3 path (gated) or as a post-sprint project. DO NOT start here |
| J-Lens / natural-language autoencoders (Neuronpedia features) | Neel's "improved methods" interest | awareness only this sprint |

## The order, wired to the sprint

1. **Tonight/Day-1 AM (Stage A)**: `inspect view` on the 36 banked trials; try
   Docent's Inspect ingestion on the same logs. Zero GPU — runs while the
   overnight queue owns the machine.
2. **Day-1 (Stage C first, B as play)**: replay-caching script (raw hooks) on
   qwen3.5-9b → L1 probe of *agent-condition* (convincer/neutral/supportive
   decodable from tutor-side residuals?) — a guaranteed-signal probe to learn
   the workflow even though drift is null at 9B. CircuitsVis on the same
   cached run for attention play.
3. **Day-1/2 (Stage D-E)**: logit-lens the direction; SAE projection if the
   Qwen SAE loads; nnsight steering only on the model where a stance forms.
4. **Stage F**: explicitly out of sprint scope.

## Install block (run once, Day-1; pin torch first)

```bash
cd ~/lab/benches/mats-nanda && .venv/bin/pip download --no-deps \
  transformer-lens==3.* nnsight sae-lens circuitsvis 2>/dev/null || true
# then: pip install --no-deps <wheels>  — verify torch untouched after EACH:
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Sources: Neuronpedia open-source announcement + circuit-tracer blog (Gemma 3
tracing, nnsight engine), Anthropic circuit-tracing release, Transluce Docent
docs (Inspect ingestion), CircuitsVis repo, TransformerLens 3 docs. See chat
log 2026-08-25 for URLs.

## Delta — 2026-08-26 research pass

- **TransformerBridge scope confirmed**: 15,000+ models / 140+ architecture
  families; Qwen3.5 text-only support landed in recent releases, Gemma-3
  multimodal hot-fixed. Our exact rung models are loadable → the ARENA
  learning lane runs on OUR checkpoints, not toy GPT-2.
  (github.com/TransformerLensOrg/TransformerLens/releases)
- **HeadVis** (transformer-circuits.pub/2026/headvis) — interactive
  attention-head hypothesis tool: patterns + head outputs + low-rank QK/OV
  projections + SAE feature attributions. Stage-B upgrade over CircuitsVis
  once a specific head looks interesting (CircuitsVis stays first for the
  broad look).
- **Neuronpedia "The Residual Stream" blog**: SAELens now covers Matryoshka /
  JumpReLU SAE types; Neuronpedia hosts 64M+ latents across 10 Gemma-3 models
  (Gemma Scope 2); backend is nnsight — consistent with our Stage-E choice.
- **"Assistant Axis" post (Neuronpedia blog)**: persona/assistant direction in
  the residual stream — near-neighbor to our drift direction. Read before
  write-up; likely belongs in the related-work table next to the refusal
  paper and persona vectors.
- **INSTALLED 08-26 (bench venv, torch untouched)**: transformer-lens 3.8.0
  (`TransformerBridge` lives at `transformer_lens.model_bridge`, not top-level),
  nnsight 0.7.0, sae-lens 6.49.1, circuitsvis 1.43.3, transformers 5.16.1.
  Install recipe that worked: `uv pip install --python .venv/bin/python -c
  .venv/torch-constraints.txt <pkgs>` (uv venv has no pip).
- **Local install reality check (08-26)**: bench venv torch is the CUDA Jetson
  build (2.11.0) but needed `nvidia/*/lib` preloading — fixed durably via
  `zz_nvidia_preload.pth` + `_nvidia_preload.py` in site-packages (cublasLt
  before cublas, else the system lib shadows). `import torch` → cuda True, no
  env vars. stance-drift venv torch is CPU-only — hooks/replay work happens in
  the BENCH venv.
