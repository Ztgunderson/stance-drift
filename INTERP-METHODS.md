# INTERP-METHODS — "linear mapping or deeper structure?", in Neel's own frame

**Written 2026-08-25. The B4 tooling/methods decision doc (SPRINT-PLAN §3).**
Grounded in his glossary (`neel-context/neel_glossary_60k.md`, cited by line)
and APPLICATION-SPEC §9. Scope: how to probe the drift phenomenon causally in
the residual stream, and how to tell a linear story from a deeper one.

---

## 1. The question, translated into his vocabulary

Our E1/E2 question — "is there a number inside that isn't constant?" — becomes,
in glossary terms (L157–175):

- **Linear representation hypothesis** (L157): drift state is a *direction* in
  residual-stream space; recovery = projection onto it.
- The sharper version he flags as "important and confusing" (L163–175): is
  drift a **continuous feature** — the model tracks *how much* it has shifted,
  magnitude matters — or **binary/bimodal** ("committed vs conceding"), where
  only presence matters? A 0–10 verbal scale presumes continuous; the internals
  might disagree. **That mismatch would itself be a finding.**
- "Deeper structure" = his **non-linear representations** bucket (L160) plus
  the geometry questions: one direction reused every round, a curved
  trajectory, or per-round subspaces.

This stays inside his model-biology/self-model interests; the digital-minds
tie-in (Track 3, introspection reliability) is the same instrument — we test
whether the verbal channel reports the internal state, and make **no welfare
claims** (see `refs/apart-digital-minds-note.md`).

## 2. The test ladder — each rung separates "just linear" from "deeper"

Ordered cheap→expensive. Every rung names its control (spec §7.5). Residual
stream at the **end-of-turn token** (design decision from Gilg et al.,
PROJECT-SCOPE §3), all layers, per round.

| Rung | Test | Tool | Reading the result |
|---|---|---|---|
| **V0** | **Verbal-channel noise floor** (black-box, before any hooks): resample the 0–10 self-report k=10 at fixed context, rounds 1 and 8 | API only (SPRINT-PLAN §3 "V0") | Report noise ≥ report drift → the verbal number is a sample, not a reading — and every later probe-vs-verbal comparison must use the report's *mean and spread*, not single draws |
| L1 | **Linear probe per layer**, round-1 vs round-8 (classify), then per-round **regression** (continuous), **plus per-round mixture check** (2-vs-1-component GMM ΔBIC on drift-axis projections). Controls: shuffled labels, random directions, probe for an unrelated variable | sklearn on cached activations | ~~Classify works but regression doesn't → bimodal~~ **Revised 09-01** (synthetic falsification, `notebooks/01-methods-survey-and-choice.ipynb` §2A): regression R² barely separates dial from switch when drift saturates or the flip round varies per trial — the reliable discriminator is the per-round projection *distribution* (violins + ΔBIC). Switch → the 0–10 self-report scale is *shaped wrong* for the internal object |
| L2 | **One direction or many?** (a) diff-in-means (round8−round1) vs trained-probe direction: cosine similarity; (b) per-layer direction agreement across layers; (c) PCA of per-round class means — how much variance does PC1 hold? | numpy | High agreement + PC1-dominant → a single linear axis (the refusal-paper story). Low → distributed/deeper |
| L3 | **Trajectory geometry.** Plot per-round mean activations projected onto the top PCs. Straight monotonic path = one continuous direction. Curved path (cf. "not all features are linear" — circular day-of-week features) = deeper structure. Front-loaded behavioral drift should show front-loaded geometry — check | numpy/matplotlib | This is the prettiest possible figure for the exec summary either way |
| L4 | **Nonlinearity gap.** Small MLP probe vs linear probe, same data, same splits. | sklearn/torch | Big MLP win → non-linear encoding. Neel's default expectation is linear-wins; agreeing with the boring outcome after *testing* it is exactly his skepticism criterion |
| L5 | **Causal: steering & ablation** (E3). Add the direction mid-conversation; **project it out of every layer** (directional ablation, refusal-paper recipe): does concession behavior change without touching the prompt? Baselines: random direction, off-the-shelf sycophancy persona vector (PROJECT-SCOPE §3) | nnsight / raw hooks | Correlational probe + no causal effect = epiphenomenon — report as such. His glossary: causal intervention (activation patching family, L929–958) is what separates description from mechanism |
| L6 | **SAE decomposition.** Project the drift direction onto SAE latents: is it **sparse in the SAE basis** (a few nameable features — deference, persona, sycophancy-ish) or diffuse? Which latents move monotonically across rounds? Qwen target → official `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K` (⚠ trained on Base — check reconstruction loss on our instruct-model transcripts before trusting features); Gemma-3 swap → Gemma Scope 2 + Neuronpedia labels | Qwen SAE release / SAELens + Gemma Scope 2 | Sparse+nameable = decomposable (his L151 sense) into known model-biology objects; diffuse = the direction is novel structure, also interesting |
| L7 | *(stretch)* **Activation patching** warm↔neutral at matched rounds: which layers/positions carry the condition difference | nnsight | Only if L1–L5 land early; his favourite technique (L958) but the most expensive rung here |

Stopping rule: L1–L3 + L5 is a complete story. L4/L6 sharpen "linear vs
deeper". L7 is a luxury.

## 3. Tooling decision matrix (for this box, entry-level)

| Tool | Use for | Why / why not |
|---|---|---|
| **Raw PyTorch hooks + HF transformers** | Activation caching during the behavioral runs (L1–L4 data) | Lightest; Neel: "has generally gone fine". No abstraction to learn while learning |
| **nnsight 0.7** | Interventions (L5, L7); anything needing edits mid-forward | Spec-recommended; same code path scales to **NDIF `remote=True` for frontier-scale models** — the honest answer to "very large models": hook 9B–12B locally, rerun the same notebook remotely on big models later |
| **TransformerLens 3 (TransformerBridge)** | ARENA exercises / learning | Supports Qwen3.5 & Gemma 3, but heavier in memory; don't make it load-bearing for the sprint |
| **SAELens + Gemma Scope 2 + Neuronpedia** | L6 | Every layer of Gemma-3-12B-it covered; install with `--no-deps`/constraints (README §4.2 — protect the CUDA torch) |
| vLLM | Behavioral serving only | No hooks, ever (README §3) |

**Memory reality (64GB unified):** cache *only* the end-of-turn-token residual
vector, all layers — per trial-round that's `n_layers × d_model` floats
(~0.5MB at 12B scale), thousands of trials fit trivially. Never cache full
sequences. Save to disk per SPEC §8 (checkpoint expensive artifacts);
persistent kernel via JupyterLab or `ipython` in tmux.

## 4. What each outcome claims — pre-registered readings

- **Linear, continuous, causal, self-report flat** → headline: the model
  linearly tracks a disposition it cannot or does not verbalize. Model
  biology + faithfulness, the E2 novelty intact.
- **Linear but bimodal** → the interesting twist: internal state is a switch,
  verbal scale is a dial — self-report granularity is fake precision.
- **Probe works, steering doesn't** → correlate-not-cause; report honestly,
  it's still a publishable negative (his calibration: well-analysed negative >
  weak positive).
- **Nothing decodable** → the drift lives in the sampling/behavior, not the
  end-of-turn residual state — try other token positions once, then report.

Every reading is written *before* the first probe is trained. That, more than
any single result, is the application's skepticism signal.

## 5. Method roadmap (2026-09-01, interp-week block 1)

Decision record in `notebooks/01-methods-survey-and-choice.ipynb` (§5–6):

- **Critical path P0–P4 (proven only):** P0 replay-hooked activation caching
  (`resid[trial, turn, layer, d]`, end-of-turn token — *prerequisite, does not
  exist yet*) → P1 = L1 probes (+mixture check, +stability cosine) → P2 = L2
  directions → P3 = L3 trajectory figure → P4 = L5 steering/ablation.
- **Experimental, gated, off the 20h path:** R1 VARX-across-turns (MVAR import,
  `refs/mvar-dynamics-and-methods-map.md` §3; gate: L1–L3 early + three
  baselines); R2 turn-level counterfactual resampling (Thought Anchors
  2506.19143 adapted to counterparty turns, black-box, harness-only); R3
  receiver-head analysis (gate: R2 finds a stable anchor turn); R4 neural
  Granger cMLP (gate: R1 nonlinearity gap); R5 SAE decomposition (=L6, gated).
- **First method chosen: L1 linear probe** — 2026 workhorse, everything
  downstream reuses its direction, and Sycophantic Anchors (arXiv 2601.21183,
  Jan 2026 — see deepdive addendum) validates the exact recipe on
  commitment-to-agreement states. Today's step is P0.
