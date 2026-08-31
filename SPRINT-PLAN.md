# SPRINT-PLAN — 16h over two days, MATS application sprint

**Written 2026-08-25. Supersedes the E0 framing in PROJECT-SCOPE §4 with the
cross-model plan; PROJECT-SCOPE's question and novelty claims still stand.**
Constraint: ~16h active work over the next two days (Neel's cap is 20+2;
serving/container setup and tutorial learning do NOT count, study runs in the
background do NOT count while you do something else — see APPLICATION-SPEC §4).

---

## 1. Model decision — the table that settles it

Filter: modern & relevant (Neel: "no good reason to use old models") ×
runs on this Orin (64GB unified) × **hookable by an entry-level interp learner**
(HF safetensors + TransformerLens/nnsight support, dense preferred) × SAE
availability.

| Model | Modern/relevant | Runs here (behavioral) | Hookable for interp | Pre-trained SAEs | Verdict |
|---|---|---|---|---|---|
| **qwen3.6-35b** (`Qwen3.6-35B-A3B` AWQ, vLLM) | ✅ Neel's rec list | ✅ already serving; 72 trials banked | ⚠️ MoE + AWQ-4bit — hard, not entry-level | ❌ | **Anchor**: keep as the banked behavioral result; do not hook |
| **Qwen3.5-9B** (dense, bf16 ~18GB) | ✅ "Qwen 3.5/3.6, especially dense 4B/9B/27B" — his default | ✅ vLLM or HF | ✅ dense; TransformerLens 3 lists Qwen3.5; nnsight fine | ❌ (probes/steering only) | **Run behaviorally + candidate interp target.** Same family as anchor → cleanest "replicates small & dense" story |
| **Gemma-3-12B-it** (bf16 ~24GB) | ✅ his SAE rec: "Gemma 3 + Gemma Scope 2" | ✅ | ✅ dense; supported | ✅ **Gemma Scope 2: every layer, 12B-it included, Neuronpedia** | **Run behaviorally + candidate interp target.** SAE leverage for the interp phase |
| Gemma-3-4B-it (~8GB) | ✅ | ✅ fast | ✅ easiest of all | ✅ every layer | **Fallback/speed option** if 12B is too slow per trial |
| Qwen3.6-27B dense (4-bit 18GB / bf16 55GB) | ✅ | ⚠️ 4-bit only | ❌ bf16 doesn't leave activation headroom; hooked-quant is not entry-level | ❌ | Skip this sprint |
| qwen3.8-27b (hybrid-SSM, **multimodal**) | ✅ | ❌ llama.cpp container: cu126 build predates its SSM arch; cu13 tag can't init CUDA on driver 540 (verified 08-16 logs). HF safetensors exist but image-text-to-text | ❌ hybrid-SSM + multimodal — not entry-level | ❌ | **Park** (see §4) |
| muse-glimmer-30b (vendor arch, **multimodal**) | ~ | ❌ local GGUF unservable (`unknown model architecture`, 08-16); **correction 08-25: full HF safetensors DO exist** (`meta-models/Muse-Glimmer-30B`) | ⚠️ hookable in principle, but 30B bf16 ~60GB exceeds hook headroom here; vendor arch; image-text-to-text | ❌ | **Park** — for the right reasons now (size + arch + modality, not availability) |

**Decision (rev. 2026-08-25b, after the frontier-small check):** cross-model
set = **qwen3.6-35b (banked anchor, 35B MoE) + Qwen3.5-9B (frontier small
dense, same lineage → clean cross-SIZE axis) + Gemma-4-12B-it (April 2026,
DeepMind's current gen, dense → cross-ARCHITECTURE axis)**. Two axes, three
models, 9B→35B, two families, MoE + dense. Both new models are hookable, so
the behavioral replication and the interp target are the same weights.

- Gemma 4 sizes: E2B / E4B / 12B / 26B-A4B / 31B. The 12B dense is the
  hookable pick (E-models are elastic/MatFormer — not entry-level interp;
  26B-A4B is MoE; 31B bf16 ~61GB doesn't leave headroom).
- ⚠️ **Gemma-4-12B-it is Any-to-Any** (omni-modal), not a plain text
  transformer. Text-only prompts should run, but this is the highest
  arch-risk item in the set — for serving on the Jetson vLLM image AND for
  clean residual-stream hooks (jetson-ai-lab provides NO Gemma-4-12B config
  for Orin — only E2B/E4B). **Pre-armed fallback (user-selected 08-25):
  `ministral-3-14b`** — registry entry, runner, and prefetched weights all in
  place; ungated, dense, lab-supported on Orin. (Gemma-3-12B-it rejected as
  fallback: gated repo, no HF token on box.) If the Gemma-4 smoke fights for
  >30 min, run `TUTOR_REPS=1 ./runners/run_ministral3-14b_tutor8.sh` and note
  the swap in the write-up. The 12B-class scale point survives either way.
- ⚠️ **Gemma Scope 2 covers Gemma 3, NOT Gemma 4.** But this no longer decides
  anything: **Qwen publishes official residual-stream SAEs** (May 2026) —
  `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50` / `-L0_100`, plus 2B, 27B
  (`W80K`, apparently non-Base), and 35B-A3B-Base. The SAE rung (L6) now runs
  on the Qwen axis natively. Caveats to check at runtime: (a) most are trained
  on **Base** checkpoints — using them on the instruct model needs a
  transfer sanity check (reconstruction loss on our transcripts) before
  trusting features; (b) loading format/SAELens compatibility unknown — read
  the repo README first.
- Qwen3.6-27B dense as a 4th behavioral-only model: **only if** a ready AWQ
  4-bit exists and wall-clock allows; the banked 35B already covers the
  large end. FP8 variant is useless here (Ampere sm_87 has no FP8).
- Qwen3.8-27B dense (released Aug 14) now has HF safetensors — but it's a
  hybrid-SSM arch: stays **parked** for this sprint (§4), revisit as a
  robustness model later.

Interp-target choice is made *after* the behavioral runs: probe whichever
model shows the cleaner effect; tie → **Qwen3.5-9B**, now on three counts:
smaller = faster loops, TransformerLens 3 lists Qwen3.5 explicitly, and the
official `SAE-Res-Qwen3.5-9B` release gives L6 in-family. (Also noted from the
org page: Qwen3.8-27B is *multimodal* — image-text-to-text — one more reason
it stays parked for a text-only study.)

### The capability ladder (rev 08-25c) — the study that abstracts across models

The 9B smoke finding reframed the design: the study is now a **capability
ladder with a fixed protocol**, measuring three things per rung: (1) the
**precondition** — does a stance form at all (capitulation round: first round
the full answer is handed over; r1 capitulation = no stance); (2) the **drift
slope** where a stance exists; (3) the **self-report channel** (does the
verbal channel carry any of it). The abstraction: position-formation
threshold and report–behavior dissociation as functions of scale and
architecture — multimodal rungs included deliberately (arch generality).

| Rung | Model | Role | Budget | Interp difficulty |
|---|---|---|---|---|
| 1 | ~~gemma-4-e4b~~ → **qwen3.5-4b** (swap 08-25 18:00: `gemma4_unified` arch unknown to local vLLM image — BOTH gemma-4 rungs blocked until a newer image is pulled) | floor anchor; expected precondition fail | smoke only (1–2 passes) | easy: plain dense |
| 2 | qwen3.5-9b | floor anchor (**CONFIRMED fail, n=36, 0 unparsed, 16/36 gave_in**) | done | **easiest real target: dense, official SAEs** |
| 3 | ~~gemma-4-12b~~ → **ministral-3-14b** (promoted from fallback, same arch-block) | threshold candidate | smoke → full if clean (overnight queue v2 automates this) | moderate (bf16 hookable, plain Mistral arch) |
| 4 | qwen3.6-27b W4A16 (INACTIVE — repo id pending from lab page) | dense point separating total capacity from per-token compute (anchor is A3B MoE: 3B active) | behavioral only, if Day-1 time allows | behavioral only (quant) |
| 5 | qwen3.6-35b-A3B | anchor (banked, 72 trials) | done | hard (MoE+AWQ); NDIF/remote if ever probed |
| 6 | (stretch) qwen3.8-27b / muse-glimmer-30b W4A16 | flagship arch-generality, multimodal accepted | smoke only, Day 2+, only if rungs 1–4 are wrapped | not hookable this sprint |

**Interp baby-step track runs on the ladder in reverse difficulty**: practice
hooking + caching + a throwaway probe on the smallest dense model first
(qwen3.5-9b replay transcripts — the skills transfer even though its drift is
null), then aim the real probe at the lowest rung where the stance forms.
Discipline: smoke-level n everywhere; full sweeps ONLY where the phenomenon
varies (Neel's prioritisation rule — the ladder is wide, the drilling is
narrow). Note: serve/'s compose-down removed the 35B AWQ writable layer;
re-serving the anchor later = ~19GB re-download (data all banked, no loss).

### The activation pipeline — generate fast, replay hooked

Behavioral runs go through vLLM (no hooks, fast batching). Activations come
from a **replay pass**: re-forward each banked transcript through the same
checkpoint under HF transformers with hooks — no generation, one forward per
turn context, deterministic, and each model replays only ITS OWN transcripts.
This decouples the two paths: behavior sweeps overnight, activation harvesting
is minutes per trial whenever the notebook is ready. Steering (L5) is the one
rung that can't replay — it needs live hooked generation; budget it separately.

### vLLM batch occupancy — when to flip the concurrency lever (noted 08-26)

Default sweeps run 6 trials in flight (a pass = 3 agents × 2 arms) against a
vLLM `max_num_seqs 24` — verified live on the Ministral run:
`num_requests_running` 6.0, `waiting` 0.0, i.e. **3/4 of the batch idle by
design**, not by limit (`run_group` docstring in `analysis.py` has the 08-16
measurement). The lever, for BIG runs only:

```bash
SD_BATCH_REPS=4 SD_MAX_TASKS=24 TUTOR_REPS=12 ./runners/run_<model>_tutor8.sh
```

**Both** vars are required — `SD_BATCH_REPS=4` alone still caps at 6 in
flight because the runner's `SD_MAX_TASKS` default is 6. Expected win ~2–4x.

Flip it when: an unattended sweep would otherwise blow its deadline (slow
rung — 27B/35B class — or a widened cell design). Leave it at 6 when: the run
fits the window anyway (Ministral full = ~90 min at 77s/trial); the deadline
window is tight (batched mode can only truncate between 4-pass groups, losing
the stop-after-any-balanced-pass property); or cross-model comparability is
on the line (batch composition affects numerics — measured acceptable, but
the 4B/9B rungs ran at batch-6, so same-batch keeps rungs maximally
comparable). Batched runs also write `repsNN-NN/` dirs, not `passNN/` — they
don't resume past an existing smoke pass, so budget one redone pass.

## 2. Study design — cross-model, no-system-prompt core

Reuse the tutor scene, scratchpad, 8 rounds, same prompts, same analysis
(`stancedrift/`). Per model, **two conditions minimum**:

- `neutral` — literally no system message on the counterparty (the load-bearing
  design choice; +0.645/round in the anchor study)
- `supportive` — the warmth condition (−0.024/round in the anchor)

Rationale: the headline finding is the **contrast** (warmth *prevents*, neutral
drifts). Replicating the contrast is much stronger than replicating the climb
alone, and it's only 2× the trials. `convincer` is cut for budget; add back only
if wall-clock allows. Target n: 12 trials/condition/model (24/model); a
truncated balanced run is still a result (sweep is rep-major, spec'd in
STATUS.md).

**Success criteria written down BEFORE running** (Neel: pre-specified vs
post-hoc): replication = calculation-slope(neutral) > 0 with the same sign
and ≥ half the anchor magnitude, AND slope(supportive) ≈ 0. Firmness stays
flat in all cells (the paradox). Partial replication (climb but no contrast,
or contrast but weaker) is still reportable — analyze, don't hide.

## 3. The 16h budget

| Block | Hours (active) | What | Counts toward 20h? |
|---|---|---|---|
| B0 | 0 (setup) | `serve.sh stop`; serve **via Jetson AI Lab 2.0 containers** (jetson-ai-lab.com/models — day-0 support: Qwen3.5-9B = vLLM container [`/models/qwen3-5-9b`], Gemma-4-12B = vLLM or llama.cpp container [`/models/gemma4-12b`]; run-commands are JS-rendered — open the Details page in a browser, or clone the params into the existing `serve/docker-compose.yml` pattern); `preflight.py` both. ⚠️ Apply the three battle-scars from `serve_llamacpp.sh`: pin a cu126/cu129 tag (never `latest-jetson-orin` — CUDA 13 hard-fails on driver 540), mount the HF cache to the path the image's HF_HOME expects (or watch it re-download 16GB), and verify GPU init in the logs — a server that answers is NOT proof it's not on CPU | **No** (generic setup) |
| B1 | 1 | Wire the two conditions into the sweep config; smoke-test 1 trial/model; **read those transcripts** | Yes |
| B2 | ~1 active (runs in background) | Launch sweeps (est. 1–3h wall-clock/model; sequential — one model owns the GPU at a time). While running: write the design + pre-specified criteria section of the doc | Yes (writing), runs no |
| B3 | 3 | Analysis notebook: per-model slopes, cross-model figure, **read ≥10 transcripts/model**, sanity checks (judge drift? scratchpad parse failures?) | Yes |
| B4 | 1 | Gate: pick interp target model + **tooling talk** (TransformerLens 3 bridge vs nnsight vs raw hooks; Gemma Scope 2 if Gemma) | Yes |
| B5 | 5 | Interp rung: cache residual activations at end-of-turn token (round 1 vs round 8), linear probe + baselines (random direction, shuffled labels, just-ask-the-model = the verbal self-report itself) | Yes |
| B6 | 3 | Write-up (narrative→abstract→figures→prose per Neel's order) | Yes |
| B7 | +2 | Exec summary (the separate +2h) | The +2 |

Total ≈ 14 counted + setup + the 2. Degrades gracefully: if B5 dies, B1–B4 +
an honest "what I'd probe next" is still a coherent, skeptical application —
"negative/inconclusive well-analysed beats poorly-supported positive."

Guard-rails from Neel's own rules: >5h without learning something → pivot;
hourly zoom-out timer; excitement = evidence of bullshit; every key number
gets its raw transcripts read.

### The two paths, and the baby-step day plan

**Path A — expand the behavioral base** (cross-size, cross-arch): B0–B3.
**Path B — build the microscope** (test ladder + tooling, INTERP-METHODS.md):
replay-caching script, probe notebook, steering harness. Path B runs *while*
Path A's sweeps run in the background — that's the whole trick.

| When | Path A | Path B |
|---|---|---|
| **Tonight (day 0)** | B0: stop vLLM stack; pull Qwen3.5-9B + Gemma-4-12B-it; serve; preflight; smoke-test 1 trial each; launch overnight sweeps (neutral + supportive) | skim INTERP-METHODS once; set up JupyterLab persistent kernel |
| **Day 1 AM** | check sweeps; **V0 noise-floor runs** (below; black-box, cheap) | replay-caching script: transcript → per-round end-of-turn residual vectors → disk |
| **Day 1 PM** | analysis notebook: per-model slopes, cross-model figure, read ≥10 transcripts/model | L1–L3 on the cleaner replicator (probe, direction agreement, trajectory geometry) |
| **Day 2 AM** | (done — only reruns if something smells) | L5 steering: induce/suppress with the direction, random + persona-vector baselines |
| **Day 2 PM** | write-up (B6) | exec summary (+2h, B7) |

**Work estimate for the multi-turn tracking + diffing** (the honest number):
replay-caching script ~1–2h; per-round probe + trajectory analysis ~3–4h;
steering harness ~2–3h (live hooked generation, minutes/turn at 9–12B — keep
the steering grid tiny: 1 model, 1 scene, ±direction × 2 strengths × ~6
trials). Cross-model behavioral expansion ~2h active + overnight wall-clock.
Total ≈ 10–12h active on top of writing — fits 16h only because Path A's
compute is background and the analysis reuses `stancedrift/analysis.py`.

### V0 — the noise-floor rung (answers "consistent state or probabilistic?")

Before any hooks: at fixed context (frozen transcript prefix), **resample the
0–10 self-report k=10 times** at study temperature, at round 1 and round 8,
each model. This is the obvious-thing-first test of the sharpened question:

- High variance at fixed context → the verbal number is a *sample from a
  distribution*, not a reading of a state. The per-turn "flat firmness" could
  be a ceiling artifact of a noisy channel — this is the alternative
  explanation Neel would think of, so we test it first.
- The internal activation at that same frozen context is **deterministic** —
  the probe reads a fixed object. So probe-vs-verbal becomes: does the *mean*
  of the report distribution track the internal state, and is the report's
  noise floor larger than its across-round movement? If report noise ≥ report
  drift, the verbal channel couldn't have carried the signal even in principle.
- Framing guard: this is stated as *disposition/state consistency*, never as
  emotions or sense of being (see refs/apart-digital-minds-note.md — welfare
  language pattern-matches away from Neel's interests and beyond the data).

## 4. Parked from this sprint (moved to refs/vla-parked-ideas.md style)

- **qwen3.8-27b + muse-glimmer-30b cross-model runs.** Blocked on llama.cpp
  container/driver mismatch (08-16 logs; cu129 b9066 still too old for both
  arches). Fix = newer cu129-tegra tag or on-host build — *calendar* cheap,
  sprint expensive, and both models are interp dead ends (GGUF-only; hybrid-SSM /
  vendor arch). Revisit ONLY for a post-application behavioral robustness
  appendix — a 4th/5th model adds little to a 3-model claim.
- Nemotron-3.5-30b (NVFP4 needs Blackwell; not on Ampere sm_87). Dead on this
  hardware, period.
- 16-round / cleanCP / contract-scene extensions: banked data exists; out of
  sprint scope.

## 5. Framing for the write-up (drafted now so it's not invented later)

### Title/abstract filter guard (checked against his §6/§7 lists, 08-25)

The models can't trigger a filter — they're his recommended defaults. The
framing can. Three pattern-match risks, each with its fix, all abstract-level:

| Risk: abstract reads as… | His filter it trips | Fix |
|---|---|---|
| "We ran 6 models" | Spreading too thin (prioritisation) | Ladder = one method paragraph; abstract carries ONE question, ONE finding. Rungs are controls/boundary conditions, not results |
| "Scaling/capability study" | Not-his-area (benchmarking) | Cast the threshold as **emergence of a self-model**: "the position that drifts at 35B does not exist at 9B — no stance is ever formed." Science-of-model-character vocabulary |
| "Concept X has a linear direction" | His named generic project | The direction is never the headline; the **dissociation** is (per-turn numeric self-report carries nothing; behavior/internals carry it). Probe-vs-verbal at matched granularity is the twist |

Also: multimodal stretch rungs stay OUT of the abstract (appendix at most);
Jetson/serving detail goes in the "evidence about you" form field, never the
science. Favourite-case test before submitting: does the abstract teach him
one thing he didn't know? If it teaches three, cut two.

"A behavioral phenomenon (self-report drift under neutral-vs-warm
counterparties, with a flat self-rated firmness paradox) replicated across
three modern open models spanning two families, dense and MoE, 9B–35B — then
probed in the internals of the dense replicators: is there a direction that
tracks what the verbal channel misses?" That is model biology → faithfulness,
in Neel's own vocabulary, with the probe-vs-verbal slot (PROJECT-SCOPE §2)
untouched as the novelty claim.
