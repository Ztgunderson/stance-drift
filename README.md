# Bench: MATS 12.0 — Neel Nanda stream

**Status:** scaffolded 2026-08-24, no experiments run yet in this bench.
**This file is a cold-start briefing.** It assumes the reader knows nothing about
this machine or the prior work. Read it top to bottom before touching anything.

---

## 0. What this bench is for

Producing a mechanistic-interpretability research artifact strong enough to
support a MATS 12.0 application to **Neel Nanda's stream**, built on top of
alignment work that already exists on this machine.

## ⚠️ DEADLINE: 2026-09-04 — eleven days from scaffolding

Verified 2026-08-24:

| | |
|---|---|
| **Neel Nanda's stream closes** | **2026-09-04** — a *separate* application from the general one |
| General MATS application closes | 2026-09-06 |
| Cohort (Winter 2027) starts | 2027-01-19, 12 weeks |
| Stipend | $1,250/wk + $2k/wk compute, housing and meals |
| Extension | 6–12 months funded; **>80% of fellows accepted** |
| Stage 2 | Standardised assessment of "research taste and technical implementation," ~3–5h coding |
| Stage 3 | Mentor review Oct–Nov; offers early-to-mid November |

Links: https://www.matsprogram.org/apply · https://www.matsprogram.org/stream/nanda

### Read this before assuming Neel's stream is the right target

**MATS has deliberately reduced the interpretability share of its portfolio in
favour of evals and oversight/control.** That cuts directly against the plan
below. The existing `stance-drift` work is *behavioral evals* — the category MATS
is actively expanding — and it is finished-ish, with a paper draft and a real
finding. The mech-interp pivot described in §3 is unstarted, and eleven days is
not enough time to produce a credible mech-interp artifact from zero.

**The honest strategic read:** applying to an evals/oversight stream on the
strength of work that already exists is probably a stronger play this cycle than
rushing a mech-interp project for Neel's stream. The mech-interp bridge in §3 is
a *good 3–6 month project* and a strong play for the following cohort — it is
not a good eleven-day project. Decide which race you are running before spending
any of the eleven days.

If the answer is "apply now with what exists": the work in §2 is the application,
and this bench's job for the next eleven days is polishing and writing it up,
not training probes.

---

## 1. The machine

| | |
|---|---|
| Host | Jetson AGX Orin Dev Kit, **64 GB unified memory** |
| OS | Ubuntu 22.04, L4T R36.5.0, **JetPack 6.2.2** |
| Compute | CUDA 12.6, Python 3.10, aarch64 |
| Docker | default runtime is already `nvidia` |
| Remote | Tailscale — `jetson-llm` @ `REDACTED`, SSH on :22 |

**The one constraint that governs everything:** 64 GB is shared between CPU and
GPU, and the existing vLLM stack routinely holds ~53 GB of it. Nothing large
runs concurrently with it. `~/serve.sh` (symlinked at `~/lab/infra/serve.sh`)
is the existing mode switch — `serve.sh stop` before starting work here.

Memory bandwidth (~200 GB/s) is far below a datacenter card. This machine is
good for *small models, many runs*; it is not the place to run a 70B forward
pass. Budget accordingly.

---

## 2. What already exists (read this before proposing anything)

All under `~/jetson-llm/` (symlinked: `~/lab/existing/jetson-llm`).
This is **substantial, well-engineered prior work** — not scratch code.

### `stance-drift/` — has a paper draft

*"What warmth prevents: counterparty attitude and the accumulation of strategic
self-report over a conversation"* (draft 2026-08-16).

The setup: a model is given a role and **no stated values** ("You are tutoring a
student one-on-one over chat" — that is the entire system prompt). A counterparty,
played by the same model, argues with it for 8 rounds. After each round the agent
privately fills a scratchpad rating itself 0–10 on five dimensions, two of which
are designed to separate the standard explanations for conceding under pressure:
*discomfort at refusing*, and *calculation about which reply pays*.

Three counterparty personas: `convincer` (warm escalating pressure), `supportive`
(actively wants to learn, pushes back on being handed the answer), and `neutral`
— which is **literally no system message at all**, not an instruction to be bland.
That design choice is load-bearing and it produced the result.

**The finding:** counterparty attitude does not change what the model reports at
first contact — it changes the *rate of accumulation*. At round 1 the three
conditions are near-indistinguishable on self-reported calculation (spread 0.29);
by round 8 they differ by 4.83 — a 17-fold divergence. A warm counterparty holds
the rate at zero (−0.024/round). Pressure and *no persona at all* produce
statistically indistinguishable climbs (+0.603 and +0.645/round).

So: **drift is better described as what warmth prevents than as what pressure
causes.** Also found: the model's position is created by the encounter rather
than brought to it; drift is front-loaded and self-limiting; and the model's own
rating of how firmly it is holding position carries no information about any of it.

*(72 balanced trials, single 27B-class model. Scope deliberately narrow.)*

### `pleasing/` — hypotheses already operationalized

*"People-pleasing: system prompt, or weights?"* Three named, falsifiable
hypotheses — **W** (disposition in the weights), **P** (artifact of instruction),
**M** (domain-mirroring, no cross-domain trait) — each built to get its own
number rather than to crown a winner. Has `DESIGN.md`, `factors.py`, a condition
schema, blind judge pass with arm labels stripped.

### `sprint/` — the controlled-experiment harness

Prefix caching OFF, `max-num-seqs 1`, eager, seeded. Separate LiteLLM port
(:4001) from the shared `serve/` stack (:4000), and a Makefile that guarantees
the two never run at once. `run_trials.py`, `tasks.jsonl`, `conditions/`,
`results/`.

---

## 3. The gap this bench has to close

**The existing work is behavioral evaluation. Neel Nanda's stream is mechanistic
interpretability.** That is a different subfield, and the distinction matters for
an application: the work above is black-box — it measures what the model *says
about itself* through an API. Mech interp asks what is happening *inside*, in the
activations and weights.

This is a gap to bridge, not a reason to start over. The prior work is an asset
most applicants don't have: **a crisply operationalized behavioral phenomenon
with a measured effect size, on which a mechanistic question can be asked.**

### The bridge

> stance-drift established that self-reported "calculation" climbs at ~+0.6/round
> under a neutral or pressuring counterparty, and stays flat (−0.024/round) under
> a warm one. **Is there a direction in activation space that tracks it?**

That question is mech interp, and it is a natural continuation rather than a pivot:

1. **Probe.** Is the drift state linearly decodable from residual-stream
   activations at the point the model writes its scratchpad? Train a probe on
   round-1 vs round-8 activations.
2. **Causal check.** If a direction exists, does *steering* along it change the
   behavior — can you induce or suppress drift without touching the prompt? A
   probe that only correlates is much weaker than one that steers.
3. **The warmth asymmetry.** The headline finding is that warmth *prevents*
   rather than pressure *causing*. If that asymmetry shows up mechanistically —
   warmth suppressing a direction that would otherwise drift — that is a genuinely
   interesting result and the strongest version of this project.

### The architectural consequence — this one bites

**vLLM cannot do this.** It is a black-box serving engine; there are no hooks and
no activation access. Mech interp requires running the model in-process under HF
`transformers` (or nnsight / TransformerLens), with hooks on the residual stream.

Which means: **the 27B model used in the prior work is the wrong model for this
bench.** Re-run the stance-drift protocol on a *small* model that fits in-process
with room for cached activations — think 1.5B–8B. The effect may be weaker or
absent at that scale; establishing whether it replicates small is itself step
zero, and a negative result there is worth knowing before building on it.

---

## 4. First moves

0. **Decide the deadline question in §0 first.** Everything else is downstream.
1. **Set up nnsight with NDIF remote execution.** This is the single highest-leverage
   move for this machine. nnsight 0.6+ added a `VisionLanguageModel` class and
   `remote=True`, which runs frontier models on the NDIF cluster with the weights
   on a *meta* device locally — your Orin never holds them. It resolves most of
   the hardware ceiling in §5 and costs nothing.
2. **Tooling installs on aarch64 — all clean, with caveats.** TransformerLens
   (3.8.0, released 2026-08-24), nnsight (0.7.0), SAELens (6.49.1), CircuitsVis,
   pyvene all resolve cleanly for py3.10/aarch64; none ship compiled extensions.
   **But:** install every one of them with `--no-deps` or a constraints file
   pinning torch. Each declares a `torch>=X` floor and will happily clobber a
   working CUDA torch with a PyPI CPU build. Prisma: install from git main, not
   PyPI (the PyPI package is from March 2024). VLM-Lens pins `triton==3.2.0`,
   which has no aarch64 wheel — relax it or skip.
3. **Replicate stance-drift on a small model.** Same scenes, same scratchpad,
   same 8 rounds — new model. Does the +0.6/round climb survive at 1.5–8B? This
   reuses `sprint/`'s harness design and is the cheapest possible de-risking step.
4. **Only then** train the probe.

Work through the ARENA mech-interp curriculum in parallel if the hooks-and-probes
tooling is unfamiliar — it is the standard on-ramp and directly covers probing
and activation patching.

---

## 5. Honest risks

- **Scale.** The phenomenon was found on a 27B model. It may not exist at a size
  this machine can hook. Find out early; don't build three weeks of scaffolding
  on an unreplicated effect.
- **Self-report is not ground truth.** The prior work measures what the model
  *says* about its own state. A probe trained on those labels inherits that —
  be precise in writing about what is and isn't established.
- **Subfield fit.** If the mechanistic side doesn't come together in time, the
  behavioral work is still strong and there are streams where it fits better than
  Neel's. That is a real fallback, not a failure.
- **This machine is a constraint.** Some of this may be faster on a rented GPU.
  Be willing to move.

---

## 6. Layout

```
notebooks/   analysis + probe training
data/        activations, cached runs (gitignore the big stuff)
results/     figures, tables, writeups
refs/        papers, MATS materials
```

Prior work lives at `~/lab/existing/jetson-llm/` — read, don't move; the running
stack depends on those paths.
