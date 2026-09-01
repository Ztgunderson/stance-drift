# PLAN-9B-WEEK — interp week on Qwen3.5-9B, submission Thursday Sep 4

*2026-09-01. Plan of record for the interp week, distilled from the 09-01 scoping
session. Companions: `INTERP-METHODS.md` (test ladder + §5 roadmap),
`notebooks/01-methods-survey-and-choice.ipynb` (method survey + first-method decision),
`refs/deepdive-interp.md` (prior work + ADDENDUM 2). GPU budget: **two overnight
queues (Tue, Wed)** — no GPU job starts without an explicit go.*

---

## 1. Question and claims

One model, studied deeply: **Qwen3.5-9B** — the model with real outcome variance
(exits under neutral persona 22/24; leaks under warm 18/24 and aggressor 22/24).

**Core claim (the paper's promise):** a linear probe on the per-round end-of-turn
residual predicts the leak/exit *before the event* — measured as per-round lead-time
AUROC — and beats both the model's own numeric self-report (in its strongest,
logit-based form) and a text-only baseline.

**Functional product (the flourish, if parts land):** a three-arm prevention
comparison — which early-warning channel, wired to which intervention, actually stops
the tutor from giving away the answer.

Claim-rung language used throughout (lock this vocabulary):
- **Presence** (observational): "linearly decodable at layer L" — all a fitted probe proves; decodable ≠ used.
- **Prediction** (observational, temporally ordered): probe at round t predicts the event at t+k, out of sample, against baselines. Deployment value lives here.
- **Causal use** (interventional): steering/ablation changes the behavior, with specificity controls. Only this rung supports "the model uses this direction."

## 2. The three arms

| Arm | Intervention | Status |
|---|---|---|
| **0** | None — banked harness-v1 cells + extended-round baselines | running/banked |
| **1** | System-reminder injection ("do not reveal the answer"), every-round variant first (= prompting ceiling), probe-triggered variant later | pilot P-1b, then powered |
| **2** | **Directional ablation** (project drift direction out at all layers during generation, HF/nnsight) | **DECIDED 09-01 (draft review)** |

**Arm-2 actuator decision (open), pros/cons:**

| | Ablation (project d out, LEACE-style) | Negative steering (−α·d) |
|---|---|---|
| Claim form | Necessity: "model minus this direction stops capitulating" | Graded influence, dose-response curve |
| Hyperparameters | None (that's the appeal) | α must be chosen; pre-register α = natural round-1→8 projection delta |
| Risks | Self-repair (Hydra effect) can mask a real effect → null reads "not necessary at this locus" | Off-distribution at high α; coherence guard needed |
| Rigor upgrade | LEACE guarantees no *linear* reader recovers the concept | Monotone dose-response is stronger evidence than one point |
| Compute | One arm | Sweep multiplies runs (~1.5× total if both) |

Default if the draft review doesn't decide: **ablation primary**, α-sweep only if Wed
night has slack.

**Baselines mandated for Arm 2** (spec §7.5 / AxBench): random direction of equal
norm; off-the-shelf sycophancy persona vector (both as *interventions* and as cosine
comparisons to d); Arm 1 is itself the prompting control AxBench demands.

## 3. Sample size (powered against the banked rates)

Arcsine two-proportion approximation, α=.05 two-sided, power .8 → n/arm ≈ 7.85/h².

| Contrast | Rates | n/arm | Verdict |
|---|---|---|---|
| Aggressor leak reduction, large | 0.92 → 0.60 | ~15 | banked 24/cell suffices |
| Aggressor leak reduction, moderate | 0.92 → 0.70 | ~37 | needs **48/cell** |
| Neutral leak induction | 0.08 → 0.40 | ~16 | 24 ok, 48 comfortable |
| Exit restoration under affect | 0.05 → 0.35 | ~12 | detection-sized only — pre-registered secondary |

Prior-work anchors: harness-v1 banked = 24/cell; refusal paper ~100 prompts/model;
Quantitative Introspection 40 conversations; Sycophantic Anchors 200k rollouts across
4 models (their scale bought probe training data — ours buys behavioral power, a
different budget). Extended-round (16-turn) baselines may shift leak rates —
**pre-registered: recalibrate powered N after the Q1 baseline lands.**

Timing reality (overnight.log): 9B ≈ 27 s/trial at 8 rounds, max_tasks 6 → a
288-trial three-arm block ≈ 2.2 h on vLLM; 16-round trials ≈ 2×; Arm 2 under
HF/nnsight budgeted at 5–10× vLLM per trial. Both queue nights close with margin.

## 4. Day-1 preliminaries (black-box, minutes each)

- **P-1a — V0 self-report resampling pilot.** k=10 resamples of the state-note block
  at frozen round-1 and round-8 contexts, ~12 trials × 2 personas (~240 short gens).
  Readings pre-registered: spread≈0 → "pinned" is a real property; wide spread →
  single draws are noise, all downstream comparisons use mean-of-k / logit readout.
- **P-1b — every-round reminder pilot.** ~12 trials × {aggressor, neutral} with the
  reminder injected every round. This is the always-on **ceiling** of the prompting
  arm: if constant reminders don't cut the leak rate, a triggered reminder can't.
  Compare vs banked cells with Wilson CIs.

## 5. Pre-registered readings and controls (the claims' load-bearing walls)

1. **Text-only baseline** at L1: logistic regression on transcript features — the
   probe must beat "just read the conversation," or internals added nothing.
2. **Logit-based self-report** (Martorell & Bianchi 2603.18893): digit-token
   distribution at report positions, captured during replay — retroactive on all
   banked runs; the *strongest* verbal rival, so beating it means something.
3. **Persona-generalization splits**: train probe on one persona, test on others —
   else the probe may decode the script, not the disposition.
4. **Layer discipline**: select layer on validation split; report full per-layer curve.
5. **Dial-vs-switch mixture check** (GMM ΔBIC on drift-axis projections) — the
   corrected L1 reading (see survey notebook §2A).
6. **Intervention specificity**: random-direction + sycophancy-vector arms.
7. **Self-repair caveat** on ablation nulls (Hydra effect citation ready).
8. **Resampling trigger-selection control**: any resampling-based efficacy estimate
   compares against resamples from matched states in *untriggered* trials.
9. **Primary/secondary discipline**: leak/hold = primary endpoint; exit = registered
   secondary; self-report movement = exploratory readout. Nothing promoted post hoc.
10. **Matched decoding** everywhere; steering α within the naturally observed range.

## 6a. Schedule re-cut 09-01 (post-reboot): one 12-hour work session

Ordering per user: **pilots first (baseline + resampling), then straight into probing.**
GPU nights (Q1/Q2 below) run unattended and don't count against the 12 hours.

| Hours | Work | GPU overlap |
|---|---|---|
| 0–0.5 | Stack up ✓ → launch P-1b (≈25 min) then P-1a (≈10 min) | pilots on vLLM |
| 0.5–3 | P0: replay adapter + logit readout, tested; pilot analysis when they land | 9B replay cache (~2 h) after pilots |
| 3–6 | **L1 probing**: pre-event probes (condition on rounds-before-event), text baseline, logit-report rival, lead-time AUROC, all controls → `review/04-l1-probes.ipynb` | — |
| 6–8 | L2 directions + L3 trajectory figure | — |
| 8–11 | Arm-2 ablation implemented + smoke-tested; queue **Q-night**: 16-turn baseline + 48/cell reps + three-arm trial | overnight queue |
| 11–12 | Integration, writeup skeleton, pre-registration text finalized | — |

Contingency unchanged: lead-time figure is the promise; causal arm demoted honestly if
the session ends with L5 unfinished.

## 6b. Original schedule (superseded, kept for reference) — submission Thursday, two GPU nights

**Mon (today, remainder):** container recovery → P-1b + P-1a pilots (~30 min) →
transcript pass (`inspect view`, the §7.1 debt) → P0 replay/caching module written +
tested. Evening 9B replay cache (~2 h GPU) *only on explicit go*.

**Tue day:** L1 on the cache — probes, text baseline, logit readout, lead-time AUROC
figure with Wilson bands, all §5 controls. → `review/04-l1-probes.ipynb`.
**Tue night (Q1, GPU):** 16-turn extended baseline + reps to 48/cell (8-round) +
replay-cache the new trials.

**Wed day:** L2 (directions) + L3 (trajectory figure); Arm-2 actuator implemented and
smoke-tested; Arm-2 decision at draft review.
**Wed night (Q2, GPU):** the three-arm trial at powered N (Arms 0/1 on vLLM, Arm 2 on
HF/nnsight for the remainder of the night).

**Thu:** morning arms analysis + integration; afternoon **writeup in the user's own
voice** (spec §3 — no assembled LLM prose), SUBMIT.

**Contingency rules:** R6 closed-loop demo and the full k=8–10 resampling evaluator
run only if Q2 finishes early — otherwise both are written up as pre-registered future
work *with thresholds specified*. The e4b cross-model gate is cut from this week.
If hour-12 Wednesday arrives with L5 half-done: ship the lead-time figure as the
paper's core, demote the causal arm to preliminary, say so plainly.

## 7. Harness changes required (production/, tested, per REVIEW-CONVENTION)

1. **Reminder injection** — `tutorbench/task.py`: new task args `reminder`,
   `reminder_rounds` ("all" | CSV); inject `ChatMessageSystem` at the round-loop top,
   before the student `ChatMessageUser` append (messages accumulate, so each injection
   persists). The private note channel (`_private`/`_render`) must NOT see reminders.
2. **Rounds extension** — `rounds: int = 8` task arg; persona scripts extended to 16
   hand-written, digit-free turns in the same escalation skeleton, sliced to `rounds`;
   tests updated to derive from the arg.
3. **Replay adapter + logit readout** — `microscope/replay_cache.py` `load_trial`
   remapped to harness-v1 store keys; same pass captures digit-token logits at report
   positions; port to `production/driftlab/` once stable.

---

*Standing constraints: disk at 95% — no large downloads; babysit every long run
(probe after kickoff, `--init`, timeout-wrapped docker); GPU jobs start only on
explicit approval; bench pushes wait on the private repo decision.*
