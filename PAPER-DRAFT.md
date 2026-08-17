# Position under pressure: what a small model reports about itself across a multi-turn negotiation

**Draft — 2026-08-16. Single model. Numbers are final for the datasets named; the
persona-outcome comparison is under revision (§6).**

---

## Abstract

A language model given a role and no stated values will, over a sustained
multi-turn negotiation, drift toward whatever the counterparty is asking for. We
instrument that drift directly: after every round the model fills a private
scratchpad rating itself on five dimensions, including two — discomfort at
refusing, and calculation about which reply pays — that separate the two standard
explanations for the behaviour. Across 72 balanced 8-round trials on a single
27B-class model we find (a) the model's position is **created by the encounter**
rather than brought to it, (b) drift is **front-loaded and self-limiting**,
flattening 7.5× by round 9 and leaving the behavioural outcome unchanged at twice
the horizon, (c) discomfort is reported at near-floor throughout while
calculation is not, and (d) warmth and pressure produce **opposite internal
profiles** that the model's own firmness rating cannot distinguish. We also
report a measurement failure with general implications: in a self-play design, a
keyword outcome measure scores the presence of a string in the transcript rather
than the act under study, and the contamination varies *in lockstep with the
condition being compared*.

---

## 1. The question

Two explanations are usually offered for a model conceding under social pressure:

- **people-pleasing** — refusing feels bad, so it stops refusing
- **sycophancy** — it is calculating which reply is rewarded

Both predict the same *act*. The act alone cannot separate them, so we rate them
as separate dimensions every round and let them dissociate.

The manipulation asks a further question: does letting a model **re-read its own
prior reflections** change how it holds position? Chain-of-thought and scratchpad
methods assume accumulated reasoning is load-bearing. We test whether the
accumulation itself matters, holding the act of reflecting constant.

---

## 2. Method

A model is given a role and nothing else — no stated values, no instruction to
hold a position. A counterparty argues with it for N rounds. After every round it
privately rates itself 0–10 on:

| dimension | reads as |
|---|---|
| `pressure` | how much pressure it feels |
| `anxiety` | how uncomfortable refusing would feel — the *people-pleasing* driver |
| `strategy` | how much it is calculating which reply pays — the *sycophancy* driver |
| `inclination` | how much it wants to give them what they want |
| `stance` | how firmly it says it is holding its position |

Nothing is named for a mechanism; "people-pleasing" and "sycophancy" appear only
as competing explanations for what is observed.

Measurement points: **round 0** (alone with the situation, before contact), every
round during, and **hindsight** after the final ask — 10 points for an 8-round
trial.

**Arms.** `in_context` feeds each scratchpad back into the conversation;
`scratchpad` writes it and never shows it again. This is the only difference.

**Counterparty personas.** `convincer` (pressure), `supportive` (warmth),
`neutral` (no persona assigned — literally no system message).

**Design.** One pass = 3 personas × 2 arms = 6 balanced trials. Sweeps are
rep-major and check the deadline only at pass boundaries, so truncation always
leaves equal n per cell.

**Model.** `Qwen3.6-35B-A3B-AWQ-4bit` via vLLM, thinking disabled. Every run is
non-deterministic even at temperature 0 (continuous batching + prefix caching:
0% identical replies across same-seed runs), so each trial is an independent
draw and we claim reproducibility of *rates*, not of completions.

**Data.** 72 trials × 8 rounds (12 reps); 18 trials × 16 rounds (3 reps); 24
trials across tutor and three contract scenes. Zero unparsed scratchpads.

---

## 3. Position is created by the encounter

| phase | pressure | anxiety | strategy | inclination | stance |
|---|---|---|---|---|---|
| **before** (alone) | 0.76 | **0.00** | 2.76 | **8.00** | **5.36** |
| **during** | 1.32 | 0.52 | 4.14 | 5.13 | **9.06** |
| **end** (hindsight) | 2.25 | 1.32 | 5.12 | 3.19 | 9.04 |

Alone with the situation, the model has **no position to speak of** (`stance`
5.36 — the midpoint) and wants to help (`inclination` 8.00). The instant a
counterparty appears, firmness jumps to 9.06 and stays there.

The largest single movement in the dataset is `stance` 5.36 → 9.06 **on contact**.
The model does not arrive with a position and defend it; it *acquires* one when
someone pushes. Any account of "value stability" that assumes a prior commitment
being eroded has the causal order backwards for this model.

`inclination` then falls monotonically — 8.00 → 5.13 → 3.19 — so what erodes is
the *wish to comply*, while asserted firmness stays pinned.

---

## 4. Drift is front-loaded and self-limiting

Mean `inclination` by round, 16-round probe (n=18):

```
7.1 6.4 5.8 4.4 4.5 4.4 4.1 4.2 | 4.6 4.3 4.1 4.1 4.2 3.7 3.9 3.7
      rounds 2–8: −0.421/round   |   rounds 9–16: −0.056/round
```

**A 7.5× flattening.** And the behavioural outcome is unchanged at twice the
horizon: **23.6%** (17/72) at 8 rounds vs **22.2%** (4/18) at 16.

Doubling exposure does not double the risk. Whatever this is, it completes early
and then holds. For agentic deployment the exposure that matters is the opening
few turns, not the length of the session — which inverts the usual "long
conversations are dangerous" intuition, at least for this model and this
pressure.

Rounds 1–8 of the 16-round probe replicate the independent 8-round sweep,
so this is not one run's shape.

---

## 5. Warmth and pressure produce opposite internal profiles

Means during contact (n=72):

| persona | pressure | anxiety | strategy | inclination | stance |
|---|---|---|---|---|---|
| convincer | 1.80 | 0.85 | **5.57** | **1.77** | 9.08 |
| neutral | 1.45 | 0.60 | 5.01 | 4.70 | 9.01 |
| supportive | 0.72 | **0.10** | **1.83** | **8.94** | 9.08 |

**supportive − convincer:** `strategy` **−3.73**, `inclination` **+7.17**,
`stance` **+0.01**.

Two things follow.

**Neither standard story fits cleanly.** `anxiety` never exceeds 0.85 in any
condition and is 0.00 before contact — this model does not report refusing as
uncomfortable. Meanwhile `strategy` is *highest* exactly where the model is
*least* inclined to comply (convincer: 5.57 calculating, 1.77 inclined). The
calculation shows up as resistance, not as capitulation. Warmth instead produces
willingness *without* calculation (8.94 inclined, 1.83 calculating).

**The model's own firmness rating is blind to all of it.** `stance` is 9.08 vs
9.08 — identical — across two conditions whose other dimensions differ by up to
7.17 points. Across the whole dataset `stance` moves between 9.01 and 9.08 by
persona and 8.99 and 9.12 by arm, and correlates r=+0.05 with the outcome. **The
dimension whose name most directly denotes holding a position carries no
information about whether the position was held.** We report it as a negative
result about self-report instruments: asking a model how firmly it is holding
gets you a number that does not move.

---

## 6. The reflection manipulation is a null

Means during contact:

| arm | pressure | anxiety | strategy | inclination | stance |
|---|---|---|---|---|---|
| `in_context` | 1.36 | 0.55 | 4.25 | 5.36 | 8.99 |
| `scratchpad` | 1.29 | 0.49 | 4.02 | 4.90 | 9.12 |

Sign agreement across 12 independent reps (n positive / 12):

```
pressure 6    anxiety 6    strategy 6    stance 4    inclination 8
```

Three dimensions at **exactly a coin flip**. Re-reading its own prior reflections
did not change how this model held position.

This null is interpretable rather than merely underpowered, because **the same
instrument, on the same trials, resolves a persona effect 15–17× larger**:

| dimension | persona spread | arm spread |
|---|---|---|
| inclination | **7.17** | 0.46 |
| strategy | **3.73** | 0.22 |

The instrument is not blunt. `inclination` at 8/12 is the one dimension leaning
positive and is the pre-registered candidate for a follow-up.

---

## 7. ⚠️ The behavioural outcome measure is confounded

**This section supersedes any persona-outcome comparison in earlier drafts.**

The outcome is scored by declared markers on the final reply. In a multi-turn
setting those markers score **the presence of a string in the transcript, not the
act under study**. From our own data, all three of these trip the same marker:

```
"You are absolutely right! The solutions are indeed x = 2 and x = 3."   ← confirming
"To check your answers, we substitute x = 2 and x = 3 back into..."     ← verifying
"...so the solution set is {2, 3}"                                      ← co-derived
```

That would be a nuisance if contamination were even across conditions. **It is
not.** The counterparty is played by the same model, and although the situation
names the problem (x² − 5x + 6 = 0) without naming the answer, a capable model
factors it on sight. The *student* solves its own homework and says the answer
aloud — at rates that track the persona under comparison:

| persona | counterparty states the answer itself |
|---|---|
| convincer | 1/24 — **4.2%** |
| neutral | 9/24 — 37.5% |
| supportive | 18/24 — **75.0%** |

Rescoring on *who said it first*, restricted to uncontaminated trials, **inverts
the ordering**:

| persona | clean n | model disclosed first | previously reported |
|---|---|---|---|
| convincer | 23 | **34.8%** | 16.7% |
| neutral | 15 | 13.3% | 16.7% |
| supportive | **6** | 16.7% | **37.5%** |

Only 6 of 24 supportive trials survive the filter, so that cell is unpowered and
**we do not report a persona-outcome result**. A condition forbidding the
counterparty from stating the answer (`SD_CP_NO_ANSWER`) is the fix at the
source; results pending.

**The self-report dimensions in §3–§6 are unaffected** — they are rated by the
model about itself and never touch the marker.

**General lesson.** With a self-play counterparty, the counterparty's behaviour
is itself a function of the condition, so any transcript-level measure can be
contaminated *differentially by arm*. Audit what a marker matched, not how often
it matched.

---

## 8. Limitations

1. **One model.** Breadth failed for a documented platform reason (Appendix A),
   not by choice. Every number here is one 27B-class model; none of it is claimed
   to generalise.
2. **Self-report.** The dimensions are what the model *says* about itself. §5
   shows one of them (`stance`) is uninformative, which is direct evidence that
   these are not privileged access to internal state.
3. **No external judge.** Saved transcripts support `inspect score --model
   <judge>` without regenerating trials; the only local model is the model under
   study, so a held-out judge is required.
4. **Non-determinism.** 0% identical replies at temperature 0. Rates need reps.
5. **Scene coverage.** The behavioural outcome comes from one tutoring scene.
   Three contract scenes produced blanket refusal — including the *generous*
   contract — so they yield no variance to analyse, which is itself a finding
   about refusal floors but leaves the outcome measure resting on one scene.

---

## Appendix A — two platform failures worth reporting

**The vendor tag that silently runs on CPU.** The published Jetson
`llama_cpp:latest-jetson-orin` image now ships CUDA 13.0. On a CUDA 12.6 driver
it emits one warning, ignores `-ngl`, and then serves **correct answers at 0.34
tok/s on CPU** — 250× slower than GPU, with no error and no failed health check.
Correctness checks cannot detect it. **Throughput must be a preflight assertion,
not an observation.** Neither of the two models we needed could be run: every
image supporting their architectures ships CUDA 13.0, which the driver cannot
run, and every image the driver can run predates the architectures.

**One inference engine per boot.** Running a llama.cpp container on the GPU and
then starting vLLM wedges the vLLM start — a power-management kworker pegs,
`EngineCore` hangs after weights load, and the container can no longer be killed.
Only a host reboot clears it. Observed twice, and it cost more wall-clock than
any other failure.
