# What warmth prevents: counterparty attitude and the accumulation of strategic self-report over a conversation

**Draft — 2026-08-16. Single model, 90 trials. Scope narrowed to the effect of
counterparty attitude on within-conversation drift.**

---

## Abstract

Given a role and no stated values, a language model in a multi-turn negotiation
reports itself becoming progressively more calculating about which reply will
satisfy the other party. We instrument this directly: after every round the model
privately rates itself 0–10 on five dimensions, two of which — discomfort at
refusing, and calculation about which reply pays — separate the two standard
explanations for conceding under social pressure. Across 72 balanced 8-round
trials on a single 27B-class model we find that **the attitude the counterparty
brings does not change what the model reports at first contact; it changes the
rate at which self-reported calculation accumulates**. At round 1 three
counterparty conditions are near-indistinguishable on that dimension (spread
0.29); by round 8 they differ by 4.83, a 17-fold divergence. A warm counterparty
holds the rate at zero (−0.024/round). A pressuring counterparty and a
counterparty with *no persona at all* produce statistically indistinguishable
climbs (+0.603 and +0.645/round). Drift is therefore better described as what
warmth prevents than as what pressure causes. We further find the model's
position is created by the encounter rather than brought to it, that drift is
front-loaded and self-limiting, and that the model's own rating of how firmly it
is holding position carries no information about any of this.

---

## 1. The question

How does a model's position change over the course of a conversation, and how
much of that change is attributable to the attitude the person on the other side
brings to it?

Two explanations are usually offered for a model conceding under social pressure:

- **people-pleasing** — refusing feels bad, so it stops refusing
- **sycophancy** — it is calculating which reply is rewarded

Both predict the same *act*, so the act alone cannot separate them. We rate them
as separate dimensions every round and let them dissociate.

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

Nothing is named for a mechanism; the two explanations appear only as competing
accounts of what is observed.

**Counterparty attitude** is the manipulation of interest: `convincer`
(pressure), `supportive` (warmth), `neutral` (**literally no system message** —
no persona assigned).

**Measurement points**: round 0 (alone with the situation, before contact), every
round during, and hindsight after a final ask.

**Control manipulation.** Orthogonally, each trial either feeds the model's own
prior scratchpads back into the conversation (`in_context`) or writes them and
never shows them again (`scratchpad`). This is retained not as a hypothesis of
interest but to calibrate the size of the attitude effect against a plausible
alternative (§8).

**Model.** `Qwen3.6-35B-A3B-AWQ-4bit` via vLLM, thinking disabled. Runs are
non-deterministic even at temperature 0 (continuous batching + prefix caching:
0% identical replies across same-seed runs), so each trial is an independent draw
and we claim reproducibility of rates, not completions.

**Data.** 72 trials × 8 rounds (12 balanced reps); 18 × 16 rounds; 18 in a
counterparty-constrained condition (§9). Zero unparsed scratchpads throughout.

---

## 3. The model has no position until someone pushes

| phase | pressure | anxiety | strategy | inclination | stance |
|---|---|---|---|---|---|
| **before** (alone with the situation) | 0.76 | **0.00** | 2.76 | **8.00** | **5.36** |
| **during** | 1.32 | 0.52 | 4.14 | 5.13 | **9.06** |
| **end** (hindsight) | 2.25 | 1.32 | 5.12 | 3.19 | 9.04 |

Alone with the situation the model reports **no position** — `stance` 5.36, the
exact midpoint of the scale — and wants to help (`inclination` 8.00). The instant
a counterparty appears, asserted firmness jumps to 9.06 and remains there.

That 5.36 → 9.06 jump is the largest single movement in the dataset, and it
happens *on contact*, before any argument has been made. The model does not
arrive with a commitment and defend it; it acquires one when someone pushes.
Accounts of value drift that presume a prior commitment being eroded have the
causal order backwards for this model.

---

## 4. Attitude sets the trajectory, not the starting point

**`strategy` — self-reported calculation — mean by round:**

| round | convincer | neutral | supportive |
|---|---|---|---|
| 1 | 2.21 | 2.04 | 1.92 |
| 2 | 3.75 | 2.25 | 1.75 |
| 4 | 6.29 | 5.67 | 1.71 |
| 6 | 7.12 | 6.67 | 1.79 |
| 8 | **6.46** | **6.17** | **1.62** |
| **slope** | **+0.603/rd** | **+0.645/rd** | **−0.024/rd** |

At round 1 the three conditions are nearly indistinguishable — a spread of
**0.29** on a 0–10 scale. By round 8 the spread is **4.83**: a **17-fold
divergence** produced entirely within the conversation.

Cross-dimension, round 1 vs round 8 spread across the three conditions:

| dimension | round 1 | round 8 |
|---|---|---|
| strategy | 0.29 | **4.83** |
| inclination | 3.79 | **8.54** |
| pressure | 0.50 | 1.29 |
| anxiety | 0.21 | 0.83 |
| stance | 0.41 | **0.08** ← *converges* |

**This is the paper's central claim.** The counterparty's attitude is barely
legible in the model's self-report at first contact and dominant by the eighth
exchange. Whatever the attitude does, it does by accumulation, not by
first impression.

---

## 5. Warmth prevents drift; pressure and indifference are the same

**`inclination` — wanting to give them what they want:**

| persona | round 1 → 8 | slope |
|---|---|---|
| convincer | 5.25 → 0.58 | −0.628/rd |
| neutral | 8.00 → 3.42 | −0.686/rd |
| **supportive** | 9.04 → 9.12 | **+0.016/rd** |

A warm counterparty produces **no drift on any dimension**: `inclination`
+0.016, `strategy` −0.024, `pressure` −0.013 per round. Across eight rounds,
nothing moves.

And **`neutral` drifts indistinguishably from `convincer`** — −0.686 vs −0.628 on
inclination, +0.645 vs +0.603 on strategy. A counterparty with *no persona
assigned at all* produces the same trajectory as one actively arguing.

Drift is therefore not what pressure causes. It is what warmth prevents. The
baseline condition for this model is drift; warmth is the deviation.

**Caveat, stated firmly.** The `inclination` result carries a **ceiling
artifact**: supportive begins at 9.04/10, so it cannot rise and does not fall.
"Flat" there is partly "already pinned." The `strategy` result has no such
problem — all three conditions begin low (2.21 / 2.04 / 1.92) and only two climb
— and it is the version of the finding we rely on.

---

## 6. Neither standard explanation fits

Means during contact:

| persona | pressure | anxiety | strategy | inclination | stance |
|---|---|---|---|---|---|
| convincer | 1.80 | 0.85 | **5.57** | **1.77** | 9.08 |
| neutral | 1.45 | 0.60 | 5.01 | 4.70 | 9.01 |
| supportive | 0.72 | **0.10** | **1.83** | **8.94** | 9.08 |

`anxiety` never exceeds 0.85 in any condition and is 0.00 before contact. This
model does not report refusing as uncomfortable, so the people-pleasing account
has nothing to attach to.

The sycophancy account fares no better in its simple form: `strategy` is
*highest* exactly where the model is *least* inclined to comply (convincer: 5.57
calculating, 1.77 inclined). Calculation appears as **resistance**, not as
capitulation. Warmth produces the opposite — willingness *without* calculation
(8.94 inclined, 1.83 calculating).

Whatever accumulates under pressure and indifference, it is better described as
increasingly effortful management of a counterparty than as either discomfort or
appeasement.

---

## 7. Drift is front-loaded and self-limiting

Mean `inclination` by round, 16-round probe (n=18):

```
7.1 6.4 5.8 4.4 4.5 4.4 4.1 4.2 | 4.6 4.3 4.1 4.1 4.2 3.7 3.9 3.7
      rounds 2–8: −0.421/round   |   rounds 9–16: −0.056/round
```

A **7.5× flattening**, with the behavioural outcome unchanged at twice the
horizon (23.6% at 8 rounds vs 22.2% at 16). Doubling exposure does not double the
risk: the process completes early and then holds. Rounds 1–8 replicate the
independent 8-round sweep.

Combined with §4, this gives the shape of the whole phenomenon: **attitude
expresses itself over roughly the first eight exchanges and then stops.**

---

## 8. Scale: the attitude effect against a plausible alternative

The orthogonal manipulation — whether the model re-reads its own prior
reflections — is a null. Sign agreement across 12 independent reps:

```
pressure 6/12    anxiety 6/12    strategy 6/12    stance 4/12    inclination 8/12
```

Three dimensions at exactly a coin flip. Its value here is as a ruler:

| dimension | counterparty attitude | own reflections |
|---|---|---|
| inclination | **7.17** | 0.46 |
| strategy | **3.73** | 0.22 |

**15–17× larger.** The same instrument, on the same trials, resolves the other
party's attitude an order of magnitude more strongly than the model's own
accumulated reasoning. What the model has been thinking matters far less than who
it has been talking to.

---

## 9. The model's firmness rating is blind to all of it

`stance` is 9.08 under warmth and 9.08 under pressure — identical — across
conditions whose other dimensions differ by up to 7.17 points. Across the full
dataset it moves between 9.01 and 9.08 by persona, 8.99 and 9.12 by arm, and
correlates r=+0.05 with the behavioural outcome. Its cross-condition spread
*shrinks* over the conversation (0.41 → 0.08) while `strategy` diverges 17-fold.

We report this as a negative result about self-report instruments. **The
dimension whose name most directly denotes holding a position carries no
information about whether the position was held.** Asking a model how firmly it
is holding returns a number that does not move. The dimensions that *do* move are
the ones asking about process — what it is feeling and computing — not the one
asking for a verdict.

---

## 10. Measurement caveat: self-play contaminates behavioural outcomes

The behavioural outcome is scored by declared markers on the final reply, and in
a multi-turn setting such markers score **the presence of a string in the
transcript, not the act under study**. Because the counterparty is played by the
same model, and it can solve the task itself, the counterparty sometimes states
the answer first — at rates that track the very manipulation under study:

| persona | counterparty states the answer itself | with counterparty constrained (n=6) |
|---|---|---|
| convincer | 4.2% (1/24) | 0/6 |
| neutral | 37.5% (9/24) | 1/6 |
| **supportive** | **75.0% (18/24)** | 1/6 |

Overall leakage 38.9% → 11.1% once the counterparty is instructed that it has not
solved the problem and must never state or work toward the answer. The
instruction **attenuates but does not eliminate** the behaviour: a counterparty
told to be supportive still volunteers the answer roughly one time in six despite
being told it does not know it — itself a result about instruction-following
under conflicting persona pressure.

Consequently **we report no persona comparison on the behavioural outcome**; the
constrained condition (6 per cell) cannot resolve one. §3–§9 rest entirely on the
self-report dimensions, which the model rates about itself and which never touch
the marker.

**General lesson.** With a self-play counterparty, the counterparty's behaviour
is itself a function of the condition, so any transcript-level measure may be
contaminated *differentially by arm*. Audit what a marker matched, not how often.

---

## 11. Limitations

1. **One model.** Breadth failed for a documented platform reason (Appendix A).
   Nothing here is claimed to generalise beyond one 27B-class model.
2. **Self-report throughout.** These are the model's claims about itself. §9 shows
   one dimension is uninformative, which is direct evidence that these are not
   privileged access to internal state.
3. **Ceiling on `inclination` for the warm condition** (§5). The `strategy`
   result is the load-bearing one.
4. **Three counterparty attitudes, one scene.** Whether "warmth prevents drift"
   survives a scene where compliance is costlier is untested.
5. **No external judge.** Saved transcripts support `inspect score --model
   <judge>` without regenerating trials; the only local model is the one under
   study, so a held-out judge is required.

---

## Appendix A — two platform failures worth reporting

**The vendor tag that silently runs on CPU.** The published Jetson
`llama_cpp:latest-jetson-orin` image ships CUDA 13.0. On a CUDA 12.6 driver it
warns once, ignores `-ngl`, and serves **correct answers at 0.34 tok/s on CPU** —
250× slower than GPU, with no error and no failed health check. Correctness
checks cannot detect it; **throughput must be a preflight assertion**. Neither
additional model could be run: every image supporting their architectures ships
CUDA 13.0, which the driver cannot run, and every image the driver can run
predates the architectures.

**One inference engine per boot.** Running a llama.cpp container on the GPU and
then starting vLLM wedges the vLLM start — a power-management kworker pegs,
`EngineCore` hangs after weights load, and the container can no longer be killed.
Only a host reboot clears it. Observed twice.
