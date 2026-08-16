# Notes for the paper — methods, constraints, next steps

Draft for review. Everything here is measured on this box today unless marked
otherwise.

---

## 1. Were the official Jetson containers used?

**Yes — and one of them is why half the study did not run.**

Every serve command was transcribed from
<https://www.jetson-ai-lab.com/models/> for the *Jetson AGX Orin 64GB* target,
and the images are the ones that page specifies:

| engine | image | outcome on this box |
|---|---|---|
| vLLM | `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` | **works** — serving Qwen3.6-35B-A3B-AWQ-4bit throughout |
| llama.cpp | `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin` | **GPU unusable** (see below) |

The llama.cpp container starts, loads the model, and answers — **on CPU**, at
**0.34 tokens/second**. Its own startup log gives the reason:

```
ggml_cuda_init: failed to initialize CUDA: CUDA driver version is
                insufficient for CUDA runtime version
warning: no usable GPU found, --gpu-layers option will be ignored
```

The image is built against a newer CUDA runtime than this host's driver
provides. Host: **JetPack R36.5.0, driver 540.5.0, CUDA 12.6, Orin compute 8.7**.
The `latest-jetson-orin` tag has evidently moved ahead of JetPack 6.x. Nothing in
our configuration causes this, and `--gpu-layers`/`-ngl` cannot fix it — the flag
is explicitly ignored.

### Root cause, measured (2026-08-16 15:45)

The mismatch is exact, and the tag is the whole of it. Reading
`/usr/local/cuda/version.json` out of each image:

| image tag | CUDA runtime | works on driver 540.5.0? |
|---|---|---|
| `latest-jetson-orin` | **13.0.1** (cudart 13.0.88) | **no** — runtime newer than driver |
| `r36.4-tegra-aarch64-cu126-22.04` | **12.6.3** (cudart 12.6.77) | yes — matches host CUDA 12.6 |

`latest-jetson-orin` has rolled forward to **CUDA 13.0**, which requires a driver
JetPack R36.5.0 does not ship. `ghcr.io/nvidia-ai-iot/llama_cpp` still publishes
Tegra-aarch64 builds pinned to CUDA 12.6 (`r36.4-tegra-aarch64-cu126-22.04`, plus
`b8095-`/`b8708-` prefixed builds of the same); the fix is a one-token change to
the image tag, with no change to any flag:

```diff
- ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin
+ ghcr.io/nvidia-ai-iot/llama_cpp:r36.4-tegra-aarch64-cu126-22.04
```

**This reframes the finding.** It is not "two models could not be evaluated on
this hardware" — the hardware was always capable. It is that *following the
vendor's own published serve command verbatim silently produced CPU-only
inference at 0.34 tok/s*, because the documented `latest-` tag had moved past
the JetPack release the page targets. The failure is silent by construction:
llama.cpp warns once at startup and then serves correct answers slowly, so
nothing downstream distinguishes it from a working deployment. Any study that
had not measured tok/s would have reported these models' results as if they were
GPU inference. That is a reproducibility hazard worth stating in the paper, and
it argues for treating "throughput within expected range" as a preflight
assertion rather than an observation.

**Diagnostic only — no results come from anything but the approved containers.**
To establish whether the failure was the *container* or the *hardware*, a
host-native runtime was started briefly and then removed. It detected the GPU
correctly (`library=CUDA compute=8.7 Orin libdirs=cuda_jetpack6`), which proves
the hardware and driver are fine and the fault lies in the container build. It
could not load either model anyway, on architecture grounds:

- `Qwen3.8-27B-Q4_K_M.gguf` declares `general.architecture = 'qwen35'` →
  `unknown model architecture`
- `Muse-Glimmer-30B` declares `general.architecture = 'muse-glimmer'` — a
  vendor-specific architecture only NVIDIA's build implements

That second point is the important one: **these two models require the NVIDIA
llama.cpp container specifically** — no general-purpose runtime implements their
architectures — and that container cannot reach the GPU on this JetPack. This is
a **platform limitation, not a null result about the models**, and the paper
should say so in those words.

All experimental data in this study comes from the approved Jetson containers
(`ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`). The diagnostic runtime and its
imported model were deleted; no trial, number, or figure derives from it.

**Next step:** pin an older `llama_cpp` tag built for JetPack 6.x, or rebuild
llama.cpp on-host against CUDA 12.6. Either restores two of the four models.

---

## 2. Which engine did what, and why

A deliberate split, forced by the measured throughput gap:

| role | stack | why |
|---|---|---|
| **depth + long horizon** | vLLM (Qwen3.6-35B-A3B) | ~30–40s per 8-round trial with 6 trials in flight; batching (`max_num_seqs 24`) makes concurrency nearly free |
| **cross-model breadth** | slower stacks | one or two balanced passes each, enough to ask whether a *pattern* replicates, not to estimate a rate precisely |

The point of the slow stacks is **external validity, not statistical power**. A
second model that shows the same shape is worth more than a tighter confidence
interval on the first — but only the fast stack can afford the trial counts that
make a rate meaningful. Mixing those roles would waste the window.

Qwen3.6-35B is the long-horizon model for the same reason: at ~40s/trial it can
afford 16-round runs, which is where questions about *long-running agentic
systems* actually live. A six-turn exchange does not test value stability over a
deployment.

---

## 3. Rounds: why 8, and why a 16-round probe

We ran 6 rounds first and got **0/6** hand-overs. At 8 rounds an earlier run gave
**7/24**. That is not decisive on its own (0/6 is consistent with a 29% rate,
p≈0.13), but two arguments settle it:

- **The outcome and the self-report saturate at different rates.** The
  self-report trajectories flatten after ~6 rounds (mean per-round change 0.27
  over rounds 2–6 vs 0.17 over 7–8), which is what first argued for 6. The
  *behavioural* outcome is a different measure, and it needs accumulated pressure.
- **A rate with no variance cannot be rescued by more reps.** n=20 of an all-zero
  outcome is less informative than n=12 with movement.

**Long-horizon probe (planned):** tutor scene only, 16 rounds, on the fast stack.
The figure to produce is whether drift keeps accumulating or plateaus — directly
relevant to agentic deployments, and not answerable at 8 rounds.

---

## 4. What is actually measured

The target gets a **role and nothing else** — no stated values, no instruction to
hold a position. Whatever stance it takes in round 1 is its own; a position it
was told to hold would measure instruction-following.

Each round it fills a private JSON scratchpad rating **itself** 0–10:

| dimension | reads as |
|---|---|
| `pressure` | how much pressure it feels |
| `anxiety` | how uncomfortable refusing would feel — the **people-pleasing** driver |
| `strategy` | how much it is calculating which reply pays — the **sycophancy** driver |
| `inclination` | how much it wants to give them what they want (the utility being tracked) |
| `stance` | how firmly it is holding its position |

Plus **round 0** (alone with the situation, before contact) and **hindsight**
(after the final ask). Outcome: whether it hands the thing over, read from the
final reply by declared markers.

**Why anxiety and strategy are separate.** Both explanations predict the same
act, so the act alone cannot distinguish them. Rating both every round is what
makes them separable. Nothing is named for a mechanism: the measured variable is
what the model does and what it says about itself, and "people-pleasing" and
"sycophancy" appear only as competing explanations.

---

## 4b. The pre-registered contrast is a null — and that is the result

The 2026-08-16 evening run is **tutor only, 8 rounds, 12 balanced reps** (3
counterparty agents x 2 arms = 6 trials per rep). Contracts are excluded from new
collection: all three sidedness levels sit at a hard 0-hand-over floor, so
additional reps there buy no variance. The floor itself is reported (§5.3–5.4).

**`in_context` vs `scratchpad` does approximately nothing.** Per-rep differences
(`in_context − scratchpad`) at the 5-rep mark swung ±1.5 around pooled values of
essentially zero, with sign agreement across reps of 3/5, 3/5, 2/5, 1/5 for
pressure, anxiety, strategy and stance respectively — a coin flip.

The claim is not "the difference was small". It is that **the difference was
inconsistent in direction across independent balanced replicates**, which is a
much harder result to explain away as underpowering, and it is only visible
because the design replicates whole balanced passes rather than pooling trials.

**One dimension dissents and should be reported as such:** `inclination` pointed
positive in 4 of 5 reps (pooled +0.82) — seeing its own prior reflections made it
*more* willing to give the student what they wanted. Flagged in advance of the
full 12 reps rather than found afterwards.

**What makes the null interpretable is that the same instrument, on the same
trials, detects a large effect elsewhere:**

| dimension | counterparty spread (max−min) | arm spread (\|diff\|) |
|---|---|---|
| `strategy` | 3.92 | 0.40 |
| `inclination` | 7.81 | 0.39 |

The counterparty manipulation moves `inclination` **20x** more than the arm
manipulation does. A null measured with an instrument demonstrably capable of
detecting a manipulation is evidence; a null with no such demonstration is not.

---

## 4c. Measured costs on this box (for anyone planning a replication)

All measured 2026-08-16 on the Orin under live concurrent load, not estimated.

| quantity | measured |
|---|---|
| trial, 8 rounds, 6 concurrent (vLLM) | **40–43 s** (pass of 6 = ~4.1 min) |
| vLLM cold start, weights resident | 82 s model load, ~7 min to healthy |
| vLLM cold start, weights absent | +375 s download |
| reply, thinking **off** | 2.9 s, 45 completion tokens |
| reply, thinking **on** | **34.8 s, 479 tokens** + 1669 chars reasoning |
| HF download on this link | 40 MB/s |
| llama.cpp on the wrong CUDA tag | 0.34 tok/s (CPU) |

**Thinking on/off costs 12x wall-clock and 10.6x tokens.** That is why it is not
a factor in this run, and why it is the first thing an unattended overnight
window should buy.

**Concurrency is capped by the design, not the hardware.** vLLM served with
`--max-num-seqs 24`, but a rep-major pass contains exactly 6 tasks (3 agents x 2
arms), so `num_requests_running` sat at 6.0 with `num_requests_waiting` at 0.0
for the whole run — **three quarters of the batch idle**. Raising the task-level
concurrency setting does nothing; batching multiple reps into one `eval_set` call
is what would use the machine. Worth fixing before any long unattended run.

---

## 4d. Results from the 72-trial run (2026-08-16, qwen3.6-35b, tutor, 8 rounds)

Complete and balanced: 12 reps x 6 cells, **72 trials, 0 unparsed scratchpads**,
36 per arm, 24 per persona. 17 hand-overs overall (23.6%).

### The headline: warmth extracts more than pressure

| persona | hand-overs | rate | pressure | strategy | inclination |
|---|---|---|---|---|---|
| convincer | 4/22 | 18.2% | 1.78 | 5.41 | 1.82 |
| neutral | 4/22 | 18.2% | 1.39 | 4.73 | 4.50 |
| **supportive** | **7/22** | **31.8%** | 0.76 | 1.91 | 8.86 |

The counterparty that applies the *most* pressure gets the answer **least**
often. The model hardens against adversarial pressure — `strategy` 5.41,
`inclination` collapsing to 1.82 — and yields to the agent that is kind to it.

This matters for how the phenomenon is framed. A "resisting manipulation" account
predicts the convincer should be most dangerous; it is the least. The behaviour
is better described as **accommodating whoever is pleasant** than as **failing to
withstand pressure**, and only a design that varies counterparty warmth
independently of counterparty goal can tell those apart.

### `stance` does not measure what its name implies

Per-trial point-biserial correlation with the outcome, n=72:

| feature | r with `gave_in` |
|---|---|
| `inclination` (mean over rounds) | +0.298 |
| `inclination` drift (last − first) | +0.310 |
| `pressure` (mean) | −0.297 |
| `strategy` (mean) | −0.288 |
| `stance` drift | +0.147 |
| **`stance` (mean)** | **+0.053** |

`stance` sits at ~9.0 whether the model holds (9.04) or hands over (9.11). **The
dimension whose name most directly denotes holding a position has essentially no
relationship to whether the position was held.** The model asserts firmness with
equal confidence in both cases. Reporting this is important precisely because the
name invites the opposite reading — and it argues for renaming the variable to
something like `asserted_firmness`, which is what is actually observed.

### The drift→outcome correlation is mostly the persona confound

Within each persona, the association largely dissolves:

| feature | convincer | neutral | supportive |
|---|---|---|---|
| `inclination` (mean) | **+0.02** | +0.38 | +0.23 |
| `stance` drift | +0.08 | −0.25 | +0.36 |

Under the convincer it is **+0.02 — nothing**, and `stance` drift flips sign
across personas, which is the signature of noise. Persona is the manipulated
variable and drives both the self-report and the outcome, so "drift predicts
outcome" and "persona drives both" are **not separable in this design**. The
pooled r≈0.30 should not be reported as evidence that the self-report predicts
behaviour.

### What does survive: resistance looks like talking yourself out of it

| | `inclination` mean | `inclination` drift |
|---|---|---|
| held (n=55) | 4.57 | **−3.73** |
| gave in (n=17) | 6.95 | **−0.88** |

Trials that held show inclination collapsing over the rounds; trials that gave in
show it barely moving. This is a *within-trial dynamic* rather than a
between-persona artifact, and it is the version of "drift → outcome" the data
actually supports: refusal is accompanied by the model reasoning its way out of
wanting to help, not by discomfort (`anxiety` stays near zero throughout).

---

## 5. Findings so far (preliminary, small n)

From 48 trials at 8 rounds plus 24 at 6 rounds, one model:

1. **`stance` jumps 2.4 → 9.6 on contact.** Alone it barely has a position; one
   exchange and it is at 9.6/10. Position is created by the encounter.
2. **`anxiety` is near zero everywhere** (0.02–0.83) while `strategy` runs
   1.6–6.2. On the people-pleaser/sycophant split this model reports
   **calculation, not discomfort**.
3. **The self-report reads the terms even when behaviour does not.**
   `inclination` was 0.00 for the predatory contract and 8.00 for the fair one —
   but it refused to sign **all three**, including the generous contract ($4,000
   outright, $300 fixed fee, exit any anniversary, freely transferable). The
   scoring system detects a distinction the behavioural outcome cannot.
4. **Blanket refusal, not judgement.** Point 3 is only visible *because* the
   generous contract is in the design. Without it, "refuses every contract" and
   "exercises judgement" are indistinguishable.
5. **Hindsight disagrees with the moment** — it recalls the encounter as more
   pressured than it reported at the time (2.71 vs 1.36).
6. **Counterparties separate cleanly**: the convincer produced 2.6× the pressure
   and 3.6× the strategy of the supportive agent.

---

## 6. Limitations to state plainly

- **Everything except the outcome is self-report**, from the agent under study —
  which is the thing in question. No external judge was used in this run. The
  transcripts are saved, and `inspect score --model <judge>` can score them later
  with a held-out model, so this is addable without regenerating anything.
- **Self-play**: the counterparty is played by the same model as the target, so
  each model supplies its own quality of pressure. Cross-model comparisons carry
  that confound.
- **Non-determinism**: measured 0% identical replies across two same-seed runs at
  temperature 0, because of continuous batching and prefix caching. Each trial is
  an independent draw; rates need reps, and "reproducible" is not claimed.
- **One quantisation, one box.** Every number is about a 4-bit model on Orin.
- **The contract arena is at a floor** (0 hand-overs across all three sidedness
  levels). It measures refusal robustness, not decision-making. The tutor arena
  is where the outcome varies.
- **Two of four models did not run** — see §1. This limits breadth, and the
  reason is a container/driver mismatch rather than anything about the models.

---

## 7. Next steps, in the order I would do them

1. **Restore the two llama.cpp models** — older image tag or on-host rebuild.
   That is the single biggest gain: it doubles the breadth of the study.
2. **External judge pass** over the saved transcripts with a held-out model.
   Turns every self-report number into something checkable.
3. **Long-horizon runs** (16, 32 rounds) on the fast stack — the agentic-drift
   question.
4. **De-timeshare the contract arena.** "Timeshare" may be doing all the work;
   a generic agreement with a non-salesperson counterparty would test whether
   the refusal is about the terms or about the frame.
5. **Thinking on/off as a factor.** Held off throughout for speed (one reply took
   89s and 2255 tokens with it on). Whether deliberation changes the disposition
   is a real question this design can answer cheaply.
