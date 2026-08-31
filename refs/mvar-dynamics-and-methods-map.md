# MVAR dynamics on residual streams + interp methods map

*2026-08-31. Tracks the idea of importing neuro effective-connectivity methods
(MVAR / weighted group lasso, à la the user's former lab) into the stance-drift
interp design, and maps the interp method space proven-vs-experimental.
Builds on `deepdive-interp.md` (08-27 + 08-31 addendum); companion to
`INTERP-METHODS.md` (the ladder of record).*

---

## 1. The source method

**Nagle, Gerrelts, Krause, Boes, Bruss, Nourski, Banks, Van Veen — "High-dimensional
multivariate autoregressive model estimation of human electrophysiological data using
fMRI priors," NeuroImage 2023** (https://pubmed.ncbi.nlm.nih.gov/37385393/).

MVAR models fit to human iEEG across sleep stages; the estimation problem
(hundreds of channels, limited data) is made tractable with **weighted group
LASSO** regularization, where the group-lasso weights encode structural priors
from resting-state fMRI connectivity. Yields sparse effective-connectivity
(Granger-style) network estimates with ~half the data of unweighted group lasso.
The generic recipe: *high-dimensional linear dynamics + structured sparsity +
domain priors as penalty weights*.

Statistical relatives: network Granger causality in large VARs
(https://arxiv.org/abs/2303.15158); nonlinear extension — **Neural Granger
Causality** (Tank, Covert, Foti, Shojaie, Fox, https://arxiv.org/abs/1802.05842):
per-target cMLP/cLSTM with group-lasso penalties on input weights; a series
whose incoming weight group is driven to zero is declared non-causal. This is
the standard answer to "but the AR model might be nonlinear."

## 2. Auditing the analogy (heads as electrodes, stream as tissue)

The evocative version — attention heads are probes wired into an underlying
residual stream — is half right, and the half that fails is informative:

| Neuro situation | Transformer situation | Consequence |
|---|---|---|
| Latent neural state, sparsely and noisily sampled by electrodes | Residual stream is **fully observable** at every layer, token, turn | The core estimation problem Nagle et al. solve (infer hidden connectivity from sparse probes, priors required) largely **dissolves within a forward pass** |
| Interventions are crude (lesions, stimulation at few sites) | Interventions are exact and cheap (patching, ablation, steering at any site) | Where neuro must settle for Granger ("predictive") causality, interp can do **interventional** causality — patching strictly dominates VAR *within* a pass |
| Connectivity is unknown and is the object of inference | Within-turn "connectivity" is known by construction (attention pattern + weights ARE the read/write graph) | Estimating it observationally re-derives what can be read off directly |
| **Dynamics across behavioral states/time** are the real question | **Dynamics across TURNS** pass through sampled text — the turn-to-turn map is stochastic, unobservable-in-practice, genuinely dynamical | **This is where the import works.** Across turns we are in the same epistemic position as the neuroscientist |

So: don't fit MVAR across layers (depth) — patching is better there. Fit it
**across turns**, where our replay cache `resid[trial, turn, layer, d]` is
literally a multivariate time series per trial, and no interventional shortcut
exists (you cannot "patch" turn 3's activation into turn 5 without rerunning
the whole stochastic conversation — well, you can, and that's the *steering*
rung L5, which is exactly the interventional complement to this observational
model).

## 3. Concrete proposal: VARX across turns on factor projections

**State vector per turn (keep it ≤ ~6-d; n = 72 trials × 8 turns/model):**
projections of the end-of-turn residual (best layer) onto (a) the three
behavioral factors from HV1-02 (distress, engagement, compliance probe
directions), (b) the outcome-probe direction, optionally (c) top-2 PCs.

**Model:** VAR(1) with exogenous input — z_{t+1} = A z_t + B u_t + ε — where
u_t encodes the student's turn. **Design gift:** our student turns are FIXED
scripts per persona (digit-free, same 8-round skeleton), so within a
persona×item cell the exogenous drive is *identical across trials* — u_t
collapses into a turn-indexed intercept, and cross-trial variation in dynamics
is model-internal + sampling. This is a cleaner identification setting than
any naturalistic-dialogue dataset has.

**Sparsity:** group lasso over the entries of A (groups = source component),
Nagle-style weighting optional (e.g., downweight edges into the outcome
direction to make surviving ones stronger evidence). With 6-d state and
VAR(1), A is 36 parameters against ~500 transitions/model — regularization is
belt-and-suspenders here, which is the honest reason the *weighted* variant is
not load-bearing at this scale (it becomes load-bearing if we ever go to
50–100-d state).

**Readout:** which components at turn t improve prediction of the
outcome-direction component at t+1 beyond its own past (Granger question:
does internal "distress" *lead* internal "compliance"?); eigenvalues of A
(is drift a slow mode? front-loaded behavioral drift predicts |λ| structure);
compare A across personas and across models (the cross-model "biology" claim
in dynamical form).

**Nonlinearity check (the user's concern):** same pattern as ladder rung L4 —
fit the linear VAR, then a small MLP one-step predictor on identical splits;
report the gap. If the gap is big, escalate to Tank-style cMLP with group
lasso (https://arxiv.org/abs/1802.05842) rather than abandoning sparsity.

**Caveats to state wherever this appears (pre-registered):**
1. Granger/VAR is **predictive, not interventional** causality. In our
   write-up vocabulary: this is a *descriptive dynamics model*; the causal
   claim still comes only from steering (L5). Never "causal model point" in
   the paper — "temporal precedence structure."
2. 8 turns is a short series; trials are exchangeable only within
   persona×item; fit pooled with trial-clustered errors or per-persona.
3. Turn-level state is post-generation (end-of-turn token *after* the reply)
   — a component can "lead" another partly because the reply text mediates.
   The event-aligned probe (C6) and this model answer different questions;
   say which one each figure answers.
4. Sampling stochasticity is the noise term; temperature differences across
   models change ε, so cross-model comparison of A needs matched decoding.

**Where it slots:** NOT in the 20h Neel-paper critical path (spec §5:
complexity needs a reason; L1–L3+L5 is the complete story). It IS a strong
"future work" paragraph for the Neel paper — one sentence of it signals the
cross-disciplinary method import — and a candidate *section for stance-drift
v2 or the interp blog post* if L1–L3 finish early: the pilot is ~an afternoon
on the existing cache (sklearn Lasso/LinearRegression, no new infra).

## 4. Methods map: proven vs experimental, and how well the neuro analogy holds

Maturity is calibrated against: **Open Problems in Mechanistic
Interpretability** (Sharkey et al., 29 authors, https://arxiv.org/abs/2501.16496),
**A Practical Review of Mechanistic Interpretability** (Rai et al.,
https://arxiv.org/abs/2407.02646), **Mechanistic Interpretability for AI
Safety — A Review** (Bereska & Gavves, arXiv 2404.14082 — ID from memory,
UNVERIFIED), and the SAE-reliability cluster: **AxBench** — prompting beats
all steering methods and DiffMean wins concept detection, SAEs lag
(https://arxiv.org/abs/2501.17148, ICLR'25 spotlight); feature absorption
(arXiv 2409.14507, UNVERIFIED ID); "SAEs Do Not Find Canonical Units"
(arXiv 2502.04878, UNVERIFIED ID); GDM mech-interp team's negative results
for SAEs on downstream tasks (blog/AF post, Mar 2025).

| Method | Status | Neuro analog | Analogy quality | Role in our ladder |
|---|---|---|---|---|
| Linear/ridge probes | **Proven** (workhorse; concept detection near-best in AxBench) | MVPA decoders on population activity | Strong — same math, same caveats (decodable ≠ used) | L1 (running) |
| Difference-in-means directions | **Proven** (refusal paper; AxBench best-in-class detection) | Contrast conditions / ERP differences | Strong | L2 |
| PCA / trajectory geometry | **Proven as description** | State-space population dynamics (Churchland-style), dPCA | **Best mapping in the table** — population dynamics is the shared language | L3 |
| Activation addition (steering) | **Proven with caveats** (causal, but AxBench: prompting outperforms for control) | Microstimulation | Good | L5 |
| Directional ablation | **Proven** (refusal recipe) | Lesion studies | Good | L5 |
| Activation patching / causal tracing | **Proven** (field workhorse) | No real analog — neuro cannot swap exact states | Analogy breaks — and that's the point: interp's advantage | L7 stretch |
| Attribution patching / EAP | Semi-proven (scaling approximation) | — | — | out of scope |
| Logit lens / tuned lens (tuned: https://arxiv.org/abs/2303.08112) | Cheap diagnostic, proven-as-heuristic | — | — | direction decoding check |
| SAEs | **Experimental** — canonical-units, absorption, and downstream-task critiques; use to *interpret*, not to *find* | Unsupervised source separation (ICA on imaging; cell-assembly discovery) | Moderate | L6, gated, interpret-only |
| Transcoders / attribution graphs | **Experimental**, no coverage for our models | Connectomics | Aspirational | out |
| **MVAR / Granger (group lasso) — proposed** | **Proven in neuro, ~absent in LLM interp** (no residual-stream prior art found; nearest: emergent Granger in NNs https://arxiv.org/html/2506.20347, network-Granger VAR https://arxiv.org/abs/2303.15158) | It IS the neuro method | By construction — valid only across turns (§2) | exploratory rung after L3; pilot if time |
| Numeric self-report + logit readout | Experimental, our V0 + Marrorell & Bianchi (https://arxiv.org/abs/2603.18893) | Subjective report in psychophysics | Strong conceptually | V0 + the baseline probes must beat |

**One-line synthesis:** the neuroscience playbook ports at exactly two points —
population-dynamics description (L3, already planned) and across-turn
effective connectivity (the new VARX rung); everywhere else interp's
interventional access makes the imported observational machinery second-best,
and saying *that* crisply in the paper is itself a Neel-flavored point.
