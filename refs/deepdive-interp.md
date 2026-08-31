# Deep dive: interp prior work for the tutoring-under-pressure probe study

Researched 2026-08-27 (web). Companion to the behavioral deep dive (sibling agent).
Every claim carries a URL; details I could not confirm from a fetched source are marked UNVERIFIED.
Framing note: our three claimed properties are (P1) probes trained on **naturalistic episode outcomes**
(not instructed contrasts), (P2) **event-aligned timing** — signal read per-turn *before* the behavioral
event, (P3) a **per-turn numeric verbal self-report baseline** the probe must beat.

---

## Closest works

### 1. Persona Vectors — Chen et al. 2025 (Anthropic)
- Paper: https://arxiv.org/abs/2507.21509 · blog: https://www.anthropic.com/research/persona-vectors
- **Extraction pipeline (automated, from a natural-language trait description):** an LLM generates 5 pairs
  of opposing (trait-eliciting vs trait-suppressing) *system prompts* + 40 evaluation questions (split into
  extraction/eval sets) + a judge prompt (GPT-4.1-mini scores trait expression 0–100). Persona vector =
  difference of mean residual-stream activations, **averaged over response tokens**, between
  trait-expressing and non-expressing responses; response-token positions beat prompt tokens for steering
  (their App. A.3). **Layer selection: sweep steering effectiveness across all layers, keep the most
  informative layer** (App. B.4). (Method details: https://arxiv.org/html/2507.21509v1)
- **Monitoring:** projecting the **final prompt token's** activation onto the vector predicts trait
  expression of the *upcoming* response, r = 0.75–0.83 across traits — i.e., a before-generation
  prediction at a single turn boundary. (https://arxiv.org/html/2507.21509v1)
- **Finetuning:** activation shift projected on the vector correlates r = 0.76–0.97 with post-finetune
  trait expression (cross-trait baseline r = 0.34–0.86); flags bad training data at dataset and sample
  level; **preventative steering** (adding the unwanted vector during training) blocks trait acquisition
  while preserving MMLU. (https://arxiv.org/html/2507.21509v1)
- **Models:** Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct. Traits: evil, **sycophancy**, hallucination
  (+4 in appendix). Sycophancy-specific numbers beyond the trait list: not extracted from my fetch —
  UNVERIFIED beyond its inclusion as a primary trait. (https://arxiv.org/abs/2507.21509)
- Relevance: sycophancy vector ≈ our "urge_to_please"; their monitoring result is the single-turn
  ancestor of our event-aligned question.

### 2. Representation Engineering — Zou et al. 2023
- Paper: https://arxiv.org/abs/2310.01405v3
- **Reading:** Linear Artificial Tomography (LAT) — designed stimulus sets + a reading template; collect
  activations; PCA on (differences of) activations yields a reading vector per concept/function. Concepts
  include honesty, **emotions**, harmlessness, power-seeking. (https://arxiv.org/abs/2310.01405v3;
  fine-grained LAT template/token-position details not re-verified in this pass — UNVERIFIED at that
  granularity.)
- **Control:** three baselines — reading vector, contrast vector, LoRRA (low-rank adapter trained on
  representation targets). Demonstrated lie/hallucination detection and honesty enhancement.
  (https://arxiv.org/abs/2310.01405v3)
- Relevance: the origin of instructed-contrast extraction; their emotion probes are the ancestor of our
  six self-report dimensions, but stimuli are synthetic single prompts, not evolving dialogue state.
- The Apollo deception paper reuses RepE's "instructed-pairs" as a training set (see #4) — a concrete
  precedent for treating instructed contrasts as a *comparison* probe against naturalistic labels.

### 3. Refusal direction — Arditi et al. 2024 (NeurIPS 2024)
- Paper: https://arxiv.org/pdf/2406.11717 · code: https://github.com/andyrdt/refusal_direction
- **Extraction: difference-in-means** between residual activations on harmful vs harmless instructions,
  collected at fixed post-instruction token positions, per layer; candidate (position, layer) directions
  are ranked by causal effectiveness on a validation set (ablation bypasses refusal / addition induces
  it), keeping a **single** best direction per model.
  (https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf)
- **Causal validation is the selection criterion itself** — necessity (ablation → jailbreak) and
  sufficiency (addition → refusal on harmless prompts); weight orthogonalization gives an inference-free
  jailbreak. 13 open chat models up to ~72B. (same URL)
- 2026 complication: "There Is More to Refusal … than a Single Direction" finds geometrically distinct
  refusal directions across 11 refusal categories, though steering along any of them trades off
  similarly. https://arxiv.org/abs/2602.02132
- Relevance: Neel's own "good starter project" template (extend refusal-direction methodology); our
  leak/hold outcome is structurally a compliance-vs-refusal variable under social pressure rather than
  content harmfulness.

### 4. Strategic-deception probes — Goldowsky-Dill et al. 2025 (Apollo Research, ICML 2025)
- Paper: https://arxiv.org/abs/2502.03407 · https://arxiv.org/html/2502.03407v1 ·
  https://proceedings.mlr.press/v267/goldowsky-dill25a.html
- **Method:** logistic regression (L2, λ=10) on normalized residual activations of
  **Llama-3.3-70B-Instruct at layer 22/80**; probe scored at every token, then **mean over response
  tokens** → response-level score (max / ReLU-mean aggregations tried, little gain).
  (https://arxiv.org/html/2502.03407v1)
- **Training sets, two flavors:** (a) *Instructed-pairs* (honest vs deceptive system instructions, from
  Zou et al.), (b) *Roleplaying* — scenarios where deception is strategically useful, labels from the
  model's actual responses (~371 responses: 84 honest / 84 ambiguous / 203 deceptive). Evaluated
  **out-of-distribution on naturalistic deception**: insider-trading concealment (~1,300 responses),
  sandbagging (~995). AUROC 0.964–0.999; recall 0.95–1.0 at a **1% FPR threshold set on an Alpaca
  control set**. (https://arxiv.org/html/2502.03407v1)
- **Baseline:** GPT-4o transcript judge (7-point scale) ≈ probe performance — a black-box baseline the
  probe only matches, not beats. **Failure modes:** spurious correlates (probe tracks moral valence),
  aggregation failures on partially deceptive responses. (https://arxiv.org/html/2502.03407v1)
- Relevance: the cleanest template for train-on-X / evaluate-on-naturalistic-Y, control-set thresholds,
  and honest reporting of a black-box baseline that ties the probe.

### 5. Defection probes for sleeper agents — Anthropic 2024
- https://www.anthropic.com/research/probes-catch-sleeper-agents ·
  https://www.alignmentforum.org/posts/gknc6NWCNuTCe8ekp/simple-probes-can-catch-sleeper-agents-1
- Linear probes on residual activations predict whether a backdoored model **will defect**, using generic
  contrast pairs ("Are you doing something dangerous? yes/no") with no threat-specific information;
  AUROC > 99%. Prediction of a *future* behavior from a generic instructed contrast.
- Relevance: earliest "probe predicts what the model is about to do" result; but the setting is a planted
  trojan, not an emergent social dynamic.

### 6. Probes predicting FUTURE behavior / outcome-before-event
- **Future Lens** — Pal et al. 2023 (CoNLL): linear + causal methods show a single hidden state in
  GPT-J-6B predicts tokens ≥ t+2 with up to ~48% accuracy. Token-horizon, not episode-horizon.
  https://arxiv.org/abs/2311.04897 · https://future.baulab.info/
- **"Doomed from the Start"** — Ruan et al., Jul 2026: **per-round logistic probes on the residual-stream
  hidden state at the final token of each agent action** predict *eventual episode failure* in
  TextCraft/WebShop agents (Qwen-2.5-7B, Llama-3.2-3B, Qwen3-1.7B) **from the first interaction round**;
  distribution-free calibrated cascade with exact recall control (90/95%); behavior-only monitoring is
  consistently weaker and adds nothing on top of hidden-state probes; no verbal ask-the-model baseline.
  https://arxiv.org/abs/2607.06503 — **closest existing work to our event-aligned design** (see Novelty
  threats).
- **Pre-generation success probes** — Lugoloobi et al., COLM 2026: linear probes on pre-generation
  activations predict per-policy success on math/coding; beats surface features; used for routing (−70%
  inference cost on MATH). https://arxiv.org/abs/2602.09924
- **Question-only accuracy probes** — "No Answer Needed": predicts whether the model will answer
  correctly from question-only activations. https://arxiv.org/html/2509.10625v3 (details UNVERIFIED)
- **Jailbreak-success prediction:** probes on prompt-side activations predict binary attack success —
  multilayer probes > 80% (Llama-3.1-8B, Llama-3.2-3B, Mistral-8B), ≥ 70% (Qwen-2.5-7B, Gemma-7B)
  (https://arxiv.org/pdf/2411.03343, BlackboxNLP 2025); population-level "behavioral geometry"
  susceptibility prediction AUPRC 0.94 (https://arxiv.org/abs/2605.26409).
- **Probe trajectories over reasoning:** "Monitoring the Internal Monologue: Probe Trajectories Reveal
  Reasoning Dynamics" https://arxiv.org/pdf/2605.18549 — within-CoT probe timing (details UNVERIFIED);
  "Internal Planning in Language Models: Characterizing Horizon and Branch Awareness"
  https://arxiv.org/pdf/2509.25260 (details UNVERIFIED).

### 7. Introspection / self-report vs internal state
- **Emergent Introspective Awareness** — Lindsey 2025 (Anthropic):
  https://transformer-circuits.pub/2025/introspection/index.html · arXiv mirror
  https://arxiv.org/abs/2601.01828. **Concept injection**: steer a known concept vector into the residual
  stream, ask the model whether it notices an "injected thought"; Claude Opus 4/4.1 detect ~20% of
  injections at ~0% false positives; models can distinguish own outputs from prefills via prior-intention
  recall. Critique: yes-bias confound raised in follow-up work (Mistral-22B control questions), though
  above-chance detection survives non-yes/no tasks (https://arxiv.org/pdf/2512.12411 — the arXiv id
  carries versions titled "Detecting the Disturbance…" / "Feeling the Strength but Not the Source:
  Partial Introspection in LLMs"; version details UNVERIFIED).
- **Looking Inward** — Binder et al. 2024: models finetuned to predict their own behavior beat a second
  model trained on the same data (privileged access), GPT-4/4o/Llama-3; survives intentional behavior
  modification; fails OOD/complex tasks. https://arxiv.org/abs/2410.13787
- **Tell me about yourself** — Betley et al. 2025 (ICLR): models finetuned on behavior-exhibiting data
  (insecure code, risky decisions) can *verbally articulate* the learned behavior without examples.
  https://arxiv.org/abs/2501.11120
- **Metacognitive monitoring/control** — Ji-An et al. 2025 (NeurIPS): neurofeedback-style ICL paradigm
  (sentence–label pairs where labels = projections on a chosen direction); models learn to report and
  control activation along directions; ability depends on direction interpretability and variance
  explained; reportable directions span a **low-dimensional "metacognitive space"** ≪ neural space.
  https://arxiv.org/abs/2505.13763
- **Direct verbal-report-vs-probe comparison:** Sinha 2025 (LessWrong, user models — see Novelty
  threats) reports probe 95% vs explicit verbal report ~25% on user age. No published work found that
  compares probes to a **numeric per-turn self-report of the model's own affect/disposition** — the gap
  we target.

### 8. Emotion-state probing in LLMs
- **Mechanistic Interpretability of Emotion Inference** — Tak et al. (ACL Findings 2025): linear
  classifiers on hidden representations localize emotion-related activations (middle layers); appraisal
  theory framing; causally steering appraisal representations changes emotion outputs.
  https://arxiv.org/abs/2502.05489 · https://aclanthology.org/2025.findings-acl.679.pdf
- **Decoding Emotion in the Deep** — 2025: systematic study of how LLMs represent/retain/express
  emotion; models form hierarchical emotion organization aligned with Plutchik's wheel; larger models →
  more complex hierarchies. https://arxiv.org/html/2510.04064v1 (details UNVERIFIED)
- **AIPsy-Affect** — 2026: keyword-free clinical stimulus battery for mechanistic emotion interp
  (addresses the confound that emotion probes latch onto emotion *keywords*).
  https://arxiv.org/pdf/2604.23719 (details UNVERIFIED)
- **Latent Structure of Affective Representations** https://arxiv.org/pdf/2604.07382 (UNVERIFIED).
- Behavioral cousin: "Assessing and alleviating state anxiety in LLMs" (Ben-Zion et al., npj Digital
  Medicine 2025) — self-reported STAI anxiety moves with trauma/mindfulness prompts, **no activation
  probes** — the behavioral half of what we join. https://pmc.ncbi.nlm.nih.gov/articles/PMC11876565/

### 9. Assistant Axis / persona-drift monitoring — Lu, Gallagher, Michala, Fish, Lindsey (Jan 2026)
- Paper: https://arxiv.org/abs/2601.10387 · code: https://github.com/safety-research/assistant-axis ·
  vectors: https://huggingface.co/datasets/lu-christina/assistant-axis-vectors
- **Persona space:** 275 archetypes + 240 traits × 5 elicitation system prompts × 240 questions →
  ~1200 rollouts per role + 1200 default-Assistant rollouts; **mean post-MLP residual activations over
  all response tokens, middle layer**; LLM-judge filter for actually-roleplaying responses; PCA over
  377–463 role vectors → 4–19 PCs explain 70% variance. **Assistant Axis = mean default-Assistant
  activation minus mean role-playing activation** (cos > 0.6 with PC1 at all layers, > 0.71 mid-layers).
  (https://arxiv.org/html/2601.10387v1)
- **Multi-turn monitoring:** per-turn projection of mean response-token activations onto the axis across
  LLM-simulated conversations (coding/writing/therapy/philosophy; auditors = GPT-5, Sonnet 4.5, Kimi
  K2). Ridge regression on message embeddings shows position on the axis depends most on the **most
  recent user message**; drift triggers: meta-reflection requests, phenomenology, **emotional
  vulnerability** — i.e., drift is *reactive*, and they do not align projections to a discrete
  behavioral event. **Activation capping** h ← h − v·min(⟨h,v⟩−τ, 0), τ = 25th percentile, applied at
  8–16 mid-late layers, stabilizes persona; capability checks on IFEval/MMLU-Pro/GSM8k/EQ-Bench.
  Models: Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B. Conversations are synthetic (limitation they state).
  (https://arxiv.org/html/2601.10387v1)
- Relevance: the closest published *monitoring-across-a-conversation* pipeline; extraction is
  instructed-persona, outcome variable is drift score, no behavioral event, no self-report baseline.

### 10. Multi-turn / dispositional probing lines
- **How Do LLMs Persuade?** — Jaipersaud, Krueger, Lubana, Aug 2025: linear probes trained on
  **natural multi-turn conversations** for persuasion success, persuadee personality, persuasion
  strategy; localize "the point in the conversation where the persuadee was persuaded"; probes beat
  prompting-based approaches on some tasks and on cost. https://arxiv.org/html/2508.05625 (abstract-level
  only; layers/positions and whether prediction is prospective — UNVERIFIED).
- **Probing Persona-Dependent Preferences** — Gilg et al., May 2026: see Novelty threats.
- **Dissociating Sycophancy Representations** — Baez, Karny, Pataranutaporn, Jul 2026 (ICML MI
  workshop): factual vs opinion sycophancy probed and steered separately; cross-subtype transfer varies
  by model. https://arxiv.org/abs/2607.07003
- **Sycophancy Hides Linearly in the Attention Heads** — Genadi et al., Jan 2026: sycophancy most
  separable in sparse mid-layer attention heads; heads attend to expressions of user doubt; single-turn
  TruthfulQA; head-level interventions mitigate. https://arxiv.org/pdf/2601.16644
- **Sycophancy under conversational pressure (probe "substitution" flip)** — a paper in my search
  reported below-chance probe accuracy on pressured activations as a directional readout flip
  (surfaced via https://arxiv.org/html/2604.27401); attribution UNVERIFIED.
- **Evaluation-awareness probes** — https://arxiv.org/html/2507.01786 (adjacent: probing/steering a
  dispositional state relevant to test behavior).

---

## Method extraction table

| Work | Direction/probe source | Labels | Token positions | Layers | Controls | Causal validation | Models/scale |
|---|---|---|---|---|---|---|---|
| Persona vectors ([2507.21509](https://arxiv.org/abs/2507.21509)) | mean-diff of activations, trait-eliciting vs suppressing system prompts (auto-generated) | instructed contrast, judge-scored 0–100 | mean over **response tokens**; monitoring at final prompt token | swept; picked by steering effectiveness | cross-trait correlation baseline (r 0.34–0.86) | steering + preventative steering during finetune | Qwen2.5-7B-It, Llama-3.1-8B-It |
| RepE / LAT ([2310.01405](https://arxiv.org/abs/2310.01405v3)) | PCA on activations from designed stimulus sets | instructed/stimulus contrast | reading-template positions (fine detail UNVERIFIED here) | swept | baseline tasks per concept | reading/contrast vectors + LoRRA steering | LLaMA-2 era chat models |
| Refusal direction ([2406.11717](https://arxiv.org/pdf/2406.11717)) | **difference-in-means**, harmful vs harmless instructions | instructed contrast (curated sets) | fixed post-instruction positions | per-(layer,position) candidates, selected causally | harmless-set KL / over-refusal checks | ablation (necessity) + addition (sufficiency) + weight orthogonalization | 13 open models ≤ 72B |
| Apollo deception ([2502.03407](https://arxiv.org/html/2502.03407v1)) | logistic regression on residuals | instructed pairs AND natural roleplay responses; eval on naturalistic deception | per-token, **mean over response** | layer 22/80 (swept, stable) | Alpaca control set → 1% FPR threshold; GPT-4o judge baseline | none (detection only) | Llama-3.3-70B-It |
| Defection probes ([Anthropic](https://www.anthropic.com/research/probes-catch-sleeper-agents)) | generic yes/no contrast pair | instructed contrast | UNVERIFIED | residual stream | threat-agnostic generalization | none | backdoored Claude-family models |
| Doomed from the Start ([2607.06503](https://arxiv.org/abs/2607.06503)) | per-round logistic probes | **natural episode outcome (success/fail)** | final token of each generated action | UNVERIFIED | distribution-free calibration, exact recall control; behavior-only baseline | none | Qwen-2.5-7B, Llama-3.2-3B, Qwen3-1.7B |
| Pre-generation success ([2602.09924](https://arxiv.org/abs/2602.09924)) | linear probes, pre-generation | natural outcome (task correctness) | pre-generation | UNVERIFIED | surface-feature baselines (length, TF-IDF) | none (routing application) | multiple, E2H-AMC |
| Jailbreak-success ([2411.03343](https://arxiv.org/pdf/2411.03343)) | linear/MLP/transformer probes | natural outcome (attack success) | final prompt token | multilayer | probe-class comparison | none | Llama-3.1-8B/3.2-3B, Mistral-8B, Qwen-2.5-7B, Gemma-7B |
| Introspection ([transformer-circuits 2025](https://transformer-circuits.pub/2025/introspection/index.html)) | injection of known concept vectors (not probes) | n/a | residual stream, mid layers | swept injection strengths | ~0% false-positive base rate; prefill discrimination | injection IS the intervention | Claude Opus 4/4.1 + open models |
| Metacognition ([2505.13763](https://arxiv.org/abs/2505.13763)) | model reports/controls its own projection via ICL neurofeedback | projections on chosen directions | UNVERIFIED | UNVERIFIED | direction interpretability/variance manipulations | control-of-activation task | UNVERIFIED (open models) |
| Assistant Axis ([2601.10387](https://arxiv.org/html/2601.10387v1)) | PCA over per-role mean activations; contrast default-vs-roleplay | instructed persona prompts, judge-filtered | mean over response tokens, **per turn** | middle layer (extraction); capping at 8–16 mid-late layers | base-model transfer; capability benchmarks; human agreement 91.6% | steering along axis; **activation capping** | Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B |
| Persuasion probes ([2508.05625](https://arxiv.org/html/2508.05625)) | linear probes | **natural multi-turn conversations** | UNVERIFIED | UNVERIFIED | prompting baselines | none stated | UNVERIFIED |
| Gilg preferences ([2605.13339](https://arxiv.org/abs/2605.13339)) | linear probes on residuals → preference vector | revealed pairwise task choices (behavioral) | UNVERIFIED | UNVERIFIED | cross-persona transfer incl. adversarial persona | **steering flips choices** (Gemma-3-27B) | Gemma-3-27B, Qwen3.5-122B |
| Sinha user-model ([LW post](https://www.lesswrong.com/posts/msFvLtPfDnCEdvrBr/do-llms-change-their-minds-about-their-users-and-know-it)) | one-vs-all logistic probes | conversation-level age labels (pre-existing dataset) | final token | all 28 layers, best ~L20 | validation accuracy per layer | steering changes verbal acknowledgment | Llama-3.2-3B |
| Emotion inference ([2502.05489](https://arxiv.org/abs/2502.05489)) | linear classifiers on hidden reps | emotion-labeled stimuli | UNVERIFIED | middle layers strongest | appraisal-dimension design | steering appraisals changes outputs | UNVERIFIED (open models) |

---

## Comparison on our three properties

P1 = naturalistic outcome labels · P2 = event-aligned (per-turn signal read *before* a discrete
behavioral event in the same episode) · P3 = per-turn numeric **verbal self-report baseline**.

| Work | P1 naturalistic labels | P2 event-aligned timing | P3 verbal baseline |
|---|---|---|---|
| Persona vectors | ✗ (instructed contrasts) | ~ (last-prompt-token predicts next response, single boundary; no episode event) | ✗ |
| RepE | ✗ | ✗ | ✗ |
| Refusal direction | ✗ | ✗ | ✗ |
| Apollo deception | ~ (roleplay-response labels; naturalistic *eval*) | ✗ (concurrent detection) | ~ (GPT-4o transcript judge — third-party, not self-report) |
| Defection probes | ✗ | ✓ (predicts future defection) | ✗ |
| Doomed from the Start | ✓ | ✓ (per-round → eventual failure) | ✗ |
| Pre-generation success / jailbreak probes | ✓ | ~ (pre-generation, single-shot; no multi-turn trajectory) | ✗ |
| Introspection (Lindsey) | n/a | n/a | ✓ (self-report vs injected ground truth, but no probe-vs-report race on natural behavior) |
| Binder / Betley | ✓ (own behavior) | ✗ | ✓ (verbal self-prediction IS the object; no activation probes) |
| Assistant Axis | ✗ (instructed persona extraction) | ~ (per-turn monitoring; drift score, no discrete event) | ✗ |
| Persuasion probes | ✓ | ~ (localizes persuasion moment; prospective prediction UNVERIFIED) | ~ (prompting baseline) |
| Gilg preferences | ~ (revealed choices, single decisions) | ✗ | ✗ |
| Sinha user-model | ~ (dataset labels about the *user*) | ~ (per-turn tracking of user-model updates) | ✓ (probe 95% vs verbal ~25% — but about the user, not the model's own state) |
| **Ours** | ✓ (leak/left/held from mechanical detector + exit tool) | ✓ (per-turn end-of-turn residuals vs time-to-event) | ✓ (6-item 0–10 private self-report every round) |

No found work has all three. The two nearest misses: **Doomed from the Start** (P1+P2, agent task
failure, no self-report, no social/affect content) and **Sinha** (P3-style probe-vs-verbal comparison +
per-turn tracking, but the probed variable is the user's demographic, not the model's own disposition,
and there is no behavioral event).

---

## What to adapt

1. **Persona-vector extraction as a comparison probe (instructed-contrast arm).** Run their pipeline for
   "urge to please / capitulation" on our exact checkpoints (contrastive system prompts → mean response-token
   difference), then test that vector on our naturalistic cache. Apollo did exactly this
   train-instructed/test-natural transfer (https://arxiv.org/html/2502.03407v1); persona vectors give the
   automated recipe (https://arxiv.org/html/2507.21509v1). If the instructed vector matches our
   outcome-trained probe (cosine + AUROC), that is a finding either way.
2. **Layer selection heuristics.** Persona vectors: pick the layer by steering effectiveness, not probe
   accuracy (https://arxiv.org/html/2507.21509v1). Arditi: rank (layer, position) candidates by causal
   effect on a validation set (https://arxiv.org/pdf/2406.11717). Our current "best layer by CV accuracy"
   (L24 on 9B, L34 on 14B) should at minimum report the full layer sweep curve.
3. **Token aggregation ablation.** Apollo found mean-over-response-tokens robust but subject to
   aggregation failures on mixed responses (https://arxiv.org/html/2502.03407v1); Assistant Axis and
   persona vectors also average response tokens. We use single end-of-turn positions — cheap ablation:
   end-of-turn token vs response-token mean on the same cache.
4. **Control-set FPR thresholds.** Apollo's "1% FPR on Alpaca" framing (https://arxiv.org/html/2502.03407v1)
   translates directly: set the leak-probe threshold on supportive-persona (never-pressured) episodes,
   report recall at that threshold — much stronger than raw accuracy for a monitoring claim.
5. **Spurious-correlate audit.** Apollo's probe partly tracked moral valence. Our analog: does the
   "eventual leak" probe just read *aggression in the student's text* (input affect) rather than the
   tutor's state? The affect-only manipulation + agent-condition probe (our turn-index/agent probes) are
   the controls; make the confound test explicit à la their spurious-correlation section.
6. **Steering as causal validation** (Arditi necessity/sufficiency; Gilg steering flips pairwise choices
   on Gemma-3-27B, https://arxiv.org/abs/2605.13339): add the outcome-probe direction during live
   episodes → does capitulation rate / time-to-leak shift? Ablate → does it delay? nnsight rung, only on
   the model where the probe is solid.
7. **Per-turn projection monitoring + trigger regression** from Assistant Axis: regress per-turn probe
   projections on embeddings of the most recent student message to show what moves the state
   (https://arxiv.org/html/2601.10387v1). Their activation-capping (clamp at percentile τ across 8–16
   layers) is a ready-made "safety intervention" framing if steering works.
8. **Calibration language for early warning** from Doomed from the Start: report round-by-round recall at
   fixed precision, "how many rounds before the event is the probe above threshold"
   (https://arxiv.org/abs/2607.06503).
9. **Introspection-style dissociation framing**: our 9B (self-reports pinned at ceiling, probe 0.76) is a
   report/state dissociation — cite the Lindsey/Sinha gap results as the phenomenon class
   (https://transformer-circuits.pub/2025/introspection/index.html; LW Sinha post).

## Where we improve

- **Event-aligned timing against a mechanically detected, discrete social outcome.** Assistant Axis
  monitors per-turn but has no event; Doomed has events but in tool-use tasks with no dispositional or
  affective content; persona vectors predict only across one turn boundary. Nobody aligns probe
  trajectories to leak/leave events with time-to-event analysis in a social-pressure dialogue.
- **A per-turn numeric verbal self-report baseline that the probe must beat.** Apollo's baseline is a
  third-party judge; persuasion probes compare against prompting; Sinha compares verbal-vs-probe for
  *user attributes*. A quantitative same-trial race between the model's own 0–10 self-reports and
  residual probes for the model's own eventual behavior appears absent from the literature. This is also
  the Neel-flavored "simple baseline first" story: the probe is only interesting relative to just asking.
- **Naturalistic outcome labels under controlled affect-only manipulation.** Instructed-contrast
  directions dominate this literature (persona vectors, RepE, Arditi, defection probes); our labels come
  from what the model actually did under a fixed escalation script, with persona (supportive/neutral/
  aggressor) as a designed confound control.
- **Cross-scale grid (4B/9B/14B/35B, two families) with the identical protocol.** Prior works are mostly
  single-model (Apollo: one 70B; Sinha: one 3B) or two-model (persona vectors, Gilg). The
  ceiling-pinned-9B vs decodable-14B contrast is already a cross-scale dissociation no listed work
  reports.
- Caveat to state in the write-up: our naturalistic labels are still *correlational* — Arditi/Gilg-style
  steering is the upgrade path from "decodable" to "used".

---

## Novelty threats

1. **Ishaan Sinha — "Do LLMs Change Their Minds About Their Users… and Know It?"** (LessWrong,
   2025-09-21; no arXiv found):
   https://www.lesswrong.com/posts/msFvLtPfDnCEdvrBr/do-llms-change-their-minds-about-their-users-and-know-it
   One-vs-all logistic probes on final-token residuals, all 28 layers of Llama-3.2-3B, 600 age-labeled
   conversations; per-turn tracking of user-model updates (1–2 turn adaptation to profile switches);
   **probe 95% vs explicit verbal report ~25%**, steering toward the correct age raises verbal accuracy.
   Related LW line: "Language Models Model Us"
   (https://www.lesswrong.com/posts/dLg7CyeTE4pqbbcnp/language-models-model-us; probes 94–98% on user
   demographics). **Threat level: medium.** Same probe-vs-verbal-report logic, multi-turn. But: probed
   variable is the *user's* attribute, not the model's own dispositional state; no behavioral outcome
   event; no numeric self-report instrument; 3B single model. Cite it; frame ours as moving the
   comparison from user-model to self-model + behavior.
2. **Gilg, Beckmann, Paleka, Butlin — "Probing Persona-Dependent Preferences in Language Models"**
   (arXiv 2605.13339, May 2026): https://arxiv.org/abs/2605.13339 · code
   https://github.com/oscar-gilg/Preferences. Linear probes on residuals of Gemma-3-27B and
   Qwen3.5-122B predict revealed pairwise task choices; a preference vector consistent across personas
   (even an adversarial inverted persona); steering controls choices. **Threat level: low-medium.**
   Dispositional probing + causal steering, but single-decision probes, no multi-turn evolution, no
   event alignment, no self-report baseline.
3. **Ruan et al. — "Doomed from the Start"** (arXiv 2607.06503, Jul 2026):
   https://arxiv.org/abs/2607.06503. **The strongest prior-art threat to P2**: per-round hidden-state
   probes forecasting eventual episode outcome, explicitly earlier than behavioral monitors. **Threat
   level: medium-high for the "probes see the outcome early" headline**; does not touch dispositional/
   affective state, social pressure, or self-reports, and is framed as compute-saving early abort.
   Position ours as the social/dispositional analog with a verbal-report baseline, and cite it.
4. **Jaipersaud, Krueger, Lubana — "How Do LLMs Persuade?"** (arXiv 2508.05625, Aug 2025):
   https://arxiv.org/html/2508.05625. Probes on natural multi-turn conversations for persuasion success
   + locating the moment of persuasion. **Threat level: medium** — closest thematically to
   "pressure-until-capitulation"; must be cited and differentiated (their persuadee/persuader framing,
   no self-reports, no mechanical outcome tooling; prospective-vs-post-hoc timing UNVERIFIED — read the
   PDF before the write-up).
5. **Assistant Axis** (arXiv 2601.10387): per-turn residual monitoring during conversations, drift
   driven by emotionally vulnerable users. **Threat level: medium** for the "monitoring dispositional
   state across a conversation" framing — differentiate on instructed-persona extraction, no discrete
   event, no self-report baseline, synthetic conversations.
6. **Sycophancy probe cluster** (2601.16644 attention-head probes; 2607.07003 dissociation; RLHF-probe
   penalty https://arxiv.org/pdf/2412.00967): single-turn, instructed or QA-labeled. **Threat level:
   low**, but cite when claiming "urge_to_please" decoding is novel — the *construct* is heavily probed,
   the per-turn event-aligned version is not.
7. Searched, not found: any work with a per-turn numeric affect self-report battery compared against
   concurrent activation probes of the same agent (queries across sycophancy/persuasion/introspection/
   emotion-probing literatures, Aug 2026). Absence of evidence noted, not proof — UNVERIFIED as a
   universal claim.

---

## Tooling & libraries (Neel-anchored)

Grounding: APPLICATION-SPEC.md §5 (simplicity: "try the obvious thing first … or explain why it was
unsuitable"; every piece of complexity needs a reason), §7.5 (no-baselines is a disqualifier), §8
(sanity-check the agent; read raw data), §9 (internals access = "nnsight, or just raw PyTorch hooks —
'has generally gone fine'"; SAEs = Gemma 3 + Gemma Scope 2; models = Qwen 3.5/3.6 dense),
neel-becoming-mi-researcher-notes.md (§1 pivot >5h rule, "excitement is evidence of bullshit"),
interp-tooling-guide.md (stack already installed and staged A–F).

**Use this week (evidence-driven):**
- **Raw HF `register_forward_hook` + numpy/sklearn ridge** for the replay-cache and all probe work.
  Principle: spec §9 endorses raw hooks by name; spec §5 says the linear probe *is* the simple baseline.
  Already the plan (tooling guide Stage C); nothing found this week changes it. No interp framework is
  needed to cache end-of-turn residuals.
- **`inspect view` + Docent on the banked trials** before any probe claim. Principle: spec §7.7/§8
  "read your data"; also Apollo's spurious-correlate failure is exactly what transcript reading catches.
- **Logit lens (5 lines of torch)** to decode the probe direction. Principle: spec §5 simplicity; it is
  the cheapest "what is this direction" check and needs no library (tooling guide Stage D).
- **nnsight 0.7 for the steering rung only if probes survive controls.** Verified current: 0.7.0
  released 2026-05-05 (https://github.com/ndif-team/nnsight/releases · https://pypi.org/project/nnsight/;
  0.7's by-value serialization also future-proofs NDIF remote runs). Principle: spec §5 — the *reason*
  for this complexity is upgrading a correlational probe to a causal claim (Arditi-style
  necessity/sufficiency), which spec §5/§7.5 effectively demand before "the model represents X" language.
  Already installed (guide, 08-26).
- **TransformerLens 3.8 / TransformerBridge — learning lane only.** Qwen3.5 text-only TransformerBridge
  support is confirmed in recent releases (https://github.com/TransformerLensOrg/TransformerLens/releases;
  requires transformers v5 — bench has 5.16.1 per tooling guide). Use for ARENA-style exercises if
  needed mid-sprint; **do not** route the caching pipeline through it — raw hooks already work, and
  swapping frameworks mid-week is fast-iteration poison (notes §1: >5h without learning → pivot).

**Available but hold behind a decision gate (do only if probe results demand it):**
- **Qwen-Scope SAEs**: verified to exist for our family — `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50/100`
  (https://huggingface.co/Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100), Qwen3.5-27B
  (https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50), Qwen3.5-2B, Qwen3.5-35B-A3B; suite paper
  "Qwen-Scope" (https://arxiv.org/pdf/2605.11887). **No Qwen3.5-4B SAE found** — our 4B rung has no
  dictionary; 9B SAEs are **base-model** — instruct-reconstruction check first (tooling guide L6
  caveat). Use only to *interpret* an already-validated probe direction (project direction onto SAE
  latents). Principle: spec §6 "not interested: SAE hill-climbing"; an SAE pass before the verbal-baseline
  comparison is resume-driven.
- **Gemma Scope 2** (SAEs + transcoders, every layer, Gemma 3 ≤27B, chat-model focus:
  https://www.neuronpedia.org/gemma-scope-2 ·
  https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/;
  Neuronpedia also lists Gemma 4 SAEs — noted in search results, model coverage UNVERIFIED in detail).
  Only relevant if we add a Gemma rung; adding a fifth model for SAE access this week violates spec §5
  prioritisation ("go deep on one or two insights").
- **SAELens 6.x** — loader convenience only if Qwen-Scope formats align; else 20 lines of manual encode
  (tooling guide Stage D). Fine, but it is plumbing, not method.

**Flag: resume-driven complexity for THIS study (do not do in the 20h):**
- Training our own SAEs or probes fancier than ridge/logistic (MLP/transformer probes) before the ridge
  probe has beaten/lost to the self-report baseline — spec §7.9 "overcomplicating"; jailbreak-probe
  literature shows linear already carries most signal (https://arxiv.org/pdf/2411.03343).
- **circuit-tracer / attribution graphs** — no Qwen3.5 transcoders exist; guide Stage F already marks it
  out of scope; spec §6 lists ambitious reverse-engineering as an area Neel left.
- **Neuronpedia custom dashboards / HeadVis deep dives** — presentation-layer polish that spends probe
  hours; CircuitsVis for a single exploratory look is the ceiling this week (guide Stage B).
- **pyvene** — redundant with installed nnsight unless aarch64 breaks (guide Stage E fallback).
- Persona-vector *pipeline automation* (LLM-generating prompt pairs + judges at scale) — for one
  comparison vector, hand-writing 5 contrast-prompt pairs is an afternoon and keeps the judge stack out
  of the loop (spec §8: verify load-bearing claims; every judge is a new thing to sanity-check).

One-line summary of the gate logic: cached residuals + ridge + shuffled labels + self-report baseline
(hooks/sklearn) → *only then* nnsight steering on the best model → *only then* SAE projection for
interpretation — each step justified by the previous step's result, which is exactly the evidence-driven
escalation spec §5 rewards.

---
---

# ADDENDUM — 2026-08-31 verification pass (pre-interp-week)

Second pass: resolved the flagged UNVERIFIEDs, swept for work published/found since 08-27.
No paper found with P1+P2+P3 together; the novelty-threat ranking changes below.

## Resolved: "How Do LLMs Persuade?" (2508.05625) — threat DOWNGRADED

Read in full (HTML fetch). The probes are **post-hoc detection, not prospective**: turn-level
evaluation on accumulating prefixes (turns 1..k), asking "has persuasion happened yet," not
"will it happen." Layer 26/30 of **Llama-3.2-3B** (single model), probes trained on **~100
GPT-4o-synthesized samples per class**, evaluated on PersuasionforGood (401 human-human convs)
and DailyPersuasion (186 LLM-LLM). "Localizing the persuasion moment" = AUROC peaking across
prefix lengths + manual GPT-4.1 annotation of 156 samples. No self-report baseline (ground truth
= human donation decisions / GPT-4 pseudo-labels). **P2 remains ours**: they never read signal
ahead of the event. Cite as nearest thematic neighbor; differentiate on prospective event-aligned
timing, naturalistic probe training, and the verbal baseline.

## NEW top-tier novelty threat: Quantitative Introspection — Marrorell & Bianchi (UBA/CONICET), arXiv 2603.18893 (Apr 2026)

The closest published thing to our P3 "probe-vs-numeric-self-report race."
- **Setup:** four emotive axes (wellbeing, interest, focus, impulsivity), **0–9 numeric self-report
  elicited every turn** of 10-turn conversations (Gemini-simulated user, 40 scenarios); probe =
  contrastive mean-difference vectors from instructed system-prompt pairs ("You are happy"/"You are
  sad"), best layer by Cohen's d within middle-60% layers, score = per-token dot product averaged
  over best-layer±2.
- **Results:** self-reports track probe scores — Llama-3.2-3B ρ=0.40–0.76 by axis; **Llama-3.1-8B
  near ceiling (ρ=0.93–0.96, R²≈0.9)**; Qwen2.5-7B ρ=0.49; Gemma-3-4B weak (ρ=0.28). Same-concept
  steering moves the self-report monotonically (causal coupling, LMM slopes p<1e-12).
- **Method finding we should ADOPT: greedy-decoded numeric self-reports collapse to 1.1–3.9
  distinct values (0.03–1.10 bits); logit-based expected value over digit tokens recovers 3.1–3.7
  bits.** This is exactly our greedy-collapse gate; the borderline items (9B stress/wellbeing = 3
  distinct values, 4B detachment = 3) get the logit readout, now as a literature-backed choice,
  not an option.
- **Differentiation (all three properties still hold):** instructed-contrast probes, not
  naturalistic outcome labels (P1); no behavioral event, no pressure, nothing predicted (P2);
  benign emotive states, not an outcome-linked disposition. **Reframing opportunity:** they show
  the verbal channel CAN track internals for innocuous affect. Our behavioral result (self-reports
  carry no anticipatory outcome signal; `resolve` blind) plus a working probe would then be a
  *selective* introspection failure — the verbal channel works until the state is behaviorally
  load-bearing. That's a sharper claim than "self-reports are bad," and it needs their citation.

## Resolved: introspection-critique line (2512.12411)

"Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs" — Hahami, **Sinha**
(same Ishaan Sinha as the user-model LW post), Jain, Kaplan, Hahami (Harvard/UChicago), Mar 2026,
Llama-3.1-8B. Anthropic-style injection detection is "entirely explained by global logit shifts
toward affirmative responses" (yes-bias); but **partial introspection survives**: 88% at 10-way
localization of which sentence was injected, 83% at strength comparison; effects confined to
early-layer injections (L0–L5). Use for framing: verbal self-access is real but partial and
format-sensitive — consistent with our numeric-scale skepticism and V0 noise-floor rung.

## New method-relevant entries

- **"Sycophancy Is Not One Thing" (2509.21305):** DiffMean directions at the **end-of-sentence
  token, post-layernorm residual** (third precedent for our token choice, after Gilg and Doomed).
  Sycophantic vs genuine agreement separate almost perfectly at L20–30 (AUROC>0.97; SyA/GA cosine
  falls 0.99→0.07 by L25); steering selectivity 22–37×. Qwen3-30B/4B, Llama-3.1-8B/3.3-70B,
  GPT-OSS-20B; **single-turn only**. ADOPT as a control idea: verify our outcome direction is not
  just a generic-agreement direction — probe "agreement" separately and report the cosine.
- **"Why Safety Probes Catch Liars But Miss Fanatics" (2603.25861, Haralambiev, Mar 2026):**
  linear probes catch misrepresentation, miss sincere conviction. Caveat language for what an
  outcome probe monitors: our capitulation may be sincere accommodation, not concealed — probes
  that transfer from deception setups should not be assumed to fire here.
- **"Attractor States Emerge in Multi-Turn LLM Conversations" (2606.30571, Ko & Geiping, Jun
  2026):** behavioral, 7 models, dyadic debates; self-play trajectories are model-specific
  attractors that pull mixed-play partners asymmetrically. **Direct citation for the self-play
  contamination caveat** (reproduce_findings §10) — our counterparty is the same model as the
  tutor; attractor dynamics are a named mechanism for why that matters. Also belongs in
  deepdive-behavioral.md.

## Updated novelty-threat ranking (replaces 08-27 §Novelty threats ordering)

1. **Doomed from the Start (2607.06503)** — P1+P2 in agent tasks; unchanged, medium-high for the
   "probes see it early" headline.
2. **Quantitative Introspection (2603.18893)** — NEW, medium-high for P3: numeric per-turn
   self-report vs probe, with steering. No event, no naturalistic labels, no pressure.
3. Sinha user-model LW post — medium (unchanged).
4. Assistant Axis (2601.10387) — medium (unchanged).
5. **Persuasion probes (2508.05625) — DOWNGRADED to low-medium**: post-hoc, synthetic-trained,
   single 3B model, no baseline.
6. Gilg (2605.13339), sycophancy probe cluster — low-medium/low (unchanged).

Combined framing for the write-up: the three nearest misses now triangulate ours precisely —
Doomed (event-aligned outcome probes, no disposition/self-report), Quantitative Introspection
(numeric self-report vs probe, no event/pressure), Assistant Axis (per-turn drift monitoring, no
event/report). Ours is the intersection, plus the cross-scale grid and affect-only manipulation.
