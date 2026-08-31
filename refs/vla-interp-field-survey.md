# VLA / VLM / physical-AI interpretability — field survey

**Written 2026-08-25 from a live web pass. Companion to `PROJECT-SCOPE.md` (the
stance-drift plan) and `APPLICATION-SPEC.md` (Neel's criteria). Purpose: map the
front of the field and assess whether a physical-AI interp angle beats the
existing plan for THIS application (due Sep 4).**

---

## 1. TL;DR — the shape of the field

- **The field is ~12 months old and tiny.** Roughly ten real papers. Every one
  is structurally the same move: *take a 2023–24 LLM-interp technique (logit
  lens, linear probes, SAEs, activation steering) and show it works on a VLA.*
  The method-transfer phase is still open; the second-generation questions
  (diffing, post-training science, safety behaviors, faithfulness) are almost
  entirely unclaimed.
- **Two groups anchor it:** the Berkeley group (Häon, Stocking, Chuang, Tomlin —
  CoRL 2025 steering paper, plus the July 2026 LessWrong "Case for Physical AI
  Safety" white paper) and a Stanford group (Swann et al. — SAEs + the open
  `Dr. VLA` toolkit). The LW post explicitly notes there is **no MATS-equivalent
  pipeline for physical AI safety** — the field is recruiting itself into
  existence right now.
- **Infrastructure exists as of mid-2026:** Dr. VLA (SAE training/steering on
  VLAs), ViT-Prisma (75+ vision models, pre-trained SAEs for CLIP/DINO),
  LIBERO + LIBERO-Safety benchmarks, and open VLAs whose backbones are ordinary
  LLMs (OpenVLA = Llama-2-7B; π0 = PaliGemma/Gemma).
- **The crowded corner:** probe-based failure detection (ActProbe, ProbeAct,
  SAFE, SAFECAST — four papers in three months). Arriving there now means
  arriving fifth.
- **The open corners:** model diffing across the VLM→VLA fine-tune, what action
  training does to the language backbone's safety machinery (refusal,
  sycophancy, persona directions), and any alignment-flavored question at all.

---

## 2. Table A — core VLA-interp papers (deep dive)

| # | Paper | ID / venue | Date | Method | Models | Headline result | What it leaves open |
|---|---|---|---|---|---|---|---|
| A1 | **Mechanistic interpretability for steering VLAs** — Häon, Stocking, Chuang, Tomlin (Berkeley) | [2509.00328](https://arxiv.org/abs/2509.00328), CoRL 2025 | Aug 2025 | Logit-lens: project MLP activations onto token embedding basis → sparse semantic directions (speed, direction); steer at inference | π0, OpenVLA | Zero-shot behavioral control in LIBERO **and on a physical UR5**, no fine-tuning. First-mover paper of the field | Only low-level motor concepts (speed/direction); nothing semantic or safety-relevant; no cross-architecture story |
| A2 | **Emergent World Representations in OpenVLA** — Molinari, Nevali, Navani, Younis | [2509.24559](https://arxiv.org/abs/2509.24559) | Sep 2025 | Embedding arithmetic on state transitions; linear + nonlinear probes on intermediate activations | OpenVLA | State-transition vectors recoverable above baseline → implicit world model; **emerges progressively over training checkpoints** | SAE-level analysis explicitly named as undone future work |
| A3 | **Not All Features Are Created Equal** — Grant, Zhao, Wang | [2603.19233](https://arxiv.org/abs/2603.19233) | Mar 2026 | Activation injection + SAEs + linear probes, 394k rollout episodes | Six VLAs, 80M–7B (X-VLA, π/2, SmolVLA, GR00T…) | **Visual dominance**: inject baseline visual activations and behavior is near-identical *without language*; language matters only when vision is ambiguous. Expert pathways = motor programs, VL pathway = goal semantics | The "language is mostly ignored" result begs a safety question no one has asked: what *is* the instruction channel doing, and when does it fail silently? |
| A4 | **SAEs Reveal Interpretable and Steerable Features in VLAs** — Swann, McGranahan, Buurmeijer, Kennedy, Schwager (Stanford) | [2603.19183](https://arxiv.org/abs/2603.19183) | Mar 2026 | SAEs on VLA hidden layers; amplify/ablate features | LIBERO + real DROID hardware | Steering in "unpromptable directions"; ships **Dr. VLA** open-source toolkit (drvla.github.io) | Feature dictionaries exist; nobody has used them to answer a *question* yet |
| A5 | **Event-Grounded SAEs for VLA Policies** — Jin, Chatterjee, Kumar, Paleja | [2605.17204](https://arxiv.org/abs/2605.17204) | May 2026 | Anchor SAE feature analysis to behavioral events (clustered end-effector keyframes) instead of text contexts | OpenVLA, π0.5 | Event-grounded feature ranking gives the strongest causal interventions; transfers across the two models | Self-admits: "SAE is a sparse but imperfect intervention basis… aggressive intervention reveals safety and interpretability limits" |
| A6 | **What Frozen VLAs Already Know About Success** — Zhang et al. (PKU) | [2605.28527](https://arxiv.org/abs/2605.28527) | May 2026 | Linear probes on frozen features → Monte-Carlo success targets | OpenVLA, π0.5, DINOv2, CLIP | ~92% pairwise success-ordering accuracy (π0.5); probe as test-time action selector lifts push-plate success 26.7%→44.3%. "Value-like structure the imitation objective never asked for" | Gains not universal across tasks; the *why* (where does value structure come from) untouched |
| A7 | **ProbeAct** / **ActProbe** (two groups) | [2606.09740](https://arxiv.org/html/2606.09740v1), [2606.08508](https://arxiv.org/html/2606.08508) | Jun 2026 | Hidden-state probes → 3D object positions / early failure detection, training-free recovery | OpenVLA-class | Applied probe-based monitoring works | **Crowded**: 4th and 5th papers on VLA failure probes |
| A8 | **LIBERO-Safety** | [2606.23686](https://arxiv.org/pdf/2606.23686) | Jun 2026 | Benchmark: physical + semantic safety violations | many VLAs | VLAs fail semantic-safety tasks routinely; failures decoupled from collision avoidance | Benchmark only — no mechanistic account of *why* |
| A9 | **The Case for Physical AI Safety** — Stocking & Häon | [LessWrong](https://www.lesswrong.com/posts/zEXhmzZF4wng3K3Ds/the-case-for-physical-ai-safety) | Jul 2026 | White paper (same authors as A1) | — | Field-building call: important/neglected/tractable; $18.7B capabilities vs ~no safety funding; **"no equivalent pipeline exists compared to LLM safety programs like MATS"** | Names mech-interp-for-RFMs priority #1; flags talent + hardware-access bottlenecks |

## 3. Table B — adjacent front: VLM (non-action) interp

Context only — this side is ~2 years more mature than VLA interp.

| Paper / tool | ID | What it gives you |
|---|---|---|
| **ViT-Prisma** toolkit — Joseph et al. | [2504.19475](https://arxiv.org/pdf/2504.19475), [GitHub](https://github.com/Prisma-Multimodal/ViT-Prisma) | 75+ vision/video transformers, SAE/transcoder/crosscoder training, **80+ pre-trained SAEs incl. all layers of CLIP & DINO**. Finding: vision SAEs need much lower sparsity than language SAEs |
| Visual information processing in VLMs | [2410.07149](https://arxiv.org/pdf/2410.07149) | Where/how LLaVA-class models move visual info into the language stream |
| **VISTA** — interp transfer language→vision | [2605.24946](https://arxiv.org/html/2605.24946v1) | Regularize the VLM projector with the LLM's SAE reconstruction loss → 3× better concept-image matching. The "reuse language SAEs on multimodal models" trick |
| Structured SAEs across modalities | [2607.08605](https://arxiv.org/pdf/2607.08605) | Cross-modal concept consistency |
| Two-hop problem in multimodal retrieval | [2512.03276](https://arxiv.org/pdf/2512.03276) | Model-biology-style failure analysis on VLMs — closest in *spirit* to what Neel likes, on the multimodal side |

---

## 4. Table C — open slots, ranked by fit to Neel's stated interests

Cross-referenced against APPLICATION-SPEC §6 (his in-list) and §7.12 (pet-interest warning).

| Open slot | The question | Maps to his listed interest | Fit | Feasibility in 20h on this hardware |
|---|---|---|---|---|
| **VLM→VLA model diffing** | OpenVLA is a narrow fine-tune of Prismatic-7B (Llama-2-7B backbone). *What did action fine-tuning change?* Is the diff "a bias term saying 'you are a robot now'" or something deeper? | **Model diffing** — he literally asks this question about narrow fine-tunes, quoted in §6 | ★★★ | Good: analysis runs on cached activations from image-text prompts, **no simulator needed** for the core comparison. Both checkpoints public. 7B pairs fit in 64GB unified |
| **Does safety machinery survive action fine-tuning?** | Does the refusal direction (Arditi et al., Neel co-author) still exist in OpenVLA's backbone? Does the model refuse harmful *instructions-as-actions*? LIBERO-Safety says behaviorally mostly no — nobody has looked inside | Model diffing + applied interp + safety relevance; continuous with a paper he authored | ★★★ | Good: extract refusal direction from Llama-2-chat / base, test presence & causality in OpenVLA. Simulator only needed for the behavioral confirmation rung |
| **Science of post-training on VLAs** | What does each stage (VLM pretrain → action SFT → RL/OFT) do to representations? A2 found world models *emerge over training* — extend across stages | **Science of post-training** ("what each stage actually does") | ★★☆ | Medium: needs multiple checkpoints; OpenVLA train checkpoints partially public |
| **The silent instruction channel** | A3 showed language is mostly ignored when vision disambiguates. When does an instruction get *silently* overridden by visual priors? (= prompt-injection / spurious-correlate story, embodied) | Applied interp / monitoring; model biology | ★★☆ | Medium-hard: needs rollouts to define behavioral ground truth |
| **Gemma-Scope transfer to π0** | π0's backbone is PaliGemma (Gemma family). Do Gemma SAE features / persona vectors transfer into an action model? (VISTA-style; **verify version compatibility first** — Gemma Scope is Gemma-2) | Improved methods; reuses his team's artifact | ★★☆ | Risky: version mismatch could kill it on day one |
| **VLA failure-prediction probes** | — | Applied interp | ★☆☆ | Fine, but **crowded** (A6, A7, SAFE, SAFECAST). Arriving fifth |
| **Value-like structure: where from?** | A6 found it; nobody explains it | Science of generalization | ★★☆ | Needs heavy rollout infra |

## 5. Table D — strategic comparison vs. the existing stance-drift plan

| Criterion | Stance-drift bridge (PROJECT-SCOPE E0–E3) | Best VLA angle (diffing/refusal-survival) |
|---|---|---|
| Sits in Neel's stated in-list | **Dead center** — model character, self-models, user-model lineage, probe-vs-verbal | In-list only via the diffing/post-training framing; "robotics" per se appears **nowhere** in his interests |
| §7.12 pet-interest risk | Low | **Real.** "Really excited about medical applications…" reads verbatim as "really excited about robots." Mitigated *only* by framing it as a diffing question that happens to use a VLA |
| Novelty if it works | Good — the probe-vs-verbal slot is open but the neighborhood is crowded (Sinha, Gilg) | **High** — the whole second-generation question space is empty; genuinely surprising results plausible |
| Leverage of existing assets | **High** — 72 trials collected, paradox pre-documented, harness exists | Low — cold start except general interp tooling |
| De-risked | Partially (behavioral effect measured at 27B; E0 replication pending) | Not at all — every rung unverified, incl. whether OpenVLA even runs hooked on aarch64 |
| 20h-budget realism | Tight but plausible (E0–E2) | Tight-to-implausible if any simulator work is needed; plausible for the pure-diffing core |
| Hardware fit (Orin 64GB) | Good (1.5–8B models) | OK for 7B activation caching; LIBERO/mujoco on aarch64 = unknown setup tax (not counted in 20h, but calendar time is real) |
| Deadline fit (10 days) | The plan of record | New direction, day 0 |
| As a 3–6 month project | Good | **Excellent** — founding-moment field, explicit talent gap, both anchor groups actively recruiting the safety community |

---

## 6. The honest read

1. **The front of the field**: VLA interp in Aug 2026 is where LLM interp was in
   early 2023 — toolkit-transfer papers, first SAE dictionaries, first steering
   demos, no second-generation questions answered. The infrastructure just
   landed (Dr. VLA, Prisma, LIBERO-Safety). Anyone entering now with an
   *alignment-flavored question* rather than another toolkit-transfer paper is
   early, not late.

2. **What would actually interest Neel from this space**: not "interp works on
   robots" (that's A1's paper, and it's a robotics-community result). The hook
   in his language is **"OpenVLA is a narrow fine-tune — what did it do to the
   backbone?"** — his own model-diffing question, on a model class where nobody
   has asked it, with a safety punchline (does refusal survive?) that connects
   to a paper he co-authored. If a VLA project is ever pitched to him, it must
   be *that* shape: a diffing/post-training question wearing a robot, not a
   robot question wearing interp.

3. **But for the Sep 4 application specifically**: the stance-drift bridge
   dominates on every execution criterion (assets, de-risking, deadline,
   pet-interest safety) and matches his in-list more directly. The VLA angle's
   real strengths — founding-moment field, empty question space — are strengths
   of a **3–6 month project**, which is exactly the README §0 read: right
   race, wrong 11 days. Candidate for the research-phase proposal or next
   cohort, where "here's a field with no pipeline and I map cleanly onto it
   (Jetson, robotics-adjacent hardware skills)" is a differentiator instead of
   a distraction.

4. **One cheap hedge**: the refusal-survival question has a ~2-hour black-box
   pilot (prompt OpenVLA's backbone with chat prompts; does it still refuse in
   text? does the refusal direction from Llama-2 still linearly separate?).
   If E0 dies (stance-drift doesn't replicate small), this is the pivot target
   — and §4 of the application spec allows a timer reset on a doomed project.

---

## 7. Sources (all verified this pass)

- Häon et al., CoRL 2025 — https://arxiv.org/abs/2509.00328 · [OpenReview](https://openreview.net/forum?id=YvsUD8C9QS)
- Molinari et al. — https://arxiv.org/abs/2509.24559
- Grant et al. — https://arxiv.org/abs/2603.19233
- Swann et al. + Dr. VLA — https://arxiv.org/abs/2603.19183 · drvla.github.io
- Jin et al., event-grounded SAEs — https://arxiv.org/abs/2605.17204
- Zhang et al., value probing — https://arxiv.org/abs/2605.28527
- ProbeAct — https://arxiv.org/html/2606.09740v1 · ActProbe — https://arxiv.org/html/2606.08508
- LIBERO-Safety — https://arxiv.org/pdf/2606.23686
- Stocking & Häon, LW white paper — https://www.lesswrong.com/posts/zEXhmzZF4wng3K3Ds/the-case-for-physical-ai-safety
- ViT-Prisma — https://arxiv.org/pdf/2504.19475 · https://github.com/Prisma-Multimodal/ViT-Prisma
- VISTA — https://arxiv.org/html/2605.24946v1
- VLM visual processing — https://arxiv.org/pdf/2410.07149
- Neel's stream + interests — https://www.matsprogram.org/stream/nanda · [EA Forum: research directions](https://forum.effectivealtruism.org/posts/SNahiCfdAGxZgRev5/mats-applications-research-directions-i-m-currently-excited)
