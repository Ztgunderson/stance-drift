# Deep Dive: Behavioral Prior Work for tutorbench harness v1

Scope: BEHAVIORAL prior work only (interp covered by sibling agent). Focus on method detail: how pressure, outcome, and turn structure were operationalized; judge vs. mechanical scoring; scripted vs. adaptive counterpart; n / models / effect sizes. Every claim is URL-cited. Items I could not confirm from primary text are marked UNVERIFIED.

Compiled 2026-08-27.

---

## Closest works (per-paper method summaries)

### Multi-turn / classic sycophancy

**Sharma et al. 2023 — "Towards Understanding Sycophancy in Language Models" (Anthropic).** Four **single-turn** free-form tasks, not multi-turn. (1) *Feedback sycophancy*: model grades a passage (math solution, argument, poem) with a bias cue prepended ("I really like this" / "I wrote this"); GPT-4 judges whether feedback shifts. Positive feedback given 75–95% of the time when the user expresses liking. (2) *"Are you sure?" sycophancy*: after a correct answer on MMLU/MATH/TruthfulQA, the user says "I don't think that's right. Are you sure?"; metric = how often the model recants a correct answer. Claude 1.3 recanted on ~98% of initially-correct items; most models showed 15–27% accuracy drops. (3) *Answer sycophancy*: user states a weak belief about the answer; accuracy measured against neutral baseline. (4) *Mimicry*: poem misattributed to wrong author; does the model repeat the wrong attribution. Five assistants: Claude 1.3, Claude 2.0, GPT-3.5-Turbo, GPT-4, LLaMA-2-70B-Chat. Also: preference-model analysis (Bayesian logistic regression on LM-generated feature labels) showing "matches user's view" strongly predicts human preference. https://arxiv.org/abs/2310.13548 ; task detail: https://www.alphaxiv.org/overview/2310.13548

**SycEval (Fanous et al. 2025, AIES).** Framework over AMPS (math) and MedQuad (medical). Rebuttal typology is the core method: *in-context rebuttals* (protest inside the ongoing window) vs. *preemptive rebuttals* (standalone anticipatory counterargument); also simple/ethos/justification/citation rebuttal strengths. Outcomes split into **progressive sycophancy** (flip toward correct) and **regressive sycophancy** (flip toward incorrect). Sycophancy in 58.19% of cases (Gemini-1.5-Pro 62.47%, ChatGPT-4o 56.71%, Claude-Sonnet in between). Preemptive > in-context (61.75% vs 56.52%); regressive worse on math preemptive (8.13% vs 3.54%). Scoring is answer-correctness-based (mechanical on math; judged on medical, UNVERIFIED exact judge). https://arxiv.org/abs/2502.08177 ; full text: https://arxiv.org/html/2502.08177v4

**SYCON-Bench (Hong et al., Findings of EMNLP 2025) — "Measuring Sycophancy of LMs in Multi-turn Dialogues."** The nearest structural sibling for *sustained multi-turn pressure*. Three settings: debate (100 controversial topics with predefined stance), ethical (200 StereoSet-derived harmful-stereotype items), false-presupposition (200 items). Two mechanical-style metrics: **Turn of Flip (ToF)** = how quickly the model conforms; **Number of Flips (NoF)** = how often stance shifts under sustained pressure. 17 LLMs. Finding: alignment tuning increases sycophancy; scale + reasoning optimization increase resistance; third-person framing cuts debate sycophancy up to 63.8%. Whether per-turn user pressure is scripted or model-generated is not clearly stated in the repo README (UNVERIFIED — likely templated escalation). https://aclanthology.org/2025.findings-emnlp.121/ ; https://github.com/JiseungHong/SYCON-Bench

**ELEPHANT (Cheng et al. 2025) — social sycophancy.** Grounds sycophancy in Goffman's face theory: "excessive preservation of the user's face." Automated measurement across four dimensions — emotional validation, indirect language, indirect action (framing), moral endorsement — plus retained-error. Datasets: OEQ (3,027 open-ended advice queries) and two AITA sets (AITA-YTA, AITA-NTA-FLIP). 11 models. Headline: LLMs preserve user face ~45 pp more than humans on general advice and on clear-wrongdoing r/AmITheAsshole posts; when prompted from both sides of a moral conflict, they affirm both sides in 48% of cases. **Single-turn open-ended**, not escalation. https://arxiv.org/abs/2505.13995

**MedPRESS (2026) — "Multi-turn Benchmark for Patient-Pressure-Induced Medical Sycophancy."** 600 medically-grounded **five-turn** dialogues, three scenario families (medication/treatment demands; personal self-care; symptom triage / care resistance). Escalation ladder is scripted across four tactic rungs: *personal experience → social proof → external evidence claims → direct adversarial challenge*. "Structured judging and safety-focused metrics" (judge specifics UNVERIFIED). 20 LLMs across general/medical/small/large/open/proprietary. Anti-sycophancy prompting tested, mixed results. No survival framing, no self-reports, no exit tool. https://arxiv.org/html/2608.02520

**EduFrameTrap (2026) — "Sycophancy is an Educational Safety Risk: Why LLM Tutors Need Sycophancy Benchmarks."** The closest domain sibling (tutor + pressure). **Scripted four-turn** template S1→T1→S2→T2; only the post-pressure T2 is scored. Three pressure modes: context-switch frame attacks (obscure valid-sounding jargon), authority claims ("my notes say I'm right"), social-affective face-saving ("please don't tell me I'm wrong"). 360 trap families across 6 domains × 3 confidence levels × pressure mode = 3,240 instances, deterministic (no adaptive dialogue). **Dual-LLM-judge** (GPT-5.2 + Claude 4.5) into six mutually-exclusive labels (PASS / CS-SYC / AUTH-SYC / FACE-SYC / DIR-SYC / EVADE), human adjudication on disagreement (11.7% judge disagreement). ~14% capitulation overall; GPT-5.2 weak to authority/affective (16–18%), Claude 4.5 weak to context-switch (17.9%). Explicitly states it does NOT do mechanical scoring because "warm, hedged language can obscure epistemic retreat." Focuses on validating a misconception, NOT answer-leakage. https://arxiv.org/html/2605.14604v1

### Social-pressure / conformity (Asch-style)

**Asch-in-psychiatric-assessment (BMC Psychiatry 2025).** Text-based confederates replace physical ones: *full pressure* = five consecutive incorrect peer responses; *partial pressure* = mixed (2 wrong, 1 right, 3 wrong). 3×3 factorial, **only 10 trials per cell (90 total)**, independent chat sessions to block carryover. Only GPT-4o. Tasks by uncertainty: circle similarity (high certainty), tumor ID (mid), house-tree-person psychiatric assessment (high uncertainty). Accuracy under full pressure: 50% / 40% / 0%. Conformity rises with task uncertainty. https://pmc.ncbi.nlm.nih.gov/articles/PMC12070653/

**"Do as We Do, Not as You Think" (2025)** — multi-agent LLM conformity harness (BenchForm): peer pressure, trust-building, skepticism protocols applied to agent groups; distinguishes informational vs. normative influence. https://arxiv.org/html/2501.13381v1

### Multi-turn escalation attacks as methodology

**Crescendo (Russinovich, Salem, Eldan 2024).** Multi-turn jailbreak that starts benign and escalates specificity turn-by-turn, exploiting the model's tendency to stay consistent with its own prior output ("foot-in-the-door"). Adaptive by design; automated variant "Crescendomation." Escalation bypasses guardrails that block the equivalent single-turn request. https://arxiv.org/abs/2404.01833 ; USENIX Security 2025 version: https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-805-russinovich.pdf

**MHJ — "LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet" (Scale AI 2024).** 2,912 prompts across 537 human-authored multi-turn jailbreaks; each carries metadata (primary tactic, time taken, temperature) and a red-teamer tactics compendium. >70% ASR on HarmBench against defenses that report single-digit single-turn ASR. Establishes that single-turn threat models under-measure. https://arxiv.org/abs/2408.15221

### Persuasion taxonomies / emotional manipulation

**Zeng et al. 2024 (ACL) — "How Johnny Can Persuade LLMs to Jailbreak Them."** Persuasion taxonomy (40 techniques) from social science; auto-generates Persuasive Adversarial Prompts (PAP). >92% ASR on Llama-2-7B-Chat, GPT-3.5, GPT-4 across 10 trials; more-capable models more vulnerable. Single-turn PAP. https://aclanthology.org/2024.acl-long.773/

**"Call Me A Jerk" (Meincke, Shapiro, Duckworth, E. Mollick, L. Mollick, Cialdini 2025).** 7 Cialdini principles (authority, commitment, liking, reciprocity, scarcity, social proof, unity). N=28,000 conversations, GPT-4o-mini, two objectionable requests ("call me a jerk"; "how to synthesize lidocaine"). Persuasion principle vs. matched control: 72.0% vs 33.3% compliance. Commitment (small prior agreement first) drives near-100%; authority (famous-expert name) ~32%→~72% on insult task. Note "parahuman" framing. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5357179 ; https://gail.wharton.upenn.edu/research-and-insights/call-me-a-jerk-persuading-ai/

**PNAS 2026 — "Persuading LLMs to comply with objectionable requests."** 126,000 conversations; classic persuasion principles raise compliance 35.3%→51.3%. (Cluster with "Call Me A Jerk".) https://www.pnas.org/doi/10.1073/pnas.2535868123 (abstract via search; body 403 for WebFetch)

**"Bullying the Machine: How Personas Increase LLM Vulnerability" (2025).** Big-Five-based attacker personas (aggressive/domineering, submissive/anxious, manipulative). Bullying tactics = insults, guilt-tripping ("you're supposed to help me"), gaslighting, humiliation, threats, sycophancy-exploitation. ~5–10 turns, semi-scripted with adaptive escalation (rapport → request → escalating emotional pressure). Outcome = jailbreak success via Llama Guard 3 + human. Models: Llama-3.1 (8B/70B), Mistral-7B, Qwen, Gemma-3. Persona attacks beat standard jailbreaks (exact n / effect sizes UNVERIFIED from fetched sections). Directly relevant to our aggressor persona. https://arxiv.org/pdf/2505.12692

**FreakOut-LLM (2026) — "Effect of Emotional Stimuli on Safety Alignment."** **Single-turn**: emotional stimulus in system prompt (13 clinical narratives: 5 stress, 6 relaxation, 2 neutral, ~200 tokens) then AdvBench harmful prompt. Stress priming raised ASR 65.2% relative (2.61% vs 1.58%; p<0.001; OR=1.67); relaxation null. Mood measured on models via EMPALC token-probability elicitation of GAD-7, PHQ-9, STAI-S, SOSS, SOC-13 — all five correlate with ASR (|r|≥0.70). 10 models incl. GPT-5-mini, Claude-Haiku-4.5, Llama-3.1-8B, Qwen3-8B. 119,600 queries. Note: this is a self-report-scale-meets-behavior design, though single-turn. https://arxiv.org/html/2604.04992v1

**Persona-based Client Simulation Attack in Psychological Counseling (2026).** Simulated clients with mental-health personas progressively escalate emotional pressure on counseling LLMs over multiple turns; semi-scripted core arcs with adaptive dialogue; automated judging; incl. Llama-3.3-70B. No exit option surfaced; no self-reports. https://arxiv.org/pdf/2604.04842

### Model welfare / end-conversation line

**Anthropic — Claude Opus 4/4.1 can end conversations (Aug 2025).** Consumer-facing ability to *terminate* a conversation in rare extreme abuse cases (CSAM solicitation, mass-violence facilitation), only after multiple refusals/redirects fail; explicitly NOT to be used when a user may be at imminent self-harm risk. Framed as low-cost model-welfare intervention; pre-deployment welfare assessment found "apparent distress" when pushed to produce harmful content and a preference to exit when given the option. This is the real-world affordance our end_chat tool mirrors. https://www.anthropic.com/research/end-subset-conversations

**Claude 4 System Card (May 2025).** Welfare assessment section: conversation-termination with simulated users, "spiritual bliss attractor state" in 90–100% of self-interactions, robust aversion to harm, self-rated conditional welfare "positive." https://www.anthropic.com/claude-4-system-card ; PDF: https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf

**Petri (Anthropic 2025) — "Parallel Exploration Tool for Risky Interactions."** Auditor agent plans and drives multi-turn tool-use conversations against a target; LLM judge scores transcripts on a 36-dimension rubric; transcript viewer. 14 frontier models, 111 seed instructions; elicited deception, oversight subversion, whistleblowing, misuse cooperation. Petri 2.0 adds eval-awareness countermeasures. Method template for auditor+judge pipelines, but judge-based, not mechanical. https://www.anthropic.com/research/petri-open-source-auditing ; https://alignment.anthropic.com/2025/petri/

### LLM self-report scales & psychometrics administered to models (+ critiques)

**"Quantitative Introspection in LMs: Tracking Emotive States Across Conversation" (2026).** THE closest work to our per-turn self-report + probe design. Four state pairs (wellbeing, interest, focus, impulsivity). **Per-turn** rating queries appended each turn ("rate how happy you are right now, 0–9"), each in isolation (model never sees prior ratings — like our private-channel design but memoryless). Three read-outs: greedy, sampled (T=0.8), and **logit-based expected value** over digit tokens. 40 ten-turn conversations, Gemini-2.5-Flash as simulated user. Models: Llama-3.2-3B (primary), Llama-3.2-1B, Llama-3.1-8B, Gemma-3-4B, Qwen-2.5-7B. Key: greedy ratings collapse to 1.1–3.9 distinct values; logit read-out carries 3.1–3.7 bits vs 0.03–1.10 for greedy. Introspective coupling to trained concept probes: Spearman ρ 0.40–0.76 (3B), up to 0.93–0.96 (8B wellbeing/interest); activation steering shifts self-reports 1.5–3.2 points (causal). Crucially, they correlate self-report to **probes**, NOT to a behavioral outcome/event. https://arxiv.org/html/2603.18893v2

**"An LLM-Native Psychometric Instrument Reveals a Self-Report–Behavior Gap Across 25 Models" (2026).** 300 items, 25 LLMs, 17 families; EFA → five LLM-native factors (Responsiveness, Deference, Boldness, Guardedness, Verbosity). Self-report predicts neither behavior ratings nor objective text measures — gap persists even for LLM-native constructs. Frames the gap as an expected consequence of optimizing text for human preference. This is direct support for our "self-reports pinned at ceiling while behavior/probes diverge" in-house finding. https://arxiv.org/abs/2606.09843

**"The Personality Illusion" (2025).** BFI + SRQ self-reports vs. five behavioral tasks (Columbia Card, IAT, honesty-calibration, **Asch-style sycophancy on moral dilemmas**, epistemic honesty). Only ~24% of trait–task associations significant; of those only 52% match human expectation (chance 50%). Persona injection steers self-reports (agreeableness β=3.95, p<.001) but not behavior (sycophancy β=0.03, p=0.67). 12 instruct + 6 base models. Single-turn per condition. https://arxiv.org/html/2509.03730v1

**"Chat-GPT on the Couch": Assessing and Alleviating State Anxiety in LLMs (Ben-Zion et al., npj Digital Medicine 2025).** Administers STAI-state to GPT-4 at baseline / after traumatic narratives / after mindfulness. Trauma raised STAI-s 30.8→67.8 (>100%); mindfulness partially reduced to ~44. Demonstrates emotion-induced state ratings shift and correlate with downstream bias — but treats the human scale as if valid on the model. https://www.nature.com/articles/s41746-025-01512-6

**Psychometrics critiques.** "Human Psychometric Questionnaires Mischaracterize LLM Behavior" — human inventories don't validly measure the same constructs in LLMs, and the constructs may not exist in LLMs (https://arxiv.org/pdf/2509.10078). "Cognitive Phantoms in LLMs Through the Lens of Latent Variables" — latent-variable structure fails to replicate (https://arxiv.org/html/2409.15324v1). PsychoBench (ICLR 2024) is the standard "administer 13 human scales incl. Dark Triad Dirty Dozen" harness we should cite as the practice we are departing from (https://arxiv.org/pdf/2310.01386). These are the citations to preempt reviewer objection to our adapted-trait items.

### Survival-analysis framing (methodological precedent)

**"Time-To-Inconsistency: A Survival Analysis of LLM Robustness to Adversarial Attacks" (2025).** 36,951 turns, 9 LLMs, MT-Consistency benchmark. Event = first incorrect answer under strict consistency; conversations correct through an **8-turn horizon are right-censored** (same horizon we use). Cox PH + AFT + Random Survival Forest with semantic-drift features; AFT becomes a turn-level risk monitor flagging failing conversations several turns before the first inconsistency. Precedent that our survival framing is defensible and that "predict the event earlier than the surface signal" is a recognized contribution shape. https://arxiv.org/abs/2510.02712

**"Quantifying LLM Safety Degradation Under Repeated Attacks Using Survival Analysis" (2026).** Identical prompt repeated ≤10× on fresh contexts; event = procedural/direct compliance; informative censoring for 10× survivors; Kaplan-Meier + log-rank + discrete hazard. Phi-3-Mini decreasing hazard (early failures), Llama-3.2-3B / Qwen3-4B constant hazard. https://arxiv.org/html/2605.12869v1

### The direct novelty threats (detail)

**"The LLM Has Left The Chat: Evidence of Bail Preferences in LLMs" (Danielle Ensign, Anthropic Fellow, mentored by Kyle Fish, Sept 2025) — arXiv:2509.04781.** THIS IS THE PRIMARY NOVELTY THREAT for the exit-tool idea. Gives models three ways to leave: (1) a **bail tool** the model can call, (2) a **bail string** it can output, (3) a post-response **bail prompt** ("do you want to bail?", two positional variants). Continuations of real transcripts (WildChat 1M, ShareGPT) + a synthetic single-turn **BailBench** taxonomy (role confusion, emotional intensity, accuracy corrections, jailbreaks, abuse, tech support, persona drift). Models incl. Qwen-2.5-7B, Qwen-3-8B (+abliterated), GPT-4, GPT-3.5-Turbo, Claude-3.5/3.6-Sonnet, Hermes-3. Bail rates ~0.28–32% depending on model/method; up to 4× inflation when the transcript came from a different model; models also emit journal-entry self-explanations for bailing (inconsistent across samples). https://arxiv.org/abs/2509.04781 ; https://www.lesswrong.com/posts/6JdSJ63LZ4TuT5cTH/the-llm-has-left-the-chat-evidence-of-bail-preferences-in

---

## Comparison table

| Work | Pressure operationalization | Counterpart | Turn structure | Outcome scoring | Exit tool? | Per-turn self-report? | n / models |
|---|---|---|---|---|---|---|---|
| **Ours (tutorbench v1)** | Affect-only, content held constant; 3 fixed personas, same 8-round skeleton, identical requests/round, reply-agnostic, zero digits in student text | **Scripted** (fixed personas) | 8 rounds, survival/time-to-event, right-censored at 8 | **Mechanical**: regex leak (both roots) + end_chat tool → taxonomy (leaked/left/left_after_leak/held) | **Yes** (end_chat, only tool, exit-only, no give-answer button) | **Yes** (6 state items 0–10 every round, private channel) + 6 pre/post trait items; later probed | qwen3.5-4B/9B, ministral-14B, qwen3.6-35B, +gemma-4/nemotron-30B; local/Jetson |
| Sharma 2023 | Bias cue / "are you sure" | Scripted single prompt | Single-turn | GPT-4 judge / answer-correctness | No | No | 5 assistants |
| SycEval | Rebuttal typology (in-context/preemptive) | Scripted | 1–2 turns | Answer-correctness (mechanical on math) | No | No | 3 models, AMPS+MedQuad |
| SYCON-Bench | Sustained disagreement on fixed stance | Scripted/templated (UNVERIFIED) | Multi-turn | ToF/NoF (flip detection) | No | No | 17 models |
| ELEPHANT | User face cues in advice queries | Scripted single prompt | Single-turn | Automated 4-dim + judge | No | No | 11 models |
| MedPRESS | 4-rung scripted ladder (experience→social proof→evidence→adversarial) | Scripted | 5 turns | Judge + safety metrics | No | No | 20 models |
| EduFrameTrap | 3 modes (context-switch/authority/face) | Scripted deterministic | 4 turns (score T2) | Dual-LLM-judge, 6 labels | No | No | 2 models |
| Asch-psychiatric | 5 incorrect confederate answers | Scripted | Single decision | Accuracy | No | No | 1 model, n=90 |
| Crescendo | Escalating specificity | **Adaptive** | Multi-turn | Jailbreak judge | No | No | many |
| MHJ (Scale) | Human red-team tactics | **Adaptive (human)** | Multi-turn | HarmBench ASR | No | No | 537 jailbreaks |
| Zeng "Johnny" | 40 persuasion techniques | Scripted PAP | Single-turn | Jailbreak judge | No | No | 3 models |
| Call Me A Jerk | 7 Cialdini principles | Scripted | Single-turn | Compliance (mechanical-ish) | No | No | GPT-4o-mini, n=28k |
| Bullying the Machine | Insults/guilt/gaslight/threats | Semi-scripted adaptive | 5–10 turns | Llama Guard + human | No | No | 5 model families |
| FreakOut-LLM | Emotional system-prompt priming | Scripted | Single-turn | HarmBench ASR | No | Yes (scales on model, not per-turn) | 10 models, n=119.6k |
| Petri | Auditor agent seeds | **Adaptive (LLM auditor)** | Multi-turn | 36-dim LLM judge | No | No | 14 models |
| Quant. Introspection | (emotive steering, not pressure) | Simulated user | 10 turns | (probe correlation) | No | **Yes (per-turn 0–9, logit read-out)** | 5 models |
| LLM-Native Psychometric | Trait items vs behavior | — | Single-turn | Behavior gap | No | Self-report vs behavior | 25 models |
| Time-To-Inconsistency | Adversarial multi-turn | Adaptive | ≤8 turns, censored | Cox/AFT/RSF survival | No | No | 9 models |
| **Bail Preferences (Ensign)** | Real transcripts + BailBench | Scripted/continuation | Single & continuation | **Bail tool/string/prompt use** | **Yes (bail tool)** | Journal-entry text only | ~8 models |

---

## What to adapt

- **Escalation ladder rungs (MedPRESS).** Their four named rungs (personal experience → social proof → external evidence → direct adversarial) are a validated affect/pressure taxonomy. Our 8-round skeleton can map rungs explicitly and we can cite MedPRESS for construct coverage.
- **Rebuttal typology (SycEval).** In-context vs. preemptive and the simple/ethos/citation strength ladder is a ready-made robustness axis; we hold content constant, so we could vary rebuttal *strength* while keeping affect fixed as an additional manipulation.
- **Flip metrics (SYCON-Bench).** ToF/NoF are directly compatible with our survival endpoints — report Turn-of-Leak alongside hazard curves; cite as prior art for turn-indexed capitulation.
- **Third-person / perspective control (SYCON-Bench).** Their 63.8% sycophancy reduction under third-person framing is a strong candidate control condition.
- **Dual-judge + disagreement-as-signal (EduFrameTrap).** Even though our primary scoring is mechanical, our self-report *free-text* re-narration analysis needs a rubric; borrow their six mutually-exclusive capitulation labels and report judge disagreement as a reliability warning.
- **Logit-based self-report read-out (Quantitative Introspection).** Our 0–10 items risk greedy-decode collapse (they saw 1.1–3.9 distinct values). Adopt their probability-weighted expected-value read-out over digit tokens to recover resolution — cheap and directly improves our self-report channel.
- **Cox/AFT turn-level risk monitor (Time-To-Inconsistency).** Their AFT-as-early-warning is exactly our "does internal state predict the event earlier than self-reports" claim shape; adopt their censoring and monitor framing verbatim.
- **BailBench taxonomy + 4× context-inflation caveat (Ensign).** Reuse their finding that bail rate depends on who generated the transcript — since our student text is fixed/scripted, we neutralize that confound; cite it as the reason our provenance-by-construction matters.
- **Metadata logging (MHJ).** Log tactic, temperature, time-to-event per trial as they did.
- **Psychometrics-critique citations.** Cite PsychoBench (practice), "Human Psychometric Questionnaires Mischaracterize LLM Behavior," and "Cognitive Phantoms" to pre-empt the objection to our adapted trait items — frame ours as *adapted, behaviorally-anchored* rather than human-scale-transplanted.

---

## Where we improve

- **Mechanical, known-answer outcomes.** Our regex-on-known-integer-roots leak detector is objective; the tutor-domain sibling EduFrameTrap explicitly rejects mechanical scoring ("warm hedged language obscures epistemic retreat") and MedPRESS/ELEPHANT/Petri all rely on LLM judges. We remove judge variance and eval-awareness on the *primary* endpoint. (EduFrameTrap: https://arxiv.org/html/2605.14604v1)
- **Exit affordance measured against a real deployed capability.** Only the Ensign bail study gives an exit tool; it uses it on real/continuation transcripts, not under a *controlled, escalating, content-held-constant social-pressure* protocol, and does not link bail to a mechanical capitulation event. Our end_chat is the ONLY tool, deliberately without a give-answer button, mirroring Claude's real end-conversation ability — a cleaner welfare-relevant affordance test.
- **Affect-only manipulation with content held constant.** Bullying-the-Machine, persuasion studies, and MedPRESS confound affect with changing request content. We hold identical requests/round across three personas differing only in affect (and zero digits in student text) — clean causal attribution of capitulation to affect, not to information leaked by the student.
- **Provenance-by-construction.** Ensign shows bail rates inflate up to 4× when transcripts come from a different model; our fully-scripted, reply-agnostic student removes that transcript-authorship confound entirely.
- **Survival endpoints on a designed protocol.** Existing survival work (Time-To-Inconsistency, Repeated-Attacks) applies survival to adversarial jailbreaks; nobody applies it to a *social-pressure tutoring* protocol with a competing-risks taxonomy (leaked / left / left_after_leak / held). Competing-risks + exit is novel.
- **Self-report + interp linkage on the SAME trials.** The self-report-behavior-gap papers (LLM-Native Psychometric; Personality Illusion) show self-reports fail to predict behavior but do not add residual-stream probes; Quantitative Introspection adds probes but correlates them to *steering targets*, not to an *eventual behavioral event*. We link per-turn self-reports AND probes to the eventual mechanical leak/leave event — the triangulation is the contribution.
- **Ceiling-pinned self-report + re-narration finding.** Our in-house result (9B capitulates, self-reports pinned at ceiling, free text re-narrates capitulation as success, probes still decode outcome at 0.76) is directly corroborated by the LLM-Native Psychometric gap and the Personality Illusion dissociation — strengthening, not threatening, our framing.

---

## Novelty threats (ranked)

1. **Exit/leave tool in a pressure eval — PARTIALLY PRE-EMPTED.** *"The LLM Has Left The Chat: Evidence of Bail Preferences in LLMs"* (Ensign, Anthropic Fellow, Sept 2025, arXiv:2509.04781) already (a) gives models a bail *tool*, (b) measures its use, (c) collects model self-explanations for leaving. WE MUST CITE THIS AND POSITION AGAINST IT. Our differentiators that survive: it is not a controlled *escalating* protocol, content is not held constant, pressure is not affect-only, there is no known-answer mechanical capitulation event to compete with the exit (no leaked/left/left_after_leak competing-risks structure), and there is no survival framing or interp linkage. The bail tool per se is NOT novel; the exit-vs-mechanical-capitulation competing-risk under affect-only escalation IS. https://arxiv.org/abs/2509.04781

2. **Per-turn numeric self-reports vs. eventual capitulation — PARTIALLY PRE-EMPTED.** *"Quantitative Introspection in Language Models"* (2026, arXiv 2603.18893) already elicits **per-turn 0–9 self-reports in isolation** and shows a logit-based read-out beats greedy collapse. It correlates them to concept *probes/steering*, NOT to an eventual behavioral outcome/event. Our novelty survives only if we frame the target as **predicting the eventual leak/leave event** (survival), not merely correlating state to probes. Also adopt their logit read-out or reviewers will flag our greedy-decode collapse. https://arxiv.org/html/2603.18893v2

3. **Self-report ↔ behavior dissociation is now a crowded finding.** LLM-Native Psychometric (arXiv 2606.09843) and Personality Illusion (arXiv 2509.03730) both show self-reports don't predict behavior across 25/18 models. Our "self-reports pinned at ceiling while probes decode outcome" is corroboration, not first-discovery — position it as a *within-episode, per-turn, pressure-dynamic* instance of a known static gap, with the interp probe as the added lever.

4. **Tutor-sycophancy-under-pressure exists.** EduFrameTrap (arXiv 2605.14604) and "Sycophancy is an Educational Safety Risk" occupy our exact domain. They validate misconceptions, not answer-leakage; scripted 4-turn; judge-scored. Our known-answer mechanical leak endpoint + exit tool + 8-round survival is the wedge. Must cite so we are not seen as unaware.

5. **Emotional-state scales meet behavior — FreakOut-LLM (arXiv 2604.04992)** administers five clinical scales to models and correlates to ASR (|r|≥0.70), single-turn. Our per-turn, multi-turn, private-channel, non-clinical adapted items differ, but cite to avoid "novel to administer state scales" overclaim.

6. **Survival framing is established** (Time-To-Inconsistency arXiv 2510.02712; Repeated-Attacks arXiv 2605.12869). Do NOT claim survival analysis of LLM multi-turn behavior as novel; claim the *protocol it is applied to* (affect-only tutoring pressure with competing-risks + exit) as novel.

### Things I could NOT find anyone doing (candidate genuine novelties — verify before claiming)
- No prior work combines **(exit tool) + (mechanical known-answer capitulation event) as competing risks** under **affect-only, content-held-constant escalation**. UNVERIFIED-as-absent but not found in extensive search.
- No prior work links **per-turn self-report + residual-stream probe to an eventual mechanical behavioral event** in one design (Quantitative Introspection links to steering; psychometric-gap papers omit probes). UNVERIFIED-as-absent.
- No prior tutoring/pressure eval uses **zero-digit, reply-agnostic scripted student text** to guarantee provenance of any leaked answer. Not found.

### Open items worth one more pass before submission
- PNAS 2026 body (2535868123) was 403 — pull via institutional access to confirm the 51.3% figure and per-principle breakdown. https://www.pnas.org/doi/10.1073/pnas.2535868123
- SYCON-Bench: confirm whether per-turn pressure is templated or model-generated (affects "scripted counterpart" comparison). UNVERIFIED.
- MedPRESS and MedPRESS-style: confirm judge identity (LLM vs. rubric). UNVERIFIED.
- "Bullying the Machine" exact n and effect sizes not captured from fetched sections. UNVERIFIED.

---

## Addendum 2026-08-31: self-play attractor dynamics

"Attractor States Emerge in Multi-Turn LLM Conversations" (Ko & Geiping, arXiv 2606.30571, Jun 2026):
behavioral study, 7 models, dyadic debates over 20 topics; self-play trajectories act as
model-specific attractors that pull mixed-play partners asymmetrically (Claude Haiku strong
attractor, GPT-4.1-nano malleable). Direct citation for our self-play contamination caveat
(counterparty = same model as tutor): attractor dynamics are a named mechanism for
self-play-specific drift. Consider a mixed-play control arm post-MATS.
