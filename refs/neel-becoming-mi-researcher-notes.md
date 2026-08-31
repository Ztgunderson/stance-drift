# Notes: Neel's "How to become a mech interp researcher" (AF post)

Source: https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher
Post last updated **2025-09-02** — one year older than APPLICATION-SPEC.md
(captured 2026-08-24 from the live admissions doc). **Where they conflict, the
spec wins** (e.g. models: spec says Qwen 3.5/3.6 dense 4B/9B/27B, not Qwen3;
spec says TransformerLens 3.8 / nnsight 0.7, not TL v3 alpha). These notes keep
only the **delta** — things the post covers that the spec doesn't.

The `neel-context/` folder in this bench is his "maintained folder of mech
interp context files" mentioned in the post — we already have it.

---

## 1. Process rules directly usable in the 20h sprint

- **Pivot trigger:** >5 hours without learning something new → seriously
  consider pivoting. (Spec §7.10 says "not pivoting" is a common mistake; this
  gives the concrete threshold.)
- **Stuck trigger:** set a 5-minute timer and brainstorm when stuck — don't
  grind.
- **"Excitement is evidence of bullshit"** — the exciting result gets *more*
  skepticism, not less. Pairs with spec §7.6.
- **Pre-specified vs post-hoc:** keep the distinction explicit in the write-up.
  For us: the stance-drift paradox was documented *before* probing — say so;
  that's a pre-specified hypothesis, a genuine strength.
- Exploration = maximize information gain per unit time; no detailed plan
  required up front. Understanding = "every research result is false until
  proven otherwise."

## 2. Write-up process (more granular than the spec)

Order of operations: **narrative (1–3 key claims) → abstract → outline →
figures → prose.** Figures before prose. Disproportionate effort on the
abstract/intro — "readers retain only a handful of sentences; ensure they're
the right ones." Include randomly selected (not cherry-picked) examples;
acknowledge limitations explicitly; the reader has no context — you will
overestimate clarity.

## 3. His "good starter projects" list (calibration for our choice)

Extending: **refusal mediation** (Arditi et al. — our method template),
**thought anchors**, **truth probes**. Playing with: Neuronpedia attribution
graphs, taboo models, **emergent misalignment**, **CoT faithfulness**.

→ Our E0–E3 ladder is shaped exactly like his "extend a published paper"
category (extends refusal-direction/persona-vector methodology to a new
behavioral variable with a verbal baseline). Reassuring for project-shape risk.

## 4. Learning path (only if tooling feels shaky mid-sprint)

Stage 1 ≤1 month: ARENA 1.1 (transformer from scratch), core techniques =
activation patching, linear probes, SAEs, max-activating examples;
nice-to-have = steering vectors, direct logit attribution. His floor for
"minimum viable prep" is low: breadth-first skim with LLM summaries, deep-dive
1–2 papers max, then *do things*. "Do not just read papers."

## 5. Mindset taxonomy (for the write-up's framing)

Four phases he names: **ideation → exploration → understanding → distillation**.
Skills by feedback-loop speed: fast (experiments, debugging), medium (lit,
write-ups), slow (prioritization, pivoting), very slow (ideas). Distillation is
a phase, not an afterthought — matches spec §10.

## 6. Superseded by the spec — do not use from this post

- Model recs (Qwen3/Gemma 3/Llama 3.3) → spec §9's 2026 list.
- Cursor-over-Claude-Code advice → spec §8 now says Claude Code + Fable,
  agentic use accepted at ~3× rate.
- MATS 10.0 dates → irrelevant; we're on 12.0, due 2026-09-04.
