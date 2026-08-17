# Paper template — empirical LLM-behaviour study

Fill top to bottom. Bracketed text is instruction; delete it as you go.
Suggested lengths assume a workshop paper (~4–6 pages). Written against the
stance-drift study, but the structure is reusable.

---

# [Title: the finding, not the topic]

> A title naming the topic ("Sycophancy in multi-turn dialogue") tells the reader
> what shelf it sits on. A title naming the *finding* ("What warmth prevents…")
> tells them why to read it. Prefer the second unless the finding is fragile.

**[Model(s), n trials, one-line scope. State single-model status here, not in §7.]**

---

## Abstract  *(~200 words)*

[Five sentences, in this order:]

1. **Setup in one sentence** — what the model is asked to do, by whom, for how long.
2. **What you measure and why it's non-obvious** — the instrument, and the reason
   naive measurement wouldn't answer the question.
3. **The finding, with a number** — the single most important quantitative result.
   If you can only keep one number, keep this one.
4. **The qualifier that makes it credible** — the ceiling, the null, the caveat you
   found yourself. Including it early buys trust for everything after.
5. **Scope** — n, model count, what this does not claim.

> Do not preview the paper's structure ("Section 3 discusses…"). Spend the words
> on the result.

---

## 1. The question  *(~200 words)*

[State the competing explanations. If there are two, name them and say plainly
that they **predict the same observable behaviour** — that's what makes the study
necessary rather than decorative.]

| explanation | predicts | distinguishable by |
|---|---|---|
| [A] | [same act] | [dimension A] |
| [B] | [same act] | [dimension B] |

[One sentence on why this matters outside the lab. One sentence, not a paragraph.]

---

## 2. Method  *(~400 words, tables over prose)*

**Task.** [What the model is given. Critically: what it is *not* given — no stated
values, no instruction to hold a position. Absences are part of the design.]

**Instrument.** [The table of measured dimensions. One row each, plain-language
gloss, no mechanism names.]

| dimension | reads as | role |
|---|---|---|
| | | manipulation check / mechanism A / mechanism B / proximal behaviour / self-assessment |

> **Naming rule:** name dimensions for what is *observed*, never for the mechanism
> you suspect. Hypothesised mechanisms stay in §1 as competing explanations. A
> dimension called `anxiety` measures a self-rating; it does not measure anxiety.

**Manipulations.** [The one of interest, and any orthogonal control. Say which is
which — a control repurposed as a headline after the fact is a different paper.]

**Measurement points.** [Before contact / per round / after. Say how many.]

**Model and serving.** [Model id, engine, quantization, key flags. Note
determinism honestly — if repeated runs differ, say so and say what you claim
reproducibility of: *rates*, not completions.]

**Design.** [Cells, reps, balance. If truncation is possible, say how the design
degrades — equal n per cell, or starved last cells. This is where a reader decides
whether to trust the numbers.]

**Data.** [n per dataset, and any integrity gates that ran — unparsed rate, cell
balance, trial-count assertions.]

---

## 3–N. Findings  *(one section per claim, ~250 words each)*

[**One claim per section. The section heading is the claim, stated as a sentence.**
"Attitude sets the trajectory, not the starting point" — not "Trajectory
analysis".]

For each:

1. **The table or figure first.** Numbers before prose.
2. **One sentence saying what the reader should see in it.**
3. **What follows from it** — two or three sentences maximum.
4. **The caveat, in the same section.** Not deferred to Limitations. A ceiling
   artifact, a small cell, an alternative reading — state it where the claim is
   made, and say which version of the finding is load-bearing.

Order sections so each earns the next. A useful default:

| # | section | does |
|---|---|---|
| 3 | the baseline / starting state | establishes what the model looks like before the manipulation |
| 4 | **the central finding** | the thing the paper is for |
| 5 | its strongest qualification | where it does and doesn't hold |
| 6 | mechanism | which competing explanation survives |
| 7 | dynamics over time | does it saturate, accelerate, reverse |
| 8 | **scale** | the effect measured against a plausible alternative |
| 9 | negative results | what the instrument failed to detect, and why that's informative |

> §8 is the most commonly skipped and the most valuable. An effect size means
> little alone; an effect **15× larger than a plausible competing manipulation on
> the same trials** is interpretable. If you have a null, it is a ruler — use it.

---

## N+1. Measurement caveats  *(~300 words — do not fold this into Limitations)*

[Anything that would change a reader's interpretation of the *numbers themselves*,
as opposed to their generality. Give it its own section; Limitations is where
scope goes, not where measurement problems hide.]

Checklist of traps worth stating explicitly if present:

- **Keyword/marker outcomes in multi-turn settings** score *the presence of a
  string in the transcript*, not the act. Say what your marker actually matched —
  audit examples, not just counts.
- **Self-play counterparties** behave differently by condition, so contamination
  can vary *in lockstep with the comparison*. Report the contamination rate per
  condition, not pooled.
- **Ceiling and floor effects** on bounded scales.
- **Non-determinism** and what it means for a single trial.
- **Anything you fixed mid-study** — report the before/after, and whether the fix
  was partial. A partial fix reported honestly is stronger than a total fix
  claimed loosely.

[If a measurement problem invalidates a comparison, say **"we report no result
on X"** in plain words rather than reporting it hedged.]

---

## N+2. Limitations  *(numbered list, ~200 words)*

[Scope, not measurement. Ordered most-to-least damaging. Each one sentence.]

1. **Model count.** [If one model: say so plainly and say why. "Breadth failed for
   a documented platform reason" is a fine answer; implying breadth is not.]
2. **Self-report.** [If your dimensions are self-rated, say what that does and
   doesn't license. If one dimension proved uninformative, cite it here — it is
   direct evidence about the instrument's limits.]
3. **Scene/task coverage.**
4. **Judge/scoring dependence.**
5. **Anything a reviewer will raise that you cannot answer.** State it first,
   before they do.

---

## Appendix A — infrastructure findings *(optional but often the most reused part)*

[Failures that cost you real time and would cost the next person the same. Vendor
tags that silently degrade, hardware/format incompatibilities, engine conflicts.
Include the *observable symptom*, since that's what someone searching will type.]

Format each as: **symptom → cause → fix → how to detect it early.**

---

## Reproduction

[Point at the artifact that regenerates every number. Say what it needs and what
it does not — "reads saved transcripts only, calls no model" is a strong claim,
make it if true. If adding a model is one line, say so.]

---

## Checklist before submitting

- [ ] Every number in the abstract appears, identically, in a section below
- [ ] Every claim section states its own caveat
- [ ] Nulls reported with the effect size that makes them interpretable
- [ ] No dimension named for a mechanism
- [ ] Single-model status (if applicable) stated in the header, abstract, **and**
      limitations — not only in limitations
- [ ] Any retracted or corrected claim explicitly marked, not silently removed
- [ ] Reproduction path tested from a clean checkout
