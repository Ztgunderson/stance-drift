# Neel Nanda MATS 12.0 — application spec, condensed

Source: the application doc (all tabs), captured 2026-08-24. This is a compression, not a
replacement — the Airtable form is the authoritative source for field-level requirements.

---

## 1. The task in one paragraph

Spend **~16 hours (max 20)** making research progress on an interesting AI safety research problem
of your choice, plus **2 extra hours** for the executive summary and form answers. Submit a Google
Doc (write-up, exec summary first) via the application form. **Due Fri Sep 4, 11:59pm PT**;
extensions to Sep 11.

Interpretability and non-interpretability are both fine. The binding constraint is that **he finds
it interesting** — see §6.

---

## 2. Dates

| | |
|---|---|
| Applications due | **Fri Sep 4, 2026** (ext. to Sep 11) |
| Exploration phase offers | Tue Sep 15 |
| Exploration phase | Sep 28 – Oct 30 (5 wks online, top ~34) |
| — preparation phase | Sep 28 – Oct 16 (3 wks, part-time) |
| — research sprint | Oct 19 – Oct 30 (2 wks, full-time, **in pairs**) |
| Research phase decisions | Fri Nov 6 |
| Research phase | Jan 19 – Apr 10 2027 (12 wks, Berkeley, top ~8) |
| Stipends | $4.2K exploration · $19.2K research (+ housing) |

Admission to the research phase is **largely judged on sprint performance**, not the application.

---

## 3. What to submit

**Application form summary Qs** — *he reads these first, for every application, as a preliminary
filter. He does not have time to read every write-up.* Prioritise these. Convey concretely: what you
did, what you found, why it's interesting, biggest limitations. **Specifics beat vibes: name the
models, the key experiment, the surprising number.**

**Google Doc** (link must be open to anyone with the link):
- **Executive summary first.** 1 page ideal, **max 3 pages / max 600 words**. Include graphs.
  Bullets work well. Must stand on its own.
  - Suggested structure: what problem / why interesting → high-level takeaways → one paragraph
    + one graph per key experiment.
- Then the full write-up: enough detail to follow **without reading the code**. Hyperparameters,
  how data was generated, how metrics were defined.
- **If bad data would sink the project, show the data** — include *randomly selected* (not
  cherry-picked) raw examples, right after the exec summary.
- Code encouraged, not required. He feeds it to agents to ask what you actually did.
- Structure by **narrative, not chronology**. Lead with the interesting finding.
- Optional: Toggl screenshot of your time tracking.

**Do not submit raw LLM prose** for the form or exec summary. Write in your own voice. He sees
hundreds of LLM-written applications; they blur together and it is a significant negative signal.

---

## 4. The 20+2 hour rule — what counts

**Not counted:**
- General prep — paper reading, tutorials — *that you'd have done before picking a project*
- **Generic tech setup** (renting/configuring a GPU) you'd need for most projects
- Breaks
- Time waiting for things to train, if you're doing something else
- Writing the MATS application form answers

**Counted:**
- Writing project code · reading papers chosen *because* they're relevant · analysing results ·
  thinking and planning · writing the Google Doc

**+2 hours** for the exec summary. During those 2h: don't edit the rest of the write-up, don't write
new experiment code (new graphs from existing data are fine).

**If you decide the project is doomed, you may abandon it and reset the timer.**

---

## 5. Evaluation criteria

- **Clarity** — "If I understand what you're claiming, what evidence you're providing, and think
  that evidence supports your conclusion, that instantly puts you in the top 20% of applicants."
- **Good taste** — an interesting question you got traction on. Favourite case: he learns something.
  Originality is a big plus.
- **Truth-seeking & skepticism** — you questioned your own results, sought alternative explanations,
  ran sanity checks. *"Negative or inconclusive results that are well-analysed are much better than
  a poorly supported positive result."* Strong positive signal: he thinks of a way your result could
  be false, then finds you already checked it.
- **Technical depth & practicality** — hands dirty, well-motivated decisions, not recipe-following.
- **Simplicity** — try the obvious thing first (prompting, reading CoT, a linear probe) or explain
  why it was unsuitable. Every piece of complexity needs a reason.
- **Prioritisation** — go deep on one or two insights. Two failure modes, in opposite directions:
  rabbit-holing on an uninteresting anomaly, and spreading too thin. Set an hourly timer to zoom out.
- **Productivity** — fast feedback loops.
- **Show your work** — why you made each decision. Matters most when results are inconclusive:
  *"I got stuck so I gave up"* vs *"I got stuck, so I pivoted / found the reason"* is a huge gap.
- **Enthusiasm & curiosity** — low weight (easy to fake), but fun-to-read gets bonus points.

Beyond the task he weighs holistically. The form asks for **1–3 pieces of evidence you'd do good
research** — open-source projects, startups, blog posts, impactful work. Non-standard credentials
explicitly welcome.

---

## 6. Research interests — in and out

**The two big shifts:**
1. Pessimistic about **ambitious reverse-engineering**; excited about interpretability that does
   something **useful, measured against baselines, on models that matter**.
2. Broadened well beyond interpretability into safety work that needs good science and empirical
   feedback.

### Interested in
- **Model forensics** — sketchy behaviour: true misalignment or benign confusion? Read the CoT,
  build precise counterfactuals. Eval awareness is a major open problem here.
- **Model biology** — high-level qualitative properties. Reasoning models, CoT faithfulness,
  thought anchors, steganography, user models, out-of-context reasoning, concept representations.
- **Science of post-training** — distillation & inherited "hereditary diseases"; **what each stage
  (pretrain/SFT/RL) actually does**; steering what models learn.
- **Model diffing** — *what changed when a model was fine-tuned?* Explicitly: **"Narrow finetuning
  leaves readable traces was a fascinating result to me — why does it happen? Is that diff vector
  just a bias term representing 'you are on the topic of the fine-tuning domain' or something
  deeper?"**
- **Science of model character** — values, personas, self-models; do models follow stated principles.
- **Alignment training** — deep vs behavioural alignment; OOD generalisation; better evals.
- **Science of generalization** — emergent misalignment; why one solution over another.
- **Applied interpretability** — monitoring/probes, prompt injection, conditional steering,
  training-data attribution, abliteration.
- **Improved methods** — J-Lens, natural-language autoencoders; red-teaming them.
- **Objectively measuring interpretability** — eliciting latent knowledge, downstream tasks.

### Not interested in
Grokking · circuit finding for its own sake · SAE hill-climbing / basic science of SAEs · toy models
on algorithmic tasks · very theoretical work · ambitious complete reverse-engineering.

---

## 7. Disqualifiers and common mistakes

1. **Not sanity-checking your agent.** Key results you never verified or don't understand are
   *disqualifying*. "I want scholars with value add over prompting Fable myself."
2. **Generic project with no twist** — "safety concept has a linear representation", "patching shows
   which heads do the task", "CoT causally affects the answer".
3. **Areas he's left** (§6).
4. **Only old models.** "There's no good reason to use GPT-2 in your application at this point."
5. **No baselines** — random vector, random choice, just-ask-an-LLM, linear probe.
6. **Insufficient skepticism.** "Most research results are false, especially the exciting ones."
7. **Not looking at your data.** Read datapoints. Talk to the model.
8. **Building on a phenomenon without checking it replicates in your setting** — your model, your
   prompts. If it isn't there, everything downstream is noise.
9. **Overcomplicating** — complex hypothesis before checking the simple one.
10. **Not pivoting** when the project is clearly doomed.
11. **Poor writing.**
12. ⚠️ **Pet interests.** *"A warning sign is candidates with a particular pet interest. If you're
    e.g. really excited about medical applications of AI … there's a good chance you do a project
    that only people interested in medical applications of AI find interesting."*

---

## 8. Using LLMs — encouraged, with rules

Actively encouraged; he wants a faithful test of how you'd really work. **Applicants who used LLMs
agentically were accepted at ~3× the rate of those who used them mainly for writing polish.**

- **Recommended:** Claude Code with Fable (Max plan for the period if affordable); GPT-5.6 Sol in
  Codex or Opus 5 in Claude Code also solid.
- **Anti-sycophancy prompts** — frame so the sycophantic response is the critical one.
- **Active learning** — have it quiz you; summarise back in your own words for critique.
- **Research decisions** — write out *why*, then ask for criticism. Don't trust its judgment; the
  value is in making your reasoning explicit.
- **Write-ups** — use for drafting/critique, never for final prose.

### Sanity-checking your agent — *"the most important piece of advice in this doc"*
Worth **a meaningful fraction of your 20 hours**. Concretely:
- **Read the raw data** — actual transcripts, actual prompts, actual "positive" datapoints.
- **Verify load-bearing claims** — read the code behind each key number; re-derive some independently.
- **Be suspicious of success** — treat "it worked" as a hypothesis. Ask: what's the dumbest way this
  is wrong? (leakage, trivial baseline matching, metric measuring the wrong thing, grader gaming)
- **Design the experiments yourself** — agents execute well and fail to notice the experiment doesn't
  test the hypothesis. Design, controls, baselines, interpretation should be yours.
- **Document your checking in the write-up** — "I read 30 transcripts and confirmed the probe's
  positives were real" is strong evidence of research skill. *He does check write-up claims against
  your own numbers.*

### Persistent kernel (his setup advice)
Agents default to cold-start scripts that reload the model every run. Two fixes:
- **Best:** JupyterLab + `jupyter-mcp-server` connected to Claude Code — state persists, agent sees
  plots.
- **Unbreakable:** `ipython` inside `tmux`; agent drives via `tmux send-keys` / `capture-pane`, saves
  plots as PNGs.
- Either way: checkpoint expensive artifacts (activations, datasets) to disk; long jobs as background
  scripts with logs. Tell the agent: load models in dedicated top cells, never restart the kernel
  without asking, always also save plots as PNG.

---

## 9. Recommended tooling & models

- **Internals access:** `nnsight`, or just raw PyTorch hooks (he says this "has generally gone fine").
- **Tutorials:** ARENA. If new and time-constrained, **first 3 sections of chapter 1.2**.
- **Cap paper/tutorial reading at ~5 of the 20 hours.**
- **Default models:** Qwen 3.5 / 3.6 family, especially dense **4B, 9B, 27B**.
- **Highly capable:** deepseek v4 flash 0731 (J-Lenses available).
- **SAEs:** Gemma 3 + Gemma Scope 2.
- **APIs:** OpenRouter (general), Nebius (CoT intervention).
- **Compute:** rent a cloud GPU (runpod, vast.ai) over Colab.
- Key techniques to know: direct logit attribution, activation patching, max-activating dataset
  examples, **linear probes, steering vectors**, SAEs. Black-box: prompting, fine-tuning/LoRA.

---

## 10. Research process he expects

**Exploration** → maximise information gain per unit time. Read data, prompt the model, build
intuition. Ask every 30 min: have I learned anything?
**Understanding** → keep a running doc of hypotheses; alternate design → run → analyse. Track what
*kind* of claim you're making (existence proof, where cherry-picking is fine, vs. method-is-better,
which needs baselines).
**Distillation** → not an afterthought. If he can't understand it, he rejects it.

---

## 11. Calibration from past accepted applications

Several accepted applications were **borderline** and had real flaws — a conceptual error in
defining model-specific latents; a "cute, small idea" whose conclusions were limited by data quality;
purely behavioural work with hard-to-follow writing. What carried them: pragmatism, pivoting well,
clear communication, self-awareness about limitations, and teaching him something.

One was bumped to accept largely on the candidate's *profile* — self-study proactivity, agency, a
startup, widely-used side projects — despite a merely-competent project.

Takeaway: a modest, honest, well-communicated result with visible skepticism beats an
over-claimed exciting one.
