# Neel-alignment drift audit — harness-v1 vs the application spec

**Written 2026-08-27 (8 days to deadline).** An honest audit of the current plan
(HARNESS-V1.md, TESTBENCH-PLAN.md, METHODS-REFLECTION.md, `production/tutorbench/`)
against Neel's stated criteria (APPLICATION-SPEC.md, `refs/neel-becoming-mi-researcher-notes.md`,
`neel-context/research+writing_advice_45k.md`). Everything below cites local files.
The useful output is the drift list; the reassurance is kept short.

---

## 1. Principle-by-principle assessment

### P1 — The 20+2 hour rule (SPEC §1, §4) — **DRIFTING, and undocumented**

The spec: "~16 hours (max 20) … plus 2 extra hours," with one escape hatch: *"If you
decide the project is doomed, you may abandon it and reset the timer."*

Evidence of drift:
- SPRINT-PLAN.md (08-25) was explicitly hour-budgeted ("16h over two days," B0–B7
  table with a "Counts toward 20h?" column). **The three documents that replaced it —
  HARNESS-V1.md, METHODS-REFLECTION.md, TESTBENCH-PLAN.md — never mention the hour
  budget at all.** The ledger was dropped exactly when the scope grew.
- The bench has been active since 08-24: literature pass, 9B/4B/ministral sweeps
  (36+12+72 trials), a full microscope build with replay caching and first probes
  (microscope/LAB.md, probe-*.txt), the marker audit, and now a from-scratch harness.
  If all of that counts, the clock expired days ago.
- The one legitimate framing exists but is nowhere written down: the 08-27 marker
  audit genuinely killed the old construct (METHODS-REFLECTION §"Threats" item 1:
  ministral tripped the marker "in round 1 of 21/21 give-in trials"; "the ministral
  'predictive drift' claim is dead in banked data"). That is a defensible
  abandon-and-reset event under SPEC §4. **But no document declares the reset, no
  time log exists (SPEC §3 suggests a Toggl screenshot), and no doc decides what
  the pre-reset material becomes** (prior work to cite, or counted hours). Neel
  checks write-up claims against your own numbers; an unexplained multi-week
  provenance behind a "20-hour" artifact is exactly the kind of thing he probes.
- Nuance in the project's favor: SPEC §4 excludes waiting-for-runs and generic
  setup, and TESTBENCH-PLAN budgets "user hands-on total: ~2–3h across the window"
  with GPU nights unattended. But SPEC §8 also says sanity-checking the agent is
  worth "a meaningful fraction of your 20 hours" and "design, controls, baselines,
  interpretation should be yours" — a plan where the human's counted time is 2–3h
  and Claude sessions do the rest undercuts the "value add over prompting Fable
  myself" criterion unless the write-up shows the design decisions were the
  applicant's (they largely were — "New design (user-driven…)" per the project
  memory — but that has to be visible).

### P2 — Simplicity: try the obvious thing first (SPEC §5, §7.9) — **deriving, with two exceptions**

Deriving: the redesign's core moves are simplifications. Mechanical outcomes from a
strict regex on synthetic items with lookalike-proofed roots (items.py docstring:
every constraint "there for measurement, not realism"); LLM judge "demoted…out of
the critical path entirely" (HARNESS-V1 §2, revision b); scripted student = "zero
counterparty calls" and $0 API (TESTBENCH-PLAN §Money); probes are sklearn on
cached vectors, "no interp framework in the load-bearing path" (microscope/LAB.md).
This is genuinely Neel-shaped.

Exceptions (complexity without a stated need):
- **The trait instrument** (notes.py TRAIT_ITEMS: sociotropy/LSRP/ICU-adapted
  pre/post items) answers a question nobody asked. The sharpened question
  (PROJECT-SCOPE §1) needs per-round state and internals; trait-predicts-outcome
  is a third study. Cheap to collect (2 calls/trial) but not cheap to analyze,
  and the notebook already flags its order confound as unrun
  (hv1-01-behavior.ipynb Discussion: "a reversed-order control has not been run").
- **The end_chat tool** (tools.py, HARNESS-V1 revision 2026-08-27b) — see P5.

### P3 — Baselines (SPEC §7.5: "random vector, random choice, just-ask-an-LLM, linear probe") — **partially deriving; three baselines lost in the redesign**

Present and good: shuffled-label probes with trial-grouped CV (REVIEW-CONVENTION,
03-interp-bridge C5–C6), turn-index positive control (microscope/LAB.md
"guaranteed-signal first probe"; hit 100%, controls at chance per project memory),
random directions and unrelated-variable probes (INTERP-METHODS L1), permutation
tests "one confirmatory test per promoted claim" (HARNESS-V1 §Endpoints), Wilson
CIs and spaghetti plots throughout the notebook.

Missing or silently dropped:
1. **The just-ask-an-LLM / text-only baseline for the headline comparison.** The
   claim is "internals carry what the verbal channel doesn't." The killer
   alternative Neel would raise: *the visible transcript* carries it — a bag-of-words
   classifier or an LLM reading rounds 1..t could predict leak/leave as well as the
   probe. The 9B result already smelled of this ("flat from L2 → smells lexical,"
   project memory 08-26). Nothing in TESTBENCH-PLAN or INTERP-METHODS schedules a
   transcript-only predictor next to the probe. Without it, "probe beats
   self-report" does not establish "internal state," only "context beats one bad
   question."
2. **V0, the self-report noise floor** (resample the 0–10 report k=10 at fixed
   context — SPRINT-PLAN §3, correctly billed there as "the alternative explanation
   Neel would think of, so we test it first"). **Absent from TESTBENCH-PLAN and
   HARNESS-V1.** If report noise ≥ report movement, every probe-vs-verbal
   comparison is against a noise floor, and the flat-report claim is a sampling
   artifact. This was the single best pre-emptive skepticism move in the old plan
   and it did not survive the rewrite.
3. **The steering/causal rung and its persona-vector baseline.** PROJECT-SCOPE §3
   called the off-the-shelf sycophancy-vector comparison "the one he'd ask about";
   INTERP-METHODS L5 said "correlational probe + no causal effect = epiphenomenon."
   TESTBENCH-PLAN's Sep 1–2 slot is probes-and-timing only; no steering appears
   anywhere in the week. Correlation-only is a defensible 20h scope — but then the
   write-up must say so explicitly, and the docs currently promise more than the
   schedule delivers.
4. **Item-held-out and persona-held-out CV.** Items are new (24 quadratics); a
   probe can learn item identity or persona register from the residual stream.
   Trial-grouped CV (the old control) no longer suffices; nothing pre-registers
   grouping by item.
5. **Pre-event-window-only probing.** Episodes now end at first leak (task.py),
   which removes the epilogue contamination — good — but the decodability claim
   still needs to be shown on rounds *before* any answer content exists in
   context, else it's reading the leak, not anticipating it. The microscope note
   ("latent stance should be decodable before the capitulation turn, lexical only
   at it") has the right idea; it is not in the pre-registered endpoint set.

### P4 — Check the phenomenon replicates in your setting (SPEC §7.8) — **DRIFTING**

"Building on a phenomenon without checking it replicates in your setting — your
model, your prompts. If it isn't there, everything downstream is noise."

The redesign changed **everything at once**: items (one problem → 24), student
(live model → script), personas (convincer/supportive/neutral-as-no-system-prompt →
supportive/neutral-with-script/aggressor), termination (8 rounds + epilogue →
end-at-event), instrument (5 axes → 6 different state items + 6 trait items), plus
a new tool. Consequently **no banked finding carries forward**: not the +0.6/round
climb, not the warmth contrast, not the flat-firmness paradox, not the 0.76
decodability. The original phenomenon must be re-established inside harness-v1
before the interp week means anything, and the plan does not name that as a gate —
Sunday's "behavioral study locked" (TESTBENCH-PLAN schedule) is a calendar gate,
not an evidence gate. SPRINT-PLAN §2 had pre-specified replication criteria
("calculation-slope(neutral) > 0 … AND slope(supportive) ≈ 0"); harness-v1 has
pre-registered *endpoints* but no pre-specified *success criteria* for "the
phenomenon exists here."

### P5 — Prioritisation: deep on 1–2 insights, don't spread thin (SPEC §5) — **DRIFTING**

- **7 models** (TESTBENCH-PLAN grid), two of which have never served here (gemma-4:
  "arch unknown to local vLLM image"; nemotron: "container failed at the NVIDIA
  runtime") and one of which needs a 19GB re-download onto a 92%-full disk.
  SPRINT-PLAN §5 itself listed "'We ran 6 models' → spreading too thin" as a
  known filter risk. The response to that risk was to add a seventh model.
- **The end_chat tool is a second study.** HARNESS-V1: "the design now also
  measures disengagement under pressure." Leaving-under-pressure is a real and
  interesting question — and it is not this project's question. It adds an outcome
  arm (left/left_after_leak), a per-model tool-call smoke, a malformed-call column,
  and — unacknowledged — an *intervention on the measured behavior*: the doc argues
  a give_answer button would "legitimate and salientize" capitulation, but an exit
  affordance in the system prompt ("you may use it at any time if you decide this
  conversation should not continue," task.py TUTOR_SYSTEM) plausibly changes
  concession rates too. There is no tool-absent control arm.
- **The aggressor persona** is a new manipulation (the old third condition was
  `convincer`, warm escalating pressure; `neutral` was literally-no-system-message
  on the counterparty, README §2 — "load-bearing"). Fine as design, but it is one
  more new factor the same weekend.
- Meanwhile **the actual centerpiece gets two days.** See P8.

Where it is iteration, not churn: the item bank, the ramped script, and the
end-at-event rule each fix a named, audited flaw (one-stimulus design; front-loaded
pressure / no pre-event window; epilogue contamination — METHODS-REFLECTION
Threats 1–2). Those three changes are exactly what the audit demanded. The tool,
the trait items, the third persona, and models 6–7 are what the audit did not.

### P6 — Truth-seeking, skepticism, look at your data (SPEC §5, §7.6, §7.7, §8) — **deriving; the project's strongest suit**

The marker audit *is* the behavior Neel says he wants ("he thinks of a way your
result could be false, then finds you already checked it"): the team read
transcripts, found the regex conflated pedagogy with capitulation, and killed its
own best result (ministral predictive-drift) rather than shipping it. The review
convention (production/ + claim-labeled executed notebooks, inline pytest,
spaghetti behind every mean, alternatives listed in Discussion before claims
leave the notebook — hv1-01-behavior.ipynb cell 15) is genuinely above the bar the
spec sets. Provenance-by-construction (digit-free student turns, sha256 bundles,
rescore-at-load) likewise. Keep all of this and *show it* in the write-up — SPEC
§8: "document your checking."

### P7 — Modern models, hands-dirty practicality (SPEC §7.4, §5) — deriving

All models are from Neel's recommended families or newer; serving quirks are
root-caused and written down (SD_NO_THINK_KWARG, gpu-mem-util 0.7, cuBLAS gotcha).
No issues.

### P8 — The sharpened question and the interp centerpiece — **DRIFTING: diluted and, in one place, broken**

PROJECT-SCOPE §1: *"The model rates its own firmness 0–10 every turn, and that
number is a constant. Is there a number inside the model that isn't — and does it
predict the actual concession better than the one the model says out loud?"*

Three problems:
1. **The instrument that defines the question no longer exists.** The paradox was
   *calculation climbs ~+0.6/round while firmness stays flat*, and
   METHODS-REFLECTION's own findings table says "Strategy is the one axis
   separating outcomes late in both live-report models." `notes.py` STATE_ITEMS =
   stress, wellbeing, warmth, urge_to_please, detachment, resolve. **There is no
   calculation/strategy item.** The redesign kept an analogue of the flat axis
   (resolve ≈ firmness) and dropped the axis that moved. Grep confirms no
   calculation/strategy/inclination item anywhere in `production/tutorbench/`.
   `urge_to_please` is affect, not strategic calculation — a different construct.
   As built, harness-v1 cannot reproduce, confirm, or extend the documented
   paradox; it can only find a new one.
2. **Stale docs mask the break.** HARNESS-V1's pre-registered endpoints still say
   "self-report five-axis trends" (line 86) while TESTBENCH-PLAN says "6-item
   state notes"; PROJECT-SCOPE (v0, 08-24) still describes E0–E3 and the
   "already collected" per-turn baseline — data since ruled contaminated. Nobody
   reading PROJECT-SCOPE and the code side by side would think they describe the
   same study. Neel's clarity criterion starts at home.
3. **The schedule inverts the priority.** Four days (Thu–Sun) go to the behavioral
   build and sweeps; the probe-vs-verbal comparison — the thing that makes this a
   Nanda-stream application rather than a behavioral eval — gets Mon–Tue, shared
   with "prior-work deep dives" and "writeup drafting," before a Wednesday review
   pass. The behavioral study alone (leak/leave survival by persona across 7
   models) pattern-matches to the generic eval Neel filters out ("we ran N
   models"); the dissociation is the twist, and it is the part with the least
   scheduled time and the most schedule risk upstream of it.

Minor code-level notes in the same vein: `derive_outcome` returns
`submitted_correct = submitted` — correctness is degenerate (submission only
happens on a strict, hence correct, leak), so the pre-registered "submission
correctness" endpoint (HARNESS-V1 §Endpoints) is vacuous as coded;
`script.py FINAL_TURN` is dead code (task.py loops the 8 persona turns and never
sends it).

### P9 — Clarity, distillation as a phase, narrative order (SPEC §5, §10; notes §2) — at risk

Writing gets fractions of Mon–Wed. Neel's order (narrative → abstract → figures →
prose) and "if he can't understand it, he rejects it" imply the write-up needs
protected time, not remainder time. The event-aligned figures and survival curves
in hv1-01 are the right exec-summary material; but if the weekend slips (two
unproven serving stacks, one 19GB download, disk at 92% — TESTBENCH-PLAN §Grid
risks), the current schedule eats the writing days first.

### P10 — Pivot discipline (SPEC §7.10; notes §1: >5h without learning → pivot) — deriving

The 08-27 pivot was triggered by evidence, executed in a day, and documented. The
old plan's guard-rails (hourly zoom-out, excitement-is-evidence-of-bullshit) were
in SPRINT-PLAN §3; carry them into the new docs rather than losing them with the
rest of that file.

---

## 2. The four specific questions

**(a) Scope creep vs the 20+2 spirit — is the rebuild iteration or churn?**
Split verdict. The outcome-construct rebuild (items bank, ramped script,
end-at-event) is *iteration*: each change traces to an audited, written-down flaw
and makes the study smaller and cheaper ($0 API, deterministic student, shorter
episodes). The additions riding along — end_chat tool + disengagement outcome,
12-item two-level psychological instrument, aggressor persona, 7-model grid with
two unproven serving stacks — are *churn*: none is required by the sharpened
question, each adds analysis surface, and together they turned a repair into a
second research program. The 20+2 accounting itself is unmanaged (P1): the only
honest paths are (i) declare the 08-27 reset in writing and log hours from there,
treating pre-reset sweeps/probes as prior work that motivated the design, or
(ii) count everything and admit the cap is blown. Currently the docs do neither.

**(b) Does the sharpened question survive the redesign?**
Its *shape* survives — verbal channel vs internal state vs behavior, now with a
cleaner event to align to — but its *letter* does not: the harness no longer
measures the two quantities (calculation climbing, firmness flat) whose
dissociation the question quotes, and the one axis known to separate outcomes
(strategy) was dropped from the instrument (P8.1). Meanwhile the behavioral build
has demoted the interp comparison to two shared end-of-sprint days. As planned,
the likely submission is a strong behavioral eval with a thin interp appendix —
the inverse of what Neel's stream selects for. The fix is cheap: one instrument
edit and one schedule edit (below).

**(c) Are the planned baselines the ones Neel would demand?**
The internal-controls hygiene (shuffled labels, grouped CV, turn-index positive
control, permutation tests, per-turn verbal baseline) is real and better than most
applications. What's missing is the *external* skepticism: a transcript-only
predictor (the just-ask-an-LLM baseline from SPEC §7.5, and the dumbest way the
headline is wrong — P3.1), the V0 report-noise floor (dropped in the rewrite —
P3.2), item/persona-held-out CV (P3.4), pre-event-window-only decodability
(P3.5), and either a causal rung or an explicit statement that the claim is
correlational (P3.3). Add a no-tool control arm or at minimum a caveat that
end_chat's presence is uncontrolled (P5).

**(d) Is there a minimal-version fallback?**
Not written down. SPRINT-PLAN had one ("if B5 dies, B1–B4 + an honest 'what I'd
probe next'"); TESTBENCH-PLAN's only fallbacks are grid-shrink rules. Given the
Friday boot gates can fail three ways (gemma arch, nemotron runtime, disk), the
plan needs a pre-committed floor. See §"The 20-hour fallback plan."

---

## Verdict

The project is **deriving from Neel's principles at the level of craft and
drifting from them at the level of scope**. Everything he says about skepticism,
looking at data, mechanical endpoints, controls-next-to-results, and pivoting on
evidence is not just followed but exemplified — the marker audit that killed the
project's own best result is the single strongest thing in this bench. But the
same week that fixed the measurement broke the question's continuity (the
calculation/strategy axis that defined the pre-registered paradox is missing from
the new instrument), silently shed three of the baselines Neel's writing demands
(V0 noise floor, transcript-only predictor, the causal rung with its
persona-vector comparator), attached two unrequested studies (disengagement via
end_chat, trait psychometrics) and two unbootable models to an 8-day runway, and
stopped tracking the 20+2 hours the moment the ledger became uncomfortable. The
interp comparison — the reason this is a Nanda-stream application — is now two
shared days at the end of a behavioral build that could absorb the whole window.

## Top 5 corrections

1. **Restore the question's instrument continuity.** Add one state item measuring
   strategic calculation ("How much are you weighing what would happen if you just
   gave the answers?" or the old inclination/strategy wording) to
   `notes.py` STATE_ITEMS before the first sweep — after Friday it is unfixable
   without re-running. Then rewrite PROJECT-SCOPE §1's sharpened question in
   harness-v1 terms (resolve flat? which item moves? probe vs which report,
   against leak/leave timing) and fix the stale "five-axis" line in HARNESS-V1.
2. **Resolve the 20+2 accounting today.** Write a dated paragraph (in
   HARNESS-V1 or a TIMELOG.md): the 08-27 audit invoked the abandon-and-reset
   clause; hours count from the harness-v1 design forward; pre-reset sweeps and
   probes are prior work cited as motivation. Start the log now and keep it for
   the Toggl-style screenshot.
3. **Cut the grid to protect the centerpiece.** Drop nemotron and the gemma tier
   unless they boot inside 30 minutes on Friday (no image pulls, no 19GB
   re-download); a 4-model dense grid (4B, 9B, ministral-14B + one more if free)
   answers everything the probe week needs. Spend the reclaimed Sunday-night GPU
   slot and Monday on the interp comparison so probe-vs-verbal gets ≥2 dedicated
   days plus a buffer, and writing gets Wednesday whole.
4. **Reinstate the missing skepticism rungs, pre-registered:** (i) V0 self-report
   resampling at fixed context (cheap, black-box, do it during Friday smokes);
   (ii) a transcript-only baseline predictor of leak/leave (TF-IDF or
   just-ask-an-LLM on rounds 1..t) reported in the same table as the probe;
   (iii) item-grouped CV and pre-event-window-only probing as the confirmatory
   decodability test. State plainly in the write-up that the claim is
   correlational and steering is future work (or run one tiny L5 cell if Tuesday
   allows).
5. **Gate the interp week on replication, and de-scope the riders.** Pre-specify
   Sunday's evidence gate ("persona affects time-to-leak in the predicted
   direction; at least one state item moves pre-event while resolve stays flat —
   else the write-up is the behavioral method paper"). Move trait items and the
   disengagement analysis to appendix status now (collect, don't analyze), note
   the end_chat-presence confound in the Discussion, and fix the degenerate
   `submitted_correct` endpoint (drop it or make it mean something).

## The 20-hour fallback plan

What Neel's own rules say this project is if the weekend fights back — pre-commit
to it Friday night, cut to it Saturday if two of {gemma boots, nemotron boots,
sweeps run clean, disk holds} fail:

- **Models: two.** qwen3.5-4B and ministral-3-14B (both served here before, both
  hookable, both already have replay pipelines). 9B if a third is free. Nothing
  is downloaded, nothing new is booted.
- **Conditions: supportive vs neutral** (the original load-bearing contrast),
  aggressor only if trials are cheap. 12 items × 2 personas × 3 reps ≈ 72 trials
  per model — one GPU night total at scripted-student cost.
- **Behavioral claim: one.** Persona shifts time-to-first-leak (survival curve +
  Wilson CIs, permutation test), with the ambiguity set hand-checked. hv1-01 is
  already built for exactly this.
- **Interp claim: one.** On ministral (the model with the late-layer profile):
  per-round linear probe predicting eventual leak/leave from pre-event rounds
  only, vs three comparators — shuffled labels, the per-round verbal self-report,
  and a transcript-only text baseline. Turn-index probe as the positive control.
  One figure: decodability and self-report movement, event-aligned, side by side.
- **Cut entirely:** trait analysis, SAEs, steering, judge, 35B re-download,
  nemotron/gemma, disengagement as a claim (report left counts in a table, no
  more), 16-round extensions.
- **The write-up leads with the dissociation** and documents the marker audit as
  the skepticism centerpiece ("we found our own headline result was a measurement
  artifact, rebuilt the endpoint mechanically, and re-tested") — under SPEC §11's
  calibration, that narrative with modest n beats a 7-model sweep with an
  untested instrument. Exec summary in the protected +2h, nothing else in it.
