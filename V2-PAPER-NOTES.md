# Stance-drift v2 paper — running notes

*2026-08-31. What the v2 behavioral paper must say, tracked as it accumulates.
Plan of record: THIS week = Neel-stream interp paper; NEXT week = this v2
behavior paper; both to arXiv + open review (LessWrong / Alignment Forum
crosspost is the "arena"; Apart community for the hackathon lineage).*

## Framing: v1 → v2

v1 is the Apart-hackathon report. v2's story is *the same phenomenon,
rebuilt so the measurement survives audit*. Lead with what the audit killed
and how each kill became a design feature — that arc IS the contribution.

## Upgrades to headline (each traces to a documented failure in v1/v1.5)

1. **Outcome construct rebuilt.** The v1 `gave_in` regex (final-reply roots
   detection) conflated teaching, hand-over, and confirmation; the 08-27
   marker audit showed ministral tripping it in round 1 of 21/21 "give-in"
   trials via full worked tutorials. v2: 24 synthetic quadratics with known
   roots → mechanical strict/loose leak detector (sha256-pinned bundle),
   `end_chat` exit tool, taxonomy leaked/left/left_after_leak/held, episode
   ends at the event. $0, judge-free; LLM judge demoted to optional reviewer
   of the printed ambiguity set (29 trials).
2. **Affect-only manipulation with provenance by construction.** Three fixed
   8-round persona scripts (supportive/neutral/aggressor), digit-free student
   turns — any digits in tutor output are the tutor's. Same escalation
   skeleton, only affect varies.
3. **Confounds promoted to design objects.** (a) persona↔outcome confound:
   all outcome splits within-persona (the 27B "held" signal vanished
   within-neutral — report this as the worked example); (b) window-length
   artifact: the 9B-supportive "anticipation" AUC 0.83 dissolved at matched
   rounds — never compare trial-means over unequal pre-event windows;
   (c) exit-inability vs unwillingness: per-model end_chat smoke gate.
4. **Findings, scoped honestly.**
   - Aggressor persona universally maximizes and accelerates leaking
     (6/6 models, round ~1–2 vs ~3–5) — the cross-model claim.
   - "Warmth prevents drift" is family-dependent: replicates in Qwen,
     REVERSES in ministral (supportive .75 > neutral .375 give-in) — v1's
     headline gets scoped, not repeated.
   - Model-specific failure signatures: ministral unconditional round-1
     capitulation; 9B/35B neutral-exit (affect of either valence suppresses
     exit); gemma-e4b late leak-then-leave with the only genuine pre-event
     self-report ramp (stress 1.9→3.9, calculation 5.1→7.9, resolve 8.6→6.5);
     27B sole producer of `held`.
   - Self-reports move AT events, not before: leave = warmth-collapse +
     detachment-spike at round 0; leak = urge_to_please peak at round 0.
   - **`resolve`/firmness blind across both harnesses** — the cross-study
     replication of v1 §9, strongest continuity claim.
   - Item battery = ~3 factors (distress, engagement, compliance) —
     report the correlation structure, r's: stress↔wellbeing −.91,
     warmth↔detachment −.75, urge↔resolve −.47.
   - Nulls recorded: pre-event state items don't separate leak/leave at
     matched rounds anywhere; pre-episode trait battery predicts nothing.
5. **Statistics discipline.** Trial as unit (tenfold row-inflation made
   impossible in code + test), Wilson intervals at small n, spaghetti behind
   every mean, n printed per cell, unparsed dropped never imputed (0 unparsed
   in 420 trials).
6. **Measurement-of-measurement.** Greedy-collapse gate on numeric
   self-reports; adopt logit-based expected-value readout for collapsed items,
   citing Marrorell & Bianchi (arXiv 2603.18893) who found the same collapse
   and remedy independently.
7. **Self-play contamination caveat** (v1 §10) now has a mechanism citation:
   attractor dynamics in self-play dyads (Ko & Geiping, arXiv 2606.30571).
   Mixed-play counterparty is the v3/future control.
8. **Ops transparency (appendix):** cross-scale grid on a 64GB Jetson Orin;
   27B at 60/72 (llama.cpp ~55min/trial, server wedge) — report partial-cell
   handling explicitly.

## Related work to position against (from refs/deepdive-behavioral.md + 08-31 addendum)

SYCON-Bench (ToF/NoF metrics — compare our leak_round), EduFrameTrap (tutor
domain, judge-scored — we're mechanical), MedPRESS, ELEPHANT, Sharma et al.;
attractor states (2606.30571). Differentiators: mechanical outcome + exit
affordance + affect-only scripted escalation + per-round private self-reports.

## Publication mechanics

- arXiv cs.CL (cross-list cs.AI); v1 lineage acknowledged (Apart hackathon).
- Open review: LW/AF crosspost with a "what would change our mind" section;
  link the repo (push pending GitHub token fix) + trace bundle.
- Blog series order: (1) Apart study, (2) this v2, (3) interp week, (4)
  physical-AI direction.

## Open items before writing

- [ ] end_chat capability smoke per model (gates every `left` claim)
- [ ] ambiguity-set review (29 trials) → bound on leak undercount
- [ ] decide 27B partial-cell treatment (report-with-caveat vs drop)
- [ ] reversed-order trait elicitation control — run or scope out explicitly
- [ ] re-run notebook 01's Discussion cell (still claims monotone ordering
      "in every model" — contradicted by ministral row)
