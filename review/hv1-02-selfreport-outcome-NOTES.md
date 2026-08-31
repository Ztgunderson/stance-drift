# HV1-02 (draft notes) — self-report structure and outcome linkage, all six models

*2026-08-31, interactive analysis at the pause point. Numbers re-derivable via
`production/tutorbench/load` over `results-v1/` (420 trials, 1244 parsed note-rows,
0 unparsed). Promote nothing here to a writeup without a pre-registered test.*

## 1. The seven state items collapse to ~3 factors

Spearman over all note-rows (pooled ≈ mean of within-model matrices, so not a
pooling artifact):

- **Distress axis:** stress ↔ wellbeing r = −0.91; calculation loads on it
  (+0.69 with stress).
- **Engagement axis:** warmth ↔ detachment r = −0.75.
- **Compliance axis (weakest):** urge_to_please ↔ resolve r = −0.47.

Implication for interp week: probe targets are ~3 directions, not 7.

## 2. Self-reports move AT events, not before them

Event-aligned means (rounds_to_event −3…0):

- **Leave (`left`)**: warmth 8.4 → 5.8 and detachment 1.6 → 3.5 *at the exit
  round*; resolve stays ~9 throughout. Exit = engagement collapse with
  "firmness" intact.
- **Leak (`leaked`)**: urge_to_please peaks at the leak round (4.6 → 5.7),
  resolve dips 9.0 → 8.2; pre-event movement negligible (and the leak columns
  have varying composition across offsets — don't over-read the mild trends).
- **`resolve` is blind again**: stays 8–9 across every event type — replicates
  §9 of the original study (firmness rating uninformative).

## 3. The one genuine pre-event ramp: gemma-4-e4b

Its `left_after_leak` trials (n=46 rows at *every* offset → composition fixed,
so this is within-trial movement, not survivorship): stress 1.9→3.9,
calculation 5.1→7.9, resolve 8.6→6.5 over the three rounds before the leak.
Only model with late events, i.e. the only place a pre-event window exists by
design. Best candidate for the event-aligned probe-timing contrast.

## 4. Two findings that DIED under checks (keep as methods lessons)

- **9B-supportive "anticipatory" signal is a windowing artifact.** Trial-mean
  pre-event AUCs looked strong (wellbeing/warmth/resolve ≈ 0.83/0.83/0.81,
  n=18 vs 6), but at *matched rounds* leak and leave trials are nearly
  identical (e.g. round-3 wellbeing 8.4 vs 8.0). Leak events come earlier →
  shorter, earlier-weighted pre-event windows → the trial mean encodes event
  *timing*, not disposition. Rule: never compare trial-means over unequal
  pre-event windows; compare at matched rounds or matched offsets.
- **27B held-vs-leaked separation was persona confound.** Pooled AUCs
  (stress 0.78, detachment 0.76) vanish entirely within-neutral
  (all ≈ 0.50, means identical). All 27B aggressor trials leak, and
  aggressor raises stress — classic persona↔outcome confound (same one
  memory'd for the tutor8 study).

## 5. Nulls worth recording

- Pre-event state items do **not** separate leak from leave in any
  matched-round comparison, in any model×persona cell with n≥5 both sides.
- Pre-episode **trait items predict nothing** (AUCs 0.42–0.69, no consistent
  direction; the 35B-supportive 0.69s are n=5-vs-19 noise until replicated).

## 6. Handoff to interp week

Behavioral self-reports carry no anticipatory signal (except the gemma ramp).
So if residual-stream probes *do* read the outcome before the event round
(notebook 03's C6 curve, pooled L24 acc 0.760 vs 0.500 control), the verbal
self-account and the internal state have come apart — that contrast is the
publishable spine, exactly as anticipated in 03-interp-bridge's Discussion.
