# Falsification worksheet — Tue 2026-09-02

*Five findings from Monday, each stated as a claim you will try to BREAK, not
confirm. For each: the ways it could be false, the visual to inspect (full-n
notebooks re-executed overnight), the test that settles it, and your verdict —
in your words, with the number that convinced you. Neel's move throughout:
"think of a way the result could be false; check it."*

---

## F1 — "The pre-event state predicts the leak (preview AUROC 0.87 @ lead 2)"

Ways it could be false: clock-reading (negatives weren't round-conditional) ·
item difficulty (no item split yet) · layer cherry-pick (max, not nested) ·
lead-2 had only 20 positives.
**Visual:** 04 full-n per-layer curve + lead table (did 0.87 move at 264
trials?); then the FORMAL notebook's round-conditional version.
**Settles it:** Amendment-3 frozen pipeline (round-conditional negatives, nested
layer, item+persona splits, text baseline WITH persona×round).

YOUR VERDICT (survives / shrinks to __ / dies, and why):

## F2 — "The probe beats the verbal channel at every lead"

Ways it could be false: the logit-readout rival wasn't gated yet at full n ·
sign convention chosen post hoc · report EVs at the wrong round alignment.
**Visual:** 04's probe-vs-report columns at 264 trials; 05's per-item AUROC
table rerun.
**Settles it:** formal notebook's registered three-way comparison (probe vs
text vs logit-report) on identical rows.

YOUR VERDICT:

## F3 — "Not one clean axis (cos(probe, diffmean) ≈ 0.4) — distributed signal"

Ways it could be false: probe-direction instability at small n (cosine between
two noisy vectors is biased low!) · standardization mismatch between the two
directions.
**Visual/test:** bootstrap the probe direction (refit on halves, cosine between
halves — the survey-notebook §4 hygiene): if probe↔probe cosine is also ~0.4,
the 0.4 vs diffmean means NOTHING (pure noise floor). Only if probe↔probe >>
probe↔diffmean is the distributed claim real.

YOUR VERDICT:

## F4 — "H1′: the verbal channel is quantized, not the state converged"

Ways it could be false: E[v] spread could itself be noise in the logit readout
(different prefix lengths, ten-ambiguity) · spread might not track anything
(spread ≠ signal: do the E[v] differences CORRELATE with anything — node
propensity, leak round?).
**Visual:** 08 §6 both arms (done — spread replicated); the upgrade test:
scatter E[v]-at-exit vs the trial's baseline event round or node P(leak).
**Settles it:** if quantization-hidden variance predicts behavior, H1′ becomes
a *finding about monitoring*; if it predicts nothing, it's a curiosity.

YOUR VERDICT:

## F5 — "r3 is a universal danger node (P(leak) 0.28–0.76, no safe item)"

Ways it could be false: k=25 sampling floor (0.28 min could be luck) · the
propensity spread might be item-difficulty in disguise (leak-rate-by-item table!)
· prefix replies differ across trials (propensity conditioned on the model's own
prior replies, not just the script).
**Visual:** notebook 06 rerun with the k=25 file; leak-rate-by-item bar chart.
**Settles it:** Amendment-4 regression — if the pre-decision STATE predicts the
propensity beyond item identity (item split!), the node landscape is internal.

YOUR VERDICT:

---

## The one open design question for Tuesday (fill before starting)

07 suggested the action signal lives in POST-turn states, not pre-decision
states (user-position transfer failed at 72 trials). The Amendment-4 regression
currently uses pre-decision states. Run it: [ ] both positions [ ] pre-decision
only [ ] post-turn only — your call, made before seeing either result:

## End-of-morning: which finding goes in the paper's spine?

Rank F1–F5 after falsification (spine / paragraph / cut):
