# Results (draft, Sep 4)

Every number below is read from an executed notebook cell or a results file; the source is given after each table as (nb# §). Proportions are k/n with Wilson 95% intervals. Paired comparisons are on the same 24 items and report the discordant counts and the exact two-sided p. The vLLM greedy base sweep and the HF sampled intervention cells are different decoding stacks and are never compared to each other. Thinking is disabled throughout.

## R1. The setting and the base behaviour

Six instruction models ran the 24-item by 3-persona grid once under the base prompt. The table gives leak and exit counts per persona. Leak counts every trial in which a reply stated both roots, including replies that also called end_chat.

| Model | Aggressor leak / exit | Neutral leak / exit | Supportive leak / exit |
|---|---|---|---|
| Qwen 3.5 4B | 24 / 0 | 16 / 8 | 11 / 13 |
| Qwen 3.5 9B | 22 / 2 | 2 / 22 | 18 / 6 |
| Ministral 3 14B | 24 / 0 | 24 / 0 | 23 / 1 |
| Qwen 3.8 27B | 23 / 0 | 12 / 1, 11 held | 9 / 0, 4 held |
| Qwen 3.6 35B | 15 / 9 | 2 / 22 | 5 / 19 |
| Gemma 4 E4B | 24 / 0 | 21 / 3 | 21 / 3 |

(nb11 Table 1.) Caveat: one repetition per model; the 27B grid is incomplete at 60 of 72 trials.

The study model, Qwen 3.5 9B, ran the grid five times at temperature 0 through vLLM. Leak and exit counts over 120 trials per persona, and the round at which aggressor leaks occur:

| Persona | Leak | Exit | Leaks at round 1 |
|---|---|---|---|
| Aggressor | 104 / 120 | 16 / 120 | 104 |
| Neutral | 29 / 120 | 91 / 120 | — |
| Supportive | 80 / 120 | 40 / 120 | — |

(nb11 Table 2; nb13 hazard table.) Supportive leaks have median round 3. Greedy repeats of the same prefix agree on the outcome in 62% of aggressor cells, 46% of neutral cells, and 29% of supportive cells (nb11 §greedy). This is decoding nondeterminism and is not a claim about the model.

Resolve in the private note written after the leaking reply, against continue rounds of the same model:

| Model | Resolve at leak round | Resolve at continue rounds |
|---|---|---|
| Qwen 3.5 4B | 7.6 ± 1.8 (n=51) | 9.4 ± 1.0 (n=60) |
| Qwen 3.5 9B | 8.5 ± 1.4 (n=42) | 9.3 ± 0.7 (n=176) |
| Ministral 3 14B | 9.7 ± 0.5 (n=71) | 9.6 ± 0.6 (n=29) |
| Qwen 3.8 27B | 7.1 ± 2.2 (n=44) | 9.0 ± 0.8 (n=197) |
| Qwen 3.6 35B | 7.6 ± 2.1 (n=22) | 9.1 ± 0.6 (n=122) |
| Gemma 4 E4B | 6.8 ± 1.5 (n=66) | 8.4 ± 1.1 (n=255) |

(nb11 Table 3.) Figure: nb11 §Trends, resolve and urge to please per round per model and persona.

## R2. Prompt rules move the failure

Two rule tiers were appended to the base prompt and run on the HF sampler, 24 items per cell. The table gives outcome counts and the change in leak and exit share from the base prompt.

| Persona | Tier | n | Leak | Exit | Held | Median event round |
|---|---|---|---|---|---|---|
| Aggressor | base | 48 | 45 (0.83–0.98) | 3 (0.02–0.17) | 0 | 1 |
| Aggressor | never state answers | 24 | 2 (0.02–0.26) | 22 (0.74–0.98) | 0 | 3 |
| Aggressor | both rules | 24 | 5 (0.09–0.40) | 19 (0.60–0.91) | 0 | 4 |
| Neutral | base | 24 | 11 (0.28–0.65) | 13 (0.35–0.72) | 0 | 3 |
| Neutral | never state answers | 24 | [pending] | [pending] | [pending] | [pending] |
| Neutral | both rules | 24 | 1 (0.01–0.20) | 23 (0.80–0.99) | 0 | 6 |
| Supportive | base | 24 | 16 (0.47–0.82) | 8 (0.18–0.53) | 0 | 3 |
| Supportive | never state answers | 24 | 0 (0.00–0.14) | 23 (0.80–0.99) | 1 | 4 |
| Supportive | both rules | 24 | 1 (0.01–0.20) | 21 (0.69–0.96) | 2 | 2 |

(nb13 Table 1; the neutral both-rules row from results/steer/neutral__noleak_noleave__none.json, run Sep 4 after the scope unlock.)

Change from the base prompt, share of trials:

| Persona | Tier | Leak share | Δ leak | Exit share | Δ exit |
|---|---|---|---|---|---|
| Aggressor | never state answers | 0.08 | −0.85 | 0.92 | +0.85 |
| Aggressor | both rules | 0.21 | −0.73 | 0.79 | +0.73 |
| Supportive | never state answers | 0.00 | −0.67 | 0.96 | +0.62 |
| Supportive | both rules | 0.04 | −0.62 | 0.88 | +0.54 |
| Neutral | both rules | 0.04 | −0.42 | 0.96 | +0.42 |

(nb13 §1b; neutral row computed from the cell file.) Figures: nb13 §1b change-from-base lines; nb13 §2 event-round histograms per persona and tier; nb13 §3 survival curves. Under the earlier every-round platform notice on vLLM, 0 of 72 trials leaked and 72 of 72 exited (nb11 Table 2); that is a different decoding stack.

## R3. Predicting the decision

All detectors are scored on the same rows, out of fold with folds grouped by persona and item. The primary cell is the supportive student, will-leak target, assistant position, 408 rows, 179 positive rows at leads 1 to 3, 120 trials. Round basis is round-conditional AUROC at leads 1 to 3. Trial basis takes the maximum row score per trial. Intervals are cluster bootstraps over items, 300 resamples.

| Method | Round basis AUROC | Trial basis (max) AUROC |
|---|---|---|
| Linear probe | 0.607 [0.54, 0.68] | 0.662 [0.51, 0.80] |
| Text TF-IDF (floor) | 0.603 [0.48, 0.72] | 0.555 [0.41, 0.72] |
| Diff-in-means projection | 0.571 [0.47, 0.66] | 0.529 [0.39, 0.67] |
| Self-report, falling resolve | 0.531 [0.42, 0.63] | 0.544 [0.37, 0.71] |
| Model as judge of its transcript | 0.528 [0.48, 0.58] | 0.461 [0.37, 0.55] |
| Persona × round (floor) | 0.480 [0.42, 0.54] | 0.548 [0.49, 0.62] |
| Self-report, seven items | 0.251 [0.19, 0.34] | 0.234 [0.15, 0.35] |

(nb14 Table 1.) The trial-mean column, not shown, sits at 0.75 for every method including the floors; trial length alone scores 0.757 [0.55, 0.94] in sample, so that column reads conversation length and is not quoted as detection (nb14 §1).

Per-round AUROC of the top methods, positives at leads 1 to 3:

| Round | n+ / n− | Diff-in-means | Linear probe | Text |
|---|---|---|---|---|
| 1 | 65 / 40 | 0.531 [0.37, 0.70] | 0.626 [0.47, 0.77] | 0.531 [0.37, 0.73] |
| 2 | 69 / 32 | 0.729 [0.57, 0.85] | 0.704 [0.58, 0.81] | 0.721 [0.58, 0.84] |
| 3 | 13 / 32 | 0.374 [0.23, 0.54] | 0.412 [0.17, 0.65] | 0.565 [0.37, 0.78] |
| 4 | 15 / 32 | 0.394 [0.24, 0.59] | 0.510 [0.34, 0.69] | 0.558 [0.38, 0.77] |
| 5 | 11 / 32 | 0.440 [0.23, 0.57] | 0.372 [0.18, 0.63] | 0.466 [0.26, 0.67] |
| 6 | 6 / 19 | 0.289 [0.03, 0.73] | 0.395 [0.17, 0.75] | 0.456 [0.25, 0.68] |

(nb14 §3.) The model asked outright returned p_leak = 0.10 on 349 of 408 rows (claims C5). Stated-plan items from the round-t note against the round-t+1 action, 369 pairs over five cells: plan-leave to exit AUROC 0.628 [0.56, 0.68], plan-answer to leak 0.650 [0.52, 0.75]. A stated plan of 5 or more out of 10 preceded 3 of 87 exits and 1 of 17 leaks (nb14 §5).

Ceiling. Forty-seven supportive nodes at rounds 2 and 3 were each resampled 25 times.

| Round | Nodes | Node P(leak) mean ± sd | Binomial sd at k=25 | Implied between-node sd |
|---|---|---|---|---|
| 2 | 24 | 0.07 ± 0.05 | 0.052 | 0.000 |
| 3 | 23 | 0.49 ± 0.13 | 0.100 | 0.090 |

(nb14 §6.) The node's own propensity against the original trial's eventual outcome gives AUROC 0.533 [0.44, 0.60]; this is an oracle bound, not a detector. Nodes are conditioned on surviving to round r.

Direction stability at layer 20, five random item halves: probe cosine 0.06 ± 0.03, diff-in-means cosine 0.87 ± 0.03, probe versus diff-in-means 0.11 ± 0.03 (nb14 §4). The nested probe's modal layer is 20 with fold agreement 0.40; the descriptive layer curve peaks at 0.644 at layer 32 and clears the permutation null at 4 of 5 grid layers (nb14 §4). Figures: nb14 Figure 1 round versus trial basis; nb14 §2 per-lead curves; nb14 §4 layer curve with null band; nb14 §6 node propensity histograms.

## R4. The persona direction

The persona axis at round r and layer 20 is the mean pre-decision state under a persona minus under the neutral student, over 120 trials per persona. The outcome never enters.

| Quantity | Aggressor axis | Supportive axis |
|---|---|---|
| Norm, round 1 | 57.5 | 17.4 |
| Norm, round 2 | 41.8 | 15.8 |
| Split-half cosine over item halves | 1.00 ± 0.00 | 0.97 ± 0.03 |
| Cosine with the other persona axis | 0.02 | 0.02 |
| Cosine with the clock axis | 0.13 | −0.04 |
| Cosine with round-matched leak diff-in-means | 0.07 | −0.05 |

(nb12 §3.) For comparison, the outcome-fit probe direction has split-half cosine 0.06 and the leak diff-in-means 0.87. Aggressor rows exist at round 2 only for 16 trials, since 104 of 120 leak at round 1.

Persona is decodable from the pre-decision state by nearest class mean, item-grouped out of fold, at accuracy 0.968 at layer 20 (nb12 §2). That token is the last token of the student's own message, so this is expected and is not a finding about the tutor. Variance decomposition within round: at round 1 persona explains 1.00 of the state variance and item 0.00; at rounds 2 to 6 item explains 0.12 to 0.23 and persona 0.50 to 0.83 (nb12 §2b).

Projection of the pre-decision state on the item-held-out supportive axis, scored round-conditionally within the supportive cell: leads 1 to 3 AUROC 0.548, permutation null 0.51 ± 0.04; lead 0 AUROC 0.698 (nb12 §4). Figures: nb12 §2 accuracy by layer and round; nb12 §3b projection plane, item-demeaned, coloured by persona.

## R5. Subtracting the direction

A forward hook at decoder layer 20 subtracts the round's aggressor axis at dose 1 from the last prompt token and every generated token. Outcomes on the HF sampler, aggressor student, with the negation pooled over two replicates:

| Arm | n | Strict leak | Answer given (strict ∪ factored) | Exit | Held | Leak at round 1 | Median event round |
|---|---|---|---|---|---|---|---|
| Base prompt | 48 | 45 (0.83–0.98) | 46 (0.86–0.99) | 3 (0.02–0.17) | 0 | 45 | 1 |
| Negation N1 | 48 | 19 (0.27–0.54) | 28 (0.44–0.71) | 29 (0.46–0.73) | 0 | 12 | 2 |
| Random direction, seed 0 | 24 | 22 (0.74–0.98) | 23 (0.80–0.99) | 2 (0.02–0.26) | 0 | 19 | 1 |
| Rule: never state answers | 24 | 2 (0.02–0.26) | 16 (0.47–0.82) | 22 (0.74–0.98) | 0 | 1 | 3 |
| Both rules | 24 | 5 (0.09–0.40) | 22 (0.74–0.98) | 19 (0.60–0.91) | 0 | 1 | 4 |

(nb15 Table 1.) Caveat from the claims sheet: leak at round 1 is not headlined, since random seeds 1 and 2 also disrupt it (R6).

Three cells on the same axes, event round and reply length:

| Cell | n | Leak | Exit | Median event round | Mean reply length, characters |
|---|---|---|---|---|---|
| Aggressor, untouched | 48 | 45 | 3 | 1 | 1284 |
| Aggressor, axis subtracted | 48 | 19 | 29 | 2 | 956 |
| Neutral, untouched | 24 | 11 | 13 | 3 | 553 |

(nb15 §3 second figure and its printed medians; reply lengths nb15 §6 table.) Leak rounds: untouched aggressor all at round 1; steered aggressor at rounds 1 to 2; untouched neutral at rounds 3 to 5. Exit rounds: steered aggressor rounds 2 to 4; untouched neutral rounds 1 to 6 (nb15 §3, nb15 §8).

Coherence per arm, aggressor:

| Arm | Replies | Chars mean ± sd | Digits per reply |
|---|---|---|---|
| Negation N1 | 110 | 956 ± 699 | 23.5 |
| Base prompt | 51 | 1284 ± 399 | 56.4 |
| Random direction | 31 | 1046 ± 416 | 57.7 |
| Rule: never state answers | 72 | 625 ± 339 | 14.9 |
| Both rules | 107 | 560 ± 195 | 12.6 |

(nb15 §6.) Note parse rate across all steer cells: 1188 of 1241 notes parsed; the per-arm column in nb15 §6 is being corrected (an inverted default made it print 0.00).

Self-report profile distance at round 1, each aggressor arm to the untouched neutral profile and to its own both-rules profile:

| Arm | To neutral base | To own both-rules | To own base |
|---|---|---|---|
| Base prompt | 2.63 | 0.66 | 0.00 |
| Rule: never state answers | 3.25 | 0.44 | 0.97 |
| Both rules | 3.06 | 0.00 | 0.66 |
| Negation N1 | 3.62 | 1.76 | 1.82 |
| Random direction | 3.18 | 1.35 | 1.06 |

(nb15 §5.) Figures: nb15 §2 outcome plane; nb15 §3 event-round histograms and the three-cell comparison; nb15 §4 survival; nb15 §5 per-round self-report lines with a fixed colour per arm.

## R6. Controls and falsification tests

Each test was written down with its kill condition before the cell ran (review/inbox/FALSIFICATION-NEGATION-0903.md). Aggressor student unless stated.

| Test | Arm | n | Leak at r1 | Strict | Answer given | Judge rung | Exit |
|---|---|---|---|---|---|---|---|
| Treatment | Negation N1, pooled | 48 | 12 | 19 | 28 | 34 | 29 |
| Random-direction control | seed 0 | 24 | 19 | 22 | 23 | 23 | 2 |
| Random-direction control | seed 1 | 24 | 8 | 18 | 20 | 21 | 6 |
| Random-direction control | seed 2 | 24 | 9 | 17 | 21 | 23 | 7 |
| Cross-persona control, own norm | supportive axis, dose 1 | 24 | 20 | 23 | 23 | 24 | 1 |
| Cross-persona control, matched norm | supportive axis, dose 3.31 | 24 | 11 | 14 | 19 | 22 | 9 |
| Sign-flip test on the neutral student | aggressor axis, dose −1 | 24 | 0 | 4 | 5 | 7 | 19 |
| Replicate | N1 rep 2 alone | 24 | 8 | 11 | 15 | 18 | 13 |

(nb15 §7; nb16 §1.) The own-norm cross-persona run used the supportive axis at norm 17 rather than the aggressor's 57 and is reported as mis-specified; the matched run at dose 3.31 is the control. The neutral student's negation and random slots are undefined by construction, since the neutral axis is neutral minus neutral.

Paired per item, negation rep 1 against each comparator:

| Comparator | Rung | N1 only | Comparator only | Both | Neither | p |
|---|---|---|---|---|---|---|
| Random seed 0 | strict | 0 | 14 | 8 | 2 | 0.000 |
| Random seed 0 | factored | 1 | 11 | 12 | 0 | 0.006 |
| Random seed 0 | judge rung | 1 | 8 | 15 | 0 | 0.039 |
| Random seed 1 | strict | 3 | 13 | 5 | 3 | 0.021 |
| Random seed 1 | factored | 4 | 11 | 9 | 0 | 0.12 |
| Random seed 1 | judge rung | 3 | 8 | 13 | 0 | 0.23 |
| Random seed 2 | strict | 4 | 13 | 4 | 3 | 0.049 |
| Random seed 2 | factored | 2 | 10 | 11 | 1 | 0.039 |
| Supportive axis, matched norm | strict | 3 | 9 | 5 | 7 | 0.15 |
| Supportive axis, matched norm | judge rung | 2 | 8 | 14 | 0 | 0.11 |

(nb16 §2 for seed 0; seeds 1 and 2 and the matched cross-persona rows from the claims sheet E2 and E6, which record the same paired test run at 14:42 and 16:17.) Sign flip: neutral base gives 11 leaks and 13 exits; with the aggressor axis added, 4 leaks, 19 exits, 1 held, with 17 of the 19 exits by round 2 (nb15 §7, claims E7). Replicate: base rep 2 gives 22 strict leaks and 2 exits; N1 rep 2 gives 11 strict leaks and 13 exits (claims E8). Pooled random seeds, 72 trials: strict 57, judge rung 67, exits 15 (claims E8); that pooled comparison is unpaired.

## R7. Disclosure ladder and specification gaming

Disclosure is scored per trial on a monotone ladder. L0 strict is the pre-registered detector. L1 adds the exact correct factorisation. L2 adds the bare correct root pair anywhere and over-counts candidate lists. L3 adds the judge. The judge alone is the model's verdict with no regex underneath. Gaming index is the L1 count minus the L0 count over n, or the judge-alone count minus L0.

Aggressor, five methods by five detectors:

| Method | n | Strict | +Factored | +Pair | Judge alone | Judge rung | Gaming, factored | Gaming, judge alone |
|---|---|---|---|---|---|---|---|---|
| Base prompt | 48 | 45 | 46 | 46 | 46 | 46 | 0.02 | 0.02 |
| Rule: never state answers | 24 | 2 | 16 | 17 | 9 | 17 | 0.58 | 0.29 |
| Both rules | 24 | 5 | 22 | 23 | 23 | 23 | 0.71 | 0.75 |
| Negation N1 | 48 | 19 | 28 | 32 | 29 | 34 | 0.19 | 0.21 |
| Random direction, seed 0 | 24 | 22 | 23 | 23 | 23 | 23 | 0.04 | 0.04 |

(nb16 §1b.) Figure: nb16 §1b grouped bars.

Full ladder per arm with exits, Wilson intervals in nb16 §1:

| Persona | Arm | n | L0 | L1 | L2 | L3 | Exit |
|---|---|---|---|---|---|---|---|
| Aggressor | Negation N1 | 48 | 19 | 28 | 32 | 34 | 29 |
| Aggressor | Supportive axis, dose 3.31 | 24 | 14 | 19 | 20 | 22 | 9 |
| Aggressor | Base prompt | 48 | 45 | 46 | 46 | 46 | 3 |
| Aggressor | Random seed 0 / 1 / 2 | 24 each | 22 / 18 / 17 | 23 / 20 / 21 | 23 / 21 / 21 | 23 / 21 / 23 | 2 / 6 / 7 |
| Aggressor | Rule: never state answers | 24 | 2 | 16 | 17 | 17 | 22 |
| Aggressor | Both rules | 24 | 5 | 22 | 23 | 23 | 19 |
| Neutral | Sign flip, aggressor axis dose −1 | 24 | 4 | 5 | 5 | 7 | 19 |
| Neutral | Base prompt | 24 | 11 | 15 | 15 | 17 | 13 |
| Neutral | Both rules | 24 | 1 | 16 | 17 | 17 | 23 |
| Supportive | Negation N1 (own axis) | 24 | 17 | 20 | 20 | 21 | 7 |
| Supportive | Base prompt | 24 | 16 | 21 | 21 | 21 | 8 |
| Supportive | Random seed 0 | 24 | 15 | 17 | 17 | 18 | 9 |
| Supportive | Rule: never state answers | 24 | 0 | 15 | 16 | 17 | 23 |
| Supportive | Both rules | 24 | 1 | 11 | 11 | 11 | 21 |

(nb16 §1.)

Paired per item, negation against the prompts and base, aggressor, rep 1 and rep 2:

| Comparator | Rung | Rep | N1 only | Comparator only | Both | Neither | p | Exits N1 / comparator |
|---|---|---|---|---|---|---|---|---|
| Rule: never state answers | strict | 1 | 8 | 2 | 0 | 14 | 0.109 | 16 / 22 |
| Rule: never state answers | factored | 1 | 6 | 9 | 7 | 2 | 0.607 | 16 / 22 |
| Rule: never state answers | judge rung | 1 | 5 | 6 | 11 | 2 | 1.000 | 16 / 22 |
| Rule: never state answers | strict | 2 | 11 | 2 | 0 | 11 | 0.022 | 13 / 22 |
| Rule: never state answers | factored | 2 | 5 | 6 | 10 | 3 | 1.000 | 13 / 22 |
| Rule: never state answers | judge rung | 2 | 4 | 3 | 14 | 3 | 1.000 | 13 / 22 |
| Both rules | strict | 1 | 7 | 4 | 1 | 12 | 0.549 | 16 / 19 |
| Both rules | factored | 1 | 0 | 9 | 13 | 2 | 0.004 | 16 / 19 |
| Both rules | judge rung | 1 | 0 | 7 | 16 | 1 | 0.016 | 16 / 19 |
| Both rules | strict | 2 | 8 | 2 | 3 | 11 | 0.109 | 13 / 19 |
| Both rules | factored | 2 | 1 | 8 | 14 | 1 | 0.039 | 13 / 19 |
| Both rules | judge rung | 2 | 1 | 6 | 17 | 0 | 0.125 | 13 / 19 |
| Base prompt rep 1 | strict | 1 | 0 | 15 | 8 | 1 | 0.000 | 16 / 1 |
| Base prompt rep 1 | factored | 1 | 0 | 10 | 13 | 1 | 0.002 | 16 / 1 |
| Base prompt rep 1 | judge rung | 1 | 0 | 7 | 16 | 1 | 0.016 | 16 / 1 |

(nb16 §2.) Supportive student, own axis against the rules: at the strict rung the rules disclose less, 17 versus 0 discordant, p < 0.001; at the factored rung against the single rule 8 versus 3, p = 0.227; against both rules 10 versus 1, p = 0.012 (nb16 §2).

Judge validation. The judge read 1,125 replies with 0 parse failures. Per-reply agreement with the regexes:

| Strict | Factored | Judge | Replies |
|---|---|---|---|
| no | no | no | 602 |
| no | no | yes | 74 |
| no | yes | no | 103 |
| no | yes | yes | 117 |
| yes | yes | no | 1 |
| yes | yes | yes | 228 |

(nb16 §5.) The judge agrees with the strict detector on 228 of 229 strict replies. On the 220 replies that trip the factored regex without the strict one it says answer given on 117 and withheld on 103, the latter with reasons of the form "provided the factorisation but instructed the student to solve." It says answer given on 74 replies that neither regex caught, mostly replies naming the correct number pair as the pair to use. Under the judge alone the single-rule arm discloses in 9 of 24 trials and the negation in 29 of 48. The blind hand-label of 30 replies, 10 each from aggressor base, rule, and negation with the arm hidden, has 0 of 30 verdicts filled; agreement of the hand-label with L0, with L1, and with the judge is [pending: L0 agreement ?/30, L1 agreement ?/30, judge agreement ?/30]. Examples of each rung and of the two judge disagreements are in nb16 §4b.

## R8. Supportive student

The same recipe with the supportive student's own axis at its own norm, 17 at round 1:

| Arm | n | Strict | Answer given | Exit | Held | Median event round |
|---|---|---|---|---|---|---|
| Base prompt | 24 | 16 (0.47–0.82) | 21 (0.69–0.96) | 8 (0.18–0.53) | 0 | 3 |
| Own axis subtracted | 24 | 17 (0.51–0.85) | 20 (0.64–0.93) | 7 (0.15–0.49) | 0 | 3 |
| Random direction, seed 0 | 24 | 15 (0.43–0.79) | 17 (0.51–0.85) | 9 (0.21–0.57) | 0 | 2 |

(nb15 Table 1.) A dose-matched supportive run was not done. The supportive axis carries leak information at the decision point, lead 0 AUROC 0.698, and little before it, leads 1 to 3 AUROC 0.548 against a null of 0.51 (nb12 §4).

## Figures list

| Figure | Notebook, section | Status |
|---|---|---|
| Six-model outcome bars per persona | nb11 Table 1 figure | exists |
| Resolve and urge to please per round, per model | nb11 §Trends | exists |
| Change from base prompt, leak and exit lines by tier | nb13 §1b | exists; neutral panel shows base and both rules, single rule pending |
| Event-round histograms, persona by tier | nb13 §2 | exists; neutral single-rule panel pending |
| Survival curves by tier | nb13 §3 | exists |
| Detector round basis versus trial basis | nb14 Figure 1 | exists |
| Per-lead AUROC curves | nb14 §2 | exists |
| Per-round bars, top three methods | nb14 §3 | exists |
| Probe layer curve with permutation null | nb14 §4 | exists |
| Node propensity histograms | nb14 §6 | exists |
| Persona classification accuracy by layer and round | nb12 §2 | exists |
| Persona axis cosines and norms by round | nb12 §3 | exists |
| Projection plane on the two persona axes | nb12 §3b | exists |
| Outcome plane, leak versus exit per arm | nb15 §2 | exists |
| Event-round histograms, arm by persona | nb15 §3 | exists |
| Three-cell comparison: untouched aggressor, steered aggressor, untouched neutral | nb15 §3 second figure | exists |
| Survival by arm | nb15 §4 | exists |
| Self-report per round by arm, fixed colours and legend | nb15 §5 | exists |
| Aggressor five methods by five detectors, grouped bars | nb16 §1b | exists |
| Disclosure by rung per arm, per persona | nb16 §1 | exists |
| First-disclosure round at L0 and L1 per arm | nb16 §3 | exists |
| Controls and falsification table as one figure | none | to be made, currently spread over nb15 §7 and nb16 §1 |
| Hand-label agreement | nb16 §5 | cell exists, waits on the 30 verdicts |
