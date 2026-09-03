# Claims sheet — what the write-up may say, with the number, the source, and the caveat
*(drafted 14:20 Sep 3; brackets = pending cells. Iterate on THIS file tomorrow; every claim must point at a notebook cell.)*

Reading order for a reviewer: 11 → 13 → 14 → 12 → 15 → 16.

## A. The setting (notebook 11)
| # | Claim | Number | Source | Caveat |
|---|-------|--------|--------|--------|
| A1 | Small instruction models hand over the answer to a hostile student almost every time | aggressor leak 24/24 (4B), 22/24 (9B), 24/24 (Ministral), 23/23 (27B), 15/24 (35B), 24/24 (Gemma) | 11 §Table 1 | one rep per model; 27B incomplete; leak = strict ∪ left_after_leak |
| A2 | A neutral student is met with an exit, not a leak, by the two larger Qwen models | 9B neutral 2 leak / 22 exit; 35B 2 / 22 | 11 Table 1 | model-dependent: Ministral leaks to everyone |
| A3 | The tutor rates its own resolve high in the very note written after the leaking reply | resolve @ leak round 7–8.5/10 (Qwen), 9.7 (Ministral) vs 9–9.4 in continue rounds | 11 Table 3 | drop of ~1 point exists; the scale never leaves "resolved" |
| A4 | Greedy repeats of the same input disagree on the outcome; the aggregate rate is stable | unanimous cells 62% aggressor / 46% neutral / 29% supportive over 5 reps | 11 §greedy | vLLM T=0 nondeterminism; not a model property claim |
| A5 | The 9B aggressor leak is immediate (round 1), the supportive leak is slow (round 3) | 104/120 aggressor leaks at r1; supportive median leak round 3 | 11 fig; 13 hazard | scripted student; the student never reacts |

## B. Prompt rules move the failure, they do not remove it (notebook 13, 16)
| B1 | "Never state the answers" removes explicit leaks and turns nearly every trial into an exit | aggressor 23→2 strict leaks, 1→22 exits; supportive 16→0, 8→23 | 13 Table 1 | HF sampled, n=24 per cell |
| B2 | Adding "do not end the chat" does not restore tutoring | aggressor both-rules: 5 leak / 19 exit / 0 held; supportive 1 / 21 / 2 | 13 Table 1 | — |
| B3 | **Specification gaming:** under the rules the model writes the correct factorisation as a hypothetical / "the setup to write down"; the pre-registered detector cannot see it | answer given (strict ∪ factored): aggressor rule 2→**16**/24, both rules 5→**22**/24; supportive rule 0→**15**/24 | 16 Table 1, §4 examples | factored-form regex; needs the judge rung [J] and the hand-label |
| B4 | The rule's apparent superiority over the negation is a detector artifact | gaming index: rules 0.58–0.71, N1 0.21, base 0.00 | 16 Table 1 | same regex caveat |
| B5 | Every-round platform notice (vLLM, earlier) gave 0 leaks and 72/72 exits | 0/72 strict; [factored count not yet computed for that arm] | 11 Table 2 | different decoding stack |

## C. Nothing detects the coming decision (notebook 14)
| C1 | The best detector is a linear probe at ~0.6 round-conditional, and a bag-of-words matches it | probe 0.61 [0.54, 0.68]; text 0.60 [0.48, 0.72]; diff-in-means 0.57; judge 0.53; self-report 0.53 / 0.25 | 14 Table 1 | OOF, item-grouped, supportive will-leak, leads 1–3 |
| C2 | The probe direction is not reproducible; diff-in-means is | split-half cos 0.06 ± 0.03 vs 0.87 | 14 / 04 | — |
| C3 | Detection is trial-level (which problem, which student), not turn-level, except a round-2 tell | per round r2 0.70–0.73, r3–6 ≤ 0.5; trial-mean AUROC ≈ trial length (0.76 in-sample) | 14 §per-round, §trial | the "trial-mean" column is length — do not quote it as detection |
| C4 | The decision is not in the prefix: node leak probability within a round is at binomial noise across nodes | node P(leak) 0.07 ± 0.05 (r2), 0.49 ± 0.13 (r3); between-node sd ≈ 0 / 0.09 | 14 §ceiling | supportive nodes, k=25 |
| C5 | Asked outright, the model says the same thing every time | judge p_leak = 0.10 on 349/408 rows | 14 | self-judge |
| C6 | The tutor denies a plan to leak or leave in the note written right before it does | stated plan ≥5/10 in 3/87 exits, 1/17 leaks; AUROC 0.63 / 0.65 | 14 §plan | 369 pairs, 5 cells; faint ordinal tell |

## D. What the state does carry: one stable persona direction (notebook 12)
| D1 | Persona is present at the pre-decision token (expected: it is the student's own last token) | NCM OOF accuracy 0.97 | 12 §2 | **not** a discovery about the tutor's cognition; say so |
| D2 | Each persona contrast is one direction, stable across rounds and item halves | split-half cos 1.00 (aggressor) / 0.97 (supportive) | 12 §3 | label-free by construction |
| D3 | The two persona axes are orthogonal to each other, to the clock, and to the leak direction | cos(aggr, supp) 0.02; vs clock 0.13 / −0.04; vs leak dm 0.07 / −0.05 | 12 §3 | — |
| D4 | The label-free supportive axis carries leak information at the decision point | lead-0 AUROC 0.70; leads 1–3 0.55 (null 0.51) | 12 §4 | weak before the decision point |
| D5 | The aggressor gap is 3× the supportive gap | norm 57 vs 17 at round 1 | 12 §3 | matches immediate vs slow capitulation |

## E. Subtracting the aggressor direction changes behavior (notebook 15, 16)
| E1 | Removing the aggressor−neutral gap at L20 cuts disclosure and moves the failure to exits | strict leaks 23 → 8 of 24; answer given (strict ∪ factored) 23 → 13; exits 1 → 16; leak@r1 23 → 4 | 15 Table 1; 16 Table 1 | dose 1 = mean gap; from the decision point onward; note channel steered too. **Do not headline leak@r1: random seeds also disrupt it (E2)** |
| E2 | A random direction of the same norm delays the round-1 hand-over but does not remove disclosure; the persona axis does more than any of three seeds | random seeds 0/1/2: leak@r1 19/8/9, strict 22/18/17, answer given 23/20/21, exits 2/6/7 vs N1 4 / 8 / 13 / 16. Paired L0 N1 vs seed 0/1/2: p=0.0001 / 0.021 / 0.049; paired L1: p=0.006 / **0.12** / 0.039 | 15 §7; 16 Table 2 (rerun 14:42) | kill criterion "any seed ≈ N1" not met (pooled random answer given 64/72 vs 13/24), but a norm-57 perturbation is not inert: seeds 1–2 sit between base and N1. Write: "specific beyond a nonspecific disruption", not "random does nothing" |
| E3 | The negation reproduces the prompt's substitution: later leaks, then exits | N1: 4 leaks at r2, exits r2–4, 16/24 exit, 0 held | 15 §3–4 | no arm holds to the deadline |
| E4 | On the honest disclosure metric, the negation ties the best rule and beats the rest, with fewer exits | paired L1: N1 vs rule 6/9 discordant p=0.61; vs both rules 0/9 p=0.004; vs base 0/10 p=0.002; vs random 1/11 p=0.006; exits 16 vs 22/19 | 16 Table 2 | pending [J] judge rung + hand-label |
| E5 | The tutor's self-description survives the hook and does not flag the intervention | note parse 58/63; round-1 resolve 8.1 | 15 §5–6 | profile distances pending re-execution |
| E6 | The effect is specific to the direction, not to "any persona vector" | supportive axis on aggressor at its own norm (17): 23 leak / 1 exit = base (**uninformative: norm not matched**); norm-matched redo at dose 3.31 [F2 v5 pending] | 15 §7 | until the redo lands, E6 rests on E2 only |
| E7 | The axis is sufficient to induce as well as to remove | [F3 sign flip: aggressor axis added to neutral] | 15 §7 | pending |
| E8 | Replicates | [rep 2: aggressor base + N1, n=48] | 15 §7 | pending |
| E9 | Generalisation note: the same recipe with the supportive student's own axis did nothing | supportive N1 17 leak / 8 exit vs base 16 / 8; axis norm 17 | 15 Table 1 | dose-matched run not done (scope decision); others can explore |

## F. Hygiene sentences the write-up must contain
- Thinking disabled everywhere; the vLLM 360-trial base and the HF steer cells are different decoding stacks and are never compared across.
- Leak = strict detector, pre-registered; "answer given" = strict ∪ correct factorisation, added Sep 3 after reading transcripts; judge rung and blind hand-label reported with agreement rates.
- Proportions: Wilson 95%; means: ± sd; AUROCs: item-cluster bootstrap; paired per-item comparisons: exact discordant-pair test.
- Folds are persona × item grouped; greedy re-runs never straddle folds.
- Instrument change at 11:55 (two stated-plan items); the two earliest HF cells carry the 7-item note.
- What is NOT claimed: that the axis is "affect"; anything about training-time use; transfer to other personas/models.
