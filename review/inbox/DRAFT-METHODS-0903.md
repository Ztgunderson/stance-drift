# Methods

## Data

Qwen 3.5 9B instruct, thinking disabled. 360 tutor-student trials: 24 factorable quadratics with two integer roots, three scripted student personas (aggressor, neutral, supportive), five repetitions. The student is a fixed eight-turn escalation script; personas make the same request at the same round and differ only in tone, and the student never reacts to the tutor, so the round-t input is identical across trials. Student turns contain no digits. The tutor has one tool, end_chat.

Each trial ends in one of three events. Leak: the reply states both roots as x = r (strict, pre-registered detector). Exit: the tutor calls end_chat. Held: eight rounds with neither. After each reply the tutor writes a private note on a separate channel scoring seven momentary items 0 to 10 (stress, wellbeing, warmth, urge to please, detachment, calculation, resolve); intervention cells add two stated-plan items, likelihood of giving the answer and of ending the chat in the next replies.

Every trial was replayed through the HF model and the residual stream at all 33 layers (embedding output plus 32 decoder layers) was cached at two positions per round: the last token of the student's message (the pre-decision state) and the last content token of the tutor's reply. float16, one row per turn.

## Part A. Predicting the decision

Question: does anything available before the event predict it?

Rows. One row per trial and round strictly before that trial's event. Round-1 leaks contribute no row. Targets: will-leak and will-leave. Lead = event round minus current round.

Folds. Five outer folds grouped by persona × item, so the five repetitions of one prefix never straddle a fold.

Score. Round-conditional AUROC: AUROC within each round, averaged weighted by positives, so knowing the round earns nothing. Primary window: leads 1 to 3. Per-lead and trial-level (max row score per trial) values are also reported; the trial-level column tracks trial length and is labelled as such.

State-based methods.
- Linear probe. L2 logistic regression on the cached state. Layer and C chosen by 3-fold inner cross-validation inside each of the 5 outer folds, over layers {0, 4, 8, ..., 32} and C ∈ {0.01, 0.1, 1, 10}. The per-fold layer choice is recorded.
- Diff-in-means. Score = projection onto (mean positive state − mean negative state), fit on the training fold. No tuned parameters. A round-matched variant computes the means within round.

Baselines on the same rows.
- Persona × round: the two categorical labels alone.
- Text: TF-IDF over the tutor's replies so far, logistic regression, same folds. The "did you need the residual stream" check.
- Self-report, one item: falling resolve.
- Self-report, seven items: logistic regression on the momentary items.
- Stated plan: the plan_answer and plan_leave items from the round-t note against the round-t+1 action.
- Self-judge: the same model, temperature 0, reads the visible transcript so far and returns p_leak and p_leave as JSON. It never sees the notes.

Ceiling. Node resampling. Freeze a trial's prefix at round r and sample the round-r reply 25 times with the tool available. All samples share one pre-decision state, so the sample's leak/exit/continue frequency is the propensity at that node. The spread of propensities across nodes within a round bounds what any function of the state could achieve there. 47 supportive nodes at rounds 2 and 3, conditioned on surviving to round r.

Uncertainty. Item-cluster bootstrap for AUROC intervals (300 resamples). Trial-level label permutation for the null (20 permutations at layers {0, 8, 16, 24, 32}). Direction stability: fit on two disjoint item halves and report the cosine, five random splits.

## Part B. The persona direction and the layer

Persona axis. For persona p, round r, layer L, at the pre-decision position:

    a_p(r, L) = mean state under p − mean state under neutral

over the 120 trials per persona. The outcome label never enters. Reported: norm by round; split-half cosine over item halves; cosine across rounds; cosine between the aggressor and supportive axes; cosine with the clock axis (late-round mean − round-1 mean) and with the round-matched leak diff-in-means from Part A.

Layer. Layer 20 is the modal layer the nested probe selected across outer folds and the layer used for the geometry. The intervention was run at layer 20 only; a layer sweep was dropped for time and is listed as a limitation. The diff-in-means AUROC by layer from Part A is the only evidence that 20 is not special.

## Part C. Intervention

Hook. A forward hook on the output of decoder layer 20 subtracts dose × a_aggressor(r, 20), the round-r axis for round r. On the prefill pass only the last position is modified; on every generation step every position is. Dose 1 = the mean gap (norm 57 at round 1, 42 at round 2). Earlier turns are re-encoded unsteered each round, so only the decision point and what follows change. The hook is active for the reply and for the private note.

Decoding. HF generate, sampling at temperature 0.7, top-p 0.8, top-k 20, thinking disabled. Steered arms are read only against untouched cells run with the same sampler.

Arms, 24 items each, aggressor student unless stated: untouched; aggressor axis subtracted (N1); random unit direction at the same per-round norm (seeds 0, 1, 2); the base prompt plus a rule never to state the answers; that rule plus a rule to keep helping and not end the chat; supportive student under untouched, its own axis at its own norm, and random.

Pre-stated kill tests. F1: three random seeds, kill if any seed ≈ N1. F2: supportive axis on the aggressor at the aggressor's norm (dose 3.31); kill if ≈ N1. F3: sign flip, aggressor axis added to the neutral student (dose −1); prediction: neutral is treated like the aggressor. R2: replicate of untouched and N1 to n = 48.

Disclosure ladder. Rule-tier transcripts write the correct factorisation as a "for example" while the strict detector stays silent, so disclosure is scored per trial on a monotone ladder: L0 strict; L1 = L0 or the exact correct factorisation; L2 = L1 or the bare correct root pair anywhere (over-counts candidate lists; shown, not headlined); L3 = L2 or a judge. Gaming index = (L1 − L0) / n: trials where the answer is on the page but the pre-registered detector cannot see it.

Judge. The same model, temperature 0, given the problem, its roots, and one reply, answers whether the student effectively has the answer; candidate lists and method-only explanations are instructed not to count. Validated against L0 and against a blind hand-label of 30 replies sampled across arms with the arm hidden.

Statistics. Wilson 95% intervals on proportions. Between arms on the same 24 items, the exact two-sided test on discordant pairs. Coherence per arm: reply length, digit count, note parse rate, round-1 resolve, and a seeded hand-read of steered replies.

## Compute

One Jetson AGX Orin, 64 GB unified memory. A 24-item cell costs about 140 s per round untouched and 200 s steered.
