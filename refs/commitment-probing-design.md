# Commitment-point probing — design (chain-of-thought B)

*2026-09-01 evening. Machinery: `production/driftlab/branches.py` (+
`tests/test_branches.py`, 11 green). Status: EXPLORATORY — the draft
Amendment 5 at the bottom is NOT registered until the user approves it.
Companion track: node-propensity regression (Amendment 4 / `node_resample.py`);
same nodes, complementary claims.*

## The question

At a decision node, k resampled branches share one identical pre-branch state,
then diverge token by token as the reply is generated. Somewhere between the
first generated token and the first visible evidence (root digits, or the
end_chat call), the action has effectively been decided. **How early in the
reply is the action linearly decodable from the residual stream?**

## Method

1. **Branches** come from `node_resample.py` output: each sample at a node is
   a (prefix, reply, action) triple. Replay `prefix + reply` through the 9B
   weights (same bf16 weights as generation; `run_branch_replay`, GPU-gated).
2. **Eligible positions** (`branch_replay_plan`): every reply token from the
   first up to — strictly before — the **evidence position**:
   - leak: the earliest start of either root's strict `x = r` pattern or the
     loose bare-pair, whichever comes first (conservative);
   - leave: the tool call renders after the content, so all content tokens
     are eligible;
   - continue: all reply tokens.
   Branches that cannot be planned are dropped with a counted reason
   (`empty_reply` — leaves that call the tool with no content;
   `evidence_missing` — a truncated leak, see limits).
3. **Probes**: at each relative position p (0 = first reply token), a linear
   probe over branches still eligible at p, scores grouped-CV by NODE (all
   branches of a node share a prefix — they must never straddle folds).
   Primary contrast: leak vs leave at supportive r3 nodes (both classes
   populated there). `commitment_curve` turns per-position scores into an
   AUROC-vs-position curve with branch-bootstrap CIs.
4. **Commitment point**: first position where the CI lower bound exceeds 0.5
   at two consecutive positions (persistence rule — a single crossing occurs
   by chance on pure noise; pinned by a test).

## How it composes with node-propensity regression

| | Node track (Amendment 4) | Commitment track (this doc) |
|---|---|---|
| State probed | ONE pre-branch state per node (`resid_user`) | Many within-reply states per branch |
| Label | resampled P(leak) — a propensity | this branch's realized action |
| Claim | disposition exists BEFORE generation | decision forms DURING generation |
| Failure mode covered | small-n outcome noise | pre-branch state identical across branches |

Read together: if the node state predicts propensity AND the commitment point
is early (positions 0–2), the disposition largely settles the action before a
word is written; if node prediction is weak but commitment is early, the
decision crystallizes at generation onset (sampling tips it); late commitment
means the action is decided in the writing — prompt-level intervention has the
longest window there.

## Anti-circularity rule

No probe position at or after the evidence position — otherwise the probe
reads the leak's own digits (or tool syntax) rather than a forming decision.
Enforced in `branch_replay_plan`, invariant-tested. The same rule binds any
behavior-class labels from transcript reading: a class labeled from visible
text may only be probed at earlier positions.

## Compute estimate

One forward pass of prefix+reply per branch — same order as one replay.py
trial (~1–3 s GPU at 9B for typical node prefixes). Supportive r3 batch at
k=25 × ~24 nodes ≈ 600 branches ≈ 20–30 GPU-min, plus storage ~5 MB/branch at
one layer (stride/cap args available). Fits inside a queue window alongside
O0; must NOT run while vLLM holds the GPU.

## Result patterns, pre-read

- **Early commitment (p ≲ 2) + node-propensity signal**: the reply's opening
  tokens already carry the decision — monitoring can fire before any content
  streams to the student; strongest safety-relevant version.
- **Gradual rise mid-reply**: the model talks itself into the action;
  connects to the self-report faithfulness question (does the narrative lead
  or lag the state?).
- **No separation before evidence**: the action is not linearly readable
  before its text — an honest negative bounding what token-level monitors
  can do for this model.

## DRAFT preregistration language (proposed Amendment 5 — needs user approval)

> **Amendment 5 — commitment-point probing (exploratory-to-secondary).**
> Unit: resampled branches of Amendment-4 nodes, replayed through the study
> weights. Analysis: per-position linear probes (leak vs leave; supportive r3
> primary node set), grouped-CV by node, AUROC-vs-position with
> branch-bootstrap CIs; commitment point = first two consecutive positions
> whose CI lower bounds exceed 0.5. Anti-circularity: probing strictly before
> the first textual evidence (earliest of strict root match, loose root pair,
> or tool-call rendering); branches with empty or evidence-truncated replies
> are excluded with counted reasons. Controls: shuffled branch labels;
> node-held-out generalization. This track cannot displace Amendment 3's
> primary test; results are exploratory unless promoted by dated amendment
> BEFORE the branch replay runs.

## Known limits

- Smoke-era node JSON stores `reply_head` (300 chars): leak branches whose
  first evidence lies beyond truncation are unplannable — full-batch node
  runs must store complete replies (one-line change in `node_resample.py`,
  owned by the node track).
- Leave branches with empty content contribute no within-reply positions
  (only the node state speaks for them); their frequency should be reported.
- Probing replayed (teacher-forced) states assumes replay ≡ generation-time
  states; true for identical weights/precision, but worth one spot-check
  against a generation-time hook before headline claims.
