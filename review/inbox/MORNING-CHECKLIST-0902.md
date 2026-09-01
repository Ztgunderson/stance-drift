# Morning checklist — Tue 2026-09-02 (submission Thu; Tue = L1 formal day)

## 1. Run health (2 min, before anything else)

- [ ] `results/MORNING-STATUS.md` — every line; any `ERROR:` → tell Claude first,
      read results second.
- [ ] Expected overnight artifacts: Q0 complete · notebooks 04+07 executed ·
      POST-Q0 complete (node batch k=25 → `results/nodes-supportive-k25.json`,
      reminder-arm replay → `microscope/cache/qwen35-9b-reminder-v1/` ~72 npz,
      notebook 08 re-executed with both arms) · O0 only if it was green-lit.
- [ ] Disk: `df -h /` — we were at 95%; caches added ~a few hundred MB.
- [ ] If the server misbehaved overnight: check for the zombie signature
      (docker hangs) BEFORE starting anything GPU.

## 2. Results, in reading order

- [ ] **Predictions first if still unfilled** — `predictions-0901.md`; once you
      open 04/07 the calibration is burned.
- [ ] **04-l1-preview** vs your predictions: per-layer AUROC — anything above
      the shuffled band at lead ≥ 1? Round-1-leak exclusion count sane?
- [ ] **07-layer-analysis**: leak-vs-exit decodable? Which position wins
      (`user` pre-decision vs `assistant` post-turn)? Persona-swap survive?
      Action-axis vs drift-axis cosine (two knobs or one?).
- [ ] **08 §6 both arms**: does the reminder arm reproduce the H1′ spread
      (identical sampled reports, spread logit E[v])? Baseline already did.
- [ ] **Node propensities k=25** (`nodes-supportive-k25.json`): leak/continue
      mix at r2 vs r3; did `leave` reappear at k=25 (the 4/10-vs-0/20 puzzle)?
      Propensity spread across nodes (the regression target's variance).
- [ ] Skim one or two fresh transcripts behind any surprising number
      (viewer: 7676) — §7.1 always.

## 3. Your open deliverables (from Monday)

- [ ] H0-1 + H0-2 readings in `results/2026-09-01-pilots.md` (yours before
      Claude's).
- [ ] `log-0901.md` end-of-day entry if not written (corrections already
      flagged: supportive DID leak 18/24 in baseline; script is 8 rounds).
- [ ] Probe exercise (`notebooks/02-probe-exercise.ipynb`) if not done — before
      the formal notebook lands, so your implementation stays an independent
      cross-check.

## 4. Decisions Tuesday needs from you

- [ ] **Amendment 5** (commitment-point probing) — draft language in
      `refs/commitment-probing-design.md`; approve/edit/defer.
- [ ] **O0 expansion** if it did not run overnight (supportive+neutral →
      48/cell; powers the formal L1).
- [ ] Anything in 04/07 that changes the Tue plan (e.g., if `user` position
      dominates, the formal notebook leads with it).

## 5. Tuesday plan of record (PLAN-9B-WEEK §6a2)

Formal `review/04-l1-probes.ipynb` (frozen Amendment-3 spec: round-conditional
negatives, nested layer freeze, text baseline WITH persona×round, logit-readout
rival, item + persona splits, lead-time AUROC with per-lead counts) → L2
directions + L3 trajectory figure → node-propensity regression join
(`driftlab/nodes.py` on k25 + cache) → B2 16-round scripts → 6 pm Q1 queue →
9 pm check.

## Standing cautions

Round-1 leaks excluded from all lead analyses (report the count) · supportive is
the primary cell · exit-round reports are POST-decision (T-1 is the predictive
reading) · nothing promoted past its prereg status · GPU jobs only on explicit go.
