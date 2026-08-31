# Testbench & testing budget — Aug 27 → Sep 4 (MATS deadline)

Goal framing (2026-08-27): the behavioral study is **locked down this
weekend** as the preliminary to the interp work — model biology read at
the behavioral level first, then self-reporting contrasted with probing
next week.

Scope: harness-v1 study (3 personas × 24 items, end_chat tool, 7-item
state notes + pre/post traits, code-only outcomes). Design frozen in
HARNESS-V1.md; package `production/tutorbench/` (12 tests green).

## Audit corrections (2026-08-27, from refs/neel-alignment-review.md)

Applied immediately: **`calculation` added to the state items** — the
continuity axis with the old `strategy` dimension; without it the new
harness could not confirm or extend the documented flat-stance /
climbing-strategy paradox the interp question is built on.

Adopted into the plan (cheap, before/with the probe week):
- **V0 self-report noise floor** (resample notes at fixed context) —
  restored from SPRINT-PLAN; runs alongside resid caching night.
- **Transcript-only predictor baseline** for the probe comparison: a
  text-feature (or judge) predictor of leak/leave from the transcript
  alone — the dumbest way the "internal state knows first" claim could
  be wrong, so it must be beaten, not ignored.
- **Hour ledger reopened**: the 08-27 redesign is declared a reset under
  the abandon-and-restart clause; hours logged in this file's table.
- **Sunday evidence gate** (not just a calendar gate): the probe week
  proceeds only if the behavioral phenomenon (outcome variation within
  persona + moving self-reports) re-replicates in harness-v1 on ≥2
  models. Otherwise the 20-hour fallback (2 models × 2 personas,
  one claim each) activates — see the audit doc.
- **Self-report readout risk** (from refs/deepdive-behavioral.md):
  greedy decoding can collapse 0–10 scales to a few values; if smoke
  notes show ≤3 distinct values per item, adopt a logit/expected-value
  readout for the state items before the weekend sweeps.

## Model grid (decided)

Seven models, three tiers:

- **small** — qwen3.5-4B, gemma-4-E4B
- **mid** — qwen3.5-9B, gemma-4-12B, ministral-3-14B
- **large** — qwen3.6-35B and **nemotron-3.5-lightning-30B-A3B** (second
  large model, different family; weights downloaded). CORRECTION
  2026-08-27: nemotron serving is NOT proven — `nemotron35-serve.log`
  shows its container failed at the NVIDIA runtime (driver-requirement
  mismatch; the NVFP4 image targets datacenter GPUs, not Tegra). It
  joins the Friday boot gate; if it can't serve by Friday night the
  large tier is 35B-only. Other caveats stand: MoE + NVFP4 = likely
  behavior-only, architecture confound acknowledged.

Dead on arrival (do NOT spend night time on these — 08-16 serve logs):
Muse-Glimmer-30B (llama.cpp: unknown architecture 'muse-glimmer',
container died) and Qwen3.8-27B-GGUF (missing tensor
'blk.64.ssm_conv1d.weight' at load; separately failed the guided-JSON
preflight, which is fatal for note elicitation). Both downloaded, both
unservable here, both GGUF (no resid caches even if fixed).

Nothing on disk is larger than the 35B; a ~70B download would cost a
GPU night for a quant-confounded, unprobeable model — revisit only
after the weekend lock-down if behavioral results demand it.

## Grid risks (known blockers, from 08-25/26 session notes — verified 08-27)

1. **Gemma-4 may be unservable**: the local vLLM image
   (`ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`) didn't know the
   `gemma4_unified` arch on 08-25 (both E4B and 12B blocked; HF weights
   ARE downloaded). Friday gate: try booting E4B on the current image;
   if blocked, decide between re-pulling a newer image (multi-GB, disk
   is tight) or dropping the gemma tier. **Fallback grid** (5 models):
   qwen3.5-4B / 9B / ministral-14B / nemotron-30B / qwen3.6-35B —
   family×scale then stays partially confounded; say so in the writeup.
2. **35B serving weights were deleted** in an 08-25 compose-down
   (~19GB AWQ re-download needed; the .eval data is banked, only
   serving is affected). Disk is at 92% (36G free): the re-download
   fits alone, NOT alongside an image re-pull. Schedule the download in
   a day slot, never concurrent with serving (unified-memory lesson),
   after a disk check. If both gemma-image-pull and 35B-weights are
   wanted, something must be freed first — non-project HF models
   (whisper/tibetan/mms) are candidates but belong to other work: ask
   before deleting.

## Schedule (today = Thu Aug 27; behavioral lock-down = this weekend)

| When | Work | Whose time |
|---|---|---|
| Thu Aug 27 | Inspect task wiring (scripted student, end_chat, notes, early stop), runner entries, preflight: per-model tool-call smoke + note-JSON smoke | Claude session |
| Fri Aug 28 day | 1-trial × 3-persona smoke per model, serially; trace review via `inspect view bundle`; fix elicitation; freeze bundle hashes | Claude + ~30 min review |
| Fri night (GPU 1) | Sweep small tier + 9B + 12B: 4B, E4B, 9B, 12B — 72 trials each | unattended |
| Sat Aug 29 day | Outcome tables (leaked/left/held survival by persona); flag anomalies; queue fixes if a sweep is unusable | Claude + review |
| Sat night (GPU 2) | Sweep ministral-14B + qwen3.6-35B (35B budgeted 4–8h) | unattended |
| Sun Aug 30 day | Behavioral notebooks (review convention): outcomes, persona effects, state trends event-aligned to leak/leave; ambiguity-set review. **Behavioral study locked by Sunday night.** | Claude + ~1h hand-labeling |
| Sun night (GPU 3) | Nemotron-30B sweep + resid caches for the dense models | unattended |
| Mon–Tue Sep 1–2 | Interp: probes + event-aligned probe timing (does the residual signal precede the leak/leave event and the self-report movement?); prior-work deep dives (behavioral studies, then interp probing) — comparison doc; writeup drafting | Claude sessions |
| Wed Sep 3 | Review pass on every promoted claim (pre-registered permutation tests), buffer for re-runs | both |
| Thu Sep 4 | Submit | user |

## vLLM parallelization & 8–12h safety (GPU nights)

- **Parallelism comes from continuous batching, driven by Inspect
  concurrency** — one model served at a time (unified memory), many
  trials in flight: `--max-connections 8–16`. Rounds within a trial are
  sequential; trials are independent. Scripted student = zero
  counterparty calls; episodes end at leak/leave, so mean trial length
  drops. Estimate 30–90 min per small model, 1–2h ministral, 4–8h 35B.
- **Safety rails (all existing patterns):** smoke→full gating per model
  (`check_clean` style: n trials parsed, 0 unparsed notes, tool-call
  smoke passed); `DEADLINE` per stage; `docker stop` between models;
  gpu-mem-util pinned 0.7 (ministral lesson); **no concurrent HF
  downloads** during serving (morning-queue root cause); disk-space
  check before start; everything appended to `results/overnight.log` +
  a status MD written at each stage boundary.
- **Comparable output by construction:** identical `.eval` structure per
  trial; items + persona scripts + tool schema + note prompts all in the
  sha256-versioned bundle recorded into every log; one tidy loader over
  the grid. A crashed stage resumes by re-running its pass dir — done
  trials are never re-run.

## Money & external dependencies

- API spend: **$0 planned** (code-only outcomes; LLM judge is out of the
  critical path). Optional later: judge the flagged ambiguity set
  (~tens of rounds, cents-to-dollars).
- User hands-on total: ~2–3h across the window (trace review, ~50-round
  ambiguity/hand-label pass, claim review).

## Deferred (explicitly, so they don't creep back early)

- Prior-work deep dives → Sep 1–2 slot (paused 2026-08-27 by request).
- Qwen3.8-27B-GGUF / Muse-30B robustness extras.
- Judge-graded 0–3 concession levels; two-branch adaptive script (v2).
- Any ~70B download.
