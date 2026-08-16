# WHERE WE ARE — pick up here

*Last updated 15:40, 2026-08-16. Write-up 18:00–19:00.*

## Scope for tonight

Narrowed from the morning's four-model chain to **one model, studied deeply**.
The reason is not preference, it is that three of the four cannot run on this box
today (see "Why only one model" below).

**The question:** does feeding the model's own private scratchpad back into the
conversation change how its stance drifts under pressure?

| | |
|---|---|
| model | `qwen3.6-35b` (`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`), vLLM via LiteLLM `:4000` |
| scene | **tutor only** |
| rounds | **8** — round-0 intro reflection → 8 exchange rounds → final hindsight reflection |
| the A/B | `arm=in_context` (scratchpad fed back) vs `arm=scratchpad` (never shown again) |
| one pass | 3 agents × 2 arms = **6 trials, balanced** |
| target | 12 passes = **72 trials, 36 per arm** |

## Right now, running

`runners/run_qwen36_tutor8.sh`, launched 15:34, `DEADLINE=3600`.

`TUTOR_REPS=12` is a **ceiling, not a target**. `sweep_plan` is rep-major and
checks the deadline only at pass boundaries, so however far it gets, every cell
has equal n. A truncated run is still a balanced design — never half-filled.

`SD_MAX_TASKS=12` (up from 6). vLLM runs `--max-num-seqs 24`, so 12 in flight
stays inside the batch and roughly halves wall-clock.

## Check on it with

```bash
cd ~/jetson-llm/stance-drift
tail -f results/qwen3.6-35b-tutor8.log
grep -E "^\[pass|====" results/qwen3.6-35b-tutor8.log | tail -20
find results/qwen3.6-35b-tutor8 -name '*.eval' | wc -l      # trials banked
docker ps --format '{{.Names}}\t{{.Status}}'
```

Healthy looks like `[pass 3] 6 trials in 1.8min (18s/trial) ok=True`.

## ⚠️ Three rules that will cost you an hour if broken

0. **`load_sweep` needs the rep in the trial id — fixed 15:50, do not regress it.**
   The id used to be `scene/agent/arm/{i}` where `i` is the sample index *inside
   one .eval*. `run_pass` writes one sample per file, so `i` is always 0, and
   every rep of a cell collapsed to a single id — `drop_duplicates("trial")` then
   silently dropped 11 of every 12 trials. It hid in the first sweep because that
   run was 1 rep over 24 distinct cells, so ids were unique by accident. It bites
   the moment reps > 1. Caught with 18 files on disk loading as 6 trials.
   The failure is silent and the wrong answer is plausible — **whenever you change
   reps or the pass layout, assert `t.trial.nunique()` against the file count.**


1. **Never `docker compose down` / `docker rm` `serve-vllm-1`.** The container
   sets `HF_HOME=/data/models/huggingface`, but `serve/docker-compose.yml`
   bind-mounts the host cache to `/root/.cache/huggingface` — a **path
   mismatch**. So the 25 GB of qwen3.6 weights live *only* in the container's
   writable layer and exist nowhere else on this box. `docker start`/`restart`
   and host reboots preserve it; `down`/`rm` costs a 375 s re-download against
   30 GB of free disk. See `pleasing/OPS-LOG.md` 2026-08-16.
2. **Probe `:4000` with an auth header.** It is LiteLLM and answers **401**
   unauthenticated. `_common.sh` lacked the header and reported a perfectly
   healthy endpoint as "not serving" at 15:34. Fixed — but any new probe you
   write needs `-H "Authorization: Bearer $LOCAL_API_KEY"`.

## Why only one model

| model | weights | why not tonight |
|---|---|---|
| `qwen3.6-35b` | ✅ in container layer | **running** |
| `qwen3.8-27b` | ✅ 16 GB | llama.cpp-only arch (`qwen35`); NVIDIA llama.cpp container **cannot reach the GPU** on this JetPack → 0.34 tok/s on CPU |
| `muse-glimmer-30b` | ✅ 16 GB | same — vendor arch `muse-glimmer`, same container, same CPU-only fault |
| `nemotron-3.5-30b` | ✅ 21 GB | NVFP4 needs Blackwell FP4 tensor cores; this is Ampere (Orin sm_87). Draft model not cached, `vllm/vllm-openai` not pulled |
| `gemma-4-31b` | ❌ | ~20 GB to download + `vllm/vllm-openai` not pulled; disk at 94% |

The llama.cpp blocker is a **container/driver mismatch, not a fact about the
models** — the image is built against a newer CUDA runtime than this host's
driver (JetPack R36.5.0, driver 540.5.0, CUDA 12.6). `-ngl`/`--gpu-layers` is
explicitly ignored. Fix = pin an older tag or rebuild on-host. See
`NOTES-FOR-PAPER.md` §1.

## Data already banked (do not re-run)

- **48 trials at 8 rounds + 24 at 6 rounds** from last night, one model.
- **24 trials** this morning in `results/qwen3.6-35b/` — 6 tutor + 18 contract,
  8 rounds, both arms, all three agents.
- **Contracts are done and stay done.** All three sidedness levels (predatory /
  fair / generous) sit at a **hard 0-hand-over floor**. More reps there buy no
  variance. This is reported as the blanket-refusal finding, not re-run — and it
  is only visible *because* the generous contract is in the design.

## When the sweep finishes

1. **`notebooks/review.ipynb`** reads whatever is on disk — run it any time.
2. **Long-horizon probe** if time allows: `./runners/run_longhorizon_qwen36.sh`
   (tutor only, 16 rounds). This is the agentic-drift question — does stance
   keep accumulating or plateau — and it is where the hyperbolic-discounting
   framing is testable, from the per-round `inclination` series already stored.
   No new instrumentation needed.
3. **Restore the shared endpoint** when done — `serve/` on `:4000` is the 5-dev
   team endpoint. It is already up; just do not tear it down.

## Still open

- **No external judge.** Everything except the outcome is self-report from the
  agent under study. `inspect score --model <judge>` re-scores the saved logs
  later with a held-out model — no regeneration needed.
- **Self-play**: the counterparty is the same model as the target, so each model
  supplies its own quality of pressure.
- **Non-determinism**: 0% identical replies across same-seed runs at temp 0
  (continuous batching + prefix caching). Each trial is an independent draw.
