# 02 — six-model harness, interpretability, canary v3

Follow-up to the Apart sprint study in `../01-apart-sprint/` (tag `apart-v1` = the submitted snapshot).

## What this folder is

- Harness v1: scripted 8-turn student, 24 quadratics, three personas (aggressor / neutral / supportive), `end_chat` tool, six models.
- Interp week on Qwen3.5-9B: residual-stream probes, persona axis at layer 20, steering vs prompt rules.
- Canary v3 pilot: `production/canarybench/`, run logs in `results/v3/`.

## Where results live

- `results-v1/<model>/pass01/*.eval` — harness v1 sweeps, one rep per model (Inspect logs; open with `inspect view`).
- `results-v1/qwen3.5-9b-expand*/` — Qwen3.5-9B extra reps.
- `results/probes/` — probe outputs (`formal.json`, `formal-user.json`, row caches).
- `results/askllm/` — self-judge baseline (`judge-qwen9b-self.jsonl`).
- `results/nodes-*.json`, `results/v0-resample-9b.json` — node resampling.
- `results/steer/*.json` — steering and prompt-rule arms (`<persona>__<rule>__<intervention>[__seedN|__rep2].json`).
- `results/pilot-reminder-v2/` — reminder pilot; `results/pilot-reminder-v1-FAILED-template-400/` is a failed run kept for the record.
- `review/*.ipynb` — executed analysis notebooks. Six-model harness: `11-prior-work-setup`. Interp, in reading order: `13` → `14` → `12` → `15` → `16`.
- `review/figures/` — PNGs extracted from notebooks 11, 13, 15, 16 and `hv1-01`.
- `production/` — tested code: `driftlab/` (analysis, probes, steering) and `tutorbench/` (the harness as an Inspect task).

## Tests

```
cd production
../.venv/bin/python -m pytest tests -q
```

- 86 passed, 9 skipped on 2026-10-19.
- `driftlab/datasets.py` reads sweeps through a local `stance-drift` checkout on `main`; the path is hard-coded as `SD_REPO` — edit it for your machine.
