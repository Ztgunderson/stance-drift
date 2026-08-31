# The review/production convention

Standing requirement (2026-08-27) for how code is built and reviewed in this
lab, starting with this bench and applying to all future work.

## Two folders

```
production/     the package (here: driftlab) + tests/. Every piece of logic a
                claim rests on lives here, unit-tested. Notebooks never define
                logic — they import it, so what is reviewed is what runs.
review/         Jupyter notebooks that review production/ section by section.
```

## What a review notebook must contain

For **every claim** it makes: **Methods / Results / Discussion**, plus:

- **Intent & expectations** up front: what the code is for, what claims the
  notebook establishes (numbered C1, C2, …).
- **The code in runnable cells**, reviewable line by line, with its **unit
  tests executed in an adjacent cell** (`pytest` run inline, output visible).
- **Visualize, visualize, visualize**: print tensor/DataFrame shapes and
  sizes at every load; matplotlib for every result; the honest per-trial
  view (spaghetti) behind every mean curve.
- Controls next to results (shuffled-label twin heatmaps, Wilson intervals),
  and confounds stated in Discussion, not footnotes.

## Serving

`./serve-review.sh` serves JupyterLab bound to the Tailscale IP only
(port 8890, token in `.jupyter-token`, log in `results/jupyter-review.log`).
Kernel = the bench venv, registered as `mats-bench`.

## Current contents

- `01-data-and-outcomes.ipynb` — loaders, rescore-at-read, gave_in
  true/false per model × counterparty with Wilson CIs (C1–C2).
- `02-selfreport-trends.ipynb` — the five self-report axes over rounds,
  split by outcome, faceted by agent (C3–C4).
- `03-interp-bridge.ipynb` — ridge probes residual→self-report per layer,
  and per-turn outcome decodability, each with trial-level shuffled
  controls, qwen3.5-9b + ministral-3-14b (C5–C6).
