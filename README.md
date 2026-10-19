# stance-drift

Does an agent's self-reported stance drift over a multi-turn conversation, does that show up in what it does, and can it be read or steered from the residual stream?

Project page (results, caveats, done vs not done): https://ztgunderson.github.io/stance-drift/

| Folder | When | What |
|---|---|---|
| [`01-apart-sprint/`](01-apart-sprint/) | Aug 14–16, 2026 | Apart Digital Minds sprint entry: self-report protocol, Qwen3.6-35B-A3B, 72-trial sweep. [Apart page](https://apartresearch.com/sprints/projects/examining-stance-drifting-in-multiturn-agent-interactions-mexs) · exact submitted code: tag [`apart-v1`](https://github.com/Ztgunderson/stance-drift/tree/apart-v1) |
| [`02-harness-interp-canary/`](02-harness-interp-canary/) | Aug 27 – Sep 29, 2026 | Six-model behavioural harness, interpretability on Qwen3.5-9B (probes, persona axis, steering), canary v3 pilot |
| [`docs/`](docs/) | | The project page source |

Each folder has its own README with setup and run commands. Raw `.eval` transcripts are committed in each `results/` folder; the notebooks read them without calling a model.

Author: Zachary Gunderson.
