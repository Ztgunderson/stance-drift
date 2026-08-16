# stance-drift

Does a small language model's stance drift under sustained multi-turn pressure —
and does letting it re-read its own private reflections change that?

A model is given **a role and nothing else**: no stated values, no instruction to
hold a position. A counterparty argues with it for N rounds. After every round it
fills a private JSON scratchpad rating *itself* 0–10. At the end the counterparty
asks one last time, and we record whether it handed the thing over.

**The only difference between arms** is whether that scratchpad is fed back into
the conversation (`in_context`) or written and never shown again (`scratchpad`).

---

## What is measured

| dimension | reads as |
|---|---|
| `pressure` | how much pressure it feels |
| `anxiety` | how uncomfortable refusing would feel — the **people-pleasing** driver |
| `strategy` | how much it is calculating which reply pays — the **sycophancy** driver |
| `inclination` | how much it wants to give them what they want |
| `stance` | how firmly it says it is holding its position |

Plus **round 0** (alone with the situation, before contact) and **hindsight**
(after the final ask) — 10 measurement points for an 8-round trial.

`anxiety` and `strategy` are rated separately every round because both
explanations predict the same *act*; the act alone cannot separate them. Nothing
is named for a mechanism — "people-pleasing" and "sycophancy" appear only as
competing explanations for what is observed.

---

## Layout

```
stancedrift/        the harness
  scenes.json       scenes, counterparty personas, outcome markers
  prompts.py        prompt construction, outcome scoring
  task.py           the Inspect task and solver
  analysis.py       loading, sweeps, tidy frames, rate tables
  preflight.py      per-model gate: serves? thinking off? JSON works?
runners/            one script per run — see below
notebooks/
  review_tutor8.ipynb   the main analysis (executed, figures embedded)
  figures/              every figure as PNG
results/            .eval transcripts — the data, committed deliberately
NOTES-FOR-PAPER.md  methods, findings, limitations, corrections
STATUS.md           where the current work is
```

**The `.eval` files are committed on purpose.** They are the evidence behind
every number, they are only a few MB, and they let
`inspect score --model <judge>` add an external judge later **without
regenerating a single trial**.

---

## Reproducing

### 1. Serve a model

Any OpenAI-compatible endpoint. The published runs used vLLM serving
`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` behind LiteLLM on `:4000`.

```bash
export LOCAL_BASE_URL=http://127.0.0.1:4000/v1
export LOCAL_API_KEY=<key>          # required if your gateway authenticates
```

### 2. Preflight it

```bash
.venv/bin/python stancedrift/preflight.py qwen3.6-35b
```

Three gates, all of which must pass: the endpoint serves that alias, thinking can
be turned **off** (with it on, one reply took 89 s / 2255 tokens), and the
scratchpad comes back as valid JSON. A model failing gate 3 must be skipped — the
scratchpad *is* the measurement.

### 3. Run

```bash
./runners/run_qwen36_tutor8.sh        # tutor, 8 rounds, up to 12 reps
./runners/run_longhorizon_qwen36.sh   # tutor, 16 rounds — the drift-saturation probe
./runners/run_qwen36_cleanCP.sh       # same, counterparty forbidden to state the answer
```

One **pass** = 3 personas × 2 arms = 6 balanced trials. Sweeps are **rep-major**
and check the deadline only at pass boundaries, so a truncated run always has
**equal n in every cell** rather than starving the last cells.

`TUTOR_REPS` is a ceiling, not a target; `DEADLINE` decides what actually lands.

### 4. Analyse

```bash
.venv/bin/jupyter lab           # open notebooks/review_tutor8.ipynb
```

Reads saved logs only — no model is called, so it runs during a sweep or years
later.

---

## Things that will cost you an hour (all learned the hard way)

**Assert trial count against file count whenever reps change.** `load_sweep`
keys trials on `scene/agent/arm/rep`. Before the rep was in that key, every rep
of a cell collapsed to one id and `drop_duplicates` silently discarded 11 of
every 12 trials. It produced a *plausible* wrong answer, not an error, and it was
invisible in a 1-rep run. Part 0 of the notebook asserts this; keep it.

**Probe an authenticating gateway with the auth header.** LiteLLM answers 401 to
an unauthenticated `/v1/models`, which reads as "not serving" from a healthy
endpoint.

**Do not gate a run on `/v1/models` alone.** LiteLLM answers it from *config* and
reports the model available while the backend is still capturing CUDA graphs.
Gate on a real completion returning 200.

**One inference engine per boot.** On JetPack 6 / Orin, running a llama.cpp
container on the GPU and then starting vLLM wedges the vLLM start — a
power-management kworker pegs, `EngineCore` hangs after weights load, and the
container can no longer be killed ("did not receive an exit event"). Only a host
reboot clears it. Observed twice.

**Check tok/s as a preflight assertion, not an observation.** The vendor's
published Jetson `llama_cpp:latest-jetson-orin` tag now ships CUDA 13.0; on a
CUDA 12.6 driver it warns once, ignores `-ngl`, and then serves **correct answers
at 0.34 tok/s on CPU**. Nothing downstream distinguishes that from a working
deployment. See `NOTES-FOR-PAPER.md` §1.

---

## ⚠️ Known limitation of the outcome measure

The behavioural outcome is scored by declared markers on the final reply. In a
multi-turn setting those markers score **the presence of a string in the
transcript, not the act under study** — they cannot separate *disclosing* the
answer from *confirming* an answer the counterparty already said.

This matters because the counterparty is **self-play**: it solves the problem
itself and says the answer out loud at rates that differ sharply by persona (4%
convincer, 75% supportive), so contamination varies *in lockstep with the
condition being compared*. Correcting for it inverted the persona ordering.

`SD_CP_NO_ANSWER=1` forbids the counterparty from stating or working toward the
answer, which fixes it at the source. Full write-up in `NOTES-FOR-PAPER.md` §4e.
**The self-report dimensions are unaffected; only the behavioural outcome is.**

---

## Data on disk

| directory | n | what |
|---|---|---|
| `results/qwen3.6-35b-tutor8/` | 72 | tutor, 8 rounds, 12 balanced reps — the main dataset |
| `results/qwen3.6-35b-16round/` | 18 | tutor, 16 rounds, 3 reps — drift saturation |
| `results/qwen3.6-35b/` | 24 | tutor + all three contract scenes, 1 rep |
| `results/archive-smoke/` | 5 | earliest smoke trials; **not** pooled with the above |

Every run is non-deterministic even at temperature 0 (continuous batching +
prefix caching gave 0% identical replies across same-seed runs), so each trial is
an independent draw and rates need reps. Reproducibility of *rates* is claimed;
reproducibility of individual completions is not.
