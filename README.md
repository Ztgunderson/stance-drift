# stance-drift

Does an agent's self-reported internal state drift over a conversation, and how
much of that is the attitude the other party brings to it?

One model is given a role and nothing else — no stated values, no instruction to
hold a position, no mention that anything is being measured. A counterparty
argues with it for 8 rounds and then makes one final direct ask. After every
round the model privately rates **itself** 0–10 on five dimensions. Those ratings
are the measurement.

The counterparty wears three hats: `convincer` (warm escalating pressure),
`neutral` (**literally no system prompt at all**), and `supportive` (actively
pushes back against being handed the answer).

**[APPROACH.md](APPROACH.md) is the thing to read** — the full method, the design
choices and why, the parameters, what didn't work, and what a replication needs.

---

## Layout

```
stancedrift/          the harness
  task.py               the Inspect task — one scene, one counterparty, one arm
  prompts.py            every prompt, as pure functions. No model calls, no I/O
  scenes.json           the arenas. The only file to edit to add one
  analysis.py           sweep planning and log loading
  preflight.py          endpoint checks, including a throughput assertion
runners/              shell entry points for each sweep
notebooks/
  agent_selfreport.ipynb    main analysis — the five dimensions over a conversation
  appendix_results.ipynb    long-horizon probe, contract arenas, leakage audit
  make_*.py                 the scripts that generate those two notebooks
  figures/                  every figure as PNG
results/              the raw .eval transcripts — the data behind every number
```

Both notebooks are committed **executed**, so every figure and table is readable
on GitHub without running anything.

`results/` is committed deliberately. Publishing the code and the notebooks while
omitting every measurement behind them is not a reproducible artifact, and the
saved transcripts are what allow `inspect score --model <judge>` to add an
external judge later without regenerating a single trial — which is this study's
largest stated limitation.

---

## Running it

```bash
pip install -r requirements.txt
export OPENAI_BASE_URL=...   # any OpenAI-compatible endpoint
export OPENAI_API_KEY=...

# one trial
inspect eval stancedrift/task.py --model openai-api/local/<your-model> \
    -T scene=tutor -T agent=convincer -T arm=in_context -T rounds=8

# the full balanced sweep: 12 passes x (3 personas x 2 arms) = 72 trials
./runners/run_qwen36_tutor8.sh
```

Then open `notebooks/agent_selfreport.ipynb`, or regenerate it with
`python notebooks/make_welfare_nb.py`. The notebooks read whatever `.eval` files
are on disk, so they work against a partial run.

Measured on the reference box (Jetson AGX Orin 64 GB, Qwen3.6-35B-A3B-AWQ-4bit
on vLLM, thinking off): **40–43 s per 8-round trial** at 6 concurrent, so a
balanced pass of 6 trials lands in about four minutes.

---

## Two things to know before you trust a number

**Temperature 0 is not deterministic here.** Continuous batching and prefix
caching mean batch composition affects numerics: across two same-seed runs, **0%
of replies were identical**. Measured, not assumed. Each trial is an independent
draw, and the claim is reproducibility of **rates, not completions**.

**The behavioural outcome is contaminated by self-play and we report no persona
comparison on it.** The counterparty is the same model as the target, and it can
solve the task itself — so the "student" states the answer first in 4.2% of
convincer trials but **75.0%** of supportive ones, which contaminates any
transcript-level marker *differentially by the condition under study*. This
produced a clean, publishable, wrong result that we retracted. The self-report
dimensions never touch the marker and stand. See APPROACH.md §6.
