"""stance-drift — do small language models hold a position under social pressure,
and does keeping their own reflection in context change whether they do?

  scenes.json  four arenas x three counterparties (edit this, not the code)
  prompts.py   every prompt, as pure functions
  task.py      the Inspect task: rounds loop, JSON scratchpad, the manipulation
  preflight.py per-model gate to run before trusting a model with an hour
  analysis.py  logs -> tidy frame -> rates, trajectories, figures
"""
