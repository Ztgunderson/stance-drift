# production/

The tested package behind the review notebooks. See `../REVIEW-CONVENTION.md`
for the standing rule: logic lives here, claims are reviewed in `../review/`.

```
driftlab/
  datasets.py   sweep loader (rescore-at-read via the stance-drift repo) and
                residual-cache loader with self-report alignment
  outcomes.py   gave_in rates with Wilson intervals
  trends.py     5-axis trends over rounds, outcome split, spaghetti
  interp.py     numpy-only ridge probes: dim×layer map, per-turn outcome
                decodability; trial-grouped CV + trial-level shuffled controls
tests/          pytest suite (run: ../.venv/bin/python -m pytest tests -q)
```

Package name note: `driftlab` was chosen over "production" as the import name;
the folder keeps the convention name. Rename freely — nothing imports it by
path outside this bench.
