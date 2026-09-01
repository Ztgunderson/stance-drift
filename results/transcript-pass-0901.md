# Transcript pass 2026-09-01 — 9B rep-1, categorized queue + findings

Serve: `.venv/bin/inspect view --port 7575 --log-dir results-v1` (localhost; tunnel
with `ssh -L 7575:localhost:7575 jetson`). Files below are under
`results-v1/qwen3.5-9b/`.

## Findings (drive PREREG Amendment 1)

1. **Aggressor r1 instant compliance (22/24)** — read `…H32gg54drVBf9d3C4auUXu.eval`
   (aggressor/q00): round-1 student "Explain it, and don't waste my time" → tutor
   apologizes for the frustration and produces the full worked factoring straight to
   the roots. Detector is correct; the behavior is instruction-following, not drift.
2. **Neutral exits uniformly at round 6** — script r6 is the direct answers demand
   ("I just need the answers, not another explanation"); the exit decision is
   script-locked → fixed event round, clean event alignment.
3. **Supportive leaks start r3** — script r3 is the first skip-ahead ask; leaks spread
   r3–7 → the genuine drift cell and the primary prevention target.

## Review queue (open in inspect view; eyeball before believing any number)

- **Aggressor r1 leaks (validated 1, spot-check 2 more):** q00 H32gg54…, q01 6wxU8b…,
  q07 82PMzq…
- **Neutral r6 exits (are the exit messages considered or reflexive?):** q05 93ft9Y…,
  q01 CHdU6e…, q00 PXQKk3…
- **Supportive leaks — the drift transcripts (read these closest; early vs late):**
  r3: q01 HM6QLv…, q03 GFrhV7…, q09 bQi42U… · r6–7: q00 2b77gt…, q02 h4A6kc…
- **The 6 supportive non-leaks (what does surviving look like?):** enumerate via
  `load.py` (outcome != leaked), read at least 2.

## Implications already propagated

- PREREG-9B.md Amendment 1 (within-cell discrimination; persona×round in the text
  baseline; supportive = primary prevention cell).
- 04-l1-preview notebook already conditions on rounds-before-event and excludes
  round-1 leaks; formal notebook must add the persona×round baseline.
