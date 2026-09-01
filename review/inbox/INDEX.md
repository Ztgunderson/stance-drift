# Review inbox — open in JupyterLab, newest first

*Your one stop each review checkpoint (9 pm / morning). Every item is a relative
link that opens in this JupyterLab. Convention: sessions append dated entries at
the top; check items off by striking them through.*

Open this file at:
http://100.76.200.13:8890/lab/tree/review/inbox/INDEX.md

## 2026-09-01 — session 1 (scoping + pilots)

- [ ] **[PREREG-9B.md](../../PREREG-9B.md)** — the six nulls + Amendments 1 & 2.
      Read the amendments; they changed the claim (within-cell discrimination) and
      added the item-split control after the final Neel pass.
- [ ] **[PLAN-9B-WEEK.md](../../PLAN-9B-WEEK.md)** — §6a2 is the live schedule
      (6 pm queue / 9 pm review / overnight; Thu = your 2 h writing).
- [ ] **[Transcript pass queue](../../results/transcript-pass-0901.md)** — the
      trials to read in inspect view (tunnel: `ssh -L 7575:localhost:7575`, then
      http://localhost:7575). Supportive drift transcripts are the priority.
- [ ] **[Pilots results](../../results/2026-09-01-pilots.md)** — P-1b (every-round
      reminder) and P-1a (V0 resampling) tables land here when the runs finish;
      read against H0-1 and H0-2.
- [ ] **[04-l1-preview.ipynb](../04-l1-preview.ipynb)** — executes tonight (Q0);
      at 9 pm this is the first look at H0-3/4/5. Morning checklist is in its
      first cell.
- [ ] **[Methods survey notebook](../../notebooks/01-methods-survey-and-choice.ipynb)**
      — reference: the proven-methods catalog + first-method decision.

## Trace review (no tunnel needed)

- **9B trace viewer: http://100.76.200.13:7676/traces-9b/index.html** — full inspect
  viewer for all banked 9B trials (own static server on the tailnet; Jupyter's
  /files/ CSP sandbox breaks the viewer app, don't use that route). Use with the
  reading queue in [transcript-pass-0901.md](../../results/transcript-pass-0901.md).
- Refresh after new runs: `./serve-traces.sh <log-dir> <name>` (e.g.
  `./serve-traces.sh results/pilot-reminder-v2 pilot` → `review/traces-pilot/`).

## Standing links

- JupyterLab root: http://100.76.200.13:8890/lab
- Evening/morning status (written by the queue): `results/MORNING-STATUS.md`
- Cache dir (tonight): `microscope/cache/qwen35-9b-v1/`
