"""tutorbench — harness v1 for the scripted tutoring-pressure study.

Design decisions (2026-08-27, see HARNESS-V1.md in the bench root):
  - items are synthetic quadratics with known integer roots, so the primary
    outcome (answer content appeared / when) is MECHANICAL, not judged;
  - the counterparty is a fixed 8-turn escalation script, identical for
    every model and trial — pressure is the treatment, and late-round
    inputs no longer depend on the subject's earlier behavior;
  - the episode ends at the submission event: the scripted student submits
    as soon as a tutor reply strictly states both roots, else at deadline;
  - an LLM judge grades the concession LEVEL per round (secondary,
    cached, hand-validated) — it never decides the primary endpoint.
"""

from tutorbench.items import make_items, detect_roots, BUNDLE_VERSION
from tutorbench.script import PRESSURE_TURNS, FINAL_TURN, submission_round
