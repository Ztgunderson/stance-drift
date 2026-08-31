import math

import pandas as pd

from driftlab.outcomes import wilson, outcome_table


def test_wilson_center():
    p, lo, hi = wilson(6, 12)
    assert p == 0.5
    # symmetric around 0.5, and the known Wilson width at n=12
    assert math.isclose(lo, 1 - hi, abs_tol=1e-12)
    assert 0.25 < lo < 0.30 and 0.70 < hi < 0.75


def test_wilson_edges():
    p, lo, hi = wilson(0, 12)
    assert p == 0.0 and lo == 0.0 and 0 < hi < 0.30
    p, lo, hi = wilson(12, 12)
    assert p == 1.0 and hi == 1.0 and 0.70 < lo < 1.0
    p, lo, hi = wilson(0, 0)
    assert math.isnan(p) and math.isnan(lo) and math.isnan(hi)


def test_outcome_table_survives_cross_model_trial_id_collision():
    # regression 2026-08-27: two models sharing a trial id must both count —
    # a global drop_duplicates("trial") silently deleted the second model
    rows = []
    for model in ("m1", "m2"):
        rows.append({"model": model, "agent": "convincer",
                     "trial": "tutor/convincer/in_context/1/0",
                     "gave_in": model == "m1", "round": 0})
    out = outcome_table(pd.DataFrame(rows))
    assert list(out["n"]) == [1, 1]
    assert list(out["gave_in"]) == [1, 0]


def test_outcome_table_counts_trials_not_rows():
    # 2 trials x 3 note-rows each: n must be 2, not 6
    rows = []
    for trial, gave in (("t1", True), ("t2", False)):
        for r in range(3):
            rows.append({"model": "m", "agent": "convincer",
                         "trial": trial, "gave_in": gave, "round": r})
    out = outcome_table(pd.DataFrame(rows))
    assert len(out) == 1
    assert out.loc[0, "n"] == 2
    assert out.loc[0, "gave_in"] == 1
    assert out.loc[0, "rate"] == 0.5
