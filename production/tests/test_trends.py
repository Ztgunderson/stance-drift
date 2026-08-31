import numpy as np
import pandas as pd

from driftlab.datasets import DIMS
from driftlab.trends import trend_table


def toy_df():
    rows = []
    for trial, gave, base in (("a", True, 2.0), ("b", True, 4.0),
                              ("c", False, 8.0)):
        for r in (1, 2):
            row = {"trial": trial, "gave_in": gave, "round": r,
                   "agent": "convincer", "phase": "during"}
            row.update({d: base + r for d in DIMS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_trend_table_means_and_n():
    tt = trend_table(toy_df())
    g = tt[(tt["gave_in"]) & (tt["round"] == 1)].iloc[0]
    assert np.isclose(g["pressure"], 4.0)  # mean of 3 and 5
    assert g["n"] == 2
    h = tt[(~tt["gave_in"]) & (tt["round"] == 2)].iloc[0]
    assert np.isclose(h["stance"], 10.0)
    assert h["n"] == 1


def test_trend_table_drops_unparsed_rows():
    df = toy_df()
    df.loc[0, list(DIMS)] = np.nan
    tt = trend_table(df)
    assert tt[(tt["gave_in"]) & (tt["round"] == 1)].iloc[0]["n"] == 1
