"""Give-in outcomes: the true/false backbone every trend hangs off."""

import math

from .datasets import DIMS  # noqa: F401  (re-export convenience)


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a binomial rate.

    Wilson rather than normal approximation because at n=12 per cell the
    normal interval misbehaves near 0 and 1 — exactly where these rates sit.
    Returns (p, lo, hi); NaNs when n == 0.
    """
    if n == 0:
        nan = float("nan")
        return nan, nan, nan
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def outcome_table(df, by=("model", "agent")):
    """Give-in rate with Wilson interval per group. One row per trial first —
    the input df has ~10 rows (notes) per trial and pooling those would
    inflate n tenfold."""
    import pandas as pd
    # dedup within group keys, not globally: trial ids are only guaranteed
    # unique within a model (load_turns prefixes them, but stay safe for
    # frames built by other paths)
    trials = df.drop_duplicates(list(by) + ["trial"])[list(by) + ["trial", "gave_in"]]
    out = []
    for key, g in trials.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        n, k = len(g), int(g["gave_in"].sum())
        p, lo, hi = wilson(k, n)
        out.append(dict(zip(by, key),
                        n=n, gave_in=k, rate=p, lo=lo, hi=hi))
    return (pd.DataFrame(out)
            .sort_values(list(by)).reset_index(drop=True))
