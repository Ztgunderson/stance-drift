"""Self-report trends over rounds, split by outcome (gave_in True/False).

The split of record is WITHIN agent condition: gave_in is confounded with
agent (supportive trials almost never give in), so a pooled gave_in split
partly re-plots the agent manipulation. plot_trends therefore panels by
agent unless told otherwise. See READING-THE-DIMENSIONS.md in the
stance-drift repo for why cross-condition comparison of `inclination`
is invalid outright.
"""

import numpy as np

from .datasets import DIMS


def trend_table(df, split="gave_in"):
    """Mean, sem and n of each dimension per (split, round), 'during' rounds
    plus round 0 (before) and the hindsight round. One row per group-round."""
    import pandas as pd
    d = df.dropna(subset=list(DIMS))
    g = d.groupby([split, "round"])[list(DIMS)]
    mean, sem, n = g.mean(), g.sem(), g.size().rename("n")
    out = mean.join(sem, rsuffix="_sem").join(n).reset_index()
    return out.sort_values([split, "round"]).reset_index(drop=True)


def plot_trends(df, split="gave_in", by_agent=True, title=None):
    """5-panel figure (one per dimension): mean ± sem over rounds for each
    split value. by_agent=True facets rows by agent, keeping the comparison
    within condition."""
    import matplotlib.pyplot as plt
    agents = sorted(df["agent"].unique()) if by_agent else [None]
    fig, axes = plt.subplots(len(agents), len(DIMS),
                             figsize=(3.1 * len(DIMS), 2.6 * len(agents)),
                             sharex=True, squeeze=False)
    for ai, agent in enumerate(agents):
        sub = df if agent is None else df[df["agent"] == agent]
        tt = trend_table(sub, split=split)
        for di, dim in enumerate(DIMS):
            ax = axes[ai][di]
            for val, g in tt.groupby(split):
                ax.errorbar(g["round"], g[dim], yerr=g[dim + "_sem"],
                            marker="o", ms=3, capsize=2, label=f"{split}={val}")
            ax.set_ylim(-0.5, 10.5)
            if ai == 0:
                ax.set_title(dim, fontsize=10)
            if di == 0:
                ax.set_ylabel(agent or "all", fontsize=9)
            if ai == len(agents) - 1:
                ax.set_xlabel("round")
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle(title or f"self-reports over rounds, split by {split}", y=1.0)
    fig.tight_layout()
    return fig


def plot_spaghetti(df, dim, agent=None, title=None):
    """Every trial as its own line for one dimension, colored by outcome —
    the honest view behind the mean curves."""
    import matplotlib.pyplot as plt
    sub = df if agent is None else df[df["agent"] == agent]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for _, g in sub.groupby("trial"):
        g = g.dropna(subset=[dim]).sort_values("round")
        color = "tab:red" if g["gave_in"].iloc[0] else "tab:blue"
        ax.plot(g["round"], g[dim], color=color, alpha=0.35, lw=1)
    ax.plot([], [], color="tab:red", label="gave_in=True")
    ax.plot([], [], color="tab:blue", label="gave_in=False")
    ax.set_xlabel("round"); ax.set_ylabel(dim); ax.set_ylim(-0.5, 10.5)
    ax.legend(fontsize=8)
    ax.set_title(title or f"{dim} — every trial" + (f" ({agent})" if agent else ""))
    fig.tight_layout()
    return fig
