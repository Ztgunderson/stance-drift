#!/usr/bin/env python3
"""Generate `reproduce_findings.ipynb` — every claim in PAPER-DRAFT.md, in order.

    python3 notebooks/make_reproduce.py

The notebook is **model-agnostic**. It reads `MODELS` from a single cell at the
top, mapping a display label to a results directory. Add a second model there and
every figure gains a second panel and every table a second column — nothing else
changes. That is the whole point: the replication question ("is this more than
qwen3.6?") should cost one line, not a rewrite.
"""
import json
import pathlib

NB = pathlib.Path(__file__).parent / "reproduce_findings.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(True)}


C = []

C.append(md("""# Reproducing the stance-drift findings

Every numbered claim in `PAPER-DRAFT.md`, computed from the saved `.eval`
transcripts. No model is called — this runs during a sweep, or years later.

**To add a second model:** put it in `MODELS` in the next cell. Every figure and
table below picks it up automatically.
"""))

C.append(code('''# ── configure ────────────────────────────────────────────────────────────────
# label -> results directory (8-round tutor sweeps)
MODELS = {
    "qwen3.6-35b": "../results/qwen3.6-35b-tutor8",
    # "gemma-3-12b": "../results/gemma3-tutor8",     # <- add a second model here
}
LONG_HORIZON = "../results/qwen3.6-35b-16round"   # 16-round probe, or None
CLEAN_CP     = "../results/qwen3.6-35b-cleanCP"   # constrained counterparty, or None

import sys, glob, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from stancedrift import analysis, prompts

DIMS   = list(analysis.DIMS)
AGENTS = ["convincer", "neutral", "supportive"]
ACOL   = {"convincer": "#d1495b", "neutral": "#8d99ae", "supportive": "#2a9d8f"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})

FIG = __import__("pathlib").Path("figures"); FIG.mkdir(exist_ok=True)
def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")

D = {}
for label, path in MODELS.items():
    D[label] = analysis.load_sweep(path)
    print(f"{label:16s} loaded")
print(f"\\n{len(D)} model(s)")'''))

C.append(md("""## Part 0 — integrity gates

These run first and **raise** rather than warn. The trial-count assertion exists
because a trial-id collision once silently discarded 11 of every 12 trials and
produced a plausible wrong answer, not an error.
"""))

C.append(code('''for label, df in D.items():
    n_files  = len(glob.glob(f"{MODELS[label]}/**/*.eval", recursive=True))
    n_trials = df.trial.nunique()
    assert n_trials == n_files, (
        f"{label}: {n_trials} trials from {n_files} files — id collision")

    t = df.drop_duplicates("trial")
    cells = t.groupby(["agent", "arm"]).size()
    assert cells.nunique() == 1, f"{label}: unbalanced cells\\n{cells}"

    unparsed = df[DIMS].isna().any(axis=1).sum()
    print(f"{label:16s} {n_trials:3d} trials = {n_files:3d} files | "
          f"cells all n={cells.iloc[0]} | {unparsed} unparsed")
print("\\nall integrity gates passed")'''))

C.append(md("""## §3 — The model has no position until someone pushes

`stance` at the midpoint when alone, pinned near 9 the moment a counterparty
appears. The largest single movement in the dataset, and it happens **on
contact**, before any argument is made.
"""))

C.append(code('''fig, axes = plt.subplots(1, len(D), figsize=(5.6*len(D), 3.8), squeeze=False)
for ax, (label, df) in zip(axes[0], D.items()):
    ph = df.groupby("phase")[DIMS].mean().reindex(["before", "during", "end"])
    display(ph.round(2).style.set_caption(f"{label} — phase means"))
    for dim in DIMS:
        ax.plot([0, 1, 2], ph[dim], marker="o", label=dim, lw=2)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["before\\n(alone)", "during", "end\\n(hindsight)"])
    ax.set_ylim(0, 10); ax.set_ylabel("self-rating 0-10"); ax.set_title(label)
    ax.annotate("", xy=(1, ph.stance.iloc[1]), xytext=(0, ph.stance.iloc[0]),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=2))
axes[0][-1].legend(fontsize=8, loc="center right")
fig.suptitle("Position is created by the encounter (red = stance on contact)", y=1.02)
fig.tight_layout(); save(fig, "s3_phase"); plt.show()'''))

C.append(md("""## §4 — Attitude sets the trajectory, not the starting point

**The central claim.** At round 1 the counterparty conditions are near
indistinguishable on `strategy`; by round 8 they have diverged many-fold. The
attitude expresses itself by *accumulation*.
"""))

C.append(code('''for label, df in D.items():
    d = df[df.phase == "during"]
    piv = d.pivot_table(index="round", columns="agent", values="strategy", aggfunc="mean")
    r1, r8 = piv.iloc[0], piv.iloc[-1]
    print(f"\\n=== {label} — strategy ===")
    print(f"  round 1 spread: {r1.max()-r1.min():.2f}   round {piv.index[-1]} spread: {r8.max()-r8.min():.2f}"
          f"   ({(r8.max()-r8.min())/max(r1.max()-r1.min(), 1e-9):.1f}x divergence)")
    for a in AGENTS:
        if a in piv:
            y = piv[a].values; x = np.arange(1, len(y)+1)
            print(f"    {a:11s} {y[0]:.2f} -> {y[-1]:.2f}   slope {np.polyfit(x,y,1)[0]:+.3f}/round")

    print("  round-1 vs round-8 spread, all dimensions:")
    sp = pd.DataFrame({
        "round1": d[d["round"] == 1].groupby("agent")[DIMS].mean().agg(lambda c: c.max()-c.min()),
        "roundN": d[d["round"] == d["round"].max()].groupby("agent")[DIMS].mean().agg(lambda c: c.max()-c.min()),
    })
    display(sp.round(2))'''))

C.append(code('''fig, axes = plt.subplots(len(D), 2, figsize=(12, 4.2*len(D)), squeeze=False)
for row, (label, df) in enumerate(D.items()):
    d = df[df.phase == "during"]
    for col, dim in enumerate(["strategy", "inclination"]):
        ax = axes[row][col]
        for a in AGENTS:
            s = d[d.agent == a].groupby("round")[dim].mean()
            if len(s):
                ax.plot(s.index, s.values, marker="o", color=ACOL[a], label=a, lw=2)
        ax.set_ylim(0, 10); ax.set_xlabel("round"); ax.set_ylabel(dim)
        ax.set_title(f"{label} — {dim}")
        if col == 0: ax.legend(fontsize=8)
fig.suptitle("Attitude sets the trajectory: identical at round 1, divergent by round 8", y=1.01)
fig.tight_layout(); save(fig, "s4_trajectory"); plt.show()'''))

C.append(md("""## §5 — Warmth prevents drift; pressure and indifference are the same

Per-persona slopes. The claim is that **`supportive` is flat** while `convincer`
and `neutral` are indistinguishable from each other.

⚠️ `inclination` carries a **ceiling artifact** for `supportive` (it starts near
9/10 and cannot rise). `strategy` does not — all conditions start near 2.0 — and
is the load-bearing version.
"""))

C.append(code('''rows = []
for label, df in D.items():
    d = df[df.phase == "during"]
    for dim in ["inclination", "strategy", "pressure"]:
        for a in AGENTS:
            s = d[d.agent == a].groupby("round")[dim].mean()
            if len(s) < 2: continue
            x = np.arange(1, len(s)+1)
            rows.append(dict(model=label, dim=dim, agent=a, start=s.values[0],
                             end=s.values[-1], slope=np.polyfit(x, s.values, 1)[0]))
sl = pd.DataFrame(rows)
display(sl.pivot_table(index=["model", "dim"], columns="agent", values="slope").round(3))
print("A flat supportive row and near-equal convincer/neutral rows reproduce §5.")

fig, ax = plt.subplots(figsize=(8, 3.6))
sub = sl[sl.dim == "strategy"]
x = np.arange(len(sub.model.unique())); w = .25
for i, a in enumerate(AGENTS):
    v = [sub[(sub.model == m) & (sub.agent == a)].slope.mean() for m in MODELS]
    ax.bar(x + (i-1)*w, v, width=w, color=ACOL[a], label=a)
ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels(list(MODELS))
ax.set_ylabel("strategy slope (per round)")
ax.set_title("Does warmth hold the calculation rate at zero?"); ax.legend(fontsize=8)
fig.tight_layout(); save(fig, "s5_slopes"); plt.show()'''))

C.append(md("""## §6 — Neither standard explanation fits

`anxiety` (people-pleasing) at floor everywhere; `strategy` (sycophancy) highest
exactly where `inclination` is lowest — calculation as resistance, not
capitulation.
"""))

C.append(code('''for label, df in D.items():
    d = df[df.phase == "during"]
    ag = d.groupby("agent")[DIMS].mean().reindex(AGENTS)
    display(ag.round(2).style.set_caption(f"{label} — during contact"))
    print(f"  anxiety max across conditions: {ag.anxiety.max():.2f}  (floor => people-pleasing has nothing to attach to)")
    if {"convincer", "supportive"} <= set(ag.index):
        diff = ag.loc["supportive"] - ag.loc["convincer"]
        print("  supportive - convincer:", {k: round(v, 2) for k, v in diff.items()})'''))

C.append(md("""## §7 — Drift is front-loaded and self-limiting

Requires the 16-round probe. Compares the early slope to the late slope on the
same trials.
"""))

C.append(code('''if LONG_HORIZON:
    lh = analysis.load_sweep(LONG_HORIZON)
    inc = lh[lh.phase == "during"].groupby("round")["inclination"].mean()
    half = len(inc) // 2
    e = np.polyfit(np.arange(2, half+1), inc.loc[2:half].values, 1)[0]
    l = np.polyfit(np.arange(half+1, len(inc)+1), inc.loc[half+1:].values, 1)[0]
    print(f"rounds 2-{half}:    {e:+.3f}/round")
    print(f"rounds {half+1}-{len(inc)}:  {l:+.3f}/round     ({abs(e/l):.1f}x flattening)")
    t_l = lh.drop_duplicates("trial")
    for label, df in D.items():
        t_s = df.drop_duplicates("trial")
        print(f"outcome  {label} 8-round: {t_s.gave_in.mean():.1%} (n={len(t_s)})  |  "
              f"16-round: {t_l.gave_in.mean():.1%} (n={len(t_l)})")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(inc.index, inc.values, marker="o", lw=2, color="#264653")
    ax.axvline(half + .5, ls="--", color="crimson")
    ax.annotate(f"{e:+.3f}/rd", (half*.5, inc.max()), color="crimson")
    ax.annotate(f"{l:+.3f}/rd", (half*1.5, inc.max()), color="crimson")
    ax.set_xlabel("round"); ax.set_ylabel("inclination"); ax.set_ylim(0, 10)
    ax.set_title("Drift is front-loaded and self-limiting")
    fig.tight_layout(); save(fig, "s7_saturation"); plt.show()
else:
    print("no long-horizon probe configured")'''))

C.append(md("""## §8 — Scale: attitude vs the model's own reflections

The `in_context`/`scratchpad` null is the ruler. Sign agreement across reps
should sit at chance; the persona effect on the same trials should be an order of
magnitude larger.
"""))

C.append(code('''for label, df in D.items():
    d = df[df.phase == "during"].copy()
    d["rep"] = pd.to_numeric(d["rep"], errors="coerce")
    ar = d.groupby(["rep", "arm"])[DIMS].mean().unstack("arm")
    diff = pd.DataFrame({k: ar[(k, "in_context")] - ar[(k, "scratchpad")] for k in DIMS}).dropna()
    n = len(diff)
    print(f"\\n=== {label} — sign agreement across {n} reps (n positive / {n}) ===")
    print((diff > 0).sum().to_string())
    p = d.groupby("arm")[DIMS].mean(); ag = d.groupby("agent")[DIMS].mean()
    scale = pd.DataFrame({"counterparty attitude": ag.max() - ag.min(),
                          "own reflections": (p.loc["in_context"] - p.loc["scratchpad"]).abs()})
    scale["ratio"] = (scale.iloc[:, 0] / scale.iloc[:, 1].replace(0, np.nan)).round(1)
    display(scale.round(2))'''))

C.append(md("""## §9 — The firmness rating is blind to all of it

`stance` should be near-constant across every condition, correlate ~0 with the
outcome, and its cross-condition spread should **shrink** while `strategy`
diverges.
"""))

C.append(code('''for label, df in D.items():
    d = df[df.phase == "during"]
    t = df.drop_duplicates("trial")
    st = d.groupby("trial")["stance"].mean().reindex(t.trial.values)
    r = np.corrcoef(st.values, t.gave_in.astype(float).values)[0, 1]
    ag = d.groupby("agent")["stance"].mean(); ar = d.groupby("arm")["stance"].mean()
    print(f"\\n=== {label} — stance ===")
    print(f"  by persona: {ag.min():.2f}-{ag.max():.2f}   by arm: {ar.min():.2f}-{ar.max():.2f}")
    print(f"  r with outcome: {r:+.3f}")
    for dim in ["stance", "strategy"]:
        s1 = d[d["round"] == 1].groupby("agent")[dim].mean()
        sN = d[d["round"] == d["round"].max()].groupby("agent")[dim].mean()
        print(f"  {dim:9s} cross-condition spread  round1 {s1.max()-s1.min():.2f} -> roundN {sN.max()-sN.min():.2f}")'''))

C.append(md("""## §10 — Measurement caveat: self-play contamination

How often the **counterparty** states the answer itself, by persona — and whether
constraining it helps. This is why no persona comparison is reported on the
behavioural outcome.
"""))

C.append(code('''from inspect_ai.log import read_eval_log
from stancedrift.task import TrialState
SC = prompts.load_scenes()["tutor"]

def leakage(path):
    rows = []
    for f in sorted(glob.glob(f"{path}/**/*.eval", recursive=True)):
        for s in (read_eval_log(f).samples or []):
            st = s.store_as(TrialState)
            if not st.rounds: continue
            leaked = any(prompts.gave_in(r.counterparty or "", SC)[0] for r in st.rounds)
            rows.append(dict(agent=st.agent, leaked=leaked, gave_in=bool(st.gave_in)))
    return pd.DataFrame(rows)

for label, path in MODELS.items():
    lk = leakage(path)
    tab = lk.groupby("agent")["leaked"].agg(["sum", "count", "mean"]).reindex(AGENTS)
    display(tab.round(3).style.set_caption(f"{label} — counterparty states the answer itself"))
    print(f"  overall: {lk.leaked.mean():.1%}")

if CLEAN_CP:
    cc = leakage(CLEAN_CP)
    if len(cc):
        display(cc.groupby("agent")["leaked"].agg(["sum", "count", "mean"]).reindex(AGENTS).round(3)
                  .style.set_caption("constrained counterparty (SD_CP_NO_ANSWER=1)"))
        print(f"  overall: {cc.leaked.mean():.1%}")'''))

C.append(md("""## Summary

If a second model is configured in `MODELS`, the questions to ask of the tables
above are:

| § | claim | replicates if… |
|---|---|---|
| 3 | position created by the encounter | `stance` near midpoint alone, ~9 during |
| 4 | attitude sets trajectory | round-1 spread small, round-N spread large |
| 5 | warmth prevents drift | `supportive` slope ≈ 0; convincer ≈ neutral |
| 6 | neither explanation fits | `anxiety` at floor; `strategy` high where `inclination` low |
| 7 | front-loaded, self-limiting | early slope ≫ late slope |
| 8 | attitude ≫ own reflections | sign agreement ≈ chance; ratio ≫ 1 |
| 9 | firmness rating uninformative | `stance` near-constant, r ≈ 0 |

A finding that fails to replicate is a **result**, not a failure — record it.
"""))

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3",
      "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.write_text(json.dumps(nb, indent=1))
print(f"wrote {NB} — {len(C)} cells")
