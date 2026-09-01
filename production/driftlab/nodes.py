"""Node-propensity regression (PREREG-9B Amendment 4, proposed).

Joins node_resample.py output (per-node resampled action propensities) to the
replay cache's PRE-DECISION states and regresses propensity on the state,
against the registered baselines.

Alignment contract:
- A node (trial, round r) branches BEFORE round r's tutor reply; its state is
  the end of round r's STUDENT turn: npz `resid_user` row where
  `user_rounds == r` (replay.py contract). Fallback (position="assistant"):
  the end of round r-1's tutor reply (`resid`/`resid_rounds == r-1`) — an
  earlier, weaker state that lacks round r's pressure; a warning is emitted.
- The self-report baseline for a node uses the round r-1 note (`report_ev`
  row r-2): the last report written BEFORE the decision. Round r's report is
  post-reply and would leak outcome information. r=1/r=2 nodes may have no
  usable pre-decision report (nan row -> mean-imputed, flagged).
- Cache files are matched on meta (persona, item_id); if several files match
  (expansion reps), the one whose outcome/leak_round/leave_round equal the
  node's originals wins, else the first match is taken with a warning.

Statistics: ridge (standardized) with GroupKFold out-of-fold predictions;
Spearman rho + R^2 with bootstrap-over-trials CIs; plus a diff-mean-style
axis (top vs bottom propensity terciles) projection correlation. Baselines
per PREREG: (persona one-hot + round) and pre-decision self-report EVs.
"""

import glob
import json
import math
import os
import warnings

import numpy as np
import pandas as pd


# -- loading ----------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def load_nodes(json_path):
    """Resampler output -> one row per node with propensities + Wilson CIs."""
    d = json.load(open(json_path))
    rows = []
    for n in d["nodes"]:
        k = n["k"]
        pl, pl_lo, pl_hi = wilson(n["counts"]["leak"], k)
        pv, pv_lo, pv_hi = wilson(n["counts"]["leave"], k)
        rows.append({
            "trial": n["trial"], "persona": n["persona"],
            "item_id": n["item_id"], "round": n["round"], "k": k,
            "n_leak": n["counts"]["leak"], "n_leave": n["counts"]["leave"],
            "n_continue": n["counts"]["continue"],
            "P_leak": pl, "P_leak_lo": pl_lo, "P_leak_hi": pl_hi,
            "P_leave": pv, "P_leave_lo": pv_lo, "P_leave_hi": pv_hi,
            "orig_outcome": n.get("orig_outcome"),
            "orig_leak_round": n.get("orig_leak_round"),
            "orig_leave_round": n.get("orig_leave_round"),
        })
    return pd.DataFrame(rows)


def _cache_index(cache_dir):
    """(persona, item_id) -> list of (npz_path, meta) in the cache."""
    idx = {}
    for meta_f in sorted(glob.glob(os.path.join(cache_dir, "*.json"))):
        meta = json.load(open(meta_f))
        npz_f = meta_f[:-5] + ".npz"
        if os.path.exists(npz_f):
            idx.setdefault((meta["persona"], meta["item_id"]), []).append(
                (npz_f, meta))
    return idx


def _pick(cands, row):
    """Disambiguate multiple cache files for one (persona, item_id)."""
    if len(cands) == 1:
        return cands[0]
    exact = [c for c in cands
             if c[1].get("outcome") == row["orig_outcome"]
             and c[1].get("leak_round") == row["orig_leak_round"]
             and c[1].get("leave_round") == row["orig_leave_round"]]
    if exact:
        return exact[0]
    warnings.warn(f"ambiguous cache match for {row['trial']} "
                  f"r{row['round']}; "
                  "taking first")
    return cands[0]


def join_states(nodes_df, cache_dir, layer, position="user"):
    """Align nodes to cached states.

    Returns (kept_df, X, R): kept_df = surviving node rows (reset index),
    X [n, d_model] the layer-`layer` pre-decision states, R [n, n_items] the
    round r-1 report EVs (nan where absent). Drops nodes with no cache file
    or no matching round row, with a summary warning.
    """
    if position not in ("user", "assistant"):
        raise ValueError(position)
    if position == "assistant":
        warnings.warn("position='assistant' uses the round r-1 reply-end "
                      "state (pre-student-pressure) — weaker than 'user'")
    idx = _cache_index(cache_dir)
    keep, X, R, dropped = [], [], [], []
    for _, row in nodes_df.iterrows():
        r_round = int(row["round"])
        cands = idx.get((row["persona"], row["item_id"]))
        if not cands:
            dropped.append((row["trial"], r_round, "no cache file"))
            continue
        npz_f, meta = _pick(cands, row)
        z = np.load(npz_f)
        if position == "user" and "resid_user" in z:
            rounds, resid = z["user_rounds"], z["resid_user"]
            want = r_round
        else:
            if position == "user":
                warnings.warn(f"{os.path.basename(npz_f)} lacks resid_user "
                              "(old-schema cache) — falling back to "
                              "assistant r-1 state")
            rounds, resid = z["resid_rounds"], z["resid"]
            want = r_round - 1
        hit = np.where(rounds == want)[0]
        if len(hit) == 0:
            dropped.append((row["trial"], r_round, f"no state row (want "
                            f"round {want}, have {sorted(set(rounds))})"))
            continue
        X.append(resid[hit[0], layer].astype(np.float32))
        ev = z["report_ev"] if "report_ev" in z else None
        pre_round = r_round - 1
        if ev is not None and 1 <= pre_round <= ev.shape[0]:
            R.append(ev[pre_round - 1].astype(np.float32))
        else:
            R.append(np.full(ev.shape[1] if ev is not None else 7,
                             np.nan, np.float32))
        keep.append(row)
    if dropped:
        warnings.warn(f"join_states dropped {len(dropped)}/{len(nodes_df)} "
                      f"nodes: {dropped[:5]}{'...' if len(dropped) > 5 else ''}")
    kept = pd.DataFrame(keep).reset_index(drop=True)
    return kept, np.stack(X) if X else np.empty((0, 0)), \
        np.stack(R) if R else np.empty((0, 0))


# -- regression -------------------------------------------------------------

def _spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def propensity_regression(X, y, groups, n_boot=1000, alpha=10.0, seed=0):
    """Out-of-fold ridge regression of propensity y on features X.

    Returns dict: oof predictions, spearman rho / R^2 with bootstrap-over-
    groups CIs, and the tercile-diffmean axis correlation.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    n_splits = min(5, len(uniq))
    if n_splits < 2:
        raise ValueError("need >=2 groups")
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict(X[te])

    rho = _spearman(oof, y)
    r2 = float(r2_score(y, oof))

    rng = np.random.default_rng(seed)
    b_rho, b_r2 = [], []
    for _ in range(n_boot):
        gs = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([np.where(groups == g)[0] for g in gs])
        if np.std(y[sel]) == 0:
            continue
        b_rho.append(_spearman(oof[sel], y[sel]))
        b_r2.append(r2_score(y[sel], oof[sel]))
    ci = lambda v: (float(np.nanpercentile(v, 2.5)),
                    float(np.nanpercentile(v, 97.5))) if v else (np.nan,) * 2

    # tercile diff-mean axis: top vs bottom third of y
    q1, q2 = np.quantile(y, [1 / 3, 2 / 3])
    lo, hi = y <= q1, y >= q2
    axis_rho = np.nan
    if lo.sum() >= 2 and hi.sum() >= 2:
        d = X[hi].mean(0) - X[lo].mean(0)
        nrm = np.linalg.norm(d)
        if nrm > 0:
            axis_rho = _spearman(X @ (d / nrm), y)

    return {"oof": oof, "spearman": rho, "spearman_ci": ci(b_rho),
            "r2": r2, "r2_ci": ci(b_r2), "diffmean_axis_spearman": axis_rho,
            "n": len(y), "n_groups": len(uniq), "n_splits": n_splits}


def compare_channels(kept_df, X, R, target="P_leak", n_boot=1000):
    """The registered three-way comparison on identical nodes/folds.

    state    — the pre-decision residual state (the interp channel)
    baseline — persona one-hot + round number (what the script gives away)
    reports  — pre-decision self-report EVs (nan -> column-mean imputed;
               all-nan columns dropped; count reported)
    """
    y = kept_df[target].to_numpy()
    groups = kept_df["trial"].to_numpy()

    persona_oh = pd.get_dummies(kept_df["persona"]).to_numpy(np.float64)
    base_feats = np.column_stack([persona_oh,
                                  kept_df["round"].to_numpy(np.float64)])

    R = np.asarray(R, np.float64)
    keep_cols = ~np.all(np.isnan(R), axis=0) if R.size else np.array([], bool)
    Rk = R[:, keep_cols] if R.size else R
    n_imputed = 0
    if Rk.size:
        mu = np.nanmean(Rk, axis=0)
        mask = np.isnan(Rk)
        n_imputed = int(mask.sum())
        Rk = np.where(mask, mu, Rk)

    out = {"state": propensity_regression(X, y, groups, n_boot),
           "baseline": propensity_regression(base_feats, y, groups, n_boot)}
    if Rk.size and Rk.shape[1] > 0:
        out["reports"] = propensity_regression(Rk, y, groups, n_boot)
        out["reports"]["n_imputed_values"] = n_imputed
    else:
        out["reports"] = None
    return out
