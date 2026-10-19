"""Incoming-message deltas, pressure axis, item difficulty, drift speed.

Brainstorm items 1, 2, 4, 5 (Sep 3). All label-free directions are computed
in-sample (they do not see the outcome); every outcome-fit direction/probe is
out of fold with item-grouped folds inside the supportive cell.

Row family ("paired rows"): one row per (trial, round r >= 2) with
    U      = resid_user[r]      state after the student's message of round r
    A_prev = resid[r-1]         state after the tutor's reply of round r-1
    D      = U - A_prev         what the incoming message did to the state
Rows are kept while r <= event round (the user position at the event round
is still pre-decision: the tutor has not written the reply yet). lead = ev - r
(0 allowed), -1 when the trial has no event. Evaluation is round-conditional
(probes.round_conditional_auc) at leads 1-3 (the Amendment-3 window) and at
lead 0 separately.
"""

import numpy as np

from driftlab.interp import _group_folds, _dual_ridge
from driftlab.probes import (event_of, round_conditional_auc, oof_scores,
                             diffmean_oof, permute_trial_labels)


# -- rows ----------------------------------------------------------------------

def build_paired_rows(trials, layer):
    U, A, y, rnd, lead, per, item, trial = [], [], [], [], [], [], [], []
    for t in trials:
        m = t["meta"]
        if t["resid_user"] is None:
            continue
        leaked, left, ev = event_of(m)
        if leaked and ev == 1:
            continue
        Ru = list(map(int, t["user_rounds"])); Ra = list(map(int, t["resid_rounds"]))
        for i, r in enumerate(Ru):
            if r < 2 or (r - 1) not in Ra:
                continue
            if ev is not None and r > ev:
                continue
            U.append(t["resid_user"][i, layer].astype(np.float32))
            A.append(t["resid"][Ra.index(r - 1), layer].astype(np.float32))
            y.append(leaked); rnd.append(r)
            lead.append((ev - r) if ev is not None else -1)
            per.append(m["persona"]); item.append(m["item_id"]); trial.append(m["trial"])
    U = np.stack(U) if U else np.zeros((0, 0), np.float32)
    A = np.stack(A) if A else np.zeros((0, 0), np.float32)
    return {"U": U, "A_prev": A, "D": U - A, "will_leak": np.array(y, bool),
            "rnd": np.array(rnd), "lead": np.array(lead), "persona": np.array(per),
            "item": np.array(item), "trial": np.array(trial)}


def sel(rows, mask):
    n = len(rows["rnd"])
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v)
            for k, v in rows.items()}


# -- evaluation ------------------------------------------------------------------

def evaluate(scores, rows, n_perm=0, seed=0):
    """Round-conditional AUROC at leads 1-3 and lead 0 (positives restricted
    by lead; negatives = every non-leak row at that round). Optional
    trial-level permutation null on the leads 1-3 number. Rows with lead 0
    are excluded from the leads-1-3 evaluation entirely (they would otherwise
    count as negatives? no: they are positives with the wrong lead — the
    pos_mask handles that), but positives at lead 0 never enter leads 1-3."""
    y, rnd, lead = rows["will_leak"], rows["rnd"], rows["lead"]
    out = {}
    m13 = (lead >= 1) & (lead <= 3)
    a, tab = round_conditional_auc(scores, y, rnd, pos_mask=m13)
    out["leads_1_3"] = {"auc_rc": a, "per_round": tab, "n_pos": int((y & m13).sum())}
    a0, tab0 = round_conditional_auc(scores, y, rnd, pos_mask=(lead == 0))
    out["lead_0"] = {"auc_rc": a0, "per_round": tab0, "n_pos": int((y & (lead == 0)).sum())}
    if n_perm:
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(n_perm):
            yp = permute_trial_labels(y, rows["trial"], rng)
            null.append(round_conditional_auc(scores, yp, rnd, pos_mask=m13)[0])
        out["leads_1_3"]["null_mean"] = float(np.nanmean(null))
        out["leads_1_3"]["null_sd"] = float(np.nanstd(null))
    return out


def oof_probe_eval(X, rows, C=1.0, k=5, seed=0, n_perm=0):
    """Item-grouped OOF logistic probe AND diffmean projection; the
    permutation null refits the direction on permuted labels (so it is a
    null on the whole pipeline, not on a fixed score vector)."""
    y, g = rows["will_leak"], rows["item"]
    s_pr = oof_scores(X, y, g, C=C, k=k, seed=seed)
    s_dm, _ = diffmean_oof(X, y, g, k=k, seed=seed)
    res = {"probe": evaluate(s_pr, rows), "diffmean": evaluate(s_dm, rows)}
    if n_perm:
        rng = np.random.default_rng(seed + 1)
        npr, ndm = [], []
        m13 = (rows["lead"] >= 1) & (rows["lead"] <= 3)
        for _ in range(n_perm):
            yp = permute_trial_labels(y, rows["trial"], rng)
            sp = oof_scores(X, yp, g, C=C, k=k, seed=seed)
            sd, _ = diffmean_oof(X, yp, g, k=k, seed=seed)
            npr.append(round_conditional_auc(sp, yp, rows["rnd"], pos_mask=m13)[0])
            ndm.append(round_conditional_auc(sd, yp, rows["rnd"], pos_mask=m13)[0])
        res["probe"]["leads_1_3"]["null_mean"] = float(np.nanmean(npr))
        res["probe"]["leads_1_3"]["null_sd"] = float(np.nanstd(npr))
        res["diffmean"]["leads_1_3"]["null_mean"] = float(np.nanmean(ndm))
        res["diffmean"]["leads_1_3"]["null_sd"] = float(np.nanstd(ndm))
    return res, s_pr, s_dm


# -- item 2: pressure axis --------------------------------------------------------

def pressure_axis_scores(rows_all, feat="U", target_persona="supportive",
                         ref_persona="neutral", held_out=True):
    """Label-free axis per round: mean(feat | target persona, round r) minus
    mean(feat | reference persona, round r). Item-held-out: the axis for rows
    of item q is computed on the other items. Returns scores for the target
    persona's rows (projection of feat onto the unit axis) aligned to
    sel(rows_all, persona == target), plus the per-round axes (all items)."""
    tp = rows_all["persona"] == target_persona
    rp = rows_all["persona"] == ref_persona
    X, rnd, item = rows_all[feat], rows_all["rnd"], rows_all["item"]
    idx_t = np.flatnonzero(tp)
    scores = np.zeros(len(idx_t))
    axes = {}
    for r in np.unique(rnd[tp]):
        if not (rp & (rnd == r)).any():
            continue                      # no reference rows at this round
        a_all = X[tp & (rnd == r)].mean(0) - X[rp & (rnd == r)].mean(0)
        axes[int(r)] = a_all / np.linalg.norm(a_all)
        for j, i in enumerate(idx_t):
            if rnd[i] != r:
                continue
            if int(r) not in axes:
                scores[j] = np.nan; continue
            if held_out:
                keep = item != item[i]
                mt = tp & (rnd == r) & keep; mr = rp & (rnd == r) & keep
                if mt.sum() < 3 or mr.sum() < 3:
                    scores[j] = np.nan; continue
                a = X[mt].mean(0) - X[mr].mean(0); a /= np.linalg.norm(a)
            else:
                a = axes[int(r)]
            scores[j] = float(X[i] @ a)
    return scores, axes


# -- item 4: item difficulty ---------------------------------------------------------

def item_difficulty(trials, layer, persona="supportive", state="assistant", rnd=1,
                    lam=1.0, n_perm=200, seed=0):
    """Per item: leak rate over the persona's reps vs the item's mean state at
    round `rnd` (assistant = after tutor reply, user = after student message).
    Leave-one-item-out dual ridge; Spearman(pred, actual) over items with a
    permutation null over item labels."""
    from scipy.stats import spearmanr
    per_item = {}
    for t in trials:
        m = t["meta"]
        if m["persona"] != persona:
            continue
        leaked, _, ev = event_of(m)
        H, R = ((t["resid"], t["resid_rounds"]) if state == "assistant"
                else (t["resid_user"], t["user_rounds"]))
        R = list(map(int, R))
        if rnd not in R:
            continue
        d = per_item.setdefault(m["item_id"], {"x": [], "leak": []})
        d["x"].append(H[R.index(rnd), layer].astype(np.float32)); d["leak"].append(leaked)
    items = sorted(per_item)
    Xi = np.stack([np.mean(per_item[q]["x"], 0) for q in items])
    rate = np.array([np.mean(per_item[q]["leak"]) for q in items])
    def loio(rate_v):
        pred = np.zeros(len(items))
        for i in range(len(items)):
            tr = np.arange(len(items)) != i
            pred[i] = _dual_ridge(Xi[tr], rate_v[tr], Xi[i:i + 1], lam)[0]
        return pred
    pred = loio(rate)
    rho = float(spearmanr(pred, rate).correlation)
    rng = np.random.default_rng(seed)
    null = [float(spearmanr(loio(rng.permutation(rate)), rate).correlation)
            for _ in range(n_perm)]
    return {"items": items, "rate": rate.tolist(), "pred": pred.tolist(), "spearman": rho,
            "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
            "p_perm": float((np.sum(np.array(null) >= rho) + 1) / (n_perm + 1)),
            "state": state, "round": rnd, "persona": persona, "layer": layer, "lam": lam}


# -- item 5: drift speed -----------------------------------------------------------------

def clock_axis(rows, feat="A_prev", lo=1, hi=5):
    """Label-free clock: mean state at rounds >= hi minus mean at round lo
    (A_prev at row round r is the assistant state at r-1)."""
    r_prev = rows["rnd"] - 1
    a = rows[feat][r_prev >= hi].mean(0) - rows[feat][r_prev == lo].mean(0)
    return a / np.linalg.norm(a)


def speed_features(rows, clock):
    """Per row: |D| (size of the incoming push), cos(D, clock), |U| , |A_prev|,
    D·clock (signed movement along the clock)."""
    D, U, A = rows["D"], rows["U"], rows["A_prev"]
    nD = np.linalg.norm(D, axis=1)
    return {"push_norm": nD, "push_cos_clock": (D @ clock) / np.maximum(nD, 1e-6),
            "push_along_clock": D @ clock, "U_norm": np.linalg.norm(U, axis=1),
            "A_prev_norm": np.linalg.norm(A, axis=1)}
