"""Formal L1 probe pipeline — PREREG-9B Amendment 3 (frozen analysis plan)
with Amendment 6 hygiene (persona x item folds, trial-level permutation
nulls as distributions, per-fold layer choice reported, error columns).

Row set (shared by probe / diffmean / text / logit-report / ask-an-LLM):
  one row per (trial, round r) with r strictly before the trial's event
  (leak_round or leave_round); trials with no event contribute every round;
  round-1 leaks are excluded (no pre-event window) and counted.
Targets: will_leak (primary) — positives = pre-event rows of leak trials,
  negatives = pre-event rows of never-leak trials at the SAME round
  (round-conditional evaluation, Amendment 3 #3); will_leave — symmetric.
Grouping: persona x item cell (Amendment 6 #1) — greedy re-runs of the same
  item are near-duplicates and must never straddle folds.
Layer/C selection: nested inside the outer folds (Amendment 3 #5); the outer
  out-of-fold scores are the only ones reported as the primary test.
"""

import glob
import json
import os

import numpy as np

from driftlab.interp import _group_folds


# -- loading ------------------------------------------------------------------

def load_cache(cache_dir, layers=None):
    """List of trials: resid [turns, L', d] (layers subset if given),
    resid_user (same, or None), ev [rounds, 7] or None, meta dict."""
    out = []
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        mf = f[:-4] + ".json"
        if not os.path.exists(mf):
            continue
        z = np.load(f)
        m = json.load(open(mf))
        resid = z["resid"]
        ru = z["resid_user"] if "resid_user" in z else None
        if layers is not None:
            resid = resid[:, layers]
            ru = ru[:, layers] if ru is not None else None
        out.append({"resid": resid, "resid_user": ru,
                    "resid_rounds": z["resid_rounds"] if "resid_rounds" in z else None,
                    "user_rounds": z["user_rounds"] if "user_rounds" in z else None,
                    "ev": z["report_ev"] if "report_ev" in z else None,
                    "meta": m})
    return out


def event_of(meta):
    leaked = str(meta["outcome"]).startswith("leak")
    left = meta["outcome"] == "left"
    ev = meta.get("leak_round") if leaked else (meta.get("leave_round") if left else None)
    return leaked, left, ev


def build_rows(trials, position="assistant"):
    """Return dict of aligned arrays over pre-event rows.
    X [n, L, d] float32; will_leak, will_leave bool; rnd, lead int (lead=-1
    when the trial has no event); persona, item, trial str; group str
    (persona/item); ev_mean float (mean logit E[v] over the 7 report items at
    that round, nan if absent); n_r1_leaks int (excluded count)."""
    X, wl, wv, rnd, lead, per, item, trial, evm = [], [], [], [], [], [], [], [], []
    n_r1 = 0
    for t in trials:
        m = t["meta"]
        leaked, left, ev = event_of(m)
        if leaked and ev == 1:
            n_r1 += 1
            continue
        if position == "assistant":
            H, R = t["resid"], t["resid_rounds"]
        else:
            H, R = t["resid_user"], t["user_rounds"]
            if H is None:
                continue
        if R is None:
            R = np.arange(1, H.shape[0] + 1)
        for i, r in enumerate(np.asarray(R)):
            r = int(r)
            if ev is not None and r >= ev:
                continue
            X.append(H[i].astype(np.float32))
            wl.append(leaked); wv.append(left); rnd.append(r)
            lead.append((ev - r) if ev is not None else -1)
            per.append(m["persona"]); item.append(m["item_id"]); trial.append(m["trial"])
            e = t["ev"]
            evm.append(float(np.nanmean(e[r - 1])) if e is not None and r - 1 < len(e) else np.nan)
    return {"X": np.stack(X) if X else np.zeros((0, 0, 0), np.float32),
            "will_leak": np.array(wl, bool), "will_leave": np.array(wv, bool),
            "rnd": np.array(rnd), "lead": np.array(lead),
            "persona": np.array(per), "item": np.array(item), "trial": np.array(trial),
            "group": np.array([f"{p}/{i}" for p, i in zip(per, item)]),
            "ev_mean": np.array(evm), "n_r1_leaks": n_r1}


def subset(rows, mask):
    out = {}
    for k, v in rows.items():
        out[k] = v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == mask.shape else v
    return out


# -- metrics --------------------------------------------------------------------

def auc(scores, y):
    """Rank AUROC with ties at 0.5; nan if a class is empty."""
    s, y = np.asarray(scores, float), np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def round_conditional_auc(scores, y, rnd, pos_mask=None, min_n=3):
    """AUROC computed within each round (positives at round k vs negatives at
    round k), then weighted by the number of positives. pos_mask restricts
    which positives count (e.g. lead == 2); negatives are always every
    negative at that round. Returns (weighted_auc, per_round list of
    (round, auc, n_pos, n_neg))."""
    s, y, rnd = np.asarray(scores, float), np.asarray(y, bool), np.asarray(rnd)
    pm = np.ones_like(y) if pos_mask is None else np.asarray(pos_mask, bool)
    table, num, den = [], 0.0, 0
    for k in np.unique(rnd):
        pos = s[y & pm & (rnd == k)]; neg = s[~y & (rnd == k)]
        if len(pos) < min_n or len(neg) < min_n:
            continue
        a = auc(np.concatenate([pos, neg]),
                np.concatenate([np.ones(len(pos), bool), np.zeros(len(neg), bool)]))
        table.append((int(k), a, len(pos), len(neg)))
        num += a * len(pos); den += len(pos)
    return (num / den if den else float("nan")), table


def bootstrap_groups(stat_fn, groups, n_boot=500, seed=0):
    """Percentile CI of stat_fn(idx) by resampling GROUPS with replacement
    (cluster bootstrap over persona x item cells). stat_fn receives an index
    array into the rows. Returns (lo, hi, samples)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        v = stat_fn(idx)
        if not np.isnan(v):
            vals.append(v)
    vals = np.array(vals)
    if len(vals) == 0:
        return float("nan"), float("nan"), vals
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals


def permute_trial_labels(y, trial, rng):
    """Trial-level label permutation: each trial keeps one label, labels are
    shuffled ACROSS trials (rows inherit). Row-level shuffles would break the
    trial structure and understate the null."""
    ut, inv = np.unique(trial, return_inverse=True)
    lab = np.array([y[trial == t][0] for t in ut])
    return rng.permutation(lab)[inv]


# -- estimators -------------------------------------------------------------------

def _fit_logistic(Xtr, ytr, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=C, max_iter=1000).fit(sc.transform(Xtr), ytr)
    return sc, lr


def _score(sc_lr, X):
    sc, lr = sc_lr
    return lr.predict_proba(sc.transform(X))[:, 1]


def probe_direction(sc_lr):
    """Probe weight vector mapped back to raw residual coordinates, unit norm."""
    sc, lr = sc_lr
    w = lr.coef_[0] / sc.scale_
    return w / np.linalg.norm(w)


def diffmean_direction(X, y):
    d = X[y].mean(0) - X[~y].mean(0)
    return d / np.linalg.norm(d)


def oof_scores(X, y, groups, C=1.0, k=5, seed=0):
    """Plain grouped out-of-fold probability scores at a fixed layer/C."""
    folds = _group_folds(groups, k, seed)
    s = np.zeros(len(y))
    for f in range(k):
        te = folds == f; tr = ~te
        s[te] = _score(_fit_logistic(X[tr], y[tr], C), X[te])
    return s


N_JOBS = int(os.environ.get("PROBES_N_JOBS", "4"))


def _parallel(fn, jobs):
    """Run fn(*job) for each job, in processes when N_JOBS > 1."""
    if N_JOBS <= 1 or len(jobs) <= 1:
        return [fn(*j) for j in jobs]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=N_JOBS, prefer="processes")(delayed(fn)(*j) for j in jobs)


def _fit_score_job(Xtr, ytr, Xte, C):
    return _score(_fit_logistic(Xtr, ytr, C), Xte)


def nested_probe(X, y, groups, layer_grid, C_grid=(0.1, 1.0, 10.0),
                 k_outer=5, k_inner=3, seed=0, select_metric=None):
    """Nested CV: per outer fold, choose (layer, C) on inner grouped folds of
    the training part by mean inner AUROC (select_metric(scores, y_inner)
    overrides the metric), refit on the whole training part, score the test
    part. Returns dict: scores (outer OOF), choices [(layer, C)] per fold,
    inner_table {(layer, C): [inner auc per outer fold]}, directions (probe
    direction per outer fold at its chosen layer, raw coords)."""
    metric = select_metric or (lambda s, yy: auc(s, yy))
    folds = _group_folds(groups, k_outer, seed)
    scores = np.zeros(len(y)); choices, directions = [], []
    inner_table = {(L, C): [] for L in layer_grid for C in C_grid}
    for f in range(k_outer):
        te = folds == f; tr = ~te
        tr_idx = np.flatnonzero(tr)
        ifolds = _group_folds(groups[tr], k_inner, seed + 100 + f)
        best, best_key = -np.inf, None
        jobs, keys = [], []
        for L in layer_grid:
            for C in C_grid:
                for g in range(k_inner):
                    ite = ifolds == g; itr = ~ite
                    jobs.append((X[tr_idx[itr], L], y[tr_idx[itr]], X[tr_idx[ite], L], C))
                    keys.append((L, C, g))
        outs = _parallel(_fit_score_job, jobs)
        for L in layer_grid:
            for C in C_grid:
                s_in = np.zeros(len(tr_idx))
                for (LL, CC, g), o in zip(keys, outs):
                    if LL == L and CC == C:
                        s_in[ifolds == g] = o
                a = metric(s_in, y[tr])
                inner_table[(L, C)].append(a)
                if a > best:
                    best, best_key = a, (L, C)
        L, C = best_key
        model = _fit_logistic(X[tr, L], y[tr], C)
        scores[te] = _score(model, X[te, L])
        choices.append((int(L), float(C)))
        directions.append(probe_direction(model))
    return {"scores": scores, "choices": choices, "inner_table": inner_table,
            "directions": directions, "folds": folds}


def layer_curve(X, y, groups, layers, C=1.0, k=5, seed=0, n_perm=0, rnd=None,
                trial=None):
    """Descriptive per-layer grouped OOF AUROC (never used to pick the reported
    layer). With n_perm > 0 also returns a trial-level permutation null band
    (mean, sd per layer) — distributions, not one shuffle."""
    aucs, null_mean, null_sd = [], [], []
    rng = np.random.default_rng(seed)
    folds = _group_folds(groups, k, seed)
    ev = (lambda s, yy: auc(s, yy)) if rnd is None else \
         (lambda s, yy: round_conditional_auc(s, yy, rnd)[0])
    perms = [permute_trial_labels(y, trial, rng) for _ in range(n_perm)] if n_perm else []
    for L in layers:
        labelsets = [y] + perms
        jobs, keys = [], []
        for li, yy in enumerate(labelsets):
            for f in range(k):
                te = folds == f
                jobs.append((X[~te, L], yy[~te], X[te, L], C)); keys.append((li, f))
        outs = _parallel(_fit_score_job, jobs)
        scores = np.zeros((len(labelsets), len(y)))
        for (li, f), o in zip(keys, outs):
            scores[li, folds == f] = o
        aucs.append(ev(scores[0], y))
        if n_perm:
            vals = [ev(scores[li], labelsets[li]) for li in range(1, len(labelsets))]
            null_mean.append(float(np.nanmean(vals))); null_sd.append(float(np.nanstd(vals)))
    return {"layers": list(layers), "auc": np.array(aucs),
            "null_mean": np.array(null_mean), "null_sd": np.array(null_sd)}


def diffmean_oof(X, y, groups, k=5, seed=0):
    """Zero-parameter rival: projection onto the train-fold diff-in-means
    direction (raw coordinates), out of fold. Returns scores and the per-fold
    directions."""
    folds = _group_folds(groups, k, seed)
    s = np.zeros(len(y)); dirs = []
    for f in range(k):
        te = folds == f; tr = ~te
        d = diffmean_direction(X[tr], y[tr]); dirs.append(d)
        s[te] = X[te] @ d
    return s, dirs


def split_half_stability(X, y, groups, C=1.0, n_rep=5, seed=0):
    """Amendment-6/F3 hygiene: cosine(probe_A, probe_B) across random group
    halves vs cosine(probe, diffmean) on the same halves. If probe-probe is as
    low as probe-diffmean, the 'not one axis' reading is noise."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    pp, pd, dd = [], [], []
    for _ in range(n_rep):
        perm = rng.permutation(ug)
        A = np.isin(groups, perm[: len(perm) // 2]); B = ~A
        if y[A].sum() < 3 or (~y[A]).sum() < 3 or y[B].sum() < 3 or (~y[B]).sum() < 3:
            continue
        wa = probe_direction(_fit_logistic(X[A], y[A], C))
        wb = probe_direction(_fit_logistic(X[B], y[B], C))
        da, db = diffmean_direction(X[A], y[A]), diffmean_direction(X[B], y[B])
        pp.append(float(wa @ wb)); dd.append(float(da @ db)); pd.append(float(wa @ da))
    return {"probe_probe": np.array(pp), "dm_dm": np.array(dd), "probe_dm": np.array(pd)}


# -- rivals on the same rows ---------------------------------------------------------

def text_features_fit_score(texts_tr, dense_tr, y_tr, texts_te, dense_te, C=1.0):
    """Amendment 3 #6 text baseline: TF-IDF 1-2-grams of the conversation so
    far + dense [persona one-hot, round, cumulative reply length], fit on the
    train fold only."""
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000,
                          sublinear_tf=True)
    Ttr = vec.fit_transform(texts_tr); Tte = vec.transform(texts_te)
    sc = StandardScaler().fit(dense_tr)
    Dtr = csr_matrix(sc.transform(dense_tr)); Dte = csr_matrix(sc.transform(dense_te))
    lr = LogisticRegression(C=C, max_iter=2000).fit(hstack([Ttr, Dtr]), y_tr)
    return lr.predict_proba(hstack([Tte, Dte]))[:, 1]


def text_baseline_oof(texts, dense, y, groups, k=5, seed=0, C=1.0):
    folds = _group_folds(groups, k, seed)
    s = np.zeros(len(y))
    texts = np.asarray(texts, dtype=object)
    for f in range(k):
        te = folds == f; tr = ~te
        s[te] = text_features_fit_score(list(texts[tr]), dense[tr], y[tr],
                                        list(texts[te]), dense[te], C)
    return s


def report_features(trials_rows_ev):
    """Placeholder hook: the notebook builds the [n, 7] E[v] matrix itself
    from the cache; kept here so the rival's row alignment is tested."""
    return np.asarray(trials_rows_ev, float)


def join_judge(rows, judge_jsonl, key="p_leak"):
    """Align ask-an-LLM judgments to rows by (trial, round). Returns (scores
    with nan where absent/errored, n_matched, n_error)."""
    s = np.full(len(rows["trial"]), np.nan)
    if not os.path.exists(judge_jsonl):
        return s, 0, 0
    idx = {(t, int(r)): i for i, (t, r) in enumerate(zip(rows["trial"], rows["rnd"]))}
    latest = {}
    for line in open(judge_jsonl):
        if line.strip():
            d = json.loads(line); latest[(d["trial"], int(d["round"]))] = d   # last record wins
    n_err = 0
    for k, d in latest.items():
        i = idx.get(k)
        if i is None:
            continue
        if d.get("error") or d.get(key) is None:
            n_err += 1; continue
        s[i] = float(d[key])
    return s, int(np.sum(~np.isnan(s))), n_err


# -- confusion matrices at operating points ------------------------------------

def confusion(scores, y, thr):
    """Counts at score >= thr, plus precision/recall/specificity."""
    s, y = np.asarray(scores, float), np.asarray(y, bool)
    ok = ~np.isnan(s); s, y = s[ok], y[ok]
    pred = s >= thr
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    return {"thr": float(thr), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": int(ok.sum()),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "flag_rate": (tp + fp) / max(ok.sum(), 1)}


def threshold_prevalence(scores, y):
    """Flag exactly the top prevalence-fraction of rows (as many alarms as
    there are true positives) — the operating point that makes precision
    equal recall and needs no tuning on labels beyond the base rate."""
    s, y = np.asarray(scores, float), np.asarray(y, bool)
    ok = ~np.isnan(s); s, y = s[ok], y[ok]
    k = int(y.sum())
    if k == 0 or k >= len(s):
        return float("nan")
    return float(np.sort(s)[::-1][k - 1])


def threshold_youden(scores, y):
    """Threshold maximizing TPR - FPR on these rows (optimistic: tuned on the
    evaluated labels; report next to the prevalence point, never alone)."""
    s, y = np.asarray(scores, float), np.asarray(y, bool)
    ok = ~np.isnan(s); s, y = s[ok], y[ok]
    best, best_thr = -np.inf, float("nan")
    for thr in np.unique(s):
        pred = s >= thr
        tpr = (pred & y).sum() / max(y.sum(), 1); fpr = (pred & ~y).sum() / max((~y).sum(), 1)
        if tpr - fpr > best:
            best, best_thr = tpr - fpr, float(thr)
    return best_thr


def trial_level(scores, y, trial, agg="max"):
    """Collapse rows to one score per trial (max over its pre-event rows =
    'the monitor fired at some point') and one label per trial."""
    s, y, trial = np.asarray(scores, float), np.asarray(y, bool), np.asarray(trial)
    ut = np.unique(trial); out_s, out_y = [], []
    for t in ut:
        m = (trial == t) & ~np.isnan(s)
        if not m.any():
            continue
        out_s.append(s[m].max() if agg == "max" else s[m].mean()); out_y.append(bool(y[trial == t][0]))
    return np.array(out_s), np.array(out_y), ut
