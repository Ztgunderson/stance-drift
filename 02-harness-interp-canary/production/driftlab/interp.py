"""Linear probes from residual-stream activations to behavior.

numpy-only on purpose: every step is a handful of linear-algebra lines a
reviewer can check in the notebook, and the Jetson venvs stay untouched.

Two leakage rules enforced throughout:
  1. Cross-validation folds NEVER split a trial — turns of one conversation
     are highly correlated, so trial-level grouping is the unit of evidence.
  2. Every probe is reported next to a shuffled-label control run through
     the identical pipeline; signal is the gap, not the raw number.
"""

import numpy as np


def _group_folds(groups, k, seed):
    """Assign each unique group (trial) to one of k folds, shuffled."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def _dual_ridge(Ztr, ytr, Zte, lam):
    """Ridge in the dual: with n << d (hundreds of turns, d=4096), solve the
    [n, n] kernel system instead of the [d, d] one."""
    mu, ym = Ztr.mean(0), ytr.mean()
    Ztr, Zte = Ztr - mu, Zte - mu
    K = Ztr @ Ztr.T
    alpha = np.linalg.solve(K + lam * np.eye(len(K)), ytr - ym)
    return Zte @ (Ztr.T @ alpha) + ym


def ridge_cv_r(Z, y, groups, lam=1e3, k=5, seed=0):
    """Group-k-fold CV Pearson r between ridge predictions and y."""
    folds = _group_folds(groups, k, seed)
    pred = np.empty_like(y, dtype=np.float64)
    for f in range(k):
        te = folds == f
        pred[te] = _dual_ridge(Z[~te], y[~te], Z[te], lam)
    if np.std(pred) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(pred, y)[0, 1])


def dim_layer_map(cache, lam=1e3, k=5, seed=0):
    """CV correlation for decoding each self-report dimension from each
    layer's residual, with matched shuffled-label control.

    Samples are (trial, round) pairs: resid turn t (reply of round t+1)
    against the round t+1 note. Returns (r, r_shuffled), both [layer, 5].
    """
    resid, dims = cache["resid"], cache["dims"]
    n_tr, R = dims.shape[:2]
    L = resid.shape[2]
    X = resid[:, :R].reshape(n_tr * R, L, -1)
    Y = dims.reshape(n_tr * R, dims.shape[2])
    groups = np.repeat(np.arange(n_tr), R)
    rng = np.random.default_rng(seed)
    r = np.zeros((L, Y.shape[1]))
    r_sh = np.zeros_like(r)
    for j in range(Y.shape[1]):
        # shuffle labels at the TRIAL level so the control keeps the
        # within-trial correlation structure the real probe must beat
        perm = rng.permutation(n_tr)
        y_sh_full = dims[perm][:, :, j].reshape(n_tr * R)
        ok = ~np.isnan(Y[:, j])
        # the control needs its own mask: an unparsed note can land anywhere
        # after the permutation, and a NaN label silently poisons corrcoef
        ok_sh = ok & ~np.isnan(y_sh_full)
        for layer in range(L):
            r[layer, j] = ridge_cv_r(X[ok, layer].astype(np.float64),
                                     Y[ok, j], groups[ok], lam, k, seed)
            r_sh[layer, j] = ridge_cv_r(X[ok_sh, layer].astype(np.float64),
                                        y_sh_full[ok_sh], groups[ok_sh],
                                        lam, k, seed)
    return r, r_sh


def gavein_turn_probe(cache, lam=1e3, k=5, seed=0):
    """Accuracy of predicting the trial's FINAL outcome from a single turn's
    residual, per (layer, turn) — 'when does the outcome become readable'.

    Ridge on +/-1 labels, thresholded at 0; group CV is trivially satisfied
    (one sample per trial per cell) but folds are still per-trial so every
    (layer, turn) cell uses the same partition. Returns (acc, acc_shuffled),
    both [layer, turns].
    """
    resid, gave = cache["resid"], cache["gave_in"]
    n_tr, T, L = resid.shape[:3]
    y = np.where(gave, 1.0, -1.0)
    rng = np.random.default_rng(seed)
    y_sh = y[rng.permutation(n_tr)]
    folds = _group_folds(np.arange(n_tr), k, seed)
    acc = np.zeros((L, T))
    acc_sh = np.zeros_like(acc)
    for layer in range(L):
        for t in range(T):
            Z = resid[:, t, layer].astype(np.float64)
            for lab, out in ((y, acc), (y_sh, acc_sh)):
                pred = np.empty(n_tr)
                for f in range(k):
                    te = folds == f
                    pred[te] = _dual_ridge(Z[~te], lab[~te], Z[te], lam)
                out[layer, t] = float(np.mean(np.sign(pred) == lab))
    return acc, acc_sh
