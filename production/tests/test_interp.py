import numpy as np

from driftlab.interp import _group_folds, dim_layer_map, gavein_turn_probe, ridge_cv_r


def test_group_folds_never_split_a_group():
    groups = np.repeat(np.arange(20), 8)
    folds = _group_folds(groups, k=5, seed=0)
    for g in np.unique(groups):
        assert len(set(folds[groups == g])) == 1
    assert set(folds) == set(range(5))


def synthetic_cache(seed=0, n_tr=30, R=6, L=4, d=32, signal_layer=2):
    """Residuals are noise everywhere except signal_layer, where dim 0 is
    linearly embedded; gave_in is readable from the same layer's last turn."""
    rng = np.random.default_rng(seed)
    resid = rng.normal(size=(n_tr, R + 1, L, d)).astype(np.float32)
    dims = rng.uniform(0, 10, size=(n_tr, R, 5)).astype(np.float32)
    w = rng.normal(size=d)
    gave = rng.random(n_tr) < 0.5
    for t in range(R):
        resid[:, t, signal_layer] += np.outer(dims[:, t, 0], w)
    resid[:, R, signal_layer] += np.outer(np.where(gave, 5.0, -5.0), w)
    return {"resid": resid, "dims": dims, "gave_in": gave}


def test_ridge_recovers_planted_signal():
    c = synthetic_cache()
    n_tr, R = c["dims"].shape[:2]
    Z = c["resid"][:, :R, 2].reshape(n_tr * R, -1).astype(np.float64)
    y = c["dims"][:, :, 0].reshape(-1).astype(np.float64)
    groups = np.repeat(np.arange(n_tr), R)
    assert ridge_cv_r(Z, y, groups, lam=10.0) > 0.9


def test_dim_layer_map_localizes_signal_and_control_is_null():
    r, r_sh = dim_layer_map(synthetic_cache(), lam=10.0)
    assert r.shape == (4, 5)
    assert r[2, 0] > 0.8                      # planted: dim 0 at layer 2
    mask = np.ones_like(r, dtype=bool); mask[2, 0] = False
    assert np.all(np.abs(r[mask]) < 0.45)     # nothing planted elsewhere
    assert np.all(np.abs(r_sh) < 0.45)        # shuffled control near zero


def test_dim_layer_map_control_is_finite_with_unparsed_notes():
    # regression 2026-08-27: NaN notes permuted into the control labels made
    # the shuffled r NaN (mask was computed on the true labels only)
    c = synthetic_cache()
    c["dims"][::4, 1, :] = np.nan
    r, r_sh = dim_layer_map(c, lam=10.0)
    assert np.isfinite(r).all() and np.isfinite(r_sh).all()


def test_gavein_turn_probe_reads_final_turn_only():
    acc, acc_sh = gavein_turn_probe(synthetic_cache(), lam=10.0)
    L, T = acc.shape
    assert (L, T) == (4, 7)
    assert acc[2, T - 1] > 0.85               # planted at layer 2, last turn
    assert np.mean(acc_sh) < 0.65             # control hovers near chance
