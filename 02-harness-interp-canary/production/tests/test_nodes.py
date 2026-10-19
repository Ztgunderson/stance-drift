"""Tests for driftlab.nodes (Amendment 4 machinery) + node_resample helpers.

No GPU, no model, no server: synthetic cache npz/meta files and a synthetic
resampler JSON with a planted linear propensity signal.
"""

import json

import numpy as np
import pytest

from driftlab.nodes import (compare_channels, join_states, load_nodes,
                            propensity_regression, wilson)
from tutorbench.node_resample import collect_samples

RNG = np.random.default_rng(11)
D, LAYERS = 16, 4          # n_layers+1 = 4; planted layer = 2
SIG_LAYER = 2
TRUE_DIR = RNG.normal(size=D)
TRUE_DIR /= np.linalg.norm(TRUE_DIR)


@pytest.fixture()
def synth(tmp_path):
    """Synthetic cache dir + nodes json: 15 trials x rounds {2,3}."""
    cache = tmp_path / "cache"
    cache.mkdir()
    nodes = []
    for i in range(15):
        item = f"q{i:02d}"
        user_rounds = np.array([1, 2, 3])
        resid_user = RNG.normal(size=(3, LAYERS, D)).astype(np.float16)
        node_p = {}
        for r in (2, 3):
            p = float(RNG.uniform(0, 1))
            node_p[r] = p
            resid_user[r - 1, SIG_LAYER] += (6 * p * TRUE_DIR).astype(np.float16)
        np.savez(cache / f"trial{i:02d}.npz",
                 resid=RNG.normal(size=(3, LAYERS, D)).astype(np.float16),
                 resid_rounds=np.array([1, 2, 3]),
                 resid_user=resid_user, user_ends=np.array([5, 15, 25]),
                 user_rounds=user_rounds,
                 report_ev=RNG.uniform(0, 10, size=(3, 7)).astype(np.float32),
                 report_probs=np.zeros((3, 7, 11), np.float32))
        (cache / f"trial{i:02d}.json").write_text(json.dumps({
            "trial": f"trial{i:02d}", "persona": "supportive",
            "item_id": item, "outcome": "leaked",
            "leak_round": 4, "leave_round": None}))
        k = 25
        for r in (2, 3):
            nleak = int(round(node_p[r] * k))
            nodes.append({"trial": f"supportive/{item}", "round": r,
                          "persona": "supportive", "item_id": item,
                          "orig_outcome": "leaked", "orig_leak_round": 4,
                          "orig_leave_round": None, "k": k,
                          "counts": {"leak": nleak, "leave": k - nleak,
                                     "continue": 0},
                          "samples": []})
    out = tmp_path / "nodes.json"
    out.write_text(json.dumps({"model": "synth", "log_dir": "synth",
                               "nodes": nodes}))
    return cache, out


def test_wilson_basic():
    p, lo, hi = wilson(18, 24)
    assert 0.74 < p < 0.76 and lo < p < hi
    assert wilson(0, 0) == (pytest.approx(np.nan, nan_ok=True),) * 3 \
        or np.isnan(wilson(0, 0)[0])


def test_load_nodes(synth):
    _, nodes_json = synth
    df = load_nodes(nodes_json)
    assert len(df) == 30
    assert {"P_leak", "P_leak_lo", "P_leak_hi", "round"} <= set(df.columns)
    assert ((df.P_leak >= 0) & (df.P_leak <= 1)).all()
    assert (df.P_leak_lo <= df.P_leak).all() and (df.P_leak <= df.P_leak_hi).all()


def test_join_alignment_and_drops(synth, tmp_path):
    cache, nodes_json = synth
    df = load_nodes(nodes_json)
    # add a node with a round that has no user row, and one with no cache file
    import pandas as pd
    df2 = pd.concat([df, pd.DataFrame([
        {**df.iloc[0].to_dict(), "round": 7},
        {**df.iloc[0].to_dict(), "item_id": "qZZ",
         "trial": "supportive/qZZ"}])], ignore_index=True)
    with pytest.warns(UserWarning, match="dropped 2"):
        kept, X, R = join_states(df2, str(cache), layer=SIG_LAYER)
    assert len(kept) == 30 and X.shape == (30, D) and R.shape == (30, 7)
    # pre-decision report row is round-1's note for a round-2 node
    assert not np.isnan(R).all()


def test_join_assistant_fallback(synth):
    cache, nodes_json = synth
    # strip resid_user from one file -> that trial's nodes fall back
    f = sorted(cache.glob("*.npz"))[0]
    z = dict(np.load(f))
    z.pop("resid_user"); z.pop("user_rounds"); z.pop("user_ends")
    np.savez(f, **z)
    df = load_nodes(nodes_json)
    with pytest.warns(UserWarning, match="old-schema"):
        kept, X, R = join_states(df, str(cache), layer=SIG_LAYER)
    assert len(kept) == 30            # fallback keeps the rows


def test_regression_recovers_planted_signal(synth):
    cache, nodes_json = synth
    df = load_nodes(nodes_json)
    kept, X, R = join_states(df, str(cache), layer=SIG_LAYER)
    y = kept.P_leak.to_numpy()
    res = propensity_regression(X, y, kept.trial.to_numpy(), n_boot=100)
    assert res["spearman"] > 0.55, res
    assert res["diffmean_axis_spearman"] > 0.5, res
    # wrong layer carries no signal
    kept0, X0, _ = join_states(df, str(cache), layer=0)
    res0 = propensity_regression(X0, kept0.P_leak.to_numpy(),
                                 kept0.trial.to_numpy(), n_boot=100)
    assert res0["spearman"] < res["spearman"] - 0.3


def test_compare_channels_baselines_flat(synth):
    cache, nodes_json = synth
    df = load_nodes(nodes_json)
    kept, X, R = join_states(df, str(cache), layer=SIG_LAYER)
    out = compare_channels(kept, X, R, n_boot=100)
    assert out["state"]["spearman"] > 0.55
    # persona is constant and round is independent of propensity by design
    assert not (out["baseline"]["spearman"] > 0.4)
    # reports were pure noise by construction
    assert out["reports"] is None or not (out["reports"]["spearman"] > 0.4)


def test_collect_samples_threaded_and_sequential():
    calls = []

    def draw():
        calls.append(1)
        return {"action": "leave"}

    seq = collect_samples(5, 1, draw)
    thr = collect_samples(20, 4, draw)
    assert len(seq) == 5 and len(thr) == 20
    assert len(calls) == 25
    assert all(s == {"action": "leave"} for s in seq + thr)
