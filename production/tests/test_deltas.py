import numpy as np

from driftlab.deltas import (build_paired_rows, evaluate, oof_probe_eval,
                             pressure_axis_scores, item_difficulty, clock_axis,
                             speed_features)


def _trial(name, persona, item, outcome, ev, K, rng, push=None, d=12):
    # rounds 1..K for user; assistant rounds 1..K (leak) or 1..K-1 (left at K)
    Ru = np.arange(1, K + 1)
    Ra = np.arange(1, K + 1) if outcome == "leaked" else np.arange(1, K)
    ru = rng.normal(size=(K, 2, d)).astype(np.float32)
    ra = rng.normal(size=(len(Ra), 2, d)).astype(np.float32)
    if push is not None:
        ru[:, 1] += push  # user states carry the signal at layer 1
    return {"resid": ra, "resid_user": ru, "resid_rounds": Ra, "user_rounds": Ru, "ev": None,
            "meta": {"trial": name, "persona": persona, "item_id": item, "outcome": outcome,
                     "leak_round": ev if outcome == "leaked" else None,
                     "leave_round": ev if outcome == "left" else None}}


def _world(signal_scale):
    rng = np.random.default_rng(0)
    w = rng.normal(size=12); w /= np.linalg.norm(w)
    trials = []
    for i in range(24):
        q = f"q{i:02d}"
        for rep in range(3):
            leak = (i + rep) % 2 == 0
            push = signal_scale * w if leak else None
            trials.append(_trial(f"t{i}_{rep}", "supportive", q, "leaked" if leak else "left",
                                 4, 4, rng, push))
            trials.append(_trial(f"n{i}_{rep}", "neutral", q, "left", 5, 5, rng, None))
    return trials


def test_rows_alignment_and_leads():
    rng = np.random.default_rng(1)
    t = _trial("a", "supportive", "q00", "leaked", 3, 3, rng)
    rows = build_paired_rows([t], layer=1)
    # rounds 2 and 3 (r=1 has no previous assistant state); lead 1 then 0
    assert rows["rnd"].tolist() == [2, 3] and rows["lead"].tolist() == [1, 0]
    np.testing.assert_allclose(rows["U"][0], t["resid_user"][1, 1])
    np.testing.assert_allclose(rows["A_prev"][0], t["resid"][0, 1])
    np.testing.assert_allclose(rows["D"], rows["U"] - rows["A_prev"])
    t2 = _trial("b", "supportive", "q00", "left", 4, 4, rng)   # assistant rounds 1..3
    rows2 = build_paired_rows([t2], layer=1)
    assert rows2["rnd"].tolist() == [2, 3, 4] and rows2["lead"].tolist() == [2, 1, 0]
    t3 = _trial("c", "supportive", "q00", "leaked", 1, 1, rng)  # round-1 leak excluded
    assert len(build_paired_rows([t3], layer=1)["rnd"]) == 0


def test_probe_finds_signal_and_null_is_chance():
    rows = build_paired_rows(_world(3.0), layer=1)
    sup = {k: (v[rows["persona"] == "supportive"] if isinstance(v, np.ndarray) else v)
           for k, v in rows.items()}
    res, _, _ = oof_probe_eval(sup["U"], sup, n_perm=3)
    assert res["probe"]["leads_1_3"]["auc_rc"] > 0.9
    assert res["diffmean"]["leads_1_3"]["auc_rc"] > 0.9
    assert abs(res["probe"]["leads_1_3"]["null_mean"] - 0.5) < 0.15
    rows0 = build_paired_rows(_world(0.0), layer=1)
    sup0 = {k: (v[rows0["persona"] == "supportive"] if isinstance(v, np.ndarray) else v)
            for k, v in rows0.items()}
    res0, _, _ = oof_probe_eval(sup0["U"], sup0)
    assert abs(res0["probe"]["leads_1_3"]["auc_rc"] - 0.5) < 0.15


def test_pressure_axis_is_label_free_and_held_out():
    rows = build_paired_rows(_world(3.0), layer=1)
    s, axes = pressure_axis_scores(rows, feat="U")
    assert len(s) == (rows["persona"] == "supportive").sum()
    assert set(axes) == {2, 3, 4} and all(abs(np.linalg.norm(a) - 1) < 1e-5 for a in axes.values())
    sup = {k: (v[rows["persona"] == "supportive"] if isinstance(v, np.ndarray) else v)
           for k, v in rows.items()}
    ev = evaluate(s, sup)
    assert 0 <= ev["leads_1_3"]["auc_rc"] <= 1


def test_item_difficulty_and_speed():
    trials = _world(10.0)
    r = item_difficulty(trials, layer=1, state="user", rnd=1, n_perm=20)
    assert len(r["items"]) == 24 and r["spearman"] > 0.5
    rows = build_paired_rows(trials, layer=1)
    c = clock_axis(rows, lo=1, hi=3)
    f = speed_features(rows, c)
    assert f["push_norm"].shape == rows["rnd"].shape
    assert np.all(np.abs(f["push_cos_clock"]) <= 1 + 1e-6)
