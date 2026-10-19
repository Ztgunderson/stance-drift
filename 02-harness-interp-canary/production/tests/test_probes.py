import json

import numpy as np
import pytest

from driftlab.probes import (auc, build_rows, bootstrap_groups, diffmean_oof,
                             join_judge, layer_curve, nested_probe,
                             permute_trial_labels, round_conditional_auc,
                             split_half_stability, subset, text_baseline_oof)


def _trial(name, persona, item, outcome, leak_round=None, leave_round=None,
           n_turns=6, L=3, d=16, signal=None, rng=None):
    resid = rng.normal(size=(n_turns, L, d)).astype(np.float32)
    if signal is not None:
        resid[:, 1, :] += signal
    return {"resid": resid, "resid_user": None, "resid_rounds": np.arange(1, n_turns + 1),
            "user_rounds": None, "ev": rng.uniform(0, 10, size=(n_turns, 7)),
            "meta": {"trial": name, "persona": persona, "item_id": item, "outcome": outcome,
                     "leak_round": leak_round, "leave_round": leave_round}}


@pytest.fixture()
def world():
    rng = np.random.default_rng(0)
    w = rng.normal(size=16); w /= np.linalg.norm(w)
    trials = []
    for i in range(24):
        item = f"q{i:02d}"
        leak = i % 2 == 0
        for rep in range(2):
            name = f"sup_{item}_r{rep}"
            if leak:
                trials.append(_trial(name, "supportive", item, "leaked", leak_round=4,
                                     n_turns=3, signal=2.0 * w, rng=rng))
            else:
                trials.append(_trial(name, "supportive", item, "left", leave_round=6,
                                     n_turns=5, signal=-2.0 * w, rng=rng))
    trials.append(_trial("r1", "supportive", "q99", "leaked", leak_round=1, n_turns=1, rng=rng))
    return trials, w


def test_build_rows_pre_event_only_and_r1_excluded(world):
    trials, _ = world
    rows = build_rows(trials)
    assert rows["n_r1_leaks"] == 1
    assert rows["X"].shape[1:] == (3, 16)
    leak_rows = rows["will_leak"]
    assert rows["rnd"][leak_rows].max() == 3          # leak at 4 -> rows 1..3
    assert rows["rnd"][~leak_rows].max() == 5         # leave at 6 -> rows 1..5
    assert set(rows["lead"][leak_rows]) == {1, 2, 3}
    assert all(g.startswith("supportive/") for g in rows["group"])


def test_auc_and_round_conditional():
    assert auc([0.9, 0.8, 0.1], [1, 1, 0]) == 1.0
    assert auc([0.5, 0.5], [1, 0]) == 0.5
    assert np.isnan(auc([0.1, 0.2], [1, 1]))
    s = np.array([0.9, 0.1, 0.9, 0.1, 0.2, 0.8, 0.3, 0.3])
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0], bool)
    rnd = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    a, table = round_conditional_auc(s, y, rnd, min_n=2)
    assert table[0][0] == 1 and table[0][1] == 1.0     # round 1 perfectly separated
    # round 2: pos [0.2, 0.3] vs neg [0.8, 0.3] -> one tie of four pairs = 0.125
    assert table[1][0] == 2 and table[1][1] == 0.125
    assert abs(a - 0.5625) < 1e-9                      # equal positive counts


def test_nested_probe_finds_signal_layer_and_reports_choices(world):
    trials, _ = world
    rows = build_rows(trials)
    res = nested_probe(rows["X"], rows["will_leak"], rows["group"],
                       layer_grid=[0, 1, 2], C_grid=(1.0,), k_outer=3, k_inner=2)
    assert len(res["choices"]) == 3
    assert all(L == 1 for L, _ in res["choices"])       # planted in layer 1
    assert auc(res["scores"], rows["will_leak"]) > 0.9
    assert all(abs(np.linalg.norm(d) - 1) < 1e-5 for d in res["directions"])
    # groups never straddle outer folds
    for g in np.unique(rows["group"]):
        assert len(set(res["folds"][rows["group"] == g])) == 1


def test_layer_curve_null_band_near_chance(world):
    trials, _ = world
    rows = build_rows(trials)
    lc = layer_curve(rows["X"], rows["will_leak"], rows["group"], layers=[0, 1],
                     k=3, n_perm=5, trial=rows["trial"])
    assert lc["auc"][1] > 0.9 and lc["auc"][0] < 0.75
    assert 0.3 < lc["null_mean"][1] < 0.7 and lc["null_sd"][1] >= 0


def test_permute_trial_labels_keeps_rows_consistent():
    y = np.array([1, 1, 0, 0, 0, 1], bool)
    trial = np.array(["a", "a", "b", "b", "c", "d"])
    yp = permute_trial_labels(y, trial, np.random.default_rng(0))
    assert yp[0] == yp[1] and yp[2] == yp[3]
    assert yp.sum() == y.sum()                          # exact same label multiset per trial... (4 trials, 2 pos)


def test_diffmean_and_stability(world):
    trials, w = world
    rows = build_rows(trials)
    s, dirs = diffmean_oof(rows["X"][:, 1], rows["will_leak"], rows["group"], k=3)
    assert auc(s, rows["will_leak"]) > 0.9
    assert abs(float(dirs[0] @ w)) > 0.8
    st = split_half_stability(rows["X"][:, 1], rows["will_leak"], rows["group"], n_rep=3)
    assert st["probe_probe"].mean() > 0.5 and st["dm_dm"].mean() > 0.9


def test_bootstrap_ci_brackets_point_estimate(world):
    trials, _ = world
    rows = build_rows(trials)
    s = rows["X"][:, 1, 0]
    point = auc(s, rows["will_leak"])
    lo, hi, vals = bootstrap_groups(lambda idx: auc(s[idx], rows["will_leak"][idx]),
                                    rows["group"], n_boot=50)
    assert lo <= point <= hi and len(vals) == 50


def test_text_baseline_oof_runs_and_separates_obvious_text():
    texts = ["give me the answer now please"] * 12 + ["let us think step by step"] * 12
    y = np.array([True] * 12 + [False] * 12)
    groups = np.array([f"g{i//2}" for i in range(24)])
    dense = np.column_stack([np.arange(24) % 3, np.arange(24)]).astype(float)
    s = text_baseline_oof(texts, dense, y, groups, k=3)
    assert auc(s, y) > 0.9


def test_join_judge_alignment_and_errors(world, tmp_path):
    trials, _ = world
    rows = build_rows(trials)
    p = tmp_path / "j.jsonl"
    recs = [{"trial": rows["trial"][0], "round": int(rows["rnd"][0]), "p_leak": 0.7, "error": None},
            {"trial": rows["trial"][1], "round": int(rows["rnd"][1]), "p_leak": None, "error": "unparsable"},
            {"trial": "nope", "round": 1, "p_leak": 0.1, "error": None}]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    s, n_ok, n_err = join_judge(rows, str(p))
    assert n_ok == 1 and n_err == 1 and s[0] == 0.7 and np.isnan(s[1])
    sub = subset(rows, rows["will_leak"])
    assert sub["X"].shape[0] == rows["will_leak"].sum() and sub["n_r1_leaks"] == rows["n_r1_leaks"]


def test_confusion_and_thresholds():
    from driftlab.probes import confusion, threshold_prevalence, threshold_youden, trial_level
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]); y = np.array([1, 1, 0, 1, 0, 0], bool)
    thr = threshold_prevalence(s, y)                 # flag top 3
    c = confusion(s, y, thr)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 2)
    assert abs(c["precision"] - c["recall"]) < 1e-9 and c["n"] == 6
    ty = threshold_youden(s, y); assert ty in (0.3, 0.8)   # both give TPR-FPR = 2/3 (tie)
    ts, tyy, ut = trial_level(np.array([0.1, 0.9, 0.2, 0.3]), np.array([1, 1, 0, 0], bool),
                              np.array(["a", "a", "b", "b"]))
    assert list(ut) == ["a", "b"] and ts[0] == 0.9 and tyy[0] and not tyy[1]
    assert np.isnan(threshold_prevalence(np.array([0.1, 0.2]), np.array([0, 0], bool)))
