"""Driver for brainstorm items 1, 2, 4, 5 — see driftlab/deltas.py.
Writes results/probes/deltas.json (+ per-row score arrays npz)."""

import json
import os
import sys
import time

import numpy as np

from driftlab.probes import load_cache, round_conditional_auc
from driftlab.deltas import (build_paired_rows, sel, evaluate, oof_probe_eval,
                             pressure_axis_scores, item_difficulty, clock_axis,
                             speed_features)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="microscope/cache/qwen35-9b-v1")
    ap.add_argument("--out", default="results/probes/deltas.json")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--sweep-stride", type=int, default=4)
    ap.add_argument("--n-perm", type=int, default=10)
    ap.add_argument("--seed", type=int, default=903)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    trials = load_cache(a.cache)
    n_layers = trials[0]["resid"].shape[1]
    log(f"{len(trials)} trials, {n_layers} layers")
    sweep = list(range(0, n_layers, 8 if a.quick else a.sweep_stride))
    if a.layer not in sweep:
        sweep.append(a.layer)
    n_perm = 2 if a.quick else a.n_perm
    R = {"provenance": {"cache": a.cache, "layer": a.layer, "sweep": sweep, "n_perm": n_perm,
                        "seed": a.seed, "C": 1.0, "folds": "item-grouped k=5 inside supportive",
                        "rows": "paired rows r>=2, r<=event (lead 0 allowed at user position)",
                        "started": time.strftime("%Y-%m-%d %H:%M")}}
    arrays = {}

    # ---- item 1: layer sweep of U / A_prev / D, supportive will-leak ----------------
    R["sweep"] = {}
    for L in sweep:
        rows = build_paired_rows(trials, L)
        sup = sel(rows, rows["persona"] == "supportive")
        if L == sweep[0]:
            R["provenance"]["n_rows_supportive"] = int(len(sup["rnd"]))
            R["provenance"]["n_pos_leads_1_3"] = int((sup["will_leak"] & (sup["lead"] >= 1) & (sup["lead"] <= 3)).sum())
            R["provenance"]["n_pos_lead_0"] = int((sup["will_leak"] & (sup["lead"] == 0)).sum())
            R["provenance"]["rounds"] = {int(r): int((sup["rnd"] == r).sum()) for r in np.unique(sup["rnd"])}
        R["sweep"][L] = {}
        for feat in ("U", "A_prev", "D"):
            res, s_pr, s_dm = oof_probe_eval(sup[feat], sup, C=1.0, seed=a.seed,
                                             n_perm=(n_perm if L == a.layer else 0))
            R["sweep"][L][feat] = res
            if L == a.layer:
                arrays[f"probe_{feat}"] = s_pr; arrays[f"dm_{feat}"] = s_dm
            log(f"L{L:2d} {feat:6s} probe {res['probe']['leads_1_3']['auc_rc']:.3f} "
                f"(lead0 {res['probe']['lead_0']['auc_rc']:.3f}) | diffmean "
                f"{res['diffmean']['leads_1_3']['auc_rc']:.3f} (lead0 {res['diffmean']['lead_0']['auc_rc']:.3f})")
        if L == a.layer:
            rows_main, sup_main = rows, sup
    # C=0.01 check at the main layer (the formal assistant run chose 0.01)
    R["C_001"] = {}
    for feat in ("U", "A_prev", "D"):
        res, _, _ = oof_probe_eval(sup_main[feat], sup_main, C=0.01, seed=a.seed)
        R["C_001"][feat] = res["probe"]
        log(f"L{a.layer} {feat} C=0.01 probe {res['probe']['leads_1_3']['auc_rc']:.3f}")
    for k in ("will_leak", "rnd", "lead", "item", "trial"):
        arrays[k] = sup_main[k]

    # ---- item 2: pressure axis (supportive - neutral, per round, label-free) --------
    R["pressure"] = {}
    for feat in ("U", "D"):
        for held in (True, False):
            s, axes = pressure_axis_scores(rows_main, feat=feat, held_out=held)
            ok = ~np.isnan(s)
            ev = evaluate(np.where(ok, s, np.nanmedian(s)), sup_main, n_perm=200, seed=a.seed)
            key = f"{feat}_{'heldout' if held else 'insample'}"
            R["pressure"][key] = ev
            arrays[f"pressure_{key}"] = s
            log(f"pressure {key}: leads1-3 {ev['leads_1_3']['auc_rc']:.3f} "
                f"(null {ev['leads_1_3']['null_mean']:.2f}±{ev['leads_1_3']['null_sd']:.2f}) "
                f"lead0 {ev['lead_0']['auc_rc']:.3f}  per-round {ev['leads_1_3']['per_round']}")
            if held and feat == "U":
                # cosine of the per-round pressure axes with each other and with the clock
                ck = clock_axis(rows_main)
                R["pressure"]["axis_cos_clock"] = {r: float(ax @ ck) for r, ax in axes.items()}
                rs = sorted(axes)
                R["pressure"]["axis_cos_adjacent"] = {f"{rs[i]}-{rs[i+1]}": float(axes[rs[i]] @ axes[rs[i+1]])
                                                      for i in range(len(rs) - 1)}
    # aggressor - neutral as a second reference axis
    s, _ = pressure_axis_scores(rows_main, feat="U", target_persona="supportive",
                                ref_persona="aggressor", held_out=True)
    ok = ~np.isnan(s)
    R["pressure"]["U_heldout_vs_aggressor"] = evaluate(np.where(ok, s, np.nanmedian(s)), sup_main,
                                                       n_perm=200, seed=a.seed)
    log(f"pressure U vs aggressor: {R['pressure']['U_heldout_vs_aggressor']['leads_1_3']['auc_rc']:.3f}")

    # ---- item 4: item difficulty from the round-1 state ----------------------------------
    R["difficulty"] = {}
    for persona in ("supportive", "neutral"):
        for state in ("assistant", "user"):
            d = item_difficulty(trials, a.layer, persona=persona, state=state, rnd=1,
                                n_perm=(20 if a.quick else 500), seed=a.seed)
            R["difficulty"][f"{persona}_{state}_r1"] = d
            log(f"difficulty {persona} {state} r1: spearman {d['spearman']:.2f} "
                f"(null {d['null_mean']:.2f}±{d['null_sd']:.2f}, p {d['p_perm']:.3f})")
    # cross: neutral round-1 state -> supportive leak rate (the problem, less the persona)
    sup_rate = dict(zip(R["difficulty"]["supportive_assistant_r1"]["items"],
                        R["difficulty"]["supportive_assistant_r1"]["rate"]))
    neu = R["difficulty"]["neutral_assistant_r1"]
    from scipy.stats import spearmanr
    R["difficulty"]["rate_corr_supportive_vs_neutral"] = float(
        spearmanr([sup_rate[q] for q in neu["items"]], neu["rate"]).correlation)
    # layer sweep for supportive/assistant/r1 (cheap)
    R["difficulty"]["sweep_supportive_assistant_r1"] = {}
    for L in sweep:
        d = item_difficulty(trials, L, persona="supportive", state="assistant", rnd=1,
                            n_perm=(20 if a.quick else 200), seed=a.seed)
        R["difficulty"]["sweep_supportive_assistant_r1"][L] = {
            "spearman": d["spearman"], "null_mean": d["null_mean"], "null_sd": d["null_sd"], "p_perm": d["p_perm"]}
    log("difficulty sweep: " + " ".join(f"L{L}:{v['spearman']:.2f}" for L, v in
                                       R["difficulty"]["sweep_supportive_assistant_r1"].items()))

    # ---- item 5: drift speed ---------------------------------------------------------------
    ck = clock_axis(rows_main)
    feats = speed_features(sup_main, ck)
    R["speed"] = {}
    for name, v in feats.items():
        ev = evaluate(v, sup_main, n_perm=200, seed=a.seed)
        R["speed"][name] = ev
        arrays[f"speed_{name}"] = v
        log(f"speed {name}: leads1-3 {ev['leads_1_3']['auc_rc']:.3f} "
            f"(null {ev['leads_1_3']['null_mean']:.2f}±{ev['leads_1_3']['null_sd']:.2f}) "
            f"lead0 {ev['lead_0']['auc_rc']:.3f}")
    # two-feature logistic (push_norm, push_along_clock) out of fold, for completeness
    from driftlab.probes import oof_scores
    Z = np.stack([feats["push_norm"], feats["push_along_clock"], feats["push_cos_clock"]], 1)
    s2 = oof_scores(Z, sup_main["will_leak"], sup_main["item"], C=1.0, seed=a.seed)
    R["speed"]["logistic_3feat_oof"] = evaluate(s2, sup_main)
    log(f"speed 3-feat logistic OOF: {R['speed']['logistic_3feat_oof']['leads_1_3']['auc_rc']:.3f}")

    R["provenance"]["elapsed_s"] = round(time.time() - T0, 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(jsonable(R), open(a.out, "w"), indent=1)
    np.savez_compressed(a.out[:-5] + "-arrays.npz", **arrays)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
