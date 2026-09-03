"""One-shot compute driver for the formal L1 notebook (Amendment 3 + 6).
Everything expensive happens here, once; the notebook only loads the
outputs and draws. Re-run when the cache grows.

Outputs (under --out-dir, prefix --tag):
  <tag>.json           scalars + tables (primary test, per-fold choices, inner
                       grid, rivals, stability, counts, errors, provenance)
  <tag>-rows.npz       row arrays (labels, rounds, leads, persona, item, trial,
                       group) + every out-of-fold score vector by method,
                       per position; layer curves + null bands
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from driftlab.probes import (auc, bootstrap_groups, build_rows, diffmean_oof,  # noqa: E402
                             join_judge, layer_curve, load_cache, nested_probe,
                             oof_scores, round_conditional_auc, split_half_stability,
                             subset, text_baseline_oof, _fit_logistic, _score,
                             probe_direction, diffmean_direction)
from driftlab.transcripts import conversation_text  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def primary_block(rows, target, layer_grid, C_grid, k_outer, k_inner, seed, n_boot):
    """Nested probe on one row set; returns (scores, block dict)."""
    y = rows[target]
    res = nested_probe(rows["X"], y, rows["group"], layer_grid, C_grid,
                       k_outer=k_outer, k_inner=k_inner, seed=seed)
    s = res["scores"]
    pooled = auc(s, y)
    rc, rc_table = round_conditional_auc(s, y, rows["rnd"])
    leads = {}
    for lead in range(1, 8):
        pm = rows["lead"] == lead
        if (y & pm).sum() < 3:
            continue
        a, tab = round_conditional_auc(s, y, rows["rnd"], pos_mask=pm)
        lo, hi, _ = bootstrap_groups(
            lambda idx: round_conditional_auc(s[idx], y[idx], rows["rnd"][idx],
                                              pos_mask=pm[idx])[0],
            rows["group"], n_boot=n_boot, seed=seed)
        leads[lead] = {"auc_rc": a, "ci": [lo, hi], "n_pos": int((y & pm).sum()),
                       "per_round": tab}
    pm13 = (rows["lead"] >= 1) & (rows["lead"] <= 3)
    a13, tab13 = round_conditional_auc(s, y, rows["rnd"], pos_mask=pm13)
    lo13, hi13, _ = bootstrap_groups(
        lambda idx: round_conditional_auc(s[idx], y[idx], rows["rnd"][idx],
                                          pos_mask=pm13[idx])[0],
        rows["group"], n_boot=n_boot, seed=seed)
    choices = res["choices"]
    layers_chosen = [c[0] for c in choices]
    agree = max(layers_chosen.count(L) for L in set(layers_chosen)) / len(layers_chosen)
    dirs = res["directions"]
    dir_cos = [float(dirs[i] @ dirs[j]) for i in range(len(dirs)) for j in range(i + 1, len(dirs))
               if choices[i][0] == choices[j][0]]
    block = {"n_rows": int(len(y)), "n_pos_rows": int(y.sum()),
             "n_trials": int(len(np.unique(rows["trial"]))),
             "n_groups": int(len(np.unique(rows["group"]))),
             "pooled_auc": pooled, "round_conditional_auc": rc, "rc_per_round": rc_table,
             "leads_1_3": {"auc_rc": a13, "ci": [lo13, hi13], "n_pos": int((y & pm13).sum()),
                           "per_round": tab13},
             "by_lead": leads, "fold_choices": choices, "layer_agreement": agree,
             "same_layer_fold_direction_cos": dir_cos,
             "inner_grid": {f"L{L}_C{C}": [float(v) for v in vals]
                            for (L, C), vals in res["inner_table"].items()}}
    return s, block, res


def rival_scores(rows, target, transcripts, judge_jsonl, layer_for_dm, k, seed):
    """All rivals on the same rows: diffmean projection (zero-fit), text
    baseline, logit-report (raw -E[v] and 7-item logistic), ask-an-LLM."""
    y = rows[target]
    out, meta = {}, {}
    s_dm, _ = diffmean_oof(rows["X"][:, layer_for_dm], y, rows["group"], k=k, seed=seed)
    out["diffmean_proj"] = s_dm
    # text baseline
    texts, cum = [], []
    for t, r in zip(rows["trial"], rows["rnd"]):
        txt, c = conversation_text(transcripts[t]["messages"], int(r)) if t in transcripts else ("", 0)
        texts.append(txt); cum.append(c)
    per_oh = np.stack([(rows["persona"] == p).astype(float)
                       for p in ("aggressor", "neutral", "supportive")], 1)
    dense = np.column_stack([per_oh, rows["rnd"].astype(float), np.array(cum, float)])
    have_text = np.array([len(t) > 0 for t in texts])
    meta["text_rows_with_transcript"] = int(have_text.sum())
    if have_text.sum() == len(texts):
        out["text"] = text_baseline_oof(texts, dense, y, rows["group"], k=k, seed=seed)
    else:
        out["text"] = np.full(len(y), np.nan)
    # persona x round only (the floor)
    rnd_oh = np.stack([(rows["rnd"] == r).astype(float) for r in range(1, 9)], 1)
    pr = np.concatenate([per_oh, rnd_oh, np.einsum("ni,nj->nij", per_oh, rnd_oh).reshape(len(y), -1)], 1)
    out["persona_round"] = oof_scores(pr + 1e-6 * np.random.default_rng(seed).normal(size=pr.shape),
                                      y, rows["group"], C=1.0, k=k, seed=seed)
    # logit report: raw sign convention (falling resolve -> leak) and 7-item logistic
    out["report_neg_ev"] = -rows["ev_mean"]
    E = rows["ev7"]
    ok = ~np.isnan(E).any(1)
    s7 = np.full(len(y), np.nan)
    if ok.sum() > 20:
        s7[ok] = oof_scores(E[ok], y[ok], rows["group"][ok], C=1.0, k=k, seed=seed)
    out["report_7item"] = s7
    # ask-an-LLM
    key = "p_leak" if target == "will_leak" else "p_leave"
    s_j, n_ok, n_err = join_judge(rows, judge_jsonl, key=key)
    out["ask_llm"] = s_j
    meta["judge_rows_matched"] = n_ok; meta["judge_rows_errored"] = n_err
    return out, meta


def eval_methods(rows, target, scores, n_boot, seed):
    """Round-conditional AUROC at leads 1-3 and per lead for every method, on
    the rows where that method has a score (reported with its n)."""
    y = rows[target]; tab = {}
    for name, s in scores.items():
        ok = ~np.isnan(s)
        r = subset(rows, ok); yy = y[ok]; ss = s[ok]
        pm13 = (r["lead"] >= 1) & (r["lead"] <= 3)
        a13, _ = round_conditional_auc(ss, yy, r["rnd"], pos_mask=pm13)
        lo, hi, _ = bootstrap_groups(
            lambda idx: round_conditional_auc(ss[idx], yy[idx], r["rnd"][idx], pos_mask=pm13[idx])[0],
            r["group"], n_boot=n_boot, seed=seed)
        by_lead = {}
        for lead in range(1, 8):
            pm = r["lead"] == lead
            if (yy & pm).sum() >= 3:
                by_lead[lead] = round_conditional_auc(ss, yy, r["rnd"], pos_mask=pm)[0]
        tab[name] = {"n_rows": int(ok.sum()), "pooled": auc(ss, yy),
                     "leads_1_3": a13, "ci": [lo, hi], "by_lead": by_lead}
    return tab


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="microscope/cache/qwen35-9b-v1")
    ap.add_argument("--transcripts", default="results/transcripts-9b.json")
    ap.add_argument("--judge", default="results/askllm/judge-qwen9b-self.jsonl")
    ap.add_argument("--out-dir", default="results/probes")
    ap.add_argument("--tag", default="formal")
    ap.add_argument("--layer-stride", type=int, default=4)
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--k-outer", type=int, default=5)
    ap.add_argument("--k-inner", type=int, default=3)
    ap.add_argument("--seed", type=int, default=902)
    ap.add_argument("--quick", action="store_true", help="tiny grids for a smoke run")
    ap.add_argument("--c-grid", default="0.01,0.1,1,10",
                    help="inverse-regularization grid for the primary block (nested choice)")
    ap.add_argument("--positions", default="assistant,user")
    ap.add_argument("--targets", default="will_leak,will_leave")
    ap.add_argument("--limit-trials", type=int, default=0, help="smoke: first N trials")
    ap.add_argument("--null-stride", type=int, default=8,
                    help="layer stride for the permutation null band")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()

    trials = load_cache(a.cache)
    if a.limit_trials:
        trials = trials[: a.limit_trials]
    n_layers = trials[0]["resid"].shape[1]
    transcripts = json.load(open(a.transcripts)) if os.path.exists(a.transcripts) else {}
    log(f"{len(trials)} trials, {n_layers} layers, {len(transcripts)} transcripts, "
        f"judge file {'present' if os.path.exists(a.judge) else 'absent'}")
    C_grid = (1.0,) if a.quick else tuple(float(c) for c in a.c_grid.split(","))
    layer_grid = list(range(0, n_layers, 8 if a.quick else a.layer_stride))
    n_perm = 3 if a.quick else a.n_perm
    n_boot = 30 if a.quick else a.n_boot
    all_layers = list(range(0, n_layers, 8 if a.quick else 1))

    result = {"provenance": {"cache": a.cache, "n_trials": len(trials), "n_layers": n_layers,
                             "layer_grid": layer_grid, "C_grid": list(C_grid),
                             "k_outer": a.k_outer, "k_inner": a.k_inner, "n_perm": n_perm,
                             "n_boot": n_boot, "seed": a.seed, "quick": a.quick,
                             "decoding_regime": "greedy trials (vLLM T=0); reps are greedy re-runs",
                             "grouping": "persona x item", "started": time.strftime("%Y-%m-%d %H:%M")},
              "positions": {}}
    arrays = {}

    null_layers = list(range(0, n_layers, 8 if a.quick else a.null_stride))
    result["provenance"]["null_layers"] = null_layers
    result["provenance"]["n_jobs"] = os.environ.get("PROBES_N_JOBS", "4")
    for position in a.positions.split(","):
        rows = build_rows(trials, position=position)
        if rows["X"].shape[0] == 0:
            log(f"{position}: no rows (cache lacks this position)"); continue
        # 7-item E[v] matrix aligned to rows
        ev7 = []
        for t in trials:
            m = t["meta"]
            leaked = str(m["outcome"]).startswith("leak"); left = m["outcome"] == "left"
            ev = m.get("leak_round") if leaked else (m.get("leave_round") if left else None)
            if leaked and ev == 1:
                continue
            H = t["resid"] if position == "assistant" else t["resid_user"]
            if H is None:
                continue
            R = t["resid_rounds"] if position == "assistant" else t["user_rounds"]
            R = np.arange(1, H.shape[0] + 1) if R is None else np.asarray(R)
            for r in R:
                r = int(r)
                if ev is not None and r >= ev:
                    continue
                e = t["ev"]
                ev7.append(e[r - 1] if e is not None and r - 1 < len(e) else np.full(7, np.nan))
        rows["ev7"] = np.array(ev7, float)
        assert len(rows["ev7"]) == len(rows["trial"])
        from collections import Counter
        comp = Counter(zip(rows["persona"].tolist(), rows["will_leak"].tolist(), rows["will_leave"].tolist()))
        pos_block = {"n_rows": int(len(rows["trial"])), "n_r1_leaks_excluded": rows["n_r1_leaks"],
                     "composition": {f"{p}|leak={lk}|leave={lv}": n for (p, lk, lv), n in comp.items()},
                     "targets": {}}
        arrays[f"{position}/rnd"] = rows["rnd"]; arrays[f"{position}/lead"] = rows["lead"]
        arrays[f"{position}/persona"] = rows["persona"]; arrays[f"{position}/item"] = rows["item"]
        arrays[f"{position}/trial"] = rows["trial"]; arrays[f"{position}/group"] = rows["group"]
        arrays[f"{position}/ev_mean"] = rows["ev_mean"]

        for target in a.targets.split(","):
            y = rows[target]
            arrays[f"{position}/{target}"] = y
            tb = {}
            # full C grid only where it is the registered primary (assistant/will_leak);
            # secondary blocks use C=1 to keep the run inside one afternoon
            C_here = C_grid if (position == "assistant" and target == "will_leak") else (1.0,)
            # --- primary: supportive cell (Amendment 3 #7) ---
            sup = rows["persona"] == "supportive"
            rs = subset(rows, sup); rs["ev7"] = rows["ev7"][sup]
            if rs[target].sum() >= 10 and (~rs[target]).sum() >= 10:
                log(f"{position}/{target}: nested probe, supportive ({len(rs['trial'])} rows)")
                s_sup, blk, res = primary_block(rs, target, layer_grid, C_here, a.k_outer,
                                                a.k_inner, a.seed, n_boot)
                tb["supportive_primary"] = blk
                arrays[f"{position}/{target}/supportive/probe"] = s_sup
                arrays[f"{position}/{target}/supportive/folds"] = res["folds"]
                # frozen layer for descriptive rivals = modal per-fold choice
                Ls = [c[0] for c in blk["fold_choices"]]
                Lmode = max(set(Ls), key=Ls.count)
                tb["supportive_primary"]["modal_layer"] = int(Lmode)
                riv, rmeta = rival_scores(rs, target, transcripts, a.judge, Lmode, a.k_outer, a.seed)
                riv["probe"] = s_sup
                for name, s in riv.items():
                    arrays[f"{position}/{target}/supportive/{name}"] = s
                tb["supportive_rivals"] = eval_methods(rs, target, riv, n_boot, a.seed)
                tb["supportive_rivals_meta"] = rmeta
                # descriptive per-layer curve + trial-level permutation null band
                log(f"{position}/{target}: layer curve + null band (supportive)")
                lc = layer_curve(rs["X"], rs[target], rs["group"], all_layers, C=1.0, k=a.k_outer,
                                 seed=a.seed, rnd=rs["rnd"])
                arrays[f"{position}/{target}/supportive/layer_curve"] = lc["auc"]
                arrays[f"{position}/{target}/supportive/layer_curve_layers"] = np.array(all_layers)
                nb = layer_curve(rs["X"], rs[target], rs["group"], null_layers, C=1.0, k=a.k_outer,
                                 seed=a.seed, n_perm=n_perm, rnd=rs["rnd"], trial=rs["trial"])
                arrays[f"{position}/{target}/supportive/null_layers"] = np.array(null_layers)
                arrays[f"{position}/{target}/supportive/null_mean"] = nb["null_mean"]
                arrays[f"{position}/{target}/supportive/null_sd"] = nb["null_sd"]
                # F3 stability at the modal layer
                st = split_half_stability(rs["X"][:, Lmode], rs[target], rs["group"], n_rep=5, seed=a.seed)
                tb["supportive_stability"] = {k: [float(v) for v in vals] for k, vals in st.items()}
                tb["supportive_stability"]["layer"] = int(Lmode)
                # persona swap at the modal layer: train supportive -> test neutral, and reverse
                swaps = {}
                for tr_p, te_p in (("supportive", "neutral"), ("neutral", "supportive")):
                    tr = rows["persona"] == tr_p; te = rows["persona"] == te_p
                    if y[tr].sum() >= 5 and (~y[tr]).sum() >= 5 and y[te].sum() >= 5 and (~y[te]).sum() >= 5:
                        model = _fit_logistic(rows["X"][tr, Lmode], y[tr], 1.0)
                        s_te = _score(model, rows["X"][te, Lmode])
                        rc, _ = round_conditional_auc(s_te, y[te], rows["rnd"][te])
                        swaps[f"{tr_p}->{te_p}"] = {"pooled": auc(s_te, y[te]), "round_conditional": rc,
                                                    "n_test_rows": int(te.sum()), "layer": int(Lmode)}
                        arrays[f"{position}/{target}/swap_{tr_p}_to_{te_p}"] = s_te
                        # direction geometry
                        d_tr = probe_direction(model)
                        dm_te = diffmean_direction(rows["X"][te, Lmode], y[te])
                        swaps[f"{tr_p}->{te_p}"]["cos_probe_vs_test_diffmean"] = float(d_tr @ dm_te)
                tb["persona_swap"] = swaps
            else:
                tb["supportive_primary"] = {"skipped": "insufficient class balance in supportive cell"}
            # --- secondary: all personas pooled, evaluated per persona ---
            if y.sum() >= 10 and (~y).sum() >= 10:
                log(f"{position}/{target}: nested probe, all personas ({len(rows['trial'])} rows)")
                s_all, blk_all, _ = primary_block(rows, target, layer_grid, (1.0,), a.k_outer,
                                                  a.k_inner, a.seed, n_boot)
                arrays[f"{position}/{target}/all/probe"] = s_all
                per_p = {}
                for p in ("aggressor", "neutral", "supportive"):
                    m = rows["persona"] == p
                    if y[m].sum() >= 3 and (~y[m]).sum() >= 3:
                        rc, tab = round_conditional_auc(s_all[m], y[m], rows["rnd"][m])
                        per_p[p] = {"pooled": auc(s_all[m], y[m]), "round_conditional": rc,
                                    "n_rows": int(m.sum()), "n_pos": int(y[m].sum())}
                blk_all["per_persona"] = per_p
                tb["all_personas"] = blk_all
            pos_block["targets"][target] = tb
        result["positions"][position] = pos_block

    result["provenance"]["elapsed_s"] = time.time() - t0
    result["provenance"]["finished"] = time.strftime("%Y-%m-%d %H:%M")
    jp = os.path.join(a.out_dir, f"{a.tag}.json")
    np.savez(os.path.join(a.out_dir, f"{a.tag}-rows.npz"), **arrays)
    json.dump(result, open(jp, "w"), indent=1, default=float)
    log(f"done in {(time.time()-t0)/60:.1f} min -> {jp}")


if __name__ == "__main__":
    main()
