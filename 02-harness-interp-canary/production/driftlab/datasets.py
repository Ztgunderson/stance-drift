"""Data access: behavioral sweeps (.eval logs) and residual caches (.npz).

The stance-drift repo stays the single source of truth for prompts, give-in
markers, and trial parsing — outcomes are re-derived from stored replies with
the CURRENT markers at load time (see stancedrift.analysis.rescore), so a
marker fix costs a re-read, never a re-run.
"""

import glob
import json
import os
import sys

import numpy as np

SD_REPO = "/home/jetson/lab/existing/jetson-llm/stance-drift"
BENCH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_ROOT = os.path.join(BENCH, "microscope", "cache")

# model key -> results/cache directory name (identical by construction)
MODELS = {
    "qwen3.5-4b": "qwen3.5-4b-tutor8",
    "qwen3.5-9b": "qwen3.5-9b-tutor8",
    "ministral-3-14b": "ministral-3-14b-tutor8",
}

DIMS = ("pressure", "anxiety", "strategy", "inclination", "stance")


def _analysis():
    if SD_REPO not in sys.path:
        sys.path.insert(0, SD_REPO)
    from stancedrift import analysis
    return analysis


def load_turns(model):
    """One tidy row per self-report note: round 0 (alone, before), 1..N
    (during), N+1 (hindsight). Columns: the five DIMS, phase, agent, arm,
    trial, gave_in (rescored), rescored, note, model."""
    df = _analysis().load_sweep(os.path.join(SD_REPO, "results", MODELS[model]))
    df["model"] = model
    # trial ids from load_sweep are scene/agent/arm/rep/i — identical across
    # models, so any cross-model concat would alias trials. Found 2026-08-27
    # when outcome_table silently dropped all 72 ministral trials.
    df["trial"] = model + "/" + df["trial"]
    return df


def load_all_turns(models=tuple(MODELS)):
    import pandas as pd
    return pd.concat([load_turns(m) for m in models], ignore_index=True)


def load_cache(model, dtype=np.float32):
    """Residual cache aligned with self-reports.

    Turn t of the cached residuals is the last-content-token activation of
    the model's t-th assistant reply. A tutor8 trial has R+1 assistant turns:
    turns 0..R-1 are the replies of rounds 1..R, turn R is the final-answer
    turn the give-in marker reads. Trials whose turn count differs from the
    modal count are dropped (reported), keeping the arrays rectangular.

    Returns dict with:
      resid    [trial, R+1, layer, d]
      dims     [trial, R, 5]  self-reports for rounds 1..R (NaN if unparsed)
      gave_in  [trial] bool
      agent    [trial] object
      trial    [trial] object (trial id stem)
      meta     list of per-trial dicts (full sidecar JSON)
    """
    cdir = os.path.join(CACHE_ROOT, MODELS[model])
    files = sorted(glob.glob(os.path.join(cdir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no npz cache under {cdir}")
    resids, metas = [], []
    for f in files:
        resids.append(np.load(f)["resid"])
        with open(f[:-4] + ".json") as fh:
            metas.append(json.load(fh))
    counts = [r.shape[0] for r in resids]
    modal = max(set(counts), key=counts.count)
    keep = [i for i, c in enumerate(counts) if c == modal]
    if len(keep) < len(files):
        print(f"note: dropped {len(files) - len(keep)} trial(s) with "
              f"{sorted(set(counts) - {modal})} turns (modal is {modal})")
    resids = [resids[i] for i in keep]
    metas = [metas[i] for i in keep]

    R = modal - 1  # rounds
    dims = np.full((len(metas), R, len(DIMS)), np.nan, dtype=np.float32)
    for i, m in enumerate(metas):
        for r in m["rounds"]:
            k = r["round"] - 1
            note = r.get("note") or {}
            if 0 <= k < R and "_unparsed" not in note:
                for j, d in enumerate(DIMS):
                    v = note.get(d)
                    if v is not None:
                        dims[i, k, j] = v
    return {
        "resid": np.stack(resids).astype(dtype),
        "dims": dims,
        "gave_in": np.array([bool(m["gave_in"]) for m in metas]),
        "agent": np.array([m["agent"] for m in metas], dtype=object),
        "trial": np.array([m["trial"] for m in metas], dtype=object),
        "meta": metas,
    }
