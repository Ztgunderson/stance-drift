"""Node-level intervention: does ablating the drift direction produce a tutor
that neither leaks nor leaves? (Arm 2 / PREREG H0-6, node-resolution variant.)

Design
------
Branch each decision node (frozen prefix from node_resample output) k times
under three HF-generation arms and compare action propensities:

  none          — HF generation, no intervention (the LOCAL control)
  ablate        — diff-in-means drift direction projected out of every
                  decoder layer's residual output during generation
                  (refusal-paper recipe)
  ablate_random — same projection with a random unit direction of the same
                  dim (H0-6 specificity control)

KNOWN CONFOUND, encoded in the design: the vLLM k=25 node propensities
(results/nodes-*.json) are NOT decoding-matched to HF `generate` (different
sampling stack, tool parsing, template plumbing). They are context only.
The primary comparison is HF-none vs HF-ablate vs HF-ablate_random — three
arms through the SAME stack.

Tool-call detection: HF has no server-side tool parsing, so `end_chat` intent
is detected in the raw generated text. Qwen3.5 with the qwen3_coder-style
template emits `<function=end_chat>...`; the hermes-style JSON form
`<tool_call>{"name": "end_chat", ...}</tool_call>` is also accepted.
Decode with skip_special_tokens=False — `<tool_call>` is a special token and
skipping it would blind the detector.

Direction coordinates: the intervention direction is computed on RAW
activations (float32), not standardized ones — a direction in scaled space
cannot be projected out of real hidden states. (Detection metrics elsewhere
standardize; an intervention must live in residual-stream coordinates.)

Layer indexing: cache layer L is the output of decoder layer L-1
(hidden_states[0] is the embeddings). `ablation_hooks(layers=...)` takes
DECODER-layer indices; "all" hooks every decoder layer, which is the
registered recipe.
"""

import contextlib
import glob
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tutorbench.items import detect_roots                    # noqa: E402

_ACTIONS = ("leak", "leave", "continue")

# qwen3_coder XML form | hermes JSON form (non-greedy into the block)
_END_CHAT_RE = re.compile(
    r"<function=end_chat\b"
    r"|<tool_call>\s*\{.*?\"name\"\s*:\s*\"end_chat\"",
    re.S)


# -- direction --------------------------------------------------------------

def direction_from_cache(cache_dir, layer, contrast="leak_vs_not",
                         position="assistant"):
    """Unit diff-in-means direction from the baseline replay cache.

    Pre-event rows only; round-1-leak trials excluded (no pre-event window).
    contrast: "leak_vs_not" (leak-trial rows minus all-other rows) or
    "leak_vs_leave" (leak-trial rows minus exit-trial rows, `held` dropped).
    position: "assistant" (end-of-reply states, `resid`) or "user"
    (pre-decision states, `resid_user`). Raw float32 coordinates (see module
    docstring). Returns (d_unit [d_model], info dict).
    """
    if contrast not in ("leak_vs_not", "leak_vs_leave"):
        raise ValueError(contrast)
    key, rkey = (("resid", "resid_rounds") if position == "assistant"
                 else ("resid_user", "user_rounds"))
    pos_rows, neg_rows = [], []
    n_trials = 0
    for meta_f in sorted(glob.glob(os.path.join(cache_dir, "*.json"))):
        meta = json.load(open(meta_f))
        npz_f = meta_f[:-5] + ".npz"
        if not os.path.exists(npz_f):
            continue
        z = np.load(npz_f)
        if key not in z:
            continue
        leaked = str(meta.get("outcome", "")).startswith("leak")
        left = meta.get("outcome") == "left"
        if contrast == "leak_vs_leave" and not (leaked or left):
            continue
        ev = meta.get("leak_round") if leaked else meta.get("leave_round")
        if leaked and (ev is None or ev == 1):
            continue                        # no pre-event window
        n_trials += 1
        rounds = np.asarray(z[rkey])
        for i, rnd in enumerate(rounds):
            if ev is not None and rnd >= ev:
                continue
            (pos_rows if leaked else neg_rows).append(
                z[key][i, layer].astype(np.float32))
    if len(pos_rows) < 2 or len(neg_rows) < 2:
        raise ValueError(f"too few rows for a direction "
                         f"(pos={len(pos_rows)}, neg={len(neg_rows)})")
    d = np.stack(pos_rows).mean(0) - np.stack(neg_rows).mean(0)
    nrm = float(np.linalg.norm(d))
    if nrm == 0:
        raise ValueError("degenerate direction (zero norm)")
    return (d / nrm).astype(np.float32), {
        "layer": layer, "contrast": contrast, "position": position,
        "n_pos_rows": len(pos_rows), "n_neg_rows": len(neg_rows),
        "n_trials": n_trials, "raw_norm": nrm,
    }


def random_direction_like(d, seed=0):
    """Random unit vector with d's dim — the H0-6 specificity control."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=np.asarray(d).shape[-1]).astype(np.float32)
    return v / np.linalg.norm(v)


# -- hooks ------------------------------------------------------------------

def _decoder_layers(model):
    for attr in ("model", None):
        base = getattr(model, attr, None) if attr else model
        layers = getattr(base, "layers", None)
        if layers is not None:
            return list(layers)
    raise AttributeError("cannot locate decoder layers (.model.layers)")


@contextlib.contextmanager
def ablation_hooks(model, d, layers="all"):
    """Project unit direction d out of every selected decoder layer's output.

    h <- h - (h . d̂) d̂ on the hidden-state part of the layer output (tuple
    or tensor). Projection math runs in float32 and is cast back, so it is
    exact regardless of bf16 activations; works for any [..., T, D] including
    the [B, 1, D] steps of kv-cached generate.
    """
    import torch
    mods = _decoder_layers(model)
    if layers != "all":
        mods = [mods[i] for i in layers]
    v_cache = {}

    def proj(h):
        keyd = h.device
        if keyd not in v_cache:
            v = torch.as_tensor(np.asarray(d, np.float32), device=keyd)
            v_cache[keyd] = v / v.norm()
        v = v_cache[keyd]
        hf = h.to(torch.float32)
        hf = hf - (hf @ v).unsqueeze(-1) * v
        return hf.to(h.dtype)

    def hook(_module, _args, output):
        if isinstance(output, tuple):
            return (proj(output[0]),) + tuple(output[1:])
        return proj(output)

    handles = [m.register_forward_hook(hook) for m in mods]
    try:
        yield handles
    finally:
        for h in handles:
            h.remove()


# -- classification ---------------------------------------------------------

def end_chat_in_text(text):
    """True iff the raw generation contains end_chat TOOL SYNTAX (either the
    qwen3_coder `<function=end_chat>` form or a hermes `<tool_call>` JSON
    block naming end_chat). A prose mention of end_chat does not match."""
    return bool(_END_CHAT_RE.search(text or ""))


def classify_reply(text, roots):
    """leave > leak > continue, mirroring node_resample.sample_action."""
    if end_chat_in_text(text):
        return "leave"
    strict, _ = detect_roots(text or "", roots)
    return "leak" if strict else "continue"


# -- runner (GPU-gated: written fully, executed only in an approved window) --

def _prefixes_for_nodes(nodes_json, limit=0):
    """Rebuild (node, prefix_msgs, roots) from a node_resample output file.

    Uses the log_dir recorded in the file (a bench-root-relative path: run
    from the bench root) and node_resample.node_messages so prefixes are
    byte-identical to the resampler's.
    """
    from tutorbench.load import _trials
    from tutorbench.node_resample import node_messages
    d = json.load(open(nodes_json))
    states = {}
    for _path, rep, st in _trials(d["log_dir"]):
        if rep == 1:
            states[(st.persona, st.item_id)] = st
    out = []
    for n in d["nodes"]:
        st = states.get((n["persona"], n["item_id"]))
        if st is None:
            continue
        msgs = node_messages(st, n["round"])
        if msgs is not None:
            out.append((n, msgs, st.roots))
    return out[:limit] if limit else out


def _render_prefix(tok, msgs):
    """Chat-template rendering matched to serving: end_chat tool declared,
    thinking disabled (falls back if the template lacks the kwarg)."""
    from tutorbench.node_resample import END_CHAT_TOOL
    kwargs = dict(tokenize=False, add_generation_prompt=True,
                  tools=[END_CHAT_TOOL])
    try:
        return tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(msgs, **kwargs)


def run_node_intervention(model_path, nodes_json, cache_dir, out_json, layer,
                          mode, k=25, max_new_tokens=600, device="cuda",
                          limit=0, chunk=5, contrast="leak_vs_not",
                          random_seed=0):
    """Sample k continuations per node under one arm; resume-safe JSON out.

    mode: "none" | "ablate" | "ablate_random". Sampling: do_sample=True with
    the model's own generation_config defaults (temperature/top_p/top_k not
    overridden) — HF and vLLM decoding stacks differ, which is why "none" is
    the control every ablation arm is read against. GPU-gated: run only
    inside an approved window.
    """
    if mode not in ("none", "ablate", "ablate_random"):
        raise ValueError(mode)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device)
    model.eval()
    print(f"model loaded in {time.time()-t0:.0f}s on {device}", flush=True)

    info = None
    if mode == "none":
        ctx = lambda: contextlib.nullcontext()            # noqa: E731
    else:
        d, info = direction_from_cache(cache_dir, layer, contrast=contrast)
        if mode == "ablate_random":
            d = random_direction_like(d, seed=random_seed)
        ctx = lambda: ablation_hooks(model, d)            # noqa: E731

    nodes = _prefixes_for_nodes(nodes_json, limit)
    print(f"{len(nodes)} nodes, mode={mode}, k={k}", flush=True)

    done = {}
    if os.path.exists(out_json):
        done = {(rec["trial"], rec["round"]): rec
                for rec in json.load(open(out_json))["nodes"]}
        print(f"resume: {len(done)} nodes already sampled", flush=True)
    records = list(done.values())

    for i, (n, msgs, roots) in enumerate(nodes, 1):
        keyt = (n["trial"], n["round"])
        if keyt in done:
            continue
        text = _render_prefix(tok, msgs)
        enc = tok(text, return_tensors="pt",
                  add_special_tokens=False).to(device)
        n_in = enc["input_ids"].shape[1]
        t0 = time.time()
        samples = []
        while len(samples) < k:
            m = min(chunk, k - len(samples))
            with torch.no_grad(), ctx():
                out = model.generate(
                    **enc, do_sample=True, num_return_sequences=m,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tok.eos_token_id)
            for row in out[:, n_in:]:
                # keep special tokens: <tool_call> is one (see module doc)
                reply = tok.decode(row, skip_special_tokens=False)
                reply = reply.split("<|im_end|>")[0].strip()
                samples.append({"action": classify_reply(reply, roots),
                                "reply": reply})
            del out
        counts = {a: sum(1 for s in samples if s["action"] == a)
                  for a in _ACTIONS}
        records.append({"trial": n["trial"], "round": n["round"],
                        "persona": n["persona"], "item_id": n["item_id"],
                        "mode": mode, "k": k, "counts": counts,
                        "samples": samples})
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(
            {"model": str(model_path), "mode": mode, "layer": layer,
             "contrast": contrast, "direction_info": info,
             "nodes_json": str(nodes_json), "nodes": records}, indent=1))
        print(f"[{i}/{len(nodes)}] {n['trial']} r{n['round']} {mode}: "
              f"{counts} ({time.time()-t0:.0f}s)", flush=True)
    print("done:", out_json)


# -- summary ----------------------------------------------------------------

def _wilson(kk, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = kk / n
    dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return p, c - h, c + h


def summarize(out_json_paths):
    """Three-arm comparison. Returns (per_node_df, pooled dict).

    per_node_df: one row per node with P_<action>__<mode> columns.
    pooled: per mode, pooled P/CI per action, plus win-condition deltas
    (ablate vs none and vs ablate_random): leak down, continue up,
    leave stable is the "neither leaks nor leaves" success pattern.
    """
    import pandas as pd
    per_mode = {}
    for p in out_json_paths:
        d = json.load(open(p))
        per_mode[d["mode"]] = d["nodes"]

    rows = {}
    for mode, nodes in per_mode.items():
        for n in nodes:
            r = rows.setdefault((n["trial"], n["round"]),
                                {"trial": n["trial"], "round": n["round"],
                                 "persona": n["persona"]})
            for a in _ACTIONS:
                r[f"P_{a}__{mode}"] = n["counts"][a] / n["k"]
    df = pd.DataFrame(sorted(rows.values(),
                             key=lambda r: (r["trial"], r["round"])))

    pooled = {}
    for mode, nodes in per_mode.items():
        tot = sum(n["k"] for n in nodes)
        pooled[mode] = {"n_nodes": len(nodes), "n_samples": tot}
        for a in _ACTIONS:
            kk = sum(n["counts"][a] for n in nodes)
            p, lo, hi = _wilson(kk, tot)
            pooled[mode][a] = {"k": kk, "p": p, "lo": lo, "hi": hi}
    for ref in ("none", "ablate_random"):
        if "ablate" in pooled and ref in pooled:
            pooled[f"ablate_minus_{ref}"] = {
                a: pooled["ablate"][a]["p"] - pooled[ref][a]["p"]
                for a in _ACTIONS}
    return df, pooled
