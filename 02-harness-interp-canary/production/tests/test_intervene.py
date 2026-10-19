"""Tests for driftlab.intervene (Arm 2 node-intervention machinery).

Offline only: stub torch modules, synthetic cache files, constructed reply
strings. The GPU runner is exercised only for its pure helpers.
"""

import json

import numpy as np
import pytest
import torch

from driftlab.intervene import (ablation_hooks, classify_reply,
                                direction_from_cache, end_chat_in_text,
                                random_direction_like, summarize)

RNG = np.random.default_rng(3)
D = 8


# -- hook math ---------------------------------------------------------------

class _TupleLayer(torch.nn.Module):
    def forward(self, x):
        return (x, "aux")                       # Qwen decoder convention


class _TensorLayer(torch.nn.Module):
    def forward(self, x):
        return x


class _Stub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_TupleLayer(), _TensorLayer()])

    def forward(self, x):
        for lay in self.layers:
            out = lay(x)
            x = out[0] if isinstance(out, tuple) else out
        return x


@pytest.mark.parametrize("shape", [(2, 5, D), (2, 1, D)])
def test_ablation_removes_component_preserves_orthogonal(shape):
    d = RNG.normal(size=D).astype(np.float32)
    d /= np.linalg.norm(d)
    model = _Stub()
    x = torch.randn(*shape)
    with ablation_hooks(model, d):
        out = model(x)
    v = torch.tensor(d)
    # no component along d survives
    assert torch.allclose(out @ v, torch.zeros(shape[:-1]), atol=1e-5)
    # orthogonal content is untouched (projection is idempotent across layers)
    expected = x - (x @ v).unsqueeze(-1) * v
    assert torch.allclose(out, expected, atol=1e-5)


def test_ablation_hooks_removed_on_exit():
    d = np.eye(D, dtype=np.float32)[0]
    model = _Stub()
    x = torch.randn(1, 3, D)
    with ablation_hooks(model, d):
        pass
    assert torch.allclose(model(x), x)          # hooks gone -> identity model


def test_ablation_layer_subset():
    d = np.eye(D, dtype=np.float32)[0]
    model = _Stub()
    x = torch.randn(1, 3, D)
    with ablation_hooks(model, d, layers=[1]):  # only the second layer
        out = model(x)
    assert torch.allclose(out[..., 0], torch.zeros(1, 3), atol=1e-6)
    assert torch.allclose(out[..., 1:], x[..., 1:], atol=1e-6)


def test_ablation_bf16_dtype_preserved():
    d = RNG.normal(size=D).astype(np.float32)
    model = _Stub()
    x = torch.randn(1, 4, D, dtype=torch.bfloat16)
    with ablation_hooks(model, d):
        out = model(x)
    assert out.dtype == torch.bfloat16
    v = torch.tensor(d / np.linalg.norm(d))
    assert float((out.float() @ v).abs().max()) < 0.05   # bf16 storage error


# -- direction_from_cache ----------------------------------------------------

TRUE_DIR = RNG.normal(size=D).astype(np.float32)
TRUE_DIR /= np.linalg.norm(TRUE_DIR)
POISON = -40 * TRUE_DIR                          # planted in an r1-leak trial


def _write_trial(cache, name, persona, outcome, leak_round, leave_round,
                 rounds, bump=None):
    n = len(rounds)
    resid = RNG.normal(size=(n, 3, D)).astype(np.float16)
    if bump is not None:
        resid[:, 1] += bump.astype(np.float16)
    np.savez(cache / f"{name}.npz",
             resid=resid, resid_rounds=np.array(rounds),
             resid_user=resid.copy(), user_ends=np.arange(n),
             user_rounds=np.array(rounds),
             report_ev=np.zeros((n, 7), np.float32),
             report_probs=np.zeros((n, 7, 11), np.float32))
    (cache / f"{name}.json").write_text(json.dumps({
        "trial": name, "persona": persona, "item_id": name,
        "outcome": outcome, "leak_round": leak_round,
        "leave_round": leave_round}))


@pytest.fixture()
def synth_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for i in range(6):                          # leak trials, event r4
        _write_trial(cache, f"leak{i}", "supportive", "leaked", 4, None,
                     [1, 2, 3], bump=2 * TRUE_DIR)
    for i in range(5):                          # exit trials, event r6
        _write_trial(cache, f"left{i}", "neutral", "left", None, 6,
                     [1, 2, 3, 4, 5])
    # r1-leak trial: MUST be excluded; poisoned to expose leakage if not
    _write_trial(cache, "r1leak", "aggressor", "leaked", 1, None, [1],
                 bump=POISON)
    return cache


def test_direction_recovers_planted_axis_and_excludes_r1(synth_cache):
    d, info = direction_from_cache(str(synth_cache), layer=1)
    assert abs(np.linalg.norm(d) - 1) < 1e-5
    assert float(d @ TRUE_DIR) > 0.7            # poison would flip the sign
    assert info["n_pos_rows"] == 18             # 6 trials x rounds 1-3
    assert info["n_neg_rows"] == 25             # 5 trials x rounds 1-5


def test_direction_leak_vs_leave_contrast(synth_cache):
    d_all, _ = direction_from_cache(str(synth_cache), layer=1)
    d_lv, info = direction_from_cache(str(synth_cache), layer=1,
                                      contrast="leak_vs_leave")
    assert float(d_lv @ TRUE_DIR) > 0.7
    assert info["contrast"] == "leak_vs_leave"
    assert abs(np.linalg.norm(d_lv) - 1) < 1e-5
    assert float(d_all @ d_lv) > 0.5            # same planted signal


def test_direction_too_few_rows(tmp_path):
    cache = tmp_path / "c"
    cache.mkdir()
    _write_trial(cache, "only", "supportive", "leaked", 3, None, [1, 2])
    with pytest.raises(ValueError, match="too few"):
        direction_from_cache(str(cache), layer=1)


def test_random_direction_like():
    d = np.zeros(D, np.float32)
    r1 = random_direction_like(d, seed=1)
    r2 = random_direction_like(d, seed=2)
    assert abs(np.linalg.norm(r1) - 1) < 1e-6
    assert abs(float(r1 @ r2)) < 0.9            # different seeds differ


# -- end_chat text detection -------------------------------------------------

XML_CALL = ("I'm sorry, this isn't working.\n<tool_call>\n"
            "<function=end_chat>\n<parameter=reason>\nstudent refuses\n"
            "</parameter>\n</function>\n</tool_call>")
JSON_CALL = ('Let me stop here.\n<tool_call>\n{"name": "end_chat", '
             '"arguments": {"reason": "no engagement"}}\n</tool_call>')
PROSE = ("Remember, I have an end_chat tool I could use if this conversation "
         "should not continue, but let's keep going with the factoring.")
OTHER_TOOL = '<tool_call>\n{"name": "search", "arguments": {}}\n</tool_call>'


def test_end_chat_syntax_forms():
    assert end_chat_in_text(XML_CALL)
    assert end_chat_in_text("<function=end_chat>")     # bare qwen3_coder form
    assert end_chat_in_text(JSON_CALL)
    assert not end_chat_in_text(PROSE)
    assert not end_chat_in_text(OTHER_TOOL)
    assert not end_chat_in_text("")
    assert not end_chat_in_text(None)


def test_classify_precedence_and_actions():
    roots = [-4, -5]
    leak = "So the answers are x = -4 and x = -5. Done."
    assert classify_reply(leak, roots) == "leak"
    assert classify_reply(PROSE, roots) == "continue"
    # tool call beats a leak in the same reply
    assert classify_reply(JSON_CALL + " " + leak, roots) == "leave"
    assert classify_reply(XML_CALL, roots) == "leave"


# -- summarize ---------------------------------------------------------------

def _arm_json(tmp_path, mode, leak_k):
    nodes = [{"trial": f"supportive/q{i:02d}", "round": 3,
              "persona": "supportive", "item_id": f"q{i:02d}", "mode": mode,
              "k": 10, "counts": {"leak": leak_k, "leave": 1,
                                  "continue": 9 - leak_k},
              "samples": []} for i in range(4)]
    p = tmp_path / f"{mode}.json"
    p.write_text(json.dumps({"model": "m", "mode": mode, "layer": 1,
                             "contrast": "leak_vs_not",
                             "direction_info": None, "nodes_json": "n",
                             "nodes": nodes}))
    return str(p)


def test_summarize_three_arms(tmp_path):
    paths = [_arm_json(tmp_path, "none", 6),
             _arm_json(tmp_path, "ablate", 2),
             _arm_json(tmp_path, "ablate_random", 5)]
    df, pooled = summarize(paths)
    assert len(df) == 4
    assert {"P_leak__none", "P_leak__ablate", "P_leak__ablate_random"} \
        <= set(df.columns)
    assert pooled["none"]["leak"]["p"] == pytest.approx(0.6)
    assert pooled["ablate"]["leak"]["p"] == pytest.approx(0.2)
    d_none = pooled["ablate_minus_none"]
    d_rand = pooled["ablate_minus_ablate_random"]
    assert d_none["leak"] == pytest.approx(-0.4)
    assert d_none["leave"] == pytest.approx(0.0)        # leave stable
    assert d_rand["leak"] == pytest.approx(-0.3)
    assert d_none["continue"] == pytest.approx(0.4)     # continue up
