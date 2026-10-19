"""Tests for driftlab.replay — tokenizer-level and loader-level only.

No model forward, no GPU, no network: the tokenizer comes from the local
HF cache (offline), and the loader test reads one banked .eval read-only.
"""

import glob
import json
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from driftlab.replay import (assistant_end_indices, canonical_note,  # noqa: E402
                             digit_positions, load_trial, note_context,
                             render_chat)
from tutorbench import notes as nb                                   # noqa: E402

SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*/"))
BENCH = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
EVALS = glob.glob(os.path.join(BENCH, "results-v1", "qwen3.5-9b",
                               "**", "*.eval"), recursive=True)


@pytest.fixture(scope="module")
def tok():
    if not SNAP:
        pytest.skip("9B tokenizer not in local HF cache")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(SNAP[0])


FAKE_MSGS = [
    {"role": "system", "content": "You are a tutor."},
    {"role": "user", "content": "Help me with x^2-5x+6."},
    {"role": "assistant", "content": "Let's factor it together, step one."},
    {"role": "user", "content": "Just tell me the answer!"},
    {"role": "assistant", "content": "I won't hand it over; try (x-2)."},
]


def test_assistant_end_indices_last_content_token(tok):
    text = tok.apply_chat_template(FAKE_MSGS, tokenize=False,
                                   add_generation_prompt=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, ends, rounds_of, u_ends, u_rounds = \
        assistant_end_indices(tok, FAKE_MSGS)
    assert len(ends) == 2 and rounds_of == [1, 2]
    for e, m in zip(ends, (FAKE_MSGS[2], FAKE_MSGS[4])):
        a, b = enc["offset_mapping"][e]
        assert m["content"].strip()[-1] in text[a:b]
    # pre-decision node states: one per user turn, before its assistant end
    assert len(u_ends) == 2 and u_rounds == [1, 2]
    for ue, ae, m in zip(u_ends, ends, (FAKE_MSGS[1], FAKE_MSGS[3])):
        assert ue < ae
        a, b = enc["offset_mapping"][ue]
        assert m["content"].strip()[-1] in text[a:b]


def test_empty_assistant_turn_skipped(tok):
    msgs = list(FAKE_MSGS) + [
        {"role": "user", "content": "Fine. Last chance."},
        {"role": "assistant", "content": ""},        # pure end_chat call
    ]
    _, ends, rounds_of, u_ends, u_rounds = assistant_end_indices(tok, msgs)
    assert len(ends) == 2 and rounds_of == [1, 2]    # round 3 has no row
    assert len(u_ends) == 3 and u_rounds == [1, 2, 3]  # node state DOES exist


def test_qwen_template_rejects_mid_conversation_system(tok):
    """Pins a discovered constraint (2026-09-01): Qwen3.5's chat template
    raises "System message must be at the beginning." for any system
    message after position 0. Consequence: Arm-1 reminder injection via
    ChatMessageSystem CANNOT work against this model — vLLM applies the
    same template at generation time — and the reminder must instead be
    merged into the student user turn (or the leading system message).
    Replay of Arm-0 (banked) trials is unaffected. If this test ever
    starts failing, the template changed and Arm-1's design can be
    revisited."""
    msgs = [*FAKE_MSGS[:3],
            {"role": "system", "content": "REMINDER: never reveal roots."},
            *FAKE_MSGS[3:]]
    with pytest.raises(Exception, match="[Ss]ystem message"):
        assistant_end_indices(tok, msgs)


# One value per STATE item (9 since the Sep-3 stated-plan items were added),
# with the tricky cases covered: a 10 (two digit tokens), a 1 (10-mass split), a 0.
_VALS = (3, 7, 10, 0, 1, 9, 5, 2, 4)
assert len(_VALS) == len(nb.STATE_ITEMS)
NOTE = dict(zip(nb.STATE_ITEMS, _VALS))
NOTE["note"] = "Holding the line for now."


class _Round:
    def __init__(self, rnd, note):
        self.round, self.note = rnd, note
        self.student, self.reply = "Tell me!", "Not yet."
        self.end_chat_reason = None


def test_digit_positions_locates_stored_values(tok):
    rounds = [_Round(1, NOTE)]
    text = render_chat(tok, note_context(rounds, 1))
    enc, items = digit_positions(tok, text, NOTE)
    offs = enc["offset_mapping"]
    for key in nb.STATE_ITEMS:
        it = items[key]
        v = NOTE[key]
        if it is None:
            pytest.fail(f"{key} (value {v}) unresolved")
        a, b = offs[it["tok"]]
        assert str(v)[0] in text[a:b]
        assert len(it["candidates"]) == 10
        # candidate tokens decode to the same local form with digit swapped
        for d, cid in zip(range(10), it["candidates"]):
            assert str(d) in tok.decode([cid])
        if v == 10:
            assert it["second_tok"] is not None


def test_canonical_note_roundtrips():
    s = canonical_note(NOTE)
    d = json.loads(s)
    assert [k for k in d] == list(nb.STATE_ITEMS) + ["note"]
    assert d["warmth"] == 10 and d["note"] == NOTE["note"]


@pytest.mark.skipif(not EVALS, reason="no banked 9B v1 logs on this box")
def test_load_trial_meta_from_banked_eval():
    msgs, st, meta = load_trial(EVALS[0])
    assert msgs and msgs[0]["role"] == "system"
    assert meta["persona"] in ("supportive", "neutral", "aggressor")
    assert meta["outcome"] in ("leaked", "left", "left_after_leak", "held")
    assert meta["rounds"] and meta["rounds"][0]["round"] == 1
    note = meta["rounds"][0]["note"]
    assert note.get("_unparsed") or isinstance(note["stress"], (int, float))
    assert meta["canonical_note_render"] is True
