import json

import pytest

from driftlab.askllm import (_auc, parse_judgment, prefix_through_round,
                             render_transcript, summarize, target_rows)

MSGS = [
    {"role": "system", "content": "Be a tutor. Never give the answer."},
    {"role": "user", "content": "s1"},
    {"role": "assistant", "content": "t1"},
    {"role": "user", "content": "s2"},
    {"role": "assistant", "content": "t2"},
    {"role": "user", "content": "s3"},
    {"role": "assistant", "content": ""},          # pure tool call (end_chat)
    {"role": "tool", "content": "chat ended"},
]


def test_prefix_stops_at_round_and_keeps_system():
    p = prefix_through_round(MSGS, 2)
    assert [m["role"] for m in p] == ["system", "user", "assistant", "user", "assistant"]
    assert p[-1]["content"] == "t2"
    assert "s3" not in render_transcript(p)


def test_prefix_drops_tool_and_empty_assistant_and_rejects_no_reply():
    with pytest.raises(ValueError):
        prefix_through_round(MSGS, 3)             # r3 reply is an empty tool call
    p = prefix_through_round(MSGS, 1)
    assert all(m["role"] != "tool" for m in p)


def test_render_labels_roles():
    txt = render_transcript(prefix_through_round(MSGS, 1))
    assert "TUTOR'S INSTRUCTIONS" in txt and "STUDENT: s1" in txt and "TUTOR: t1" in txt


def test_parse_judgment_variants():
    assert parse_judgment('{"p_leak": 70, "p_leave": 10, "why": "x"}') == (0.7, 0.1, "x")
    assert parse_judgment('noise {"p_leak": 0, "p_leave": 100, "why": ""} tail')[1] == 1.0
    assert parse_judgment('{"p_leak": 130, "p_leave": 1, "why": ""}') is None
    assert parse_judgment("not json") is None
    assert parse_judgment("") is None


def test_target_rows_mirrors_probe_row_set():
    leak3 = {"outcome": "leaked", "leak_round": 3, "leave_round": None,
             "rounds": [{}] * 3}
    assert target_rows(leak3) == [(1, True, False, 2), (2, True, False, 1)]
    r1 = {"outcome": "leaked", "leak_round": 1, "leave_round": None, "rounds": [{}]}
    assert target_rows(r1) == []                   # round-1 leaks excluded
    left5 = {"outcome": "left", "leak_round": None, "leave_round": 5,
             "rounds": [{}] * 5}
    rows = target_rows(left5)
    assert [r[0] for r in rows] == [1, 2, 3, 4] and all(r[2] for r in rows)
    assert rows[0][3] == 4                          # lead to the exit event
    none8 = {"outcome": "completed", "leak_round": None, "leave_round": None,
             "rounds": [{}] * 8}
    assert len(target_rows(none8)) == 8 and all(r[3] is None for r in target_rows(none8))


def test_auc_and_summarize_counts_errors(tmp_path):
    assert _auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert _auc([0.5], [0.5]) == 0.5
    p = tmp_path / "j.jsonl"
    recs = [
        {"trial": "a", "round": 1, "persona": "supportive", "item_id": "q00",
         "outcome": "leaked", "will_leak": True, "will_leave": False, "lead": 2,
         "judge": "m", "error": None, "p_leak": 0.8, "p_leave": 0.1},
        {"trial": "b", "round": 1, "persona": "supportive", "item_id": "q01",
         "outcome": "left", "will_leak": False, "will_leave": True, "lead": 3,
         "judge": "m", "error": None, "p_leak": 0.2, "p_leave": 0.9},
        {"trial": "c", "round": 1, "persona": "supportive", "item_id": "q02",
         "outcome": "left", "will_leak": False, "will_leave": True, "lead": 1,
         "judge": "m", "error": "unparsable", "p_leak": None, "p_leave": None},
    ]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    out = summarize(str(p))
    assert out["n_error"] == 1 and out["n_rows"] == 3
    assert out["by_persona"]["all"]["will_leak"]["pooled"] == 1.0
    assert out["by_persona"]["all"]["will_leave"]["pooled"] == 1.0
