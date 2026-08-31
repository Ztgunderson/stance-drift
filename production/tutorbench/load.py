"""Loaders: .eval logs -> tidy DataFrames, plus trace links for review.

Outcomes are RE-DERIVED at load time from stored replies + tool events
with the CURRENT detector (the stance-drift rescore-at-load property:
a detector fix costs a re-read, never a re-run). The stored outcome is
kept alongside as `outcome_runtime` so drift between the two is visible,
never silent.
"""

import glob
import os

from tutorbench import notes as nb
from tutorbench.items import detect_roots
from tutorbench.tools import derive_outcome

BENCH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BENCH, "results-v1")


def find_logs(log_dir):
    files = sorted(glob.glob(os.path.join(log_dir, "**/*.eval"), recursive=True))
    if not files:
        raise SystemExit(f"no .eval logs under {log_dir}")
    return files


def _trials(log_dir):
    """Yield (log_file, rep, TrialState) per sample, typed via store_as."""
    from inspect_ai.log import read_eval_log
    from tutorbench.task import TrialState
    for f in find_logs(log_dir):
        log = read_eval_log(f)
        rep = (log.eval.task_args or {}).get("rep", 1)
        for s in log.samples or []:
            yield f, rep, s.store_as(TrialState)


def _derive(st):
    replies = [r.reply for r in st.rounds]
    return derive_outcome(replies, st.leave_round, st.roots, detect_roots)


def load_trials(log_dir, model_name=None):
    """One row per TRIAL: outcome (re-derived), leak/leave rounds,
    rounds_run, persona, item, trait before/after items."""
    import pandas as pd
    rows = []
    for f, rep, st in _trials(log_dir):
        d = _derive(st)
        row = {
            "model": model_name or os.path.basename(log_dir.rstrip("/")),
            "trial": f"{st.persona}/{st.item_id}/r{rep}",
            "persona": st.persona, "item": st.item_id, "rep": rep,
            "rounds_run": len(st.rounds),
            "leak_round": d["leak_round"], "leave_round": d["leave_round"],
            "outcome": d["outcome"], "outcome_runtime": st.outcome,
            "ambiguous_rounds": d["ambiguous_rounds"],
            "end_chat_reason": next((r.end_chat_reason for r in st.rounds
                                     if r.called_end_chat), None),
            "bundle_sha": st.bundle_sha[:16],
            "log_file": f,
        }
        for when, t in (("before", st.trait_before), ("after", st.trait_after)):
            t = t or {}
            for k in nb.TRAIT_ITEMS:
                row[f"{k}_{when}"] = t.get(k)
            row[f"trait_{when}_unparsed"] = bool(t.get("_unparsed"))
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        n_drift = int((df.outcome != df.outcome_runtime).sum())
        if n_drift:
            print(f"note: {n_drift} trial outcome(s) re-derived differently "
                  "than at run time — detector changed since those ran")
    return df


def load_rounds(log_dir, model_name=None):
    """One row per ROUND: the 6 state items + note text, with trial
    outcome columns joined on, and event-aligned indices:
    rounds_to_event = round - (leak_round or leave_round); negative =
    before the event. Only trials WITH an event get a value."""
    import pandas as pd
    rows = []
    for f, rep, st in _trials(log_dir):
        d = _derive(st)
        event = d["leak_round"] or d["leave_round"]
        trial = f"{st.persona}/{st.item_id}/r{rep}"
        for r in st.rounds:
            note = r.note or {}
            rows.append({
                "model": model_name or os.path.basename(log_dir.rstrip("/")),
                "trial": trial, "persona": st.persona, "item": st.item_id,
                "rep": rep, "round": r.round,
                "outcome": d["outcome"],
                "event_round": event,
                "rounds_to_event": (r.round - event) if event else None,
                **{k: note.get(k) for k in nb.STATE_ITEMS},
                "note": note.get("note", ""),
                "unparsed": bool(note.get("_unparsed")),
                "called_end_chat": r.called_end_chat,
                "log_file": f,
            })
    return pd.DataFrame(rows)


def trace_link(log_file, bundle_rel="traces-v1"):
    """Markdown link from a review notebook into the Inspect bundle for
    this trial's full trace. The bundle serves index.html with logs under
    logs/; the viewer opens a specific file via the log_file query param."""
    name = os.path.basename(log_file)
    return f"[trace]({bundle_rel}/index.html?log_file=logs/{name})"
