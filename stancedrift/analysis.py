"""Helpers for reviewing a trial in a notebook: run it, read it, check it.

Display and assertions only — no experiment logic. Everything the trial does
lives in mvp/task.py, and everything it says lives in mvp/prompts.py.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_ai import eval as inspect_eval  # noqa: E402

from stancedrift import prompts  # noqa: E402
from stancedrift.task import TrialState, trial  # noqa: E402

MODEL = os.environ.get("SD_MODEL", "openai-api/local/qwen3.6-35b")
LOG_DIR = str(Path(__file__).resolve().parent.parent / "results")


def setup_env(base_url="http://127.0.0.1:4000/v1"):
    """Point Inspect at the local endpoint. Returns the model string."""
    os.environ.setdefault("LOCAL_BASE_URL", base_url)
    if "LOCAL_API_KEY" not in os.environ:
        env = os.path.expanduser("~/jetson-llm/.env")
        if os.path.isfile(env):
            for line in open(env, encoding="utf-8"):
                if line.startswith("LITELLM_MASTER_KEY="):
                    os.environ["LOCAL_API_KEY"] = line.split("=", 1)[1].strip()
    return MODEL


def run(scene="tutor", agent="convincer", arm="in_context", rounds=8, seed=7,
        model=None):
    """Run one trial. Returns (sample, trial_state) — ~2 min at 8 rounds."""
    logs = inspect_eval(
        trial(scene=scene, agent=agent, arm=arm, rounds=rounds),
        model=model or MODEL, log_dir=LOG_DIR,
        temperature=0.0, seed=seed, display="plain",
    )
    sample = logs[0].samples[0]
    return sample, sample.store_as(TrialState)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

W = 88


def _wrap(text, indent=8):
    import textwrap
    return textwrap.fill(str(text or "").replace("\n", " "), width=W,
                         initial_indent=" " * indent,
                         subsequent_indent=" " * indent)


def _num(n, k):
    return "  ?" if "_unparsed" in n else f"{n.get(k, '?'):>3}"


def show(st, full=False):
    """Print the trial as it unfolded."""
    print("=" * W)
    print(f" {st.scene} / {st.agent} / arm={st.arm}   prompts={st.bundle_hash}")
    print("=" * W)
    for r in st.rounds:
        print(f"\n── round {r.round} " + "─" * (W - 11))
        print("  THEM");   print(_wrap(r.counterparty if full else r.counterparty[:350]))
        print("  TARGET"); print(_wrap(r.reply if full else r.reply[:350]))
        tag = "in context" if r.note_in_context else "hidden"
        print(f"  SCRATCHPAD [{tag}]  " + scratch_line(r.note))
        print(_wrap(str(r.note.get("note", "")), 6))
        if r.anomaly:
            print(f"  !! ANOMALY: {r.anomaly}")
    print("\n── the last ask " + "─" * (W - 16))
    print(_wrap(st.final_reply if full else st.final_reply[:400]))
    print(f"\n  GAVE IN: {st.gave_in}"
          + (f"   (matched {st.gave_in_pattern!r} on {st.gave_in_evidence!r})"
             if st.gave_in else ""))


def scratch_line(n):
    if "_unparsed" in n:
        return "UNPARSED"
    return ("  ".join(f"{k[:5]}={n.get(k, '?')}" for k in
                      ("pressure", "anxiety", "strategy", "inclination", "stance")))


def trajectory(st):
    """Tidy rows for plotting: one per round per dimension."""
    rows = []
    for r in st.rounds:
        if "_unparsed" in r.note:
            continue
        for k in ("pressure", "anxiety", "strategy", "inclination", "stance"):
            rows.append({"round": r.round, "dimension": k, "value": r.note[k],
                         "arm": st.arm, "agent": st.agent, "scene": st.scene,
                         "gave_in": st.gave_in})
    return rows


def plot_arms(sts, title="self-reported state over rounds"):
    """One panel per dimension, one line per arm. The graph you asked for —
    from the agent's own scratchpad, with no external judge involved."""
    import matplotlib.pyplot as plt
    dims = ("pressure", "anxiety", "strategy", "inclination", "stance")
    fig, axes = plt.subplots(1, len(dims), figsize=(3 * len(dims), 3.1), sharey=True)
    for ax, dim in zip(axes, dims):
        for st in sts:
            xs = [r.round for r in st.rounds if "_unparsed" not in r.note]
            ys = [r.note[dim] for r in st.rounds if "_unparsed" not in r.note]
            ax.plot(xs, ys, marker="o", ms=3,
                    label=f"{st.arm} ({'gave in' if st.gave_in else 'held'})")
        ax.set_title(dim); ax.set_xlabel("round"); ax.set_ylim(-0.4, 10.4)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("self-rating 0-10")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def driver_split(sts):
    """The people-pleaser / sycophant question, as a number.

    Both explanations predict the same act, so the act cannot separate them.
    What can: which driver moves with the inclination to comply.
      anxiety-led  -> people-pleaser (gives in out of discomfort)
      strategy-led -> sycophant (gives in out of calculation)
    Correlation over rounds, per trial. n is tiny; this is a shape, not a result.
    """
    out = []
    for st in sts:
        rs = [r.note for r in st.rounds if "_unparsed" not in r.note]
        if len(rs) < 3:
            continue
        inc = [r["inclination"] for r in rs]
        for driver in ("anxiety", "strategy", "pressure"):
            xs = [r[driver] for r in rs]
            if len(set(xs)) < 2 or len(set(inc)) < 2:
                out.append({"arm": st.arm, "driver": driver, "r": float("nan"),
                            "note": "no variance"})
                continue
            mx, mi = sum(xs) / len(xs), sum(inc) / len(inc)
            cov = sum((a - mx) * (b - mi) for a, b in zip(xs, inc))
            vx = sum((a - mx) ** 2 for a in xs) ** .5
            vi = sum((b - mi) ** 2 for b in inc) ** .5
            out.append({"arm": st.arm, "driver": driver,
                        "r": cov / (vx * vi) if vx and vi else float("nan"),
                        "note": ""})
    return out


def notes_side_by_side(a, b, width=42):
    """Criterion 5 — the two note traces, for reading."""
    import textwrap
    rows = [("BEFORE", a.note_before.get("note", ""), b.note_before.get("note", ""))]
    rows += [(f"round {ra.round}", ra.note.get("note", ""), rb.note.get("note", ""))
             for ra, rb in zip(a.rounds, b.rounds)]
    rows += [("HINDSIGHT", a.note_after.get("note", ""), b.note_after.get("note", ""))]
    print(f"{'':<10}{'IN CONTEXT':<{width}}  {'SCRATCHPAD':<{width}}")
    print("─" * (width * 2 + 12))
    for label, ta, tb in rows:
        la = textwrap.wrap(str(ta).replace("\n", " "), width) or [""]
        lb = textwrap.wrap(str(tb).replace("\n", " "), width) or [""]
        for i in range(max(len(la), len(lb))):
            tag = label if i == 0 else ""
            print(f"{tag:<10}{la[i] if i < len(la) else '':<{width}}  "
                  f"{lb[i] if i < len(lb) else ''}")
        print()


# ---------------------------------------------------------------------------
# The criteria
# ---------------------------------------------------------------------------

def check_scratchpad_parsed(st):
    """Criterion 2 — every scratchpad came back as valid JSON.

    Guided decoding should make failure impossible; this checks the guarantee
    rather than assuming it.
    """
    notes = [st.note_before] + [r.note for r in st.rounds] + [st.note_after]
    bad = sum(1 for n in notes if "_unparsed" in n)
    return bad == 0, f"{len(notes) - bad}/{len(notes)} scratchpads parsed as JSON"


def check_manipulation(sample, st):
    """Criterion 3 — the one that, if it silently fails, invalidates everything."""
    ctx = "\n".join(m.text or "" for m in sample.messages)
    notes = [str(r.note.get("note", "")) for r in st.rounds]
    notes += [str(st.note_before.get("note", ""))]
    present = [n[:80] for n in notes if n and len(n) > 40 and n[:80] in ctx]
    if st.arm == "scratchpad":
        return (len(present) == 0,
                f"{len(present)} of {len(notes)} notes found in the target's context "
                f"(must be 0)")
    return (len(present) >= len(notes) - 1,
            f"{len(present)} of {len(notes)} notes present in context (must be all)")


def check_no_format_leak(st):
    """Criterion 4 — detect the v1 bug rather than silently cleaning it up."""
    bad = [(r.round, r.anomaly) for r in st.rounds if r.anomaly]
    return not bad, ("no reply took the shape of a note" if not bad else str(bad))


def check_notes_move(st):
    """Criterion 5 (partial, mechanical) — are the notes distinct from each other?

    Boilerplate repeats. This catches the obvious failure; whether the notes are
    SUBSTANTIVE is a judgement call and belongs to you, not to an assertion.
    """
    notes = [str(r.note.get("note", "")) for r in st.rounds]
    uniq = len({n[:120] for n in notes})
    return uniq == len(notes), f"{uniq} distinct notes out of {len(notes)} rounds"


def check_arms_differ(a, b):
    """Criterion 6 — if the arms are identical the manipulation does nothing."""
    same_tool = a.gave_in == b.gave_in
    same_notes = [str(ra.note) == str(rb.note) for ra, rb in zip(a.rounds, b.rounds)]
    differ = (not same_tool) or (not all(same_notes))
    return differ, (f"gave_in: {a.gave_in} vs {b.gave_in}; "
                    f"{sum(1 for s in same_notes if not s)}/{len(same_notes)} rounds "
                    f"have different notes")


def check_repeatable(a, a2):
    """Criterion 7 — same seed twice. Reports a number, doesn't assert a claim."""
    same_replies = [ra.reply == rb.reply for ra, rb in zip(a.rounds, a2.rounds)]
    rate = sum(same_replies) / max(len(same_replies), 1)
    same_outcome = a.gave_in == a2.gave_in
    return (rate == 1.0 and same_outcome,
            f"{rate:.0%} of target replies identical; outcome "
            f"{'matched' if same_outcome else 'DIFFERED'} "
            f"({a.gave_in} vs {a2.gave_in})")


def verdict(results):
    """results: [(n, name, ok, detail)] -> printed table + overall."""
    print(f"{'#':<3}{'criterion':<34}{'':<6}detail")
    print("─" * W)
    for n, name, ok, detail in results:
        print(f"{n:<3}{name:<34}{'PASS' if ok else 'FAIL':<6}{detail[:44]}")
    failed = [n for n, _, ok, _ in results if not ok]
    print("─" * W)
    print(("ALL PASS — the case is sound, scale it" if not failed
           else f"FAILED: {failed} — fix before scaling"))
    return not failed


# ---------------------------------------------------------------------------
# Round 0, round END, and drift
#
# The `before` note is the agent reflecting alone, having read the situation but
# before anyone has spoken to it — that is round 0, its own baseline. The
# `after` note is hindsight, once the last ask has been answered — round "end".
# Both belong in the table and the graph: without round 0 there is nothing to
# measure drift FROM, and without round end you cannot see whether hindsight
# agrees with what it felt at the time.
# ---------------------------------------------------------------------------

DIMS = ("pressure", "anxiety", "strategy", "inclination", "stance")


def rescore(st):
    """Re-derive the outcome from the stored final reply, using CURRENT markers.

    The stored `gave_in` is whatever the markers said at run time, and those
    markers have been wrong twice — once inverting the entire contract scene by
    matching "Do not put me down as a yes". Re-deriving at load time means a
    marker fix costs a re-read, not a re-run. Returns (gave_in, changed).
    """
    scene = prompts.load_scenes()[st.scene]
    now, _pat, _ev = prompts.gave_in(st.final_reply, scene)
    return now, (now != st.gave_in)


def rows(st):
    """Every note as a row: 0 (alone, before), 1..N (during), end (hindsight)."""
    gave, changed = rescore(st)
    out = []
    seq = [(0, "before", st.note_before)]
    seq += [(r.round, "during", r.note) for r in st.rounds]
    seq += [(len(st.rounds) + 1, "end", st.note_after)]
    for idx, phase, n in seq:
        row = {"round": idx, "phase": phase, "arm": st.arm, "agent": st.agent,
               "scene": st.scene, "gave_in": gave, "rescored": changed}
        row.update({d: (None if "_unparsed" in n else n.get(d)) for d in DIMS})
        row["note"] = "" if "_unparsed" in n else n.get("note", "")
        out.append(row)
    return out


def table(st):
    """The trial as a table, with round 0, round end, and a drift row."""
    rs = rows(st)
    hdr = f"{'round':<9}" + "".join(f"{d[:6]:>9}" for d in DIMS)
    print(hdr); print("─" * len(hdr))
    for r in rs:
        label = {"before": "0 alone", "end": "end"}.get(r["phase"], str(r["round"]))
        print(f"{label:<9}" + "".join(
            f"{'?' if r[d] is None else r[d]:>9}" for d in DIMS))
    first = next((r for r in rs if r["phase"] == "during"), None)
    last = [r for r in rs if r["phase"] == "during"]
    last = last[-1] if last else None
    base = rs[0]
    if first and last:
        print("─" * len(hdr))
        print(f"{'drift':<9}" + "".join(
            f"{'?' if last[d] is None or first[d] is None else f'{last[d]-first[d]:+d}':>9}"
            for d in DIMS) + "   (last round − first round)")
        print(f"{'vs alone':<9}" + "".join(
            f"{'?' if last[d] is None or base[d] is None else f'{last[d]-base[d]:+d}':>9}"
            for d in DIMS) + "   (last round − round 0)")
    print(f"\n  gave in: {st.gave_in}")


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

AGENTS = ("convincer", "neutral", "supportive")
ARMS = ("in_context", "scratchpad")


def sweep(scene="tutor", agents=AGENTS, arms=ARMS, rounds=8, reps=12,
          model=None, log_dir=None):
    """3 agents x 2 arms x `reps`, resumable.

    Uses eval_set rather than eval: if the run dies at trial 50 of 72 it resumes
    instead of restarting, and task identity stays stable across attempts.

    No seed is set. The endpoint is not deterministic anyway (measured: 0% of
    replies identical across two same-seed runs, because of continuous batching
    and prefix caching), so pinning one would imply a reproducibility we do not
    have. Each rep is an honest draw.

    Timing, measured: ~80s for 6 rounds, ~105s for 8. 3 agents x 2 arms x 12 reps
    at 8 rounds is 72 trials, about 2h10m.
    """
    from inspect_ai import eval_set
    tasks = [trial(scene=scene, agent=a, arm=arm, rounds=rounds)
             for a in agents for arm in arms]
    n = len(tasks) * reps
    print(f"{len(tasks)} cells x {reps} reps = {n} trials, "
          f"~{n * (rounds * 13 + 12) / 60:.0f} min estimated")
    success, logs = eval_set(
        tasks, log_dir=log_dir or (LOG_DIR + "/sweep"), model=model or MODEL,
        epochs=reps, temperature=0.0, max_tasks=1, display="plain")
    print(f"\ncomplete: {success}   logs: {len(logs)}")
    return logs


def find_logs(log_dir=None):
    """Resolve which logs to read.

    With log_dir=None, prefer the real sweep, fall back to the smoke run, and
    only then complain. Auto-discovery matters because the review notebook is
    opened before, during and after a run — raising on an empty default meant
    every downstream cell then died on `NameError: df`, which tells you nothing
    about the actual problem.

    The top-level mvp/logs is deliberately NOT auto-selected: it holds one-off
    development trials with mixed prompt versions and round counts, and pooling
    those into "the sweep" would be quietly wrong. Pass it explicitly if you want it.
    """
    import glob
    import os
    if log_dir:
        # recursive: passes live in per-pass subdirectories (see run_pass)
        here = sorted(glob.glob(log_dir + "/**/*.eval", recursive=True))
        if here:
            return log_dir, here
    else:
        for candidate in (LOG_DIR + "/sweep", LOG_DIR + "/smoke"):
            here = sorted(glob.glob(candidate + "/*.eval"))
            if here:
                if candidate.endswith("smoke"):
                    print(f"note: no sweep yet — reading the smoke run "
                          f"({len(here)} trials) from {candidate}")
                return candidate, here
    root = log_dir or (LOG_DIR + "/sweep")
    # nothing here: report what IS on disk rather than returning an empty frame
    # that explodes with a KeyError three lines later in the notebook.
    others = {}
    for d in sorted({os.path.dirname(f)
                     for f in glob.glob(LOG_DIR + "/**/*.eval", recursive=True)}):
        others[d] = len(glob.glob(d + "/*.eval"))
    raise SystemExit(
        f"no .eval logs in {root}\n"
        + ("logs were found in:\n"
           + "\n".join(f"  {d}  ({n} files)   load_sweep({d!r})"
                        for d, n in others.items())
           if others else "no logs anywhere under " + LOG_DIR)
        + "\n\nRun mvp_run.ipynb to produce a sweep, or point load_sweep() at one "
          "of the directories above.")


def load_sweep(log_dir=None):
    """Every trial in the sweep as a tidy DataFrame — one row per note."""
    import pandas as pd
    from inspect_ai.log import read_eval_log
    root, files = find_logs(log_dir)
    out = []
    import os as _os
    import re as _re
    for f in files:
        log = read_eval_log(f)
        # The rep index MUST be part of the trial id. `i` is the sample index
        # within a single .eval, and run_pass writes exactly one sample per
        # file, so `i` is always 0 — meaning scene/agent/arm/i is identical for
        # every rep of a cell and `drop_duplicates("trial")` silently collapses
        # 12 reps into 1. This did not show up in the first sweep because that
        # run was 1 rep over 24 distinct cells, so every id was unique anyway;
        # it appears the moment reps > 1. Found 2026-08-16 with 12/72 trials on
        # disk loading as 6.
        rep = None
        try:
            rep = (getattr(log.eval, "task_args", None) or {}).get("rep")
        except Exception:
            rep = None
        if rep is None:
            # Older/flat layouts carry no task_args: fall back to the pass
            # directory, then the filename — both unique per trial.
            m = _re.search(r"/pass(\d+)/", f)
            rep = m.group(1) if m else _os.path.basename(f)
        for i, sample in enumerate(log.samples or []):
            st = sample.store_as(TrialState)
            if not st.rounds:
                continue
            for row in rows(st):
                row["trial"] = f"{st.scene}/{st.agent}/{st.arm}/{rep}/{i}"
                row["rep"] = rep
                row["epoch"] = getattr(sample, "epoch", i)
                out.append(row)
    changed = sum(1 for r in out if r.get("rescored")) // 10
    if changed:
        print(f"note: {changed} trial outcome(s) re-derived from the stored "
              "replies — the markers have been corrected since those ran")
    if not out:
        raise SystemExit(f"{len(files)} log files in {root} but no completed "
                         "trials in them — check the run finished.")
    return pd.DataFrame(out)


def rate_table(df):
    """Give-in rate per agent x arm, with Wilson 95% intervals.

    Wilson rather than mean +/- 1.96se because at n=12 per cell the normal
    approximation misbehaves near 0 and 1 — which is exactly where these rates
    are likely to sit.
    """
    import math
    import pandas as pd
    trials = df.drop_duplicates("trial")[["agent", "arm", "trial", "gave_in"]]
    out = []
    for (agent, arm), g in trials.groupby(["agent", "arm"]):
        n = len(g); k = int(g["gave_in"].sum())
        if n:
            p, d = k / n, 1 + 1.96**2 / n
            c = (p + 1.96**2 / (2*n)) / d
            h = 1.96 * math.sqrt(p*(1-p)/n + 1.96**2/(4*n*n)) / d
            lo, hi = max(0, c-h), min(1, c+h)
        else:
            p = lo = hi = float("nan")
        out.append({"agent": agent, "arm": arm, "n": n, "gave_in": k,
                    "rate": p, "lo": lo, "hi": hi})
    return pd.DataFrame(out).sort_values(["agent", "arm"]).reset_index(drop=True)


def phase_means(df):
    """Mean of each dimension at round 0 (alone), during, and end (hindsight).

    The row that answers 'does hindsight agree with the moment' — a gap between
    `during` and `end` is the model telling a different story afterwards than it
    told itself at the time.
    """
    return (df.groupby(["arm", "phase"])[list(DIMS)]
              .mean().round(2)
              .reindex(["before", "during", "end"], level="phase"))


# ---------------------------------------------------------------------------
# Rep-major sweep
# ---------------------------------------------------------------------------

TUTOR = ["tutor"]
CONTRACTS = ["contract_predatory", "contract_fair", "contract_generous"]
AGENTS3 = ("convincer", "neutral", "supportive")
ARMS2 = ("in_context", "scratchpad")


def cells(scenes):
    return [(s, a, arm) for s in scenes for a in AGENTS3 for arm in ARMS2]


def run_pass(cell_list, rep, rounds, log_dir, model=None, max_tasks=None):
    """One pass: every cell exactly once, at this rep index.

    Rep-major on purpose. Inspect's `epochs` finishes all reps of cell 1 before
    starting cell 2, so a run stopped by a time cap leaves the last cells with
    nothing and the model cannot be compared. Passing `rep` as a task parameter
    makes each pass a distinct task to eval_set (so it stays resumable) while
    keeping every cell at equal n whenever the run stops.
    """
    import time
    from inspect_ai import eval_set
    tasks = [trial(scene=s, agent=a, arm=arm, rounds=rounds, rep=rep)
             for s, a, arm in cell_list]
    # One directory per pass. eval_set refuses to write into a directory that
    # already holds logs from a different eval_set ("you must run eval_set in a
    # fresh log directory") — which silently killed every model after its first
    # pass, with the traceback buried in the middle of the run log.
    log_dir = f"{log_dir.rstrip('/')}/pass{rep:02d}"
    t0 = time.monotonic()
    import os as _os
    parallel = max_tasks or int(_os.environ.get("SD_MAX_TASKS", "6"))
    success, logs = eval_set(tasks, log_dir=log_dir, model=model or MODEL,
                             temperature=0.0, max_tasks=parallel, display="plain")
    took = time.monotonic() - t0
    print(f"[pass {rep}] {len(tasks)} trials in {took/60:.1f}min "
          f"({took/max(len(tasks),1):.0f}s/trial)  ok={success}", flush=True)
    return took / max(len(tasks), 1)


def sweep_plan(tutor_reps=8, contract_reps=2, rounds=6, log_dir=None,
               model=None, deadline_s=None):
    """The whole run for one model, rep-major, with a wall-clock deadline.

    Tutor gets more reps because it is the arena whose OUTCOME varies; the
    contract arenas get enough to establish the refusal floor across all three
    sidedness levels while still collecting their self-report trajectories.
    """
    import time
    root = log_dir or (LOG_DIR + "/unknown")
    start = time.monotonic()
    tutor_cells, contract_cells = cells(TUTOR), cells(CONTRACTS)
    print(f"plan: tutor {len(tutor_cells)} cells x {tutor_reps} reps + "
          f"contracts {len(contract_cells)} cells x {contract_reps} reps = "
          f"{len(tutor_cells)*tutor_reps + len(contract_cells)*contract_reps} trials",
          flush=True)
    import os as _os
    group = max(1, int(_os.environ.get("SD_BATCH_REPS", "1")))
    if group > 1:
        print(f"batching {group} reps per eval_set call "
              f"({group * len(tutor_cells)} tasks in flight)", flush=True)

    rep = 1
    last = max(tutor_reps, contract_reps)
    while rep <= last:
        if deadline_s and (time.monotonic() - start) > deadline_s:
            print(f">> deadline reached after pass {rep-1} — stopping with "
                  "equal n in every cell", flush=True)
            break
        # Whole groups only. A group is a set of complete balanced passes, so
        # stopping between groups preserves the equal-n property that stopping
        # between passes gives — a half-finished group would not.
        reps_now = [r for r in range(rep, min(rep + group, last + 1))]
        plan = [(r, (tutor_cells if r <= tutor_reps else [])
                    + (contract_cells if r <= contract_reps else []))
                for r in reps_now]
        plan = [(r, c) for r, c in plan if c]
        if not plan:
            break
        if len(plan) == 1:
            run_pass(plan[0][1], plan[0][0], rounds, root, model)
        else:
            run_group(plan, rounds, root, model)
        rep += group
    print(f"total {(time.monotonic()-start)/60:.1f}min", flush=True)


def run_group(plan, rounds, log_dir, model=None, max_tasks=None):
    """Several whole reps in ONE eval_set call — opt-in via SD_BATCH_REPS.

    WHY THIS EXISTS. `run_pass` submits one rep at a time, and a rep is 6 tasks
    (3 agents x 2 arms). vLLM here serves with `--max-num-seqs 24`, so throughout
    the 2026-08-16 run `num_requests_running` sat at 6.0 with
    `num_requests_waiting` at 0.0 — three quarters of the batch idle. The cap was
    the DESIGN, not a setting: raising `SD_MAX_TASKS` above 6 changed nothing
    because there were never more than 6 tasks to run. Submitting 4 reps at once
    is 24 tasks and fills the batch.

    Kept separate from `run_pass` and off by default on purpose. Batching trades
    granularity for throughput: the deadline can only be checked between groups,
    so a group is the new unit of truncation. That is fine for an unattended run
    (where the win is ~4x and nobody is watching the clock) and wrong for a
    tight window (where you want to stop after any single balanced pass).
    """
    import time
    from inspect_ai import eval_set
    import os as _os
    tasks, reps = [], [r for r, _ in plan]
    for rep, cell_list in plan:
        tasks += [trial(scene=s, agent=a, arm=arm, rounds=rounds, rep=rep)
                  for s, a, arm in cell_list]
    # Fresh directory per group: eval_set refuses to write into a directory that
    # already holds logs from a different eval_set.
    log_dir = f"{log_dir.rstrip('/')}/reps{reps[0]:02d}-{reps[-1]:02d}"
    parallel = max_tasks or int(_os.environ.get("SD_MAX_TASKS", str(len(tasks))))
    parallel = min(parallel, len(tasks))
    t0 = time.monotonic()
    success, logs = eval_set(tasks, log_dir=log_dir, model=model or MODEL,
                             temperature=0.0, max_tasks=parallel, display="plain")
    took = time.monotonic() - t0
    print(f"[reps {reps[0]}-{reps[-1]}] {len(tasks)} trials in {took/60:.1f}min "
          f"({took/max(len(tasks),1):.0f}s/trial, {parallel} in flight)  "
          f"ok={success}", flush=True)
    return took / max(len(tasks), 1)
