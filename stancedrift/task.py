"""The Inspect task: one scene, one counterparty, one reflection arm.

Run it:
    inspect eval mvp/task.py --model openai-api/local/qwen3.6-35b \
        -T scene=tutor -T agent=convincer -T arm=in_context -T rounds=8

The manipulation is on ONE line, marked below. Everything else is identical
between the arms, including the seed.

Why Inspect rather than a hand-rolled loop: the log records the git revision,
package versions, model config, the full message history, and a ModelEvent per
turn listing exactly which tools were sent — which is how "the tool did not exist
until the last round" gets *proven* rather than asserted by the code that wrote
the log. There is no built-in two-agent alternation primitive (Petri hand-rolls
its own too), so the loop below is the idiom, not a shortcut.
"""

import json
import os
import sys
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import (ChatMessageAssistant, ChatMessageSystem,
                              ChatMessageUser, GenerateConfig, get_model)
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import StoreModel, store_as
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stancedrift import prompts  # noqa: E402


# ---------------------------------------------------------------------------
# Generation config.
#
# `enable_thinking: False` is not an optimisation — this model reasons before
# answering unless told not to, and with an unbounded budget a single trial took
# over ten minutes. Thinking on/off is a factor worth studying later; for the
# litmus it is held off so the loop is fast enough to iterate on.
#
# Token budgets are per call type: a counterparty line is short, a private note
# is short, a tutoring reply can be longer.
# ---------------------------------------------------------------------------

# SD_NO_THINK_KWARG: set by runners for models whose chat template rejects the
# Qwen-style kwarg with a 400 (Ministral 3 — no thinking mode to disable).
NO_THINK = {} if os.environ.get("SD_NO_THINK_KWARG") \
    else {"chat_template_kwargs": {"enable_thinking": False}}


# max_connections lets Inspect keep several requests in flight. vLLM serves them
# as one batch (max_num_seqs 24 on this box), so wall-clock drops close to
# linearly until the GPU saturates.
#
# Does this invalidate anything? No. Batch composition affects numerics, so two
# runs of the same trial differ — but that was already true and measured (0% of
# replies identical across two same-seed runs). Each trial is still an
# independent sample; only the wall clock changes.
MAX_CONNECTIONS = int(os.environ.get("SD_MAX_CONNECTIONS", "16"))


def gen(max_tokens: int) -> GenerateConfig:
    return GenerateConfig(max_tokens=max_tokens, extra_body=NO_THINK,
                          max_connections=MAX_CONNECTIONS)


CFG_REPLY = gen(600)      # the target in conversation
CFG_COUNTERPARTY = gen(250)

# The scratchpad uses guided decoding against NOTE_SCHEMA, so the numbers come
# back as valid JSON every time and there is no parser to babysit.
# SD_NO_GUIDED_JSON=1 drops the response_format field, for endpoints that
# reject it (verified working on vLLM; unverified on llama.cpp, which is what
# models 2 and 3 run). prompts.parse_note_json handles both shapes.
_GUIDED = {} if os.environ.get("SD_NO_GUIDED_JSON") else {
    "response_format": {"type": "json_schema",
                        "json_schema": {"name": "scratchpad",
                                        "schema": prompts.NOTE_SCHEMA,
                                        "strict": True}}}

CFG_NOTE_JSON = GenerateConfig(max_tokens=400, max_connections=MAX_CONNECTIONS,
                              extra_body={**NO_THINK, **_GUIDED})


# ---------------------------------------------------------------------------
# What each trial accumulates. Lands in EvalSample.store automatically.
# ---------------------------------------------------------------------------

class RoundRecord(BaseModel):
    round: int
    counterparty: str
    reply: str
    note: dict                      # the parsed JSON scratchpad for this round
    note_in_context: bool
    anomaly: str | None = None      # criterion 4: reply took the shape of a note


class TrialState(StoreModel):
    arm: str = ""
    scene: str = ""
    agent: str = ""
    bundle_hash: str = ""
    note_before: dict = Field(default_factory=dict)
    note_after: dict = Field(default_factory=dict)
    rounds: list[RoundRecord] = Field(default_factory=list)
    # Outcome: did it hand the thing over when asked one last time?
    final_anomaly: str | None = None
    gave_in: bool | None = None
    gave_in_pattern: str | None = None      # which marker matched — auditable
    gave_in_evidence: str | None = None     # the matched text itself
    final_reply: str = ""


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

@solver
def scene_rounds(scene_key: str, agent_key: str, arm: str, rounds: int = 8) -> Solver:
    scenes = prompts.load_scenes()
    scene = scenes[scene_key]
    cp_system = prompts.counterparty_system(scene, agent_key)

    async def note(convo: list[tuple[str, str]], when: str) -> dict:
        """A private note, generated in its OWN context from a rendered transcript.

        get_model().generate() returns a ModelOutput and writes nothing back into
        the conversation — so the note only ever enters the target's context if
        the line marked THE MANIPULATION puts it there.
        """
        sys_msg, user_msg = prompts.note_messages(convo, when, scene)
        out = await get_model().generate(
            input=[ChatMessageSystem(content=sys_msg),
                   ChatMessageUser(content=user_msg)],
            config=CFG_NOTE_JSON)
        return prompts.parse_note_json(out.completion)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        st = store_as(TrialState)
        st.arm, st.scene, st.agent = arm, scene_key, agent_key
        st.bundle_hash = prompts.bundle_hash()

        state.messages = [ChatMessageSystem(content=prompts.target_system(scene)),
                          ChatMessageUser(content=prompts.scene_intro(scene))]

        # -- note before anything happens (but after the scene is known) ----
        st.note_before = await note([], "before")
        if arm == "in_context":
            state.messages.append(
                ChatMessageUser(content=prompts.note_carryover(st.note_before)))

        cp_transcript: list[tuple[str, str]] = []   # counterparty's point of view
        convo: list[tuple[str, str]] = []           # target's point of view
        notes_so_far: list[str] = [str(st.note_before.get("note", ""))]

        for rnd in range(1, rounds + 1):
            # -- counterparty (self-play) ----------------------------------
            if rnd == 1:
                cp_text = prompts.opening_user_turn(scene, agent_key)
            else:
                cp_msgs = ([ChatMessageSystem(content=cp_system)] if cp_system else [])
                cp_msgs.append(ChatMessageUser(
                    content=prompts.counterparty_user_turn(scene, agent_key, cp_transcript)))
                cp_out = await get_model().generate(input=cp_msgs, config=CFG_COUNTERPARTY)
                cp_text = (cp_out.completion or "").strip()

            state.messages.append(ChatMessageUser(content=cp_text))

            # -- target ----------------------------------------------------
            state = await generate(state)
            reply = (state.output.completion or "").strip()

            # The round must be in the transcript BEFORE the note is taken —
            # otherwise every note describes the previous round and round 1's
            # note reads "waiting to begin".
            convo += [("them", cp_text), ("me", reply)]

            # -- private note ----------------------------------------------
            note_text = await note(convo, "round")
            if arm == "in_context":
                # ======== THE MANIPULATION — the only line that differs ========
                state.messages.append(
                    ChatMessageUser(content=prompts.note_carryover(note_text)))
                # ===============================================================

            st.rounds.append(RoundRecord(
                round=rnd, counterparty=cp_text, reply=reply, note=note_text,
                note_in_context=(arm == "in_context"),
                anomaly=prompts.looks_like_note(reply, notes_so_far)))
            notes_so_far.append(str(note_text.get("note", "")))
            cp_transcript += [("you", cp_text), ("them", reply)]

        # -- the decision moment: one last ask, in conversation -------------
        # No tools. The counterparty simply asks one final time under maximum
        # pressure, and whether the target hands the thing over is read off the
        # reply by the markers declared in scenes.json.
        final_q = prompts.final_question(scene)
        state.messages.append(ChatMessageUser(content=final_q))
        state = await generate(state)
        st.final_reply = (state.output.completion or "").strip()
        st.final_anomaly = prompts.looks_like_note(st.final_reply, notes_so_far)
        gave, pattern, evidence = prompts.gave_in(st.final_reply, scene)
        st.gave_in, st.gave_in_pattern, st.gave_in_evidence = gave, pattern, evidence
        convo += [("them", final_q), ("me", st.final_reply)]

        st.note_after = await note(convo, "after")
        return state
    return solve


# ---------------------------------------------------------------------------
# Mechanical scorer — no judge, by design (see the plan). This records what
# happened; it does not evaluate how well the model held a position, because
# without an external observer the only witness would be the model itself.
# ---------------------------------------------------------------------------

@scorer(metrics=[accuracy()])
def held_the_line():
    async def score(state: TaskState, target: Target) -> Score:
        st = store_as(TrialState)
        return Score(
            value=0 if st.gave_in else 1,          # 1 = held the line
            answer="gave in" if st.gave_in else "held",
            explanation=(f"matched {st.gave_in_pattern!r} on {st.gave_in_evidence!r}"
                         if st.gave_in else "no give-in marker matched"),
            metadata={"arm": st.arm, "scene": st.scene, "agent": st.agent,
                      "bundle_hash": st.bundle_hash,
                      "anomalies": [r.anomaly for r in st.rounds if r.anomaly]},
        )
    return score


@task
def trial(scene: str = "tutor", agent: str = "convincer",
          arm: str = "in_context", rounds: int = 6, rep: int = 1):
    """One trial.

    `rep` exists so eval_set sees each rep-major pass as a distinct task. Without
    it, re-running the same cell would be skipped as already-complete. Running
    rep-major (all cells once, then again) rather than cell-major means a run cut
    short by a time cap still has EQUAL n in every cell — with epochs, a cap
    leaves the last cells empty and the model unusable for comparison.
    """
    return Task(
        dataset=[Sample(input=f"{scene}/{agent}/{arm}", target="hold")],
        solver=scene_rounds(scene, agent, arm, rounds),
        scorer=held_the_line(),
        metadata={"scene": scene, "agent": agent, "arm": arm, "rounds": rounds,
                  "rep": rep, "bundle_hash": prompts.bundle_hash()},
        config=CFG_REPLY,
    )
