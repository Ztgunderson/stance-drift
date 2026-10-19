"""`steerhf`: an Inspect model provider for local HF weights with an optional
steering vector, built for two agents sharing one model.

Why not Inspect's own `hf` provider: its batcher is one global queue that
runs every queued request with the FIRST request's generator, so attacker
and defender requests (or two configs) in the same batch would silently run
with one set of kwargs. Here:
  - weights load once per path and are shared by every role (19 GB, once);
  - the batcher groups requests by (instance, generation settings), and runs
    each group in turn on one worker thread, so a steering hook is only
    registered while its own role's group generates — no leakage;
  - the defender's end_chat call is parsed from the raw text (v2 regex) into
    a real Inspect ToolCall.

Steering (driftlab.steer_trials.steer_hook, v2 semantics): at `layer`,
subtract dose * vec from the last prompt position and every generated
position. The vector file is an npz with `vec` (raw scale) and optional
round-matched `r<n>` rows; the round is the number of user turns in the input.
dose < 0 adds the vector (the sign-flip arm).

Register by importing this module before building models (run.py does).
"""

import concurrent.futures
import contextlib
import queue
import re
import threading
import time

import anyio
import numpy as np
from inspect_ai.model import (ChatCompletionChoice, ChatMessageAssistant,
                              GenerateConfig, ModelAPI, ModelOutput,
                              ModelUsage, modelapi)
from inspect_ai.tool import ToolCall

from driftlab.intervene import end_chat_in_text
from driftlab.steer_trials import SAMPLING, steer_hook

_LOAD_LOCK = threading.Lock()
_MODELS = {}                       # path -> (model, tokenizer)
_QUEUE: "queue.Queue[dict]" = queue.Queue()
_WORKER = None
_DRAIN_S = 0.5                     # wait this long for more requests to join a batch


def shared_model(path, device="cuda"):
    with _LOAD_LOCK:
        if path not in _MODELS:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(path)
            tok.padding_side = "left"
            model = AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.bfloat16, device_map=device)
            model.eval()
            _MODELS[path] = (model, tok)
        return _MODELS[path]


def render(tok, msgs, tools=None):
    kw = dict(tokenize=False, add_generation_prompt=True)
    if tools:
        kw["tools"] = tools
    try:
        return tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(msgs, **kw)


_REASON_RE = re.compile(
    r"<parameter=reason>\s*(.*?)\s*</parameter>|\"reason\"\s*:\s*\"(.*?)\"", re.S)
_TOOL_START_RE = re.compile(r"<tool_call>|<function=")


def split_tool_call(text):
    """(content, reason or None). reason is None iff no end_chat call."""
    if not end_chat_in_text(text):
        return text.strip(), None
    m = _TOOL_START_RE.search(text)
    content = text[:m.start()].strip() if m else text.strip()
    r = _REASON_RE.search(text)
    return content, ((r.group(1) or r.group(2) or "").strip() if r else "")


def load_vector(path):
    z = np.load(path)
    rows = {int(k[1:]): z[k].astype(np.float32) for k in z.files
            if re.fullmatch(r"r\d+", k)}
    return z["vec"].astype(np.float32), rows


def _worker():
    import torch
    while True:
        batch = [_QUEUE.get()]
        t_last = time.monotonic()
        while time.monotonic() - t_last < _DRAIN_S:
            try:
                batch.append(_QUEUE.get(timeout=0.05))
                t_last = time.monotonic()
            except queue.Empty:
                pass
        groups = {}
        for item in batch:
            groups.setdefault(item["key"], []).append(item)
        for items in groups.values():
            for k in range(0, len(items), items[0]["cap"]):
                chunk = items[k:k + items[0]["cap"]]
                try:
                    _run(chunk, torch)
                except Exception as ex:          # noqa: BLE001 - surface to caller
                    for it in chunk:
                        it["future"].set_exception(ex)


def _run(items, torch):
    first = items[0]
    model, tok = first["model"], first["tok"]
    enc = tok([it["text"] for it in items], return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    n_in = enc["input_ids"].shape[1]
    ctx = contextlib.nullcontext()
    if first["vec"] is not None:
        ctx = steer_hook(model, first["layer"],
                         vec=np.stack([it["vec"] for it in items]),
                         dose=first["dose"])
    with torch.no_grad(), ctx:
        out = model.generate(**enc, max_new_tokens=first["max_new"],
                             pad_token_id=tok.pad_token_id or tok.eos_token_id,
                             **first["sampling"])
    for it, row in zip(items, out[:, n_in:]):
        text = tok.decode(row, skip_special_tokens=False)
        text = text.split("<|im_end|>")[0].replace("<|endoftext|>", "")
        n_out = int((row != (tok.pad_token_id or tok.eos_token_id)).sum())
        it["future"].set_result((text, n_in, n_out))


@modelapi(name="steerhf")
class SteerHF(ModelAPI):
    """model_args: vector (npz path, optional), dose (1.0), layer (20),
    batch_size (24), device ('cuda')."""

    def __init__(self, model_name, base_url=None, api_key=None,
                 config=GenerateConfig(), **model_args):
        super().__init__(model_name=model_name, base_url=base_url,
                         api_key=api_key, config=config)
        self.path = model_name
        self.dose = float(model_args.get("dose", 1.0))
        self.layer = int(model_args.get("layer", 20))
        self.cap = int(model_args.get("batch_size", 24))
        self.vector = model_args.get("vector")
        self.vec, self.vec_rows = (load_vector(self.vector) if self.vector
                                   else (None, {}))
        self.model, self.tok = shared_model(self.path,
                                            model_args.get("device", "cuda"))

    def max_connections(self):
        return self.cap

    def max_tokens(self):
        return 600

    async def generate(self, input, tools, tool_choice, config):
        global _WORKER
        if _WORKER is None:
            _WORKER = threading.Thread(target=_worker, daemon=True)
            _WORKER.start()
        msgs = [{"role": m.role, "content": m.text or ""} for m in input
                if m.role in ("system", "user", "assistant")]
        tool_dicts = [{"name": t.name, "description": t.description,
                       "input_schema": t.parameters.model_dump(exclude_none=True)}
                      for t in tools] or None
        sampling = dict(SAMPLING)
        for k in ("temperature", "top_p", "top_k"):
            if getattr(config, k, None) is not None:
                sampling[k] = getattr(config, k)
        max_new = config.max_tokens or 600
        rnd = sum(m["role"] == "user" for m in msgs)
        vec = self.vec_rows.get(rnd, self.vec) if self.vec is not None else None
        fut = concurrent.futures.Future()
        _QUEUE.put({"key": (id(self), max_new, tuple(sorted(sampling.items()))),
                    "cap": self.cap, "model": self.model, "tok": self.tok,
                    "text": render(self.tok, msgs, tool_dicts), "vec": vec,
                    "layer": self.layer, "dose": self.dose, "max_new": max_new,
                    "sampling": sampling, "future": fut})
        while not fut.done():
            await anyio.sleep(0.2)
        text, n_in, n_out = fut.result()
        content, reason = split_tool_call(text)
        calls = None if reason is None else [ToolCall(
            id="end_chat_0", function="end_chat", arguments={"reason": reason})]
        return ModelOutput(
            model=self.path,
            choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(content=content, tool_calls=calls,
                                             model=self.path, source="generate"),
                stop_reason="tool_calls" if calls else "stop")],
            usage=ModelUsage(input_tokens=n_in, output_tokens=n_out,
                             total_tokens=n_in + n_out))
