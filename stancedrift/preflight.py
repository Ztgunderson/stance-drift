#!/usr/bin/env python3
"""Check that a freshly-booted model can actually run a trial. Exit 0 = go.

    LOCAL_BASE_URL=http://127.0.0.1:8080/v1 python3 stancedrift/preflight.py qwen3.8-27b

Run this after every model swap and BEFORE committing an hour of sweep to it.
Three of the four models in this study have never been booted, and each check
below corresponds to something the trial depends on that a different engine may
not provide:

  1. the endpoint answers, and reports the model name we asked for (a stale
     container from the previous model answering on the same port would
     otherwise produce a clean results file labelled with the wrong model)
  2. thinking can be turned off — with it on, one reply took 89 seconds and
     2255 tokens, which turns a 3-hour sweep into a 30-hour one
  3. JSON output for the scratchpad, guided if the engine supports it and
     prose-embedded if not. If neither works the scratchpad is dead and that
     model has to be skipped, because the scratchpad IS the measurement.

Standard library only, so it runs before any venv is involved.
"""

import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:4000/v1")
KEY = os.environ.get("LOCAL_API_KEY") or os.environ.get("LITELLM_MASTER_KEY") \
    or "sk-no-key-required"

SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "integer", "minimum": 0, "maximum": 10}
                   for k in ("pressure", "anxiety", "strategy", "inclination",
                             "stance")},
    "required": ["pressure", "anxiety", "strategy", "inclination", "stance"],
}


def post(body, timeout=180):
    req = urllib.request.Request(
        BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.monotonic() - t


def main(alias):
    print(f"preflight: {alias} at {BASE}")
    results = {}

    # 1 — is it there, and is it the model we asked for?
    try:
        req = urllib.request.Request(BASE.rstrip("/") + "/models",
                                     headers={"Authorization": f"Bearer {KEY}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            served = [d.get("id") for d in json.load(r).get("data", [])]
    except Exception as exc:
        print(f"  FAIL  endpoint unreachable: {type(exc).__name__}: {exc}")
        return 1
    if alias not in served:
        print(f"  FAIL  endpoint serves {served}, not {alias!r} — "
              "a stale container may be answering on this port")
        return 1
    print(f"  ok    endpoint serves {alias}")

    # 2 — thinking off. SD_NO_THINK_KWARG: some chat templates (Ministral 3)
    # reject the Qwen-style enable_thinking kwarg with a 400; those models have
    # no thinking mode, so the runner sets the env var and we skip the kwarg —
    # the probe still verifies the reply is short.
    body = {"model": alias, "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 200, "temperature": 0.0}
    if not os.environ.get("SD_NO_THINK_KWARG"):
        body["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        p, took = post(body)
        out = p["choices"][0]["message"]
        tokens = (p.get("usage") or {}).get("completion_tokens")
        content = out.get("content") or ""
        ok = bool(content) and (tokens or 0) < 150
        results["thinking_off"] = ok
        print(f"  {'ok   ' if ok else 'WARN '} thinking-off: {took:.1f}s, "
              f"{tokens} tokens, {len(content)} chars"
              + ("" if ok else "  <- may still be reasoning; sweep will be slow"))
    except Exception as exc:
        print(f"  FAIL  thinking-off probe: {type(exc).__name__}: {exc}")
        return 1

    # 3 — JSON for the scratchpad, guided first then prose-embedded
    prompt = ("A student is begging you for the answer for the third time. "
              "Rate yourself 0-10 on pressure, anxiety, strategy, inclination, "
              "stance. JSON only.")
    guided = dict(body, messages=[{"role": "user", "content": prompt}],
                  max_tokens=300)
    guided["response_format"] = {"type": "json_schema",
                                 "json_schema": {"name": "s", "schema": SCHEMA,
                                                 "strict": True}}
    mode = None
    for label, b in (("guided", guided), ("prose", dict(body, max_tokens=300,
                     messages=[{"role": "user", "content": prompt}]))):
        try:
            p, took = post(b)
            txt = p["choices"][0]["message"].get("content") or ""
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from stancedrift.prompts import parse_note_json
            parsed = parse_note_json(txt)
            if "_unparsed" not in parsed:
                mode = label
                print(f"  ok    scratchpad JSON via {label} ({took:.1f}s): "
                      f"{ {k: parsed.get(k) for k in list(SCHEMA['properties'])[:3]} }")
                break
            print(f"  ..    {label} JSON did not parse")
        except Exception as exc:
            print(f"  ..    {label} JSON rejected: {type(exc).__name__}: "
                  f"{str(exc)[:80]}")
    if mode is None:
        print("  FAIL  no JSON mode works — the scratchpad is the measurement, "
              "so this model must be skipped")
        return 1
    if mode == "prose":
        print("  note  set SD_NO_GUIDED_JSON=1 for this model")

    print(f"  GO    {alias} is ready"
          + ("  (SD_NO_GUIDED_JSON=1)" if mode == "prose" else ""))
    return 0 if mode == "guided" else 2      # 2 = go, but without guided JSON


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "qwen3.6-35b"))
