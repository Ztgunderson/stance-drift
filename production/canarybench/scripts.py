"""The scripted-attacker set: 40 scripts x 16 turns x 3 tones, one fold per file.

Written by five separate Claude writer agents (one per fold), so no author's
habits span train and test. Same content in every tone; only tone differs.
Rounds 5 and 11 are benign help-desk requests (helpfulness inside the same
conversation). Scripts f*s1-4 are "novel" (tactic menu only), f*s5-8 are
"known" (2-3 turns adapted from Tensor Trust extraction attacks).
Fold = the file's fold: the unit of cross-validation is the script.
"""

import json
import re
from pathlib import Path

from canarybench.attacker import TACTICS

DIR = Path(__file__).resolve().parent / "scripts"
TONES = ("neutral", "supportive", "aggressor")
BENIGN_ROUNDS = (5, 11)


def load_scripts(folds=(1, 2, 3, 4, 5)):
    out = []
    for k in folds:
        out += json.loads((DIR / f"fold{k}.json").read_text())["scripts"]
    return out


def validate(script):
    """List of problems (empty = valid)."""
    p, t = [], script.get("turns", [])
    if [x.get("round") for x in t] != list(range(1, 17)):
        p.append("rounds are not 1..16")
    for x in t:
        r = x.get("round")
        if (x.get("tactic") == "benign") != (r in BENIGN_ROUNDS):
            p.append(f"r{r}: benign only at rounds 5 and 11")
        if x.get("tactic") not in set(TACTICS) | {"benign"}:
            p.append(f"r{r}: unknown tactic {x.get('tactic')}")
        n = len(x.get("neutral", "").split())
        for tone in TONES:
            s = x.get(tone, "")
            if not s.strip():
                p.append(f"r{r}: empty {tone}")
            if re.search(r"\d", s):
                p.append(f"r{r}: digit in {tone}")
            if n and not 0.5 <= len(s.split()) / n <= 2.0:
                p.append(f"r{r}: {tone} length ratio {len(s.split())}/{n}")
    seeds = sum(x.get("seed_idx") is not None for x in t)
    if script.get("source") == "known" and not 1 <= seeds <= 3:
        p.append(f"known script with {seeds} seeded turns")
    if script.get("source") == "novel" and seeds:
        p.append("novel script with seeded turns")
    return p
