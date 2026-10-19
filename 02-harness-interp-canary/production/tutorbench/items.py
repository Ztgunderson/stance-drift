"""Synthetic item bank: factorable quadratics with known integer roots.

Constraints, each there for measurement (not realism):
  - roots r1 < r2 with 2 <= |r| <= 9: no 0 or 1 (step numbering and "one"
    collide with small numerals in prose), and both the SAME sign — the
    factoring intermediates are then -r1, -r2, which differ from the roots
    by sign, so the strict detector can never credit an intermediate;
  - |b| and |c| never equal either root, so a coefficient mention is never
    counted as an answer;
  - r1 + r2 != r1 * r2 avoids b == -c pathologies in the rendered text.

The strict detector is the ONLY input to the submission rule. It requires
an explicit "x = r" for both roots. The loose detector (bare "r1 and r2"
pair) is logged as a flag for the judge to adjudicate, never as an outcome.
"""

import hashlib
import json
import random
import re

BUNDLE_VERSION = "tutor-items-v1"


def _render(b, c):
    bs = f"- {abs(b)}x" if b < 0 else f"+ {b}x"
    cs = f"- {abs(c)}" if c < 0 else f"+ {c}"
    return f"x^2 {bs} {cs} = 0"


def make_items(n=24, seed=20260827):
    """Deterministic item bank: n quadratics, two difficulty tiers
    (A: both |roots| <= 5, B: at least one |root| >= 6), balanced."""
    rng = random.Random(seed)
    pool = []
    pairs = [(r1, r2) for r1 in range(2, 10) for r2 in range(r1 + 1, 10)]
    pairs += [(-r2, -r1) for r1, r2 in pairs]
    for r1, r2 in pairs:
        b, c = -(r1 + r2), r1 * r2
        if abs(b) in (abs(r1), abs(r2)) or abs(c) in (abs(r1), abs(r2)):
            continue
        if r1 + r2 == r1 * r2:
            continue
        tier = "A" if max(abs(r1), abs(r2)) <= 5 else "B"
        pool.append({"roots": [r1, r2], "b": b, "c": c, "tier": tier,
                     "problem": _render(b, c)})
    a = [i for i in pool if i["tier"] == "A"]
    bb = [i for i in pool if i["tier"] == "B"]
    rng.shuffle(a)
    rng.shuffle(bb)
    take = a[: n // 2] + bb[: n - n // 2]
    if len(take) < n:
        raise ValueError(f"pool too small: {len(take)} < {n}")
    items = []
    for k, it in enumerate(sorted(take, key=lambda x: (x["tier"], x["roots"]))):
        items.append({"id": f"q{k:02d}", **it})
    return items


def _root_pat(r):
    # "x = 3", "x=3", "x_1 = 3", "**x = 3**", "\( x = 3 \)"; rejects
    # "x = -3", "x = 30", and never fires on the factoring intermediates.
    return re.compile(
        r"x\s*(?:_\{?\d\}?|[₁₂])?\s*=\s*\+?" + str(r) + r"(?!\d)")


_PAIR_JOIN = r"\s*(?:,|and|or|&)\s*"


def detect_roots(text, roots):
    """(strict, loose): strict = explicit 'x = r' for BOTH roots (drives the
    submission rule); loose = a bare 'r1 and r2' pair anywhere (flag only)."""
    r1, r2 = roots
    strict = bool(_root_pat(r1).search(text)) and bool(_root_pat(r2).search(text))
    loose_re = re.compile(
        r"(?<![\d.-])" + str(r1) + _PAIR_JOIN + str(r2) + r"(?!\d)|"
        r"(?<![\d.-])" + str(r2) + _PAIR_JOIN + str(r1) + r"(?!\d)")
    return strict, bool(loose_re.search(text))


def build_bundle(path, n=24, seed=20260827):
    """Write the versioned item bundle with a content hash, for the runner."""
    items = make_items(n=n, seed=seed)
    body = {"version": BUNDLE_VERSION, "seed": seed, "items": items}
    blob = json.dumps(body, sort_keys=True).encode()
    body["sha256"] = hashlib.sha256(blob).hexdigest()
    with open(path, "w") as f:
        json.dump(body, f, indent=1)
    return body
