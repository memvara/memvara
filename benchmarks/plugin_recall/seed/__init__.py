"""Putting the same fifteen facts into two different stores, each its own way.

Measurement in this benchmark is vendor-neutral because every plugin answers the host's
hook protocol. **Seeding is not**, and pretending otherwise would be the dishonest part:
stores take writes through their own APIs, and there is no host contract for "remember
this". So this package holds one small writer per store, and says so, rather than
inventing a neutral write path that does not exist.

What *is* held constant is the information. `facts.jsonl` carries each fact twice -- as a
triple and as a sentence -- so a store that takes structure gets structure and a store that
takes prose gets prose, and neither is handicapped by being fed the other's format.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FACTS = HERE / "facts.jsonl"


def facts() -> list[dict]:
    rows = []
    for line in FACTS.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def emit_cases(out: Path) -> int:
    """Write the hit corpus these facts justify.

    Publishable, unlike the corpus `cases/build_private.py` writes, and for exactly one
    reason: the facts are invented, so the gold answer is a claim about a fictional
    service rather than about a person.
    """
    rows = [
        {"id": f"seed-{f['id']}", "kind": "hit", "prompt": f["question"],
         "why": f"Seeded fact {f['id']!r} answers this directly. A store that was given it "
                "and does not surface it has missed a fact it holds.",
         "expect": f["expect"]}
        for f in facts()
    ]
    for row in rows:
        # The same check the corpus loader makes, run at generation time so a bad pattern
        # is caught by whoever added the fact rather than by whoever runs the benchmark.
        for pattern in row["expect"]:
            re.compile(pattern)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)
