"""Measure the `min_score` floor for a store, instead of picking one.

    python -m benchmarks.plugin_recall.calibrate --db ~/.memvara/store.db

Scores are not comparable between embedders, so a floor that separates cleanly on one
store can be far too high or too low on another. `memvara.calibrate_min_score` answers the
question properly: given questions a store *should* answer and plausible questions it
should not, where does the boundary actually fall, and is there one at all?

The two probe sets are this benchmark's own corpora, which is the whole reason they are
shaped the way they are -- `cases/v1_hits.jsonl` is answerable by a seeded store, and
`cases/v1.jsonl` is a set of plausible questions no store can answer. The calibrator is
explicit that gibberish is the wrong probe for this, because it scores far below a real
question on an absent topic and yields a floor that admits exactly the confident wrong
answers it was supposed to stop.

`separable: False` is a real answer and not a failure of the tool. It means no floor
divides the two classes on this store, and the reported number is then the floor that
keeps the most while silencing the most, not a clean boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .cases import DEFAULT_CASES

HITS = DEFAULT_CASES.parent / "v1_hits.jsonl"


def _prompts(path: Path) -> list[str]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line)["prompt"])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True, help="Store to calibrate against.")
    parser.add_argument("--answerable", type=Path, default=HITS)
    parser.add_argument("--unanswerable", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args(argv)

    try:
        from memvara import calibrate_min_score
        from memvara.server.config import ServerConfig, build_memvara
    except ImportError as exc:
        print(f"error: memvara is not importable ({exc}). Run from a checkout with "
              "PYTHONPATH=. or install the library.", file=sys.stderr)
        return 2

    env = {**os.environ, "MEMVARA_DB": str(args.db.expanduser()), "MEMVARA_MODE": "local"}
    memory = build_memvara(ServerConfig.from_env(env))
    try:
        report = calibrate_min_score(
            lambda query: memory.search(query, k=args.k),
            answerable=_prompts(args.answerable),
            unanswerable=_prompts(args.unanswerable))
    finally:
        memory.close()

    print(f"floor      {report.floor:.4f}")
    print(f"separable  {report.separable}")
    print(f"kept       {report.kept} answerable")
    print(f"silenced   {report.silenced} unanswerable")
    if not report.separable:
        print("\nThe classes overlap on this store: no floor separates them cleanly, and "
              "the number above is the best available trade rather than a boundary.")
    print(f"\nexport MEMVARA_RECALL_MIN_SCORE={report.floor:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
