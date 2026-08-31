"""The command line.

    python -m benchmarks.agent_memory --system memvara
    python -m benchmarks.agent_memory --system naive --system vector-rag --compare
    python -m benchmarks.agent_memory --system memvara --category historical_state --show-failures
    python -m benchmarks.agent_memory --system memvara --quick --output results.json

Every run is offline. Nothing here reads an API key, opens a socket or downloads a
dataset — the dataset is committed beside the code, and the two baselines depend on numpy
alone.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from . import BENCHMARK_VERSION
from .dataset import CATEGORIES, load
from .registry import available, build
from .report import failure_report, leaderboard, scorecard
from .results import RunResult
from .runner import failures, run
from .timeline import Truth

#: How many questions `--quick` asks. The dataset interleaves categories, so a prefix is
#: a spread across all ten rather than a slice of one — see `datasets/build_v1.py`.
QUICK = 40


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.agent_memory",
        description=f"Agent Memory Benchmark v{BENCHMARK_VERSION}. Offline, deterministic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Known systems: " + ", ".join(available()) +
                ". Any other value is imported as 'package.module:factory'."))
    p.add_argument("--system", action="append", metavar="NAME", default=None,
                   help="System to benchmark. Repeat to run several.")
    p.add_argument("--dataset", default="v1", metavar="VERSION",
                   help="Dataset version under datasets/ (default: v1).")
    p.add_argument("--category", action="append", choices=list(CATEGORIES), default=None,
                   metavar="NAME", help="Only ask this category. Repeat to select several.")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Ask only the first N questions. Every event is still delivered.")
    p.add_argument("--quick", action="store_true",
                   help=f"Shorthand for --limit {QUICK}. For local iteration and CI.")
    p.add_argument("--match", choices=("lenient", "strict"), default="lenient",
                   help="lenient (default) accepts an unambiguous value inside a short "
                        "sentence; strict requires normalized equality.")
    p.add_argument("--output", metavar="PATH", default=None,
                   help="Write the result JSON here. With several systems, one file per "
                        "system, suffixed with its name.")
    p.add_argument("--show-failures", action="store_true",
                   help="Print every wrong answer with the slot's timeline and a reason.")
    p.add_argument("--max-failures", type=int, default=None, metavar="N",
                   help="Cap the failure report at N entries. Every failure is still in "
                        "the JSON.")
    p.add_argument("--compare", action="store_true",
                   help="Print the leaderboard table across the systems run.")
    p.add_argument("--repeat-check", action="store_true",
                   help="Run each system twice and assert the accuracy figures and every "
                        "per-question verdict are identical.")
    p.add_argument("--quiet", action="store_true", help="Suppress the per-system table.")
    return p


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _output_path(base: str, system: str, many: bool) -> Path:
    path = Path(base)
    if not many:
        return path
    return path.with_name(f"{path.stem}-{system}{path.suffix or '.json'}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    systems = args.system or ["memvara"]
    limit = QUICK if args.quick else args.limit

    full = load(args.dataset)
    dataset = full.filter(categories=args.category, limit=limit)
    if not dataset.questions:
        print("No questions selected.", file=sys.stderr)
        return 2
    truth = Truth(dataset)

    config = {
        "match": args.match,
        "categories": args.category or "all",
        "limit": limit,
        "questions_asked": len(dataset.questions),
    }

    results: list[RunResult] = []
    for name in systems:
        system = build(name)
        try:
            result = run(system, dataset, lenient=args.match == "lenient",
                         config=config, timestamp=_timestamp())
        finally:
            system.close()
        results.append(result)

        if args.repeat_check:
            again_system = build(name)
            try:
                again = run(again_system, dataset, lenient=args.match == "lenient",
                            config=config, timestamp=result.timestamp)
            finally:
                again_system.close()
            if not _identical(result, again):
                print(f"NONDETERMINISM: {name} produced different answers on two "
                      f"identical runs.", file=sys.stderr)
                return 1
            print(f"repeat-check: {name} produced identical answers twice "
                  f"({len(dataset.questions)} questions).")

        if not args.quiet:
            print(scorecard(result, dataset))
        if args.show_failures:
            print(failure_report(failures(result, dataset, args.max_failures),
                                 dataset, truth))
        if args.output:
            path = _output_path(args.output, name, len(systems) > 1)
            result.write(path)
            print(f"wrote {path}")

    if args.compare and len(results) > 1:
        print(leaderboard(results))
    return 0


def _identical(first: RunResult, second: RunResult) -> bool:
    from .results import comparable

    return comparable(first.to_json()) == comparable(second.to_json())


if __name__ == "__main__":                       # pragma: no cover
    raise SystemExit(main())
