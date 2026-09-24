"""Fail when LOCOMO retrieval moves away from the figures committed beside this script.

Run:  PYTHONPATH=. python3 bench/retrieval_regression.py [--update]

Needs the LOCOMO file, 2.8 MB (`python3 bench/locomo.py --download`).

`docs/BENCHMARKS.md` records a regression that three commits carried unnoticed: a change to
the graph leg's gate cost 1.6 points of LongMemEval single-session-user R@12, and nothing
failed, because no test asserted a benchmark figure. This script is that test, for the one
public benchmark cheap enough to run on every push. It runs `bench/locomo.py --score
retrieval` with no flags, which is the published configuration: all ten conversations, the
hashing embedder, no reranker and no extraction model, in about half a minute. Then it
compares each category's figures, and the overall ones, with
`bench/expected/locomo_retrieval.json`:

* `in ctx`, the share of gold answers whose words reach the context `recall()` returns,
  which is what a change to how `recall()` renders its lines moves;
* evidence recall at each cut-off, and evidence MRR, which are what a change to ranking
  moves.

It fails when any figure moved by more than `TOLERANCE`, in either direction. An
improvement fails too, because the figures in `docs/BENCHMARKS.md` would then be wrong: a
change meant to move them runs this with `--update` and commits the new file beside the
documentation that quotes them.

The run is deterministic on one machine, and three runs are byte-identical (see the report
`bench/locomo.py` prints). Across machines the cosines can differ in their last bits,
because numpy's BLAS picks its kernel for the CPU it finds, and a last-bit difference can
reorder two turns whose scores tie to seven digits. `TOLERANCE` is sized to absorb one such
question.

The dataset file is checked against the digest recorded in the expected file first, so that
a change upstream fails as a changed dataset rather than as a retrieval regression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evalkit as ek  # noqa: E402
import locomo  # noqa: E402

EXPECTED = Path(__file__).resolve().parent / "expected" / "locomo_retrieval.json"

#: How far a figure may move, in percentage points, before the gate fails. One question
#: moves an overall figure by at most 0.07 points, so the overall bound admits one question
#: and not two. One question in the smallest category, open-domain with 92 scored
#: questions, moves its figures by at most 1.09 points, so the category bound admits one
#: there, and proportionally more in a larger category, where the overall bound is the one
#: that catches a change.
TOLERANCE = {"all": 0.1, "category": 1.1}


def measure(samples: Sequence[locomo.Sample]) -> dict[str, dict[str, float]]:
    """The figures `bench/locomo.py --score retrieval` reports with no flags, by group."""
    parser = argparse.ArgumentParser()
    ek.add_common_arguments(parser)
    args = parser.parse_args([])
    budget = ek.RetrievalBudget(k=args.k, max_chars=args.max_chars,
                                include_episodes=not args.no_episodes)
    plan = ek.build_plan(args)
    scores, _, _, _ = locomo.run_retrieval(
        samples, budget=budget, plan=plan, embedder=ek.build_embedder(args.embedder),
        w_graph=args.w_graph, w_temporal=args.w_temporal)
    return figures(scores, plan.ks)


def figures(scores: Sequence[ek.RetrievalScore],
            ks: Sequence[int]) -> dict[str, dict[str, float]]:
    """Per category and overall: the question counts, `in ctx`, evidence R@k and MRR, in
    percent, as `evalkit.retrieval_tables` computes them, to two decimals."""
    groups = sorted({s.category for s in scores})
    out: dict[str, dict[str, float]] = {}
    for name in (*groups, "all"):
        items = [s for s in scores if name == "all" or s.category == name]
        answered = [s for s in items if s.answer_in_context is not None]
        evidenced = [s for s in items if s.evidence_recall_at is not None]
        row: dict[str, float] = {"n answered": len(answered),
                                 "n evidenced": len(evidenced)}
        if answered:
            row["in ctx"] = _pct([float(bool(s.answer_in_context)) for s in answered])
        if evidenced:
            for k in ks:
                row[f"R@{k}"] = _pct([s.evidence_recall_at[k] for s in evidenced
                                      if s.evidence_recall_at is not None])
            row["MRR"] = _pct([s.evidence_mrr for s in evidenced
                               if s.evidence_mrr is not None])
        out[name] = row
    return out


def _pct(values: Sequence[float]) -> float:
    return round(100 * mean(values), 2)


def compare(want: dict[str, dict[str, float]],
            got: dict[str, dict[str, float]]) -> list[str]:
    """One line per figure that moved beyond `TOLERANCE`, or that one side lacks."""
    failures = []
    for group in sorted(set(want) | set(got)):
        a, b = want.get(group, {}), got.get(group, {})
        bound = TOLERANCE["all" if group == "all" else "category"]
        for metric in sorted(set(a) | set(b)):
            if metric not in a or metric not in b:
                failures.append(f"{group} {metric}: expected {a.get(metric, 'nothing')}, "
                                f"measured {b.get(metric, 'nothing')}")
            elif metric.startswith("n "):
                if a[metric] != b[metric]:
                    failures.append(f"{group} {metric}: expected {a[metric]:g}, "
                                    f"measured {b[metric]:g}")
            # Rounded as the figures are, so that 30.1 against 30.0 is the 0.1 it reads
            # as and not the 0.10000000000000142 a float subtraction makes of it.
            elif round(abs(b[metric] - a[metric]), 2) > bound:
                failures.append(f"{group} {metric}: expected {a[metric]:.2f}, measured "
                                f"{b[metric]:.2f} ({b[metric] - a[metric]:+.2f}, bound "
                                f"{bound})")
    return failures


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true",
                        help=f"write the measured figures to {EXPECTED.name} and exit")
    parser.add_argument("--cache", default=None,
                        help="dataset cache directory (default $MEMVARA_BENCH_DATA or "
                             "~/.cache/memvara-bench)")
    args = parser.parse_args(argv)
    path = ek.require(ek.LOCOMO10, args.cache)
    if args.update:
        got = measure(locomo.load(path))
        EXPECTED.parent.mkdir(exist_ok=True)
        EXPECTED.write_text(json.dumps({"dataset sha256": digest(path), "figures": got},
                                       indent=1, sort_keys=True) + "\n")
        print(f"  wrote {EXPECTED}")
        return 0
    expected: dict[str, Any] = json.loads(EXPECTED.read_text())
    if expected["dataset sha256"] != digest(path):
        print(f"  {path} is not the file {EXPECTED.name} was measured on: its sha256 is "
              f"{digest(path)}, not {expected['dataset sha256']}. The dataset changed "
              "upstream; measure it with --update and review what moved.")
        return 1
    got = measure(locomo.load(path))
    failures = compare(expected["figures"], got)
    for group, row in got.items():
        print(f"  {group:<12} " + "  ".join(f"{m} {v:g}" for m, v in row.items()))
    if failures:
        print("\n  Retrieval moved away from the committed figures:\n")
        print("\n".join(f"    {line}" for line in failures))
        print(f"\n  If the change is meant to move them, run with --update, commit "
              f"{EXPECTED.relative_to(EXPECTED.parents[2])}, and update the figures "
              "docs/BENCHMARKS.md quotes in the same commit.")
        return 1
    print("\n  LOCOMO retrieval matches the committed figures.")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by hand and in CI
    try:
        sys.exit(main())
    except ek.DatasetMissing as missing:
        print(f"\n{missing}")
        sys.exit(1)
