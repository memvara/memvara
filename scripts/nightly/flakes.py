"""Flakes: rerun a failed test twice, and measure the flake rate of each layer.

A test that failed in the night is run twice more, and the majority of the three runs
decides what the failure was:

* both reruns pass: a **flake**. The majority passed, so nothing is filed.
* both reruns fail: **confirmed**. Only a confirmed failure may be filed, because its
  strict expected failure will fail every time too.
* one passes and one fails: **intermittent**. The majority failed, so it counts as a
  failure, but it is not filed: a strict expected failure on a test that sometimes
  passes would make the suite itself flaky. A person looks at it.
* a rerun could not run the test, or the test was never rerun: **unconfirmed**.

A test that passed after failing is flaky, whatever its verdict. The flake rate of a
layer is its flaky tests over the tests it ran, across the last fourteen nights, and the
design's budget is 0.5% per layer.

By hand:

    python3 scripts/nightly/flakes.py rerun --worktree W --python P NODEID...
    python3 scripts/nightly/flakes.py rates [--checkout C] [--window 14]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

if __package__ in (None, ""):  # run as a script, not imported as nightly.flakes
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nightly import night, steps  # noqa: E402 - needs the path set just above

from harness.tiers import TIER_DIRS  # noqa: E402 - the nightly package puts tests/ on the path

#: How many times a failed test is run again.
RERUNS = 2
#: The nights a flake rate is measured over.
WINDOW = 14
#: The design's flake budget: at most 0.5% of a layer's tests may be flaky.
BUDGET = 0.005

PASSED, FAILED, ERROR = "passed", "failed", "error"


def layer_of(nodeid: str) -> str:
    """The layer a test belongs to, from its node id.

    Under tests/adversarial it is the first folder that is not a tier folder, such as
    `model` or `concurrency`, so a nightly concurrency test counts as concurrency; a file
    with no such folder is `adversarial`. Elsewhere it is `live` for tests/live, `unit`
    for the rest of tests/, `doctest` for the package and `other` for anything else.
    """
    parts = nodeid.split("::", 1)[0].replace("\\", "/").split("/")
    if parts[:2] == ["tests", "adversarial"]:
        return next((part for part in parts[2:-1] if part not in TIER_DIRS), "adversarial")
    if parts[:2] == ["tests", "live"]:
        return "live"
    if parts[:1] == ["tests"]:
        return "unit"
    if parts[:1] == ["memvara"]:
        return "doctest"
    return "other"


def outcome_of(returncode: int | None) -> str:
    """What one pytest run of one test says about it. pytest exits 0 when the test passed
    and 1 when it failed. Any other code means it was interrupted, misused or collected
    nothing, and None means the run was stopped at its cap: none of those say anything
    about the test."""
    return {0: PASSED, 1: FAILED}.get(returncode, ERROR) if returncode is not None else ERROR


def verdict(results: Sequence[str]) -> str:
    """What the reruns of a test that failed say the failure was: see the module's
    docstring."""
    if not results or ERROR in results:
        return "unconfirmed"
    if all(result == PASSED for result in results):
        return "flake"
    if all(result == FAILED for result in results):
        return "confirmed"
    return "intermittent"


def flaky(results: Sequence[str]) -> bool:
    """Whether a test that failed passed at least once when it was run again."""
    return PASSED in results


@dataclass(frozen=True)
class Rerun:
    nodeid: str
    results: tuple[str, ...]

    @property
    def verdict(self) -> str:
        return verdict(self.results)

    @property
    def flaky(self) -> bool:
        return flaky(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {"nodeid": self.nodeid, "results": list(self.results),
                "verdict": self.verdict, "flaky": self.flaky}


def rerun_command(python: str, nodeid: str, tier: str = "nightly") -> list[str]:
    """The command that runs one test again, in the tier the night ran."""
    return [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tier", tier, nodeid]


def rerun(nodeid: str, run: Callable[[list[str]], int | None], *, python: str,
          tier: str = "nightly", times: int = RERUNS) -> Rerun:
    """Run one test `times` more times. `run` runs a command and returns its exit code,
    or None when it was stopped at its cap."""
    command = rerun_command(python, nodeid, tier)
    return Rerun(nodeid, tuple(outcome_of(run(command)) for _ in range(times)))


@dataclass(frozen=True)
class Rate:
    """A layer's flakes over the nights measured."""

    run: int
    flaky: int
    nights: int

    @property
    def rate(self) -> float:
        return self.flaky / self.run if self.run else 0.0

    @property
    def over_budget(self) -> bool:
        return self.rate > BUDGET

    def to_dict(self) -> dict[str, Any]:
        return {"run": self.run, "flaky": self.flaky, "nights": self.nights,
                "rate": self.rate, "over_budget": self.over_budget}


def rates(records: Iterable[Mapping[str, Any]], window: int = WINDOW) -> dict[str, Rate]:
    """The flake rate of each layer over the last `window` nights of the history.

    Only night records count, one per date: a second run of the same night replaces the
    first. A night with no test results, because it failed before its tests ran, still
    takes its place in the window and adds nothing to it.
    """
    by_date: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if record.get("kind") == "night" and "date" in record:
            by_date[str(record["date"])] = record
    totals: dict[str, list[int]] = {}
    for date in sorted(by_date)[-window:]:
        for layer, counts in (by_date[date].get("layers") or {}).items():
            total = totals.setdefault(layer, [0, 0, 0])
            total[0] += int(counts.get("run", 0))
            total[1] += int(counts.get("flaky", 0))
            total[2] += 1
    return {layer: Rate(*total) for layer, total in sorted(totals.items())}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="flakes.py", description="Rerun failed tests, or show the flake rate of "
        "each layer over the last nights.")
    sub = command.add_subparsers(dest="command", required=True)
    again = sub.add_parser("rerun", help="run tests twice more and say what each was")
    again.add_argument("--worktree", required=True, help="the checkout the tests ran in")
    again.add_argument("--python", required=True, help="the interpreter the night used")
    again.add_argument("--tier", default="nightly", help="the tier the night ran")
    again.add_argument("--home", help="HOME for the reruns; the nightly home by default")
    again.add_argument("--cap", type=float, default=600.0,
                       help="seconds each rerun may take")
    again.add_argument("nodeids", nargs="+", metavar="NODEID")
    table = sub.add_parser("rates", help="the flake rate of each layer")
    table.add_argument("--checkout", help="the main checkout; the one this file is in by "
                       "default")
    table.add_argument("--window", type=int, default=WINDOW, help="how many nights")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "rerun":
        return _rerun(args)
    return _rates(args)


def _rerun(args: argparse.Namespace) -> int:
    worktree = pathlib.Path(args.worktree).resolve()
    home = (pathlib.Path(args.home) if args.home
            else night.Layout(night.main_checkout(worktree)).home)
    home.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="memvara-rerun-"))
    try:
        env = night.step_env(os.environ, worktree=worktree, home=home, tmp=tmp)
        log = worktree / "local" / "flake-reruns.log"

        def run(command: list[str]) -> int | None:
            return steps.run_command(command, cwd=worktree, env=env, log=log,
                                     deadline=time.monotonic() + args.cap).returncode

        for nodeid in args.nodeids:
            result = rerun(nodeid, run, python=args.python, tier=args.tier)
            print(json.dumps(result.to_dict()), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def _rates(args: argparse.Namespace) -> int:
    checkout = (pathlib.Path(args.checkout) if args.checkout
                else night.main_checkout(pathlib.Path(__file__).parent))
    records, _ = night.read_jsonl(night.Layout(checkout).history)
    print(f"{'layer':<16}{'nights':>7}{'tests run':>11}{'flaky':>7}{'rate':>9}  "
          f"budget {BUDGET:.2%}")
    for layer, rate in rates(records, args.window).items():
        print(f"{layer:<16}{rate.nights:>7}{rate.run:>11}{rate.flaky:>7}{rate.rate:>9.2%}  "
              f"{'over' if rate.over_budget else 'within'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
