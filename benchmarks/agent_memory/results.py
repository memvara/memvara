"""The result file: what a run produces, and what a reader needs to trust it.

One JSON object per run. It carries the scores, the timings, the cost counters, and
enough of the environment to say what produced them — benchmark version, dataset version,
system version, interpreter, platform, git commit, configuration. A score without that
is a number somebody has to take on faith, which is the opposite of the point.

## The schema is the contract for a future leaderboard

`docs/benchmarks/agent-memory-benchmark.md` documents these fields as the interface a
benchmark page would read. Adding a field is safe; renaming or repurposing one is not,
and is what `BENCHMARK_VERSION` is for.

## Nothing secret goes in

No environment variables, no file paths outside the repository, no API keys, no hostname,
no username. The environment block is a fixed list of fields assembled by name — never a
sweep of `os.environ` — because a benchmark result is written to be published, and a
result file that harvested its environment would be a credential leak with a nice table
on top.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import BENCHMARK_VERSION
from .adapters.base import Usage
from .scoring import Judgement, Scorecard


def git_commit() -> str | None:
    """The commit this ran from, or `None` outside a checkout.

    A published number should say which tree produced it. `None` is honest and is not an
    error: the benchmark runs perfectly well from an unpacked sdist.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5, cwd=Path(__file__).resolve().parent)
    except (OSError, subprocess.SubprocessError):       # pragma: no cover - defensive
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def environment() -> dict[str, Any]:
    """The fields that make a run reproducible. Assembled by name, never swept."""
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "machine": platform.machine(),
        # How many cores the timings had. Not secret, and a latency table read without it
        # cannot be compared against a re-run on different hardware.
        "cpu_count": os.cpu_count(),
        "git_commit": git_commit(),
        "numpy": _version("numpy"),
        "memvara": _version("memvara"),
    }


def _version(package: str) -> str | None:
    try:
        module = __import__(package)
    except ImportError:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version else None


@dataclass(frozen=True, slots=True)
class Latency:
    """Timings, in milliseconds, separated so a slow write cannot hide behind a fast read.

    `write_total_ms` is the whole ingestion, which is the cold path: an empty store
    filling up. It is measured once, because repeating it means emptying the store again
    and the second fill is no longer the cold path.

    `query_*` are per-question and are the warm path. With `--latency-repeats N` the
    question set is asked N times; the first pass is discarded and the rest are
    aggregated, which is what makes "warm" true rather than merely claimed — `vector-rag`
    builds its index on first search, and that construction used to land in
    `query_max_ms` and be published as a tail.

    `p50_spread_ms` is the distance between the highest and lowest per-pass p50. It is
    the number that says whether to believe the others. A published latency measured once
    on a laptop with a browser open cannot distinguish a system from the machine's mood,
    and the spread is the only thing in this block that shows the difference.
    """

    write_total_ms: float
    write_mean_ms: float
    query_mean_ms: float
    query_p50_ms: float
    query_p95_ms: float
    query_max_ms: float
    #: Timed query passes aggregated into the figures above. 1 means a single pass, which
    #: includes whatever the system does lazily on its first read.
    repeats: int = 1
    #: max(per-pass p50) - min(per-pass p50). 0.0 when a single pass was aggregated,
    #: where it means *not measured* rather than *perfectly stable*.
    p50_spread_ms: float = 0.0

    def to_json(self) -> dict[str, float]:
        return {
            "write_total_ms": round(self.write_total_ms, 3),
            "write_mean_ms": round(self.write_mean_ms, 4),
            "query_mean_ms": round(self.query_mean_ms, 4),
            "query_p50_ms": round(self.query_p50_ms, 4),
            "query_p95_ms": round(self.query_p95_ms, 4),
            "query_max_ms": round(self.query_max_ms, 4),
            "repeats": self.repeats,
            "p50_spread_ms": round(self.p50_spread_ms, 4),
        }


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Nearest-rank, in milliseconds. Deterministic, and does not interpolate between two
    measurements to produce a number that was never measured."""
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index] * 1000.0


def latency_of(write_total_s: float, writes: int,
               query_times_s: Sequence[Sequence[float]] | Sequence[float]) -> Latency:
    """Aggregate one ingestion and one or more timed query passes.

    `query_times_s` is a sequence of passes, each a sequence of per-question seconds. A
    flat sequence of floats is accepted as the single-pass case, because that is what
    every caller passed before repeats existed and the shape is unambiguous.
    """
    passes = _passes(query_times_s)
    flat = sorted(t for one in passes for t in one)
    n = len(flat)
    per_pass_p50 = [_percentile(sorted(one), 0.50) for one in passes if one]

    return Latency(
        write_total_ms=write_total_s * 1000.0,
        write_mean_ms=(write_total_s / writes * 1000.0) if writes else 0.0,
        query_mean_ms=(sum(flat) / n * 1000.0) if n else 0.0,
        query_p50_ms=_percentile(flat, 0.50),
        query_p95_ms=_percentile(flat, 0.95),
        query_max_ms=(flat[-1] * 1000.0) if flat else 0.0,
        repeats=len(passes),
        p50_spread_ms=(max(per_pass_p50) - min(per_pass_p50)) if per_pass_p50 else 0.0,
    )


def _passes(times: Sequence[Sequence[float]] | Sequence[float]) -> list[list[float]]:
    if times and isinstance(times[0], (int, float)):
        return [list(times)]                            # type: ignore[arg-type]
    return [list(one) for one in times]                 # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RunResult:
    """One system, one dataset, one configuration, one run."""

    system: str
    system_version: str
    dataset_version: str
    #: UTC ISO-8601. Supplied by the runner rather than read here, so a caller that wants
    #: a byte-identical result file for a reproducibility check can pin it.
    timestamp: str
    scorecard: Scorecard
    latency: Latency
    usage: Usage
    judgements: tuple[Judgement, ...]
    config: Mapping[str, Any] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)

    def to_json(self, *, include_judgements: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "benchmark": "agent-memory",
            "benchmark_version": BENCHMARK_VERSION,
            "dataset_version": self.dataset_version,
            "system": self.system,
            "system_version": self.system_version,
            "timestamp": self.timestamp,
            "counts": dict(self.counts),
            "config": dict(self.config),
            "environment": environment(),
            "metrics": self.scorecard.to_json(),
            "latency": self.latency.to_json(),
            "usage": self.usage.to_json(),
        }
        if include_judgements:
            payload["questions"] = [j.to_json() for j in self.judgements]
        return payload

    def write(self, path: str | Path, *, include_judgements: bool = True) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # `newline="\n"` for the reason `datasets/build_v1.py::_write` gives: without it
        # Python translates to `os.linesep` on write, and this file is the interchange
        # format a leaderboard reads and two runs get diffed in. Nothing compares its
        # bytes today, which is precisely why it would have gone unnoticed.
        target.write_text(
            json.dumps(self.to_json(include_judgements=include_judgements), indent=2) + "\n",
            encoding="utf-8", newline="\n")


def comparable(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a result that two runs of the same configuration must agree on.

    Three things are dropped, for two different reasons.

    Timings and the timestamp are measurements of the machine rather than of the system,
    and asserting on them would make the check fail on a busy laptop — which would teach
    everyone to ignore it.

    `support` is dropped because it holds whatever the system calls its rows, and some
    systems mint those per process: memvara's claim ids are `uuid4`, so they differ on
    every run of identical input while every answer stays the same. Comparing them would
    report nondeterminism that does not exist. This does narrow the check — a system whose
    retrieval wanders while its answers hold still would pass — and that is the right
    boundary, because what this benchmark scores is answers.

    What is left is every accuracy figure and every per-question verdict, and those must
    match exactly.
    """
    dropped = {"latency_s", "support"}
    return {
        "metrics": payload["metrics"],
        "questions": [
            {k: v for k, v in row.items() if k not in dropped}
            for row in payload.get("questions", [])
        ],
    }
