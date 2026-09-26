"""How long the reads and writes an agent depends on take, and the rule that judges them.

The design of the adversarial suite (`docs/superpowers/specs/2026-09-25-adversarial-test-
suite-design.md`, "Phase 4") fixes the budget decision rule before any number is
measured, so that a budget cannot be fitted to whatever the first night happened to show.
This module holds that rule as plain functions:

* **Hard ceilings from the hook contract.** The recall hook's p95 may not pass 7.5 s at
  any store size, cold or warm, and no recall may take longer than 10 s. Session start's
  p95 may not pass 20 s. These apply from the first run.
* **Library budgets.** After 14 valid nights, each budget is 1.5 times the median p95 of
  those nights, rounded up to the next step of 1, 2, 5, 10, 20, 50 and so on. They are
  committed once, together with the fingerprint of the machine that measured them.
* **A regression** needs three things at once: the p95 is more than 1.20 times the median
  of the last 7 valid nights; the increase is more than 2 ms and more than 3 times the
  median absolute deviation of those nights; and measuring the series again at once shows
  the same thing.

Percentiles are taken by nearest rank, with `evalkit.percentile`, so this script reports
a p95 the same way every other benchmark here does.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sqlite3
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evalkit  # noqa: E402

import memvara  # noqa: E402

#: The checkout this script belongs to. It is bench/perf_budget.py, one level below it.
REPO = Path(__file__).resolve().parents[1]

#: The percentiles reported for every series, by the name each is stored under.
PERCENTILES: dict[str, float] = {"p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99}

#: A regression's increase must be larger than this, whatever the spread of the history.
LATENCY_FLOOR_MS = 2.0
#: How many earlier valid nights the rolling median is taken over.
ROLLING_NIGHTS = 7
#: A p95 must exceed the rolling median by this factor before it can be a regression.
REGRESSION_RATIO = 1.20
#: The increase must also exceed this many median absolute deviations of the history.
MAD_MULTIPLIER = 3.0
#: Library budgets are derived only once this many valid nights exist.
BUDGET_NIGHTS = 14
#: A budget is this multiple of the median p95, before rounding up to a 1-2-5 step.
BUDGET_MULTIPLIER = 1.5


def series_key(operation: str, size: int, temperature: str) -> str:
    """The name a series is stored under, such as `hook.recall@1000/cold`."""
    return f"{operation}@{size}/{temperature}"


# --- percentiles and the bootstrap interval ---------------------------------------------


def _resampled_percentiles(samples: Sequence[float], quantiles: Sequence[float], *,
                           resamples: int, seed: int) -> dict[float, list[float]]:
    """Each quantile of `resamples` samples drawn with replacement from `samples`.

    One set of resamples serves every quantile, so the intervals of one series are
    computed from the same draws.
    """
    rng = random.Random(seed)
    estimates: dict[float, list[float]] = {q: [] for q in quantiles}
    for _ in range(resamples):
        draw = rng.choices(samples, k=len(samples))
        for q in quantiles:
            estimates[q].append(float(evalkit.percentile(draw, q)))
    return estimates


def _interval(estimates: list[float], confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    return (float(evalkit.percentile(estimates, tail)),
            float(evalkit.percentile(estimates, 1.0 - tail)))


def bootstrap_interval(samples: Sequence[float], q: float, *, resamples: int = 1000,
                       confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """A percentile bootstrap interval for the `q` quantile of `samples`.

    The same samples and seed give the same interval, so a report can be reproduced.
    """
    if not samples:
        raise ValueError("no samples to take an interval of")
    estimates = _resampled_percentiles(samples, [q], resamples=resamples, seed=seed)
    return _interval(estimates[q], confidence)


def summarize(samples: Sequence[float], *, resamples: int = 1000,
              seed: int = 0) -> dict[str, Any]:
    """p50, p90, p95, p99 and the maximum of `samples`, each percentile with a 95% interval.

    An empty series is refused. Reporting it as zero would make a series that measured
    nothing look like the fastest one in the run.
    """
    if not samples:
        raise ValueError("no samples to summarize: a series that measured nothing has no "
                         "percentiles")
    values = [float(v) for v in samples]
    estimates = _resampled_percentiles(values, list(PERCENTILES.values()),
                                       resamples=resamples, seed=seed)
    summary: dict[str, Any] = {"n": len(values)}
    for name, q in PERCENTILES.items():
        summary[name] = float(evalkit.percentile(values, q))
    summary["max"] = max(values)
    summary["interval"] = {name: list(_interval(estimates[q], 0.95))
                           for name, q in PERCENTILES.items()}
    return summary


# --- rounding and spread -----------------------------------------------------------------


def round_up_125(value: float) -> float:
    """The smallest of 1, 2 and 5 times a power of ten that is at least `value`.

    A value within one part in a billion of a step counts as that step, so floating-point
    noise such as 2.0000000000000004 does not push a budget to the next one.
    """
    if value <= 0:
        raise ValueError(f"a budget must be positive, not {value}")
    exponent = math.floor(math.log10(value))
    for decade in (exponent - 1, exponent, exponent + 1):
        for step in (1.0, 2.0, 5.0):
            candidate = step * 10.0 ** decade
            if candidate >= value * (1 - 1e-9):
                return candidate
    raise AssertionError("unreachable: 10 ** (exponent + 1) is above value")


def mad(values: Sequence[float]) -> float:
    """The median absolute deviation from the median, unscaled."""
    centre = statistics.median(values)
    return statistics.median(abs(v - centre) for v in values)


# --- the regression rule ------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """What the regression rule decided about one series.

    `outcome` is `ok`, `regression`, `not reproduced` (the first two conditions held and
    the re-measure did not repeat them) or `no history` (fewer than seven earlier valid
    nights, so the rule cannot be applied yet).
    """

    outcome: str
    current: float
    median: float | None
    mad: float | None
    remeasured: float | None
    detail: str


def is_regression(current: float, history: Sequence[float], *, floor: float) -> bool:
    """The first two conditions: above 1.20 times the median, by more than the noise."""
    centre = statistics.median(history)
    return (current > REGRESSION_RATIO * centre
            and current - centre > max(floor, MAD_MULTIPLIER * mad(history)))


def judge(current: float, history: Sequence[float], remeasure: Callable[[], float], *,
          floor: float) -> Verdict:
    """Apply the regression rule to one series.

    `history` holds the series' value on earlier valid nights, oldest first, and only the
    last seven count. `remeasure` measures the series again and is called only when the
    first two conditions hold, because a re-measure costs as much as the measurement.
    """
    recent = [float(v) for v in history][-ROLLING_NIGHTS:]
    if len(recent) < ROLLING_NIGHTS:
        return Verdict("no history", current, None, None, None,
                       f"{len(recent)} of the {ROLLING_NIGHTS} earlier nights the rule "
                       "needs")
    centre, spread = statistics.median(recent), mad(recent)
    if not is_regression(current, recent, floor=floor):
        return Verdict("ok", current, centre, spread, None,
                       f"{current:.3f} against a median of {centre:.3f}")
    again = float(remeasure())
    if is_regression(again, recent, floor=floor):
        return Verdict("regression", current, centre, spread, again,
                       f"{current:.3f}, and {again:.3f} measured again, against a median "
                       f"of {centre:.3f} and a deviation of {spread:.3f}")
    return Verdict("not reproduced", current, centre, spread, again,
                   f"{current:.3f} against a median of {centre:.3f}, but {again:.3f} "
                   "when measured again")


# --- library budgets ----------------------------------------------------------------------


def derive_budgets(nights: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Each series' budget: 1.5 times its median p95 over `nights`, rounded up to 1-2-5.

    `nights` maps each series to its p95 on one valid night. A series missing from any of
    them gets no budget, because a median over fewer nights is not the rule. Fewer than 14
    nights is refused.
    """
    if len(nights) < BUDGET_NIGHTS:
        raise ValueError(f"library budgets need {BUDGET_NIGHTS} valid nights, and there "
                         f"are {len(nights)}")
    everywhere = set.intersection(*(set(night) for night in nights))
    return {name: round_up_125(BUDGET_MULTIPLIER
                               * statistics.median(night[name] for night in nights))
            for name in sorted(everywhere)}


# --- the hard ceilings --------------------------------------------------------------------


@dataclass(frozen=True)
class Ceiling:
    """A limit from the hook contract on one statistic of one hook's series."""

    operation: str
    stat: str
    limit_ms: float


#: The hook contract's limits. They apply at every store size, cold and warm.
CEILINGS: tuple[Ceiling, ...] = (
    Ceiling("hook.recall", "p95", 7_500.0),
    Ceiling("hook.recall", "max", 10_000.0),
    Ceiling("hook.session_start", "p95", 20_000.0),
)


def check_ceilings(series: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One entry per ceiling that applies to a series, saying whether it was breached.

    A series that recorded a timeout breaches its maximum whatever its samples say,
    because a timed-out run is recorded at the hook's limit and would otherwise pass.
    """
    entries: list[dict[str, Any]] = []
    for key in sorted(series):
        operation = key.split("@", 1)[0]
        stats = series[key]
        for ceiling in CEILINGS:
            if ceiling.operation != operation:
                continue
            value = float(stats[ceiling.stat])
            breached = value > ceiling.limit_ms or (
                ceiling.stat == "max" and int(stats.get("timeouts", 0)) > 0)
            entries.append({"series": key, "stat": ceiling.stat,
                            "limit_ms": ceiling.limit_ms, "value_ms": value,
                            "breached": breached})
    return entries


# --- the machine, and whether a run is valid ------------------------------------------------


#: A run is under load when the one-minute load average, divided by the number of logical
#: CPUs, is above this at any check. At 0.5, half the machine is busy with something other
#: than the measurement, which is enough to move a p95 by more than the rule's 2 ms floor.
MAX_LOAD_PER_CPU = 0.5


def _tool(*command: str) -> str | None:
    """The output of a system tool that describes the machine, or None if it failed.

    These tools (`sysctl`, `pmset`, `git`) run with a minimal environment rather than
    `harness.env.child_env`: memvara never runs in them, and the fixed locale keeps
    `pmset`'s wording the English the parser below reads.
    """
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"}
    try:
        done = subprocess.run(list(command), capture_output=True, text=True, timeout=10,
                              env=env, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _cpu_model() -> str:
    if platform.system() == "Darwin":
        return _tool("sysctl", "-n", "machdep.cpu.brand_string") or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _memory_bytes() -> int | None:
    if platform.system() == "Darwin":
        text = _tool("sysctl", "-n", "hw.memsize")
        return int(text) if text and text.isdigit() else None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return None


def machine_fingerprint() -> dict[str, Any]:
    """The machine a run measured, and the software it measured with.

    `id` is a short hash of the hardware alone: the system, the processor, the number of
    logical CPUs and the memory. Budgets are keyed on it, so an operating-system or Python
    update keeps a machine's budgets while a different laptop does not inherit them. The
    software versions and the commit are recorded beside it, so a jump in the numbers can
    be traced to what changed.
    """
    hardware = {"system": platform.system(), "machine": platform.machine(),
                "cpu": _cpu_model(), "logical_cpus": os.cpu_count(),
                "memory_bytes": _memory_bytes()}
    ident = hashlib.sha256(json.dumps(hardware, sort_keys=True).encode()).hexdigest()[:16]
    return {**hardware, "release": platform.release(), "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version, "memvara": memvara.__version__,
            "commit": _tool("git", "-C", str(REPO), "rev-parse", "HEAD"), "id": ident}


@dataclass(frozen=True)
class Conditions:
    """The state of the machine at one check during a run.

    `on_battery` is None where the platform cannot say, which counts as not on battery:
    the nightly run is on a Mac, where it can always say.
    """

    on_battery: bool | None
    load_per_cpu: float


def on_battery_from_pmset(text: str) -> bool | None:
    """Read `pmset -g batt`'s first line: which power source the Mac is drawing from."""
    if "'Battery Power'" in text:
        return True
    if "'AC Power'" in text:
        return False
    return None


def on_battery_from_sysfs(root: Path) -> bool | None:
    """Read Linux's power supplies: on battery when no mains supply is online and a
    battery is present. None when there is no power supply to read, as in a container."""
    if not root.is_dir():
        return None
    mains_online = battery = False
    for supply in root.iterdir():
        kind_file = supply / "type"
        kind = kind_file.read_text(encoding="utf-8").strip() if kind_file.is_file() else ""
        if kind == "Mains":
            online = supply / "online"
            if online.is_file() and online.read_text(encoding="utf-8").strip() == "1":
                mains_online = True
        elif kind == "Battery":
            battery = True
    if mains_online:
        return False
    return True if battery else None


def read_conditions() -> Conditions:
    """The power source and the load of this machine now."""
    if platform.system() == "Darwin":
        text = _tool("pmset", "-g", "batt")
        on_battery = on_battery_from_pmset(text) if text is not None else None
    elif platform.system() == "Linux":
        on_battery = on_battery_from_sysfs(Path("/sys/class/power_supply"))
    else:
        on_battery = None
    try:
        load = os.getloadavg()[0]
    except (AttributeError, OSError):
        # Windows has no load average. The nightly tiers run only on a Mac.
        load = 0.0
    return Conditions(on_battery=on_battery,
                      load_per_cpu=float(load) / (os.cpu_count() or 1))


def invalid_reasons(checks: Sequence[Conditions]) -> list[str]:
    """Why a run with these checks is invalid, in words; empty when it is valid."""
    reasons: list[str] = []
    on_battery = sum(1 for c in checks if c.on_battery)
    if on_battery:
        reasons.append(f"the machine ran on battery at {on_battery} of {len(checks)} checks")
    busiest = max((c.load_per_cpu for c in checks), default=0.0)
    if busiest > MAX_LOAD_PER_CPU:
        reasons.append(f"the load average reached {busiest:.2f} per CPU, above the limit "
                       f"of {MAX_LOAD_PER_CPU}")
    return reasons
