"""How long the reads and writes an agent depends on take, and the rule that judges them.

Run:  PYTHONPATH=. python3 bench/perf_budget.py [--sizes 1000,10000,100000] [--out FILE]
      [--history DIR]
      PYTHONPATH=. python3 bench/perf_budget.py --write-budgets FILE --history DIR

A run builds a store of each size and times `search`, `recall` and `remember`, each in
fresh child processes, and the session-start and recall hooks through
`harness.hooks.HookRunner`. Cold is the first call in each of 30 new processes, and warm
is 200 calls in one. Each series reports p50, p90, p95 and p99 with a bootstrap interval,
and every record carries the fingerprint of the machine that measured it.

The design of the adversarial suite fixes the budget decision rule before any number is
measured, so that a budget cannot be fitted to whatever the first night happened to show.
The rule is in the section "Phase 4" of
`docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`, and this module holds
it as plain functions:

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

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evalkit  # noqa: E402

import memvara  # noqa: E402
from memvara import Memvara, NullLLM  # noqa: E402
from memvara.select import PLAIN_READ  # noqa: E402

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


# --- measuring ------------------------------------------------------------------------------

#: The user every timing store's claims belong to.
PERF_USER = "perf"
#: The file committed library budgets live in, once 14 valid nights exist.
BUDGETS_PATH = REPO / "bench" / "expected" / "perf_budgets.json"
#: What every timing record says it is, so a folder of mixed records can be read safely.
RECORD_KIND = "memvara-perf"
RECORD_VERSION = 1
#: The exit code of a run marked invalid: reported, and not failed.
EXIT_INVALID = 3
#: Untimed calls a warm series makes before its timed ones.
WARMUP_CALLS = 5
#: What each reading hook's status line says when it read a store. "not configured" and
#: "recall failed" are the two ways it can answer without having read one.
READ_OUTCOMES: dict[str, tuple[str, ...]] = {
    "recall": ("recalled", "already in context", "no matching memories",
               "standing preferences updated"),
    "session_start": ("session opened",),
}
#: The series of one store size, in the order they are measured. The reads come before
#: `remember`, which writes to a copy of the store so that the reads see the built size.
ORDER: tuple[tuple[str, str], ...] = tuple(
    (operation, temperature)
    for operation in ("search", "recall", "hook.session_start", "hook.recall", "remember")
    for temperature in ("cold", "warm"))
#: The syllables the made-up names of a timing store are built from.
SYLLABLES = ("ta", "lo", "vi", "ren", "mar", "zo", "quin", "el", "dri", "kan", "sol",
             "bex", "nu", "or", "phi", "gal", "tor", "wen", "ix", "ul", "vas", "pem",
             "rud", "yo")


class PerfError(RuntimeError):
    """A measurement that did not measure what it names, such as a hook that never read
    the store it was pointed at."""


@dataclass(frozen=True)
class PerfConfig:
    """What one timing run measures: the store sizes, and how many cold processes and
    warm calls each series takes."""

    sizes: tuple[int, ...] = (1_000, 10_000, 100_000)
    cold: int = 30
    warm: int = 200
    resamples: int = 1000
    seed: int = 0


def _harness() -> tuple[ModuleType, ModuleType]:
    """`harness.env` and `harness.hooks`, from this checkout's tests folder.

    Imported when a run needs them rather than at the top, so that importing this module
    (which `bench/soak.py` does) does not depend on the tests folder.
    """
    tests = str(REPO / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    import harness.env  # noqa: PLC0415
    import harness.hooks  # noqa: PLC0415
    return harness.env, harness.hooks


def _store(path: Path) -> Memvara:
    return Memvara(str(path), embedder=evalkit.build_embedder("hashing"), llm=NullLLM(),
                   user=PERF_USER, **PLAIN_READ)


def build_store(path: Path, claims: int, *, seed: int = 0) -> list[str]:
    """Write exactly `claims` live claims to a new store at `path`, and return the names
    of the made-up people they are about.

    Each person has four claims: a city, an employer and two different likes. They are
    written through `Memvara.remember`, as an agent writes them, in transactions of 1,000
    so that building 100,000 claims takes minutes rather than an hour.
    """
    rng = random.Random(seed)
    syllables = list(SYLLABLES)
    rng.shuffle(syllables)

    def word(number: int, length: int) -> str:
        parts = []
        for _ in range(length):
            number, digit = divmod(number, len(syllables))
            parts.append(syllables[digit])
        return "".join(parts)

    cities = [word(n, 3).capitalize() for n in range(40)]
    companies = [word(n + 1_000, 3).capitalize() for n in range(40)]
    items = [word(n + 2_000, 3) for n in range(300)]
    people = [word(n, 4).capitalize() for n in range(math.ceil(claims / 4))]
    facts: list[tuple[str, str, str]] = []
    for person in people:
        liked = rng.sample(items, 2)
        facts += [(person, "lives_in", rng.choice(cities)),
                  (person, "works_at", rng.choice(companies)),
                  (person, "likes", liked[0]), (person, "likes", liked[1])]
    mem = _store(path)
    try:
        for start in range(0, claims, 1_000):
            with mem.store.batch():
                for subject, predicate, obj in facts[start:min(start + 1_000, claims)]:
                    mem.remember(subject, predicate, obj)
    finally:
        mem.close()
    return people


def _child(spec_text: str) -> int:
    """Child mode: time one operation on one store, and print the milliseconds as JSON.

    `spec` names the store, the operation, the untimed `warmup` calls, the timed `calls`,
    the queries, the index of the first query, and an `offset` that keeps every fact a
    `remember` writes new. The store is opened before timing starts, so a sample is the
    call alone.
    """
    spec = json.loads(spec_text)
    mem = _store(Path(spec["db"]))
    queries: list[str] = spec["queries"]

    def call(index: int) -> None:
        query = queries[(spec["first"] + index) % len(queries)]
        if spec["operation"] == "search":
            mem.search(query, k=10, **PLAIN_READ)
        elif spec["operation"] == "recall":
            mem.recall(query, **PLAIN_READ)
        else:
            mem.remember("user", "likes", f"timing item {spec['offset'] + index}")

    for index in range(spec["warmup"]):
        call(index)
    samples = []
    for index in range(spec["warmup"], spec["warmup"] + spec["calls"]):
        began = time.perf_counter()
        call(index)
        samples.append((time.perf_counter() - began) * 1000.0)
    mem.close()
    print(json.dumps({"ms": samples}))
    return 0


def _run_child(spec: Mapping[str, Any], home: Path, *, timeout: float) -> list[float]:
    """Run this script in child mode, in the suite's child environment, and read its
    samples."""
    env, _ = _harness()
    command = [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(spec)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                          env=env.child_env(home), cwd=str(REPO), check=False)
    if done.returncode != 0:
        raise PerfError(f"timing {spec['operation']} failed in its child process, which "
                        f"exited with {done.returncode}: {done.stderr[-600:]}")
    return [float(v) for v in json.loads(done.stdout.strip().splitlines()[-1])["ms"]]


def _read_outcome(hook: str, reply: Mapping[str, Any] | None) -> str:
    """The hook's status line, after checking that the hook read the store."""
    reply = reply or {}
    status = str(reply.get("systemMessage", ""))
    if not any(outcome in status for outcome in READ_OUTCOMES[hook]):
        raise PerfError(f"the {hook} hook did not read the store, so its time is not the "
                        f"time of a read: it said {status!r}")
    if hook == "session_start":
        context = str((reply.get("hookSpecificOutput") or {}).get("additionalContext", ""))
        visible = re.search(r"(\d+) claim\(s\) visible", context)
        if visible is None or int(visible.group(1)) == 0:
            raise PerfError("the session_start hook saw no claim, so it did not read the "
                            "store being timed")
    return status


def time_hook(db: Path, hook: str, runs: int, *, cold: bool, workdir: Path,
              names: Sequence[str], runner_factory: Callable[..., Any] | None = None,
              ) -> tuple[list[float], int]:
    """Time `runs` runs of one reading hook against the store at `db`, in milliseconds,
    and count the runs that passed the host's time limit.

    Cold runs each get a new home directory and session, so no state the hooks keep in
    the home directory carries over. Warm runs share one. A run past the limit is recorded
    at the limit, because dropping it would hide the worst sample. `runner_factory`
    replaces `HookRunner` in tests.
    """
    _, hooks = _harness()
    make = runner_factory or hooks.HookRunner
    cwd = workdir / "hook-cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    homes = workdir / "hook-homes"
    homes.mkdir(parents=True, exist_ok=True)
    server_env = {"MEMVARA_DB": str(db), "MEMVARA_USER": PERF_USER}
    samples: list[float] = []
    statuses: list[str] = []
    timeouts = 0
    runner, home = None, homes
    for run in range(runs):
        if cold or runner is None:
            home = Path(tempfile.mkdtemp(prefix="home-", dir=homes))
            runner = make("claude", home=home, cwd=cwd, server_env=server_env)
        fields = {"session": home.name}
        if hook == "recall":
            fields["prompt"] = f"tell me where {names[run % len(names)]} lives"
        try:
            result = runner.run(hook, **fields)
        except hooks.HookTimeout:
            samples.append(float(runner.host.timeouts[hook]) * 1000.0)
            timeouts += 1
            continue
        samples.append(result.elapsed * 1000.0)
        statuses.append(_read_outcome(hook, result.reply))
    if hook == "recall" and statuses and not any("recalled" in s for s in statuses):
        raise PerfError("the recall hook never recalled anything, so it did not read the "
                        f"store being timed: it said {statuses[0]!r}")
    return samples, timeouts


class _Bench:
    """The stores of one run, and how to measure one series on them."""

    def __init__(self, config: PerfConfig, workdir: Path,
                 runner_factory: Callable[..., Any] | None) -> None:
        self.config = config
        self.workdir = workdir
        self.runner_factory = runner_factory
        self.stores: dict[int, Path] = {}
        self.copies: dict[int, Path] = {}
        self.names: dict[int, list[str]] = {}
        self.blocks = 0

    def prepare(self, size: int) -> None:
        db = self.workdir / f"store-{size}.db"
        self.names[size] = build_store(db, size, seed=self.config.seed)
        self.stores[size] = db

    def copy_for_writes(self, size: int) -> None:
        """A copy of the built store for `remember`, so the reads keep their size."""
        source = self.stores[size]
        target = self.workdir / f"writes-{size}.db"
        for path in source.parent.iterdir():
            if path.name.startswith(source.name) and not path.name.endswith(".lock"):
                shutil.copy2(path, target.parent / (target.name + path.name[len(source.name):]))
        self.copies[size] = target

    def _home(self) -> Path:
        homes = self.workdir / "child-homes"
        homes.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="home-", dir=homes))

    def series(self, operation: str, size: int, temperature: str) -> tuple[list[float], int]:
        """Measure one series: the samples in milliseconds, and the timeouts."""
        cold = temperature == "cold"
        runs = self.config.cold if cold else self.config.warm
        if operation.startswith("hook."):
            return time_hook(self.stores[size], operation.removeprefix("hook."), runs,
                             cold=cold, workdir=self.workdir, names=self.names[size],
                             runner_factory=self.runner_factory)
        # Only as many queries as a series makes calls: they travel on the child's command
        # line, which macOS caps at 1 MB, and a 100,000-claim store names 25,000 people.
        wanted = max(self.config.cold, self.config.warm + WARMUP_CALLS)
        spec: dict[str, Any] = {
            "db": str(self.copies[size] if operation == "remember" else self.stores[size]),
            "operation": operation,
            "queries": [f"tell me where {name} lives" for name in self.names[size][:wanted]]}
        samples: list[float] = []
        for run in range(runs if cold else 1):
            self.blocks += 1
            spec.update(first=run, offset=self.blocks * 1_000_000,
                        warmup=0 if cold else WARMUP_CALLS, calls=1 if cold else runs)
            samples += _run_child(spec, self._home(), timeout=120.0 + 5.0 * runs)
        return samples, 0


def _parse_key(key: str) -> tuple[str, int, str]:
    operation, rest = key.split("@", 1)
    size, temperature = rest.split("/", 1)
    return operation, int(size), temperature


def _judge_all(series: Mapping[str, Mapping[str, Any]], history: Sequence[Mapping[str, Any]],
               fingerprint_id: str, bench: _Bench, config: PerfConfig) -> dict[str, Any]:
    """The regression rule applied to every series, against this machine's valid nights."""
    prior: dict[str, list[float]] = {}
    for night in nights_for(history, fingerprint_id):
        for key, stats in night.get("series", {}).items():
            prior.setdefault(key, []).append(float(stats["p95"]))
    verdicts: dict[str, Any] = {}
    for key, stats in series.items():
        operation, size, temperature = _parse_key(key)

        def remeasure(operation: str = operation, size: int = size,
                      temperature: str = temperature) -> float:
            samples, _ = bench.series(operation, size, temperature)
            return float(summarize(samples, resamples=config.resamples,
                                   seed=config.seed)["p95"])

        verdicts[key] = dataclasses.asdict(judge(float(stats["p95"]), prior.get(key, []),
                                                 remeasure, floor=LATENCY_FLOOR_MS))
    return verdicts


def check_budgets(series: Mapping[str, Mapping[str, Any]], fingerprint: Mapping[str, Any],
                  path: Path) -> dict[str, Any] | None:
    """Each series with a committed budget, its p95 and whether it is over. None when no
    budgets are committed, or when they were measured on another machine."""
    if not path.is_file():
        return None
    committed = json.loads(path.read_text(encoding="utf-8"))
    if (committed.get("fingerprint") or {}).get("id") != fingerprint.get("id"):
        return None
    return {key: {"p95": float(series[key]["p95"]), "budget_ms": float(budget),
                  "over": float(series[key]["p95"]) > float(budget)}
            for key, budget in committed.get("budgets_ms", {}).items() if key in series}


def measure(config: PerfConfig, workdir: Path, *, history: Sequence[Mapping[str, Any]] = (),
            runner_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Build a store of each size in `workdir`, time every series on it, and judge the run.

    The machine's conditions are read before the first size, after each size and at the
    end. A run they make invalid is not judged: its numbers describe the machine, and
    measuring a series again on a busy machine would only double the time it wastes.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    fingerprint = machine_fingerprint()
    bench = _Bench(config, workdir, runner_factory)
    checks = [read_conditions()]
    series: dict[str, dict[str, Any]] = {}
    for size in config.sizes:
        bench.prepare(size)
        for operation, temperature in ORDER:
            if operation == "remember" and size not in bench.copies:
                bench.copy_for_writes(size)
            samples, timeouts = bench.series(operation, size, temperature)
            series[series_key(operation, size, temperature)] = {
                **summarize(samples, resamples=config.resamples, seed=config.seed),
                "samples_ms": [round(sample, 3) for sample in samples],
                "timeouts": timeouts}
        checks.append(read_conditions())
    regressions: dict[str, Any] = {}
    if not invalid_reasons(checks):
        regressions = _judge_all(series, history, fingerprint["id"], bench, config)
    checks.append(read_conditions())
    reasons = invalid_reasons(checks)
    return {
        "kind": RECORD_KIND, "version": RECORD_VERSION,
        "date": started.date().isoformat(), "started": started.isoformat(),
        "finished": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint, "config": dataclasses.asdict(config),
        "conditions": [dataclasses.asdict(check) for check in checks],
        "valid": not reasons, "invalid_reasons": reasons, "series": series,
        "ceilings": check_ceilings(series),
        "regressions": {} if reasons else regressions,
        "budgets": check_budgets(series, fingerprint, BUDGETS_PATH),
    }


# --- nights, budgets and the command line -----------------------------------------------------


def load_runs(directory: Path) -> list[dict[str, Any]]:
    """Every timing record in `directory`, oldest first. Other files are skipped."""
    if not directory.is_dir():
        return []
    runs = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("kind") == RECORD_KIND:
            runs.append(data)
    return sorted(runs, key=lambda run: str(run.get("started", "")))


def nights_for(runs: Sequence[Mapping[str, Any]], fingerprint_id: str) -> list[dict[str, Any]]:
    """The valid runs on the machine `fingerprint_id` names, the last one of each date,
    oldest first. These are the nights the regression rule and the budgets count."""
    chosen: dict[str, Mapping[str, Any]] = {}
    for run in runs:
        if (run.get("kind") != RECORD_KIND or not run.get("valid")
                or (run.get("fingerprint") or {}).get("id") != fingerprint_id
                or not run.get("date")):
            continue
        date = str(run["date"])
        if date not in chosen or str(run.get("started", "")) >= str(chosen[date].get("started", "")):
            chosen[date] = run
    return [dict(chosen[date]) for date in sorted(chosen)]


def write_budgets(runs: Sequence[Mapping[str, Any]], fingerprint: Mapping[str, Any],
                  path: Path) -> dict[str, Any]:
    """Derive the library budgets from the last 14 valid nights and write them to `path`,
    once. Refused when `path` exists, and when fewer than 14 valid nights qualify."""
    if path.exists():
        raise FileExistsError(f"{path} exists: the design commits budgets once, so a new "
                              "machine needs its own file rather than a rewrite of this one")
    nights = nights_for(runs, str(fingerprint["id"]))
    if len(nights) < BUDGET_NIGHTS:
        raise ValueError(f"library budgets need {BUDGET_NIGHTS} valid nights on this "
                         f"machine, and there are {len(nights)}")
    chosen = nights[-BUDGET_NIGHTS:]
    library = [{key: float(stats["p95"]) for key, stats in night["series"].items()
                if not key.startswith("hook.")} for night in chosen]
    payload = {"fingerprint": dict(fingerprint),
               "nights": [night["date"] for night in chosen],
               "rule": (f"{BUDGET_MULTIPLIER} times the median p95 of {BUDGET_NIGHTS} valid "
                        "nights, rounded up to the next 1-2-5 step"),
               "budgets_ms": derive_budgets(library)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def skip_reason(record: Mapping[str, Any]) -> str:
    """Why the nightly test skips an invalid run, in the words the skip ledger's rule for
    it matches (tests/harness/skips.py)."""
    return "the performance run is invalid: " + "; ".join(record["invalid_reasons"])


def exit_code(record: Mapping[str, Any]) -> int:
    """0 for a valid run that passes, 1 for a valid run that fails, `EXIT_INVALID` for a
    run on battery or under load, whatever its numbers say."""
    if not record.get("valid"):
        return EXIT_INVALID
    failed = (any(entry.get("breached") for entry in record.get("ceilings", ()))
              or any(verdict.get("outcome") == "regression"
                     for verdict in (record.get("regressions") or {}).values())
              or any(entry.get("over") for entry in (record.get("budgets") or {}).values()))
    return 1 if failed else 0


def report(record: Mapping[str, Any]) -> str:
    """The run as text: the machine, its validity, each series, and anything that failed."""
    machine, config = record["fingerprint"], record["config"]
    lines = [f"timing on {machine['cpu']}, {machine['logical_cpus']} CPUs, "
             f"{machine['system']} {machine['release']}, commit {str(machine['commit'])[:8]}; "
             f"sizes {', '.join(str(size) for size in config['sizes'])}; "
             f"cold {config['cold']}, warm {config['warm']}",
             "valid" if record["valid"] else "INVALID, not failed: "
             + "; ".join(record["invalid_reasons"])]
    lines.append(f"{'series':30} {'n':>4} {'p50':>9} {'p90':>9} {'p95':>9} {'p99':>9} "
                 f"{'max':>9}  p95 interval")
    for key, stats in record["series"].items():
        low, high = stats["interval"]["p95"]
        lines.append(f"{key:30} {stats['n']:>4} {stats['p50']:>9.2f} {stats['p90']:>9.2f} "
                     f"{stats['p95']:>9.2f} {stats['p99']:>9.2f} {stats['max']:>9.2f}  "
                     f"{low:.2f} to {high:.2f}")
    breached = [entry for entry in record["ceilings"] if entry["breached"]]
    lines.append("ceilings: " + ("all held" if not breached else "; ".join(
        f"{e['series']} {e['stat']} {e['value_ms']:.0f} ms above {e['limit_ms']:.0f} ms"
        for e in breached)))
    moved = {key: verdict for key, verdict in (record["regressions"] or {}).items()
             if verdict["outcome"] in ("regression", "not reproduced")}
    for key, verdict in moved.items():
        lines.append(f"{verdict['outcome']}: {key}: {verdict['detail']}")
    for key, entry in (record["budgets"] or {}).items():
        if entry["over"]:
            lines.append(f"over budget: {key}: p95 {entry['p95']:.2f} ms against "
                         f"{entry['budget_ms']:.2f} ms")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Time the store sizes, print the report, and exit 0, 1 or `EXIT_INVALID`."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--sizes", default="1000,10000,100000",
                        help="store sizes in claims, separated by commas")
    parser.add_argument("--cold", type=int, default=30, help="fresh processes per series")
    parser.add_argument("--warm", type=int, default=200, help="calls per warm series")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="write the run's record to this JSON file")
    parser.add_argument("--history", type=Path, default=None,
                        help="a folder of earlier records to apply the regression rule to")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="the folder to build the stores in, which keeps them")
    parser.add_argument("--write-budgets", type=Path, default=None, metavar="PATH",
                        help="derive the library budgets from --history and write them here")
    args = parser.parse_args(argv)
    if args.child is not None:
        return _child(args.child)
    if args.write_budgets is not None:
        if args.history is None:
            parser.error("--write-budgets reads the nights from --history")
        try:
            payload = write_budgets(load_runs(args.history), machine_fingerprint(),
                                    args.write_budgets)
        except (ValueError, FileExistsError) as refused:
            parser.exit(2, f"{refused}\n")
        print(f"wrote {len(payload['budgets_ms'])} budgets to {args.write_budgets}")
        return 0
    config = PerfConfig(sizes=tuple(int(size) for size in args.sizes.split(",")),
                        cold=args.cold, warm=args.warm, resamples=args.resamples,
                        seed=args.seed)
    history = load_runs(args.history) if args.history is not None else []
    with tempfile.TemporaryDirectory(prefix="memvara-perf-") as scratch:
        record = measure(config, args.workdir or Path(scratch), history=history)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(report(record))
    return exit_code(record)


if __name__ == "__main__":
    raise SystemExit(main())
