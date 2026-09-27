"""Mutation testing: how many small, deliberate bugs in a module the tests catch.

    PYTHONPATH=. python3 bench/mutation.py run memvara/write/reconcile.py
    PYTHONPATH=. python3 bench/mutation.py run memvara/write/reconcile.py \\
        --tests tests/test_reconcile.py tests/test_pipeline.py --max-children 6
    PYTHONPATH=. python3 bench/mutation.py report local/mutation/<run>/report.json

A mutant is a copy of a module with one small change: `<` becomes `<=`, a string gains
`XX` at both ends, `and` becomes `or`, a number moves by one. The tests are run against each
mutant. A mutant the tests fail on is **killed**, which is what a good test suite does. A
mutant they pass on **survived**, which means no test would notice that bug. The score of a
module is the share of its mutants that were caught.

The tool is mutmut 3, pinned at `MUTMUT` below. It is not a dependency of this package, so
install it beside the development extras first: `python3 -m pip install 'mutmut==3.8.0'`.
The design is section "S2: mutation testing" of
`docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`, and
`docs/claude/testing.md` describes how to read a run.

## Where it runs

mutmut writes a `mutants/` folder and a configuration beside the code it mutates, so a run
never touches this checkout. It clones the checkout's committed `HEAD` into a temporary
folder, writes the configuration there, runs mutmut there, reads the results and deletes
the clone. Uncommitted changes are therefore not measured; commit first. `--keep` leaves
the clone in place, so `python -m mutmut show <mutant>` and `mutmut browse` can be run in
it afterwards.

## Which tests run

By default, every test file under `tests/` that imports the module by name, found the way
`scripts/test_changed.py` reads imports. That is narrower than every test the module can
affect, which for a module under `memvara/` is most of the suite, and it is what keeps a
run to minutes. A mutant that only a broader test would catch then counts as undetected,
so the score is a lower bound. `--tests` replaces the default with a list you give.

mutmut first runs the selection once to record which tests reach which function, and then
runs, for each mutant, only the tests that reached its function. A mutant in a function
that no selected test reaches is reported as **no tests**, and counts as undetected.

## Equivalent mutants

Some mutants cannot be caught by any test, because the change does not change behaviour:
a mutated default that every caller overrides, for example. Those are listed in
`bench/mutation_equivalents.toml`, one table per mutant, each with a reason, and they are
left out of the score. A listed mutant that a run did not produce is reported, because its
entry has gone stale. The design allows at most `MAX_EQUIVALENTS` entries; past that, the
tool should be replaced rather than the list grown.

## The score

For each module: caught / (all mutants - equivalents - mutants mutmut skipped or did not
check). Caught is killed, timed out, or caught by the type check. A timeout counts as
caught because the mutant made a test hang, which a test run notices. Survived, no tests
and suspicious all count as not caught. `--floor 80` makes the run fail when a module
scores below 80%.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import Counter
from typing import Any, Sequence

REPO = pathlib.Path(__file__).resolve().parent.parent

#: The mutmut release this script reads the results of. Its result files are not a stable
#: interface, so a different release is refused rather than misread.
MUTMUT = "3.8.0"

EQUIVALENTS = REPO / "bench" / "mutation_equivalents.toml"

#: The most entries `EQUIVALENTS` may hold. The design says that a tool needing more than
#: ten exemptions should be replaced by cosmic-ray instead.
MAX_EQUIVALENTS = 10

#: mutmut 3's status for each exit code a mutant's test run can end with, copied from
#: `mutmut/stats.py` at `MUTMUT`. Any other code is "suspicious", as there.
STATUS_BY_EXIT_CODE: dict[int | None, str] = {
    1: "killed", 3: "killed", 0: "survived", 5: "no tests", 33: "no tests",
    2: "check was interrupted by user", None: "not checked", 34: "skipped",
    35: "suspicious", 36: "timeout", 24: "timeout", -24: "timeout", 152: "timeout",
    255: "timeout", 37: "caught by type check", -11: "segfault", -9: "segfault",
}

CAUGHT = frozenset({"killed", "timeout", "caught by type check", "segfault"})
#: Not counted either way: mutmut did not get to test these mutants.
UNCOUNTED = frozenset({"skipped", "not checked", "check was interrupted by user"})


def status_of(exit_code: int | None) -> str:
    """The status mutmut gives a mutant whose test run ended with `exit_code`.

    >>> status_of(1), status_of(0), status_of(33), status_of(99)
    ('killed', 'survived', 'no tests', 'suspicious')
    """
    return STATUS_BY_EXIT_CODE.get(exit_code, "suspicious")


def dotted(module: str) -> str:
    """`memvara/write/reconcile.py` as mutmut names it, `memvara.write.reconcile`.

    >>> dotted("memvara/write/reconcile.py")
    'memvara.write.reconcile'
    """
    path = pathlib.PurePosixPath(module)
    if path.suffix != ".py":
        raise ValueError(f"{module} is not a Python file")
    parts = path.with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def module_of(mutant: str) -> str:
    """The module a mutant belongs to. mutmut names a mutant `<module>.<function>__mutmut_<n>`,
    with a method's class folded into the function part as `xǁClassǁmethod`.

    >>> module_of("memvara.write.reconcile.x__bounds__mutmut_7")
    'memvara.write.reconcile'
    >>> module_of("memvara.write.reconcile.xǁReconcilerǁrun__mutmut_2")
    'memvara.write.reconcile'
    """
    return mutant.rsplit(".", 1)[0]


@dataclasses.dataclass(frozen=True)
class Equivalent:
    mutant: str
    reason: str


def load_equivalents(path: pathlib.Path = EQUIVALENTS) -> dict[str, Equivalent]:
    """The equivalent mutants, by name. Every entry needs a mutant and a reason, a name may
    appear once, and there may be at most `MAX_EQUIVALENTS` entries."""
    if not path.exists():
        return {}
    rows = tomllib.loads(path.read_text(encoding="utf-8")).get("equivalent", [])
    found: dict[str, Equivalent] = {}
    for row in rows:
        mutant, reason = str(row.get("mutant", "")).strip(), str(row.get("reason", "")).strip()
        if not mutant or not reason:
            raise ValueError(f"{path.name}: every entry needs a mutant and a reason: {row}")
        if mutant in found:
            raise ValueError(f"{path.name}: {mutant} is listed twice")
        found[mutant] = Equivalent(mutant, reason)
    if len(found) > MAX_EQUIVALENTS:
        raise ValueError(
            f"{path.name} lists {len(found)} equivalent mutants, more than the "
            f"{MAX_EQUIVALENTS} the design allows. Replace the tool rather than grow the list.")
    return found


@dataclasses.dataclass
class ModuleScore:
    module: str
    counts: dict[str, int]
    equivalents: int
    undetected: list[str]

    @property
    def counted(self) -> int:
        """The mutants the score is taken over."""
        return sum(n for status, n in self.counts.items() if status not in UNCOUNTED)

    @property
    def caught(self) -> int:
        return sum(n for status, n in self.counts.items() if status in CAUGHT)

    @property
    def score(self) -> float | None:
        """Caught as a percentage of the counted mutants, or None when none were counted."""
        return None if self.counted == 0 else 100.0 * self.caught / self.counted


def score(statuses: dict[str, str], equivalents: dict[str, Equivalent]) -> list[ModuleScore]:
    """One score per module, from each mutant's status. Equivalent mutants are left out.

    >>> s = score({"m.f__mutmut_1": "killed", "m.f__mutmut_2": "survived",
    ...            "m.f__mutmut_3": "survived", "m.f__mutmut_4": "not checked"},
    ...           {"m.f__mutmut_3": Equivalent("m.f__mutmut_3", "no caller passes it")})
    >>> s[0].score, s[0].undetected, s[0].equivalents
    (50.0, ['m.f__mutmut_2'], 1)
    """
    by_module: dict[str, dict[str, str]] = {}
    for mutant, status in statuses.items():
        by_module.setdefault(module_of(mutant), {})[mutant] = status
    scores = []
    for module in sorted(by_module):
        mutants = by_module[module]
        kept = {m: s for m, s in mutants.items() if m not in equivalents}
        scores.append(ModuleScore(
            module=module,
            counts=dict(Counter(kept.values())),
            equivalents=len(mutants) - len(kept),
            undetected=sorted(m for m, s in kept.items()
                              if s not in CAUGHT and s not in UNCOUNTED)))
    return scores


def stale(statuses: dict[str, str], equivalents: dict[str, Equivalent],
          modules: Sequence[str]) -> list[str]:
    """Listed equivalents in the measured modules that this run did not produce. mutmut
    numbers a function's mutants in order, so an edit to the function renames them, and an
    entry naming a mutant that no longer exists no longer exempts anything."""
    measured = {dotted(m) for m in modules}
    return sorted(name for name in equivalents
                  if module_of(name) in measured and name not in statuses)


def _test_changed() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_test_changed", REPO / "scripts" / "test_changed.py")
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def default_tests(modules: Sequence[str], repo: pathlib.Path = REPO) -> list[str]:
    """Every test file under tests/ that imports one of `modules` by name."""
    reader = _test_changed()
    wanted = {dotted(m) for m in modules}
    found = []
    for path in sorted((repo / "tests").rglob("test_*.py")):
        relative = path.relative_to(repo).as_posix()
        if not reader.is_target(relative):
            continue
        source = reader.read_source(relative, path.read_text(encoding="utf-8",
                                                             errors="replace"))
        if source.names & wanted:
            found.append(relative)
    return found


def mutmut_config(modules: Sequence[str], tests: Sequence[str]) -> str:
    """The `[mutmut]` section of the clone's setup.cfg.

    >>> print(mutmut_config(["memvara/write/reconcile.py"], ["tests/test_reconcile.py"]))
    [mutmut]
    source_paths=memvara
    only_mutate=memvara/write/reconcile.py
    pytest_add_cli_args_test_selection=tests/test_reconcile.py
    pytest_add_cli_args=-p
        no:cacheprovider
        -q
    <BLANKLINE>
    """
    def listed(values: Sequence[str]) -> str:
        return "\n    ".join(values)
    return ("[mutmut]\nsource_paths=memvara\n"
            f"only_mutate={listed(modules)}\n"
            f"pytest_add_cli_args_test_selection={listed(tests)}\n"
            "pytest_add_cli_args=-p\n    no:cacheprovider\n    -q\n")


def read_statuses(clone: pathlib.Path, modules: Sequence[str]) -> dict[str, str]:
    """Each mutant's status, from the `.meta` file mutmut writes beside each mutated module."""
    statuses: dict[str, str] = {}
    for module in modules:
        meta = clone / "mutants" / f"{module}.meta"
        if not meta.exists():
            raise FileNotFoundError(f"mutmut wrote no results for {module} ({meta})")
        data = json.loads(meta.read_text(encoding="utf-8"))
        for mutant, code in data["exit_code_by_key"].items():
            statuses[mutant] = status_of(code)
    return statuses


def _check_mutmut(python: str) -> None:
    done = subprocess.run([python, "-c", "import mutmut; print(mutmut.__version__)"],
                          capture_output=True, text=True)
    version = done.stdout.strip()
    if done.returncode != 0:
        raise SystemExit(f"{python} cannot import mutmut. Install it with "
                         f"`{python} -m pip install 'mutmut=={MUTMUT}'`.")
    if version != MUTMUT:
        raise SystemExit(f"{python} has mutmut {version}, and this script reads the results "
                         f"of mutmut {MUTMUT}. Install that release, or update MUTMUT here "
                         "after checking that its result files still read the same way.")


def _diff(clone: pathlib.Path, python: str, mutant: str) -> str:
    done = subprocess.run([python, "-m", "mutmut", "show", mutant], cwd=clone,
                          capture_output=True, text=True, timeout=60)
    return done.stdout.strip()


def run(args: argparse.Namespace) -> int:
    modules = [pathlib.PurePosixPath(m).as_posix() for m in args.modules]
    for module in modules:
        if not module.startswith("memvara/") or not (REPO / module).is_file():
            raise SystemExit(f"{module} is not a module under memvara/ in this checkout")
    tests = args.tests or default_tests(modules)
    if not tests:
        raise SystemExit("no test file imports these modules; name the tests with --tests")
    equivalents = load_equivalents()
    _check_mutmut(args.python)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = pathlib.Path(args.out) if args.out else REPO / "local" / "mutation" / stamp
    out.mkdir(parents=True, exist_ok=True)
    clone = pathlib.Path(tempfile.mkdtemp(prefix="memvara-mutation-"))
    try:
        subprocess.run(["git", "clone", "--quiet", "--shared", str(REPO), str(clone)],
                       check=True)
        subprocess.run(["git", "checkout", "--quiet", head], cwd=clone, check=True)
        (clone / "setup.cfg").write_text(mutmut_config(modules, tests), encoding="utf-8")
        print(f"== mutating {', '.join(modules)} at {head[:12]}, "
              f"with {len(tests)} test files, in {clone}")
        began = time.monotonic()
        with open(out / "mutmut.log", "w", encoding="utf-8") as log:
            done = subprocess.run(
                [args.python, "-m", "mutmut", "run", "--max-children", str(args.max_children)],
                cwd=clone, stdout=log, stderr=subprocess.STDOUT, timeout=args.cap * 60)
        seconds = time.monotonic() - began
        if done.returncode != 0:
            print(f"mutmut exited {done.returncode}; see {out / 'mutmut.log'}",
                  file=sys.stderr)
            return 2
        statuses = read_statuses(clone, modules)
        scores = score(statuses, equivalents)
        report = {
            "commit": head, "modules": modules, "tests": tests, "mutmut": MUTMUT,
            "seconds": round(seconds, 1), "max_children": args.max_children,
            "stale_equivalents": stale(statuses, equivalents, modules),
            "scores": [{"module": s.module, "score": s.score, "counted": s.counted,
                        "caught": s.caught, "equivalents": s.equivalents,
                        "counts": s.counts,
                        "undetected": [{"mutant": m, "status": statuses[m],
                                        "diff": _diff(clone, args.python, m)}
                                       for m in s.undetected]}
                       for s in scores],
        }
    finally:
        if args.keep:
            print(f"kept the clone at {clone}")
        else:
            shutil.rmtree(clone, ignore_errors=True)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(render(report))
    print(f"\nwrote {out / 'report.json'}")
    return _verdict(report, args.floor)


def render(report: dict[str, Any]) -> str:
    """The summary a person reads: one line per module, and the stale entries."""
    lines = [f"commit {report['commit'][:12]}, mutmut {report['mutmut']}, "
             f"{len(report['tests'])} test files, {report['seconds']} s"]
    for s in report["scores"]:
        value = "n/a" if s["score"] is None else f"{s['score']:.1f}%"
        counts = ", ".join(f"{n} {status}" for status, n in sorted(s["counts"].items()))
        lines.append(f"  {s['module']}: {value} caught ({s['caught']} of {s['counted']}); "
                     f"{counts}; {s['equivalents']} listed as equivalent")
    for name in report["stale_equivalents"]:
        lines.append(f"  stale entry in {EQUIVALENTS.name}: {name} was not produced")
    return "\n".join(lines)


def _verdict(report: dict[str, Any], floor: float | None) -> int:
    if floor is None:
        return 0
    low = [s for s in report["scores"] if s["score"] is not None and s["score"] < floor]
    for s in low:
        print(f"{s['module']} scored {s['score']:.1f}%, under the floor of {floor}%",
              file=sys.stderr)
    return 1 if low else 0


def report_command(args: argparse.Namespace) -> int:
    report = json.loads(pathlib.Path(args.report).read_text(encoding="utf-8"))
    print(render(report))
    if args.survivors:
        for s in report["scores"]:
            for row in s["undetected"]:
                print(f"\n# {row['mutant']}: {row['status']}\n{row['diff']}")
    return _verdict(report, args.floor)


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = top.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run", help="mutate modules in a throwaway clone and score them")
    go.add_argument("modules", nargs="+", help="modules under memvara/, as paths")
    go.add_argument("--tests", nargs="+", default=None,
                    help="test files to run; the default is those that import the modules")
    go.add_argument("--max-children", type=int, default=4,
                    help="mutants tested at once")
    go.add_argument("--cap", type=float, default=60.0,
                    help="minutes mutmut may run before it is stopped")
    go.add_argument("--python", default=sys.executable,
                    help="the interpreter that has mutmut and this package's dependencies")
    go.add_argument("--out", default="", help="folder for report.json and mutmut.log")
    go.add_argument("--keep", action="store_true", help="leave the clone in place")
    go.add_argument("--floor", type=float, default=None,
                    help="fail when a module scores below this percentage")
    go.set_defaults(func=run)
    show = sub.add_parser("report", help="print a run's summary again")
    show.add_argument("report", help="a report.json a run wrote")
    show.add_argument("--survivors", action="store_true",
                      help="also print every undetected mutant's diff")
    show.add_argument("--floor", type=float, default=None)
    show.set_defaults(func=report_command)
    return top


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
