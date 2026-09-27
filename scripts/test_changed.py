#!/usr/bin/env python3
"""Run the tests a change can affect, before you push. This is the local tier of testing.

    python3 scripts/test_changed.py                 # compare with origin/main
    python3 scripts/test_changed.py --base main     # compare with another ref
    python3 scripts/test_changed.py --dry-run       # say what would run, and run nothing
    python3 scripts/test_changed.py -- -x -n 4      # everything after -- goes to pytest

It finds the files that differ from the merge base with `--base`, including uncommitted
and untracked files, and runs pytest on:

* the test files that changed;
* the test files and doctest modules that import a changed Python file, directly or
  through other modules. A doctest module is any module under memvara/, and any other
  module under tests/ that holds a doctest example, because pyproject.toml passes
  --doctest-modules;
* the test files that name a changed file in a string, such as `ROOT / "docs"` or
  `"-m", "memvara.server"`, directly or through a module they import. That is how a test
  reaches a document, a data file or a script it runs as a separate process;
* the tests that failed on the last run in this checkout, from pytest's own cache.

It runs the full suite instead in two cases. The first is a file in FULL_SUITE or
DATA_FOLDERS below: a file that reaches the tests by a route this cannot see, such as a
conftest.py file, pyproject.toml, the test harness or test data. The second is any other
changed file that no test imports or names and that is not prose (see PROSE_EXTENSIONS),
such as a new tool's configuration file: nothing shows what reads it, so the safe answer
is everything. Every run prints which mode it chose and which file decided it.

Only fast-tier tests are run, which is what a plain `pytest` and CI run. A changed test in
a nightly, weekly, local or quarantine folder is named in the output, with the command to
run it, and is not run.

This is not a gate and does not measure coverage. The pull request's CI runs the full
suite on every interpreter, with coverage and mypy. It exists so that the local check
before a push takes seconds rather than the nine minutes of the full local gate.
`docs/claude/working-here.md` has the five tiers and the reasons for them.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from types import ModuleType

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_BASE = "origin/main"

#: Changes that dependency tracking cannot see through, so they run the full suite. Each
#: entry is a set of patterns, matched against the path from the repository root with
#: fnmatch, where `*` also matches `/`, and the reason. The first entry that matches wins.
FULL_SUITE: tuple[tuple[tuple[str, ...], str], ...] = (
    (("conftest.py", "*/conftest.py"),
     "a conftest file sets fixtures and hooks for every test below it, and no test imports it"),
    (("pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini"),
     "it holds pytest's options, the dependencies or the coverage settings, which apply to "
     "every test"),
    (("requirements*.txt", "*/requirements*.txt", "*.lock", "*-lock.json", "*-lock.yaml"),
     "a lockfile or requirements file changes the installed packages every test runs "
     "against"),
    (("tests/harness/*",),
     "the test harness is shared by every layer of the adversarial suite, and the root "
     "conftest.py loads its tiers module from a path rather than by import"),
    (("tests/fixtures/*", "tests/scenarios/*"),
     "test data is read from its path, and a data file is not something a test imports"),
    ((".github/*",),
     "CI configuration decides what CI runs, which only a full run can stand in for"),
    (("plugin/*",),
     "the plugin's hooks run as separate processes and load each host's module by name "
     "at run time, which the import graph cannot follow"),
)

#: Folders whose non-Python files run the full suite, with the reason. Python files there
#: are followed through the import graph like any other.
DATA_FOLDERS: dict[str, str] = {
    "tests/": "a non-Python file under tests/ is test data, read from its path",
    "memvara/": "a non-Python file under memvara/ is package data, such as a predicate pack "
                "or the packaged skill, which the library reads at run time",
}

#: Prose: a file with one of these extensions anywhere outside NOT_PROSE_FOLDERS. A changed
#: prose file that no test names runs nothing. Any other changed file that no test imports
#: or names runs the full suite, because nothing shows what reads it. Markdown under the
#: folders below is not prose: it ships in a package or is read by code at run time, such
#: as the packaged skill, the npm package's README or a test fixture.
PROSE_EXTENSIONS = (".md", ".rst", ".txt")
NOT_PROSE_FOLDERS = ("memvara/", "tests/", "plugin/", "npm/")

#: The reason printed for a changed file that no test reaches and that is not prose.
UNKNOWN = ("no test imports or names it, and it is not prose, so nothing shows what reads "
           "it and every test may")

#: The folders Python puts on the import path when the tests run: the repository root,
#: tests/ (it has no __init__.py, so pytest adds it), and the two folders that tests add
#: themselves before importing a script from them.
IMPORT_ROOTS = ("", "tests", "scripts", "bench")

#: Files whose own folder name means nothing in a string, because every test names it.
#: A changed file in them is followed through the import graph and its own name only.
NAMED_THROUGH_IMPORTS = ("memvara/", "tests/")

#: The library. Its strings are not searched for the names of changed files: they are
#: messages and addresses such as https://memvara.dev/docs/, and every test imports the
#: library, so one of them would make every test look as if it read the docs/ folder.
LIBRARY = "memvara/"

#: pytest's default test file names, which this repository does not change.
TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")

#: pyproject.toml passes --ignore=memvara/skills, so nothing there is collected.
NOT_COLLECTED = ("memvara/skills/",)

#: pytest exits with this when it collected no tests.
NO_TESTS_COLLECTED = 5


def git(*args: str, cwd: pathlib.Path = REPO) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"test_changed: git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def changed_files(base: str, cwd: pathlib.Path = REPO) -> tuple[str, list[str]]:
    """The merge base of `base` and HEAD, and every file that differs from it: committed,
    staged, unstaged and untracked. A renamed file is listed under both of its names,
    because the tests that imported the old name are affected too."""
    merge_base = git("merge-base", base, "HEAD", cwd=cwd).strip()
    diffed = git("diff", "--name-only", "--no-renames", "-z", merge_base, cwd=cwd)
    untracked = git("ls-files", "--others", "--exclude-standard", "-z", cwd=cwd)
    names = {name for name in (diffed + untracked).split("\0") if name}
    return merge_base, sorted(names)


def full_suite_reason(path: str) -> str | None:
    """Why a change to `path` needs the full suite, or None when selection can follow it."""
    for patterns, reason in FULL_SUITE:
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns):
            return reason
    if not path.endswith(".py"):
        for folder, reason in DATA_FOLDERS.items():
            if path.startswith(folder):
                return reason
    return None


def is_prose(path: str) -> bool:
    """Whether `path` is documentation that nothing but a test naming it can read."""
    return path.endswith(PROSE_EXTENSIONS) and not path.startswith(NOT_PROSE_FOLDERS)


def module_names(path: str) -> list[str]:
    """The dotted names under which `path` can be imported, one per import root that holds
    it. tests/harness/env.py is both `tests.harness.env` and `harness.env`."""
    if not path.endswith(".py"):
        return []
    names = []
    for root in IMPORT_ROOTS:
        prefix = f"{root}/" if root else ""
        if not path.startswith(prefix):
            continue
        parts = path[len(prefix):-len(".py")].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts and all(part.isidentifier() for part in parts):
            names.append(".".join(parts))
    return names


def _with_parents(name: str) -> list[str]:
    """`a.b.c` and the packages above it, since importing a.b.c runs a/__init__.py and
    a/b/__init__.py first."""
    parts = name.split(".")
    return [".".join(parts[:end]) for end in range(len(parts), 0, -1)]


@dataclass
class Source:
    """What one Python file imports and which strings it holds."""

    #: Absolute module names it imports, each with the packages above it. A string that
    #: reads as a dotted module name counts as well, with its `__main__`, because that is
    #: how a test starts a module in a child process: `"-m", "memvara.server"`.
    names: set[str] = field(default_factory=set)
    #: Files its relative imports reach, as paths from the repository root.
    files: set[str] = field(default_factory=set)
    #: Every string literal that holds no whitespace, whole and split into the parts of a
    #: path. A string with whitespace in it is prose, such as a docstring or a message
    #: saying "see docs/claude/testing.md", and reading prose as a path would make every
    #: test that imports a documented helper look as if it read the documents.
    strings: set[str] = field(default_factory=set)
    #: Whether it holds a doctest example. pyproject.toml passes --doctest-modules, so
    #: pytest collects the examples in every module under tests/ and memvara/, support
    #: modules such as tests/adversarial/parity/compare.py included.
    doctests: bool = False


_SEPARATORS = re.compile(r"[/\\]+")
_DOTTED = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")


def read_source(path: str, text: str) -> Source:
    """Parse one file. A file that does not parse imports nothing, as far as this can tell,
    and still counts as changed if it changed."""
    source = Source(doctests=">>>" in text)
    try:
        tree = ast.parse(text, filename=path)
    except (SyntaxError, ValueError):
        return source
    folder = pathlib.PurePosixPath(path).parent
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                source.names.update(_with_parents(alias.name))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            source.names.update(_with_parents(node.module))
            source.names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = folder
            for _ in range(node.level - 1):
                base = base.parent
            if node.module:
                base = base.joinpath(*node.module.split("."))
            for target in [base] + [base / alias.name for alias in node.names]:
                source.files.add(f"{target}.py".removeprefix("./"))
                source.files.add(f"{target}/__init__.py".removeprefix("./"))
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and not any(character.isspace() for character in node.value)):
            source.strings.add(node.value)
            source.strings.update(part for part in _SEPARATORS.split(node.value) if part)
            if _DOTTED.fullmatch(node.value):
                source.names.update(_with_parents(node.value))
                source.names.add(f"{node.value}.__main__")
    return source


class Graph:
    """The repository's Python files, what each imports, and which strings each holds."""

    def __init__(self, sources: dict[str, Source]) -> None:
        self.sources = sources
        self.by_name: dict[str, str] = {}
        for path in sorted(sources):
            for name in module_names(path):
                self.by_name.setdefault(name, path)
        self._closures: dict[str, frozenset[str]] = {}
        self._imports: dict[str, set[str]] = {}

    @classmethod
    def build(cls, repo: pathlib.Path, paths: Iterable[str]) -> Graph:
        sources = {}
        for path in paths:
            file = repo / path
            if path.endswith(".py") and file.is_file():
                sources[path] = read_source(path, file.read_text(encoding="utf-8",
                                                                 errors="replace"))
        return cls(sources)

    def imports(self, path: str) -> set[str]:
        """The files `path` imports directly and that exist here."""
        if path not in self._imports:
            source = self.sources[path]
            found = {self.by_name[name] for name in source.names if name in self.by_name}
            self._imports[path] = found | {file for file in source.files
                                           if file in self.sources}
        return self._imports[path]

    def closure(self, path: str) -> frozenset[str]:
        """`path` and every file it imports, directly or through other files."""
        if path not in self._closures:
            seen = {path}
            todo = [path]
            while todo:
                for found in self.imports(todo.pop()):
                    if found not in seen:
                        seen.add(found)
                        todo.append(found)
            self._closures[path] = frozenset(seen)
        return self._closures[path]


def tokens(path: str) -> tuple[set[str], set[str]]:
    """What a test has to import, and what it has to name in a string, to be affected by a
    change to `path`.

    A Python file is imported under its module names; read_source counts a string such as
    `"-m", "memvara.server"` as an import too. Any file can be named by its
    own file name. A file outside memvara/ and tests/ can also be named by its top-level
    folder, as in `ROOT / "docs"`, because a test that builds a path from the repository
    root has to name that folder, even when it then globs for the file. Inside memvara/
    and tests/ every test names the folder, so the folder says nothing. Deeper folders are
    not used: a folder such as docs/claude shares its name with strings that mean
    something else, such as the name of a host.
    """
    names = set(module_names(path))
    named = set()
    parts = path.split("/")
    if parts[-1] != "__init__.py":
        named.add(parts[-1])
    if len(parts) > 1 and not path.startswith(NAMED_THROUGH_IMPORTS):
        named.add(parts[0])
    return names, named


def is_target(path: str, *, doctests: bool = False) -> bool:
    """Whether pytest collects `path` as a test file or as a module of doctests.

    Under tests/, that is a test file, or any other module that holds a doctest example
    (`doctests`). Under memvara/, every module is a target, whether or not it holds an
    example today, so that a changed library module always selects at least itself."""
    if path.startswith(NOT_COLLECTED) or not path.endswith(".py"):
        return False
    if path.startswith("tests/"):
        name = path.rsplit("/", 1)[-1]
        return doctests or any(fnmatch.fnmatchcase(name, pattern)
                               for pattern in TEST_FILE_PATTERNS)
    return path.startswith("memvara/")


def affected(graph: Graph, changed: Sequence[str]) -> dict[str, set[str]]:
    """Each test file or doctest module a change reaches, with the changed files that
    reach it. A target is reached when a file in its import closure is a changed file,
    imports a changed module by name, or, outside the library, names a changed file in a
    string."""
    wanted = [(path, *tokens(path)) for path in changed]
    found: dict[str, set[str]] = {}
    for target in sorted(path for path, source in graph.sources.items()
                         if is_target(path, doctests=source.doctests)):
        closure = graph.closure(target)
        for path, names, named in wanted:
            if path in closure or any(
                    names & graph.sources[each].names
                    or (named & graph.sources[each].strings and not each.startswith(LIBRARY))
                    for each in closure):
                found.setdefault(target, set()).add(path)
    return found


def load_tiers(repo: pathlib.Path) -> ModuleType:
    """tests/harness/tiers.py, loaded from its path as the root conftest.py loads it."""
    path = repo / "tests" / "harness" / "tiers.py"
    spec = importlib.util.spec_from_file_location("_test_changed_tiers", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def last_failed(repo: pathlib.Path) -> list[str]:
    """The node ids that failed on the last pytest run in this checkout, whose files still
    exist. pytest keeps them in its cache folder; an unreadable cache means none."""
    cache = repo / ".pytest_cache" / "v" / "cache" / "lastfailed"
    try:
        entries = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(entries, dict):
        return []
    return sorted(node for node in entries
                  if isinstance(node, str) and (repo / node.split("::", 1)[0]).exists())


@dataclass
class Plan:
    """What one run decided: the mode, the reasons, and the pytest arguments."""

    mode: str  # "full", "selected" or "nothing"
    reasons: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    other_tiers: dict[str, str] = field(default_factory=dict)

    def describe(self) -> list[str]:
        headline = {"full": "Mode: the full suite, meaning the fast tier, as CI runs it.",
                    "selected": "Mode: selected tests.",
                    "nothing": "Mode: nothing to run."}[self.mode]
        lines = [headline] + [f"  {reason}" for reason in self.reasons]
        for path, tier in sorted(self.other_tiers.items()):
            lines.append(f"  Not run: {path} is in the {tier} tier. Run it with "
                         f"`python3 -m pytest {path}`.")
        return lines


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def make_plan(repo: pathlib.Path, changed: Sequence[str], failed: Sequence[str]) -> Plan:
    """Decide what to run for the `changed` files and the `failed` node ids."""
    triggers = [(path, reason) for path in changed
                if (reason := full_suite_reason(path)) is not None]
    if triggers:
        return Plan("full", [f"{path} changed: {reason}." for path, reason in triggers])

    tiers = load_tiers(repo)
    tracked = git("ls-files", "-z", "*.py", cwd=repo).split("\0")
    graph = Graph.build(repo, {path for path in tracked if path} | set(changed))
    reached = affected(graph, changed)
    reaching = set().union(*reached.values())
    unknown = [path for path in changed if path not in reaching and not is_prose(path)]
    if unknown:
        return Plan("full", [f"{path} changed: {UNKNOWN}." for path in unknown])

    plan = Plan("selected")
    for path in sorted(reached):
        tier = tiers.tier_of(repo / path)
        if tier != "fast":
            if path in changed:
                plan.other_tiers[path] = tier
            continue
        plan.targets.append(path)
    tests = [path for path in plan.targets if path.startswith("tests/")]
    library = [path for path in plan.targets if path.startswith("memvara/")]
    changed_tests = [path for path in tests if path in changed]
    plan.reasons.append(f"Changed: {', '.join(changed)}." if changed else "Nothing changed.")
    plan.reasons.append(f"{_count(len(tests), 'file')} under tests/ and "
                        f"{_count(len(library), 'library module')} import or name a changed "
                        f"file. Of those files under tests/, {len(changed_tests)} changed "
                        "themselves.")
    selected = set(plan.targets)
    extra = [node for node in failed if node.split("::", 1)[0] not in selected
             and tiers.tier_of(repo / node.split("::", 1)[0]) == "fast"]
    plan.targets += extra
    plan.reasons.append(f"{_count(len(extra), 'more test')} that failed on the last run "
                        f"{'is' if len(extra) == 1 else 'are'} run again." if extra else
                        "No other test failed on the last run.")
    if not plan.targets:
        plan.mode = "nothing"
        plan.reasons.append("No test imports or names a changed file.")
    return plan


def pytest_command(plan: Plan, passthrough: Sequence[str]) -> list[str]:
    targets = [] if plan.mode == "full" else plan.targets
    return [sys.executable, "-m", "pytest", "-q", *targets, *passthrough]


def child_env(repo: pathlib.Path) -> dict[str, str]:
    """The environment pytest runs in, with this checkout first on the import path. An
    editable install of memvara can point at another checkout, and then `import memvara`
    would test that one instead of this."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo), env.get("PYTHONPATH", "")) if part)
    return env


Runner = Callable[[list[str], pathlib.Path, dict[str, str]], int]


def run_pytest(command: list[str], cwd: pathlib.Path, env: dict[str, str]) -> int:
    return subprocess.run(command, cwd=cwd, env=env).returncode


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="test_changed.py",
        description="Run the tests a change can affect: the local tier of testing. "
                    "Arguments after -- are passed to pytest.")
    command.add_argument("--base", default=DEFAULT_BASE,
                         help=f"the ref to compare with; {DEFAULT_BASE} by default")
    command.add_argument("--dry-run", action="store_true",
                         help="print what would run, and run nothing")
    return command


def main(argv: Sequence[str] | None = None, *, repo: pathlib.Path = REPO,
         runner: Runner = run_pytest) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in args:
        split = args.index("--")
        args, passthrough = args[:split], args[split + 1:]
    options = parser().parse_args(args)

    merge_base, changed = changed_files(options.base, cwd=repo)
    print(f"test_changed: comparing this checkout with {options.base}, merge base "
          f"{merge_base[:12]}.")
    plan = make_plan(repo, changed, last_failed(repo))
    for line in plan.describe():
        print(line)
    if plan.mode == "nothing":
        return 0
    command = pytest_command(plan, passthrough)
    shown = command if len(command) <= 12 else command[:10] + [
        f"... and {len(command) - 10} more arguments"]
    print(f"Running: {' '.join(shown)}", flush=True)
    if options.dry_run:
        return 0
    code = runner(command, repo, child_env(repo))
    if code == NO_TESTS_COLLECTED and plan.mode == "selected":
        print("test_changed: the selected files hold no tests, so nothing failed.")
        return 0
    return code


if __name__ == "__main__":
    sys.exit(main())
