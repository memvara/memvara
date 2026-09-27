"""The local test selection in scripts/test_changed.py, which is the first tier of testing.

It runs the tests a change can reach through imports, the tests that name a changed file
in a string, and the tests that failed last time, and it runs the full suite when a file
changes that it cannot follow. Its one failure worth fearing is quiet: a test the change
affects that it leaves out, so a developer pushes believing the change was checked. These
tests pin both halves: what must be selected, and what must send it to the full suite.

The end-to-end tests build a small git repository in a temporary folder, with the real
tier rules copied into it, and replace pytest with a runner that records its command.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load() -> ModuleType:
    path = ROOT / "scripts" / "test_changed.py"
    spec = importlib.util.spec_from_file_location("_test_changed_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tc = _load()


# -- What sends a change to the full suite --------------------------------------------------

@pytest.mark.parametrize("path, named", [
    ("conftest.py", "conftest"),
    ("tests/adversarial/hooks/conftest.py", "conftest"),
    ("pyproject.toml", "pytest's options"),
    ("requirements-dev.txt", "lockfile"),
    ("npm/memvara/package-lock.json", "lockfile"),
    ("tests/harness/tiers.py", "tiers module"),
    ("tests/fixtures/stores/v0.1.0/memory.db", "test data"),
    ("tests/scenarios/scripted/first.json", "test data"),
    ("tests/adversarial/nightly_runner/gh_recorded.json", "test data"),
    ("memvara/packs/engineering.toml", "package data"),
    ("memvara/skills/memvara/SKILL.md", "package data"),
    (".github/workflows/ci.yml", "CI configuration"),
])
def test_a_change_selection_cannot_follow_runs_the_full_suite(path: str, named: str) -> None:
    """Each of these reaches tests by a route the import graph does not see: pytest loads
    conftest files itself, the root conftest loads the tiers module from its path,
    pyproject.toml configures every run, and data and CI configuration are read by path.
    The reason is printed, so it has to say what the file is."""
    reason = tc.full_suite_reason(path)
    assert reason is not None and named in reason


@pytest.mark.parametrize("path", [
    "memvara/core.py", "tests/test_fast.py", "tests/adversarial/sessions/runner.py",
    "scripts/nightly/filing.py", "bench/soak.py", "docs/claude/testing.md", "README.md",
    "CLAUDE.md", "examples/quickstart.py", "tests/harness/env.py", "plugin/hooks/run.py"])
def test_an_ordinary_source_test_or_document_is_not_on_the_named_list(path: str) -> None:
    """These go through selection. Whether one of them still runs the full suite depends on
    whether any test reaches it, which the end-to-end tests below check.

    The test harness and the plugin's hooks were on the named list until 2026-09-27, and
    between them they sent 15 of 19 merged pull requests to the full suite. They are
    followed now: a harness module through imports, including a conftest's imports, and a
    hook through the strings that name the plugin folder, which is how every test that
    starts a hook finds it."""
    assert tc.full_suite_reason(path) is None


@pytest.mark.parametrize("path, prose", [
    ("README.md", True), ("CLAUDE.md", True), ("docs/claude/testing.md", True),
    ("docs/notes.rst", True), ("docs/list.txt", True), ("NOTES.txt", True),
    ("benchmarks/agent_memory/README.md", True), (".claude/rules/doctests.md", True),
    (".pre-commit-config.yaml", False), ("noxfile.py", False), ("LICENSE", False),
    ("docs/diagram.svg", False), ("memvara/skills/memvara/SKILL.md", False),
    ("plugin/skills/memvara/SKILL.md", False), ("npm/memvara/README.md", False),
    ("tests/fixtures/notes.md", False)])
def test_prose_is_a_short_explicit_set(path: str, prose: bool) -> None:
    """Markdown, reStructuredText and plain text may run nothing when no test names them,
    except under the folders whose Markdown ships in a package or is read at run time.
    Every other file no test reaches runs the full suite."""
    assert tc.is_prose(path) is prose


def test_the_folders_it_skips_are_the_folders_pyproject_tells_pytest_to_ignore() -> None:
    """pytest never collects what `--ignore` names, so selecting a file there would pass a
    path pytest refuses. The two lists must not drift apart."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    addopts = re.search(r'^addopts\s*=\s*"([^"]*)"', text, re.M)
    assert addopts is not None
    ignored = {f"{folder.rstrip('/')}/"
               for folder in re.findall(r"--ignore=(\S+)", addopts.group(1))}
    assert ignored == set(tc.NOT_COLLECTED)


# -- Reading one file -----------------------------------------------------------------------

def test_module_names_follow_every_folder_the_tests_import_from() -> None:
    """tests/ and the two script folders are on the import path when the tests run, so a
    file there is imported under a shorter name as well as its full one."""
    assert tc.module_names("memvara/server/__init__.py") == ["memvara.server"]
    assert tc.module_names("tests/harness/env.py") == ["tests.harness.env", "harness.env"]
    assert tc.module_names("scripts/nightly/filing.py") == [
        "scripts.nightly.filing", "nightly.filing"]
    assert tc.module_names("bench/evalkit.py") == ["bench.evalkit", "evalkit"]
    assert tc.module_names("docs/API.md") == []
    assert tc.module_names("npm/memvara-cli/x.py") == []


def test_every_kind_of_import_is_read_and_prose_is_not_read_as_a_path() -> None:
    source = tc.read_source("tests/adversarial/test_x.py", "\n".join([
        '"""A docstring that mentions docs/claude/testing.md in passing."""',
        "import memvara.store.sqlite",
        "from harness import env, stores",
        "from . import helpers",
        "from ..shared import tools",
        "def later():",
        "    import bench_module",
        'ARGS = ["-m", "memvara.server"]',
        'DOCS = ROOT / "docs" / "API.md"',
        'NOTE = "see docs/API.md for this"',
    ]))
    assert {"memvara.store.sqlite", "memvara.store", "memvara", "harness", "harness.env",
            "harness.stores", "bench_module"} <= source.names
    assert {"memvara.server", "memvara.server.__main__"} <= source.names
    assert {"tests/adversarial/helpers.py", "tests/shared.py",
            "tests/shared/tools.py"} <= source.files
    assert {"docs", "API.md", "-m"} <= source.strings
    assert not any("testing.md" in each or " " in each for each in source.strings)


def test_a_file_that_does_not_parse_imports_nothing_rather_than_stopping_the_run() -> None:
    source = tc.read_source("tests/test_broken.py", "def broken(:\n")
    assert (source.names, source.files, source.strings) == (set(), set(), set())


# -- Selection over a small graph ------------------------------------------------------------

def _graph(files: dict[str, str]) -> object:
    return tc.Graph({path: tc.read_source(path, text) for path, text in files.items()})


FILES = {
    "memvara/__init__.py": "from .core import Memvara\n",
    "memvara/core.py": "from .store import sqlite\nURL = 'https://memvara.dev/docs/cloud'\n",
    "memvara/store/__init__.py": "",
    "memvara/store/sqlite.py": "'''>>> 1 + 1\n2\n'''\n",
    "memvara/lazy.py": "def load():\n    import memvara.gone\n",
    "tests/harness/__init__.py": "",
    "tests/harness/runner.py": "import memvara\nARGS = ['-m', 'memvara.server']\n",
    "tests/test_core.py": "from memvara import Memvara\n",
    "tests/test_runner.py": "from harness import runner\n",
    "tests/test_docs.py": "DOCS = ROOT / 'docs'\n",
    "tests/test_readme.py": "PAGE = ROOT / 'README.md'\n",
    "tests/test_alone.py": "import json\n",
    "memvara/server/__init__.py": "",
    "memvara/server/__main__.py": "from . import tools\n",
    "memvara/server/tools.py": "",
}


def _affected(changed: list[str]) -> set[str]:
    return set(tc.affected(_graph(FILES), changed))


def test_a_change_reaches_every_test_that_imports_it_through_any_chain() -> None:
    """tests/test_core.py never names sqlite, but importing memvara runs it."""
    assert "tests/test_core.py" in _affected(["memvara/store/sqlite.py"])
    assert "tests/test_alone.py" not in _affected(["memvara/store/sqlite.py"])


def test_the_changed_modules_own_doctests_and_its_importers_doctests_run() -> None:
    reached = _affected(["memvara/store/sqlite.py"])
    assert {"memvara/store/sqlite.py", "memvara/core.py", "memvara/__init__.py"} <= reached


def test_a_module_a_test_starts_in_a_child_process_reaches_that_test() -> None:
    """`python -m memvara.server` runs memvara/server/__main__.py and what it imports, and
    the test that starts it imports none of them."""
    assert "tests/test_runner.py" in _affected(["memvara/server/tools.py"])
    assert "tests/test_core.py" not in _affected(["memvara/server/tools.py"])


def test_a_document_reaches_the_tests_that_name_it_or_its_top_folder() -> None:
    assert _affected(["README.md"]) == {"tests/test_readme.py"}
    assert _affected(["docs/claude/testing.md"]) == {"tests/test_docs.py"}


def test_a_string_in_the_library_does_not_make_every_test_read_the_docs() -> None:
    """memvara/core.py holds https://memvara.dev/docs/cloud and every test imports the
    library. If the library's strings counted, a change to any document would select
    every test."""
    assert "tests/test_core.py" not in _affected(["docs/API.md"])


def test_a_deleted_module_reaches_the_tests_that_still_import_it_by_name() -> None:
    """memvara/gone.py is not in the graph any more, and the import that names it now
    fails. The tests that reach that import must run and show it."""
    assert "tests/test_core.py" not in _affected(["memvara/gone.py"])
    assert "memvara/lazy.py" in _affected(["memvara/gone.py"])


def test_a_changed_test_file_is_always_its_own_target() -> None:
    assert _affected(["tests/test_alone.py"]) == {"tests/test_alone.py"}


def test_every_module_pytest_imports_to_look_for_doctests_is_a_target() -> None:
    """pyproject.toml passes --doctest-modules, so pytest imports every module under tests/
    and memvara/ while it collects, whether or not the module holds an example. A support
    module that fails to import fails the run, so it has to be run itself. The packaged
    skill is ignored by pytest, and a conftest file is loaded rather than collected."""
    assert not tc.is_target("memvara/skills/memvara/auth.py")
    assert tc.is_target("tests/harness/runner.py")
    assert tc.is_target("tests/harness/__init__.py")
    assert tc.is_target("tests/adversarial/parity/compare.py")
    assert tc.is_target("tests/adversarial/test_adv_x.py")
    assert tc.is_target("memvara/core.py")
    assert not tc.is_target("tests/adversarial/conftest.py")
    assert not tc.is_target("tests/fixtures/notes.md")
    assert not tc.is_target("scripts/nightly/filing.py")


def test_a_support_module_that_no_test_imports_is_still_run_when_it_changes() -> None:
    """The mutation check that found this broke tests/harness/known_bugs.py so that it
    could not be imported. It failed through that module itself, collected for its
    doctests, although it holds none."""
    files = {"tests/harness/__init__.py": "", "tests/harness/ledger.py": "X = 1\n",
             "tests/test_alone.py": "import json\n"}
    assert set(tc.affected(_graph(files), ["tests/harness/ledger.py"])) == {
        "tests/harness/ledger.py"}


COMPARE = 'def normalise(x):\n    """>>> normalise(1)\n    {}\n    """\n    return x\n'


def test_a_support_module_under_tests_with_a_doctest_is_its_own_target() -> None:
    """pyproject.toml passes --doctest-modules, so pytest runs the examples in
    tests/adversarial/parity/compare.py as tests of their own. A broken example there once
    selected only the four test files that import the module, and passed, while
    `pytest tests/adversarial/parity/compare.py` failed."""
    files = {"tests/adversarial/parity/compare.py": COMPARE.format(1),
             "tests/adversarial/parity/test_adv_parity_x.py": "from . import compare\n"}
    reached = set(tc.affected(_graph(files), ["tests/adversarial/parity/compare.py"]))
    assert reached == {"tests/adversarial/parity/compare.py",
                       "tests/adversarial/parity/test_adv_parity_x.py"}


# -- Routes that are not an import of the test file itself ---------------------------------
#
# A mutation check on 2026-09-27 broke eleven files one at a time, ran the whole fast tier
# each time, and compared the failing tests with what this selection chose. The rule of
# the time missed a failing test twice: a new Markdown file that tests/test_docs.py reads
# through ROOT.rglob("*.md"), and a module that only a conftest file imports, whose break
# failed tests/adversarial/test_adv_tiers.py. The tests below pin each route it follows now.

def test_a_file_a_test_reads_through_a_glob_pattern_reaches_that_test() -> None:
    """A bare "*" matches everything and says nothing, and a glob in the library is a
    message or an address, like every other string there."""
    files = {"tests/test_docs.py": "PAGES = sorted(ROOT.rglob('*.md'))\n",
             "tests/test_any.py": "EVERYTHING = sorted(ROOT.glob('*'))\n",
             "memvara/__init__.py": "PATTERN = '*.yaml'\n",
             "tests/test_library.py": "import memvara\n"}
    assert set(tc.affected(_graph(files), ["release/NOTES.md"])) == {"tests/test_docs.py"}
    assert set(tc.affected(_graph(files), ["release/notes.yaml"])) == set()


def test_a_module_only_a_conftest_imports_reaches_every_test_below_that_conftest() -> None:
    """pytest imports a conftest file before the tests below it, so a module that fails to
    import there fails all of them, whether or not they use its fixtures. This is the
    shape of tests/adversarial/sessions/switches.py."""
    files = {"tests/adversarial/__init__.py": "",
             "tests/adversarial/sessions/__init__.py": "",
             "tests/adversarial/sessions/conftest.py": "from . import switches\n",
             "tests/adversarial/sessions/switches.py": "X = 1\n",
             "tests/adversarial/sessions/test_adv_a.py": "def test_a():\n    pass\n",
             "tests/adversarial/test_adv_elsewhere.py": "def test_b():\n    pass\n"}
    reached = set(tc.affected(_graph(files), ["tests/adversarial/sessions/switches.py"]))
    assert reached == {"tests/adversarial/sessions/__init__.py",
                       "tests/adversarial/sessions/switches.py",
                       "tests/adversarial/sessions/test_adv_a.py"}


HOOKS_CONFTEST = """import pytest

@pytest.fixture{arguments}
def runner():
    return ROOT / "plugin" / "hooks"
"""


@pytest.mark.parametrize("arguments, test_b, reached", [
    ("", "def test_b():\n    pass\n", {"tests/hooks/test_uses.py"}),
    ("", "pytestmark = pytest.mark.usefixtures('runner')\n",
     {"tests/hooks/test_uses.py", "tests/hooks/test_other.py"}),
    ("(autouse=True)", "def test_b():\n    pass\n",
     {"tests/hooks/__init__.py", "tests/hooks/test_uses.py", "tests/hooks/test_other.py"}),
], ids=["only-the-test-that-asks", "usefixtures", "autouse"])
def test_a_string_in_a_conftest_reaches_the_tests_that_use_its_fixtures(
        arguments: str, test_b: str, reached: set[str]) -> None:
    """This is how a test that starts a plugin hook finds it: a conftest fixture builds
    the path from the plugin folder's name. A test that uses none of that conftest's
    fixtures runs none of its code, so the string does not reach it. An autouse fixture
    runs for the doctests of the package's own __init__.py as well."""
    files = {"tests/hooks/__init__.py": "",
             "tests/hooks/conftest.py": HOOKS_CONFTEST.format(arguments=arguments),
             "tests/hooks/test_uses.py": "def test_a(runner):\n    pass\n",
             "tests/hooks/test_other.py": test_b}
    assert set(tc.affected(_graph(files), ["plugin/hooks/recall.py"])) == reached


def test_what_the_tests_conftest_imports_outside_the_library_is_session_wide() -> None:
    """tests/conftest.py registers the skip ledger as a plugin that sees every test in a
    run, so a change to the ledger, or to anything it imports, runs the full suite."""
    files = {"memvara/__init__.py": "",
             "tests/conftest.py": "import memvara\nfrom harness import skips\n",
             "tests/harness/__init__.py": "",
             "tests/harness/skips.py": "from . import rules\n",
             "tests/harness/rules.py": "",
             "tests/harness/other.py": ""}
    assert tc.session_wide(_graph(files)) == {
        "tests/harness/__init__.py", "tests/harness/skips.py", "tests/harness/rules.py"}
    assert tc.session_wide(_graph({"tests/test_a.py": ""})) == set()


@pytest.fixture(scope="module")
def this_repository() -> object:
    """The import graph of this repository's tests, library and plugin."""
    paths = [path.relative_to(ROOT).as_posix()
             for folder in ("tests", "memvara", "plugin")
             for path in (ROOT / folder).rglob("*.py")]
    return tc.Graph.build(ROOT, paths)


@pytest.mark.parametrize("changed, reader", [
    ("release/NOTES.md", "tests/test_docs.py"),
    ("tests/adversarial/sessions/switches.py", "tests/adversarial/sessions/test_adv_runner.py"),
    ("plugin/hooks/recall.py", "tests/adversarial/hooks/test_adv_hook_approve.py"),
])
def test_the_two_misses_and_the_plugin_route_hold_in_this_repository(
        this_repository: object, changed: str, reader: str) -> None:
    """The same routes, checked against the real files rather than a model of them."""
    assert reader in tc.affected(this_repository, [changed])


def test_every_whole_tree_reader_exists() -> None:
    """A renamed or removed reader would stop being selected without any message."""
    for path in tc.WHOLE_TREE_READERS:
        assert (ROOT / path).is_file(), path


# -- The last run's failures ---------------------------------------------------------------

def test_the_last_runs_failures_are_read_from_pytests_cache(tmp_path: pathlib.Path) -> None:
    assert tc.last_failed(tmp_path) == []
    cache = tmp_path / ".pytest_cache" / "v" / "cache"
    cache.mkdir(parents=True)
    (cache / "lastfailed").write_text("not json")
    assert tc.last_failed(tmp_path) == []
    (cache / "lastfailed").write_text("[1, 2]")
    assert tc.last_failed(tmp_path) == []
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("")
    (cache / "lastfailed").write_text(json.dumps({
        "tests/test_a.py::test_one[x-1]": True, "tests/test_deleted.py::test_two": True}))
    assert tc.last_failed(tmp_path) == ["tests/test_a.py::test_one[x-1]"]


# -- End to end, in a small repository -------------------------------------------------------

def _git(repo: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "-c", "commit.gpgsign=false", *args], cwd=repo, check=True,
                   capture_output=True)


def _write(repo: pathlib.Path, path: str, text: str) -> None:
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text)


@pytest.fixture()
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """A repository on branch work, one commit ahead of main, with nothing changed yet."""
    root = tmp_path / "repo"
    (root / "tests" / "harness").mkdir(parents=True)
    shutil.copy(ROOT / "tests" / "harness" / "tiers.py", root / "tests" / "harness")
    for path, text in FILES.items():
        _write(root, path, text)
    _write(root, "tests/adversarial/__init__.py", "")
    _write(root, "tests/adversarial/nightly/__init__.py", "")
    _write(root, "tests/adversarial/nightly/test_slow.py", "import memvara\n")
    _write(root, "pyproject.toml", "")
    _write(root, ".gitignore", ".pytest_cache/\n__pycache__/\n")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "start")
    _git(root, "checkout", "-q", "-b", "work")
    return root


def _given(command: list[str]) -> list[str]:
    """The targets and the arguments passed through, without the options main adds: `-n
    auto` when pytest-xdist is installed, and the run's own `--basetemp`."""
    rest = command[4:]
    if rest[:2] == ["-n", "auto"]:
        rest = rest[2:]
    if rest and rest[0].startswith("--basetemp="):
        rest = rest[1:]
    return rest


class Recorder:
    def __init__(self, code: int = 0) -> None:
        self.code = code
        self.calls: list[tuple[list[str], pathlib.Path, dict[str, str]]] = []

    def __call__(self, command: list[str], cwd: pathlib.Path, env: dict[str, str]) -> int:
        self.calls.append((command, cwd, env))
        return self.code


def test_a_source_change_runs_the_tests_that_reach_it_and_the_last_failures(
        repo: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(repo, "memvara/server/tools.py", "CHANGED = True\n")
    _git(repo, "commit", "-qam", "change tools")
    _write(repo, "tests/test_alone.py", "import json  # edited, not committed\n")
    cache = repo / ".pytest_cache" / "v" / "cache"
    cache.mkdir(parents=True)
    (cache / "lastfailed").write_text(json.dumps({"tests/test_readme.py::test_x": True}))
    runner = Recorder()
    assert tc.main(["--base", "main", "--", "-x"], repo=repo, runner=runner) == 0
    [(command, cwd, env)] = runner.calls
    assert command[:4] == [sys.executable, "-m", "pytest", "-q"]
    assert _given(command) == ["memvara/server/__main__.py", "memvara/server/tools.py",
                           "tests/harness/runner.py", "tests/test_alone.py",
                           "tests/test_runner.py",
                           "tests/test_readme.py::test_x", "-x"]
    assert cwd == repo
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(repo)
    printed = capsys.readouterr().out
    assert "Mode: selected tests." in printed
    assert "Changed: memvara/server/tools.py, tests/test_alone.py." in printed
    assert "1 more test that failed on the last run is run again." in printed


def test_a_change_it_cannot_follow_runs_the_full_suite_and_says_which_file(
        repo: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(repo, "pyproject.toml", "[tool.pytest.ini_options]\n")
    _write(repo, "memvara/core.py", "CHANGED = True\n")
    runner = Recorder()
    assert tc.main(["--base", "main"], repo=repo, runner=runner) == 0
    [(command, _, _)] = runner.calls
    assert command[:4] == [sys.executable, "-m", "pytest", "-q"] and _given(command) == []
    printed = capsys.readouterr().out
    assert "Mode: the full suite" in printed
    assert "pyproject.toml changed: it holds pytest's options" in printed


def test_a_changed_test_in_a_slow_tier_is_named_with_its_command_and_not_run(
        repo: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A plain pytest and CI run only the fast tier. A nightly test run here by default
    could take the time this tier exists to save, so it is named instead, with the command
    that runs it."""
    _write(repo, "tests/adversarial/nightly/test_slow.py", "import memvara  # edited\n")
    runner = Recorder()
    assert tc.main(["--base", "main"], repo=repo, runner=runner) == 0
    assert runner.calls == []
    printed = capsys.readouterr().out
    assert "Mode: nothing to run." in printed
    assert ("Not run: tests/adversarial/nightly/test_slow.py is in the nightly tier. Run it "
            "with `python3 -m pytest tests/adversarial/nightly/test_slow.py`.") in printed


def test_a_file_no_test_reaches_runs_the_full_suite_unless_it_is_prose(
        repo: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A new tool's configuration file is read by something this cannot see, so it runs
    everything. A document no test names is read by nothing that can fail, so it runs
    nothing; a document that a test names runs that test."""
    _write(repo, ".pre-commit-config.yaml", "repos: []\n")
    runner = Recorder()
    assert tc.main(["--base", "main"], repo=repo, runner=runner) == 0
    assert [_given(command) for command, _, _ in runner.calls] == [[]]
    assert (".pre-commit-config.yaml changed: no test imports or names it, and it is not "
            "prose") in capsys.readouterr().out
    (repo / ".pre-commit-config.yaml").unlink()

    _write(repo, "NOTES.md", "# Notes\n")
    runner = Recorder()
    assert tc.main(["--base", "main"], repo=repo, runner=runner) == 0
    assert runner.calls == []
    assert "Mode: nothing to run." in capsys.readouterr().out

    _write(repo, "docs/guide.md", "# Guide\n")
    runner = Recorder()
    assert tc.main(["--base", "main"], repo=repo, runner=runner) == 0
    [(command, _, _)] = runner.calls
    assert _given(command) == ["tests/test_docs.py"]
    assert "Mode: selected tests." in capsys.readouterr().out


def test_a_broken_doctest_in_a_support_module_is_run_itself(repo: pathlib.Path) -> None:
    """The case a reviewer reproduced: the example in a support module under tests/ is
    what changed, so pytest has to be given that module, not only the tests importing it."""
    _write(repo, "tests/adversarial/parity/__init__.py", "")
    _write(repo, "tests/adversarial/parity/compare.py", COMPARE.format(1))
    _write(repo, "tests/adversarial/parity/test_adv_parity_x.py", "from . import compare\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "add parity")
    _git(repo, "branch", "-f", "main", "HEAD")
    _write(repo, "tests/adversarial/parity/compare.py", COMPARE.format(2))
    runner = Recorder()
    assert tc.main(["--base", "main"], repo=repo, runner=runner) == 0
    [(command, _, _)] = runner.calls
    assert _given(command) == ["tests/adversarial/parity/compare.py",
                           "tests/adversarial/parity/test_adv_parity_x.py"]


def test_a_dry_run_prints_the_plan_and_runs_nothing(
        repo: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(repo, "README.md", "# Changed\n")
    runner = Recorder()
    assert tc.main(["--base", "main", "--dry-run"], repo=repo, runner=runner) == 0
    assert runner.calls == []
    assert "tests/test_readme.py" in capsys.readouterr().out


def test_selected_files_that_hold_no_tests_are_not_a_failure(
        repo: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A module with no doctests collects nothing, and pytest says so with exit status 5.
    Nothing failed, so the run passes. In the full suite, the same status would mean the
    suite lost its tests, so it is passed on there."""
    _write(repo, "memvara/lazy.py", "X = 1\n")
    assert tc.main(["--base", "main"], repo=repo, runner=Recorder(code=5)) == 0
    assert "hold no tests" in capsys.readouterr().out
    _write(repo, "pyproject.toml", "# changed\n")
    assert tc.main(["--base", "main"], repo=repo, runner=Recorder(code=5)) == 5
    assert tc.main(["--base", "main"], repo=repo, runner=Recorder(code=1)) == 1


def test_a_base_that_does_not_exist_stops_with_gits_own_message(repo: pathlib.Path) -> None:
    with pytest.raises(SystemExit, match="git merge-base no-such-ref HEAD failed"):
        tc.main(["--base", "no-such-ref"], repo=repo, runner=Recorder())


def test_a_renamed_module_counts_under_its_old_name_too(repo: pathlib.Path) -> None:
    """The tests that imported the old name now fail to import it, so they must run."""
    _git(repo, "mv", "memvara/lazy.py", "memvara/eager.py")
    _, changed = tc.changed_files("main", cwd=repo)
    assert changed == ["memvara/eager.py", "memvara/lazy.py"]


def test_a_new_document_a_test_globs_for_runs_that_test(repo: pathlib.Path) -> None:
    """The first miss the mutation check found, end to end: a new release note with a
    wording the project has retired failed tests/test_docs.py, which never names it."""
    _write(repo, "tests/test_wording.py", "PAGES = sorted(ROOT.rglob('*.md'))\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "add a glob reader")
    _git(repo, "branch", "-f", "main", "HEAD")
    _write(repo, "release/NOTES.md", "# Notes\n")
    runner = Recorder()
    assert tc.main(["--base", "main"], repo=repo, runner=runner) == 0
    [(command, _, _)] = runner.calls
    assert _given(command) == ["tests/test_wording.py"]


def test_a_python_change_runs_the_test_that_collects_the_whole_tree(
        repo: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The second miss: tests/adversarial/test_adv_tiers.py collects every tier in a child
    process, so a nightly test that cannot be imported fails it, and nothing else in the
    fast tier. A change to a document does not select it."""
    _write(repo, "tests/adversarial/test_adv_tiers.py", "import subprocess\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "add the whole-tree reader")
    _git(repo, "branch", "-f", "main", "HEAD")
    _write(repo, "tests/adversarial/nightly/test_slow.py", "import memvara  # edited\n")
    runner = Recorder()
    assert tc.main(["--base", "main"], repo=repo, runner=runner) == 0
    [(command, _, _)] = runner.calls
    assert _given(command) == ["tests/adversarial/test_adv_tiers.py"]
    printed = capsys.readouterr().out
    assert ("tests/adversarial/test_adv_tiers.py runs as well, because it collects every "
            "test module in a child process") in printed
    assert "Not run: tests/adversarial/nightly/test_slow.py is in the nightly tier." in printed

    _git(repo, "checkout", "-q", "--", "tests/adversarial/nightly/test_slow.py")
    _write(repo, "README.md", "# Changed\n")
    runner = Recorder()
    assert tc.main(["--base", "main"], repo=repo, runner=runner) == 0
    [(command, _, _)] = runner.calls
    assert _given(command) == ["tests/test_readme.py"]


def test_a_change_to_what_the_tests_conftest_registers_runs_the_full_suite(
        repo: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(repo, "tests/conftest.py", "from harness import skips\n")
    _write(repo, "tests/harness/skips.py", "RULES = ()\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "add the ledger")
    _git(repo, "branch", "-f", "main", "HEAD")
    _write(repo, "tests/harness/skips.py", "RULES = ('changed',)\n")
    runner = Recorder()
    assert tc.main(["--base", "main"], repo=repo, runner=runner) == 0
    [(command, _, _)] = runner.calls
    assert _given(command) == []
    assert ("tests/harness/skips.py changed: tests/conftest.py imports it and registers it "
            "as a plugin that sees every test in the run.") in capsys.readouterr().out


@pytest.mark.parametrize("installed, passed, workers", [
    (True, [], ["-n", "auto"]),
    (True, ["-n", "0"], []),
    (True, ["-n4"], []),
    (True, ["--numprocesses=2"], []),
    (True, ["-p", "no:xdist"], []),
    (False, [], []),
])
def test_it_runs_one_worker_per_core_when_pytest_xdist_is_installed(
        repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, installed: bool,
        passed: list[str], workers: list[str]) -> None:
    """Workers cost about two seconds to start and cut a large selection several times
    over. Arguments after -- that choose the workers win, and without pytest-xdist the
    run is serial rather than an error."""
    monkeypatch.setattr(tc, "xdist_installed", lambda: installed)
    _write(repo, "README.md", "# Changed\n")
    runner = Recorder()
    assert tc.main(["--base", "main", "--", *passed], repo=repo, runner=runner) == 0
    [(command, _, _)] = runner.calls
    options = [arg for arg in command[4:] if not arg.startswith("--basetemp=")]
    assert options == [*workers, "tests/test_readme.py", *passed]


class TempRecorder(Recorder):
    """A runner that also records whether the run's base temporary directory existed."""

    def __call__(self, command: list[str], cwd: pathlib.Path, env: dict[str, str]) -> int:
        [basetemp] = [arg.split("=", 1)[1] for arg in command if arg.startswith("--basetemp=")]
        self.basetemp = pathlib.Path(basetemp)
        self.existed = self.basetemp.is_dir()
        return super().__call__(command, cwd, env)


def test_each_run_gets_a_base_temporary_directory_of_its_own(
        repo: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """pytest's default base temporary directory is shared by every run of one user, and
    pytest deletes all but the newest three runs' folders in it, so concurrent runs by
    several agents deleted each other's files. A run that passes leaves nothing behind; a
    run that fails keeps its files and says where."""
    _write(repo, "README.md", "# Changed\n")
    first, second = TempRecorder(), TempRecorder()
    assert tc.main(["--base", "main"], repo=repo, runner=first) == 0
    assert tc.main(["--base", "main"], repo=repo, runner=second) == 0
    assert first.existed and second.existed and first.basetemp != second.basetemp
    assert first.basetemp.parent == pathlib.Path(tempfile.gettempdir())
    assert not first.basetemp.exists() and not second.basetemp.exists()

    failing = TempRecorder(code=1)
    assert tc.main(["--base", "main"], repo=repo, runner=failing) == 1
    assert failing.basetemp.is_dir()
    assert (f"the temporary files of this run are kept in {failing.basetemp}"
            in capsys.readouterr().out)
    shutil.rmtree(failing.basetemp)


def test_a_base_temporary_directory_passed_through_is_used_instead(
        repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    _write(repo, "README.md", "# Changed\n")
    runner = Recorder()
    chosen = f"--basetemp={tmp_path / 'mine'}"
    assert tc.main(["--base", "main", "--", chosen], repo=repo, runner=runner) == 0
    [(command, _, _)] = runner.calls
    assert [arg for arg in command if arg.startswith("--basetemp")] == [chosen]
