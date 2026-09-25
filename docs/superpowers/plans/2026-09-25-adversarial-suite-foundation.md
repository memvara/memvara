# Adversarial suite foundation (F1 and F2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared harness that every later part of the adversarial suite uses:

- a real stdio MCP server driver;
- a real hook runner;
- a safe child environment;
- tiers selected by folder;
- a skip ledger;
- Hypothesis profiles.

On top of it, land the confirmed public bugs as strict xfail tests with GitHub issues.

**Architecture:**

- **Harness.** A support package, `tests/harness/`, holds no tests. Test code lives in `tests/adversarial/`.
- **Tiers.** `tests/conftest.py` gains a `--tier` option. `pytest_ignore_collect` leaves out folders named `nightly`, `weekly`, `local` and `quarantine` unless the tier selects them.
- **Skip ledger.** A plugin registered from `tests/conftest.py` fails the session when any skip has no rule.
- **Process drivers.** `McpProcess` and `HookRunner` start real child processes with an environment built by `harness.env.child_env`.

**Tech Stack:** Python 3.10–3.13, pytest 8, Hypothesis 6.100 or later, subprocess and threading from the standard library, and `gh` for issues and the private advisory.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`.

This plan covers PRs F1 and F2 only. F3 (fakes), F4 (scenarios), F5 (checklist), D1 (model and state machines) and D2 (concurrency and crashes) get their own plans once F1's interfaces exist in code.

## Global Constraints

- **Platforms.** Python `>=3.10`. CI runs 3.10, 3.11, 3.12 and 3.13 on Ubuntu, plus 3.13 on macOS and Windows. Every fast-tier test here must pass on all of them.
- **Offline.** No test may reach the network.
- **Child processes** get their environment from `harness.env.child_env`, and never the real home directory.
- **Embedder.** Every `Memvara(...)` built in `tests/` passes `embedder=`, or `tests/conftest.py` fails the run.
- **Deprecations are errors** (`filterwarnings = ["error::DeprecationWarning"]`). A hook implementation must not declare pytest's deprecated `path` argument.
- **Budget.** The fast tier may add at most 3 minutes to any CI job. F1 and F2 together should add under 40 seconds on Windows.
- **Prose.** All prose (docstrings, comments, docs, commit messages, PR bodies, issues) is plain English that a reader with no context understands on the first read. Do not copy the clipped voice of older files.
- **No AI attribution.** No AI attribution and no model name in any commit, PR title, PR body or issue. No `Co-Authored-By` trailer. This rule also binds any subagent that writes to GitHub.
- **Commits.** Commit files by name. Never `git add -A`, `git add .` or `git commit -a`. Never stash.
- **Documentation** ships in the same commit as the code it describes.
- **Security.** Findings that fall under the in-scope list of `SECURITY.md` never go into a public issue or a public test. They go to a private draft advisory.

## Review Focus

1. **A hung server.** It must fail the test within its timeout, never stall the suite. Pinned by `test_a_silent_server_times_out_instead_of_hanging` (Task 5).
2. **A tier folder without `__init__.py`.** It would make its modules top-level, and two files with the same name would then collide. Pinned by `test_every_tier_folder_is_a_package` (Task 2).
3. **A hook that prints something other than JSON.** That is a bug, because it desynchronises the client, and it must be reported with the output and stderr attached, not as a bare `JSONDecodeError`. Pinned by `test_output_that_is_not_json_is_reported_with_its_text` (Task 6).
4. **A suite that imported another copy of memvara.** A stale editable install makes every result describe the wrong code, so this must fail first, with the fix in the message. Pinned by `test_the_suite_imports_the_checkout_it_lives_in` (Task 1).
5. **A new skip.** A skip with a reason no rule explains must turn the run red. A skip is green in every summary, so an unexplained one hides a test that stopped running. Pinned by `test_an_unexplained_skip_fails_the_run` (Task 3).

---

## Task 0: Working environment (no commit)

- [ ] **Step 1: Create a virtual environment that matches CI.** The system Python's `memvara` is an editable install that points at a deleted scratch directory, so subprocess tests would import nothing.

```bash
cd /Applications/workstation/agent-memory/.claude/worktrees/friendly-einstein-53c8da
python3 -m venv local/venv-ci
local/venv-ci/bin/python -m pip install -q --upgrade pip
local/venv-ci/bin/python -m pip install -q -e ".[dev,cloud,ingest,encrypt]" "hypothesis>=6.100"
local/venv-ci/bin/python -c "import memvara, hypothesis; print(memvara.__file__, hypothesis.__version__)"
```

Expected: the printed path is inside this worktree. `local/` is ignored by git.

- [ ] **Step 2: Record the baseline.**

```bash
local/venv-ci/bin/python -m pytest -q -rs -p no:cacheprovider > local/baseline/pytest-ci.txt 2>&1; tail -30 local/baseline/pytest-ci.txt
```

Expected: every test passes. The `SKIPPED` lines list today's skip reasons, and Task 3's rules must cover every one of them.

---

## Task 1: The harness package and the child environment

**Files:**
- Create: `tests/harness/__init__.py`
- Create: `tests/harness/env.py`
- Create: `tests/adversarial/__init__.py`
- Create: `tests/adversarial/test_adv_env.py`
- Create: `docs/claude/testing.md`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `harness.env.REPO: pathlib.Path`, the checkout root;
  - `harness.env.REAL_HOME: pathlib.Path`;
  - `harness.env.child_env(home: pathlib.Path, extra: Mapping[str, str] | None = None) -> dict[str, str]`.

- [ ] **Step 1: Write the failing tests**

`tests/adversarial/__init__.py`:

```python
"""The adversarial test suite. See docs/claude/testing.md."""
```

`tests/adversarial/test_adv_env.py`:

```python
"""The environment the adversarial suite gives every child process."""

from __future__ import annotations

import pathlib

import pytest

import memvara
from harness.env import REAL_HOME, REPO, child_env


def test_the_suite_imports_the_checkout_it_lives_in() -> None:
    """A worktree can import another copy of memvara through a stale editable install.

    Every result in this suite would then describe code that is not under test, so this
    fails first and says how to fix it.
    """
    imported = pathlib.Path(memvara.__file__).resolve()
    assert REPO in imported.parents, (
        f"memvara was imported from {imported}, not from this checkout ({REPO}). Run "
        "with PYTHONPATH set to the checkout, or from a virtual environment that has this "
        "checkout installed in editable mode. docs/claude/testing.md explains both.")


def test_a_child_never_gets_the_real_home_directory() -> None:
    with pytest.raises(ValueError, match="real home directory"):
        child_env(REAL_HOME)


def test_a_child_does_not_inherit_memvara_or_model_variables(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMVARA_API_KEY", "mv_must_not_reach_a_child")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-must-not-reach-a-child")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.invalid")
    monkeypatch.setenv("CLAUDECODE", "1")
    env = child_env(tmp_path)
    for name in ("MEMVARA_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL", "CLAUDECODE"):
        assert name not in env


def test_a_child_runs_this_checkout_offline_in_the_home_it_was_given(
        tmp_path: pathlib.Path) -> None:
    env = child_env(tmp_path, {"MEMVARA_USER": "tester"})
    assert env["HOME"] == env["USERPROFILE"] == str(tmp_path.resolve())
    assert env["PYTHONPATH"] == str(REPO)
    assert env["MEMVARA_EMBEDDER"] == "hashing"
    assert env["MEMVARA_FEATURE_ENCRYPTION"] == "0"
    assert env["MEMVARA_FEATURE_PROJECT_SCOPE"] == "0"
    assert env["MEMVARA_DAEMON"] == "1"
    assert env["MEMVARA_USER"] == "tester"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_env.py`
Expected: FAIL. Collection errors with `ModuleNotFoundError: No module named 'harness'`.

- [ ] **Step 3: Write the implementation**

`tests/harness/__init__.py`:

```python
"""Support code for the adversarial test suite. Nothing in this package is a test.

docs/claude/testing.md explains the suite. Every module here must be safe to import with
no side effects, because `--doctest-modules` imports each one while pytest collects.
"""
```

`tests/harness/env.py`:

```python
"""The environment every child process of the adversarial suite runs in."""

from __future__ import annotations

import os
import pathlib
from typing import Mapping

#: The checkout under test. This file is tests/harness/env.py, two levels below it.
REPO = pathlib.Path(__file__).resolve().parents[2]


def _real_home() -> pathlib.Path:
    """The account's home directory, read from the password database on POSIX.

    Not read from HOME, because every test runs with HOME pointed at a temporary
    directory (tests/conftest.py), and a check against HOME would compare that temporary
    directory with itself.
    """
    if os.name == "posix":
        import pwd  # noqa: PLC0415 - POSIX only

        return pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    return pathlib.Path(os.path.expanduser("~")).resolve()


#: The real home directory, which no child process may be given.
REAL_HOME = _real_home()

#: Variables a child must not inherit from the machine running the suite. CI exports
#: MEMVARA_API_KEY for one hosted test, a developer's shell can hold model keys, and a
#: suite started from inside Claude Code carries that session's own variables.
_DROPPED = ("MEMVARA_", "ANTHROPIC_", "OPENAI_", "CLAUDECODE", "CLAUDE_CODE_")


def child_env(home: pathlib.Path, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment for one child process of the suite.

    It starts from this process's environment rather than from an empty one, because on
    Windows a child without SYSTEMROOT cannot open a socket
    (tests/test_hook_recall_requests.py found this). Then:

    * every MEMVARA_, ANTHROPIC_, OPENAI_ and Claude Code variable is removed;
    * HOME and USERPROFILE point at `home`, which must not be the real home directory;
    * PYTHONPATH is this checkout, so the child imports the code under test rather than
      whatever an editable install points at;
    * the store uses the hashing embedder, no encryption and no project read from git,
      the keyring backend is the null one, and the hooks never start their background
      daemon, which would outlive the test.

    `extra` is applied last, so a test can override any of these.
    """
    home = pathlib.Path(home).resolve()
    if home == REAL_HOME:
        raise ValueError(
            f"refusing to start a child process with the real home directory {home}")
    env = {key: value for key, value in os.environ.items() if not key.startswith(_DROPPED)}
    env.update({
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PYTHONPATH": str(REPO),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        "MEMVARA_EMBEDDER": "hashing",
        "MEMVARA_FEATURE_ENCRYPTION": "0",
        "MEMVARA_FEATURE_PROJECT_SCOPE": "0",
        "MEMVARA_DAEMON": "1",
    })
    env.update(extra or {})
    return env
```

`docs/claude/testing.md` (first version; later tasks add sections):

```markdown
# The adversarial test suite

This page is for anyone adding to or running the adversarial suite. The suite tries to break memvara, and it uses memvara the way an agent does: through the real MCP server process, the real hook scripts and a real store, rather than by calling library functions directly. The design and the reasons behind it are in `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`.

The suite lives in two places:

- `tests/harness/` is support code, and it holds no tests.
- `tests/adversarial/` holds the tests. Every file there is named `test_adv_*.py`.

## Child processes

**Every child process the suite starts gets its environment from `harness.env.child_env(home)`.** That function does five things:

- It refuses the real home directory.
- It removes every `MEMVARA_`, `ANTHROPIC_`, `OPENAI_` and Claude Code variable.
- It points `PYTHONPATH` at this checkout.
- It selects the hashing embedder, with encryption and project detection off.
- It stops the hooks from starting their background daemon.

A test that builds its own environment for a child process is the kind of test that once overwrote a developer's real credentials, so do not write one.

**The suite must import the checkout it lives in.** In a git worktree, a stale editable install can make `import memvara` load another copy. `test_adv_env.py` then fails first and explains the fix. There are two ways to fix it:

- run with `PYTHONPATH` set to the checkout;
- create a virtual environment inside the worktree (the `local/` directory is ignored by git) and install the checkout into it in editable mode.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_env.py`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/harness/__init__.py tests/harness/env.py tests/adversarial/__init__.py tests/adversarial/test_adv_env.py docs/claude/testing.md
git commit -m "Add the adversarial test harness and the environment its child processes run in"
```

---

## Task 2: Tiers selected by folder

**Files:**
- Create: `tests/harness/tiers.py`
- Modify: `tests/conftest.py`. Add the `harness` imports after the `encryption_module` import, and add `pytest_addoption` and `pytest_ignore_collect` after the existing `pytest_configure`.
- Create: `tests/adversarial/test_adv_tiers.py`
- Create the tier guards:
  - `tests/adversarial/nightly/__init__.py` and `tests/adversarial/nightly/test_adv_nightly_tier_guard.py`;
  - `tests/adversarial/weekly/__init__.py` and `tests/adversarial/weekly/test_adv_weekly_tier_guard.py`;
  - `tests/adversarial/local/__init__.py` and `tests/adversarial/local/test_adv_local_tier_guard.py`.
- Modify: `docs/claude/testing.md` (add "Tiers").

**Interfaces:**
- Consumes: `harness.env.REPO`.
- Produces:
  - `harness.tiers.TESTS: pathlib.Path`;
  - `TIERS: tuple[str, ...]` and `TIER_DIRS: tuple[str, ...]`;
  - `SELECTS: dict[str, frozenset[str]]`;
  - `tier_of(path: pathlib.Path) -> str`;
  - `ignored(path: pathlib.Path, option: str, *, is_dir: bool | None = None) -> bool`;
  - the pytest option `--tier`, one of `fast|nightly|weekly|local|quarantine`, default `fast`.

- [ ] **Step 1: Write the failing tests**

`tests/adversarial/test_adv_tiers.py`:

```python
"""Tiers: which tests a run collects, decided by the folder a test lives in."""

from __future__ import annotations

import pytest

from harness.env import REPO
from harness.tiers import SELECTS, TESTS, TIER_DIRS, ignored, tier_of


@pytest.mark.parametrize("relative, tier", [
    ("test_api.py", "fast"),
    ("adversarial/test_adv_env.py", "fast"),
    ("adversarial/model/nightly/test_adv_deep.py", "nightly"),
    ("adversarial/sessions/weekly/test_adv_every_switch.py", "weekly"),
    ("adversarial/concurrency/local/test_adv_full_disk.py", "local"),
    ("adversarial/quarantine/test_adv_flaky.py", "quarantine"),
    ("live/agents/test_claude.py", "local"),
])
def test_a_test_files_tier_comes_from_its_folders(relative: str, tier: str) -> None:
    assert tier_of(TESTS / relative) == tier


def test_code_outside_the_tests_folder_is_fast() -> None:
    assert tier_of(REPO / "memvara" / "core.py") == "fast"


def test_the_wider_tiers_include_the_narrower_ones() -> None:
    assert SELECTS["fast"] == {"fast"}
    assert SELECTS["nightly"] == {"fast", "nightly"}
    assert SELECTS["weekly"] == {"fast", "nightly", "weekly"}
    assert SELECTS["local"] == {"local"}
    assert SELECTS["quarantine"] == {"quarantine"}


def test_a_fast_run_leaves_out_every_higher_tier_folder() -> None:
    for name in TIER_DIRS:
        assert ignored(TESTS / "adversarial" / name, "fast", is_dir=True)
    assert ignored(TESTS / "live", "fast", is_dir=True)


def test_a_fast_folder_is_entered_whatever_the_tier() -> None:
    for option in SELECTS:
        assert not ignored(TESTS / "adversarial", option, is_dir=True)


def test_a_local_run_leaves_out_fast_files_but_keeps_package_files() -> None:
    assert ignored(TESTS / "test_api.py", "local", is_dir=False)
    assert ignored(REPO / "memvara" / "core.py", "local", is_dir=False)
    assert not ignored(TESTS / "adversarial" / "__init__.py", "local", is_dir=False)
    assert not ignored(TESTS / "adversarial" / "conftest.py", "local", is_dir=False)


def test_every_tier_folder_is_a_package() -> None:
    """Without __init__.py a tier folder's modules are imported as top-level modules, and
    two test files with the same name in different folders then collide."""
    for directory in (TESTS / "adversarial").rglob("*"):
        if directory.is_dir() and directory.name in TIER_DIRS:
            assert (directory / "__init__.py").is_file(), directory
```

The tier guards. Each one fails if it is ever collected by a tier that should have left it out, so the ordinary fast run is itself the proof that exclusion works.

`tests/adversarial/nightly/__init__.py`, `tests/adversarial/weekly/__init__.py` and `tests/adversarial/local/__init__.py` each contain:

```python
"""Tests in this folder run only when --tier selects it. See docs/claude/testing.md."""
```

`tests/adversarial/nightly/test_adv_nightly_tier_guard.py`:

```python
"""Proof that the nightly folder is collected only when a run selects it."""

import pytest


def test_this_file_runs_only_when_the_nightly_tier_is_selected(
        request: pytest.FixtureRequest) -> None:
    assert request.config.getoption("--tier") in ("nightly", "weekly")
```

`tests/adversarial/weekly/test_adv_weekly_tier_guard.py`:

```python
"""Proof that the weekly folder is collected only when a run selects it."""

import pytest


def test_this_file_runs_only_when_the_weekly_tier_is_selected(
        request: pytest.FixtureRequest) -> None:
    assert request.config.getoption("--tier") == "weekly"
```

`tests/adversarial/local/test_adv_local_tier_guard.py`:

```python
"""Proof that the local folder is collected only when a run selects it."""

import pytest


def test_this_file_runs_only_when_the_local_tier_is_selected(
        request: pytest.FixtureRequest) -> None:
    assert request.config.getoption("--tier") == "local"
```

- [ ] **Step 2: Run to verify they fail**

Run: `local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial`

Expected: FAIL. `test_adv_tiers.py` cannot import `harness.tiers`, and the three guard files are collected and fail with `ValueError: no option named '--tier'`.

- [ ] **Step 3: Write the implementation**

`tests/harness/tiers.py`:

```python
"""Which tier a test belongs to, decided by the folder its file lives in."""

from __future__ import annotations

import pathlib

#: tests/, the folder this package sits in.
TESTS = pathlib.Path(__file__).resolve().parents[1]

#: Every value --tier accepts. fast is the default, and the only tier CI runs.
TIERS = ("fast", "nightly", "weekly", "local", "quarantine")

#: A folder with one of these names puts every test under it in that tier.
TIER_DIRS = ("nightly", "weekly", "local", "quarantine")

#: The tiers each --tier value collects. The first three widen in turn. local and
#: quarantine collect only themselves: local tests need this machine's logins, Docker or
#: transcripts, and quarantined tests are known to be unreliable, so neither run says
#: anything about the fast tier.
SELECTS: dict[str, frozenset[str]] = {
    "fast": frozenset({"fast"}),
    "nightly": frozenset({"fast", "nightly"}),
    "weekly": frozenset({"fast", "nightly", "weekly"}),
    "local": frozenset({"local"}),
    "quarantine": frozenset({"quarantine"}),
}

#: Files that every folder needs whatever the tier. They are never left out.
_ALWAYS = ("__init__.py", "conftest.py")


def _parts(path: pathlib.Path) -> tuple[str, ...] | None:
    """`path` relative to tests/, as parts, or None when it is outside tests/."""
    try:
        return pathlib.Path(path).resolve().relative_to(TESTS).parts
    except ValueError:
        return None


def tier_of(path: pathlib.Path) -> str:
    """The tier of a test file or folder. Anything outside tests/, such as the doctests
    in memvara/, is fast."""
    parts = _parts(path)
    if parts is None:
        return "fast"
    if parts[:1] == ("live",):
        return "local"
    for part in parts:
        if part in TIER_DIRS:
            return part
    return "fast"


def ignored(path: pathlib.Path, option: str, *, is_dir: bool | None = None) -> bool:
    """Whether a run with --tier `option` leaves `path` out of collection.

    A folder is left out only when it is itself a tier folder, or tests/live, that the
    option does not select. Any other folder is entered, because a nightly or local
    folder can sit inside it. A file is left out when its tier is not selected, except
    __init__.py and conftest.py, which every folder needs.
    """
    wanted = SELECTS[option]
    if is_dir is None:
        is_dir = pathlib.Path(path).is_dir()
    if is_dir:
        parts = _parts(path)
        if not parts:
            return False
        if parts == ("live",):
            own: str | None = "local"
        else:
            own = parts[-1] if parts[-1] in TIER_DIRS else None
        return own is not None and own not in wanted
    if pathlib.Path(path).name in _ALWAYS:
        return False
    return tier_of(path) not in wanted
```

`tests/conftest.py`. Add these imports directly after `from memvara.store import encryption as encryption_module`:

```python
# The adversarial suite's tiers and skip ledger (docs/claude/testing.md). `harness` is
# importable here because tests/ has no __init__.py, so pytest puts tests/ on sys.path
# before it imports this file.
from harness import tiers as tiers_module
```

Then add these two hooks directly after the existing `pytest_configure` function:

```python
def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--tier", choices=tiers_module.TIERS, default="fast",
        help="which tier of tests to collect: fast (the default, and what CI runs), "
             "nightly, weekly, local or quarantine. See docs/claude/testing.md.")


def pytest_ignore_collect(collection_path: pathlib.Path, config: Any) -> bool | None:
    """Leave out every test whose tier --tier does not select.

    Returns True or None, never False. pytest stops at the first hook that returns a
    value, so returning False here would overrule --ignore and every other plugin's
    decision about the same path.
    """
    if tiers_module.ignored(collection_path, config.getoption("--tier")):
        return True
    return None
```

Add this section to `docs/claude/testing.md`:

```markdown
## Tiers

A test's tier comes from the folder its file lives in. You do not mark it.

| Folder | Tier | Who runs it |
|---|---|---|
| anything else | fast | every PR, in CI |
| `nightly/` | nightly | the nightly run on the maintainer's Mac |
| `weekly/` | weekly | the weekly run |
| `local/`, and all of `tests/live/` | local | only on a machine with the logins, Docker and transcripts it needs |
| `quarantine/` | quarantine | nobody by default; each test there has an issue |

`--tier` chooses what a run collects:

| Flag | Collects |
|---|---|
| none | fast |
| `--tier nightly` | fast and nightly |
| `--tier weekly` | fast, nightly and weekly |
| `--tier local` | only local |
| `--tier quarantine` | only quarantine |

A tier that a run does not select is left out when pytest collects, so it is never imported and never reported as skipped. A file you name on the command line is always collected, whatever its folder.

**Every tier folder needs an `__init__.py`.** Without one, two test files with the same name in different folders collide. `test_adv_tiers.py` checks this.

The files named `test_adv_*_tier_guard.py` fail if their folder is ever collected by a tier that should have left it out. The ordinary fast run is therefore the proof that the tiers work.
```

- [ ] **Step 4: Run the tests to verify they pass, and that the tiers hold**

```bash
local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial
local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial --tier nightly
local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial --tier weekly
local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial --tier local
```

Expected:

- the default run: `17 passed`, with no guard collected;
- nightly: `18 passed`;
- weekly: `19 passed`;
- local: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/harness/tiers.py tests/conftest.py tests/adversarial/test_adv_tiers.py tests/adversarial/nightly/__init__.py tests/adversarial/nightly/test_adv_nightly_tier_guard.py tests/adversarial/weekly/__init__.py tests/adversarial/weekly/test_adv_weekly_tier_guard.py tests/adversarial/local/__init__.py tests/adversarial/local/test_adv_local_tier_guard.py docs/claude/testing.md
git commit -m "Select test tiers by folder, so slow tests stay out of the default run without showing as skipped"
```

---

## Task 3: The skip ledger

**Files:**
- Create: `tests/harness/skips.py`
- Modify: `tests/conftest.py`. Import `skips_module`, and register the ledger in `pytest_configure`.
- Create: `tests/adversarial/test_adv_skips.py`
- Modify: `docs/claude/testing.md` (add "Skips"), and `CONTRIBUTING.md` (one paragraph under "Running it").

**Interfaces:**
- Consumes: `harness.env.REPO` and `harness.env.child_env`.
- Produces:
  - `harness.skips.SkipRule(pattern: str, why: str)`;
  - `RULES: tuple[SkipRule, ...]`;
  - `explained(reason: str) -> bool`;
  - `reason_of(longrepr: object) -> str`;
  - `SkipLedger`, a pytest plugin object.

- [ ] **Step 1: Write the failing tests**

`tests/adversarial/test_adv_skips.py`:

```python
"""The skip ledger: a skip whose reason no rule explains fails the run."""

from __future__ import annotations

import pathlib
import subprocess
import sys

from harness.env import REPO, child_env
from harness.skips import RULES, explained, reason_of

_PLUGIN = '''
from harness.skips import SkipLedger


def pytest_configure(config):
    config.pluginmanager.register(SkipLedger(), "skip-ledger")
'''


def _run_one_skip(tmp_path: pathlib.Path, reason: str) -> subprocess.CompletedProcess[str]:
    """Run pytest in a child process on one test that skips with `reason`."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "conftest.py").write_text(_PLUGIN, encoding="utf-8")
    (project / "test_one.py").write_text(
        "import pytest\n\n\ndef test_skips():\n"
        f"    pytest.skip({reason!r})\n", encoding="utf-8")
    env = child_env(tmp_path / "home", {"PYTHONPATH": str(REPO / "tests")})
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                          cwd=project, env=env, capture_output=True, text=True, timeout=120)


def test_a_listed_reason_is_explained() -> None:
    assert explained("git is not installed")
    assert explained("could not import 'pypdf': No module named 'pypdf'")


def test_an_unlisted_reason_is_not_explained() -> None:
    assert not explained("flaky on Tuesdays")


def test_the_reason_is_read_without_the_prefix_pytest_adds() -> None:
    assert reason_of(("tests/test_x.py", 3, "Skipped: git is not installed")) == (
        "git is not installed")


def test_every_rule_says_why_its_skip_is_legitimate() -> None:
    for rule in RULES:
        assert rule.why.strip(), rule.pattern


def test_an_unexplained_skip_fails_the_run(tmp_path: pathlib.Path) -> None:
    done = _run_one_skip(tmp_path, "flaky on Tuesdays")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "skips with no rule in tests/harness/skips.py" in done.stdout
    assert "flaky on Tuesdays" in done.stdout


def test_an_explained_skip_leaves_the_run_green(tmp_path: pathlib.Path) -> None:
    done = _run_one_skip(tmp_path, "git is not installed")
    assert done.returncode == 0, done.stdout + done.stderr
```

- [ ] **Step 2: Run to verify they fail**

Run: `local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_skips.py`
Expected: FAIL. `ModuleNotFoundError: No module named 'harness.skips'`.

- [ ] **Step 3: Write the implementation**

`tests/harness/skips.py`. `RULES` must cover every reason in the `SKIPPED` lines of `local/baseline/pytest-ci.txt` from Task 0, plus the CI-only and platform-only reasons below.

```python
"""The skip ledger: every reason a test here may skip, and why that is not hiding a failure.

A skip whose reason no rule below explains fails the run. The ledger exists because most
summaries show a skip as green, so a test that stops running for a new reason looks
exactly like a test that still passes. When you add a skip, add a rule that says why it
is legitimate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass(frozen=True)
class SkipRule:
    #: A regular expression, searched for in the skip reason.
    pattern: str
    #: Why a test that skips for this reason is not hiding a failure.
    why: str


RULES: tuple[SkipRule, ...] = (
    SkipRule(r"^tomllib arrive[sd] in 3\.11",
             "Python 3.10 has no tomllib. These tests run on 3.11 and later."),
    SkipRule(r"^3\.10 only$",
             "The 3.10 half of a pair whose other half needs tomllib."),
    SkipRule(r"^no dist/",
             "The wheel checks run at release time, after python3 -m build --wheel."),
    SkipRule(r"^node/npm missing or unloadable$",
             "These need Node. CI's npm-bridge job runs the bridge's own tests."),
    SkipRule(r"^the renderer is the authority for this test$",
             "Needs memvara-cloud installed, which is a separate repository."),
    SkipRule(r"^no sub-second headroom exists at this platform's ceiling$",
             "A platform limit on timestamps that the test measures before skipping."),
    SkipRule(r"^this platform stores at most \d+ days of history",
             "A platform limit on timestamps that the test measures before skipping."),
    SkipRule(r"^could not import '(openai|pypdf|httpx|zoneinfo)'",
             "An optional extra that this environment did not install."),
    SkipRule(r"^no POSIX permission bits to check$",
             "Windows has no POSIX file modes."),
    SkipRule(r"^Windows file modes do not express this$",
             "Windows has no POSIX file modes."),
    SkipRule(r"^the hooks' copy of the vectors is not in this checkout yet$",
             "A packaging state that the test detects before skipping."),
    SkipRule(r"^git is not installed$",
             "Project detection needs git."),
    SkipRule(r"^mypy is not installed",
             "The type-level tests need mypy. CI's types job has it."),
    SkipRule(r"^deployment not asked, tool surface NOT checked",
             "Needs MEMVARA_API_KEY to ask the hosted deployment. CI provides it."),
    SkipRule(r"^needs the encrypt extra",
             "The encrypt extra is optional. CI installs it."),
    SkipRule(r"^no system tz database and no tzdata package$",
             "The time zone tests need a tz database."),
)


def reason_of(longrepr: object) -> str:
    """The reason text of a skipped report, without the "Skipped: " prefix pytest adds."""
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        text = str(longrepr[2])
    else:
        text = str(longrepr)
    return text.removeprefix("Skipped: ")


def explained(reason: str) -> bool:
    """Whether some rule in RULES covers `reason`."""
    return any(re.search(rule.pattern, reason) for rule in RULES)


class SkipLedger:
    """A pytest plugin that fails the session when a skip has no rule in RULES.

    An expected failure (xfail) is reported as skipped too. It is not a skip, and it is
    ignored here.
    """

    def __init__(self) -> None:
        self.unexplained: list[str] = []

    def _record(self, report: Any) -> None:
        if not report.skipped or hasattr(report, "wasxfail"):
            return
        reason = reason_of(report.longrepr)
        if not explained(reason):
            self.unexplained.append(f"{report.nodeid}: {reason}")

    def pytest_runtest_logreport(self, report: Any) -> None:
        self._record(report)

    def pytest_collectreport(self, report: Any) -> None:
        self._record(report)

    def pytest_sessionfinish(self, session: Any) -> None:
        if not self.unexplained:
            return
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_sep("=", "skips with no rule in tests/harness/skips.py", red=True)
            for line in self.unexplained:
                reporter.write_line(line)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
```

`tests/conftest.py`. Extend the harness import block from Task 2:

```python
from harness import skips as skips_module
from harness import tiers as tiers_module
```

Then add this line at the end of `pytest_configure`:

```python
    config.pluginmanager.register(skips_module.SkipLedger(), "memvara-skip-ledger")
```

Add this section to `docs/claude/testing.md`:

```markdown
## Skips

**A skip needs a rule.** Every skip reason must match a rule in `tests/harness/skips.py`, and each rule says why that skip hides no failure. A skip with no matching rule fails the whole run, and the run lists the test and its reason.

The ledger exists because most summaries show a skip as green, so a test that stops running for a new reason looks exactly like one that passes.

An expected failure (xfail) is not a skip, and the ledger ignores it.
```

`CONTRIBUTING.md`. Add this paragraph directly after the paragraph that begins "`[dev]` is pytest":

```markdown
**Every skip needs a rule.** A test that skips for a reason with no rule in `tests/harness/skips.py` fails the run. So when you add a skip, add a rule that says why the skip is legitimate. `docs/claude/testing.md` explains the rule, and the test tiers that keep slow tests out of the default run without skipping them.
```

- [ ] **Step 4: Run the tests to verify they pass, then check the whole suite against the ledger**

```bash
local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_skips.py
local/venv-ci/bin/python -m pytest -q -rs -p no:cacheprovider
```

Expected:

- the first command: `6 passed`;
- the full run: the same pass count as the Task 0 baseline plus the new tests, and no "skips with no rule" section.

If that section appears, add a rule for each listed reason, but only after reading its test and confirming the skip is legitimate.

- [ ] **Step 5: Commit**

```bash
git add tests/harness/skips.py tests/conftest.py tests/adversarial/test_adv_skips.py docs/claude/testing.md CONTRIBUTING.md
git commit -m "Fail the test run when a test skips for a reason nobody has explained"
```

---

## Task 4: Hypothesis and its profiles

**Files:**
- Modify: `pyproject.toml`. Change the `dev` extra to:

  ```toml
  dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "coverage>=7.0", "mypy>=1.8", "hypothesis>=6.100"]
  ```

- Modify: `tests/harness/tiers.py` (add `HYPOTHESIS_PROFILE_FOR` and `load_hypothesis_profile`).
- Modify: `tests/conftest.py` (load the profile in `pytest_configure`).
- Modify: `.gitignore` (add `.hypothesis/`).
- Modify: `CONTRIBUTING.md` (the `[dev]` sentence).
- Create: `tests/adversarial/test_adv_hypothesis.py`
- Modify: `docs/claude/testing.md` (add "Property-based tests").

**Interfaces:**
- Consumes: `harness.tiers.SELECTS` and `harness.skips.explained`.
- Produces:
  - `harness.tiers.HYPOTHESIS_PROFILE_FOR: dict[str, str]`;
  - `load_hypothesis_profile(option: str) -> None`;
  - the registered profiles `memvara-fast`, `memvara-nightly` and `memvara-weekly`.

- [ ] **Step 1: Write the failing tests**

`tests/adversarial/test_adv_hypothesis.py`:

```python
"""Hypothesis runs under the profile of the selected tier."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from harness.skips import explained
from harness.tiers import HYPOTHESIS_PROFILE_FOR


def test_hypothesis_runs_the_profile_of_the_selected_tier(
        request: pytest.FixtureRequest) -> None:
    expected = settings.get_profile(HYPOTHESIS_PROFILE_FOR[request.config.getoption("--tier")])
    assert settings.default.max_examples == expected.max_examples
    assert settings.default.derandomize == expected.derandomize


def test_the_fast_profile_is_repeatable_and_keeps_no_database() -> None:
    fast = settings.get_profile("memvara-fast")
    assert fast.derandomize is True
    assert fast.database is None


@given(st.text())
def test_the_skip_ledger_answers_for_any_reason_text(reason: str) -> None:
    assert explained(reason) in (True, False)
```

- [ ] **Step 2: Run to verify they fail**

Run: `local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_hypothesis.py`
Expected: FAIL. `ImportError: cannot import name 'HYPOTHESIS_PROFILE_FOR'`.

- [ ] **Step 3: Write the implementation**

Append to `tests/harness/tiers.py`:

```python
#: The Hypothesis profile each tier runs under.
HYPOTHESIS_PROFILE_FOR: dict[str, str] = {
    "fast": "memvara-fast",
    "nightly": "memvara-nightly",
    "weekly": "memvara-weekly",
    "local": "memvara-nightly",
    "quarantine": "memvara-nightly",
}


def load_hypothesis_profile(option: str) -> None:
    """Register the suite's Hypothesis profiles, and load the one for --tier `option`.

    The fast profile is derandomized and keeps no example database, so a PR run is
    repeatable. The nightly and weekly profiles run far more examples. They keep what
    they find in ~/.cache/memvara-adversarial/hypothesis, so a failure found one night is
    tried first the next night. Hypothesis is imported here rather than at the top of
    the module, so this module stays importable in an environment without the dev extra.
    """
    try:
        from hypothesis import HealthCheck, settings  # noqa: PLC0415
        from hypothesis.database import DirectoryBasedExampleDatabase  # noqa: PLC0415
    except ImportError:
        return
    quiet = [HealthCheck.too_slow]
    settings.register_profile(
        "memvara-fast", max_examples=30, stateful_step_count=25, derandomize=True,
        database=None, deadline=None, print_blob=True, suppress_health_check=quiet)
    database = DirectoryBasedExampleDatabase(
        str(pathlib.Path.home() / ".cache" / "memvara-adversarial" / "hypothesis"))
    settings.register_profile(
        "memvara-nightly", max_examples=3000, stateful_step_count=100, database=database,
        deadline=None, print_blob=True, suppress_health_check=quiet)
    settings.register_profile(
        "memvara-weekly", parent=settings.get_profile("memvara-nightly"),
        max_examples=20000, stateful_step_count=200)
    settings.load_profile(HYPOTHESIS_PROFILE_FOR[option])
```

`tests/conftest.py`. Add this line at the end of `pytest_configure`:

```python
    tiers_module.load_hypothesis_profile(config.getoption("--tier", default="fast"))
```

`.gitignore`. Append:

```gitignore
# Hypothesis keeps a cache in the directory it runs from. The adversarial suite's nightly
# runs keep their example database under ~/.cache instead, so nothing here is worth
# keeping. Leading slash: this is the one directory at the root.
/.hypothesis/
```

`CONTRIBUTING.md`. Replace the sentence beginning "`[dev]` is pytest, pytest-asyncio, coverage and mypy — no provider SDKs." with:

```markdown
`[dev]` is pytest, pytest-asyncio, coverage, mypy and Hypothesis — no provider SDKs.
```

Add this section to `docs/claude/testing.md`:

```markdown
## Property-based tests

Hypothesis runs under the profile of the selected tier.

| Tier | Examples | Behaviour |
|---|---|---|
| fast | 30 per test, 25 steps per state machine | Derandomized, with no example database, so a PR run gives the same answer every time |
| nightly | 3,000 | Keeps the examples it finds in `~/.cache/memvara-adversarial/hypothesis`, so a failure found one night is tried first the next night |
| weekly | 20,000 | Same database as nightly |

When a property test fails, Hypothesis prints a reproduction blob. Put it in a `@reproduce_failure` decorator to replay the exact case.
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_hypothesis.py
local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_hypothesis.py --tier nightly
```

Expected: `3 passed` for each. The profile test checks a different profile in each run.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/harness/tiers.py tests/conftest.py .gitignore CONTRIBUTING.md tests/adversarial/test_adv_hypothesis.py docs/claude/testing.md
git commit -m "Add Hypothesis to the dev extra, with a repeatable profile for PRs and larger ones for nightly runs"
```

---

## Task 5: A real MCP server process (`McpProcess`)

**Files:**
- Create: `tests/harness/stdio.py`
- Create: `tests/adversarial/conftest.py` (the `mcp` fixture)
- Create: `tests/adversarial/test_adv_stdio.py`
- Modify: `docs/claude/testing.md` (add "The MCP server, in its own process")

**Interfaces:**
- Consumes: `harness.env.child_env`, and `memvara.server.config.FEATURES`.
- Produces:
  - `harness.stdio.PROTOCOL = "2025-06-18"`;
  - `McpProcessError(RuntimeError)`;
  - `RpcError(RuntimeError)`, with `.code: int` and `.message: str`;
  - `ToolResult(text: str, is_error: bool, raw: dict)`;
  - `McpProcess(db, *, home, user="tester", scope=None, features=None, read_only=False, env=None, cwd=None, timeout=30.0)`, with these methods:
    - `.initialize(protocol=PROTOCOL) -> dict` and `.list_tools() -> list[dict]`;
    - `.call(name, /, **arguments) -> ToolResult`;
    - `.request(method, params=None) -> dict` and `.notify(method, params=None) -> None`;
    - `.send_raw(data: str | bytes) -> None` and `.recv(timeout=None) -> dict`;
    - `.alive() -> bool`, `.kill() -> None` and `.close(timeout=10.0) -> int`;
    - `.stderr_text() -> str` and `.transcript: list[tuple[str, str]]`;
  - the fixture `mcp(db=None, **kwargs) -> McpProcess`, which kills every server it started when the test ends.

- [ ] **Step 1: Write the failing tests**

`tests/adversarial/conftest.py`:

```python
"""Fixtures shared by the adversarial suite."""

from __future__ import annotations

import pathlib
from typing import Any, Callable, Iterator

import pytest

from harness.stdio import McpProcess


@pytest.fixture
def mcp(tmp_path: pathlib.Path,
        tmp_path_factory: pytest.TempPathFactory) -> Iterator[Callable[..., McpProcess]]:
    """Start real memvara MCP servers. By default each one opens memory.db in tmp_path.

    Every server this fixture started is killed when the test ends, before pytest deletes
    the temporary directory. On Windows, a file that a live process holds open cannot be
    deleted.
    """
    home = tmp_path_factory.mktemp("mcp-home")
    started: list[McpProcess] = []

    def start(db: pathlib.Path | None = None, **options: Any) -> McpProcess:
        server = McpProcess(db or tmp_path / "memory.db", home=home, **options)
        started.append(server)
        return server

    yield start
    for server in started:
        server.kill()
```

`tests/adversarial/test_adv_stdio.py`:

```python
"""McpProcess drives a real `python -m memvara.server` over its stdio pipe."""

from __future__ import annotations

import pathlib
import time
from typing import Callable

import pytest

from harness.stdio import PROTOCOL, McpProcess, McpProcessError

Start = Callable[..., McpProcess]


def test_a_real_server_process_remembers_and_recalls_over_its_pipe(mcp: Start) -> None:
    server = mcp()
    hello = server.initialize()
    assert hello["protocolVersion"] == PROTOCOL
    assert hello["serverInfo"]["name"] == "memvara"
    names = {tool["name"] for tool in server.list_tools()}
    assert {"memory_remember", "memory_recall"} <= names
    stored = server.call("memory_remember", subject="user", predicate="prefers",
                         object="tabs for indentation", memory_type="procedural")
    assert not stored.is_error, stored.text
    recalled = server.call("memory_recall", query="how does the user indent code")
    assert "tabs for indentation" in recalled.text
    assert server.close() == 0


def test_the_store_outlives_the_server_process(mcp: Start, tmp_path: pathlib.Path) -> None:
    db = tmp_path / "shared.db"
    first = mcp(db)
    first.initialize()
    assert not first.call("memory_remember", subject="user", predicate="lives_in",
                          object="Lisbon").is_error
    assert first.close() == 0
    second = mcp(db)
    second.initialize()
    assert "Lisbon" in second.call("memory_history", subject="user",
                                   predicate="lives_in").text


def test_a_dead_server_is_reported_instead_of_waited_on(mcp: Start) -> None:
    server = mcp()
    server.initialize()
    server.kill()
    with pytest.raises(McpProcessError):
        server.request("ping")


def test_a_silent_server_times_out_instead_of_hanging(mcp: Start) -> None:
    server = mcp()
    server.initialize()
    started = time.monotonic()
    with pytest.raises(McpProcessError, match="no message from the server"):
        server.recv(timeout=0.5)
    assert time.monotonic() - started < 5


def test_an_unknown_feature_is_refused_before_a_server_starts(mcp: Start) -> None:
    with pytest.raises(ValueError, match="unknown feature"):
        mcp(features={"no_such_feature": True})


def test_a_switched_off_feature_hides_its_tools(mcp: Start) -> None:
    server = mcp(features={"documents": False})
    server.initialize()
    names = {tool["name"] for tool in server.list_tools()}
    assert "memory_add_document" not in names
    assert "memory_remember" in names


def test_a_read_only_server_lists_only_read_only_tools(mcp: Start) -> None:
    server = mcp(read_only=True)
    server.initialize()
    specs = server.list_tools()
    assert specs
    assert all(spec["annotations"]["readOnlyHint"] for spec in specs)
```

- [ ] **Step 2: Run to verify they fail**

Run: `local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_stdio.py`
Expected: FAIL. `ModuleNotFoundError: No module named 'harness.stdio'`.

- [ ] **Step 3: Write the implementation**

`tests/harness/stdio.py`:

```python
"""A real memvara MCP server in its own process, driven over its stdio pipe.

tests/test_server.py calls MemvaraMCPServer.handle_line() in-process. This module tests
what that cannot reach: the process an agent's client actually starts. That process reads
its configuration from the environment, frames messages on a real pipe, and exits when the
client closes stdin.
"""

from __future__ import annotations

import json
import os
import pathlib
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping

from memvara.server.config import FEATURES

from .env import child_env

#: The protocol version a client asks for when a test does not name one.
PROTOCOL = "2025-06-18"

_SCOPE_FIELDS = ("tenant", "project", "agent", "session")


class McpProcessError(RuntimeError):
    """The server exited, stopped reading, or wrote nothing within the timeout."""


class RpcError(RuntimeError):
    """The server answered a request with a JSON-RPC error object."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"JSON-RPC error {code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolResult:
    """One tools/call result: the text of its content blocks, and its error flag."""

    text: str
    is_error: bool
    raw: dict[str, Any] = field(repr=False)


class McpProcess:
    """`python -m memvara.server` in a child process, spoken to one JSON line at a time.

    `db` is the store the server opens (MEMVARA_DB). `home` becomes the child's HOME and
    must not be the real one. `user` binds MEMVARA_USER. `scope` can bind tenant,
    project, agent and session. `features` switches named features on or off. A feature
    name the server does not know is refused here, because the server would refuse to
    start with it. `env` is applied last.
    """

    def __init__(self, db: str | os.PathLike[str], *, home: pathlib.Path,
                 user: str | None = "tester", scope: Mapping[str, str] | None = None,
                 features: Mapping[str, bool] | None = None, read_only: bool = False,
                 env: Mapping[str, str] | None = None, cwd: pathlib.Path | None = None,
                 timeout: float = 30.0) -> None:
        extra: dict[str, str] = {"MEMVARA_DB": str(db)}
        if user is not None:
            extra["MEMVARA_USER"] = user
        for key, value in (scope or {}).items():
            if key not in _SCOPE_FIELDS:
                raise ValueError(f"unknown scope field {key!r}; use one of {_SCOPE_FIELDS}")
            extra[f"MEMVARA_{key.upper()}"] = value
        for name, on in (features or {}).items():
            if name not in FEATURES:
                raise ValueError(
                    f"unknown feature {name!r}; the server would refuse to start with it")
            extra[f"MEMVARA_FEATURE_{name.upper()}"] = "1" if on else "0"
        if read_only:
            extra["MEMVARA_READ_ONLY"] = "1"
        extra.update(env or {})

        self.timeout = timeout
        #: Every line written ("->") and read ("<-"), in order.
        self.transcript: list[tuple[str, str]] = []
        self._next_id = 1
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._stderr: list[bytes] = []
        options: dict[str, Any] = {}
        if sys.platform == "win32":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "memvara.server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=child_env(home, extra), cwd=str(cwd or home), **options)
        self._readers = [threading.Thread(target=self._pump_stdout, daemon=True),
                         threading.Thread(target=self._pump_stderr, daemon=True)]
        for reader in self._readers:
            reader.start()

    # -- the pipe ------------------------------------------------------------

    def _pump_stdout(self) -> None:
        stream = self.proc.stdout
        assert stream is not None
        for raw in iter(stream.readline, b""):
            self._lines.put(raw)
        self._lines.put(None)  # the end of the stream

    def _pump_stderr(self) -> None:
        stream = self.proc.stderr
        assert stream is not None
        for raw in iter(stream.readline, b""):
            self._stderr.append(raw)

    def stderr_text(self) -> str:
        """Everything the server has written to stderr so far."""
        return b"".join(self._stderr).decode("utf-8", "replace")

    def alive(self) -> bool:
        return self.proc.poll() is None

    def _dead(self, what: str) -> McpProcessError:
        try:
            code: int | None = self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            code = None
        return McpProcessError(
            f"{what} (exit code {code}); stderr: {self.stderr_text()[-800:]!r}")

    def send_raw(self, data: str | bytes) -> None:
        """Write `data` and one newline, exactly as given, with no framing checks."""
        payload = data.encode("utf-8") if isinstance(data, str) else data
        self.transcript.append(("->", payload.decode("utf-8", "replace")))
        stream = self.proc.stdin
        assert stream is not None
        try:
            stream.write(payload + b"\n")
            stream.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise self._dead(f"could not write to the server: {exc}") from exc

    def recv(self, timeout: float | None = None) -> dict[str, Any]:
        """The next message the server writes. Raises McpProcessError when none arrives."""
        wait = self.timeout if timeout is None else timeout
        try:
            raw = self._lines.get(timeout=wait)
        except queue.Empty:
            raise McpProcessError(
                f"no message from the server within {wait}s; "
                f"stderr: {self.stderr_text()[-800:]!r}") from None
        if raw is None:
            self._lines.put(None)  # later calls must see the end of the stream too
            raise self._dead("the server closed its output")
        line = raw.decode("utf-8")
        self.transcript.append(("<-", line.rstrip("\r\n")))
        message = json.loads(line)
        if not isinstance(message, dict):
            raise McpProcessError(f"the server wrote a message that is not an object: "
                                  f"{line[:200]!r}")
        return message

    # -- JSON-RPC ------------------------------------------------------------

    def request(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Send one request, wait for the reply with its id, and return its result."""
        request_id = self._next_id
        self._next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = dict(params)
        self.send_raw(json.dumps(message))
        while True:
            reply = self.recv()
            if reply.get("id") == request_id:
                break
        if "error" in reply:
            raise RpcError(int(reply["error"]["code"]), str(reply["error"]["message"]))
        result = reply["result"]
        assert isinstance(result, dict)
        return result

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        """Send one notification, which by definition gets no reply."""
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = dict(params)
        self.send_raw(json.dumps(message))

    def initialize(self, protocol: str = PROTOCOL) -> dict[str, Any]:
        """The opening handshake a client performs, followed by `initialized`."""
        result = self.request("initialize", {
            "protocolVersion": protocol, "capabilities": {},
            "clientInfo": {"name": "memvara-adversarial-suite", "version": "0"}})
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self.request("tools/list")["tools"])

    def call(self, name: str, /, **arguments: Any) -> ToolResult:
        """Call one tool. A tool that ran and failed comes back with is_error set."""
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        text = "".join(str(block.get("text", "")) for block in result.get("content", []))
        return ToolResult(text=text, is_error=bool(result.get("isError")), raw=result)

    # -- ending it -----------------------------------------------------------

    def kill(self) -> None:
        """Stop the server at once: SIGKILL on POSIX, TerminateProcess on Windows."""
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=10)
        self._finish()

    def close(self, timeout: float = 10.0) -> int:
        """Close stdin, as a client does at the end of a session, and return the exit code."""
        stream = self.proc.stdin
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except OSError:
                pass
        try:
            code = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            raise McpProcessError(
                f"the server did not exit within {timeout}s of its input closing") from None
        self._finish()
        return code

    def _finish(self) -> None:
        for reader in self._readers:
            reader.join(timeout=5)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass

    def __enter__(self) -> McpProcess:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.kill()
```

Add this section to `docs/claude/testing.md`:

```markdown
## The MCP server, in its own process

`harness.stdio.McpProcess` starts `python -m memvara.server` as a child process and speaks newline-delimited JSON-RPC to it, the way an agent's client does. Nothing about the server is faked: it imports this checkout, opens a real SQLite store, and reads its configuration from the environment.

To use it:

- **In a test,** use the `mcp` fixture. `mcp()` opens `memory.db` in the test's temporary directory, and `mcp(path)` opens a store you name.
- **Scope and switches.** Pass `features={"documents": False}` to switch a feature off, `read_only=True` for a read-only server, and `scope={"session": "s1"}` to bind a scope field.
- **Calling tools.** `call(name, **arguments)` returns the tool's text and its error flag.
- **Raw input.** `send_raw` and `recv` send and read arbitrary lines, for protocol tests.

Failures are loud and quick:

- A server that exits raises `McpProcessError`, with its exit code and the tail of its stderr.
- A server that writes nothing within the timeout raises the same error, instead of hanging the suite.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_stdio.py --durations=5`
Expected: `7 passed`. Each test takes about one to two seconds, mostly Python start-up.

- [ ] **Step 5: Commit**

```bash
git add tests/harness/stdio.py tests/adversarial/conftest.py tests/adversarial/test_adv_stdio.py docs/claude/testing.md
git commit -m "Drive a real memvara MCP server process over its stdio pipe in the adversarial suite"
```

---

## Task 6: The hook runner, and in-process stores

**Files:**
- Create: `tests/harness/hooks.py`
- Create: `tests/harness/stores.py`
- Modify: `tests/adversarial/conftest.py` (add the `hook_runner` fixture)
- Create: `tests/adversarial/test_adv_hooks.py`
- Modify: `docs/claude/testing.md` (add "Hooks" and "Stores in the test process")

**Interfaces:**
- Consumes: `harness.env.REPO` and `harness.env.child_env`, plus `memvara.Memvara`, `memvara.NullLLM`, `memvara.MemoryType` and `memvara.embed.HashingEmbedder`.
- Produces:
  - `harness.hooks.HOOKS_DIR` and `harness.hooks.RUN`;
  - `HookResult(exit_code, stdout, stderr, reply, elapsed)`;
  - `HookOutputError(AssertionError)`;
  - `host_record(host) -> Any`;
  - `parse_reply(stdout, *, what) -> dict | None`;
  - `HookRunner(host, *, home, cwd, server_env=None, env=None)`, with `.payload(hook, **fields) -> dict`, `.run(hook, *, stdin=None, timeout=None, **fields) -> HookResult` and `.write_client_config(server_env) -> Path`;
  - `harness.stores.memory(**options) -> Memvara` and `harness.stores.file(path, **options) -> Memvara`;
  - the fixture `hook_runner(host, **kwargs) -> HookRunner`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/adversarial/conftest.py`:

```python
from harness.hooks import HookRunner  # noqa: E402 - kept beside the fixture that uses it


@pytest.fixture
def hook_runner(tmp_path: pathlib.Path,
                tmp_path_factory: pytest.TempPathFactory) -> Callable[..., HookRunner]:
    """Build HookRunners that share one scratch home and one working directory."""
    home = tmp_path_factory.mktemp("hook-home")
    work = tmp_path / "work"
    work.mkdir()

    def make(host: str, **options: Any) -> HookRunner:
        return HookRunner(host, home=home, cwd=work, **options)

    return make
```

`tests/adversarial/test_adv_hooks.py`:

```python
"""HookRunner runs the plugin's real hook scripts, the way a client does."""

from __future__ import annotations

import pathlib
from typing import Callable

import pytest

from harness import stores
from harness.hooks import HookOutputError, HookRunner, host_record, parse_reply
from memvara import MemoryType

Make = Callable[..., HookRunner]
HOSTS = ("claude", "codex", "copilot", "cursor", "opencode")


@pytest.mark.parametrize("host", HOSTS)
def test_every_host_gives_each_of_its_hooks_a_timeout(host: str) -> None:
    record = host_record(host)
    for hook in record.events:
        assert record.timeouts[hook] > 0, (host, hook)


def test_session_start_without_a_store_says_not_configured(hook_runner: Make) -> None:
    result = hook_runner("claude").run("session_start")
    assert result.exit_code == 0
    assert result.reply is not None
    assert "not configured" in result.reply["systemMessage"]


def test_session_start_reads_the_store_the_client_config_names(
        hook_runner: Make, tmp_path: pathlib.Path) -> None:
    db = tmp_path / "memory.db"
    mem = stores.file(db)
    mem.scope(user="tester").remember("user", "prefers", "tabs for indentation",
                                      memory_type=MemoryType.PROCEDURAL)
    mem.close()
    runner = hook_runner("claude", server_env={"MEMVARA_DB": str(db), "MEMVARA_USER": "tester"})
    result = runner.run("session_start")
    assert result.exit_code == 0
    assert result.reply is not None
    assert "session opened with" in result.reply["systemMessage"]
    assert "tabs for indentation" in result.reply["hookSpecificOutput"]["additionalContext"]


def test_the_approve_hook_allows_a_read_only_memvara_tool(hook_runner: Make) -> None:
    result = hook_runner("claude").run("approve", tool_name="mcp__memvara__memory_search")
    assert result.exit_code == 0
    assert result.reply is not None
    assert result.reply["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_the_approve_hook_says_nothing_about_a_write_tool(hook_runner: Make) -> None:
    result = hook_runner("claude").run("approve", tool_name="mcp__memvara__memory_forget")
    assert result.exit_code == 0
    assert result.reply is None


def test_output_that_is_not_json_is_reported_with_its_text() -> None:
    with pytest.raises(HookOutputError, match="Traceback"):
        parse_reply("Traceback (most recent call last): boom", what="recall on claude")
    assert parse_reply("", what="recall on claude") is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_hooks.py`
Expected: FAIL. `ModuleNotFoundError: No module named 'harness.hooks'`.

- [ ] **Step 3: Write the implementation**

`tests/harness/stores.py`:

```python
"""Stores in the test process, built the way every test in this repository builds one:
with the hashing embedder, and with no model."""

from __future__ import annotations

import pathlib
from typing import Any

from memvara import Memvara, NullLLM
from memvara.embed import HashingEmbedder


def memory(**options: Any) -> Memvara:
    """An in-memory store."""
    return Memvara(embedder=HashingEmbedder(dim=512), llm=NullLLM(), **options)


def file(path: pathlib.Path, **options: Any) -> Memvara:
    """A store in the SQLite file at `path`. A server started with MEMVARA_DB=`path` and
    the child environment's MEMVARA_EMBEDDER=hashing opens it in the same vector space."""
    return Memvara(str(path), embedder=HashingEmbedder(dim=512), llm=NullLLM(), **options)
```

`tests/harness/hooks.py`:

```python
"""The plugin's hook scripts, run in a child process the way a client runs them."""

from __future__ import annotations

import importlib
import json
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .env import REPO, child_env

HOOKS_DIR = REPO / "plugin" / "hooks"
RUN = HOOKS_DIR / "run.py"


class HookOutputError(AssertionError):
    """A hook printed something that is not JSON. On a real client that desynchronises
    the conversation, so it is a failure in its own right."""


@dataclass(frozen=True)
class HookResult:
    """What one hook run did."""

    exit_code: int
    stdout: str
    stderr: str
    #: stdout parsed as JSON, or None when the hook printed nothing.
    reply: dict[str, Any] | None
    #: Wall-clock seconds, including Python start-up.
    elapsed: float


def host_record(host: str) -> Any:
    """The Host record that plugin/hooks/hosts/<host>.py defines."""
    if str(HOOKS_DIR) not in sys.path:
        sys.path.insert(0, str(HOOKS_DIR))
    return importlib.import_module(f"hosts.{host}").HOST


def parse_reply(stdout: str, *, what: str, stderr: str = "") -> dict[str, Any] | None:
    """A hook's stdout as JSON, or None when it printed nothing.

    Output that is not JSON raises HookOutputError, with the output and stderr attached.
    """
    text = stdout.strip()
    if not text:
        return None
    try:
        reply = json.loads(text)
    except ValueError:
        raise HookOutputError(f"{what} printed text that is not JSON: {text[:300]!r}; "
                              f"stderr: {stderr[-300:]!r}") from None
    if not isinstance(reply, dict):
        raise HookOutputError(f"{what} printed JSON that is not an object: {text[:300]!r}")
    return reply


class HookRunner:
    """Runs `plugin/hooks/run.py <hook> --host <host>` with a payload shaped for that host.

    `home` becomes the child's HOME, and `cwd` its working directory. When `server_env`
    is given, it is written into the host's first client config file as the memvara
    server's env block, which is where the hooks look for the store
    (plugin/hooks/lib/ipc.py). Without it, the hooks find no store and report
    "not configured".
    """

    def __init__(self, host: str, *, home: pathlib.Path, cwd: pathlib.Path,
                 server_env: Mapping[str, str] | None = None,
                 env: Mapping[str, str] | None = None) -> None:
        self.host = host_record(host)
        self.home = pathlib.Path(home)
        self.cwd = pathlib.Path(cwd)
        self._env = child_env(self.home, env)
        if server_env is not None:
            self.write_client_config(server_env)

    def write_client_config(self, server_env: Mapping[str, str]) -> pathlib.Path:
        """Write the host's first client config file, holding a memvara server block."""
        if self.host.config_format != "json":
            raise NotImplementedError(
                f"{self.host.id} keeps a {self.host.config_format} client config")
        path = pathlib.Path(str(self.host.client_configs[0]).replace("~", str(self.home), 1))
        path.parent.mkdir(parents=True, exist_ok=True)
        block = {"command": sys.executable, "args": ["-m", "memvara.server"],
                 "env": dict(server_env)}
        path.write_text(json.dumps({"mcpServers": {"memvara": block}}), encoding="utf-8")
        return path

    def payload(self, hook: str, **fields: Any) -> dict[str, Any]:
        """The stdin this host sends for `hook`.

        Each keyword names an Event field (session, cwd, prompt, transcript_path or
        tool_name), and it is written under this host's own key for that field. When a
        host accepts several keys, the last one is used, because the earlier ones are
        richer shapes that tests of that host build themselves. Cursor's
        workspace_roots is a list, for example.
        """
        event = self.host.events.get(hook)
        if event is None:
            raise ValueError(f"{self.host.id} has no event for the {hook} hook")
        body: dict[str, Any] = {"hook_event_name": event}
        values = {"session": "adversarial-session", "cwd": str(self.cwd), **fields}
        for name, value in values.items():
            keys = self.host.fields.get(name)
            if not keys:
                raise ValueError(f"{self.host.id} has no stdin key for the {name} field")
            body[keys[-1]] = value
        return body

    def run(self, hook: str, *, stdin: str | None = None, timeout: float | None = None,
            **fields: Any) -> HookResult:
        """Run one hook and wait for it, within this host's own timeout for that hook."""
        text = json.dumps(self.payload(hook, **fields)) if stdin is None else stdin
        limit = float(self.host.timeouts[hook]) if timeout is None else timeout
        started = time.monotonic()
        done = subprocess.run(
            [sys.executable, str(RUN), hook, "--host", self.host.id], input=text,
            capture_output=True, text=True, encoding="utf-8", env=self._env,
            cwd=str(self.cwd), timeout=limit)
        elapsed = time.monotonic() - started
        reply = parse_reply(done.stdout, what=f"{hook} on {self.host.id}", stderr=done.stderr)
        return HookResult(exit_code=done.returncode, stdout=done.stdout, stderr=done.stderr,
                          reply=reply, elapsed=elapsed)
```

Add these sections to `docs/claude/testing.md`:

```markdown
## Hooks

`harness.hooks.HookRunner` runs `plugin/hooks/run.py <hook> --host <host>` in a child process, with the stdin payload that host sends.

- **In a test,** use the `hook_runner` fixture: `hook_runner("claude").run("approve", tool_name="mcp__memvara__memory_search")`.
- **Giving the hooks a store.** Pass `server_env={"MEMVARA_DB": ..., "MEMVARA_USER": ...}` and the runner writes the host's client config, which is where the hooks look for the store. Without it, the hooks report "not configured".
- **What a run returns:** the exit code, the parsed reply and the elapsed time.
- **Non-JSON output fails the test.** A hook that prints something other than JSON raises `HookOutputError`, because on a real client that output would desynchronise the conversation.

## Stores in the test process

`harness.stores.memory()` gives an in-memory store and `harness.stores.file(path)` a SQLite file. Both use the hashing embedder and no model, as every other test here does.

A server started on the same file with the child environment opens it in the same vector space. So a test can write through the library and then read through the server, or the other way round.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_hooks.py`
Expected: `10 passed`. That is five host-timeout cases plus five others.

- [ ] **Step 5: Commit**

```bash
git add tests/harness/hooks.py tests/harness/stores.py tests/adversarial/conftest.py tests/adversarial/test_adv_hooks.py docs/claude/testing.md
git commit -m "Run the plugin's real hook scripts from the adversarial suite, the way a client does"
```

---

## Task 7: Wire the harness into CI and the docs index

**Files:**
- Modify: `.github/workflows/ci.yml`. In the `types` job, after the `benchmarks/agent_memory` mypy step, add:

  ```yaml
      # The adversarial suite's harness is shared by every tier, and a wrong signature there
      # breaks tests that only the nightly run collects. Checking it here catches that on
      # the PR that introduces it.
      - run: python -m mypy tests/harness --ignore-missing-imports
  ```

- Modify: `docs/claude/README.md`. Add a row to the table, after the "Counters, benchmark scripts" row. Its three cells:
  - "The adversarial test suite: tiers, the harness, the skip ledger, known bugs";
  - a markdown link whose text is `testing.md` and whose target is `testing.md`, in the same form as the links in the other rows (it is not written out here, because `tests/test_doc_links.py` would check it relative to this plan's folder);
  - `` `tests/harness/`, `tests/adversarial/` ``.

- Modify: `CLAUDE.md`. Add this row to the table in "Where the rest of the context lives", after the "Counters, benchmark scripts, the demo harness" row:

  ```markdown
  | The adversarial test suite: tiers, the harness, the skip ledger, known bugs | `docs/claude/testing.md` | `tests/harness/`, `tests/adversarial/` |
  ```

- Modify: `docs/claude/testing.md` (add "Known bugs and security findings").

- [ ] **Step 1: Run mypy on the harness to see its current state**

Run: `local/venv-ci/bin/python -m mypy tests/harness --ignore-missing-imports`
Expected: `Success: no issues found in 7 source files`. If mypy reports errors, fix the annotations in the named file, not the call sites of other tasks.

- [ ] **Step 2: Add the docs section**

Add this section to `docs/claude/testing.md`:

```markdown
## Known bugs and security findings

**A bug the suite finds lands at once as a failing test marked `xfail(strict=True)`.** The marker cites a GitHub issue, and the fix follows in its own PR.

- **Strict mode keeps the marker honest.** When the fix lands, the test starts passing and strict mode fails the run until the fix PR removes the marker.
- **Nothing is weakened.** Never skip, delete or weaken a test to make the run green.

**A finding that falls under the in-scope list in `SECURITY.md` never goes into a public issue or a public test.** It goes to a private draft advisory on GitHub, and its failing test lands together with its fix.
```

- [ ] **Step 3: Run the documentation tests that read these files**

Run: `local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/test_docs.py tests/test_doc_links.py`
Expected: every test passes, including the link check on `docs/claude/README.md` and the new `testing.md`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml docs/claude/README.md CLAUDE.md docs/claude/testing.md
git commit -m "Type-check the adversarial harness in CI and index its documentation"
```

---

## Task 8: Gate, pull request and review for F1

- [ ] **Step 1: Run the full gate.** This is a code change.

```bash
cd /Applications/workstation/agent-memory/.claude/worktrees/friendly-einstein-53c8da
local/venv-ci/bin/python -m pytest -q -rs -p no:cacheprovider > local/gate-f1.txt 2>&1; tail -5 local/gate-f1.txt
COVERAGE_FILE=$PWD/local/cov/.coverage.f1 local/venv-ci/bin/python -m coverage run -m pytest -q -p no:cacheprovider > local/cov-f1.txt 2>&1; tail -3 local/cov-f1.txt
COVERAGE_FILE=$PWD/local/cov/.coverage.f1 local/venv-ci/bin/python -m coverage report | tail -2
local/venv-ci/bin/python -m mypy -p memvara
local/venv-ci/bin/python -m mypy benchmarks/agent_memory --ignore-missing-imports
local/venv-ci/bin/python -m mypy tests/harness --ignore-missing-imports
local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial --tier nightly
local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial --durations=10 | tail -14
```

Expected:

- The full run is green, with no "skips with no rule" section.
- Coverage is `TOTAL ... 100%`.
- All three mypy runs are clean.
- The nightly run is green.
- The adversarial fast tier takes under 30 s.

Record each "N passed" line for the PR body.

- [ ] **Step 2: Push and open the PR.**

```bash
git push -u origin claude/memvara-testing-suite-plan-ff1278
gh pr create --repo memvara/memvara --base main --head claude/memvara-testing-suite-plan-ff1278 \
  --title "Add the adversarial suite's harness: real server and hook processes, test tiers and a skip ledger" \
  --body-file local/pr-f1.md
```

`local/pr-f1.md` contains:

1. What the PR adds and why, in plain sentences.
2. The tiers table.
3. The gate's "N passed" lines and the coverage line.
4. The fast-tier time.
5. A note that the spec is included.
6. A review line, added after Step 3, that says only that a code review ran at high effort and what it found.

No AI attribution and no model name.

- [ ] **Step 3: Review.** Dispatch a subagent pinned to the latest Sonnet (`model: "sonnet"`) to run `/code-review high <PR number>`. Its brief:

  - Report the findings. Do not post anything to GitHub.
  - Nothing it writes anywhere may carry AI attribution or a model name.

Verify every finding against the code before acting on it. Fix the real ones on this branch, rerun Step 1, and write a reason in the PR body for any finding that is wrong.

- [ ] **Step 4: Update the Confluence decision page.** Update page 17530881 ("Build a tiered adversarial test suite...") so its Links section names the PR.

---

## Task 9: File the public issues (F2)

- [ ] **Step 1: Create one issue per public bug.** Write each body to `local/issues/<id>.md`, then create the issue.

```bash
gh issue create --repo memvara/memvara --label bug --title "<title>" --body-file local/issues/<id>.md
```

Record each issue number that `gh` returns. The titles and bodies:

**B2.** Title: `A value written from a session-bound or agent-bound server ends the user-wide value, and only that session or agent can see the new one`

```markdown
**What happens**

A server bound to a session (`MEMVARA_SESSION`) or to an agent (`MEMVARA_AGENT`) writes a new value for a single-valued fact. The user-wide value is then ended for everyone, but only that session or agent can see the new value. Every other session and agent, and the user-wide scope, is left with no value at all.

**Reproduction** (offline, in memory)

    from memvara import Memvara, NullLLM
    from memvara.embed import HashingEmbedder
    m = Memvara(embedder=HashingEmbedder(dim=512), llm=NullLLM())
    user, s1, s2 = m.scope(user="u"), m.scope(user="u", session="s1"), m.scope(user="u", session="s2")
    user.remember("user", "lives_in", "Berlin")
    s1.remember("user", "lives_in", "Paris")
    [c.object for c in user.get_all()]   # [] -- expected ['Berlin']
    [c.object for c in s1.get_all()]     # ['Paris']
    [c.object for c in s2.get_all()]     # [] -- expected ['Berlin']

The same happens with `agent=` in place of `session=`.

**Decided behaviour**

A value written in a session, or by an agent, shadows the user-wide value inside that session or agent and does not end it. This mirrors how a project's own value shadows the user-wide one today (`docs/INTERNALS.md`, "A repository's own value shadows the user-wide one"):

- A user-wide write does not end a session's or agent's own value.
- A present-tense read takes the value from the narrowest scope in the reader's ancestor chain that holds one. Reads at another instant, and `count()`, are not shadowed.

This reverses the reasoning in the `owner_key` docstring in `memvara/types.py`, and the fix rewrites that docstring and INTERNALS. CONTRIBUTING asks for an issue before any change to scope resolution, and this is that issue.

**Still open:** claims that a session-bound write has already ended stay ended. Whether a migration should reopen them is undecided.
```

**B3.** Title: `The plugin's auto-approve hook still asks permission for memory_get_document and memory_list_documents`

```markdown
**What happens**

`plugin/hooks/approve.py` auto-allows the read-only memvara tools so that reading memory never interrupts the user. Its `READ_ONLY` set lists 11 tools, but the server has 13 read-only tools (`Tool.writes` is false for 13 entries of `TOOLS`). The two missing ones are `memory_get_document` and `memory_list_documents`. For those two the hook prints nothing, so the client asks the user for permission, and the hook's "searched" count misses them.

**Reproduction**

    echo '{"session_id":"s","cwd":"/tmp","hook_event_name":"PreToolUse","tool_name":"mcp__memvara__memory_get_document"}' \
      | MEMVARA_DAEMON=1 python3 plugin/hooks/run.py approve --host claude
    # prints nothing; the same line with memory_search prints an allow decision

**Expected:** every tool whose `writes` flag is false is auto-allowed. A test that compares the hook's list with `TOOLS` stops the two lists drifting apart again.
```

**B4.** Title: `One request line with deeply nested JSON kills the stdio MCP server`

```markdown
**What happens**

`memvara/server/protocol.py` decodes each line with `json.loads` and catches only `ValueError`. A line whose JSON is nested about 100,000 levels deep raises `RecursionError`, which is not a `ValueError`. It escapes the server loop and the process exits with code 1. The agent's client then loses every memory tool for the rest of its session.

**Reproduction**

    python3 - <<'PY'
    import subprocess, sys
    line = '{"jsonrpc":"2.0","id":1,"method":"ping","params":' + "[" * 100000 + "]" * 100000 + "}\n"
    p = subprocess.run([sys.executable, "-m", "memvara.server"], input=line, capture_output=True, text=True,
                       env={"MEMVARA_DB": ":memory:", "MEMVARA_EMBEDDER": "hashing", "MEMVARA_FEATURE_ENCRYPTION": "0",
                            "MEMVARA_FEATURE_PROJECT_SCOPE": "0", "PATH": "/usr/bin:/bin"})
    print(p.returncode, p.stderr.splitlines()[-1])   # 1 RecursionError: maximum recursion depth exceeded ...
    PY

**Expected:** a JSON-RPC parse error (`-32700`) for that line, and the server keeps serving. `SECURITY.md` lists resource exhaustion from your own input as out of scope for security reports, so this is filed as a bug.
```

**B5.** Title: `memory_standing with k=0 says nothing is stored when standing preferences exist`

```markdown
**What happens**

`memory_standing`'s `k` argument has no minimum in its schema. With `k=0` the tool answers "No standing preferences are stored in this scope...", although preferences exist. The default `k` returns them. An agent that reads that answer concludes the user has no preferences, which is the "nothing found" versus "did not look" confusion the hooks are built to avoid.

**Reproduction** over stdio: remember `user prefers tabs` with `memory_type=procedural`, then call `memory_standing` with `{"k": 0}`.

**Expected:** either the schema refuses `k` below 1, or `k=0` never produces the empty-store answer.
```

**B6.** Title: `remember() raises AttributeError when memory_type is given as a string`

```markdown
**What happens**

`Memvara.remember(..., memory_type="procedural")` raises `AttributeError: 'str' object has no attribute 'value'` from inside the SQLite store. The MCP tool accepts the same values as strings, so an agent that moves from the tool to the library naturally passes a string. The failed write changes nothing (the previous value stays live), but the error names neither the argument nor the values it accepts.

**Reproduction**

    from memvara import Memvara, NullLLM
    from memvara.embed import HashingEmbedder
    m = Memvara(embedder=HashingEmbedder(dim=512), llm=NullLLM())
    m.scope(user="u").remember("user", "prefers", "tabs", memory_type="procedural")   # AttributeError

**Expected:** either accept the `MemoryType` values as strings (the spelling the MCP tool takes), or raise a `ValueError` or `TypeError` that names `memory_type` and lists the accepted values.
```

---

## Task 10: The known-bugs registry and its strict xfail tests (F2)

**Files:**
- Create: `tests/harness/known_bugs.py`
- Create: `tests/adversarial/test_adv_known_bugs.py`

**Interfaces:**
- Consumes:
  - `harness.stores.memory`, and the `mcp` and `hook_runner` fixtures;
  - `harness.stdio.McpProcessError`;
  - `memvara.MemoryType`, and `memvara.server.tools.TOOLS`;
  - `plugin/hooks/approve.py` (`READ_ONLY`).
- Produces:
  - `harness.known_bugs.KnownBug(id, issue, title)` and `KNOWN_BUGS: dict[str, KnownBug]`;
  - `xfail(bug_id, *, raises=AssertionError) -> pytest.MarkDecorator`.

- [ ] **Step 1: Write the registry.** Replace each `ISSUE_B*` below with the numbers from Task 9.

`tests/harness/known_bugs.py`:

```python
"""Bugs the adversarial suite has found and not yet fixed, one entry per GitHub issue.

A test that reproduces one of these carries `xfail("B2")`, a strict expected failure
that cites the issue. When the fix lands, the test passes, and strict mode fails the run
until the fix PR removes the marker. So a fixed bug cannot keep its marker, and the marker
cannot hide a different failure: `raises` names the exception the bug produces, and any
other exception fails the test.

Security-class findings are not listed here. They follow SECURITY.md, and their tests
land together with their fixes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class KnownBug:
    id: str
    #: The issue number in memvara/memvara.
    issue: int
    title: str


KNOWN_BUGS: dict[str, KnownBug] = {bug.id: bug for bug in (
    KnownBug("B2", ISSUE_B2, "a session-bound or agent-bound write ends the user-wide value"),
    KnownBug("B3", ISSUE_B3, "auto-approve misses two read-only document tools"),
    KnownBug("B4", ISSUE_B4, "deeply nested JSON kills the stdio server"),
    KnownBug("B5", ISSUE_B5, "memory_standing with k=0 reports an empty store"),
    KnownBug("B6", ISSUE_B6, "remember() crashes on a memory_type given as a string"),
)}


def xfail(bug_id: str, *,
          raises: type[BaseException] | tuple[type[BaseException], ...] = AssertionError,
          ) -> pytest.MarkDecorator:
    """The strict expected-failure marker for a registered bug."""
    bug = KNOWN_BUGS[bug_id]
    return pytest.mark.xfail(
        strict=True, raises=raises,
        reason=f"{bug.id}, memvara/memvara#{bug.issue}: {bug.title}")
```

- [ ] **Step 2: Write the tests**

`tests/adversarial/test_adv_known_bugs.py`:

```python
"""Confirmed bugs, each pinned by a strict expected failure that cites its issue.

Each test states the behaviour the fix must produce. Until then it fails in exactly the
way the bug fails, and `known_bugs.xfail` names that failure.
"""

from __future__ import annotations

import sys
from typing import Callable

import pytest

from harness import known_bugs, stores
from harness.hooks import HOOKS_DIR, HookRunner
from harness.stdio import McpProcess, McpProcessError
from memvara import MemoryType
from memvara.server.tools import TOOLS

if str(HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(HOOKS_DIR))
import approve  # noqa: E402 - plugin/hooks is not a package; the path is set just above

LEVELS = ("session", "agent")


def _live(scoped: object) -> list[str]:
    return sorted(claim.object for claim in scoped.get_all())  # type: ignore[attr-defined]


# -- B2: a bound write ends the user-wide value --------------------------------------

@pytest.mark.parametrize("level", LEVELS)
@known_bugs.xfail("B2")
def test_a_bound_write_leaves_the_user_wide_value_live(level: str) -> None:
    mem = stores.memory()
    mem.scope(user="u").remember("user", "lives_in", "Berlin")
    mem.scope(user="u", **{level: "one"}).remember("user", "lives_in", "Paris")
    assert _live(mem.scope(user="u")) == ["Berlin"]


@pytest.mark.parametrize("level", LEVELS)
@known_bugs.xfail("B2")
def test_a_bound_write_leaves_its_siblings_on_the_user_wide_value(level: str) -> None:
    mem = stores.memory()
    mem.scope(user="u").remember("user", "lives_in", "Berlin")
    mem.scope(user="u", **{level: "one"}).remember("user", "lives_in", "Paris")
    assert _live(mem.scope(user="u", **{level: "two"})) == ["Berlin"]


@pytest.mark.parametrize("level", LEVELS)
def test_a_bound_scope_reads_its_own_value_in_place_of_the_user_wide_one(level: str) -> None:
    """Passes today and must keep passing after the B2 fix: the shadow side of the rule."""
    mem = stores.memory()
    mem.scope(user="u").remember("user", "lives_in", "Berlin")
    mem.scope(user="u", **{level: "one"}).remember("user", "lives_in", "Paris")
    assert _live(mem.scope(user="u", **{level: "one"})) == ["Paris"]


# -- B3: auto-approve misses two read-only tools --------------------------------------

@known_bugs.xfail("B3")
def test_the_approve_hook_allows_exactly_the_servers_read_only_tools() -> None:
    assert set(approve.READ_ONLY) == {tool.name for tool in TOOLS if not tool.writes}


@pytest.mark.parametrize("tool", ["memory_get_document", "memory_list_documents"])
@known_bugs.xfail("B3")
def test_the_approve_hook_allows_the_read_only_document_tools(
        hook_runner: Callable[..., HookRunner], tool: str) -> None:
    result = hook_runner("claude").run("approve", tool_name=f"mcp__memvara__{tool}")
    assert result.exit_code == 0
    assert result.reply is not None, f"the approve hook printed no decision for {tool}"
    assert result.reply["hookSpecificOutput"]["permissionDecision"] == "allow"


# -- B4: deeply nested JSON kills the server ------------------------------------------

@known_bugs.xfail("B4", raises=McpProcessError)
def test_one_deeply_nested_request_does_not_kill_the_server(
        mcp: Callable[..., McpProcess]) -> None:
    server = mcp()
    server.initialize()
    depth = 100_000
    server.send_raw('{"jsonrpc":"2.0","id":99,"method":"ping","params":'
                    + "[" * depth + "]" * depth + "}")
    reply = server.recv(timeout=20)
    assert "error" in reply and reply.get("id") in (None, 99), reply
    assert server.request("ping") == {}


# -- B5: memory_standing with k=0 --------------------------------------------------------

@known_bugs.xfail("B5")
def test_memory_standing_with_k_zero_never_reports_an_empty_store(
        mcp: Callable[..., McpProcess]) -> None:
    server = mcp()
    server.initialize()
    stored = server.call("memory_remember", subject="user", predicate="prefers",
                         object="tabs for indentation", memory_type="procedural")
    assert not stored.is_error, stored.text
    reply = server.call("memory_standing", k=0)
    assert "No standing preferences are stored" not in reply.text, reply.text


# -- B6: a string memory_type -----------------------------------------------------------

@known_bugs.xfail("B6", raises=AttributeError)
def test_remember_takes_a_string_memory_type_or_refuses_it_by_name() -> None:
    user = stores.memory().scope(user="u")
    try:
        # A string on purpose: it is the spelling the MCP tool accepts.
        user.remember("user", "prefers", "tabs", memory_type="procedural")  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        assert "memory_type" in str(exc), exc
        return
    [claim] = user.get_all()
    assert claim.memory_type is MemoryType.PROCEDURAL
```

- [ ] **Step 3: Run the tests.** Each bug test must be an expected failure, and the shadow test must pass.

Run: `local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_known_bugs.py -rxX`
Expected: `2 passed, 10 xfailed`, with no XPASS.

- [ ] **Step 4: Prove each expected failure fails for its stated reason.**

Run: `local/venv-ci/bin/python -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_known_bugs.py --runxfail 2>&1 | grep -E "^(FAILED|E )" | head -40`

Expected: 10 FAILED. The failure behind each:

| Bug | Tests | Failure |
|---|---|---|
| B2 | 4 | `assert [] == ['Berlin']` |
| B3 | 1 | the set comparison, which names the two missing tools |
| B3 | 2 | "the approve hook printed no decision" |
| B4 | 1 | `McpProcessError`: "the server closed its output (exit code 1)", with `RecursionError` in the stderr tail |
| B5 | 1 | the `No standing preferences` assertion |
| B6 | 1 | `AttributeError: 'str' object has no attribute 'value'` |

- [ ] **Step 5: Document the registry, then commit.** In `docs/claude/testing.md`, "Known bugs and security findings", add this bullet after "Strict mode keeps the marker honest":

```markdown
- **Registering a bug.** `tests/harness/known_bugs.py` lists each open bug, and `known_bugs.xfail("B2")` builds its marker.
```

```bash
git add tests/harness/known_bugs.py tests/adversarial/test_adv_known_bugs.py docs/claude/testing.md
git commit -m "Pin five confirmed bugs as strict expected failures that cite their issues"
```

---

## Task 11: Report B1 privately (F2)

- [ ] **Step 1: Check the advisory body.** It is kept in `local/advisory-b1.json`, which git ignores. It never enters the repository, because SECURITY.md keeps a vulnerability private until its fix ships. Read it once before sending.

- [ ] **Step 2: Create the draft advisory.**

```bash
gh api --method POST repos/memvara/memvara/security-advisories --input local/advisory-b1.json --jq '.ghsa_id + " " + .html_url'
```

Expected: a GHSA id and a URL, with the advisory in draft state. Record the id on the internal decision page for security findings, and never in the repository.

---

## Task 12: Gate, pull request and review for F2

- [ ] **Step 1:** Repeat Task 8, Step 1 (the full gate) on this branch.
- [ ] **Step 2:** Push and open the PR.
  - **Base:** `main` if F1 has merged, otherwise F1's branch.
  - **Title:** `Pin five confirmed bugs as strict expected failures, each citing its issue`.
  - **Body:** links to the five issues, the `--runxfail` table from Task 10 Step 4, and the gate's "N passed" lines. It must not mention the private advisory.
- [ ] **Step 3:** Review the PR exactly as in Task 8, Step 3.
- [ ] **Step 4:** Update Confluence page 17563669 (strict xfail) with the issue numbers and the PR link.
