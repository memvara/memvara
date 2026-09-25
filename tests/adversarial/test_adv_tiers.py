"""Tiers: which tests a run collects, decided by the folder a test lives in."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

from harness.env import REPO, child_env
from harness.tiers import (SELECTS, TESTS, TIER_DIRS, collection_report,
                           folders_without_init, ignored, nested_tier_folders,
                           tier_of)


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


def test_a_package_file_outside_the_tests_folder_is_an_ordinary_fast_module() -> None:
    """Only the suite's own folders need their __init__.py in every tier. The one in a
    package of memvara/ holds doctests, which are fast tests."""
    assert ignored(REPO / "memvara" / "ingest" / "__init__.py", "local", is_dir=False)
    assert not ignored(REPO / "memvara" / "ingest" / "__init__.py", "fast", is_dir=False)


def test_every_tier_folder_is_a_package() -> None:
    """Without __init__.py a tier folder's modules are imported as top-level modules, and
    two test files with the same name in different folders then collide."""
    for directory in (TESTS / "adversarial").rglob("*"):
        if directory.is_dir() and directory.name in TIER_DIRS:
            assert (directory / "__init__.py").is_file(), directory


def test_a_tier_folder_inside_live_is_collected_by_a_local_run() -> None:
    """Everything under tests/live is local, whatever its own folder is called."""
    assert not ignored(TESTS / "live" / "nightly", "local", is_dir=True)


_PARTS = st.sampled_from(["adversarial", "live", "model", *TIER_DIRS])


@given(folders=st.lists(_PARTS, max_size=4), option=st.sampled_from(sorted(SELECTS)))
def test_a_collected_file_never_sits_in_a_folder_the_run_leaves_out(
        folders: list[str], option: str) -> None:
    """A file a run collects must be reachable: no folder above it may be left out."""
    path = TESTS.joinpath(*folders, "test_adv_x.py")
    if not ignored(path, option, is_dir=False):
        for parent in path.parents:
            if parent == TESTS:
                break
            assert not ignored(parent, option, is_dir=True), (parent, option)


def test_the_collection_report_names_every_folder_a_run_left_out() -> None:
    line = collection_report(
        "fast", [TESTS / "adversarial" / "weekly", TESTS / "adversarial" / "nightly"])
    assert line == ("tier fast; left out 2 tier folders: "
                    "tests/adversarial/nightly, tests/adversarial/weekly")


def test_the_collection_report_says_when_nothing_was_left_out() -> None:
    assert collection_report("weekly", []) == "tier weekly; no tier folder left out"


def test_a_tier_folder_inside_another_is_found(tmp_path: pathlib.Path) -> None:
    for folder in ("a/nightly/weekly", "live/local", "b/nightly"):
        (tmp_path / folder).mkdir(parents=True)
    assert nested_tier_folders(tmp_path) == [tmp_path / "a" / "nightly" / "weekly",
                                             tmp_path / "live" / "local"]


def test_no_tier_folder_sits_inside_another() -> None:
    """The outermost tier folder decides a test's tier, so a tier folder inside another
    one would carry a name that means nothing."""
    assert nested_tier_folders(TESTS) == []


def test_a_folder_without_init_is_found(tmp_path: pathlib.Path) -> None:
    for folder in ("model/nightly", "sessions"):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / "model" / "nightly" / "__init__.py").write_text("")
    (tmp_path / "__pycache__").mkdir()
    assert folders_without_init(tmp_path) == [tmp_path / "model", tmp_path / "sessions"]


def test_every_folder_of_the_suite_is_a_package() -> None:
    """A folder without __init__.py makes the modules below it top-level, and a tier
    folder under it then cannot be imported by its package name."""
    for root in (TESTS / "adversarial", TESTS / "live"):
        if root.is_dir():
            assert folders_without_init(root) == [], root


def test_the_collection_report_ignores_bytecode_folders() -> None:
    line = collection_report("fast", [TESTS / "adversarial" / "nightly",
                                      TESTS / "adversarial" / "nightly" / "__pycache__"])
    assert line == "tier fast; left out 1 tier folder: tests/adversarial/nightly"


def _collect(tmp_path: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    """`pytest --collect-only` with `args`, in a child process started at the repository
    root, as a developer would run it."""
    home = tmp_path / "home"
    home.mkdir()
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         *args],
        cwd=REPO, env=child_env(home), capture_output=True, text=True, timeout=120)


def test_a_run_given_only_the_package_accepts_tier_and_leaves_out_its_doctests(
        tmp_path: pathlib.Path) -> None:
    """pytest learns an option only from a conftest file it has loaded, and before a run
    it loads only the conftest files above the paths it was given. With the option
    registered in tests/conftest.py, `pytest memvara --tier local` stopped with
    "unrecognized arguments: --tier". The doctests in memvara/ are fast tests, so a
    local run given that folder collects nothing."""
    run = _collect(tmp_path, "memvara", "--tier", "local")
    assert "unrecognized arguments" not in run.stderr, run.stderr
    assert run.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, run.stdout + run.stderr


def test_a_local_run_leaves_out_the_doctests_in_the_package(tmp_path: pathlib.Path) -> None:
    """A hook in tests/conftest.py is asked only about paths under tests/, so the doctests
    in memvara/ used to be collected by every tier, local included."""
    run = _collect(tmp_path, "--tier", "local")
    assert run.returncode == pytest.ExitCode.OK, run.stdout + run.stderr
    collected = [line for line in run.stdout.splitlines() if "::" in line]
    assert collected, "the local tier has tests, so the run must collect some"
    assert [line for line in collected if not line.startswith("tests/")] == []
