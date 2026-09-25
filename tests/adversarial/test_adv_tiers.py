"""Tiers: which tests a run collects, decided by the folder a test lives in."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

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
