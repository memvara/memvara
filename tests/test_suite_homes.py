"""The home directories that tests/conftest.py makes for every test.

Every test in this repository runs with `HOME` pointed at a directory of its own, and with
the plugin hooks' home pointed at another, so that nothing a test does reaches the
developer's real files. Those two directories used to be made with pytest's
`tmp_path_factory.mktemp()`. To number a new directory, `mktemp()` lists every entry in the
session's base temporary directory. So each test listed a directory that had grown by three
entries for every test before it: its own `tmp_path`, and these two. The suite's run time
therefore grew with the square of its size. On CI, the tests that run after the adversarial
suite took about 40% longer than the same tests run on their own.

The two directories are now made inside one directory for the whole session, with random
names that need no listing, so making them costs the same for the last test as for the
first.
"""

from __future__ import annotations

import os
import pathlib
import sys
from types import ModuleType
from typing import Iterator

import pytest

HOOKS = pathlib.Path(__file__).resolve().parents[1] / "plugin" / "hooks"


@pytest.fixture(scope="module")
def ipc() -> Iterator[ModuleType]:
    """The hooks' `lib.ipc`, imported before any per-test fixture runs.

    tests/conftest.py redirects the hooks' home only when `lib.ipc` has been imported.
    A module-scoped fixture is set up before the function-scoped ones, so the redirect
    applies to the test that asks for this.
    """
    sys.path.insert(0, str(HOOKS))
    try:
        from lib import ipc as module
    finally:
        sys.path.remove(str(HOOKS))
    yield module


def assert_made_apart(directory: pathlib.Path, base: pathlib.Path) -> None:
    """`directory` exists inside pytest's base temporary directory `base`, but not directly
    in it, where every new numbered directory costs a listing of all the others."""
    assert directory.is_dir()
    assert base in directory.parents, (directory, base)
    assert directory.parent != base, (
        f"{directory} was made directly in pytest's base temporary directory, where every "
        "new numbered directory costs a listing of all the others")


def test_a_tests_home_is_not_one_of_pytests_numbered_directories(
        tmp_path_factory: pytest.TempPathFactory) -> None:
    assert_made_apart(pathlib.Path(os.environ["HOME"]), tmp_path_factory.getbasetemp())


def test_the_hooks_home_is_not_one_of_pytests_numbered_directories(
        ipc: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> None:
    hooks_home = pathlib.Path(ipc._HOME)
    assert_made_apart(hooks_home, tmp_path_factory.getbasetemp())
    assert hooks_home != pathlib.Path(os.environ["HOME"])
