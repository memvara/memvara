"""Fixtures for the hook conformance tests. The `clis` fixture, fresh fake agent CLIs, is
in tests/adversarial/conftest.py, because tests/adversarial/test_adv_hooks.py uses it too."""

from __future__ import annotations

import pathlib
from typing import Callable, Iterator

import pytest

from harness.hooks import HookRunner

from . import support


@pytest.fixture
def hooks(tmp_path: pathlib.Path) -> Iterator[Callable[..., HookRunner]]:
    """Build HookRunners for one test, each with a short home of its own unless the test
    passes one, and close them when the test ends. Closing stops any recall daemon a
    runner allowed and any capture child it left running."""
    work = tmp_path / "work"
    work.mkdir()
    with support.runner_factory(work) as make:
        yield make


@pytest.fixture
def store(tmp_path: pathlib.Path) -> pathlib.Path:
    """A store file that holds `support.MEMORY` for `support.USER`."""
    return support.make_store(tmp_path / "memory.db")


@pytest.fixture
def store_env(store: pathlib.Path) -> dict[str, str]:
    """The variables that name `store` to the hooks."""
    return support.store_env(store)
