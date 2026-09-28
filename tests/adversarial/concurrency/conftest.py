"""Fixtures shared by the concurrency and crash tests."""

from __future__ import annotations

import pathlib

import pytest


@pytest.fixture()
def home(tmp_path: pathlib.Path) -> pathlib.Path:
    """A home directory of the test's own, for the child processes it starts."""
    path = tmp_path / "home"
    path.mkdir()
    return path
