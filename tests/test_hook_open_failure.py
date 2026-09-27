"""`lib.open.failure` tells a store that cannot open from one that is not configured (#337).

The hook conformance tests run each hook in a child process; these run `open_store` and
`fast.recall` in this process, where a value left from an earlier call would show.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

from lib import fast, hosted  # noqa: E402
from lib import open as opener  # noqa: E402


@pytest.fixture(autouse=True)
def _no_client_config(monkeypatch, tmp_path):
    """Only this process's environment configures a store, and no hosted login exists."""
    monkeypatch.setattr(opener, "client_env", lambda: {
        key: value for key, value in __import__("os").environ.items()})
    monkeypatch.setattr(opener, "_CREDENTIALS", tmp_path / "no-credentials.json")
    monkeypatch.setattr(hosted, "credentials", lambda: None)
    monkeypatch.setattr(fast, "_OPENED", None)
    monkeypatch.setattr(fast, "socket_path", lambda key: None)
    for name in ("MEMVARA_DB", "MEMVARA_MODE", "MEMVARA_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def broken(monkeypatch, tmp_path) -> pathlib.Path:
    path = tmp_path / "broken.db"
    path.write_text("this file is not a SQLite database")
    monkeypatch.setenv("MEMVARA_DB", str(path))
    monkeypatch.setenv("MEMVARA_EMBEDDER", "hashing")
    monkeypatch.setenv("MEMVARA_FEATURE_ENCRYPTION", "0")
    return path


def test_a_store_that_cannot_open_is_kept_as_the_failure(broken) -> None:
    assert opener.open_store() is None
    assert opener.failure is not None


def test_the_next_call_clears_a_failure_left_by_an_earlier_one(broken, monkeypatch) -> None:
    assert opener.open_store() is None and opener.failure is not None
    monkeypatch.delenv("MEMVARA_DB")
    assert opener.open_store() is None
    assert opener.failure is None


def test_recall_reports_a_store_that_cannot_open_as_a_failure_naming_its_class(broken) -> None:
    text, ok, reason = fast.recall("where do I live", spawn=False)
    assert (text, ok) == ("", False)
    assert reason == f"{fast.OPEN}:{type(opener.failure).__name__}"


def test_recall_reports_nothing_configured_as_none() -> None:
    assert fast.recall("where do I live", spawn=False) == ("", None, "")
