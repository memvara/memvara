"""The hooks' fixed reads are plain: no query rewrite on a local store.

The session-start hook recalls one fixed sentence at every session, and the daemon warms
its store with a one-word recall. Neither is a question worth a model call, and a rewrite
of the same words could hand one session a different block from the next. A local store
(the library's `Memvara`, or its `RemoteMemvara`) is asked with `query_rewrite=False`. The
standard-library hosted client takes no such argument, and its server decides for itself,
so it is asked exactly as before.
"""
from __future__ import annotations

import pathlib
import sys

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import daemon  # noqa: E402
import session_start  # noqa: E402


class _Store:
    """A store that records what every recall was asked with."""

    def __init__(self) -> None:
        self.asked: list[dict] = []

    def recall(self, query, **kwargs) -> str:
        self.asked.append(kwargs)
        return ""


def _run_session_start(monkeypatch, store, close) -> None:
    monkeypatch.setattr(session_start, "payload", lambda: {"session_id": "s", "cwd": ""})
    monkeypatch.setattr(session_start, "write", lambda host, reply: None)
    monkeypatch.setattr(session_start, "due_capture_alert", lambda: "")
    monkeypatch.setattr(session_start, "open_writer", lambda: (store, close))
    monkeypatch.setattr(session_start, "_local_binding", lambda s: "")
    monkeypatch.setattr(session_start, "_hosted_binding", lambda s: "")

    def standing(store, *, fallback, **kw):
        return fallback()

    monkeypatch.setattr(session_start, "standing_block", standing)
    assert session_start.main() == 0


def test_session_start_asks_a_local_store_for_plain_reads(monkeypatch):
    store = _Store()
    _run_session_start(monkeypatch, store, None)
    assert len(store.asked) == 2
    assert all(kw["query_rewrite"] is False for kw in store.asked)


def test_session_start_asks_the_hosted_client_as_before(monkeypatch):
    store = _Store()
    _run_session_start(monkeypatch, store, lambda: None)
    assert len(store.asked) == 2
    assert all("query_rewrite" not in kw for kw in store.asked)


def test_the_daemon_warms_a_local_store_with_a_plain_read(monkeypatch):
    store = _Store()
    monkeypatch.setattr(daemon, "open_store", lambda: store)
    monkeypatch.setattr(daemon.Daemon, "run", lambda self: 0)
    assert daemon.main() == 0
    assert store.asked == [{"k": 1, "query_rewrite": False}]


def test_the_daemon_warms_the_hosted_client_as_before(monkeypatch):
    import lib.hosted
    store = _Store()
    monkeypatch.setattr(daemon, "open_store", lambda: None)
    monkeypatch.setattr(lib.hosted, "open_hosted", lambda: store)
    monkeypatch.setattr(daemon.Daemon, "run", lambda self: 0)
    assert daemon.main() == 0
    assert store.asked == [{"k": 1}]
