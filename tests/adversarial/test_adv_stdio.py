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


def test_a_reply_whose_result_is_not_an_object_is_reported(
        mcp: Start, monkeypatch: pytest.MonkeyPatch) -> None:
    """The check must not be an assert, which python -O removes."""
    server = mcp()
    monkeypatch.setattr(server, "recv",
                        lambda timeout=None: {"jsonrpc": "2.0", "id": 1, "result": None})
    with pytest.raises(McpProcessError, match="not an object"):
        server.request("ping")


def test_a_request_from_the_server_is_not_taken_for_the_reply(
        mcp: Start, monkeypatch: pytest.MonkeyPatch) -> None:
    """A message with a method is a request or notification from the server, even when
    its id matches the one the client is waiting on."""
    server = mcp()
    messages = iter([
        {"jsonrpc": "2.0", "id": 1, "method": "sampling/createMessage", "params": {}},
        {"jsonrpc": "2.0", "id": 1, "result": {}},
    ])
    monkeypatch.setattr(server, "recv", lambda timeout=None: next(messages))
    assert server.request("ping") == {}


def test_each_server_gets_its_own_store_unless_a_test_shares_one(mcp: Start) -> None:
    first, second = mcp(), mcp()
    assert first.db != second.db
