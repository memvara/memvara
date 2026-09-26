"""A sentinel secret must never reach a channel a person or a log would read.

The MCP server takes its API key, its store key and its confirmation secret from the
environment. None of the three may appear on the server's standard error, in its argv (an
argument list is world-readable through `ps`), or in the `memory_stats` a model is shown.
The remote client that carries a bearer token must not print it in its repr, which is the
form that reaches a traceback or a debug log.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

from harness.stdio import McpProcess

API_KEY = "sk-SENTINELKEY-abc123"
DB_KEY = "d" * 64
CONFIRM = "confirm-SENTINELSECRET"
SENTINELS = (API_KEY, DB_KEY, CONFIRM)


def test_no_configured_secret_reaches_stderr_argv_or_stats(mcp) -> None:
    """A server started with all three secrets in its environment leaks none of them to
    its stderr, its argv, or the stats it renders to the model."""
    server: McpProcess = mcp(user="alice", env={
        "MEMVARA_API_KEY": API_KEY, "MEMVARA_DB_KEY": DB_KEY,
        "MEMVARA_CONFIRM_SECRET": CONFIRM, "MEMVARA_FEATURE_ENCRYPTION": "1"})
    server.initialize()
    server.call("memory_remember", predicate="lives_in", object="Lisbon")
    stats = server.call("memory_stats").text
    if sys.platform != "win32":
        argv = subprocess.run(["ps", "-o", "command=", "-p", str(server.proc.pid)],
                              capture_output=True, text=True).stdout
        for secret in SENTINELS:
            assert secret not in argv
    server.close()
    stderr = server.stderr_text()
    for secret in SENTINELS:
        assert secret not in stderr
        assert secret not in stats


needs_httpx = pytest.mark.skipif(
    importlib.util.find_spec("httpx") is None,
    reason="could not import 'httpx'")


@needs_httpx
def test_a_remote_client_never_prints_its_bearer_token_in_a_repr() -> None:
    """`RemoteMemvara` and its scoped views carry a bearer token and must render only their
    scope in a repr, so the token cannot ride a repr into a log or a traceback.
    Construction opens no connection, so this makes no network call."""
    import asyncio

    from memvara.remote.api import RemoteMemvara
    from memvara.remote.aio import AsyncRemoteMemvara

    sync = RemoteMemvara(api_key=API_KEY, base_url="https://example.test")
    a_sync = AsyncRemoteMemvara(api_key=API_KEY, base_url="https://example.test")
    try:
        views = [sync, sync.scope(user="alice"),
                 a_sync, a_sync.scope(user="alice")]
        for view in views:
            assert API_KEY not in repr(view)
    finally:
        sync.close()
        asyncio.run(a_sync.aclose())
