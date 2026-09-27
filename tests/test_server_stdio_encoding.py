"""The stdio server reads UTF-8 and writes its errors as UTF-8, whatever the locale says.

The adversarial fuzz tests check the same over a real pipe, in a child process, where
coverage does not reach. These run `main()` in this process with the process's own
streams replaced by streams that have bytes under them, as a real process's do.
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from memvara.server.cli import main


@pytest.fixture
def env(tmp_path):
    return {"MEMVARA_DB": str(tmp_path / "m.db"), "MEMVARA_EMBEDDER": "hashing",
            "MEMVARA_FEATURE_ENCRYPTION": "0", "MEMVARA_FEATURE_PROJECT_SCOPE": "0"}


def _stdin(monkeypatch, data: bytes, encoding: str) -> io.BytesIO:
    buffer = io.BytesIO(data)
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(buffer, encoding=encoding))
    return buffer


def test_a_line_that_is_not_utf8_gets_a_parse_error_and_the_next_line_an_answer(
        monkeypatch, env):
    """Under a strict UTF-8 locale this byte used to end the server with
    UnicodeDecodeError, and no request after it got a reply (#311)."""
    _stdin(monkeypatch, b'{"jsonrpc":"2.0","id":7,"method":"ping","params":{"x":"\xff"}}\n'
                        b'{"jsonrpc":"2.0","id":8,"method":"ping"}\n', "utf-8")
    out = io.StringIO()
    assert main([], env=env, stdout=out) == 0
    first, second = (json.loads(line) for line in out.getvalue().splitlines())
    assert first["id"] is None and first["error"]["code"] == -32700, first
    assert "column 56" in first["error"]["message"], first
    assert second == {"jsonrpc": "2.0", "id": 8, "result": {}}


def test_utf8_input_is_read_as_utf8_when_the_locale_says_cp1252(monkeypatch, env):
    """A Node client writes "Zürich" as UTF-8 bytes, and under cp1252, a Windows pipe's
    default, it was stored as "ZÃ¼rich" (#311)."""
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "memory_remember",
                          "arguments": {"predicate": "lives_in", "object": "Zürich"}}}
    _stdin(monkeypatch, json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n",
           "cp1252")
    out = io.StringIO()
    assert main([], env=env, stdout=out) == 0
    text = json.loads(out.getvalue())["result"]["content"][0]["text"]
    assert "user lives in Zürich" in text, text


def test_serving_leaves_the_process_standard_input_open(monkeypatch, env):
    """The server reads through a UTF-8 wrapper of its own over standard input's bytes.
    Closing that wrapper would close the bytes under the process's own stdin too."""
    buffer = _stdin(monkeypatch, b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n', "utf-8")
    assert main([], env=env, stdout=io.StringIO()) == 0
    assert not buffer.closed
    assert not sys.stdin.closed


def test_standard_error_is_written_as_utf8_when_the_locale_says_cp1252(monkeypatch):
    """The usage printed for a wrong argument starts with an em dash, which cp1252 writes
    as a byte a client reading UTF-8 cannot decode."""
    buffer = io.BytesIO()
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(buffer, encoding="cp1252"))
    assert main(["--no-such-option"], env={}, stdout=io.StringIO()) == 2
    sys.stderr.flush()
    written = buffer.getvalue()
    assert "—".encode("utf-8") in written
    assert written.decode("utf-8").startswith("memvara-mcp: unexpected argument")
