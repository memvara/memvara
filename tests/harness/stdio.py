"""A real memvara MCP server in its own process, driven over its stdio pipe.

tests/test_server.py calls MemvaraMCPServer.handle_line() in-process. This module tests
what that cannot reach: the process an agent's client actually starts. That process reads
its configuration from the environment, frames messages on a real pipe, and exits when the
client closes stdin.
"""

from __future__ import annotations

import json
import os
import pathlib
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping

from memvara.server.config import FEATURES

from .env import child_env

#: The protocol version a client asks for when a test does not name one.
PROTOCOL = "2025-06-18"

_SCOPE_FIELDS = ("tenant", "project", "agent", "session")


class McpProcessError(RuntimeError):
    """The server exited, stopped reading, or wrote nothing within the timeout."""


class RpcError(RuntimeError):
    """The server answered a request with a JSON-RPC error object."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"JSON-RPC error {code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolResult:
    """One tools/call result: the text of its content blocks, and its error flag."""

    text: str
    is_error: bool
    raw: dict[str, Any] = field(repr=False)


class McpProcess:
    """`python -m memvara.server` in a child process, spoken to one JSON line at a time.

    `db` is the store the server opens (MEMVARA_DB). `home` becomes the child's HOME and
    must not be the real one. `user` binds MEMVARA_USER. `scope` can bind tenant,
    project, agent and session. `features` switches named features on or off. A feature
    name the server does not know is refused here, because the server would refuse to
    start with it. `env` is applied last.
    """

    def __init__(self, db: str | os.PathLike[str], *, home: pathlib.Path,
                 user: str | None = "tester", scope: Mapping[str, str] | None = None,
                 features: Mapping[str, bool] | None = None, read_only: bool = False,
                 env: Mapping[str, str] | None = None, cwd: pathlib.Path | None = None,
                 timeout: float = 30.0) -> None:
        extra: dict[str, str] = {"MEMVARA_DB": str(db)}
        if user is not None:
            extra["MEMVARA_USER"] = user
        for key, value in (scope or {}).items():
            if key not in _SCOPE_FIELDS:
                raise ValueError(f"unknown scope field {key!r}; use one of {_SCOPE_FIELDS}")
            extra[f"MEMVARA_{key.upper()}"] = value
        for name, on in (features or {}).items():
            if name not in FEATURES:
                raise ValueError(
                    f"unknown feature {name!r}; the server would refuse to start with it")
            extra[f"MEMVARA_FEATURE_{name.upper()}"] = "1" if on else "0"
        if read_only:
            extra["MEMVARA_READ_ONLY"] = "1"
        extra.update(env or {})

        self.timeout = timeout
        #: Every line written ("->") and read ("<-"), in order.
        self.transcript: list[tuple[str, str]] = []
        self._next_id = 1
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._stderr: list[bytes] = []
        options: dict[str, Any] = {}
        if sys.platform == "win32":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "memvara.server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=child_env(home, extra), cwd=str(cwd or home), **options)
        self._readers = [threading.Thread(target=self._pump_stdout, daemon=True),
                         threading.Thread(target=self._pump_stderr, daemon=True)]
        for reader in self._readers:
            reader.start()

    # -- the pipe ------------------------------------------------------------

    def _pump_stdout(self) -> None:
        stream = self.proc.stdout
        assert stream is not None
        for raw in iter(stream.readline, b""):
            self._lines.put(raw)
        self._lines.put(None)  # the end of the stream

    def _pump_stderr(self) -> None:
        stream = self.proc.stderr
        assert stream is not None
        for raw in iter(stream.readline, b""):
            self._stderr.append(raw)

    def stderr_text(self) -> str:
        """Everything the server has written to stderr so far."""
        return b"".join(self._stderr).decode("utf-8", "replace")

    def alive(self) -> bool:
        return self.proc.poll() is None

    def _dead(self, what: str) -> McpProcessError:
        try:
            code: int | None = self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            code = None
        return McpProcessError(
            f"{what} (exit code {code}); stderr: {self.stderr_text()[-800:]!r}")

    def send_raw(self, data: str | bytes) -> None:
        """Write `data` and one newline, exactly as given, with no framing checks."""
        payload = data.encode("utf-8") if isinstance(data, str) else data
        self.transcript.append(("->", payload.decode("utf-8", "replace")))
        stream = self.proc.stdin
        assert stream is not None
        try:
            stream.write(payload + b"\n")
            stream.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise self._dead(f"could not write to the server: {exc}") from exc

    def recv(self, timeout: float | None = None) -> dict[str, Any]:
        """The next message the server writes. Raises McpProcessError when none arrives."""
        wait = self.timeout if timeout is None else timeout
        try:
            raw = self._lines.get(timeout=wait)
        except queue.Empty:
            raise McpProcessError(
                f"no message from the server within {wait}s; "
                f"stderr: {self.stderr_text()[-800:]!r}") from None
        if raw is None:
            self._lines.put(None)  # later calls must see the end of the stream too
            raise self._dead("the server closed its output")
        line = raw.decode("utf-8")
        self.transcript.append(("<-", line.rstrip("\r\n")))
        message = json.loads(line)
        if not isinstance(message, dict):
            raise McpProcessError(f"the server wrote a message that is not an object: "
                                  f"{line[:200]!r}")
        return message

    # -- JSON-RPC ------------------------------------------------------------

    def request(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Send one request, wait for the reply with its id, and return its result."""
        request_id = self._next_id
        self._next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = dict(params)
        self.send_raw(json.dumps(message))
        while True:
            reply = self.recv()
            if reply.get("id") == request_id:
                break
        if "error" in reply:
            raise RpcError(int(reply["error"]["code"]), str(reply["error"]["message"]))
        result = reply["result"]
        assert isinstance(result, dict)
        return result

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        """Send one notification, which by definition gets no reply."""
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = dict(params)
        self.send_raw(json.dumps(message))

    def initialize(self, protocol: str = PROTOCOL) -> dict[str, Any]:
        """The opening handshake a client performs, followed by `initialized`."""
        result = self.request("initialize", {
            "protocolVersion": protocol, "capabilities": {},
            "clientInfo": {"name": "memvara-adversarial-suite", "version": "0"}})
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self.request("tools/list")["tools"])

    def call(self, name: str, /, **arguments: Any) -> ToolResult:
        """Call one tool. A tool that ran and failed comes back with is_error set."""
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        text = "".join(str(block.get("text", "")) for block in result.get("content", []))
        return ToolResult(text=text, is_error=bool(result.get("isError")), raw=result)

    # -- ending it -----------------------------------------------------------

    def kill(self) -> None:
        """Stop the server at once: SIGKILL on POSIX, TerminateProcess on Windows."""
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=10)
        self._finish()

    def close(self, timeout: float = 10.0) -> int:
        """Close stdin, as a client does at the end of a session, and return the exit code."""
        stream = self.proc.stdin
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except OSError:
                pass
        try:
            code = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            raise McpProcessError(
                f"the server did not exit within {timeout}s of its input closing") from None
        self._finish()
        return code

    def _finish(self) -> None:
        for reader in self._readers:
            reader.join(timeout=5)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass

    def __enter__(self) -> McpProcess:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.kill()
