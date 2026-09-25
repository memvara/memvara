"""The plugin's hook scripts, run in a child process the way a client runs them."""

from __future__ import annotations

import importlib
import json
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .env import REPO, child_env

HOOKS_DIR = REPO / "plugin" / "hooks"
RUN = HOOKS_DIR / "run.py"


class HookOutputError(AssertionError):
    """A hook printed something that is not JSON. On a real client that desynchronises
    the conversation, so it is a failure in its own right."""


@dataclass(frozen=True)
class HookResult:
    """What one hook run did."""

    exit_code: int
    stdout: str
    stderr: str
    #: stdout parsed as JSON, or None when the hook printed nothing.
    reply: dict[str, Any] | None
    #: Wall-clock seconds, including Python start-up.
    elapsed: float


def host_record(host: str) -> Any:
    """The Host record that plugin/hooks/hosts/<host>.py defines."""
    if str(HOOKS_DIR) not in sys.path:
        sys.path.insert(0, str(HOOKS_DIR))
    return importlib.import_module(f"hosts.{host}").HOST


def parse_reply(stdout: str, *, what: str, stderr: str = "") -> dict[str, Any] | None:
    """A hook's stdout as JSON, or None when it printed nothing.

    Output that is not JSON raises HookOutputError, with the output and stderr attached.
    """
    text = stdout.strip()
    if not text:
        return None
    try:
        reply = json.loads(text)
    except ValueError:
        raise HookOutputError(f"{what} printed text that is not JSON: {text[:300]!r}; "
                              f"stderr: {stderr[-300:]!r}") from None
    if not isinstance(reply, dict):
        raise HookOutputError(f"{what} printed JSON that is not an object: {text[:300]!r}")
    return reply


class HookRunner:
    """Runs `plugin/hooks/run.py <hook> --host <host>` with a payload shaped for that host.

    `home` becomes the child's HOME, and `cwd` its working directory. When `server_env`
    is given, it is written into the host's first client config file as the memvara
    server's env block, which is where the hooks look for the store
    (plugin/hooks/lib/ipc.py). Without it, the hooks find no store and report
    "not configured".
    """

    def __init__(self, host: str, *, home: pathlib.Path, cwd: pathlib.Path,
                 server_env: Mapping[str, str] | None = None,
                 env: Mapping[str, str] | None = None) -> None:
        self.host = host_record(host)
        self.home = pathlib.Path(home)
        self.cwd = pathlib.Path(cwd)
        self._env = child_env(self.home, env)
        if server_env is not None:
            self.write_client_config(server_env)

    def write_client_config(self, server_env: Mapping[str, str]) -> pathlib.Path:
        """Write the host's first client config file, holding a memvara server block."""
        if self.host.config_format != "json":
            raise NotImplementedError(
                f"{self.host.id} keeps a {self.host.config_format} client config")
        path = pathlib.Path(str(self.host.client_configs[0]).replace("~", str(self.home), 1))
        path.parent.mkdir(parents=True, exist_ok=True)
        block = {"command": sys.executable, "args": ["-m", "memvara.server"],
                 "env": dict(server_env)}
        path.write_text(json.dumps({"mcpServers": {"memvara": block}}), encoding="utf-8")
        return path

    def payload(self, hook: str, **fields: Any) -> dict[str, Any]:
        """The stdin this host sends for `hook`.

        Each keyword names an Event field (session, cwd, prompt, transcript_path or
        tool_name), and it is written under this host's own key for that field. When a
        host accepts several keys, the last one is used, because the earlier ones are
        richer shapes that tests of that host build themselves. Cursor's
        workspace_roots is a list, for example.
        """
        event = self.host.events.get(hook)
        if event is None:
            raise ValueError(f"{self.host.id} has no event for the {hook} hook")
        body: dict[str, Any] = {"hook_event_name": event}
        values = {"session": "adversarial-session", "cwd": str(self.cwd), **fields}
        for name, value in values.items():
            keys = self.host.fields.get(name)
            if not keys:
                raise ValueError(f"{self.host.id} has no stdin key for the {name} field")
            body[keys[-1]] = value
        return body

    def run(self, hook: str, *, stdin: str | None = None, timeout: float | None = None,
            **fields: Any) -> HookResult:
        """Run one hook and wait for it, within this host's own timeout for that hook."""
        text = json.dumps(self.payload(hook, **fields)) if stdin is None else stdin
        limit = float(self.host.timeouts[hook]) if timeout is None else timeout
        started = time.monotonic()
        done = subprocess.run(
            [sys.executable, str(RUN), hook, "--host", self.host.id], input=text,
            capture_output=True, text=True, encoding="utf-8", env=self._env,
            cwd=str(self.cwd), timeout=limit)
        elapsed = time.monotonic() - started
        reply = parse_reply(done.stdout, what=f"{hook} on {self.host.id}", stderr=done.stderr)
        return HookResult(exit_code=done.returncode, stdout=done.stdout, stderr=done.stderr,
                          reply=reply, elapsed=elapsed)
