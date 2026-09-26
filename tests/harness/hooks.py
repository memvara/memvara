"""The plugin's hook scripts, run in a child process the way a client runs them."""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .env import REPO, child_env

HOOKS_DIR = REPO / "plugin" / "hooks"
RUN = HOOKS_DIR / "run.py"

#: Where the hooks keep their logs, their state and the recall daemon's socket, under the
#: home directory a run is given.
HOOKS_HOME = pathlib.Path(".memvara") / ".hooks"

#: The moment a hook log line was written, which starts every line. `HookResult.logs`
#: leaves it out, so a test can compare lines from two runs.
_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z) ")

#: The line run.py writes to hooks.log when it hands capture to a child in a new session.
_DETACHED = re.compile(r"^detached hook=capture host=\S+ pid=(\d+)")


class HookOutputError(AssertionError):
    """A hook printed something that is not JSON. On a real client that desynchronises
    the conversation, so it is a failure in its own right."""


class HookTimeout(AssertionError):
    """A hook ran past its host's time limit. A real client would kill it there and carry
    on without its answer, so it is a failure in its own right."""


class Stubs(Protocol):
    """Stub agent CLIs, such as `harness.fakes.cli.FakeClis`."""

    def path(self, rest: str | None = None) -> str:
        """A PATH value with the stubs first and `rest` after them."""
        ...


@dataclass(frozen=True)
class HookResult:
    """What one hook run did."""

    exit_code: int
    stdout: str
    stderr: str
    #: stdout parsed as JSON, or None when the hook printed nothing.
    reply: dict[str, Any] | None
    #: Wall-clock seconds, including Python start-up. For a capture that the host hands to
    #: a child in a new session, this is how long the host waited, not how long the
    #: capture ran.
    elapsed: float
    #: The lines the run added to each log in `~/.memvara/.hooks/`, keyed by the file's
    #: stem (`hooks`, `recall`, `capture`, `recall-sample`), without their timestamps. For
    #: a capture handed to a child, it includes what the child wrote before it ended.
    logs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: The pid of the child run.py handed a capture to, or None when it handed none.
    detached_pid: int | None = None

    def log(self, name: str) -> tuple[str, ...]:
        """The lines this run added to `<name>.log`, without their timestamps."""
        return tuple(self.logs.get(name, ()))


def _text(output: str | bytes | None) -> str:
    """Partial output from a timed-out run, which Python can hand back as bytes."""
    if isinstance(output, bytes):
        return output.decode("utf-8", "replace")
    return output or ""


def host_record(host: str) -> Any:
    """The Host record that plugin/hooks/hosts/<host>.py defines."""
    if str(HOOKS_DIR) not in sys.path:
        sys.path.insert(0, str(HOOKS_DIR))
    return importlib.import_module(f"hosts.{host}").HOST


def host_ids() -> tuple[str, ...]:
    """Every host the plugin has a record for: the modules in plugin/hooks/hosts."""
    return tuple(sorted(path.stem for path in (HOOKS_DIR / "hosts").glob("*.py")
                        if path.stem != "__init__"))


def agent_clis() -> frozenset[str]:
    """The program name of every agent CLI a capture can start: each host's own
    extractor, and the `claude` CLI that every host falls back to
    (`CLAUDE_CLI` in plugin/hooks/core/host.py)."""
    records = [host_record(host) for host in host_ids()]
    fallback = importlib.import_module("core.host").CLAUDE_CLI
    specs = [record.extractor for record in records] + [fallback]
    return frozenset(spec.argv[0] for spec in specs if spec is not None and spec.argv)


def path_without_agent_clis(path: str) -> str:
    """`path` without any directory that holds a real agent CLI.

    A capture finds its extractor on PATH. The stubs a test gives it come first, but the
    fakes stand in for `claude` and `codex` only, so a real `cursor-agent`, `copilot` or
    `opencode` further along would still be found, and it would reach the network and
    spend money. Leaving out the whole directory is the only way to be sure. On Windows a
    program is also found under each suffix in PATHEXT, such as `.cmd`.
    """
    names = agent_clis()
    suffixes = [""] + [suffix.lower() for suffix in
                       os.environ.get("PATHEXT", "").split(os.pathsep) if suffix]

    def holds_one(directory: str) -> bool:
        return any(os.path.exists(os.path.join(directory, name + suffix))
                   for name in names for suffix in suffixes)

    return os.pathsep.join(part for part in path.split(os.pathsep)
                           if part and not holds_one(part))


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
        raise HookOutputError(f"{what} printed JSON that is not an object: {text[:300]!r}; "
                              f"stderr: {stderr[-300:]!r}")
    return reply


def _detached_pid(lines: Sequence[str]) -> int | None:
    """The pid in run.py's line saying it handed capture to a child, if there is one."""
    for line in lines:
        found = _DETACHED.match(line)
        if found:
            return int(found.group(1))
    return None


def _alive(pid: int) -> bool:
    """Whether process `pid` is still running. POSIX only.

    A process that has ended but not yet been reaped, a zombie, counts as ended: a
    capture child's parent has exited, and on Linux nothing may reap the child promptly.
    """
    if sys.platform == "win32":
        # os.kill with signal 0 terminates the process on Windows rather than probing it.
        raise NotImplementedError("process probing is POSIX only")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if sys.platform.startswith("linux"):
        try:
            stat = pathlib.Path(f"/proc/{pid}/stat").read_bytes()
        except OSError:
            return False
        return stat.rsplit(b")", 1)[-1].split()[:1] != [b"Z"]
    return True


def _wait_for_exit(pid: int, timeout: float) -> bool:
    """Wait until process `pid` has ended. False when it still runs after `timeout`."""
    deadline = time.monotonic() + timeout
    while _alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _kill_group(pid: int) -> None:
    """Kill the process group that `pid` leads, or `pid` alone when it leads none.

    run.py starts a capture child in a new session, so the child leads a group holding
    everything it started, such as an agent CLI that never answered.
    """
    kill = getattr(signal, "SIGKILL", signal.SIGTERM)
    try:
        os.killpg(pid, kill)
    except (OSError, AttributeError):
        try:
            os.kill(pid, kill)
        except OSError:
            pass


class HookRunner:
    """Runs `plugin/hooks/run.py <hook> --host <host>` with a payload shaped for that host.

    `home` becomes the child's HOME, and `cwd` its working directory. When `server_env`
    is given, it is written into the host's first client config file as the memvara
    server's env block, which is where the hooks look for the store
    (plugin/hooks/lib/ipc.py). Without it, the hooks find no store and report
    "not configured". `env` is different: it is applied last to the hook process's own
    environment, on top of `child_env`.

    No directory that holds a real agent CLI is on the child's PATH
    (`path_without_agent_clis`). `stubs` are stub agent CLIs put first on it, such as
    `harness.fakes.cli.FakeClis`. `capture` is refused without them, because it starts
    an agent CLI to mine the turn.

    Client configs are written as JSON only. Codex keeps its config in TOML, so a Codex
    run with a store is refused here until the hook-conformance tests add a TOML writer.
    """

    def __init__(self, host: str, *, home: pathlib.Path, cwd: pathlib.Path,
                 server_env: Mapping[str, str] | None = None,
                 env: Mapping[str, str] | None = None,
                 stubs: Stubs | None = None) -> None:
        self.host = host_record(host)
        self.home = pathlib.Path(home)
        self.cwd = pathlib.Path(cwd)
        self.stubs = stubs
        environment = child_env(self.home, env)
        rest = path_without_agent_clis(environment.get("PATH", ""))
        environment["PATH"] = stubs.path(rest) if stubs is not None else rest
        self._env = environment
        #: Captures handed to a child that `run` was told not to wait for. `close` kills
        #: each one with everything it started.
        self._detached: list[int] = []
        if server_env is not None:
            self.write_client_config(server_env)

    @property
    def environment(self) -> dict[str, str]:
        """A copy of the environment each hook runs with."""
        return dict(self._env)

    def write_client_config(self, server_env: Mapping[str, str]) -> pathlib.Path:
        """Write the host's first client config file, holding a memvara server block."""
        if self.host.config_format != "json":
            raise NotImplementedError(
                f"HookRunner writes JSON client configs only, and {self.host.id} keeps a "
                f"{self.host.config_format} one; the hook-conformance tests add that writer")
        path = pathlib.Path(str(self.host.client_configs[0]).replace("~", str(self.home), 1))
        if not path.resolve().is_relative_to(self.home.resolve()):
            raise ValueError(f"refusing to write a client config outside the test's home: "
                             f"{path}")
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

    def run(self, hook: str, *, stdin: str | bytes | None = None,
            timeout: float | None = None, wait_detached: bool = True,
            **fields: Any) -> HookResult:
        """Run one hook and wait for it, within this host's own timeout for that hook.

        `stdin` replaces the payload, as text or as raw bytes. `timeout` replaces the
        host's limit; a hook that the host has no event for has no limit, so a test that
        runs one passes it.

        On a host that hands capture to a child in a new session (`detach_capture`), the
        hook returns at once and the child mines the turn afterwards. `run` then waits
        for the child too, within the same limit, so the result's logs hold what the
        capture did, while `elapsed` is still what the host waited. With
        `wait_detached=False` it does not wait, and `close()` kills the child and
        everything it started.
        """
        if hook == "capture" and self.stubs is None:
            raise NotImplementedError(
                "capture starts an agent CLI to mine the turn; give HookRunner stub CLIs "
                "(stubs=FakeClis(...)) so that a real one is never reached")
        detaches = hook == "capture" and bool(self.host.detach_capture)
        text = json.dumps(self.payload(hook, **fields)) if stdin is None else stdin
        data = text.encode("utf-8") if isinstance(text, str) else text
        limit = float(self.host.timeouts[hook]) if timeout is None else timeout
        before = self._log_sizes()
        started = time.monotonic()
        try:
            done = subprocess.run(
                [sys.executable, str(RUN), hook, "--host", self.host.id], input=data,
                capture_output=True, env=self._env, cwd=str(self.cwd), timeout=limit)
        except subprocess.TimeoutExpired as exc:
            raise HookTimeout(
                f"{hook} on {self.host.id} ran past its limit of {limit}s; "
                f"stdout so far: {_text(exc.stdout)[:300]!r}; "
                f"stderr: {_text(exc.stderr)[-300:]!r}") from None
        elapsed = time.monotonic() - started
        stderr = _text(done.stderr)
        try:
            stdout = done.stdout.decode("utf-8")
        except UnicodeDecodeError:
            raise HookOutputError(f"{hook} on {self.host.id} printed bytes that are not "
                                  f"UTF-8: {done.stdout[:300]!r}") from None
        reply = parse_reply(stdout, what=f"{hook} on {self.host.id}", stderr=stderr)
        logs = self._new_lines(before)
        pid = _detached_pid(logs.get("hooks", ())) if detaches else None
        if pid is not None:
            if not wait_detached:
                self._detached.append(pid)
            elif _wait_for_exit(pid, limit):
                logs = self._new_lines(before)
            else:
                _kill_group(pid)
                raise HookTimeout(f"the capture {self.host.id} handed to pid {pid} ran past "
                                  f"{limit}s")
        return HookResult(exit_code=done.returncode, stdout=stdout, stderr=stderr,
                          reply=reply, elapsed=elapsed, logs=logs, detached_pid=pid)

    def close(self) -> None:
        """Kill every capture child this runner did not wait for, with everything each
        one started. Safe to call more than once."""
        for pid in self._detached:
            _kill_group(pid)
            _wait_for_exit(pid, 5.0)
        self._detached.clear()

    def _log_sizes(self) -> dict[str, int]:
        return {path.stem: path.stat().st_size
                for path in (self.home / HOOKS_HOME).glob("*.log")}

    def _new_lines(self, before: Mapping[str, int]) -> dict[str, tuple[str, ...]]:
        """The lines each hook log gained since `before`, without their timestamps.

        A log the hooks truncated in the meantime, which they do past 64 KB, is read
        whole.
        """
        new: dict[str, tuple[str, ...]] = {}
        for path in sorted((self.home / HOOKS_HOME).glob("*.log")):
            data = path.read_bytes()
            start = before.get(path.stem, 0)
            if len(data) < start:
                start = 0
            lines = data[start:].decode("utf-8", "replace").splitlines()
            if lines:
                new[path.stem] = tuple(_STAMP.sub("", line, count=1) for line in lines)
        return new
