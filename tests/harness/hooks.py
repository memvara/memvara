"""The plugin's hook scripts, run in a child process the way a client runs them."""

from __future__ import annotations

import functools
import importlib
import json
import os
import pathlib
import re
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .env import REPO, child_env

HOOKS_DIR = REPO / "plugin" / "hooks"
RUN = HOOKS_DIR / "run.py"

#: Where the hooks keep their logs, their state and the recall daemon's socket, under the
#: home directory a run is given.
HOOKS_HOME = pathlib.Path(".memvara") / ".hooks"

#: The longest unix socket path macOS accepts. Its `sun_path` field holds 104 bytes, and
#: the path must end with a NUL byte inside them.
MAX_SOCKET_PATH = 103


def daemon_socket_path(home: pathlib.Path) -> pathlib.Path:
    """The path of a recall daemon's socket under `home`, which is as long as it gets.

    The hooks put the socket in `run/` under their home and name it `recall-` and 16 hex
    digits (plugin/hooks/lib/ipc.py, `RUNTIME_DIR` and `socket_path`).
    """
    return home / HOOKS_HOME / "run" / f"recall-{'0' * 16}.sock"


#: Why the recall daemon cannot run here. HookRunner refuses `daemon=True` with these
#: words, and a test that skips for the same reason uses them too, so that the skip
#: matches its rule in tests/harness/skips.py.
NO_UNIX_SOCKETS = "the recall daemon listens on a unix socket, which Windows lacks"

#: The moment a hook log line was written, which starts every line. `HookResult.logs`
#: leaves it out, so a test can compare lines from two runs.
_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z) ")

#: The line run.py writes to hooks.log when it hands capture to a child in a new session.
_DETACHED = re.compile(r"^detached hook=capture host=\S+ pid=(\d+)")

#: Where each host keeps its MCP servers, and in what shape: where a user who runs a
#: local store configures it. The hooks look for the store in the files the host record
#: lists (`client_configs`), so writing here lets a test check that the two agree.
#:
#: * Claude Code: `mcpServers` in ~/.claude.json, the first file hosts/claude.py lists.
#: * Codex: a `[mcp_servers.<name>]` table in ~/.codex/config.toml (hosts/codex.py).
#: * GitHub Copilot CLI: `mcpServers` in ~/.copilot/mcp-config.json, each server with a
#:   `type`; hosts/copilot.py lists that file second.
#: * Cursor: `mcpServers` in ~/.cursor/mcp.json, from Cursor's MCP documentation
#:   (https://cursor.com/docs/mcp), which hosts/cursor.py does not list.
#: * OpenCode: `mcp` in ~/.config/opencode/opencode.json, each local server with a
#:   `command` list and an `environment` object, from OpenCode's MCP documentation
#:   (https://opencode.ai/docs/mcp-servers/) and the opencode-memvara README.
CLIENT_CONFIGS: dict[str, tuple[str, str]] = {
    "claude": ("~/.claude.json", "mcpServers"),
    "codex": ("~/.codex/config.toml", "toml"),
    "copilot": ("~/.copilot/mcp-config.json", "copilot"),
    "cursor": ("~/.cursor/mcp.json", "mcpServers"),
    "opencode": ("~/.config/opencode/opencode.json", "opencode"),
}

#: The environment variable that names the file a daemon records its pid in.
_DAEMONS_VAR = "HOOK_TEST_DAEMONS"

#: The `sitecustomize` module that a runner allowing the daemon puts first on the hook's
#: PYTHONPATH. Python imports `sitecustomize` when it starts, in every process that
#: inherits that path, so each recall daemon the hooks start records its pid. `close()`
#: can then stop every one, including a daemon that no socket path leads to any more.
#: A `sitecustomize` it shadows, such as a Linux distribution's, still runs after it.
_DAEMON_SPY = f"""\
import importlib.machinery, importlib.util, os, sys
_log = os.environ.get("{_DAEMONS_VAR}")
if _log and sys.orig_argv[1:2] and sys.orig_argv[1].endswith("daemon.py"):
    with open(_log, "a", encoding="utf-8") as _fh:
        _fh.write(str(os.getpid()) + "\\n")
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.machinery.PathFinder.find_spec(
    "sitecustomize", [p for p in sys.path if os.path.abspath(p or os.curdir) != _here])
if _spec is not None and _spec.loader is not None:
    _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
"""

#: The environment variable that carries `HookRunner.patches` to the launcher below.
_PATCHES_VAR = "HOOK_TEST_PATCHES"

#: The launcher's exit status when a patch names an attribute the hooks do not have.
_BAD_PATCH = 97

#: What a hook process runs instead of run.py when a test patches it: it binds the host
#: first, as run.py does, because some hook modules read the host when they are imported,
#: then sets each patched module attribute, then calls run.py's own `main`. It is passed
#: with `python -c`, so it is never a file that pytest's doctest collection would import.
#:
#: A patch replaces the attribute on the one module it names, so it reaches only code that
#: reads that attribute when it runs. A module that imported the value by name keeps its
#: own copy, and so does a default argument, which Python fixes when it defines the
#: function. `lib.fast.REWRITE_WAIT_SEC` is such a value: recall.py imports it by name
#: (`from lib.fast import REWRITE_WAIT_SEC`) and passes its own copy, and lib/fast.py reads
#: it only as a default argument, so a patch of it changes nothing. Patch the copy the code
#: reads instead, here `recall.REWRITE_WAIT_SEC`. No test patches a value like that today.
_LAUNCHER = f"""\
import importlib, json, os, sys
sys.path.insert(0, sys.argv[1])
argv = sys.argv[2:]
from core import host as _host
_host.use(importlib.import_module("hosts." + argv[argv.index("--host") + 1]).HOST)
for dotted, value in json.loads(os.environ.pop("{_PATCHES_VAR}")).items():
    module_name, _, name = dotted.rpartition(".")
    module = importlib.import_module(module_name)
    if not hasattr(module, name):
        sys.stderr.write(dotted + " names nothing the hooks have\\n")
        raise SystemExit({_BAD_PATCH})
    setattr(module, name, value)
import run
try:
    status = run.main(argv)
except BaseException:
    status = 0
raise SystemExit(status)
"""


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


@functools.cache
def host_ids() -> tuple[str, ...]:
    """Every host the plugin has a record for: the modules in plugin/hooks/hosts. Read once
    per run, because the records do not change while the tests run."""
    return tuple(sorted(path.stem for path in (HOOKS_DIR / "hosts").glob("*.py")
                        if path.stem != "__init__"))


@functools.cache
def agent_clis() -> frozenset[str]:
    """The program name of every agent CLI a capture can start: each host's own
    extractor, and the `claude` CLI that every host falls back to
    (`CLAUDE_CLI` in plugin/hooks/core/host.py). Worked out once per run, like
    `host_ids`."""
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


def short_dir(prefix: str) -> pathlib.Path:
    """A new private directory with a short path, which the caller removes.

    A runner that allows the recall daemon needs a home like this. The daemon's socket
    lives under the home, and macOS refuses a unix socket path longer than
    `MAX_SOCKET_PATH`. A pytest temporary directory under a long TMPDIR can pass that
    length before the hooks add their part, so this makes the home in the system's
    temporary directory, and makes it again in /tmp when the daemon's socket would not fit
    under it.

    The length measured is that of the resolved path, because `child_env` resolves a home
    before it hands it to the hooks, and a temporary directory is often reached through a
    symbolic link: on macOS, /tmp is /private/tmp.
    """
    home = pathlib.Path(tempfile.mkdtemp(prefix=f"mv-{prefix}-"))
    fits = len(os.fsencode(daemon_socket_path(home.resolve()))) <= MAX_SOCKET_PATH
    if fits or not os.path.isdir("/tmp"):
        return home
    home.rmdir()
    return pathlib.Path(tempfile.mkdtemp(prefix=f"mv-{prefix}-", dir="/tmp"))


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


#: How a TOML basic string writes each character that it cannot hold as itself.
_TOML_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n",
                 "\f": "\\f", "\r": "\\r"}


def toml_string(text: str) -> str:
    r"""`text` as a TOML basic string.

    >>> print(toml_string('say "hi"'))
    "say \"hi\""
    >>> print(toml_string("C:\\stores\\memory.db"))
    "C:\\stores\\memory.db"
    >>> print(toml_string("tab\there, bell\x07"))
    "tab\there, bell\u0007"
    >>> toml_string("\ud800")
    Traceback (most recent call last):
    ...
    ValueError: TOML cannot hold the lone surrogate '\ud800'
    """
    out = []
    for char in text:
        code = ord(char)
        if char in _TOML_ESCAPES:
            out.append(_TOML_ESCAPES[char])
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04x}")
        elif 0xD800 <= code <= 0xDFFF:
            raise ValueError(f"TOML cannot hold the lone surrogate {char!r}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def _toml_key(name: str) -> str:
    """`name` as a TOML key: bare when TOML allows that, quoted otherwise."""
    return name if re.fullmatch(r"[A-Za-z0-9_-]+", name) else toml_string(name)


def toml_server_block(name: str, command: str, args: Sequence[str],
                      env: Mapping[str, str]) -> str:
    """One MCP server as Codex keeps it in ~/.codex/config.toml: a `[mcp_servers.<name>]`
    table, with its environment as a sub-table.

    >>> print(toml_server_block("memvara", "python3", ["-m", "memvara.server"],
    ...                         {"MEMVARA_USER": "tester"}))
    [mcp_servers.memvara]
    command = "python3"
    args = ["-m", "memvara.server"]
    <BLANKLINE>
    [mcp_servers.memvara.env]
    MEMVARA_USER = "tester"
    <BLANKLINE>
    """
    table = f"mcp_servers.{_toml_key(name)}"
    lines = [f"[{table}]",
             f"command = {toml_string(command)}",
             "args = [" + ", ".join(toml_string(arg) for arg in args) + "]",
             "",
             f"[{table}.env]"]
    lines += [f"{_toml_key(key)} = {toml_string(value)}" for key, value in env.items()]
    return "\n".join(lines) + "\n"


def _detached_pid(lines: Sequence[str]) -> int | None:
    """The pid in run.py's line saying it handed capture to a child, if there is one."""
    for line in lines:
        found = _DETACHED.match(line)
        if found:
            return int(found.group(1))
    return None


def _lines_since(path: pathlib.Path, start: int) -> tuple[str, ...]:
    """The lines of the log at `path` past byte `start`, without their timestamps, or none
    when there is no such log. A log now shorter than `start` was truncated by the hooks,
    which they do past 64 KB, so it is read whole."""
    try:
        with open(path, "rb") as log:
            if os.fstat(log.fileno()).st_size < start:
                start = 0
            log.seek(start)
            data = log.read()
    except FileNotFoundError:
        return ()
    return tuple(_STAMP.sub("", line, count=1)
                 for line in data.decode("utf-8", "replace").splitlines())


def socket_peer_pid(path: pathlib.Path) -> int | None:
    """The pid of the process listening on the unix socket at `path`, or None when
    nothing accepts a connection there.

    The kernel reports it (`LOCAL_PEERPID` on macOS, `SO_PEERCRED` on Linux), so it names
    the listener itself, whatever its command line says. Every recall daemon runs the same
    command, so a command line could not tell this test's daemon from another's. When no
    file is at `path`, it answers None without opening a socket.
    """
    if sys.platform == "win32" or not os.path.exists(path):
        return None
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        probe.connect(str(path))
        if sys.platform == "darwin":
            # SOL_LOCAL and LOCAL_PEERPID, from <sys/un.h>; Python names neither.
            raw = probe.getsockopt(0, 0x002, 4)
        elif sys.platform.startswith("linux"):
            raw = probe.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                   struct.calcsize("3i"))
        else:
            return None
        return int(struct.unpack_from("i", raw)[0])
    except OSError:
        return None
    finally:
        probe.close()


def process_alive(pid: int) -> bool:
    """Whether process `pid` is still running. POSIX only.

    A process that has ended but not yet been reaped, a zombie, counts as ended. It still
    answers signal 0, and whatever adopted it may reap it late, or never, as when the
    tests run as process 1 in a container. So the process's state is read: from /proc on
    Linux, and from `ps` elsewhere.
    """
    _refuse_windows()
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not _state_says_ended(pid)


def _refuse_windows() -> None:
    if sys.platform == "win32":
        # os.kill with signal 0 terminates the process on Windows rather than probing it.
        raise NotImplementedError("process probing is POSIX only")


def _answers_signal_0(pid: int) -> bool:
    """Whether process `pid` exists, counting one that has ended and has not been reaped.
    It costs one system call."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _state_says_ended(pid: int) -> bool:
    """Whether the process table says process `pid` has ended: it is gone, or it is a
    zombie. This is the check that sees a process nothing has reaped yet. It reads /proc on
    Linux and starts `ps` elsewhere, which takes a few milliseconds."""
    if sys.platform.startswith("linux"):
        try:
            stat = pathlib.Path(f"/proc/{pid}/stat").read_bytes()
        except OSError:
            return True
        return stat.rsplit(b")", 1)[-1].split()[:1] == [b"Z"]
    try:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                               text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return not state or state.startswith("Z")


def _runs_the_daemon(pid: int) -> bool:
    """Whether process `pid` is running this checkout's plugin/hooks/daemon.py, which
    `close()` checks before it kills a recorded pid, so a pid the system has since given
    to another process is left alone."""
    daemon = str(HOOKS_DIR / "daemon.py")
    if sys.platform.startswith("linux"):
        try:
            argv = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except OSError:
            return False
        return daemon.encode() in argv
    try:
        done = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True,
                              text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return daemon in done.stdout


#: How often `_wait_for_exit` reads a process's state where that starts `ps`, in seconds.
#: Signal 0, sent on every poll, sees a process once it has been reaped; only the state
#: sees one that has ended and not been reaped, so it is still read, only less often.
_STATE_EVERY = 0.25


def _wait_for_exit(pid: int, timeout: float) -> bool:
    """Wait until process `pid` has ended. False when it still runs after `timeout`.

    It polls every 20 ms with signal 0. The state, which also sees a process that has
    ended and not been reaped, is read on the first poll and then every `_STATE_EVERY`
    seconds where reading it starts `ps`, and on every poll on Linux.
    """
    _refuse_windows()
    every = 0.0 if sys.platform.startswith("linux") else _STATE_EVERY
    deadline = time.monotonic() + timeout
    read_state_at = time.monotonic()
    while _answers_signal_0(pid):
        now = time.monotonic()
        if now >= read_state_at:
            if _state_says_ended(pid):
                return True
            read_state_at = now + every
        if now >= deadline:
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
    is given, it is written as the memvara server's env block into the file where the
    host keeps its MCP servers, in that host's shape (CLIENT_CONFIGS). The hooks look for
    the store in the files the host record lists (plugin/hooks/lib/ipc.py); without a
    store, they report "not configured". `env` is different: it is applied last to the
    hook process's own environment, on top of `child_env`, and a `MEMVARA_DB` there wins
    over any client config.

    No directory that holds a real agent CLI is on the child's PATH
    (`path_without_agent_clis`). `stubs` are stub agent CLIs put first on it, such as
    `harness.fakes.cli.FakeClis`. `capture` is refused without them, because it starts
    an agent CLI to mine the turn.

    `daemon=True` lets the recall hook start its background daemon, which `child_env`
    otherwise forbids. The daemon outlives the hook and idles for 30 minutes, so a test
    that allows it calls `close()` when it ends. Its socket lives under `home`, and macOS
    refuses a unix socket path of 104 bytes or more, so such a home needs a short path,
    which `short_dir` makes.

    `patches` sets module attributes in the hook process before the hook runs, such as
    `{"lib.hosted.TIMEOUT_SEC": 0.25}`, so a test can shrink one of a hook's time limits
    instead of waiting it out. A patch that names an attribute the hooks do not have is
    refused with `ValueError`, because a limit that was renamed would otherwise leave the
    test waiting out the real one. A patch reaches only code that reads the attribute from
    that module when it runs; the note above `_LAUNCHER` names a value it cannot reach.
    """

    def __init__(self, host: str, *, home: pathlib.Path, cwd: pathlib.Path,
                 server_env: Mapping[str, str] | None = None,
                 env: Mapping[str, str] | None = None,
                 stubs: Stubs | None = None, daemon: bool = False,
                 patches: Mapping[str, float] | None = None) -> None:
        if daemon and not hasattr(socket, "AF_UNIX"):
            raise NotImplementedError(NO_UNIX_SOCKETS)
        self.host = host_record(host)
        self.home = pathlib.Path(home)
        self.cwd = pathlib.Path(cwd)
        self.stubs = stubs
        self.patches = dict(patches or {})
        environment = child_env(self.home, env)
        rest = path_without_agent_clis(environment.get("PATH", ""))
        environment["PATH"] = stubs.path(rest) if stubs is not None else rest
        #: Where each daemon the hooks start records its pid, or None when the daemon is
        #: not allowed.
        self._daemon_log: pathlib.Path | None = None
        if daemon:
            environment.pop("MEMVARA_DAEMON", None)
            spy = self.home / ".hookrunner"
            spy.mkdir(parents=True, exist_ok=True)
            (spy / "sitecustomize.py").write_text(_DAEMON_SPY, encoding="utf-8")
            environment["PYTHONPATH"] = os.pathsep.join(
                part for part in (str(spy), environment.get("PYTHONPATH", "")) if part)
            self._daemon_log = spy / "daemons"
            environment[_DAEMONS_VAR] = str(self._daemon_log)
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
        """Write a memvara server block into the file where this host keeps its MCP
        servers, in that host's shape (CLIENT_CONFIGS)."""
        known = CLIENT_CONFIGS.get(self.host.id)
        if known is None:
            raise NotImplementedError(
                f"HookRunner does not know where {self.host.id} keeps its MCP servers")
        where, shape = known
        path = pathlib.Path(where.replace("~", str(self.home), 1))
        if not path.resolve().is_relative_to(self.home.resolve()):
            raise ValueError(f"refusing to write a client config outside the test's home: "
                             f"{path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        command, args, env = sys.executable, ["-m", "memvara.server"], dict(server_env)
        if shape == "toml":
            text = toml_server_block("memvara", command, args, env)
        elif shape == "opencode":
            text = json.dumps({"$schema": "https://opencode.ai/config.json", "mcp": {
                "memvara": {"type": "local", "command": [command, *args],
                            "environment": env, "enabled": True}}})
        elif shape == "copilot":
            text = json.dumps({"mcpServers": {"memvara": {
                "type": "local", "command": command, "args": args, "env": env,
                "tools": ["*"]}}})
        else:
            text = json.dumps({"mcpServers": {"memvara": {
                "command": command, "args": args, "env": env}}})
        path.write_text(text, encoding="utf-8")
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
        if detaches and self.patches:
            raise ValueError(f"patches reach the hook process only, and on {self.host.id} "
                             f"capture runs in a child that run.py starts afresh")
        text = json.dumps(self.payload(hook, **fields)) if stdin is None else stdin
        data = text.encode("utf-8") if isinstance(text, str) else text
        limit = float(self.host.timeouts[hook]) if timeout is None else timeout
        command = [sys.executable, str(RUN), hook, "--host", self.host.id]
        env = self._env
        if self.patches:
            command = [sys.executable, "-c", _LAUNCHER, str(HOOKS_DIR), hook,
                       "--host", self.host.id]
            env = {**env, _PATCHES_VAR: json.dumps(self.patches)}
        before = self._log_sizes()
        started = time.monotonic()
        try:
            done = subprocess.run(command, input=data, capture_output=True, env=env,
                                  cwd=str(self.cwd), timeout=limit)
        except subprocess.TimeoutExpired as exc:
            raise HookTimeout(
                f"{hook} on {self.host.id} ran past its limit of {limit}s; "
                f"stdout so far: {_text(exc.stdout)[:300]!r}; "
                f"stderr: {_text(exc.stderr)[-300:]!r}") from None
        elapsed = time.monotonic() - started
        stderr = _text(done.stderr)
        if self.patches and done.returncode == _BAD_PATCH:
            raise ValueError(f"a patch was refused: {stderr.strip()[-300:]}")
        try:
            stdout = done.stdout.decode("utf-8")
        except UnicodeDecodeError:
            raise HookOutputError(f"{hook} on {self.host.id} printed bytes that are not "
                                  f"UTF-8: {done.stdout[:300]!r}") from None
        reply = parse_reply(stdout, what=f"{hook} on {self.host.id}", stderr=stderr)
        pid = None
        if detaches:
            # run.py names the child in hooks.log, so only that log is read here. Every log
            # is read once below, after the child has ended, so the logs hold what it did.
            pid = _detached_pid(_lines_since(self.home / HOOKS_HOME / "hooks.log",
                                             before.get("hooks", 0)))
        if pid is not None:
            if not wait_detached:
                self._detached.append(pid)
            elif not _wait_for_exit(pid, limit):
                _kill_group(pid)
                raise HookTimeout(f"the capture {self.host.id} handed to pid {pid} ran past "
                                  f"{limit}s")
        return HookResult(exit_code=done.returncode, stdout=stdout, stderr=stderr,
                          reply=reply, elapsed=elapsed, logs=self._new_lines(before),
                          detached_pid=pid)

    def daemon_sockets(self) -> list[pathlib.Path]:
        """The recall daemons' socket files under this runner's home."""
        return sorted((self.home / HOOKS_HOME / "run").glob("recall-*.sock"))

    def wait_for_daemon(self, timeout: float = 10.0) -> tuple[pathlib.Path, int]:
        """The socket and the pid of a recall daemon under this home, once it accepts a
        connection. The recall hook starts one after it has answered, and the daemon
        opens the store before it listens, so it can take a moment to appear."""
        deadline = time.monotonic() + timeout
        while True:
            for path in self.daemon_sockets():
                pid = socket_peer_pid(path)
                if pid is not None:
                    return path, pid
            if time.monotonic() >= deadline:
                raise HookTimeout(f"no recall daemon under {self.home} accepted a "
                                  f"connection within {timeout}s")
            time.sleep(0.05)

    def daemon_pids(self) -> list[int]:
        """The pid of every recall daemon the hooks started under this runner, in the
        order they started, whether or not each still runs."""
        if self._daemon_log is None or not self._daemon_log.exists():
            return []
        return [int(line) for line in self._daemon_log.read_text().split()]

    def close(self) -> None:
        """Kill every recall daemon the hooks started under this runner or that listens
        under its home, and every capture child it did not wait for, each with everything
        it started. Safe to call more than once."""
        for pid in self.daemon_pids():
            if _runs_the_daemon(pid):
                _kill_group(pid)
                _wait_for_exit(pid, 5.0)
        for path in self.daemon_sockets():
            listener = socket_peer_pid(path)
            if listener is not None:
                _kill_group(listener)
                _wait_for_exit(listener, 5.0)
            try:
                path.unlink()
            except OSError:
                pass
        for pid in self._detached:
            _kill_group(pid)
            _wait_for_exit(pid, 5.0)
        self._detached.clear()

    def _log_sizes(self) -> dict[str, int]:
        """The size of each hook log, keyed by the file's stem, from one listing of the
        log directory."""
        sizes: dict[str, int] = {}
        try:
            with os.scandir(self.home / HOOKS_HOME) as entries:
                for entry in entries:
                    if entry.name.endswith(".log") and entry.is_file():
                        sizes[entry.name[:-len(".log")]] = entry.stat().st_size
        except FileNotFoundError:
            pass
        return sizes

    def _new_lines(self, before: Mapping[str, int]) -> dict[str, tuple[str, ...]]:
        """The lines each hook log gained since `before`, without their timestamps. The
        log directory is listed once, and a log whose size has not changed is not read."""
        new: dict[str, tuple[str, ...]] = {}
        for stem, size in sorted(self._log_sizes().items()):
            if size == before.get(stem, 0):
                continue
            lines = _lines_since(self.home / HOOKS_HOME / f"{stem}.log", before.get(stem, 0))
            if lines:
                new[stem] = lines
        return new
