"""Runs one framework's checks inside that framework's virtual environment.

The nightly tests start this file with the environment's own interpreter:

    <environment>/bin/python -I -B probe.py <checks file> <report file>

`-I` keeps out every `PYTHON*` variable and the user's own site-packages, so the process
sees only what the environment installed. `-B` stops it writing bytecode into this
checkout. A checks file holds functions named `check_*`. The probe runs each one once, in
the order the file defines them, and appends its outcome to the report as one JSON line.
A checks file that raises while it loads runs no check: the probe records that failure
once, and every check reports it as the reason it never ran.

Before it loads the checks, the probe installs an audit hook that blocks every attempt to
reach the network and records it in the report: a connection or a datagram to anything
but a unix socket, a lookup of a host name other than localhost, and a reverse lookup of
any address but a loopback one. No check needs the network, so any attempt is a failure,
whichever library made it. The hook sees only what goes through Python's `socket` module. Networking done in
C, or in a child process, goes unseen, so the nightly tests also start the probe with no
credentials in its environment and with every proxy variable pointing at a closed port.

The suite imports this module as well, to list the checks in a checks file and to read a
report back. So it imports only the standard library at module level: the framework's
environment has no pytest, and the suite's environment has no framework.
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib.metadata
import importlib.util
import inspect
import ipaddress
import json
import re
import socket
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

#: How long each step of a run may take: the start, where the probe imports memvara and
#: lists what is installed; loading the checks file; each check; and the interpreter's
#: shutdown after the last one. Past it, `faulthandler` writes every thread's stack to
#: stderr and ends the process with a failure, so the report shows which step never
#: finished.
CHECK_SECONDS = 120.0

#: Host names that are answered without asking a name server.
_LOCAL_NAMES = frozenset({"", "localhost", "localhost.localdomain"})

_SENDS = frozenset({"socket.connect", "socket.sendto", "socket.sendmsg"})
#: The audit events of a name lookup. CPython raises socket.gethostbyname for
#: gethostbyname_ex as well.
_LOOKUPS = frozenset({"socket.getaddrinfo", "socket.gethostbyname",
                      "socket.gethostbyaddr", "socket.getnameinfo"})
#: Lookups that ask a name server about an address, so a numeric address counts.
_REVERSE = frozenset({"socket.gethostbyaddr", "socket.getnameinfo"})


class BlockedNetworkAccess(OSError):
    """Raised in place of a network access.

    An `OSError`, so a library that copes with a refused connection copes with this in
    the same way. That can hide the refusal from the check that caused it, which is why
    the report records every one: a network access fails the environment's network test
    even when the library swallowed the error.
    """


def network_access(event: str, args: tuple[Any, ...]) -> str | None:
    """Describe the network access an audit event stands for, or return None.

    A connection or a datagram counts unless its socket is a unix socket, which never
    leaves the machine. A lookup counts unless the name is empty, localhost or a loopback
    address. A numeric address is answered without a name server too, except by a
    reverse lookup (`gethostbyaddr` or `getnameinfo`), which asks one what the address is
    called.
    """
    if event in _SENDS:
        sock, address = args[0], args[-1]
        if sock.family == getattr(socket, "AF_UNIX", object()):
            return None
        verb = "connect to" if event == "socket.connect" else "send to"
        if address is None:
            # sendmsg on a connected socket names no address, because it sends to the
            # socket's peer.
            try:
                address = sock.getpeername()
            except OSError:
                return "send on a socket with no peer"
        return f"{verb} {_address(address)}"
    if event in _LOOKUPS:
        host = args[0]
        if event == "socket.getnameinfo" and isinstance(host, tuple):
            host = host[0]
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        if host is None or host.lower() in _LOCAL_NAMES or _loopback(host):
            return None
        if _numeric(host) and event not in _REVERSE:
            return None
        return f"look up {host}"
    return None


def _ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return None


def _numeric(host: str) -> bool:
    return _ip(host) is not None


def _loopback(host: str) -> bool:
    address = _ip(host)
    return address is not None and address.is_loopback


def _address(address: Any) -> str:
    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return repr(address)


class Report:
    """Appends one JSON object per line to the report file, from any thread.

    The file is opened for each record rather than held open, so a record written by a
    library's exit handler, after the last check, still reaches the disk.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def emit(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)


def install_guard(report: Report, phase: Callable[[], str]) -> None:
    """Block and record every network access for the rest of this process's life.

    An audit hook cannot be removed, which is why only the probe's own process installs
    one and the suite never does. `phase` names what was running: a check's name,
    "start" or "load" before the first check, or "exit" after the last.
    """

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if not event.startswith("socket."):
            return
        what = network_access(event, args)
        if what is None:
            return
        report.emit({"kind": "network", "access": what, "event": event, "phase": phase(),
                     "thread": threading.current_thread().name,
                     "stack": traceback.format_stack(limit=16)[:-1]})
        raise BlockedNetworkAccess(f"the probe blocked a network access: {what}")

    sys.addaudithook(hook)


class Context:
    """What a check is given: a temporary folder, and stores that are closed after it."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._stores: list[Any] = []

    def memvara(self, **options: Any) -> Any:
        """A new in-memory store with the hashing embedder and no model.

        It is bound to the user "alice" unless `options` names another user. The hashing
        embedder and `NullLLM` are what the rest of the suite uses, and with them a check
        can never reach a model.
        """
        from memvara import HashingEmbedder, Memvara, NullLLM

        options.setdefault("user", "alice")
        # A copy of `memory()` in tests/harness/stores.py, which this process cannot
        # import: a framework's environment holds memvara and the framework, not the
        # suite's harness. Keep the two recipes the same.
        store = Memvara(embedder=HashingEmbedder(dim=512), llm=NullLLM(), **options)
        self._stores.append(store)
        return store

    def close(self) -> None:
        for store in self._stores:
            store.close()


Check = Callable[[Context], None]


def checks(module: ModuleType) -> list[tuple[str, Check]]:
    """The check functions a module defines, in the order it defines them."""
    return [(name, value) for name, value in vars(module).items()
            if name.startswith("check_") and inspect.isfunction(value)
            and value.__module__ == module.__name__]


def load(path: Path) -> ModuleType:
    """Import a checks file from its path, so the probe needs nothing on `sys.path`."""
    name = f"memvara_probe_{Path(path).stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load checks from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def distributions() -> dict[str, str]:
    """Every distribution this interpreter can import, by its normalised name."""
    # The same normalisation as `_DESCRIBE` in environments.py, which also runs inside an
    # environment, and as packaging's `canonicalize_name`, which the suite uses to look
    # these names up. An environment cannot import packaging, so the rule is written out
    # here. Keep all three the same.
    return {re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version
            for d in importlib.metadata.distributions() if d.metadata["Name"]}


def run(checks_file: Path, report_file: Path, *,
        check_seconds: float = CHECK_SECONDS) -> int:
    """Run every check in `checks_file`, appending each outcome to `report_file`.

    One watchdog covers the whole run, from the first line here to the end of the
    process. It is restarted at the start of each step, so each step gets the whole
    limit and a hang prints the stacks of the step that hung. It is never stopped, so no
    step runs without it. The steps are the start (importing memvara and listing what is
    installed), loading the checks file, each check with its records, and the shutdown.

    A checks file that raises while it loads runs no check. Its failure is recorded once,
    and the probe exits with status 1.
    """
    _watch(check_seconds)
    report = Report(report_file)
    phase = ["start"]
    install_guard(report, lambda: phase[0])
    try:
        import memvara
        location: str | None = memvara.__file__
    except ImportError:
        location = None
    report.emit({"kind": "start", "python": sys.version, "executable": sys.executable,
                 "memvara": location, "distributions": distributions()})
    phase[0] = "load"
    _watch(check_seconds)
    try:
        module = load(checks_file)
    except Exception as exc:  # a file that cannot load runs nothing; each check says why
        report.emit({"kind": "load_error", "checks": str(checks_file), **_failure(exc)})
        phase[0] = "exit"
        _watch(check_seconds)
        return 1
    for name, function in checks(module):
        phase[0] = name
        _watch(check_seconds)
        report.emit({"kind": "begin", "check": name})
        started = time.monotonic()
        record: dict[str, Any] = {"kind": "result", "check": name, "passed": True}
        with tempfile.TemporaryDirectory(prefix="check-") as folder:
            context = Context(Path(folder))
            try:
                function(context)
            except Exception as exc:  # every failure is a result, and the next check runs
                record.update(passed=False, **_failure(exc))
            finally:
                context.close()
        record["seconds"] = round(time.monotonic() - started, 3)
        report.emit(record)
    phase[0] = "exit"
    # The interpreter still has to shut down after this: it waits for every thread that
    # is not a daemon, then runs the exit handlers, where telemetry often sends. A
    # framework that left a thread running would keep the handlers from ever running, so
    # the shutdown is a step of its own, with the whole limit.
    _watch(check_seconds)
    report.emit({"kind": "end"})
    return 0


def _watch(seconds: float) -> None:
    """Restart the watchdog. If the current step is still running after `seconds`,
    `faulthandler` writes every thread's stack to stderr and ends the process with
    status 1. There is only one watchdog, so restarting it ends the previous step's."""
    faulthandler.dump_traceback_later(seconds, exit=True)


def _failure(exc: BaseException) -> dict[str, Any]:
    """An exception as a report records it."""
    return {"error_type": type(exc).__name__, "message": str(exc)[:4000],
            "traceback": "".join(traceback.format_exception(exc))[-12000:]}


# -- reading a report back, in the suite ------------------------------------------------


@dataclass
class Result:
    """One check's outcome."""

    check: str
    passed: bool
    seconds: float = 0.0
    error_type: str = ""
    message: str = ""
    traceback: str = ""

    def describe(self) -> str:
        if self.passed:
            return f"{self.check} passed"
        return (f"{self.check} failed with {self.error_type}: {self.message}\n"
                f"{self.traceback}")


@dataclass
class Run:
    """Everything one probe run reported."""

    start: dict[str, Any] | None = None
    results: dict[str, Result] = field(default_factory=dict)
    begun: list[str] = field(default_factory=list)
    network: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False
    #: The failure that stopped the checks file from loading, if one did. Its `check` is
    #: the checks file's path.
    load_error: Result | None = None

    def result(self, check: str) -> Result:
        """The check's outcome, or a failure that says why the check has none."""
        found = self.results.get(check)
        if found is not None:
            return found
        if self.load_error is not None:
            failed = self.load_error
            return Result(check, False, error_type="ChecksNotLoaded",
                          message=(f"it never ran: loading the checks file {failed.check} "
                                   f"failed with {failed.error_type}: {failed.message}"),
                          traceback=failed.traceback)
        if self.finished:
            reason = ("it never ran: the probe finished without it, so the checks file "
                      "defines no check of that name")
        elif check in self.begun:
            reason = "it started and never finished: the probe stopped while running it"
        elif self.begun:
            reason = f"it never ran: the probe stopped while running {self.begun[-1]}"
        else:
            reason = "it never ran: the probe stopped before its first check"
        return Result(check, False, error_type="ProbeStopped", message=reason)


def read(path: Path) -> Run:
    """Read a report. A line cut short by a killed process is left out."""
    found = Run()
    path = Path(path)
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = record.get("kind")
        if kind == "start":
            found.start = record
        elif kind == "begin":
            found.begun.append(record["check"])
        elif kind == "result":
            found.results[record["check"]] = Result(
                record["check"], bool(record["passed"]), float(record.get("seconds", 0.0)),
                record.get("error_type", ""), record.get("message", ""),
                record.get("traceback", ""))
        elif kind == "network":
            found.network.append(record)
        elif kind == "load_error":
            found.load_error = Result(
                record["checks"], False, 0.0, record.get("error_type", ""),
                record.get("message", ""), record.get("traceback", ""))
        elif kind == "end":
            found.finished = True
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one framework's checks and write their outcomes as JSON lines.")
    parser.add_argument("checks", type=Path, help="the checks file to run")
    parser.add_argument("report", type=Path, help="the report file to append to")
    parser.add_argument("--check-seconds", type=float, default=CHECK_SECONDS,
                        help="how long loading, each check and the shutdown may take "
                             "before the probe stops")
    options = parser.parse_args(argv)
    return run(options.checks, options.report, check_seconds=options.check_seconds)


if __name__ == "__main__":
    sys.exit(main())
