"""Run the crash child, stop it at a named point, and check the store it leaves behind.

`Child(program, home=...)` starts `crash_child.py` with the suite's child environment and
sends it its program. `wait_for(prefix)` reads the child's lines until one starts with
`prefix`, collecting each `ACK` on the way. `kill()` kills the child, the way a crash or
`kill -9` would, and `release()` lets a held child carry on.

`after_crash(path, user, live)` then opens the store in this process, which never had it
open, and checks what the spec calls a correct recovery: the file passes the integrity
checks, every acknowledged write is present and found by `search`, an erased claim has
its erasure record and nothing else, and the next write succeeds without waiting on a
lock the dead process held.
"""

from __future__ import annotations

import json
import pathlib
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Collection, Mapping, Sequence
from types import TracebackType
from typing import Any

from memvara import Memvara

from harness import stores
from harness.crash_child import PAUSES, POINTS
from harness.env import child_env
from harness.invariants import check_store_integrity

__all__ = ["CHILD", "PAUSES", "POINTS", "Child", "CrashHarnessError", "acked_claims",
           "after_crash", "kill_at"]

CHILD = pathlib.Path(__file__).with_name("crash_child.py")
#: The next write after a crash must finish within this many seconds. A lock the dead
#: process still held would make it wait out SQLite's five-second busy timeout, so three
#: seconds tells the two apart and still leaves room for a slow or loaded machine.
NEXT_WRITE_SECONDS = 3.0

_EOF = object()


class CrashHarnessError(AssertionError):
    """The child exited, printed an error, or printed nothing within the timeout."""


class Child:
    """One run of the crash child. Use it as a context manager, so that a child the test
    left running is killed."""

    def __init__(self, program: Mapping[str, Any], *, home: pathlib.Path,
                 env: Mapping[str, str] | None = None, timeout: float = 20.0) -> None:
        self.timeout = timeout
        self.acked: list[dict[str, Any]] = []
        self._lines: queue.Queue[object] = queue.Queue()
        self._stderr: list[str] = []
        self.proc = subprocess.Popen(
            [sys.executable, str(CHILD)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=child_env(home, env), text=True, encoding="utf-8")
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self._send(json.dumps(dict(program)))

    def __enter__(self) -> Child:
        return self

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None,
                 tb: TracebackType | None) -> None:
        if self.proc.poll() is None:
            self.kill()

    # -- reading ---------------------------------------------------------------------------

    def _pump_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._lines.put(line.rstrip("\n"))
        self._lines.put(_EOF)

    def _pump_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr.append(line)

    def stderr_tail(self, lines: int = 20) -> str:
        return "".join(self._stderr[-lines:])

    def wait_for(self, prefix: str) -> str:
        """The rest of the first line that starts with `prefix`. Raises `CrashHarnessError`
        when the child prints an error, exits first, or prints nothing more in time."""
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise CrashHarnessError(
                    f"no line starting with {prefix!r} within {self.timeout:g} s; the "
                    f"child's stderr ends with:\n{self.stderr_tail()}") from None
            if line is _EOF:
                code = self.proc.wait()
                raise CrashHarnessError(
                    f"the child exited with status {code} before printing {prefix!r}; its "
                    f"stderr ends with:\n{self.stderr_tail()}")
            assert isinstance(line, str)
            if line.startswith("ACK "):
                self.acked.append(json.loads(line[4:]))
            if line.startswith("ERROR "):
                raise CrashHarnessError(f"the child failed: {line[6:]}\n{self.stderr_tail()}")
            if line.startswith(prefix):
                return line[len(prefix):].strip()

    # -- control ---------------------------------------------------------------------------

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def release(self) -> None:
        """Let a held child carry on past its point."""
        self._send("go")

    def kill(self) -> int:
        """Kill the child at once, as a crash would, and return its exit status."""
        self.proc.kill()
        return self.proc.wait(timeout=self.timeout)

    def finish(self) -> int:
        """Close the child's input and wait for it to exit; its exit status."""
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        return self.proc.wait(timeout=self.timeout)


def kill_at(program: Mapping[str, Any], home: pathlib.Path, *,
            env: Mapping[str, str] | None = None, timeout: float = 20.0) -> Child:
    """Run `program` until the child reaches its point, then kill it. The returned child
    has exited; its `acked` list says which setup ops it finished."""
    with Child(program, home=home, env=env, timeout=timeout) as child:
        child.wait_for(f"POINT {program['point']}")
        child.kill()
    return child


def acked_claims(child: Child, setup: Sequence[Sequence[Any]]) -> dict[str, str]:
    """The claims the child's `remember` setup ops acknowledged, by id, with the object
    each was written with, which is the text `after_crash` searches for."""
    assert len(child.acked) == len(setup), (
        f"the child acknowledged {len(child.acked)} of {len(setup)} setup ops")
    return {a["ids"][0]: op[1]["object"] for a, op in zip(child.acked, setup)}


def after_crash(path: pathlib.Path, user: str, live: Mapping[str, str], *,
                erased: Collection[str] = (), key: bytes | None = None) -> Memvara:
    """Check a store a killed process left behind, and return an open handle on it.

    `live` maps each acknowledged claim that must still be live to the text `search`
    must find it by. `erased` names acknowledged claims that were erased before the
    crash. `key` opens an encrypted store. The caller closes the handle.
    """
    problems = check_store_integrity(path, key=key)
    assert not problems, "the store is damaged after the crash:\n" + "\n".join(problems)
    mem = (stores.file(path) if key is None else
           stores.file(path, encryption=True, key_env={"MEMVARA_DB_KEY": key.hex()}))
    try:
        for claim_id, text in live.items():
            assert mem.store.get_claim(claim_id) is not None, (
                f"the acknowledged claim {claim_id} ({text!r}) is gone")
            assert mem.store.erasure_record(claim_id) is None, (
                f"the live claim {claim_id} ({text!r}) has an erasure record")
            found = [r.claim.id for r in mem.search(text, k=10, user=user)]
            assert claim_id in found, f"search({text!r}) does not find {claim_id}"
        for claim_id in erased:
            assert mem.store.get_claim(claim_id) is None, (
                f"the erased claim {claim_id} is back")
            assert mem.store.erasure_record(claim_id) is not None, (
                f"the erased claim {claim_id} has no erasure record")
        started = time.monotonic()
        receipt = mem.remember("user", "likes", "the first write after the crash", user=user)
        took = time.monotonic() - started
        assert receipt.added, "the first write after the crash stored nothing"
        assert took < NEXT_WRITE_SECONDS, (
            f"the first write after the crash took {took:.1f} s, as if the dead process "
            f"still held a lock")
    except BaseException:
        mem.close()
        raise
    return mem
