"""Every JSON decoder in the hooks treats nesting too deep to decode as unreadable.

`json.loads` raises `RecursionError`, not `ValueError`, on nesting deeper than Python's
recursion limit. The two readers of a hook's stdin caught only `ValueError`, and a payload
nested 100,000 levels deep crashed every hook body (#346). The same pattern stood at every
other decoder the hooks run: the agent CLI's event stream and reply, the hosted server's
reply, and the daemon's request and reply. Each one here must read such text as it reads
text that is not JSON.
"""

from __future__ import annotations

import pathlib
import socket
import subprocess
import sys
import threading

import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import daemon  # noqa: E402
from core.host import CODEX_CLI  # noqa: E402
from lib import agentic, extract, fast, hosted  # noqa: E402

DEEP = '{"q": ' + "[" * 100_000 + "]" * 100_000 + "}"

#: Deep enough to pass the recursion limit and small enough to fit under the daemon's
#: cap on a request (`daemon.MAX_REQUEST_BYTES`, 64 KB), which closes a longer one first.
DEEP_REQUEST = '{"q": ' + "[" * 20_000 + "]" * 20_000 + "}"


def test_the_payload_is_too_deep_for_json_loads() -> None:
    """The premise: without a guard, this text raises RecursionError, not ValueError."""
    import json  # noqa: PLC0415

    for text in (DEEP, DEEP_REQUEST):
        with pytest.raises(RecursionError):
            json.loads(text)


@pytest.mark.parametrize("decode, unreadable", [
    (lambda: agentic._Watch().feed(DEEP), False),
    (lambda: agentic._proposals(DEEP), None),
    (lambda: extract._decode(DEEP), None),
    (lambda: extract._facts(DEEP), []),
    (lambda: extract._stream(subprocess.CompletedProcess([], 0, DEEP + "\n", ""),
                             CODEX_CLI, "codex"), ("", {})),
    (lambda: hosted._decode(DEEP.encode()), None),
    (lambda: hosted._decode(b"data: " + DEEP.encode()), None),
    (lambda: fast._served(DEEP), None),
], ids=["agent event stream", "agent proposals", "extractor envelope", "extractor facts",
        "extractor event stream", "hosted reply", "hosted event-stream reply",
        "daemon reply"])
def test_a_decoder_reads_nesting_too_deep_as_unreadable(decode, unreadable) -> None:
    assert decode() == unreadable


def test_the_daemon_closes_a_request_nested_too_deeply_without_raising() -> None:
    """The daemon serves each connection on a thread of its own, so an exception there
    ended that thread with a traceback rather than the quiet close any other unreadable
    request gets."""
    served = daemon.Daemon.__new__(daemon.Daemon)
    ours, theirs = socket.socketpair()
    raised: list[BaseException] = []

    def serve() -> None:
        try:
            served._serve(ours)
        except BaseException as exc:  # noqa: BLE001 -- the test reports what escaped
            raised.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    assert len(DEEP_REQUEST) < daemon.MAX_REQUEST_BYTES
    theirs.sendall(DEEP_REQUEST.encode())
    theirs.shutdown(socket.SHUT_WR)
    assert theirs.recv(1) == b""
    thread.join(10)
    theirs.close()
    assert raised == []
