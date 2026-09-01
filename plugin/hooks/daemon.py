#!/usr/bin/env python3
"""A resident store handle, so recall costs a socket round trip instead of an import.

The per-prompt hook spends 148ms, and only 6ms of that is the query: 95ms is
`import memvara`, the rest interpreter startup and opening the store. None of that work is
per-prompt work — it is the same every time — so this process does it once and answers on
a unix socket in about 12ms, measured end to end including the client.

It exists to be disposable. Nothing depends on it running, nothing breaks when it dies,
and every client falls back to querying in-process. That is the property that makes a
background process acceptable here: the worst case is the old speed, never a lost prompt.

How it stops running, since nothing tells it to:

* **Idle timeout.** Claude Code sends no shutdown on exit, so a daemon that has not been
  asked anything for 30 minutes exits by itself. This is what keeps abandoned processes
  from accumulating across days of sessions.
* **Superseded by new code.** The socket name embeds a digest of the hook sources, so
  edited code addresses a different socket. The old daemon is simply never contacted again
  and idles out. No restart step to remember, and no window where stale logic is served.
* **Singleton by bind.** Two sessions starting at once both try to bind; the loser sees
  `EADDRINUSE` and exits quietly, leaving the winner serving both.

Reads only. It never writes to the store, so a crash cannot corrupt anything.

**A failed query is not an empty answer.** It used to be: every exception collapsed to `""`,
which the client could not tell from "this store has nothing relevant" and therefore treated
as authoritative. One wedged client behind this process then disabled recall for a whole
session, silently, while the fallback that exists for exactly that case never fired. The
reply is JSON now -- `{"ok": true, "text": ...}` or `{"ok": false}` -- so the client can tell
the two apart and fall through on the second.

Changing the wire format needs no version negotiation: the socket name digests the hook
sources, so a daemon running the old code is addressed by a different name and the two never
meet.
"""

from __future__ import annotations

import json
import os
import socket
import os.path
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.ipc import IDLE_TIMEOUT_SEC, socket_path, store_key  # noqa: E402
from lib.open import open_store  # noqa: E402

#: Requests are small; anything larger is not a query we generated.
MAX_REQUEST_BYTES = 64 * 1024

#: Consecutive failed queries after which this process gives up and exits.
#:
#: A backend that has broken permanently -- expired credentials, a session the server will
#: never honour again -- fails every query identically, and holding the socket for the full
#: idle timeout means every prompt in that window pays a round trip to learn nothing. Exiting
#: hands the address back: the next client finds no daemon, takes the fallback route, and
#: spawns a replacement that opens a fresh backend. Small, because the cost of being wrong is
#: one respawn and the cost of being right is the rest of the session.
MAX_CONSECUTIVE_FAILURES = 3


def _listening(path: str) -> bool:
    """Whether anything is accepting connections on `path`.

    Connect and drop it. A daemon mid-answer still completes the accept, so this says
    "alive" for a busy one where waiting for a reply would say "dead" -- which is the
    difference between tidying a dead address and unlinking a live one.
    """
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        probe.connect(path)
        return True
    except OSError:
        return False
    finally:
        probe.close()


class Daemon:
    def __init__(self, path: str, store: object) -> None:
        self.path = path
        self.store = store
        self.last_seen = time.monotonic()
        self._lock = threading.Lock()
        self.failures = 0

    # -- serving ---------------------------------------------------------------

    def _answer(self, request: dict) -> dict:
        query = str(request.get("q") or "").strip()
        if not query:
            return {"ok": True, "text": ""}
        try:
            kwargs = {
                "k": int(request.get("k") or 6),
                "budget": int(request.get("budget") or 700),
            }
            floor = float(request.get("min_score") or 0.0)
            if floor:
                # Sent only when there is a floor to apply, which is exactly what
                # `lib.fast.recall` does on the direct path. Passing it unconditionally
                # looked harmless -- `min_score=0.0` filters nothing -- and was not: a
                # backend whose `recall()` predates the argument raises `TypeError` on
                # every call, so the daemon route answered nothing at all while the direct
                # route answered normally. That is the one divergence a daemon may never
                # have, and it was introduced by the change whose comment said so.
                kwargs["min_score"] = floor
        except (TypeError, ValueError):
            # A request this process cannot parse. Reported as a failed query, but
            # deliberately NOT counted against `failures`: the backend is fine, and letting
            # a malformed request retire a healthy daemon would hand any buggy client a way
            # to turn recall off for the session. It used to raise from here, past a
            # docstring promising it never would -- `_serve` caught the ValueError and
            # dropped the connection without a reply, which the client reads as "no daemon"
            # and survives, so the bug was invisible from every side.
            # No reason field, deliberately for now. A refused query falls through to the
            # client's own hosted call, which asks the same refused question again and
            # gets the reason first-hand -- so the banner is right and the cost is one
            # extra round trip per prompt for as long as an allowance stays spent. Widening
            # this wire is the fix; `test_a_failed_query_is_not_an_empty_answer` asserts
            # the reply dict exactly, so it is a deliberate change and not a drive-by.
            return {"ok": False}
        header = request.get("header")
        if header:
            kwargs["header"] = str(header)
        if request.get("include_episodes"):
            kwargs["include_episodes"] = True
        types = request.get("memory_types")
        if isinstance(types, list) and types:
            kwargs["memory_types"] = [str(t) for t in types]
        try:
            # Serialised deliberately. The store is a read handle over SQLite and is not
            # documented as thread-safe; a per-prompt hook has no concurrency worth the
            # risk of finding out otherwise.
            with self._lock:
                # Both backends answer the same call. The local one is a `Memvara`; the
                # hosted one is a `HostedRecall` holding a kept-alive TLS connection,
                # which is the whole reason a hosted install wants a daemon: the same
                # request costs 609ms on a fresh connection and 177ms on a warm one.
                #
                # Both raise on failure and return text on success, which is what lets one
                # `except` cover both backends without knowing which one it holds.
                text = str(self.store.recall(query, **kwargs) or "")
                self.failures = 0
                return {"ok": True, "text": text}
        except Exception:
            # Still never a raised exception out of here -- but no longer an empty string
            # either, because the client cannot act on what it cannot see.
            with self._lock:
                self.failures += 1
            return {"ok": False}

    def _serve(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(5.0)
            try:
                chunks, size = [], 0
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_REQUEST_BYTES:
                        return
                    chunks.append(chunk)
                request = json.loads(b"".join(chunks).decode("utf-8", "replace"))
                if not isinstance(request, dict):
                    return
                conn.sendall(json.dumps(self._answer(request)).encode("utf-8"))
            except (OSError, ValueError, socket.timeout):
                # Client vanished mid-exchange, or sent nonsense. Neither is fatal.
                return

    def _sweep_stale(self) -> None:
        """Unlink sockets in this directory that nobody is listening on.

        The address digests the hook sources, which is what stops an edited hook being
        served by a daemon running the old code -- the daemon is stranded rather than
        reused, and it exits. What it does not do is remove the *file*, so a day of hook
        edits leaves a directory of dead addresses: five sockets, one live daemon, on the
        machine this was found on.

        Litter rather than a leak, and worth a dozen lines anyway, because `ls run/` is how
        somebody debugging recall asks whether a daemon is up, and it should not answer
        with four ghosts and one truth.

        The probe is a bare `connect`, and it has to be: asking for an *answer* would call
        a live daemon dead whenever it happened to be busy, and unlinking a live address
        leaves that daemon running and unreachable while the next client spawns a duplicate
        beside it. A listening socket accepts immediately; a stranded one refuses with
        ECONNREFUSED. That is the whole distinction, and it needs no reply.

        Our own address is skipped because we are listening on it in a moment, and every
        failure is ignored -- a socket that cannot be swept is exactly as harmless as it
        was before.
        """
        run_dir = os.path.dirname(self.path)
        try:
            names = os.listdir(run_dir)
        except OSError:
            return
        for name in names:
            path = os.path.join(run_dir, name)
            if path == self.path or not name.startswith("recall-"):
                continue
            if _listening(path):
                continue  # a live daemon for some other digest; leave it alone
            try:
                os.unlink(path)
            except OSError:
                continue

    def run(self) -> int:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(self.path)
        except OSError:
            # Either a live daemon owns this address, or a dead one left the file behind.
            # Telling those apart by connecting is the only reliable test: a stale socket
            # refuses, a live one accepts.
            from lib.ipc import send

            if send(self.path, {"q": ""}, timeout=1.0) is not None:
                return 0  # someone else is already serving this exact address
            try:
                os.unlink(self.path)
                server.bind(self.path)
            except OSError:
                return 0
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        self._sweep_stale()

        server.listen(16)
        server.settimeout(30.0)
        try:
            while True:
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    if time.monotonic() - self.last_seen > IDLE_TIMEOUT_SEC:
                        return 0
                    continue
                self.last_seen = time.monotonic()
                if self.failures >= MAX_CONSECUTIVE_FAILURES:
                    # The failures were counted by earlier clients; this one is dropped
                    # without a reply. That is deliberate and not a lost answer: a dropped
                    # connection is one of the cases `ipc.send` already collapses to None,
                    # so this client falls through to the in-process route and gets a real
                    # answer -- which is more than the reply we were about to send it.
                    return 0
                threading.Thread(target=self._serve, args=(conn,), daemon=True).start()
        finally:
            server.close()
            try:
                os.unlink(self.path)
            except OSError:
                pass


def main() -> int:
    store = open_store()
    if store is None:
        # No library, or no local store. On a paste-the-URL hosted install that is the
        # normal state, not a broken one, so fall through to the stdlib HTTP client
        # rather than exiting.
        from lib.hosted import open_hosted

        store = open_hosted()
    if store is None:
        # Nothing to serve at all. Exiting is correct: a daemon with no backend would
        # accept connections and answer every one with silence, which is indistinguishable
        # from a working daemon over a store that happens to be empty.
        return 0
    try:
        # Pay the first-query costs -- imports, page cache, TLS handshake -- before any
        # prompt is waiting on them. For hosted this is the handshake that turns a 609ms
        # first call into a 177ms one.
        store.recall("warm", k=1)
    except Exception:
        pass
    return Daemon(socket_path(store_key()), store).run()


if __name__ == "__main__":
    raise SystemExit(main())
