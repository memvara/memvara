"""Real servers under the pressure a long agent session puts on them: killed at random
moments, run two at once, and made to wait on a lock another writer holds."""

from __future__ import annotations

import json
import pathlib
import random
import threading
import time
from typing import Any

import pytest

from harness import stores
from harness.crash import Child
from harness.invariants import check_store_integrity
from harness.stdio import McpProcess

USER = "tester"     # the user `McpProcess` binds by default
ROUNDS = 200


def remember_line(request_id: int, obj: str) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
                       "params": {"name": "memory_remember", "arguments": {
                           "subject": "user", "predicate": "visited", "object": obj}}})


def test_random_kills_of_a_real_server_lose_no_acknowledged_write(
        tmp_path: pathlib.Path) -> None:
    """Each round starts a server on the same store, makes a few writes it waits for,
    sends one more it does not wait for, and kills the server a random moment later. A
    write counts as acknowledged only when its reply arrived."""
    rng = random.Random(20260925)
    db = tmp_path / "s.db"
    home = tmp_path / "home"
    home.mkdir()
    acknowledged: list[str] = []
    for round_ in range(ROUNDS):
        server = McpProcess(db, home=home)
        try:
            server.initialize()
            for call in range(rng.randint(0, 4)):
                obj = f"r{round_:03d}c{call}"
                reply = server.call("memory_remember", subject="user", predicate="visited",
                                    object=obj)
                assert not reply.is_error, reply.text
                acknowledged.append(obj)
            server.send_raw(remember_line(10_000 + round_, f"r{round_:03d}unacknowledged"))
            time.sleep(rng.uniform(0, 0.02))
        finally:
            server.kill()
        problems = check_store_integrity(db)
        assert not problems, f"after round {round_}: {problems}"
    with stores.file(db) as mem:
        stored = {c.object for c in mem.get_all(user=USER)}
    missing = sorted(set(acknowledged) - stored)
    assert not missing, f"acknowledged writes are gone: {missing}"


def test_two_servers_writing_at_once_keep_every_acknowledged_write(
        tmp_path: pathlib.Path) -> None:
    """Two servers write and read at the same time from two threads. Only the first
    writes `lives_in`, so the final state of that slot is known."""
    db = tmp_path / "s.db"
    stores.file(db).close()     # created first: two servers creating it at once meet #281
    acknowledged: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def session(name: str, home: pathlib.Path, seed: int) -> None:
        rng = random.Random(seed)
        try:
            with McpProcess(db, home=home) as server:
                server.initialize()
                for i in range(200):
                    obj = f"{name}{i:03d}"
                    kind = rng.choice(["remember", "remember", "search"])
                    if kind == "search":
                        reply = server.call("memory_search", query=obj, k=3)
                    elif name == "a" and rng.random() < 0.2:
                        reply = server.call("memory_remember", subject="user",
                                            predicate="lives_in", object=f"City{obj}")
                    else:
                        reply = server.call("memory_remember", subject="user",
                                            predicate="likes", object=obj)
                        if not reply.is_error:
                            with lock:
                                acknowledged.append(obj)
                    assert not reply.is_error, reply.text
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            errors.append(exc)

    threads = []
    for n, name in enumerate("ab"):
        home = tmp_path / name
        home.mkdir()
        threads.append(threading.Thread(target=session, args=(name, home, n)))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert not errors, errors
    with stores.file(db) as mem:
        likes = [c.object for c in mem.get_all(user=USER) if c.predicate == "likes"]
        homes = [c for c in mem.get_all(user=USER) if c.predicate == "lives_in"]
    assert sorted(likes) == sorted(acknowledged), "a write was lost or stored twice"
    assert len(homes) <= 1
    assert check_store_integrity(db) == []


def test_a_writer_that_holds_the_lock_makes_others_fail_after_the_busy_timeout(
        tmp_path: pathlib.Path) -> None:
    """One writer at a time is documented behaviour. The held child keeps the write lock:
    it stops inside `remember()` for a name it has not seen, after the entity write. A
    write through another handle raises after SQLite's five-second busy timeout, and a
    server's write returns an error result in about the same time and stays up."""
    db = tmp_path / "s.db"
    home = tmp_path / "home"
    home.mkdir()
    with stores.file(db) as mem:
        mem.remember("user", "likes", "green tea", user=USER)
    spec: dict[str, Any] = {"db": str(db), "user": USER, "setup": [], "hold": True,
                            "point": "before-claim",
                            "action": ["remember", {"predicate": "lives_in",
                                                    "object": "Reykjavik"}]}
    with Child(spec, home=home, timeout=60) as child:
        child.wait_for("POINT before-claim")
        with stores.file(db) as mem:
            started = time.monotonic()
            with pytest.raises(Exception, match="database is locked"):
                mem.remember("user", "lives_in", "Nairobi", user=USER)
            took = time.monotonic() - started
        assert 4.5 <= took <= 7.0, f"the write waited {took:.1f} s"
        with McpProcess(db, home=home) as server:
            server.initialize()
            started = time.monotonic()
            reply = server.call("memory_remember", subject="user", predicate="lives_in",
                                object="Lima")
            took = time.monotonic() - started
            assert reply.is_error, reply.text
            assert took <= 7.0, f"the server's write waited {took:.1f} s"
            assert server.request("ping") is not None
        child.release()
        child.wait_for("DONE")
        assert child.finish() == 0
    assert check_store_integrity(db) == []
