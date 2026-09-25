"""Two handles on one store file: what one commits, the other reads at once, through every
read path, the vector index included. A writer that holds the lock never blocks a reader.

A server and the hooks' daemon are two handles on one file in real use, so a write the
other handle cannot see is a memory the agent loses until something reopens the store.
"""

from __future__ import annotations

import pathlib
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from memvara import Memvara

from harness import stores
from harness.crash import Child

USER = "u1"


@pytest.fixture()
def handles(tmp_path: pathlib.Path) -> Iterator[tuple[Memvara, Memvara]]:
    path = tmp_path / "s.db"
    first = stores.file(path)
    second = stores.file(path)
    try:
        yield first, second
    finally:
        first.close()
        second.close()


def reads(mem: Memvara, text: str) -> dict[str, Any]:
    """What each read path returns for `text`, by claim id."""
    return {
        "get_all": sorted(c.id for c in mem.get_all(user=USER)),
        "search": sorted(r.claim.id for r in mem.search(text, k=5, user=USER)),
    }


def test_a_write_through_one_handle_is_read_at_once_through_the_other(
        handles: tuple[Memvara, Memvara]) -> None:
    first, second = handles
    assert reads(second, "Berlin") == {"get_all": [], "search": []}   # opened, and empty
    berlin = first.remember("user", "lives_in", "Berlin", user=USER).added[0].id
    assert reads(second, "Berlin") == {"get_all": [berlin], "search": [berlin]}
    assert second.get(berlin, user=USER) is not None


def test_the_vector_index_of_the_other_handle_sees_the_new_claim(
        handles: tuple[Memvara, Memvara]) -> None:
    """`search` joins the text and vector legs, so a hit through the text index alone
    would hide a stale vector index. `vector_search` asks the vector index only."""
    first, second = handles
    second.remember("user", "likes", "green tea", user=USER)     # maps the vector file
    claim = first.remember("user", "lives_in", "Berlin", user=USER).added[0]
    vector = second.embedder.encode([claim.text])[0]
    hits = [claim_id for claim_id, _ in second.store.vector_search(vector, [claim.scope], 5)]
    assert claim.id in hits


def test_a_reader_on_another_thread_sees_the_write_too(
        handles: tuple[Memvara, Memvara]) -> None:
    first, second = handles
    second.get_all(user=USER)
    berlin = first.remember("user", "lives_in", "Berlin", user=USER).added[0].id
    seen: list[dict[str, Any]] = []
    reader = threading.Thread(target=lambda: seen.append(reads(second, "Berlin")))
    reader.start()
    reader.join(timeout=10)
    assert seen == [{"get_all": [berlin], "search": [berlin]}]


def test_an_erase_through_one_handle_removes_the_claim_from_every_read_of_the_other(
        handles: tuple[Memvara, Memvara]) -> None:
    first, second = handles
    berlin = first.remember("user", "lives_in", "Berlin", user=USER).added[0].id
    assert reads(second, "Berlin")["get_all"] == [berlin]
    assert first.erase(berlin, user=USER)
    assert reads(second, "Berlin") == {"get_all": [], "search": []}
    assert second.get(berlin, user=USER) is None


def test_a_writer_holding_the_lock_blocks_no_reader(tmp_path: pathlib.Path) -> None:
    """A held child stops right after its claim is written and before the commit, so it
    holds the write lock. Reads through another handle return the last commit at once."""
    db = tmp_path / "s.db"
    home = tmp_path / "home"
    home.mkdir()
    with stores.file(db) as mem:
        tea = mem.remember("user", "likes", "green tea", user=USER).added[0].id
        spec = {"db": str(db), "user": USER, "setup": [], "point": "after-claim",
                "hold": True, "action": ["remember", {"predicate": "lives_in",
                                                      "object": "Berlin"}]}
        with Child(spec, home=home) as child:
            child.wait_for("POINT after-claim")
            started = time.monotonic()
            seen = reads(mem, "Berlin")
            took = time.monotonic() - started
            # Only the committed claim, whatever the query: `search` returns the nearest
            # claims it has, so the uncommitted Berlin must simply not be among them.
            assert seen["get_all"] == [tea] and set(seen["search"]) <= {tea}
            assert took < 1.0, f"reads waited {took:.1f} s for the writer"
            child.release()
            child.wait_for("DONE")
            assert child.finish() == 0
        assert sorted(c.object for c in mem.get_all(user=USER)) == ["Berlin", "green tea"]
