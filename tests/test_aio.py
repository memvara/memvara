"""The async facade.

Two things are under test, and only one of them is "does it work".

The first is coverage of the surface: an adapter that declares `aadd`/`asearch` and
finds nothing to call falls back to the synchronous method *on the loop thread*, which
is the failure this module exists to remove — so a public `Memvara` method with no
counterpart here is a regression, and `_unwrapped()` is what says so.

The second is that it genuinely leaves the loop free. A wrapper that awaited nothing
would pass every functional assertion below and fix nothing at all, so
`test_a_slow_call_does_not_block_the_event_loop` is the one that matters.

Written with `asyncio.run` rather than `pytest-asyncio` on purpose: the thing under test
is one `asyncio.to_thread` per method, and a plugin with its own event-loop policy,
fixture scoping and strict/auto modes is a larger dependency than the module it would be
testing.
"""

import asyncio
import sqlite3
import time

import pytest

from memvara import Memvara, HashingEmbedder, NullLLM, Scope
from memvara.aio import NOT_WRAPPED, AsyncMemvara, _unwrapped
from memvara.types import Claim


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def mem():
    m = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice")
    yield m
    m.close()


@pytest.fixture()
def amem(mem):
    return AsyncMemvara(mem)


# --- surface ----------------------------------------------------------------

def test_every_public_memvara_method_has_an_awaitable_counterpart():
    """The reported gap was literal: `dir(Memvara)` had exactly one method starting with
    `a`, and it was `add`. LangChain, LlamaIndex and CrewAI all declare async methods
    whose default implementation runs the sync one on the loop thread."""
    assert _unwrapped() == set()


def test_scope_is_the_one_deliberate_omission():
    assert NOT_WRAPPED == {"scope"}
    assert not hasattr(AsyncMemvara, "scope")


def test_a_method_added_to_memvara_and_not_here_is_reported():
    """The check above is only worth anything if it can fail."""
    assert _unwrapped(type("Fake", (), {"vaporize": lambda self: None})) == {"vaporize"}


def test_the_wrapper_says_what_it_wraps(amem):
    assert repr(amem).startswith("<AsyncMemvara of <Memvara default/alice/")


def test_state_is_read_without_awaiting(amem, mem):
    """Reading these touches no I/O, and making them awaitable would imply otherwise."""
    assert amem.store is mem.store
    assert amem.embedder is mem.embedder
    assert amem.default_scope == Scope("default", "alice")
    assert amem.extractor == "fast-path-only"


# --- it actually yields -----------------------------------------------------

def test_a_slow_call_does_not_block_the_event_loop(amem, monkeypatch):
    """The assertion the whole module rests on. `encode()` on a real embedding model and
    a SQLite write lock are both hundreds of milliseconds; run on the loop thread they
    stall every other task in the process."""
    monkeypatch.setattr(amem.memvara, "consolidate",
                        lambda **kw: time.sleep(0.2) or {"decayed": 0})

    async def main():
        ticks = 0
        task = asyncio.ensure_future(amem.consolidate())
        while not task.done():
            await asyncio.sleep(0.001)
            ticks += 1
        await task
        return ticks

    assert run(main()) > 10, "the loop was pinned for the duration of the call"


def test_concurrent_calls_serialize_inside_the_store_rather_than_corrupting_it(mem):
    """`SQLiteStore` guards its connection with an `RLock`, which is what makes the
    threadpool safe to point at it. Concurrency here buys overlap, not parallel writes —
    SQLite has one writer."""
    amem = AsyncMemvara(mem)

    async def main():
        await asyncio.gather(*(amem.remember("user", f"p{i}", f"v{i}")
                               for i in range(24)))
        return await amem.count()

    assert run(main()) == 24


# --- the surface, exercised -------------------------------------------------

def test_the_write_surface(amem):
    async def main():
        receipt = await amem.add("I live in Berlin")
        assert receipt.episode_ids
        claim = (await amem.get_all())[0]

        assert (await amem.get(claim.id)).object == "Berlin"
        assert (await amem.why(claim.id)) is not None
        assert (await amem.count()) == 1
        assert (await amem.stats())["claims"] == 1
        assert [c.object for c in await amem.history("user", "lives_in")] == ["Berlin"]

        await amem.supersede(claim.id, Claim(subject="user", predicate="lives_in",
                                             object="Lisbon",
                                             scope=amem.default_scope))
        assert (await amem.get(claim.id)).invalidated_by is not None

        current = (await amem.get_all())[0]
        assert await amem.delete(current.id) is True
        await amem.remember("user", "likes", "coffee")
        assert len(await amem.forget("user", "likes")) == 1

        await amem.remember("user", "speaks", "portuguese")
        assert (await amem.erase((await amem.get_all())[0].id)) is True

        await amem.remember("user", "speaks", "portuguese")
        assert (await amem.purge())["claims"] >= 1
        await amem.remember("user", "speaks", "portuguese")
        assert (await amem.reset())["claims"] >= 1
        assert (await amem.get_all()) == []

    run(main())


def test_the_read_surface(amem):
    async def main():
        await amem.add("We decided to sunset the kafka pipeline at the offsite")
        await amem.remember("user", "lives_in", "Berlin")

        assert [r.claim.object for r in await amem.search("where do they live")] \
            == ["Berlin"]
        assert (await amem.search("berlin", min_score=1.1)) == []
        assert "Berlin" in await amem.recall("where do they live")
        turns = await amem.search("kafka", include_episodes=True)
        assert any(getattr(r, "episode", None) is not None for r in turns)

    run(main())


def test_maintenance_is_awaitable(amem):
    async def main():
        await amem.add("I live in Berlin")
        assert await amem.reembed(HashingEmbedder(dim=96)) == 1
        assert (await amem.consolidate())["decayed"] >= 0

    run(main())


def test_close_is_awaitable_because_it_commits():
    memvara = Memvara(embedder=HashingEmbedder(dim=32), llm=NullLLM(), user="alice")

    async def main():
        amem = AsyncMemvara(memvara)
        await amem.remember("user", "lives_in", "Berlin")
        await amem.close()

    run(main())
    with pytest.raises(sqlite3.ProgrammingError):
        memvara.count()


def test_the_async_context_manager_closes_the_store():
    async def main():
        memvara = Memvara(embedder=HashingEmbedder(dim=32), llm=NullLLM(), user="alice")
        async with AsyncMemvara(memvara) as amem:
            await amem.remember("user", "lives_in", "Berlin")
            assert await amem.count() == 1
        return memvara

    memvara = run(main())
    with pytest.raises(sqlite3.ProgrammingError):
        memvara.count()
