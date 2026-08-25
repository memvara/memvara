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
from datetime import timedelta

import pytest

from memvara import Memvara, HashingEmbedder, NullLLM, Scope, utcnow
from memvara.aio import (
    NOT_WRAPPED,
    AsyncMemvara,
    AsyncScopedMemvara,
    _public,
    _scoped_omissions,
    _unbound,
    _unwrapped,
)
from memvara.core import ScopedMemvara
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


def test_nothing_is_deliberately_omitted_any_more():
    """`scope` was the one entry, and it is implemented. The set stays because the check
    needs somewhere to record a deliberate omission; empty is the honest answer."""
    assert NOT_WRAPPED == frozenset()
    assert hasattr(AsyncMemvara, "scope")


def test_a_method_added_to_memvara_and_not_here_is_reported():
    """The check above is only worth anything if it can fail."""
    assert _unwrapped(type("Fake", (), {"vaporize": lambda self: None})) == {"vaporize"}


# --- the same surface, scoped -----------------------------------------------
#
# `AsyncScopedMemvara` sits at the corner of a square — sync/async on one axis, unscoped/
# scoped on the other — and can fall off either edge. A method added to `AsyncMemvara`
# and forgotten here is awaitable only unscoped; one added to `ScopedMemvara` and
# forgotten here is scopable only synchronously. Neither raises: the caller reaches for
# a method that is simply not there, at runtime, in whichever request first needs it.
#
# Both checks derive the surface with `dir()`. A list of method names written out by hand
# would have to be updated by the same person who just forgot to update the class, which
# makes it a copy of the mistake rather than a check on it.

def test_the_scoped_view_covers_the_async_facade():
    assert _unbound() == set()


def test_the_scoped_view_covers_the_synchronous_scoped_view():
    """The other edge. `ScopedMemvara` is the shape being mirrored, so anything it has
    and this does not is a method you can scope in sync code and not in async code."""
    assert _unbound(ScopedMemvara) == set()


def test_a_method_added_to_the_async_facade_and_not_bound_here_is_reported():
    """Both checks above are only worth anything if they can fail."""
    assert _unbound(type("Fake", (), {"vaporize": lambda self: None})) == {"vaporize"}


def test_what_a_scoped_view_legitimately_leaves_out_is_read_off_the_sync_pair():
    """The excused names are not a literal in this file or in `aio.py`.

    `close`, `reembed` and `scope` are absent from `ScopedMemvara` because they are not
    scoped operations, and `_scoped_omissions()` reads that off the sync pair rather than
    restating it — so the day `ScopedMemvara` grows one of them, the async view stops
    being excused for not having it, without anyone remembering to edit a list.
    """
    assert _scoped_omissions() == {"close", "reembed", "scope"}
    assert _scoped_omissions() <= _public(AsyncMemvara), \
        "excusing a name the async facade does not even have would hide a real gap"


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
        assert await amem.connectivity() == {"live_claims": 1, "joinable_claims": 0}
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


def test_the_delta_read_is_awaitable(amem):
    """`since()` is two scope-wide id scans over SQLite, which is exactly the shape this
    module exists to keep off the loop thread."""
    async def main():
        await amem.remember("user", "lives_in", "Berlin")
        berlin = (await amem.get_all())[0]
        away = utcnow()
        await amem.supersede(berlin.id, Claim(subject="user", predicate="lives_in",
                                              object="Lisbon",
                                              scope=amem.default_scope),
                             close="retired")
        delta = await amem.since(away)
        return [c.object for c in delta.added], [c.object for c in delta.gone]

    assert run(main()) == (["Lisbon"], ["Berlin"])


def test_the_scoped_view_carries_the_delta_read_too(amem):
    """The other edge of the square. A method that is awaitable only unscoped is missing
    from the object a server layer actually holds per request."""
    async def main():
        view = amem.scope(user="karl")
        await view.remember("user", "working_on", "auth refactor")
        return [c.object for c in (await view.since(utcnow() - timedelta(days=1))).added]

    assert run(main()) == ["auth refactor"]


def test_the_prompt_block_returns_its_ids_through_both_async_facades(amem):
    """`recall()` still returns `str` by default on all four surfaces, and `with_ids`
    has to reach the two here or the citation is available only in synchronous code."""
    async def main():
        await amem.remember("user", "lives_in", "Lisbon")
        view = amem.scope(user="alice")
        return (await amem.recall("where do they live"),
                await amem.recall("where do they live", with_ids=True),
                await view.recall("where do they live", with_ids=True))

    plain, block, scoped = run(main())
    assert isinstance(plain, str)
    assert block.text == plain == scoped.text
    assert block.claim_ids == scoped.claim_ids
    assert len(block.claim_ids) == 1


def test_the_context_budget_reaches_both_async_facades(amem):
    """A `**self._kw` splice that dropped `budget` would still typecheck and still return
    a block — one that quietly overruns the caller's context window."""
    async def main():
        for city in ("Lisbon", "Berlin", "Porto", "Madrid", "Vienna"):
            await amem.remember("user", f"lived_in_{city.lower()}", city)
        view = amem.scope(user="alice")
        return (await amem.recall("where has the user lived", budget=40),
                await view.recall("where has the user lived", budget=40, with_ids=True))

    unscoped, scoped = run(main())
    assert "did not fit" in unscoped
    assert unscoped == scoped.text and scoped.dropped > 0


def test_maintenance_is_awaitable(amem):
    async def main():
        await amem.add("I live in Berlin")
        assert await amem.reembed(HashingEmbedder(dim=96)) == 1
        assert (await amem.consolidate())["decayed"] >= 0

    run(main())


def test_proving_an_erasure_is_awaitable_on_both_views(amem):
    """It re-queries the disk, so it is exactly the kind of call that must not run on
    the loop thread — and it is on the scoped view too, taking no scope, because a
    scoped view has no narrower version of a row count."""
    async def main():
        receipt = await amem.remember("user", "lives_in", "Berlin")
        claim_id = receipt.added[0].id
        assert not (await amem.prove_erased(claim_id)).proven
        assert await amem.erase(claim_id)
        assert (await amem.prove_erased(claim_id)).proven
        assert (await amem.scope(user="alice").prove_erased(claim_id)).proven

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


# --- the scoped view, exercised ---------------------------------------------

def test_a_scoped_view_binds_the_four_keywords_once(amem):
    """`ScopedMemvara`'s own first test, awaited. Same object, same guarantee."""
    async def main():
        alice = amem.scope(user="alice", session="s1")
        await alice.remember("user", "lives_in", "Lisbon")
        assert [c.object for c in await alice.get_all()] == ["Lisbon"]
        assert await amem.get_all(user="alice", session="s1") == await alice.get_all()

    run(main())


def test_a_scoped_view_keeps_one_scope_out_of_another(amem):
    """The reason the binding exists at all: four keywords repeated per call are four
    chances to write one user's fact into another user's scope."""
    async def main():
        await amem.scope(user="bob").remember("user", "likes", "tea")
        assert [c.object for c in await amem.get_all(user="bob")] == ["tea"]
        assert await amem.get_all(user="carol") == []

    run(main())


def test_a_scoped_view_cannot_be_talked_out_of_its_scope(amem):
    async def main():
        with pytest.raises(TypeError):
            await amem.scope(user="frank").remember("user", "lives_in", "Lisbon",
                                                    user="mallory")

    run(main())


def test_a_scoped_view_narrows_but_never_widens(amem):
    async def main():
        session = amem.scope(user="erin").bind(session="s2")
        await session.remember("user", "working_on", "auth refactor")
        assert session.scope == Scope("default", "erin", None, "s2")
        assert await amem.get_all(user="erin") == [], "session scratch stayed put"
        assert len(await amem.get_all(user="erin", session="s2")) == 1

    run(main())


def test_a_scoped_view_says_what_it_is_bound_to(amem):
    assert repr(amem.scope(user="grace", session="s3")).startswith(
        "<AsyncScopedMemvara default/grace/*/s3 of <AsyncMemvara ")


def test_binding_a_scope_touches_no_store(amem, monkeypatch):
    """`scope()` is the one method here that is not a coroutine, and it has to stay that
    way honestly: it binds four strings. If it ever reaches the store, it is lying about
    what it costs on the loop thread."""
    monkeypatch.setattr(amem.memvara.store, "get_claim",
                        lambda *a, **kw: pytest.fail("scope() went to the store"))
    assert amem.scope(user="ivan").scope == Scope("default", "ivan")


def test_the_scoped_view_reaches_both_objects_underneath(amem, mem):
    """`memvara` is the synchronous instance on all three facades — a server layer wants
    the store off it. `unscoped` is the async one, and is where `close()` and `reembed()`
    live, because a scoped view has neither and reaching them synchronously would put an
    fsync back on the loop thread."""
    view = amem.scope(user="alice")

    assert view.memvara is mem
    assert view.unscoped is amem
    assert not hasattr(view, "close") and not hasattr(view, "reembed")


def test_the_scoped_view_is_not_mistaken_for_a_synchronous_one(amem):
    """`integrations._common.bind` recognises a `ScopedMemvara` by its `_mem` attribute.

    Naming this class's slot the same thing would hand a synchronous adapter an
    `AsyncMemvara` typed as an `Memvara`: every call a coroutine nobody awaits, no
    exception anywhere, and memory that silently never gets written.
    """
    from memvara.integrations._common import bind

    view = amem.scope(user="alice")
    assert getattr(view, "_mem", None) is None
    with pytest.raises(AttributeError):
        bind(view)


def test_the_scoped_write_and_read_surface(amem):
    """Every bound method, once, against a real store — the forwarding is the whole of
    what this class does, and a wrong `**self._kw` splice is invisible to a type checker.
    """
    async def main():
        view = amem.scope(user="dave")
        await view.add("I live in Berlin")
        claim = (await view.get_all())[0]

        assert (await view.get(claim.id)).object == "Berlin"
        assert (await view.why(claim.id)) is not None
        assert await view.count() == 1
        assert [c.object for c in await view.history("user", "lives_in")] == ["Berlin"]
        assert [c.id for c in await view.produced(claim.sources[0])] == [claim.id]
        assert [r.claim.object for r in await view.search("where do they live")] \
            == ["Berlin"]
        assert "Berlin" in await view.recall("where do they live")
        assert (await view.stats())["claims"] == 1

        await view.remember("Berlin", "in_country", "Germany")
        # The path below is exactly what a join rate above zero is a count of.
        assert await view.connectivity() == {"live_claims": 2, "joinable_claims": 1}
        assert [p.render() for p in await view.paths_between("user", "Germany")] \
            == ["user -lives_in-> Berlin -in_country-> Germany"]
        assert await view.neighborhood("user") != []

        await view.supersede(claim.id, Claim(subject="user", predicate="lives_in",
                                             object="Lisbon", scope=view.scope))
        assert (await view.get(claim.id)).state == "ended"

        assert isinstance(await view.consolidate(), dict)
        assert await view.delete((await view.get_all())[0].id) is True
        await view.remember("user", "likes", "coffee")
        assert len(await view.forget("user", "likes")) == 1
        await view.remember("user", "speaks", "portuguese")
        assert await view.erase((await view.get_all())[0].id) is True
        await view.remember("user", "speaks", "portuguese")
        assert (await view.purge())["claims"] >= 1
        await view.remember("user", "speaks", "portuguese")
        assert (await view.reset())["claims"] >= 1
        assert await view.get_all() == []

    run(main())


def test_the_scoped_view_reads_the_bitemporal_keywords(amem):
    """The time keywords and the claim-state filter have to survive the splice too: they
    are the arguments a scoped server handle actually varies per request, and a `**_kw`
    splice that dropped one would still typecheck and still return claims."""
    async def main():
        now = utcnow()
        then, moved = now - timedelta(days=800), now - timedelta(days=30)
        view = amem.scope(user="judy")
        await view.remember("user", "lives_in", "Berlin",
                            valid_from=then, recorded_at=then)
        await view.remember("user", "lives_in", "Lisbon",
                            valid_from=moved, recorded_at=moved)

        back = now - timedelta(days=100)
        assert [c.object for c in await view.get_all()] == ["Lisbon"]
        assert [c.object for c in await view.get_all(valid_at=back)] == ["Berlin"]
        # `known_at` alone is the *other* question — what we believed 100 days ago about
        # the world as it is now — and Berlin has since ended in world time, so it is
        # empty. Asserted because a splice that forwarded one axis under the other's name
        # would pass every check above.
        assert await view.get_all(known_at=back) == []
        assert [c.object for c in await view.get_all(as_of=back)] == ["Berlin"]
        assert len(await view.get_all(include_invalidated=True)) == 2
        assert await view.count() == 1
        assert await view.count(include_invalidated=True) == 2
        assert [r.claim.object for r in
                await view.search("lives", as_of=now - timedelta(days=100))] == ["Berlin"]

    run(main())


def test_a_slow_scoped_call_also_leaves_the_loop_free(amem, monkeypatch):
    """The view forwards to the facade rather than re-implementing the threading, so
    this is the property it inherits — and the assertion that says it really forwards."""
    monkeypatch.setattr(amem.memvara, "consolidate",
                        lambda **kw: time.sleep(0.2) or {"decayed": 0})

    async def main():
        ticks = 0
        task = asyncio.ensure_future(amem.scope(user="alice").consolidate())
        while not task.done():
            await asyncio.sleep(0.001)
            ticks += 1
        await task
        return ticks

    assert run(main()) > 10, "the loop was pinned for the duration of the call"


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
