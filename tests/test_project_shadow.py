"""A repository's own value hides the user-wide one, for single-valued facts only.

Before project scope, a local server wrote every fact without a project. Now a new value
written inside a repository lands in that repository's slot, so it cannot end the older
value that was stored without one — and ending it would be wrong anyway, because that
older value is still the answer in every other repository. So both stay stored, and a
read bound to the repository returns its own value instead of both. A read anywhere else
is unchanged. Only single-valued predicates shadow: for a many-valued one both values are
true at once, and hiding one would lose a fact.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from memvara import AsyncMemvara, HashingEmbedder, MemoryType, Memvara, NullLLM
from memvara.schema import (BUILTIN_PREDICATES, Cardinality, PredicateRegistry,
                            PredicateSpec, Volatility)

APP = "github.com/acme/app"
WEB = "github.com/acme/web"
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2026, 2, 1, tzinfo=timezone.utc)


def make():
    registry = PredicateRegistry(specs=BUILTIN_PREDICATES + (
        PredicateSpec(name="editor", cardinality=Cardinality.ONE,
                      volatility=Volatility.SLOW, memory_type=MemoryType.PROCEDURAL),
        PredicateSpec(name="uses_tool", cardinality=Cardinality.MANY,
                      volatility=Volatility.SLOW),
    ))
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice",
                  registry=registry)
    mem.remember("user", "editor", "vim", memory_type=MemoryType.PROCEDURAL,
                 valid_from=T0, recorded_at=T0)
    mem.scope(project=APP).remember("user", "editor", "emacs",
                                    memory_type=MemoryType.PROCEDURAL,
                                    valid_from=T1, recorded_at=T1)
    return mem


def objects(claims):
    return sorted(c.object for c in claims)


def test_both_values_stay_stored_and_the_old_one_is_still_live():
    mem = make()
    assert objects(mem.get_all()) == ["vim"]
    assert objects(mem.scope(project=APP).get_all(states=["live", "ended", "retired"])) \
        == ["emacs"]


def test_every_read_bound_to_the_repository_returns_only_its_own_value():
    app = make().scope(project=APP)
    assert objects(app.get_all()) == ["emacs"]
    assert [r.claim.object for r in app.search("editor")] == ["emacs"]
    block = app.recall("which editor")
    assert "emacs" in block and "vim" not in block
    assert objects(app.standing()) == ["emacs"]
    profile = app.profile("editor", since=T0 - timedelta(days=1),
                          buckets={"tools": ["editor"]})
    assert [r.text for r in profile.standing] == ["user editor emacs"]
    assert [r.text for r in profile.recent] == ["user editor emacs"]
    assert [r.text for r in profile.relevant] == ["user editor emacs"]
    assert [r.text for r in profile.buckets["tools"]] == ["user editor emacs"]
    assert objects(app.since(T0 - timedelta(days=1)).added) == ["emacs"]


def test_a_read_outside_the_repository_is_unchanged():
    mem = make()
    assert objects(mem.get_all()) == ["vim"]
    assert objects(mem.scope(project=WEB).get_all()) == ["vim"]
    assert [r.claim.object for r in mem.scope(project=WEB).search("editor")] == ["vim"]


def test_a_many_valued_predicate_shows_both_values():
    mem = make()
    mem.remember("user", "uses_tool", "make")
    mem.scope(project=APP).remember("user", "uses_tool", "bazel")
    assert objects(c for c in mem.scope(project=APP).get_all()
                   if c.predicate == "uses_tool") == ["bazel", "make"]


def test_a_read_at_an_earlier_instant_is_not_shadowed():
    """At T0 the repository had no value of its own, so the user-wide one answered."""
    app = make().scope(project=APP)
    between = T0 + timedelta(days=1)
    assert objects(app.get_all(valid_at=between)) == ["vim"]
    assert [r.claim.object for r in app.search("editor", valid_at=between)] == ["vim"]


def test_when_the_repositorys_value_ends_the_user_wide_one_answers_again():
    mem = make()
    app = mem.scope(project=APP)
    app.forget("user", "editor", close="ended")
    assert objects(app.get_all()) == ["vim"]


def test_the_async_facade_shadows_the_same_way():
    mem = make()

    async def run():
        return await AsyncMemvara(mem).scope(project=APP).get_all()

    assert objects(asyncio.run(run())) == ["emacs"]



# -- erasure stays inside the project ---------------------------------------------------

def test_a_purge_bound_to_a_project_erases_only_that_project():
    """A purge from inside one repository must not take every repository's memory with
    it. What was written without a project is user-wide, not the repository's, so it
    stays too."""
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice")
    app, web = mem.scope(project=APP), mem.scope(project=WEB)
    app.remember("api", "depends_on", "postgres")
    app.add("the api now talks to postgres through pgbouncer")
    web.remember("site", "depends_on", "nginx")
    web.add("the site is served by nginx")
    mem.remember("user", "prefers", "pytest", memory_type=MemoryType.PROCEDURAL)

    counts = app.purge()

    assert counts["claims"] >= 1 and counts["episodes"] == 1
    assert [c.object for c in app.get_all() if c.predicate == "depends_on"] == []
    assert [c.object for c in web.get_all() if c.predicate == "depends_on"] == ["nginx"]
    assert [c.object for c in mem.standing()] == ["pytest"]
    assert mem.store.stats()["episodes"] == 1, "the other repository's turn survives"


def test_reset_is_bound_the_same_way():
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice")
    mem.scope(project=APP).remember("api", "depends_on", "postgres")
    mem.scope(project=WEB).remember("site", "depends_on", "nginx")
    mem.scope(project=APP).reset()
    assert objects(mem.scope(project=WEB).get_all()) == ["nginx"]


def test_a_purge_with_no_project_still_takes_every_project():
    """An unset project is a wildcard, as every unset scope field is: purging a user
    erases that user's memory in every repository."""
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice")
    mem.scope(project=APP).remember("api", "depends_on", "postgres")
    mem.scope(project=WEB).remember("site", "depends_on", "nginx")
    mem.purge()
    assert mem.store.stats()["claims"] == 0



# -- what shadowing costs, and stores that cannot answer it ------------------------------

class Counting:
    """A store that counts the slot lookups shadowing makes, and can hide either one."""

    def __init__(self, inner, *, hide=()):
        self._inner, self._hide = inner, set(hide)
        self.batched, self.single = 0, 0

    def __getattr__(self, name):
        if name in self._hide:
            raise AttributeError(name)
        value = getattr(self._inner, name)
        if name == "occupied_slots":
            def batched(*a, **k):
                self.batched += 1
                return value(*a, **k)
            return batched
        if name == "count_competing":
            def single(*a, **k):
                self.single += 1
                return value(*a, **k)
            return single
        return value


def wrapped(**kw):
    """`make()`'s store, re-opened behind a `Counting` wrapper."""
    base = make()
    store = Counting(base.store, **kw)
    mem = Memvara(store=store, embedder=base.embedder, llm=NullLLM(), user="alice",
                  registry=base.registry)
    return mem, store


def test_the_slot_lookups_of_one_read_are_one_store_query():
    mem, store = wrapped()
    app = mem.scope(project=APP)
    store.batched = store.single = 0
    assert [r.claim.object for r in app.search("editor")] == ["emacs"]
    assert (store.batched, store.single) == (1, 0)
    store.batched = 0
    assert objects(app.get_all()) == ["emacs"]
    assert (store.batched, store.single) == (1, 0)


def test_a_store_without_the_batched_lookup_falls_back_to_one_per_slot():
    mem, store = wrapped(hide={"occupied_slots"})
    assert objects(mem.scope(project=APP).get_all()) == ["emacs"]
    assert store.single >= 1


def test_a_store_with_neither_lookup_reads_unshadowed_rather_than_raising():
    """Both lookups are optional on the store protocol. Without either, a read cannot
    tell whether the project has its own value, so it returns what is stored."""
    mem, _ = wrapped(hide={"occupied_slots", "count_competing"})
    app = mem.scope(project=APP)
    assert objects(app.get_all()) == ["emacs", "vim"]
    assert sorted(r.claim.object for r in app.search("editor")) == ["emacs", "vim"]


def test_a_store_that_declares_the_lookups_but_cannot_answer_reads_unshadowed(monkeypatch):
    """`RemoteStore` has both methods and raises `NotImplementedError` from each, because
    the hosted API has no slot lookup."""
    mem = make()

    def cannot(*a, **k):
        raise NotImplementedError("no endpoint")
    monkeypatch.setattr(mem.store, "occupied_slots", cannot)
    monkeypatch.setattr(mem.store, "count_competing", cannot)
    assert objects(mem.scope(project=APP).get_all()) == ["emacs", "vim"]


def test_search_asks_about_shadowing_only_for_results_it_would_return(monkeypatch):
    """The cheap filters run first, so a candidate that a memory-type filter or the score
    floor drops never costs a slot lookup."""
    import memvara.retrieve.hybrid as hybrid
    seen = []
    real = hybrid.shadowed

    def recording(store, registry, claims, scope):
        claims = list(claims)
        seen.append([c.object for c in claims])
        return real(store, registry, claims, scope)

    monkeypatch.setattr(hybrid, "shadowed", recording)
    app = make().scope(project=APP)
    assert app.search("editor", min_score=10.0) == []
    assert app.search("editor", memory_types=[MemoryType.SEMANTIC]) == []
    assert seen == [], "nothing survived the cheap filters, so nothing was looked up"
    assert [r.claim.object for r in app.search("editor")] == ["emacs"]
    assert [sorted(s) for s in seen] == [["emacs", "vim"]]
