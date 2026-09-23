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
