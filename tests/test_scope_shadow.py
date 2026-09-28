"""A session's or an agent's own value shadows the user-wide one, and a session inside a
repository reads the global facts it writes.

#266. A value written in a session, or by an agent, is a local value. It answers inside
that session or agent, and it does not end the user-wide value, which is still the answer
for every other reader. A user-wide write does not end a session's or an agent's own
value either. A present-tense read takes a single-valued fact from the narrowest level of
the reader's chain (`Scope.ancestors`) that holds one, as a repository's own value already
did (`tests/test_project_shadow.py`). Reads at another instant, and `count()`, show every
value. A repeat of a value the writer can see still reinforces that value.

#273. A predicate declared global is written at the writer's scope with only the project
cleared, so a session inside a repository files it at that session with no project. The
session's chain now includes that level, so the session reads the fact back.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memvara import HashingEmbedder, MemoryType, Memvara, NullLLM
from memvara.schema import (BUILTIN_PREDICATES, Cardinality, PredicateRegistry,
                            PredicateSpec, Volatility)
from memvara.store.sqlite import SQLiteStore
from memvara.types import Claim, Scope

REPO = "github.com/acme/app"
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2026, 2, 1, tzinfo=timezone.utc)
#: The levels a handle can be bound to below the user, each with a sibling of its own.
LEVELS = [({"session": "s1"}, {"session": "s2"}),
          ({"agent": "a1"}, {"agent": "a2"}),
          ({"agent": "a1", "session": "s1"}, {"agent": "a1", "session": "s2"})]
LEVEL_IDS = ["session", "agent", "agent-and-session"]


def make(**options) -> Memvara:
    return Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice", **options)


def with_editor() -> Memvara:
    """A store whose vocabulary has `editor`: single-valued, and project-relative, which
    no built-in predicate is."""
    registry = PredicateRegistry(specs=BUILTIN_PREDICATES + (
        PredicateSpec(name="editor", cardinality=Cardinality.ONE,
                      volatility=Volatility.SLOW, memory_type=MemoryType.PROCEDURAL),))
    return make(registry=registry)


def objects(claims) -> list[str]:
    return sorted(c.object for c in claims)


def homes(mem) -> list[str]:
    return objects(c for c in mem.get_all() if c.predicate == "lives_in")


# -- separate values for supersession ----------------------------------------------------

@pytest.mark.parametrize("level,sibling", LEVELS, ids=LEVEL_IDS)
def test_a_bound_write_ends_nothing_at_the_user_level(level, sibling):
    mem = make()
    [berlin] = mem.remember("user", "lives_in", "Berlin").added
    receipt = mem.scope(**level).remember("user", "lives_in", "Paris")
    assert receipt.closed == []
    assert mem.get(berlin.id).state == "live"
    assert homes(mem) == ["Berlin"]
    assert homes(mem.scope(**sibling)) == ["Berlin"]
    assert homes(mem.scope(**level)) == ["Paris"]


@pytest.mark.parametrize("level,sibling", LEVELS, ids=LEVEL_IDS)
def test_a_user_wide_write_ends_nothing_a_session_or_agent_holds(level, sibling):
    mem = make()
    bound = mem.scope(**level)
    [paris] = bound.remember("user", "lives_in", "Paris").added
    receipt = mem.remember("user", "lives_in", "Berlin")
    assert receipt.closed == []
    assert bound.get(paris.id).state == "live"
    assert homes(bound) == ["Paris"]
    assert homes(mem) == ["Berlin"]
    assert homes(mem.scope(**sibling)) == ["Berlin"]


def test_a_session_and_its_agent_hold_separate_values_too():
    mem = make()
    agent, session = mem.scope(agent="a1"), mem.scope(agent="a1", session="s1")
    [berlin] = agent.remember("user", "lives_in", "Berlin").added
    assert session.remember("user", "lives_in", "Paris").closed == []
    [rome] = agent.remember("user", "lives_in", "Rome").added
    assert mem.get(berlin.id, agent="a1").state == "ended"
    assert homes(session) == ["Paris"]
    assert homes(agent) == ["Rome"]
    assert homes(mem.scope(agent="a1", session="s2")) == ["Rome"]
    assert rome.scope == Scope("default", "alice", "a1")


def test_a_new_value_still_ends_the_value_at_its_own_scope():
    mem = make()
    mem.remember("user", "lives_in", "Berlin")
    session = mem.scope(session="s1")
    [paris] = session.remember("user", "lives_in", "Paris").added
    receipt = session.remember("user", "lives_in", "Rome")
    assert [c.id for c in receipt.closed] == [paris.id]
    assert homes(session) == ["Rome"]
    assert homes(mem) == ["Berlin"]


def test_a_session_repeating_the_user_wide_value_reinforces_it():
    """A repeat reinforces a claim the writer can see, as it did before (#266 (c))."""
    mem = make()
    [berlin] = mem.remember("user", "lives_in", "Berlin").added
    receipt = mem.scope(session="s1").remember("user", "lives_in", "Berlin")
    assert [c.id for c in receipt.reinforced] == [berlin.id]
    assert receipt.added == []


@pytest.mark.parametrize("level,sibling", LEVELS, ids=LEVEL_IDS)
def test_a_bound_retraction_ends_the_user_wide_value_it_reads(level, sibling):
    """A retraction closes the claims its writer can see. A session or an agent reads the
    user-wide value, so its retraction still ends it."""
    mem = make()
    [berlin] = mem.remember("user", "lives_in", "Berlin").added
    receipt = mem.scope(**level).remember("user", "lives_in", "Berlin", polarity=-1)
    assert [c.id for c in receipt.closed] == [berlin.id]
    assert homes(mem) == []


@pytest.mark.parametrize("level,sibling", LEVELS, ids=LEVEL_IDS)
def test_a_user_level_retraction_leaves_a_bound_value_live(level, sibling):
    """A user-level reader cannot see a session's or an agent's value, so a user-level
    retraction that names the same value does not end it. `forget()` is the call that
    reaches down into every scope."""
    mem = make()
    bound = mem.scope(**level)
    # The session's own Berlin first: written after the user-wide one, it would reinforce
    # that claim instead of storing its own.
    [own] = bound.remember("user", "lives_in", "Berlin").added
    [berlin] = mem.remember("user", "lives_in", "Berlin").added
    receipt = mem.remember("user", "lives_in", "Berlin", polarity=-1)
    assert [c.id for c in receipt.closed] == [berlin.id]
    assert bound.get(own.id).state == "live"
    assert homes(bound) == ["Berlin"]
    assert homes(mem) == []


@pytest.mark.parametrize("level,sibling", LEVELS, ids=LEVEL_IDS)
def test_a_user_level_retraction_of_the_whole_slot_leaves_a_bound_value_live(
        level, sibling):
    """A retraction that names no value closes every value in the slot the writer can
    see, and no other."""
    mem = make()
    bound = mem.scope(**level)
    [berlin] = mem.remember("user", "lives_in", "Berlin").added
    [paris] = bound.remember("user", "lives_in", "Paris").added
    receipt = mem.remember("user", "lives_in", "", polarity=-1)
    assert [c.id for c in receipt.closed] == [berlin.id]
    assert bound.get(paris.id).state == "live"
    assert homes(bound) == ["Paris"]
    assert homes(mem) == []


def test_forget_at_the_user_level_still_reaches_a_sessions_value():
    mem = make()
    mem.remember("user", "lives_in", "Berlin")
    session = mem.scope(session="s1")
    session.remember("user", "lives_in", "Paris")
    assert sorted(c.object for c in mem.forget("user", "lives_in")) == ["Berlin", "Paris"]
    assert homes(session) == []


# -- the present-tense read takes the narrowest value --------------------------------------

@pytest.mark.parametrize("level,sibling", LEVELS, ids=LEVEL_IDS)
def test_every_present_tense_read_bound_to_the_level_returns_its_own_value(level, sibling):
    mem = make()
    mem.remember("user", "lives_in", "Berlin", valid_from=T0, recorded_at=T0)
    bound = mem.scope(**level)
    bound.remember("user", "lives_in", "Paris", valid_from=T1, recorded_at=T1)
    assert homes(bound) == ["Paris"]
    assert [r.claim.object for r in bound.search("where does the user live")] == ["Paris"]
    block = bound.recall("where does the user live")
    assert "Paris" in block and "Berlin" not in block
    assert objects(bound.since(T0 - timedelta(days=1)).added) == ["Paris"]
    assert [r.claim.object for r in mem.scope(**sibling).search("lives in")] == ["Berlin"]


def test_the_narrowest_level_of_the_chain_answers():
    mem = make()
    mem.remember("user", "lives_in", "Berlin")
    mem.scope(agent="a1").remember("user", "lives_in", "Paris")
    mem.scope(agent="a1", session="s1").remember("user", "lives_in", "Rome")
    assert homes(mem.scope(agent="a1", session="s1")) == ["Rome"]
    assert homes(mem.scope(agent="a1", session="s2")) == ["Paris"]
    assert homes(mem.scope(agent="a2")) == ["Berlin"]
    assert homes(mem.scope(session="s1")) == ["Berlin"]


def test_when_the_sessions_value_ends_the_user_wide_one_answers_again():
    mem = make()
    mem.remember("user", "lives_in", "Berlin")
    session = mem.scope(session="s1")
    session.remember("user", "lives_in", "Paris")
    session.forget("user", "lives_in", close="ended")
    assert homes(session) == ["Berlin"]
    assert homes(mem) == ["Berlin"]


def test_a_many_valued_predicate_shows_both_values():
    mem = make()
    mem.remember("user", "likes", "tea")
    session = mem.scope(session="s1")
    session.remember("user", "likes", "coffee")
    assert objects(c for c in session.get_all() if c.predicate == "likes") == ["coffee", "tea"]


def test_a_siblings_value_hides_nothing_from_a_reader_that_cannot_see_it():
    """A slot's key leaves out the session, so a sibling's value is in the same slot as
    the user-wide one. It is not in the reader's chain, so it must not hide the value the
    reader can see: before the lookup was asked about the reader's own levels only, a
    repository's reader lost the user-wide value to a session inside that repository."""
    mem = with_editor()
    mem.remember("user", "editor", "vim", memory_type=MemoryType.PROCEDURAL)
    mem.scope(project=REPO, session="s2").remember("user", "editor", "emacs",
                                                   memory_type=MemoryType.PROCEDURAL)
    editors = lambda m: objects(c for c in m.get_all() if c.predicate == "editor")  # noqa: E731
    assert editors(mem.scope(project=REPO)) == ["vim"]
    assert editors(mem.scope(project=REPO, session="s1")) == ["vim"]
    assert editors(mem.scope(project=REPO, session="s2")) == ["emacs"]


def test_the_tenants_own_value_is_not_hidden_by_a_users_value():
    """The shadow compares values of one owner, tenant plus user, as the slots do. A value
    written for the whole tenant belongs to no user, so a user's value leaves it alone."""
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM())
    mem.remember("user", "lives_in", "Berlin")
    alice = mem.scope(user="alice", session="s1")
    alice.remember("user", "lives_in", "Paris")
    assert homes(alice) == ["Berlin", "Paris"]


# -- what is not shadowed -----------------------------------------------------------------

def test_a_read_at_another_instant_is_not_shadowed():
    mem = make()
    mem.remember("user", "lives_in", "Berlin", valid_from=T0, recorded_at=T0)
    session = mem.scope(session="s1")
    session.remember("user", "lives_in", "Paris", valid_from=T1, recorded_at=T1)
    later, between = T1 + timedelta(days=1), T0 + timedelta(days=1)
    assert objects(session.get_all(valid_at=later)) == ["Berlin", "Paris"]
    assert objects(session.get_all(as_of=later)) == ["Berlin", "Paris"]
    assert objects(session.get_all(known_at=later)) == ["Berlin", "Paris"]
    assert objects(session.get_all(valid_at=between)) == ["Berlin"]
    assert sorted(r.claim.object for r in session.search("lives in", valid_at=later)) == [
        "Berlin", "Paris"]


def test_count_is_not_shadowed():
    mem = make()
    mem.remember("user", "lives_in", "Berlin")
    session = mem.scope(agent="a1", session="s1")
    session.remember("user", "lives_in", "Paris")
    assert session.count() == 2
    assert len(session.get_all()) == 1


def test_a_shadowed_value_is_still_readable_by_id():
    mem = make()
    [berlin] = mem.remember("user", "lives_in", "Berlin").added
    session = mem.scope(session="s1")
    session.remember("user", "lives_in", "Paris")
    assert session.get(berlin.id) is not None
    assert session.why(berlin.id) is not None


# -- #273: the chain, and a session's global fact inside a repository ----------------------

def test_the_chain_of_a_reader_bound_to_every_level():
    reader = Scope("t", "u", agent="a", session="s", project=REPO)
    want = [Scope("t", "u", "a", "s", project=REPO), Scope("t", "u", "a", project=REPO),
            Scope("t", "u", project=REPO), Scope("t", "u", "a", "s"), Scope("t", "u", "a"),
            Scope("t", "u"), Scope("t")]
    assert reader.ancestors() == want


@pytest.mark.parametrize("reader,want", [
    (Scope("t", "u", session="s", project=REPO),
     [Scope("t", "u", session="s", project=REPO), Scope("t", "u", project=REPO),
      Scope("t", "u", session="s"), Scope("t", "u"), Scope("t")]),
    (Scope("t", "u", "a", project=REPO),
     [Scope("t", "u", "a", project=REPO), Scope("t", "u", project=REPO),
      Scope("t", "u", "a"), Scope("t", "u"), Scope("t")]),
    (Scope("t", "u", project=REPO),
     [Scope("t", "u", project=REPO), Scope("t", "u"), Scope("t")]),
    (Scope("t", "u", "a", "s"),
     [Scope("t", "u", "a", "s"), Scope("t", "u", "a"), Scope("t", "u"), Scope("t")]),
    (Scope("t", "u"), [Scope("t", "u"), Scope("t")]),
], ids=["project-and-session", "project-and-agent", "project-only", "no-project",
        "user-only"])
def test_the_chain_leaves_out_levels_the_reader_is_not_bound_to(reader, want):
    assert reader.ancestors() == want


@pytest.mark.parametrize("level", [{"session": "s1"}, {"agent": "a1"},
                                   {"agent": "a1", "session": "s1"}],
                         ids=LEVEL_IDS)
def test_a_bound_handle_inside_a_repository_reads_the_global_facts_it_writes(level):
    mem = make()
    inside = mem.scope(project=REPO, **level)
    [claim] = inside.remember("user", "lives_in", "Berlin").added
    assert claim.scope == Scope("default", "alice", project=None, **level)
    assert [c.id for c in inside.get_all()] == [claim.id]
    assert [r.claim.id for r in inside.search("lives in Berlin")] == [claim.id]
    assert inside.get(claim.id) is not None
    explanation = inside.why(claim.id)
    assert explanation is not None and explanation.claim.id == claim.id
    assert [c.id for c in inside.history("user", "lives_in")] == [claim.id]
    # The same handle outside the repository reads it too, as it always did.
    assert [c.id for c in mem.scope(**level).get_all()] == [claim.id]


def test_a_sessions_global_fact_from_add_is_readable_by_that_session():
    """The reproduction in #273, through `add()` and the fast path."""
    session = make().scope(project=REPO, session="s1")
    [claim] = session.add("I live in Berlin.").added
    assert (claim.scope.project, claim.scope.session) == (None, "s1")
    assert [c.id for c in session.get_all()] == [claim.id]
    assert session.why(claim.id) is not None


def test_a_sessions_global_fact_is_not_visible_to_a_sibling_or_the_repository():
    mem = make()
    [claim] = mem.scope(project=REPO, session="s1").remember(
        "user", "lives_in", "Berlin").added
    for other in (mem.scope(project=REPO, session="s2"), mem.scope(project=REPO), mem):
        assert other.get_all() == []
        assert other.why(claim.id) is None


def test_a_sessions_global_value_inside_a_repository_shadows_the_user_wide_one():
    mem = make()
    mem.remember("user", "lives_in", "Berlin")
    session = mem.scope(project=REPO, session="s1")
    session.remember("user", "lives_in", "Paris")
    assert homes(session) == ["Paris"]
    assert homes(mem.scope(project=REPO)) == ["Berlin"]
    assert homes(mem.scope(project=REPO, session="s2")) == ["Berlin"]
    assert homes(mem) == ["Berlin"]


def test_the_repositorys_value_comes_before_the_same_session_outside_it():
    """The chain lists the project-bound levels first. For a project-relative predicate
    that the same session also wrote with no project, the repository's value answers."""
    mem = with_editor()
    mem.scope(session="s1").remember("user", "editor", "vim",
                                     memory_type=MemoryType.PROCEDURAL)
    mem.scope(project=REPO).remember("user", "editor", "emacs",
                                     memory_type=MemoryType.PROCEDURAL)
    inside = mem.scope(project=REPO, session="s1")
    assert objects(c for c in inside.get_all() if c.predicate == "editor") == ["emacs"]
    assert objects(c for c in mem.scope(session="s1").get_all()
                   if c.predicate == "editor") == ["vim"]


# -- the store lookup ---------------------------------------------------------------------

def test_occupied_slots_counts_only_the_scopes_it_is_given():
    store = SQLiteStore(":memory:")
    user, session, sibling = (Scope("t", "u"), Scope("t", "u", session="s1"),
                              Scope("t", "u", session="s2"))
    claim = Claim(subject="user", predicate="lives_in", object="Paris", scope=session)
    store.put_claim(claim)
    key = claim.fact_key
    assert store.occupied_slots("t", [key]) == {key}
    assert store.occupied_slots("t", [key], scopes=[session, user]) == {key}
    assert store.occupied_slots("t", [key], scopes=[sibling, user]) == set()
    store.close()


def test_a_session_read_asks_one_lookup_for_the_levels_above_it(monkeypatch):
    mem = make()
    mem.remember("user", "lives_in", "Berlin")
    session = mem.scope(agent="a1", session="s1")
    session.remember("user", "lives_in", "Paris")
    calls = []
    real = mem.store.occupied_slots

    def counting(tenant, keys, *, scopes=None):
        calls.append([s.key() for s in scopes])
        return real(tenant, keys, scopes=scopes)

    monkeypatch.setattr(mem.store, "occupied_slots", counting)
    assert homes(session) == ["Paris"]
    assert calls == [["default/alice/*/a1/s1", "default/alice/*/a1/*"]]
    calls.clear()
    assert homes(mem) == ["Berlin"]
    assert calls == [], "a user-level read has no narrower level to ask about"
