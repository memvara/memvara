"""Reconciler: contradiction resolution as an index lookup instead of an LLM call.

No LLM is constructed anywhere in this file, and that is the point — every decision here
is a pure function of stored state plus the predicate schema. The tests that matter most
are the ones asserting history survives: superseding must never delete, and an `as_of`
query must still return what we believed at the time.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from engram.schema import (
    BUILTIN_PREDICATES,
    Cardinality,
    PredicateRegistry,
    PredicateSpec,
    Volatility,
)
from engram.store import SQLiteStore
from engram.types import Claim, Episode, Scope, utcnow
from engram.write import Reconciler


@pytest.fixture()
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def registry() -> PredicateRegistry:
    return PredicateRegistry()


@pytest.fixture()
def rec(store, registry) -> Reconciler:
    return Reconciler(store, registry)


SCOPE = Scope("acme", "alice")


def claim(predicate: str, obj: str, *, subject: str = "user", polarity: int = 1,
          scope: Scope = SCOPE, sources=("ep_1",), **kw) -> Claim:
    return Claim(subject=subject, predicate=predicate, object=obj, scope=scope,
                 polarity=polarity, sources=list(sources), **kw)


def live_objects(store, c: Claim, as_of=None):
    return sorted(x.object for x in store.competing_claims(c.scope.tenant, c.fact_key, as_of))


# --- accumulate --------------------------------------------------------------

def test_new_fact_is_added(rec, store):
    res = rec.apply(claim("lives_in", "Berlin"))
    assert res.action == "add"
    assert res.invalidated == []
    assert store.get_claim(res.claim.id) is not None


def test_multi_valued_predicates_accumulate(rec, store):
    a = rec.apply(claim("likes", "coffee"))
    b = rec.apply(claim("likes", "tea"))
    assert (a.action, b.action) == ("add", "add")
    assert b.invalidated == []
    # Both stay live: "likes" takes many values, so these are not competing answers.
    assert live_objects(store, a.claim) == ["coffee", "tea"]


def test_unknown_predicates_default_to_many(rec, store, registry):
    assert not registry.known("collects")
    a = rec.apply(claim("collects", "vinyl"))
    b = rec.apply(claim("collects", "stamps"))
    # Wrongly retiring a true fact is unrecoverable; keeping two only degrades ranking.
    assert b.invalidated == []
    assert live_objects(store, a.claim) == ["stamps", "vinyl"]


# --- reinforce ---------------------------------------------------------------

def test_identical_claim_reinforces_instead_of_inserting(rec, store):
    first = rec.apply(claim("lives_in", "Berlin", sources=["ep_1"]))
    second = rec.apply(claim("lives_in", "Berlin", sources=["ep_2"]))

    assert second.action == "reinforce"
    assert second.claim.id == first.claim.id
    assert second.claim.observation_count == 2
    assert second.claim.salience > first.claim.salience
    assert store.stats()["claims"] == 1


def test_reinforce_merges_sources_rather_than_overwriting(rec):
    rec.apply(claim("works_at", "Acme", sources=["ep_1"]))
    res = rec.apply(claim("works_at", "Acme", sources=["ep_2"]))
    # Provenance is cumulative: three turns supporting a fact is different evidence from
    # one turn whose source id keeps getting replaced.
    assert res.claim.sources == ["ep_1", "ep_2"]


def test_reinforce_does_not_duplicate_a_repeated_source(rec):
    rec.apply(claim("works_at", "Acme", sources=["ep_1"]))
    res = rec.apply(claim("works_at", "Acme", sources=["ep_1"]))
    assert res.claim.sources == ["ep_1"]
    assert res.claim.observation_count == 2


def test_salience_is_capped(rec):
    rec.apply(claim("lives_in", "Berlin"))
    last = None
    for _ in range(50):
        last = rec.apply(claim("lives_in", "Berlin"))
    assert last.claim.salience <= rec.max_salience


# --- supersede ---------------------------------------------------------------

def test_single_valued_predicate_supersedes(rec, store):
    berlin = rec.apply(claim("lives_in", "Berlin")).claim
    res = rec.apply(claim("lives_in", "Lisbon"))

    assert res.action == "supersede"
    assert [c.id for c in res.invalidated] == [berlin.id]
    assert live_objects(store, res.claim) == ["Lisbon"]


def test_superseding_never_deletes(rec, store):
    berlin = rec.apply(claim("lives_in", "Berlin")).claim
    lisbon = rec.apply(claim("lives_in", "Lisbon")).claim

    old = store.get_claim(berlin.id)
    assert old is not None                       # still on disk
    assert old.invalidated_at is not None        # transaction time closed
    assert old.valid_to is not None              # valid time closed
    assert old.invalidated_by == lisbon.id       # and it points at its successor


def test_as_of_still_sees_the_superseded_claim(rec, store):
    t0 = utcnow() - timedelta(days=10)
    t1 = utcnow() - timedelta(days=1)
    berlin = rec.apply(claim("lives_in", "Berlin", valid_from=t0, recorded_at=t0), now=t0)
    rec.apply(claim("lives_in", "Lisbon", valid_from=t1, recorded_at=t1), now=t1)

    # Time travel is the whole reason invalidation is a timestamp and not a DELETE.
    assert live_objects(store, berlin.claim, as_of=t0 + timedelta(days=1)) == ["Berlin"]
    assert live_objects(store, berlin.claim) == ["Lisbon"]


def test_aliases_land_in_the_same_slot(rec, store):
    # "moved_to" is an alias of lives_in. If it stayed a separate predicate the move
    # would never contradict the old city and both would sit there live.
    berlin = rec.apply(claim("lives_in", "Berlin")).claim
    res = rec.apply(claim("moved_to", "Lisbon"))
    assert res.claim.predicate == "lives_in"
    assert [c.id for c in res.invalidated] == [berlin.id]


def test_supersede_ignores_case_and_whitespace_noise_in_the_predicate(rec):
    rec.apply(claim("lives_in", "Berlin"))
    res = rec.apply(claim("  Lives In  ", "Lisbon"))
    assert res.action == "supersede"


# --- retraction --------------------------------------------------------------

def test_retraction_invalidates_without_leaving_a_live_negative(rec, store):
    acme = rec.apply(claim("works_at", "Acme")).claim
    res = rec.apply(claim("works_at", "Acme", polarity=-1, sources=["ep_2"]))

    assert res.action == "retract"
    assert [c.id for c in res.invalidated] == [acme.id]
    assert store.competing_claims("acme", acme.fact_key) == []
    # The tombstone exists for provenance but can never be live at any instant.
    assert res.claim is not None
    assert not res.claim.is_live()
    assert not res.claim.is_live(utcnow() + timedelta(days=365))
    assert store.get_claim(acme.id).invalidated_by == res.claim.id


def test_retraction_only_retires_the_value_it_names(rec, store):
    globex = rec.apply(claim("works_at", "Globex")).claim
    res = rec.apply(claim("works_at", "Acme", polarity=-1))
    # "I no longer work at Acme" says nothing about an employer recorded as Globex.
    assert res.invalidated == []
    assert live_objects(store, globex) == ["Globex"]


def test_repeating_a_retraction_does_not_pile_up_tombstones(rec, store):
    rec.apply(claim("works_at", "Acme"))
    rec.apply(claim("works_at", "Acme", polarity=-1))
    before = store.stats()["claims"]
    res = rec.apply(claim("works_at", "Acme", polarity=-1, sources=["ep_3"]))
    assert res.action == "noop"
    assert store.stats()["claims"] == before


def test_retraction_is_visible_in_history(rec, store):
    t0 = utcnow() - timedelta(days=5)
    acme = rec.apply(claim("works_at", "Acme", valid_from=t0, recorded_at=t0), now=t0).claim
    rec.apply(claim("works_at", "Acme", polarity=-1))
    assert live_objects(store, acme, as_of=t0 + timedelta(days=1)) == ["Acme"]


# --- cross-predicate supersession -------------------------------------------

@pytest.fixture()
def supersede_registry() -> PredicateRegistry:
    reg = PredicateRegistry(BUILTIN_PREDICATES)
    reg.register(PredicateSpec(name="unemployed", cardinality=Cardinality.ONE,
                               volatility=Volatility.SLOW, supersedes=("works_at",)))
    return reg


def test_asserting_a_predicate_retires_the_slot_it_supersedes(store, supersede_registry):
    rec = Reconciler(store, supersede_registry)
    job = rec.apply(claim("works_at", "Acme")).claim
    res = rec.apply(claim("unemployed", "true"))

    # Two different predicate names covering one slot. The key for `works_at` has to be
    # derived the same way the store indexed it, or this lookup matches nothing and the
    # old employer silently stays live.
    assert res.action == "supersede"
    assert [c.id for c in res.invalidated] == [job.id]
    assert store.get_claim(job.id).invalidated_by == res.claim.id


def test_cross_predicate_supersession_stops_at_the_scope_owner(store, supersede_registry):
    rec = Reconciler(store, supersede_registry)
    alice = Scope("acme", "alice")
    bob = Scope("acme", "bob")
    bobs_job = rec.apply(claim("works_at", "Acme", scope=bob)).claim

    res = rec.apply(claim("unemployed", "true", scope=alice))

    # Same tenant, same generic subject "user", same predicate. Alice quitting must not
    # end Bob's employment.
    assert res.invalidated == []
    assert store.get_claim(bobs_job.id).invalidated_at is None


# --- multi-tenant / multi-user isolation ------------------------------------

def test_two_users_in_one_tenant_do_not_collide(rec, store):
    alice = rec.apply(claim("lives_in", "Berlin", scope=Scope("acme", "alice"))).claim
    res = rec.apply(claim("lives_in", "Lisbon", scope=Scope("acme", "bob")))

    assert res.invalidated == []
    assert store.get_claim(alice.id).invalidated_at is None


def test_a_new_session_still_retires_the_old_value(rec, store):
    # Agent and session are deliberately outside the fact key: a durable fact about a
    # person is the same fact whichever session observed it.
    old = rec.apply(claim("lives_in", "Berlin",
                          scope=Scope("acme", "alice", "asst", "s1"))).claim
    res = rec.apply(claim("lives_in", "Lisbon",
                          scope=Scope("acme", "alice", "asst", "s2")))
    assert [c.id for c in res.invalidated] == [old.id]


# --- determinism -------------------------------------------------------------

def test_same_inputs_produce_the_same_actions(registry):
    sequence = [
        ("lives_in", "Berlin", 1),
        ("likes", "coffee", 1),
        ("lives_in", "Berlin", 1),
        ("likes", "tea", 1),
        ("lives_in", "Lisbon", 1),
        ("works_at", "Acme", 1),
        ("works_at", "Acme", -1),
        ("collects", "vinyl", 1),
        ("collects", "stamps", 1),
    ]

    def run():
        s = SQLiteStore(":memory:")
        r = Reconciler(s, PredicateRegistry())
        now = utcnow()
        actions = []
        for pred, obj, pol in sequence:
            # One instant for the whole run, exactly as WritePipeline does it, so the
            # outcome cannot depend on how long the run took.
            res = r.apply(claim(pred, obj, polarity=pol, valid_from=now), now=now)
            actions.append((res.action, len(res.invalidated)))
        live = sorted((c.predicate, c.object) for c in s.iter_claims("acme"))
        s.close()
        return actions, live

    first, second = run(), run()
    assert first == second
    assert first[0] == [
        ("add", 0), ("add", 0), ("reinforce", 0), ("add", 0), ("supersede", 1),
        ("add", 0), ("retract", 1), ("add", 0), ("add", 0),
    ]


def test_a_claim_recorded_after_the_batch_instant_still_reconciles(rec):
    # Transaction time is ours to assign. A Claim built a few microseconds after the
    # batch captured `now` would otherwise fail its own is_live(now) check and silently
    # neither reinforce nor supersede anything.
    past = utcnow() - timedelta(days=1)
    now = utcnow()
    rec.apply(claim("lives_in", "Berlin", valid_from=past, recorded_at=past), now=past)

    ahead = claim("lives_in", "Berlin", valid_from=past,
                  recorded_at=now + timedelta(seconds=5), sources=["ep_2"])
    res = rec.apply(ahead, now=now)
    assert res.action == "reinforce"
    assert res.claim.recorded_at <= now


def test_backdating_a_claim_is_still_allowed(rec, store):
    # Clamping only pulls the future back; imports and replays must keep their own past.
    past = utcnow() - timedelta(days=30)
    res = rec.apply(claim("lives_in", "Berlin", valid_from=past, recorded_at=past))
    assert store.get_claim(res.claim.id).recorded_at == past


def test_victim_order_is_stable(store, registry):
    # Two live claims in a ONE slot can only arise if the schema changed under us; the
    # order they are reported in must still not depend on row order from the store.
    rec = Reconciler(store, registry)
    rec.apply(claim("collects", "vinyl"))       # unknown -> MANY, so both survive
    rec.apply(claim("collects", "stamps"))
    registry.learn("collects", Cardinality.ONE)
    res = rec.apply(claim("collects", "books"))
    assert res.action == "supersede"
    assert [c.object for c in res.invalidated] == ["vinyl", "stamps"]


# --- hygiene -----------------------------------------------------------------

def test_no_llm_is_reachable_from_the_reconciler(rec):
    # The design claim in one assertion: reconciliation has no model to call.
    assert not hasattr(rec, "llm")


def test_text_is_rendered_when_absent(rec):
    res = rec.apply(claim("lives_in", "Berlin"))
    assert res.claim.text == "user lives in Berlin"


def test_caller_supplied_text_is_preserved(rec):
    c = claim("lives_in", "Berlin")
    c.text = "Alice has been living in Berlin since 2019"
    res = rec.apply(c)
    assert res.claim.text == "Alice has been living in Berlin since 2019"


def test_episode_scoped_claim_round_trips(rec, store):
    e = Episode(content="I live in Berlin.", scope=SCOPE)
    res = rec.apply(claim("lives_in", "Berlin", sources=[e.id]))
    assert store.get_claim(res.claim.id).sources == [e.id]
