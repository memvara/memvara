"""Reconciler: contradiction resolution as an index lookup instead of an LLM call.

No LLM is constructed anywhere in this file, and that is the point — every decision here
is a pure function of stored state plus the predicate schema. The tests that matter most
are the ones asserting history survives: superseding must never delete, and an `as_of`
query must still return what we believed at the time.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from memvara.schema import (
    BUILTIN_PREDICATES,
    Cardinality,
    PredicateRegistry,
    PredicateSpec,
    Volatility,
)
from memvara.store import SQLiteStore
from memvara.types import Claim, Episode, Scope, utcnow
from memvara.write import Reconciler


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
    rec.reinforce_bump = 2.0            # a caller who wants repetition to count for a lot
    rec.apply(claim("lives_in", "Berlin"))
    last = None
    for _ in range(50):
        last = rec.apply(claim("lives_in", "Berlin"))
    assert last.claim.salience == pytest.approx(rec.max_salience)


def test_a_bump_lands_on_storage_strength_and_stamps_the_observation(rec, store):
    """Both halves, because either one alone reproduces a shipped bug.

    Written onto `salience` instead of the base, a bump is erased by the next decay
    pass. Written without the timestamp, it decays from the original `valid_from`, so
    freshness never recovers however often the fact is restated.
    """
    now = utcnow().replace(microsecond=0)
    first = rec.apply(claim("lives_in", "Berlin", valid_from=now), now=now)
    assert first.claim.last_observed is None      # a first sighting is not a re-sighting

    second = rec.apply(
        claim("lives_in", "Berlin", valid_from=now, sources=["ep_2"]), now=now)
    assert second.action == "reinforce"
    stored = store.get_claim(second.claim.id)
    assert stored.salience_base == pytest.approx(1.025)
    assert stored.salience == pytest.approx(1.025)
    assert stored.last_observed == now
    assert stored.trace_from == now


def test_a_weak_trace_earns_more_than_a_strong_one(rec, store):
    """Bjork & Bjork: the gain from a successful retrieval is inversely related to how
    available the memory already was. The shipped rule was a flat bump, which made
    massed repetition the cheapest possible way to buy salience."""
    fresh = rec.apply(claim("likes", "coffee")).claim
    faded = rec.apply(claim("likes", "tea")).claim
    faded.meta["salience_base"] = 1.0
    faded.salience = 0.1                     # nine tenths of the trace has gone
    store.put_claim(faded)

    strong = rec.reinforce(store.get_claim(fresh.id), ["ep_2"])
    weak = rec.reinforce(store.get_claim(faded.id), ["ep_2"])

    assert strong.salience == pytest.approx(1.025)     # 1.0 + 0.25 * 0.1
    assert weak.salience == pytest.approx(1.2275)      # 1.0 + 0.25 * (0.1 + 0.9 * 0.9)
    assert weak.salience_base > strong.salience_base


def test_a_replayed_observation_is_stamped_when_it_happened_not_today(rec, store):
    """A replay's whole value is reconstructing real history.

    Stamped with wall-clock now, every re-observation an importer replays claims to
    have happened today - so the recency signal the import exists to rebuild is
    destroyed at the moment it is written, and a fact last mentioned in 2024 ranks as
    freshly restated.
    """
    t0 = utcnow() - timedelta(days=800)
    t1 = utcnow() - timedelta(days=430)
    rec.apply(claim("working_on", "auth refactor", valid_from=t0, recorded_at=t0))
    res = rec.apply(claim("working_on", "auth refactor", valid_from=t1, recorded_at=t1,
                          sources=["ep_2"]))

    assert res.action == "reinforce"
    assert res.claim.last_observed == t1
    assert res.claim.trace_from == t1


def test_the_observation_stamp_only_ever_moves_forward(rec, store):
    """Replays do not arrive in order; recency must not depend on the order they do."""
    t0 = utcnow() - timedelta(days=800)
    recent = utcnow() - timedelta(days=10)
    old = utcnow() - timedelta(days=400)
    rec.apply(claim("working_on", "auth refactor", valid_from=t0, recorded_at=t0))
    rec.apply(claim("working_on", "auth refactor", valid_from=recent, recorded_at=recent))

    res = rec.apply(claim("working_on", "auth refactor", valid_from=old, recorded_at=old))
    assert res.claim.last_observed == recent


def test_a_future_dated_restatement_cannot_backdate_the_present(rec, store):
    """The same clamp `recorded_at` gets: a fact asserted as true from next month is
    not evidence about how fresh anything is today."""
    now = utcnow().replace(microsecond=0)
    rec.apply(claim("lives_in", "Berlin", valid_from=now), now=now)
    res = rec.apply(claim("lives_in", "Berlin", valid_from=now + timedelta(days=30),
                          sources=["ep_2"]), now=now)

    assert res.action == "reinforce"
    assert res.claim.last_observed == now


def test_reinforcement_works_on_a_store_whose_decay_pass_never_ran(rec, store):
    """No `salience_base` in `meta` means salience *is* the base - the honest reading
    for a claim nothing has decayed, and the one that keeps a library used without the
    consolidator from silently refusing to reinforce."""
    added = rec.apply(claim("likes", "coffee")).claim
    assert "salience_base" not in store.get_claim(added.id).meta

    again = rec.reinforce(store.get_claim(added.id), ["ep_2"])
    assert again.salience == pytest.approx(1.025)


def test_reinforcement_survives_a_zero_salience_claim(rec, store):
    """A third-party writer can leave a 0.0 in the column; dividing by it in the
    retrievability ratio would take down the write path."""
    added = rec.apply(claim("likes", "coffee")).claim
    added.salience = 0.0
    added.meta["salience_base"] = 0.0
    store.put_claim(added)

    again = rec.reinforce(store.get_claim(added.id), ["ep_2"])
    assert again.salience == pytest.approx(0.025)


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


def test_retraction_and_deduplication_use_one_notion_of_identity(rec, store):
    """The asymmetry this closes: `_retract` matched objects casefolded while
    `value_key` hashed them case-*sensitively*, so retraction folded case and
    deduplication did not. Whichever end you read it from, one of the two was wrong."""
    acme = rec.apply(claim("works_at", "Acme Corp")).claim
    res = rec.apply(claim("works_at", "acme, inc.", polarity=-1))
    assert res.action == "retract"
    assert [c.id for c in res.invalidated] == [acme.id]
    assert live_objects(store, acme) == []


def test_a_retraction_naming_a_different_entity_still_hits_nothing(rec, store):
    globex = rec.apply(claim("works_at", "Globex Ltd")).claim
    assert rec.apply(claim("works_at", "Globex Labs", polarity=-1)).invalidated == []
    assert live_objects(store, globex) == ["Globex Ltd"]


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


# --- the two time axes are not one axis ----------------------------------------

def test_a_backdated_supersession_closes_valid_time_where_the_new_value_begins(rec, store):
    """The bug this pins: `_retire` used to stamp transaction time on *both* axes.

    Learning today that someone moved in July has to close the old value in July. Stamped
    with today instead, Berlin stays "true" through a window in which Lisbon is also true
    — two live answers to a single-valued question, which is the precise failure this
    whole design exists to make impossible. It is invisible unless a write is backdated,
    because `valid_from` otherwise defaults to now and the two axes coincide.
    """
    now = utcnow()
    moved = now - timedelta(days=30)
    first = rec.apply(claim("lives_in", "Berlin",
                            valid_from=now - timedelta(days=800),
                            recorded_at=now - timedelta(days=800)),
                      now=now - timedelta(days=800))
    rec.apply(claim("lives_in", "Lisbon", valid_from=moved, recorded_at=now), now=now)

    berlin = store.get_claim(first.claim.id)
    assert berlin.valid_to == moved, "valid time closed at the wrong instant"
    # Transaction time is a different question with a different answer: we believed
    # Berlin right up until today, and the record has to keep saying so.
    assert berlin.invalidated_at == now

    # The intervals abut rather than overlap: Berlin ends exactly where Lisbon starts,
    # so no instant on the valid-time axis has two answers to a single-valued question.
    lisbon = next(c for c in store.iter_claims("acme") if c.object == "Lisbon")
    assert berlin.valid_to == lisbon.valid_from
    assert lisbon.valid_to is None


def test_a_retraction_backdated_before_the_fact_collapses_rather_than_inverting(rec, store):
    """`valid_to` can meet `valid_from` but must never precede it: an interval that ends
    before it starts is not a shorter fact, it is a corrupt row that no `as_of` window
    can return consistently."""
    now = utcnow()
    first = rec.apply(claim("lives_in", "Berlin", valid_from=now, recorded_at=now), now=now)
    rec.apply(claim("lives_in", "Berlin", polarity=-1,
                    valid_from=now - timedelta(days=500), recorded_at=now), now=now)

    berlin = store.get_claim(first.claim.id)
    assert berlin.valid_to == berlin.valid_from
    assert berlin.valid_to >= berlin.valid_from


def test_the_two_axes_stay_distinct_even_without_explicit_backdating(rec, store):
    """The general contract, stated once: valid time follows the *new value*, and
    transaction time follows the *apply instant*. They coincide only when a claim is
    applied the moment it is built, which is the common case and precisely why stamping
    one onto the other went unnoticed."""
    now = utcnow()
    first = rec.apply(claim("lives_in", "Berlin"), now=now)
    later = now + timedelta(minutes=5)
    second = rec.apply(claim("lives_in", "Lisbon"), now=later)

    berlin = store.get_claim(first.claim.id)
    assert berlin.valid_to == second.claim.valid_from   # when it stopped being true
    assert berlin.invalidated_at == later               # when we stopped believing it
    assert berlin.valid_to < berlin.invalidated_at
