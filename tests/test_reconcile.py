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
from memvara.types import Claim, Episode, Scope, close_out, utcnow
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
    # The store speaks in two axes now; this helper still speaks in one, because every
    # question the reconciler asks is about a single write instant.
    return sorted(x.object for x in store.competing_claims(
        c.scope.tenant, c.fact_key, valid_at=as_of, known_at=as_of))


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


# --- accumulation, reported --------------------------------------------------
# The default above is right and it used to be silent, which is a separate defect. These
# fix the trigger: an `add` onto an *unregistered* predicate whose slot already holds live
# values, and nothing else.

def test_a_second_value_under_an_undeclared_predicate_is_reported(rec, registry):
    """The measured case, at the layer that knows what was already there.

    `status` is not in the schema and never will be on this path, so the second write
    accumulates instead of superseding and both values go on answering. That is defensible
    behaviour and an indefensible silence: the receipt for it was byte-identical to the
    receipt for a correct replacement."""
    assert not registry.known("status")
    first = rec.apply(claim("status", "not installed", subject="quota_gate"))
    second = rec.apply(claim("status", "installed", subject="quota_gate"))

    assert first.accumulated is None          # nothing was there to pile up beside
    assert second.action == "add" and second.invalidated == []
    assert second.accumulated is not None
    assert second.accumulated.subject == "quota_gate"
    assert second.accumulated.predicate == "status"
    assert second.accumulated.existing == 1


def test_the_first_write_to_an_empty_slot_is_silent(rec, registry):
    """Nothing was displaced and nothing is competing. A report here would fire on every
    first write in the store and would be pure noise."""
    assert not registry.known("deployed_to")
    assert rec.apply(claim("deployed_to", "staging")).accumulated is None


def test_a_registered_many_predicate_is_silent(rec, registry):
    """`likes` is declared MANY by a person, so accumulating is the decision rather than
    the absence of one. This must not fire there, or the signal drowns in the predicates
    that are working exactly as designed."""
    assert registry.known("likes") and not registry.spec("likes").functional
    rec.apply(claim("likes", "coffee"))
    assert rec.apply(claim("likes", "tea")).accumulated is None


def test_a_predicate_learned_at_runtime_stops_being_reported(rec, registry):
    """The note is a request for a decision, so making the decision has to end it — in
    *both* directions. Declared MANY, this goes quiet; declared ONE, the write supersedes
    and there is nothing left to report. Either way one declaration settles the predicate
    permanently, which is what makes the report worth reading rather than noise to filter."""
    rec.apply(claim("stage", "canary"))
    assert rec.apply(claim("stage", "beta")).accumulated is not None

    registry.learn("stage", Cardinality.MANY)
    assert rec.apply(claim("stage", "ga")).accumulated is None

    registry.register(PredicateSpec("branch", Cardinality.ONE, Volatility.FAST))
    rec.apply(claim("branch", "main"))
    settled = rec.apply(claim("branch", "release"))
    assert settled.action == "supersede" and settled.accumulated is None


def test_re_stating_the_same_value_is_silent(rec, registry):
    """A reinforcement is the slot's *own* value arriving again, not a second answer to
    the question. `apply` returns before the slot is ever counted."""
    assert not registry.known("status")
    rec.apply(claim("status", "installed", subject="quota_gate"))
    again = rec.apply(claim("status", "installed", subject="quota_gate"))
    assert again.action == "reinforce" and again.accumulated is None


def test_a_retraction_is_silent(rec, registry):
    """A negative assertion removes an answer; it cannot pile one up."""
    assert not registry.known("status")
    rec.apply(claim("status", "installed", subject="quota_gate"))
    gone = rec.apply(claim("status", "installed", subject="quota_gate", polarity=-1))
    assert gone.action == "retract" and gone.accumulated is None


def test_a_slot_whose_only_value_is_no_longer_live_is_silent(rec, registry):
    """Two values a year apart, the first already ended, are history rather than a
    pile-up — nothing is competing, so nothing is reported.

    The occupancy question is asked of `competing_claims`, which is contractually
    live-only, and not of `slot_history`, which is every claim the slot ever held. That
    is the mistake this pins: a report built on the audit trail would fire on every
    ordinary sequence of values in the store's history and mean nothing."""
    assert not registry.known("status")
    first = rec.apply(claim("status", "not installed", subject="quota_gate"))
    first.claim.valid_to = utcnow() - timedelta(seconds=1)
    rec.store.put_claim(first.claim)
    assert rec.apply(claim("status", "installed", subject="quota_gate")).accumulated is None


def test_another_users_value_in_the_same_named_slot_is_silent(rec, registry):
    """The owner filter is the same one every other lookup here applies. Bob's
    `quota_gate status` is not an answer competing with Alice's."""
    assert not registry.known("status")
    rec.apply(claim("status", "not installed", subject="quota_gate",
                    scope=Scope("acme", "bob")))
    assert rec.apply(claim("status", "installed",
                           subject="quota_gate")).accumulated is None


def test_a_store_predating_count_competing_still_reports(store, registry):
    """The cheap count is an optional capability, reached through `getattr` exactly as
    `batch` and `put_spec` are. A third-party `Store` that never heard of it must get the
    same answer — it simply pays the hydration it was already paying."""
    class OldStore(SQLiteStore):
        count_competing = None

    old = OldStore(":memory:")
    older = Reconciler(old, registry)
    older.apply(claim("status", "not installed", subject="quota_gate"))
    second = older.apply(claim("status", "installed", subject="quota_gate"))
    assert second.accumulated is not None and second.accumulated.existing == 1
    # And the branch is genuinely the other one, not a method that happens to exist.
    assert getattr(old, "count_competing", None) is None
    old.close()


def test_the_reported_count_grows_with_the_slot(rec):
    """The number is the live occupants *before* this write, so a slot that keeps growing
    reports a worse number each time rather than repeating "1"."""
    rec.apply(claim("owner", "platform", subject="quota_gate"))
    rec.apply(claim("owner", "payments", subject="quota_gate"))
    third = rec.apply(claim("owner", "security", subject="quota_gate"))
    assert third.accumulated.existing == 2


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
    assert old.valid_to is not None              # valid time closed: it stopped being true
    assert old.invalidated_by == lisbon.id       # and it points at its successor


def test_supersession_ends_a_claim_rather_than_calling_it_a_mistake(rec, store):
    """The defect this whole rule exists to remove: `_retire` closed *both* clocks.

    `valid_to` was right — Berlin stopped being true when Lisbon began. `invalidated_at`
    was not: it says *we no longer believe this record*, and the record was never wrong.
    Every superseded claim in every store this library wrote was therefore marked as an
    error, which is what made "what do we now believe was true in June" return nothing.
    """
    berlin = rec.apply(claim("lives_in", "Berlin")).claim
    rec.apply(claim("lives_in", "Lisbon"))

    old = store.get_claim(berlin.id)
    assert old.invalidated_at is None, "we were never wrong about Berlin"
    assert old.state == "ended", "the world moved on, which is a different word"


def test_a_correcting_caller_can_say_so_and_gets_the_other_axis(rec, store):
    """The reading the reconciler cannot reach on its own, and the reason `close` exists.

    "Lisbon is right and Berlin was never true" is a statement about the record, not
    about the world — so belief stops and the valid interval is left exactly as written,
    because a correction witnessed no world event and must not invent one.
    """
    berlin = rec.apply(claim("lives_in", "Berlin")).claim
    rec.apply(claim("lives_in", "Lisbon"), close="retired")

    old = store.get_claim(berlin.id)
    assert old.state == "retired"
    assert old.invalidated_at is not None
    assert old.valid_to is None, "a correction saw nothing stop being true"
    assert not old.is_live()


@pytest.mark.parametrize("close,axes", [
    ("ended", ("valid_to", "invalidated_at")),
    ("retired", ("invalidated_at", "valid_to")),
])
def test_each_closure_moves_exactly_one_clock(rec, store, close, axes):
    """Stated as the general rule, because it is the invariant and not two behaviours.

    One write, one assertion, one clock. A closure that moved both would be saying two
    things — "it stopped being true" *and* "we were mistaken" — and only one of them
    can be what the caller meant.
    """
    moved, still = axes
    first = rec.apply(claim("lives_in", "Berlin")).claim
    rec.apply(claim("lives_in", "Lisbon"), close=close)

    old = store.get_claim(first.id)
    assert getattr(old, moved) is not None
    assert getattr(old, still) is None
    assert old.state == close


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


# --- authority: what a candidate has to be worth to close something ----------
#
# Until `AUTHORITY_SHARE` existed the answer was "nothing". Contradiction resolution was
# predicate cardinality plus write order, so a 0.10 guess replaced a 1.00 statement — and
# stamped it `ended`, which asserts the world changed. Nothing about the world had
# changed; a machine had guessed, and the store recorded a world event as the reason.

def test_a_guess_does_not_end_a_statement_it_is_worth_a_tenth_of(rec, store):
    """The reported defect, exactly. `confidence` appeared nowhere in this module.

    Note which half of it matters more. The bad ranking is recoverable — both values are
    live, the confident one ranks first, and a caller can settle it. The false `ended` is
    not: it is an assertion about the world that no evidence supports, written into the
    one axis whose whole purpose is answering "what do we now believe was true then", and
    `CLAUDE.md` names this distinction as the one mistake here that cannot be found by
    reading the data afterwards.
    """
    london = rec.apply(claim("lives_in", "London", confidence=1.00)).claim
    res = rec.apply(claim("lives_in", "Paris", confidence=0.10))

    assert res.action == "add", "it was stored; nothing here refuses a write"
    assert res.invalidated == [], "and it closed nothing"
    assert store.get_claim(london.id).state == "live", "no world event was invented"
    assert live_objects(store, res.claim) == ["London", "Paris"]


def test_the_dispute_names_both_values_and_what_each_is_worth(rec):
    """A caller acting on this has one decision to make about one pair of claims, so the
    report carries the pair rather than the count `Accumulation` carries. Without it the
    receipt reads `added 1, ended 0`, which is also what a correct first write reads."""
    london = rec.apply(claim("lives_in", "London", confidence=1.00)).claim
    res = rec.apply(claim("lives_in", "Paris", confidence=0.10))

    (dispute,) = res.disputed
    assert dispute.claim_id == london.id
    assert (dispute.incumbent, dispute.incumbent_confidence) == ("London", 1.00)
    assert (dispute.candidate, dispute.candidate_confidence) == ("Paris", 0.10)


def test_an_ordinary_supersession_between_comparable_claims_still_supersedes(rec, store):
    """The regression this rule could most easily have caused, named so it cannot happen
    quietly. Every confidence the shipped write paths produce sits at or above half of
    every other one — 1.00 from `remember()`, 0.95 from the fast path, 0.70 from an
    extraction whose model gave no figure, 0.50 from one that ignored the schema — so
    ordinary traffic must pass this rule untouched. A store that stopped superseding is a
    store that stopped learning."""
    berlin = rec.apply(claim("lives_in", "Berlin", confidence=1.00)).claim
    res = rec.apply(claim("lives_in", "Lisbon", confidence=0.70))

    assert res.action == "supersede"
    assert [c.id for c in res.invalidated] == [berlin.id]
    assert res.disputed == []
    assert live_objects(store, res.claim) == ["Lisbon"]


def test_the_extraction_default_still_displaces_a_stated_fact(rec, store):
    """The case the rule does **not** cover, pinned so the docstring cannot drift from it.

    0.70 is the documented default for an extraction whose model gave no figure, and
    `0.70 >= 0.5 * 1.00`, so a mined paraphrase still closes a fact a person asserted at
    1.00. Reading `AUTHORITY_SHARE` as protection against that is the wrong inference, and
    it is the one a reader actually draws — so the boundary is a test rather than a
    sentence. Blocking it would stop the store learning from conversation, which the
    comparable-claims test above holds.
    """
    stated = rec.apply(claim("lives_in", "Lisbon", confidence=1.00)).claim
    res = rec.apply(claim("lives_in", "Porto", confidence=0.70))

    assert res.action == "supersede"
    assert res.disputed == []
    assert store.get_claim(stated.id).state == "ended"


@pytest.mark.parametrize("candidate, displaces", [
    (0.50, True),    # exactly the share: an even match closes
    (0.49, False),   # just under it
])
def test_the_share_is_a_floor_and_not_a_margin(rec, store, candidate, displaces):
    """At exactly half the candidate wins, because the test is "worth at least" and not
    "worth more than". The boundary is pinned because it is the whole of the rule: 0.50 is
    `llm._shape.UNKNOWN_CONFIDENCE`, what a model that ignored the schema gets, and that
    is not evidence of a guess — it is an absence of evidence about how sure the model
    was, which must not be read as an admission."""
    rec.apply(claim("lives_in", "London", confidence=1.00))
    res = rec.apply(claim("lives_in", "Paris", confidence=candidate))

    assert (res.action == "supersede") is displaces
    assert live_objects(store, res.claim) == (["Paris"] if displaces
                                              else ["London", "Paris"])


def test_authority_is_read_against_the_incumbent_and_not_against_a_fixed_floor(rec, store):
    """A share, not a threshold. The same 0.30 claim is a guess beside a stated fact and
    an even match beside another uncertain one, and only a ratio says both."""
    rec.apply(claim("lives_in", "London", confidence=0.50))
    res = rec.apply(claim("lives_in", "Paris", confidence=0.30))

    assert res.action == "supersede", "0.30 is more than half of 0.50"
    assert live_objects(store, res.claim) == ["Paris"]


def test_the_authority_rule_does_not_reach_a_caller_who_named_the_victim(rec, store):
    """Pinned because four documents state the rule as an invariant, and an invariant with
    a silent exception is worse than a narrower one stated plainly.

    `Memvara.supersede`, `forget` and `delete` all close a claim the caller named, before
    this module is asked anything — so there is no candidate to weigh against it. That is
    the same boundary `close="retired"` sits on: the rule arbitrates an *inference* the
    write path drew, and naming the row to close is an instruction rather than an
    inference. Asserted here at the reconciler, where the absence of a victim is the
    mechanism: a claim already closed is not live, so it is never a competing claim.
    """
    london = rec.apply(claim("lives_in", "London", confidence=1.00)).claim
    close_out(london, utcnow(), None, "ended")     # what `supersede` does first
    store.put_claim(london)

    res = rec.apply(claim("lives_in", "Paris", confidence=0.10))

    assert res.disputed == [], "there was nothing live left to dispute with"
    assert res.action == "add"
    assert store.get_claim(london.id).state == "ended"


def test_a_low_confidence_retraction_still_retracts(rec, store):
    """`AUTHORITY_SHARE` is deliberately not consulted on the retraction path, and this
    pins the decision so it does not read later as a place the rule was forgotten. A
    retraction writes a tombstone that is born invalidated — "we stopped believing X" —
    and leaving the target live beside it would put both sentences in the store at once,
    which is a worse record than either."""
    berlin = rec.apply(claim("lives_in", "Berlin", confidence=1.00)).claim
    res = rec.apply(claim("lives_in", "Berlin", polarity=-1, confidence=0.10))

    assert res.action == "retract"
    assert [c.id for c in res.invalidated] == [berlin.id]
    assert store.get_claim(berlin.id).state == "ended"


# --- the interval a supersession can empty -----------------------------------
#
# `close_out` clamps a closure to the claim's own start rather than inverting the
# interval, which is right: a fact that ends before it begins is a row no `as_of` window
# can return consistently. What the clamp cannot do is make the row answer anything.

def test_a_value_replaced_at_the_instant_it_began_is_true_at_no_instant(rec, store):
    """Two writes sharing a `valid_from` — any same-day correction, and every import that
    stamps dates rather than timestamps. The displaced claim keeps `state == "ended"`,
    which `core.py` documents as "still answers `valid_at=<while it held>`", and there is
    no instant at which it held."""
    at = utcnow() - timedelta(days=30)
    delhi = rec.apply(claim("lives_in", "Delhi", valid_from=at, recorded_at=at),
                      now=at).claim
    rec.apply(claim("lives_in", "Mumbai", valid_from=at, recorded_at=at), now=at)

    old = store.get_claim(delhi.id)
    assert old.valid_from == old.valid_to, "the interval is empty"
    for probe in (at - timedelta(days=1), at, at + timedelta(days=1)):
        assert "Delhi" not in live_objects(store, old, as_of=probe)


def test_the_write_that_empties_an_interval_says_so(rec, store):
    """The reason this is a defect and not merely a shape. `invalidated 1` is what an
    ordinary supersession reports too, so the difference between "still answers about the
    period it held" and "answers nothing, ever" had no symptom at the write, and the one
    query that would reveal it is the one nobody runs against a claim they just closed."""
    at = utcnow() - timedelta(days=30)
    delhi = rec.apply(claim("lives_in", "Delhi", valid_from=at, recorded_at=at),
                      now=at).claim
    res = rec.apply(claim("lives_in", "Mumbai", valid_from=at, recorded_at=at), now=at)

    (collapse,) = res.collapsed
    assert collapse.claim_id == delhi.id
    assert (collapse.subject, collapse.predicate, collapse.object) == (
        "user", "lives_in", "Delhi")
    assert collapse.at == at


def test_an_ordinary_supersession_leaves_the_displaced_interval_alone(rec, store):
    """The other half, or the report would fire on every supersession in the store.
    Berlin held for nine days and goes on answering about them."""
    t0 = utcnow() - timedelta(days=10)
    t1 = utcnow() - timedelta(days=1)
    berlin = rec.apply(claim("lives_in", "Berlin", valid_from=t0, recorded_at=t0),
                       now=t0).claim
    res = rec.apply(claim("lives_in", "Lisbon", valid_from=t1, recorded_at=t1), now=t1)

    assert res.collapsed == []
    assert live_objects(store, berlin, as_of=t0 + timedelta(days=1)) == ["Berlin"]


def test_a_retired_closure_cannot_empty_an_interval(rec, store):
    """`close="retired"` stops the belief clock and leaves valid time exactly as written,
    so there is no interval to empty and nothing to report. Pinned because the detection
    reads `valid_to`, and a closure that never sets it must not trip it."""
    at = utcnow() - timedelta(days=30)
    delhi = rec.apply(claim("lives_in", "Delhi", valid_from=at, recorded_at=at),
                      now=at).claim
    res = rec.apply(claim("lives_in", "Mumbai", valid_from=at, recorded_at=at),
                    now=at, close="retired")

    assert res.collapsed == []
    assert store.get_claim(delhi.id).valid_to is None


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


def test_a_retraction_ends_its_target_rather_than_calling_it_a_mistake(rec, store):
    """"I no longer work at Acme" is news about the world, not a complaint about us.

    The employment was real and it finished, so valid time closes and the record stays
    believed — which is what keeps "where did they work in 2023" answerable after they
    leave. Every negative form the write path can produce is of that shape ("no longer",
    "used to", "not ... any more"), and `Claim.render` words a negative claim the same
    way, so this is the reading the data supports rather than a preference.
    """
    acme = rec.apply(claim("works_at", "Acme")).claim
    res = rec.apply(claim("works_at", "Acme", polarity=-1, sources=["ep_2"]))

    old = store.get_claim(acme.id)
    assert old.state == "ended"
    assert old.invalidated_at is None
    assert old.invalidated_by == res.claim.id, "and it still points at the retraction"


def test_a_retraction_that_is_a_correction_says_so(rec, store):
    """The other reading, reachable and different: "you misheard me, I never worked
    there." Nothing about the world changed, so nothing on the world clock moves."""
    acme = rec.apply(claim("works_at", "Acme")).claim
    rec.apply(claim("works_at", "Acme", polarity=-1, sources=["ep_2"]), close="retired")

    old = store.get_claim(acme.id)
    assert old.state == "retired"
    assert old.valid_to is None
    assert live_objects(store, acme) == []


def test_the_retraction_tombstone_is_unreachable_from_either_clock(rec, store):
    """The one row that closes both axes, and the only one that may.

    A tombstone is bookkeeping, not an assertion about the world, so it has no true
    interval to preserve and nothing an audit loses by it never being live. It exists so
    "why did you stop believing that?" has an answer with source episodes attached.
    """
    rec.apply(claim("works_at", "Acme"))
    res = rec.apply(claim("works_at", "Acme", polarity=-1, sources=["ep_2"]))

    assert res.claim.invalidated_at is not None and res.claim.valid_to is not None
    for kw in ({}, {"as_of": utcnow() + timedelta(days=365)},
               {"valid_at": utcnow() - timedelta(days=365)}):
        assert not res.claim.is_live(**kw), kw


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
        # `is_live` and not `iter_claims`'s default filter: that one reads the belief
        # axis alone, and a superseded claim now keeps its belief axis open.
        live = sorted((c.predicate, c.object)
                      for c in s.iter_claims("acme") if c.is_live())
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
    # Berlin then, we believe it now, and the record has to keep saying so. A
    # supersession is never told the old value was wrong, so it never says it was.
    assert berlin.invalidated_at is None

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
    transaction time is not the reconciler's to touch on a supersession at all.

    The two instants coincide only when a claim is applied the moment it is built, which
    is the common case and precisely why stamping one onto the other went unnoticed."""
    now = utcnow()
    first = rec.apply(claim("lives_in", "Berlin"), now=now)
    later = now + timedelta(minutes=5)
    second = rec.apply(claim("lives_in", "Lisbon"), now=later)

    berlin = store.get_claim(first.claim.id)
    assert berlin.valid_to == second.claim.valid_from   # when it stopped being true
    assert berlin.invalidated_at is None                # we never stopped believing it
    assert berlin.valid_to < later


def test_a_superseded_claim_still_answers_what_was_true_back_then(rec, store):
    """The question the conflation emptied out, at the layer that emptied it.

    Berlin was true from 2023 and Lisbon from 2026. Asked what we *now* believe was true
    in 2024, the store has to say Berlin — and it could not, because the write path had
    marked Berlin as a record we no longer believe. `as_of` kept working the whole time,
    which is exactly why this went unnoticed: it rewinds the belief clock to before the
    supersession, so it never reads the stamp that was wrong.
    """
    then = utcnow() - timedelta(days=1200)
    moved = utcnow() - timedelta(days=100)
    mid = utcnow() - timedelta(days=600)
    first = rec.apply(claim("lives_in", "Berlin", valid_from=then, recorded_at=then),
                      now=then).claim
    rec.apply(claim("lives_in", "Lisbon", valid_from=moved, recorded_at=moved), now=moved)

    assert live_objects(store, first, as_of=mid) == ["Berlin"]
    believed_now_about_then = [
        c.object for c in store.competing_claims("acme", first.fact_key, valid_at=mid)]
    assert believed_now_about_then == ["Berlin"]
