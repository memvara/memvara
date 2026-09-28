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
from memvara.types import Claim, Derivation, Episode, Scope, close_out, utcnow
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


@pytest.mark.covers("inv:CG6")
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


def test_a_restatement_with_an_earlier_start_is_kept_for_the_period_before_the_claim(
        rec, store):
    """The same value, stated as true from before the claim on record begins.
    Reinforcing that claim dropped the earlier start, so no read of the earlier period
    found the fact (#283). Moving the claim's start instead would change what reads of
    the past return. So the earlier period is stored as a claim of its own, ending where
    the claim on record begins, and the claim on record is left exactly as it was."""
    april = utcnow() - timedelta(days=150)
    january = april - timedelta(days=90)
    first = rec.apply(claim("likes", "tea", valid_from=april, recorded_at=april),
                      now=april).claim
    res = rec.apply(claim("likes", "tea", valid_from=january, sources=["ep_2"]))

    assert res.action == "add" and res.invalidated == []
    assert (res.claim.valid_from, res.claim.valid_to) == (january, april)
    assert res.restated is not None and res.restated.id == first.id
    kept = store.get_claim(first.id)
    assert (kept.valid_from, kept.valid_to, kept.observation_count) == (april, None, 1)
    february = january + timedelta(days=30)
    assert [c.id for c in store.competing_claims("acme", first.fact_key,
                                                 valid_at=february)] == [res.claim.id]
    assert store.competing_claims("acme", first.fact_key, valid_at=february,
                                  known_at=april + timedelta(days=1)) == [], (
        "what the store believed before the restatement has not changed")


def restated(rec, store, *, days_before_january: int):
    """Tea on record from April, then restated from January, which stores January to
    April. Returns the claim for that period, and a restatement of the same value that
    begins `days_before_january` days before January (a negative number is after it)."""
    april = utcnow() - timedelta(days=150)
    january = april - timedelta(days=90)
    rec.apply(claim("likes", "tea", valid_from=april, recorded_at=april), now=april)
    earlier = rec.apply(claim("likes", "tea", valid_from=january, sources=["ep_2"])).claim
    assert (earlier.valid_from, earlier.valid_to) == (january, april)
    return earlier, claim("likes", "tea", sources=["ep_3"],
                          valid_from=january - timedelta(days=days_before_january))


@pytest.mark.parametrize("days_before_january", [0, -30],
                         ids=["the same start", "a later start inside the period"])
def test_restating_an_earlier_period_the_store_holds_is_a_repeat_of_it(
        rec, store, days_before_january):
    """The claim for January to April is over, so it is not live, and the duplicate check
    looked only at live claims: the same restatement made twice stored the earlier period
    twice. A restatement whose period a believed claim of the value already covers says
    nothing new, so it reinforces that claim and stores nothing."""
    earlier, again = restated(rec, store, days_before_january=days_before_january)

    res = rec.apply(again)

    assert res.action == "reinforce" and res.claim.id == earlier.id
    assert res.restated is not None and res.restated.valid_to is None, (
        "the live claim on record, which a link proposed for this write attaches to")
    assert res.claim.observation_count == 2 and res.claim.sources == ["ep_2", "ep_3"]
    assert (res.claim.valid_from, res.claim.valid_to) == (earlier.valid_from,
                                                          earlier.valid_to)
    assert len(store.find_by_value("acme", earlier.value_key)) == 2, (
        "April's claim and the one for January to April, and nothing else")


def test_restating_from_an_even_earlier_start_adds_only_the_period_not_yet_held(
        rec, store):
    """From October, when the store already holds January to April and April onwards: only
    October to January is new, so only that is stored. Ending the new claim where April's
    begins would store January to April a second time."""
    earlier, again = restated(rec, store, days_before_january=90)

    res = rec.apply(again)

    assert res.action == "add"
    assert (res.claim.valid_from, res.claim.valid_to) == (again.valid_from,
                                                          earlier.valid_from)
    assert store.get_claim(earlier.id).observation_count == 1
    assert len(store.find_by_value("acme", earlier.value_key)) == 3


def test_a_restatement_still_covers_a_gap_between_two_stored_periods(rec, store):
    """Tea is stored for February to March and again from June, with nothing between. A
    restatement from January says it held throughout, so it covers every period before
    June that the store does not hold: January to February and March to June. The claim
    for February to March does not reach June, so it does not move the end, and stopping
    at February would drop March to June. Up to #435 the restatement was stored for all of
    January to June, and February to March was then stored twice."""
    june = utcnow() - timedelta(days=90)
    february, march = june - timedelta(days=120), june - timedelta(days=90)
    january = february - timedelta(days=30)
    live = rec.apply(claim("likes", "tea", valid_from=june, recorded_at=june),
                     now=june).claim
    between = claim("likes", "tea", valid_from=february, valid_to=march,
                    recorded_at=february)
    store.put_claim(between)

    res = rec.apply(claim("likes", "tea", valid_from=january, sources=["ep_2"]))

    assert res.action == "add"
    assert pieces_of(res) == [(january, february), (march, june)]
    assert reinforced_by(res) == [between.id]
    assert res.restated is not None and res.restated.id == live.id


def test_a_retired_earlier_period_does_not_make_a_restatement_a_repeat(rec, store):
    """Only a claim the store still believes covers a period. A retired one says the
    record was wrong, so restating that period stores it again."""
    earlier, again = restated(rec, store, days_before_january=0)
    close_out(earlier, utcnow(), None, "retired")
    store.put_claim(earlier)

    res = rec.apply(again)

    assert res.action == "add" and res.claim.id != earlier.id
    assert (res.claim.valid_from, res.claim.valid_to) == (earlier.valid_from,
                                                          earlier.valid_to)


PROJECT_A = Scope("acme", "alice", project="github.com/acme/a")
PROJECT_B = Scope("acme", "alice", project="github.com/acme/b")


def database(scope: Scope, **kw) -> Claim:
    """`api uses_database postgres`: a predicate nobody declared, so each project holds
    its own slot for it."""
    return claim("uses_database", "postgres", subject="api", scope=scope, **kw)


def test_an_earlier_start_ends_where_the_writers_own_claim_begins(rec, store):
    """Project B holds a value from June and restates it from January, so the period from
    January is stored and ends where the next claim of that value begins. Project A holds
    the same value from April, and `value_key` finds A's claim too, because it covers the
    owner and not the project. B cannot read A's claim (`Scope.sees`), so A's start must
    not decide where B's period ends. If it did, B's period would stop in April, and B
    would hold nothing from April until its own claim begins in June."""
    june = utcnow() - timedelta(days=90)
    april, january = june - timedelta(days=60), june - timedelta(days=150)
    own = rec.apply(database(PROJECT_B, valid_from=june, recorded_at=june), now=june).claim
    # Put in directly, so that the test fixes the state it needs rather than depending on
    # how a repeat written in two projects is reconciled.
    elsewhere = database(PROJECT_A, valid_from=april, recorded_at=april)
    store.put_claim(elsewhere)
    assert elsewhere.value_key == own.value_key, "the lookup by value finds both claims"

    res = rec.apply(database(PROJECT_B, valid_from=january, sources=["ep_2"]))

    assert res.action == "add" and res.claim.scope == PROJECT_B
    assert (res.claim.valid_from, res.claim.valid_to) == (january, june)
    assert store.get_claim(elsewhere.id).valid_to is None


def test_an_earlier_start_ends_where_a_user_wide_claim_the_project_reads_begins(
        rec, store):
    """A project reads the user-wide scope above it, so a user-wide claim of the same
    value counts, and the earlier period ends where it begins. A sibling project's claim
    of that value, beginning earlier, still does not count."""
    june = utcnow() - timedelta(days=90)
    april, january = june - timedelta(days=60), june - timedelta(days=150)
    wide = rec.apply(database(SCOPE, valid_from=june, recorded_at=june), now=june).claim
    store.put_claim(database(PROJECT_A, valid_from=april, recorded_at=april))

    res = rec.apply(database(PROJECT_B, valid_from=january, sources=["ep_2"]))

    assert res.action == "add" and res.claim.scope == PROJECT_B
    assert (res.claim.valid_from, res.claim.valid_to) == (january, june)
    kept = store.get_claim(wide.id)
    assert (kept.valid_from, kept.valid_to, kept.observation_count) == (june, None, 1)


def test_an_earlier_period_held_in_another_project_does_not_make_a_restatement_a_repeat(
        rec, store):
    """The check for a period already held follows the same rule: a claim that covers the
    restated period counts only when the writer can see it. Project A's claim for January
    to June is not one project B can read, so B's restatement stores B's own period."""
    june = utcnow() - timedelta(days=90)
    january = june - timedelta(days=150)
    rec.apply(database(PROJECT_B, valid_from=june, recorded_at=june), now=june)
    elsewhere = database(PROJECT_A, valid_from=january, valid_to=june, recorded_at=january)
    store.put_claim(elsewhere)

    res = rec.apply(database(PROJECT_B, valid_from=january, sources=["ep_2"]))

    assert res.action == "add" and res.claim.scope == PROJECT_B
    assert (res.claim.valid_from, res.claim.valid_to) == (january, june)
    assert store.get_claim(elsewhere.id).observation_count == 1


def test_a_value_written_twice_for_a_period_before_the_live_one_is_a_repeat(rec, store):
    """#351. A different value dated before the live one is stored already ended, where
    the live one begins. That claim is not live, so the duplicate check never found it,
    and the same write made twice stored the period twice. The second write reinforces
    the claim on record and stores nothing."""
    march = utcnow() - timedelta(days=60)
    january = march - timedelta(days=60)
    rec.apply(claim("lives_in", "Paris", valid_from=march, recorded_at=march), now=march)
    rome = rec.apply(claim("lives_in", "Rome", valid_from=january, sources=["ep_2"])).claim
    assert rome.valid_to == march, "the premise: Rome is stored as history"

    res = rec.apply(claim("lives_in", "Rome", valid_from=january, sources=["ep_3"]))

    assert res.action == "reinforce" and res.claim.id == rome.id
    assert res.claim.observation_count == 2 and res.claim.sources == ["ep_2", "ep_3"]
    assert len(store.find_by_value("acme", rome.value_key)) == 1


@pytest.mark.parametrize("expires, repeat", [(False, True), (True, False)],
                         ids=["no expiry: a repeat", "an expiry: a claim of its own"])
def test_a_value_written_twice_with_its_end_is_a_repeat_only_where_it_may_reinforce(
        rec, store, expires, repeat):
    """The same rule for a value the caller wrote with its end. A project reads the
    user-wide scope above it, so a user-wide claim for the period is the one on record. A
    repeat that names an expiry reinforces only a claim in exactly its own scope, as in
    step 1, so it is stored as the project's own claim and the user-wide one keeps no
    expiry."""
    june = utcnow() - timedelta(days=90)
    january = june - timedelta(days=150)
    wide = rec.apply(database(SCOPE, valid_from=january, valid_to=june,
                              recorded_at=january), now=january).claim
    expiry = utcnow() + timedelta(days=30) if expires else None

    res = rec.apply(database(PROJECT_B, valid_from=january, valid_to=june,
                             expires_at=expiry, sources=["ep_2"]))

    assert (res.action == "reinforce" and res.claim.id == wide.id) is repeat
    assert (res.action == "add" and res.claim.scope == PROJECT_B) is not repeat
    assert store.get_claim(wide.id).expires_at is None


def test_a_supersession_in_one_project_is_not_cut_off_by_another_projects_claim(store):
    """`supersede()` writes its new claim through the same reconciler, so the same rule
    holds there: the new value's claim ends where a claim of that value the writer can
    see begins, never where another project's does."""
    from memvara import Memvara, NullLLM
    from memvara.embed import HashingEmbedder

    june = utcnow() - timedelta(days=90)
    april, january = june - timedelta(days=60), june - timedelta(days=150)
    mem = Memvara(store=store, embedder=HashingEmbedder(dim=64), llm=NullLLM(),
                  tenant="acme", user="alice")
    mem.remember("api", "uses_database", "postgres", valid_from=june)
    store.put_claim(database(PROJECT_A, valid_from=april, recorded_at=april))
    b = mem.scope(project=PROJECT_B.project)
    mysql = b.remember("api", "uses_database", "mysql",
                       valid_from=january - timedelta(days=30)).added[0]

    receipt = b.supersede(mysql.id, Claim(subject="api", predicate="uses_database",
                                          object="postgres", valid_from=january))

    (new,) = receipt.added
    assert new.scope.project == PROJECT_B.project
    assert (new.valid_from, new.valid_to) == (january, june)
    assert [c.id for c in receipt.closed] == [mysql.id]


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
@pytest.mark.covers("inv:I3")
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


def test_a_retraction_far_less_confident_than_the_value_it_names_ends_nothing(rec, store):
    """#307. A retraction faces the rule a new value faces. This used to be pinned the
    other way, on the ground that every negative came from the fast path at 0.95 or from
    `remember()`; the model tier produces negatives too, at whatever confidence it gives,
    and one at 0.10 ended a fact stated at 1.00. The value stays live, the retraction is
    kept as its tombstone, and the write reports a dispute marked as a retraction's."""
    berlin = rec.apply(claim("lives_in", "Berlin", confidence=1.00)).claim
    res = rec.apply(claim("lives_in", "Berlin", polarity=-1, confidence=0.10))

    assert res.action == "retract" and res.invalidated == []
    assert store.get_claim(berlin.id).state == "live"
    assert res.claim.state == "retired", "the tombstone is stored, and no read returns it"
    (dispute,) = res.disputed
    assert dispute.retraction and dispute.claim_id == berlin.id
    assert (dispute.incumbent_confidence, dispute.candidate_confidence) == (1.00, 0.10)
    assert repr(dispute) == ("<Dispute user lives_in: 'Berlin' 1.00 kept, "
                             "retraction 0.10 did not end it>")


def test_a_disputed_retraction_sent_again_reinforces_its_tombstone(rec, store):
    """A repeat stores no second tombstone, and still reports the dispute: the value is
    still live, and a receipt that reported nothing would read as a retraction that named
    nothing on record."""
    berlin = rec.apply(claim("lives_in", "Berlin", confidence=1.00)).claim
    first = rec.apply(claim("lives_in", "Berlin", polarity=-1, confidence=0.10)).claim

    res = rec.apply(claim("lives_in", "Berlin", polarity=-1, confidence=0.10,
                          sources=["ep_2"]))

    assert res.action == "noop" and res.claim.id == first.id
    assert res.claim.observation_count == 2 and res.claim.sources == ["ep_1", "ep_2"]
    assert [d.claim_id for d in res.disputed] == [berlin.id]
    assert len(store.find_by_value("acme", first.value_key)) == 1


@pytest.mark.parametrize("confidence, ends", [(0.50, True), (0.49, False)])
def test_the_share_a_retraction_needs_is_the_same_floor(rec, store, confidence, ends):
    berlin = rec.apply(claim("lives_in", "Berlin", confidence=1.00)).claim
    res = rec.apply(claim("lives_in", "Berlin", polarity=-1, confidence=confidence))

    assert store.get_claim(berlin.id).state == ("ended" if ends else "live")
    assert bool(res.disputed) is not ends


def test_a_retraction_of_a_whole_slot_ends_only_what_it_outranks(rec, store):
    """A retraction that names no value closes every value in the slot, one by one
    against the rule: it ends the guess and leaves the stated value live."""
    tea = rec.apply(claim("likes", "tea", confidence=1.00)).claim
    coffee = rec.apply(claim("likes", "coffee", confidence=0.10)).claim

    res = rec.apply(claim("likes", "", polarity=-1, confidence=0.30))

    assert [c.id for c in res.invalidated] == [coffee.id]
    assert [(d.claim_id, d.candidate) for d in res.disputed] == [(tea.id, "")]
    assert live_objects(store, tea) == ["tea"]


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


def test_a_retraction_dated_in_the_future_leaves_a_tombstone_that_does_not_end_first(
        rec, store):
    """Like every other closure, the tombstone's world clock never ends before the
    row's own start. When it was closed at the write instead, a retraction dated next
    year stored a row that ended before it began, and `history()` and `why()` showed that
    inverted interval (#275). The belief clock still closes at the write."""
    now = utcnow()
    later = now + timedelta(days=365)
    tea = rec.apply(claim("likes", "tea", valid_from=now), now=now).claim
    res = rec.apply(claim("likes", "tea", polarity=-1, valid_from=later,
                          sources=["ep_2"]), now=now)

    tombstone = store.get_claim(res.claim.id)
    assert tombstone.valid_from == later
    assert tombstone.valid_to == later, "an empty interval, not an inverted one"
    assert tombstone.invalidated_at == now
    assert store.get_claim(tea.id).valid_to == later, "tea stays true until then"


def test_a_retraction_given_a_naive_instant_treats_it_as_utc(rec, store):
    """`as_utc` documents that callers build naive instants by hand, and every other
    closure reads them as UTC. The tombstone's clamp compares the write instant with the
    row's own start, so it must do the same rather than raise."""
    aware = utcnow().replace(microsecond=0)
    naive = aware.replace(tzinfo=None)
    rec.apply(claim("likes", "tea", valid_from=aware - timedelta(days=1)), now=aware)
    res = rec.apply(claim("likes", "tea", polarity=-1, valid_from=aware), now=naive)

    tombstone = store.get_claim(res.claim.id)
    assert tombstone.invalidated_at == aware
    assert tombstone.valid_to == aware


def test_a_backdated_retraction_closes_its_tombstone_where_it_is_recorded(rec, store):
    """#317. A tombstone closes both clocks at the instant its write is recorded, so it is
    believed at no instant. A retraction backdated with `recorded_at` closed them at the
    moment of the call instead, and between the two the tombstone was a live negative
    claim that reads of that past returned."""
    now = utcnow()
    january, february = now - timedelta(days=60), now - timedelta(days=30)
    tea = rec.apply(claim("likes", "tea", valid_from=january, recorded_at=january),
                    now=now).claim
    res = rec.apply(claim("likes", "tea", polarity=-1, valid_from=february,
                          recorded_at=february, sources=["ep_2"]), now=now)

    tombstone = store.get_claim(res.claim.id)
    assert tombstone.recorded_at == tombstone.invalidated_at == tombstone.valid_to == february
    between = february + timedelta(days=10)
    assert not tombstone.is_live(as_of=between)
    assert store.get_claim(tea.id).valid_to == february


def test_a_retraction_dated_in_the_future_sent_again_is_a_repeat(rec, store):
    """#349. The value stays live until the retraction's date, so a repeat still finds it,
    and used to end it again at the same instant, point it at a second tombstone and
    report it ended. Ending it there changes nothing, so the repeat reinforces the
    tombstone on record and reports nothing, as a repeat dated now does. A retraction
    dated sooner does change the value, so it is not a repeat."""
    now = utcnow()
    later, sooner = now + timedelta(days=365), now + timedelta(days=30)
    tea = rec.apply(claim("likes", "tea", valid_from=now), now=now).claim
    first = rec.apply(claim("likes", "tea", polarity=-1, valid_from=later),
                      now=now).claim

    res = rec.apply(claim("likes", "tea", polarity=-1, valid_from=later,
                          sources=["ep_2"]), now=now)

    assert res.action == "noop" and res.claim.id == first.id and res.invalidated == []
    assert store.get_claim(tea.id).invalidated_by == first.id
    assert [c.id for c in store.iter_claims(None, True) if c.polarity < 0] == [first.id]

    res = rec.apply(claim("likes", "tea", polarity=-1, valid_from=sooner), now=now)

    assert res.action == "retract" and [c.id for c in res.invalidated] == [tea.id]
    assert store.get_claim(tea.id).valid_to == sooner


def test_a_future_dated_retraction_repeated_as_a_correction_retires_the_value(rec, store):
    """Retiring the value changes it even where its world clock already stops, so a
    retraction with `close="retired"` is not taken for a repeat of an ending."""
    now = utcnow()
    later = now + timedelta(days=365)
    tea = rec.apply(claim("likes", "tea", valid_from=now), now=now).claim
    rec.apply(claim("likes", "tea", polarity=-1, valid_from=later), now=now)

    res = rec.apply(claim("likes", "tea", polarity=-1, valid_from=later), now=now,
                    close="retired")

    assert res.action == "retract" and [c.id for c in res.invalidated] == [tea.id]
    assert store.get_claim(tea.id).state == "retired"


def test_a_retraction_where_the_value_already_ends_is_recorded_and_ends_nothing(
        rec, store):
    """The value already stops being true where the retraction says, so ending it would
    change nothing and it is not reported. The retraction still named a value on record,
    so its tombstone is kept."""
    now = utcnow()
    end = now + timedelta(days=30)
    tea = rec.apply(claim("likes", "tea", valid_from=now, valid_to=end), now=now).claim

    res = rec.apply(claim("likes", "tea", polarity=-1, valid_from=end), now=now)

    assert res.action == "retract" and res.invalidated == [] and res.disputed == []
    assert res.claim.state == "retired"
    assert store.get_claim(tea.id).invalidated_by is None


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


@pytest.mark.parametrize("expiry, folded", [
    (timedelta(0), False),              # expires at the repeat's own instant: already gone
    (timedelta(microseconds=1), True),  # expires just after it: still the one on record
])
def test_a_retraction_repeated_at_its_tombstones_expiry_is_judged_at_that_instant(
        rec, store, expiry, folded):
    """A tombstone counts as expired at its `expires_at` itself, as every read counts it
    (#284). A repeat at that instant writes a tombstone of its own; a repeat one
    microsecond earlier is folded into the tombstone on record."""
    t0 = utcnow() - timedelta(days=2)
    t1 = t0 + timedelta(days=1)
    rec.apply(claim("works_at", "Acme", valid_from=t0, recorded_at=t0), now=t0)
    first = rec.apply(claim("works_at", "Acme", polarity=-1, valid_from=t0, recorded_at=t0,
                            expires_at=t1 + expiry), now=t0).claim
    res = rec.apply(claim("works_at", "Acme", polarity=-1, valid_from=t1, recorded_at=t1),
                    now=t1)
    tombstones = [c for c in store.iter_claims(None, True) if c.polarity < 0]
    if folded:
        assert res.action == "noop" and [c.id for c in tombstones] == [first.id]
    else:
        assert res.action == "retract" and len(tombstones) == 2


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


def test_a_new_session_leaves_a_sibling_sessions_value_alone(rec, store):
    # Agent and session are outside the fact key, so two sessions' values share a slot.
    # A write in s2 used to end the value s1 held, which s2 cannot read, and hand it back
    # in s2's receipt. A new value now ends only claims at exactly its own scope
    # (`Reconciler._at_scope`).
    old = rec.apply(claim("lives_in", "Berlin",
                          scope=Scope("acme", "alice", "asst", "s1"))).claim
    res = rec.apply(claim("lives_in", "Lisbon",
                          scope=Scope("acme", "alice", "asst", "s2")))
    assert res.invalidated == []
    assert store.get_claim(old.id).is_live()


def test_a_new_session_leaves_the_value_its_agent_holds_live(rec, store):
    # A session's value is a local value (#266). Up to 0.17.0, learning "I moved to
    # Lisbon" in a fresh session ended the city held for the agent, for every session of
    # that agent. Now both stay live, and the session reads its own value in place of the
    # agent's (`memvara.retrieve.shadow`).
    old = rec.apply(claim("lives_in", "Berlin", scope=Scope("acme", "alice", "asst"))).claim
    res = rec.apply(claim("lives_in", "Lisbon",
                          scope=Scope("acme", "alice", "asst", "s2")))
    assert res.action == "add"
    assert res.invalidated == []
    assert store.get_claim(old.id).is_live()


def test_a_new_value_still_ends_the_value_at_its_own_scope(rec, store):
    old = rec.apply(claim("lives_in", "Berlin",
                          scope=Scope("acme", "alice", "asst", "s2"))).claim
    res = rec.apply(claim("lives_in", "Lisbon",
                          scope=Scope("acme", "alice", "asst", "s2")))
    assert [c.id for c in res.invalidated] == [old.id]
    assert store.get_claim(old.id).state == "ended"


def test_a_retraction_in_a_session_still_ends_the_value_it_reads(rec, store):
    # A retraction closes the claims its writer can see, so it still ends the value the
    # session reads from its agent's level.
    old = rec.apply(claim("lives_in", "Berlin", scope=Scope("acme", "alice", "asst"))).claim
    res = rec.apply(claim("lives_in", "Berlin", polarity=-1,
                          scope=Scope("acme", "alice", "asst", "s2")))
    assert res.action == "retract"
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


# --- temporal ordering: bounds, not scalars ----------------------------------
#
# `valid_from` now carries the earliest boundary the source stated, which can be far
# earlier than the sentence that stated it. Ordering on the scalar alone would then let a
# vague boundary outrank a precise one purely because normalising it moved it backwards.

from datetime import datetime, timezone  # noqa: E402

UTC = timezone.utc


def at(y, m, d, hh=12):
    return datetime(y, m, d, hh, tzinfo=UTC)


def test_a_backfilled_event_does_not_retire_a_later_one(rec, store):
    """The property that already held and must keep holding: supersession runs along
    valid time, so a fact stated today but true from 2019 is history, not news."""
    rec.apply(claim("lives_in", "Berlin", valid_from=at(2026, 1, 1)), now=at(2026, 1, 1))
    rec.apply(claim("lives_in", "Lisbon", valid_from=at(2019, 1, 1)), now=at(2026, 9, 3))
    assert live_objects(store, claim("lives_in", "x")) == ["Berlin"]


def test_an_equal_valid_from_still_leaves_the_stored_claim_supersedable(rec, store):
    """The strict `>` at the old comparison put an equal boundary in `older`. The
    bounds rule has to agree, or every same-instant rewrite silently stops superseding."""
    same = at(2026, 5, 1)
    rec.apply(claim("lives_in", "Berlin", valid_from=same), now=same)
    rec.apply(claim("lives_in", "Lisbon", valid_from=same), now=same)
    assert live_objects(store, claim("lives_in", "x")) == ["Lisbon"]


def test_instant_only_ordering_is_what_it_always_was(rec, store):
    """No precision anywhere means the rule must reduce exactly to the scalar compare."""
    rec.apply(claim("lives_in", "Berlin", valid_from=at(2026, 1, 1)), now=at(2026, 1, 1))
    rec.apply(claim("lives_in", "Lisbon", valid_from=at(2026, 6, 1)), now=at(2026, 6, 1))
    assert live_objects(store, claim("lives_in", "x")) == ["Lisbon"]


def test_a_coarse_boundary_containing_a_precise_one_falls_back_to_belief(rec, store):
    """The case the whole rule exists for.

        2025-12-01  "I live in London."            -> instant
        2026-09-03  "I moved to Lisbon last year." -> the year 2025

    2025-12-01 lies inside 2025, so the two boundaries cannot be ordered. Precedence
    falls to `recorded_at`, and the later statement wins — which is what a person reading
    those two sentences concludes. Ordering on the scalar alone leaves London standing.
    """
    rec.apply(claim("lives_in", "London", valid_from=at(2025, 12, 1)),
                  now=at(2025, 12, 1))
    rec.apply(claim("lives_in", "Lisbon", valid_from=at(2025, 1, 1),
                        temporal_precision="year"), now=at(2026, 9, 3))
    assert live_objects(store, claim("lives_in", "x")) == ["Lisbon"]


def test_two_coarse_boundaries_that_do_not_overlap_still_order(rec, store):
    """Precision does not mean "give up". Year 2024 and year 2025 are disjoint, so the
    later one supersedes on valid time exactly as two instants would."""
    rec.apply(claim("lives_in", "Lisbon", valid_from=at(2025, 1, 1),
                        temporal_precision="year"), now=at(2026, 1, 1))
    rec.apply(claim("lives_in", "London", valid_from=at(2024, 1, 1),
                        temporal_precision="year"), now=at(2026, 9, 3))
    # London is confidently *earlier*, so it is history and does not displace Lisbon,
    # even though it was said later.
    assert live_objects(store, claim("lives_in", "x")) == ["Lisbon"]


@pytest.mark.parametrize("start, precision", [
    (at(2025, 6, 1), "month"),     # "in June 2025"
    (at(2025, 1, 1), "year"),      # "in 2025"
])
def test_a_repeat_dated_by_a_month_or_a_year_covers_a_period_that_begins_inside_it(
        rec, store, start, precision):
    """Rome is stored from 15 June 2025 until Paris begins in March 2026. A repeat dated
    "June 2025" or "2025" names a period that contains 15 June, so the stored claim cannot
    be said to begin after it (`_bounds`), and the repeat reinforces it (#351). Compared
    as instants, the stored claim begins later, and the repeat would store the period a
    second time."""
    rec.apply(claim("lives_in", "Paris", valid_from=at(2026, 3, 1)), now=at(2026, 3, 1))
    rome = rec.apply(claim("lives_in", "Rome", valid_from=at(2025, 6, 15)),
                     now=at(2026, 4, 1)).claim
    assert rome.valid_to == at(2026, 3, 1)

    res = rec.apply(claim("lives_in", "Rome", valid_from=start,
                          temporal_precision=precision), now=at(2026, 5, 1))

    assert res.action == "reinforce" and res.claim.id == rome.id


def test_eligibility_is_untouched_by_temporal_ordering(rec, store):
    """Ordering decides *which* competing claim wins, never *whether* two claims
    compete. A many-valued predicate collects no victims at all, so two events with
    overlapping boundaries both survive."""
    rec.apply(claim("likes", "London", valid_from=at(2025, 1, 1),
                        temporal_precision="year"), now=at(2025, 6, 1))
    rec.apply(claim("likes", "Lisbon", valid_from=at(2025, 12, 1)), now=at(2026, 9, 3))
    assert live_objects(store, claim("likes", "x")) == ["Lisbon", "London"]


# --- a write that overlaps part of a stored period (#435) ---------------------

#: The instant every write in this section is reconciled at. Every period below is over
#: by then unless a test says otherwise.
LATER = at(2026, 9, 1)


def stored(rec, predicate: str, obj: str, start, end=None, **kw) -> Claim:
    """Write a value for a period and return the claim the write stored."""
    res = rec.apply(claim(predicate, obj, valid_from=start, valid_to=end, **kw), now=LATER)
    assert res.action == "add", "the premise: the period is stored as a claim of its own"
    return res.claim


def pieces_of(res) -> list[tuple]:
    """The periods a write stored, in order: `res.claim`, then each further piece."""
    parts = [res.claim] + [p.claim for p in res.also if p.action == "add"]
    return [(p.valid_from, p.valid_to) for p in parts]


def reinforced_by(res) -> list[str]:
    """The ids of the stored claims that a write which also stored a piece reinforced."""
    return [p.claim.id for p in res.also if p.action == "reinforce"]


def values_at(store, c: Claim, when) -> list[str]:
    return sorted(x.object for x in store.competing_claims(
        c.scope.tenant, c.fact_key, valid_at=when, known_at=LATER))


@pytest.mark.covers("inv:MM11")
def test_a_write_that_overlaps_part_of_a_stored_period_stores_only_the_rest(rec, store):
    """#435, the issue's example. Rome is stored for January to March, and Rome is then
    written for February to April. February to March is already held, so the write
    reinforces the stored claim and stores Rome only for March to April. Stored whole, the
    second claim overlapped the first, and a read of 15 February returned Rome twice."""
    first = stored(rec, "lives_in", "Rome", at(2026, 1, 1), at(2026, 3, 1))

    res = rec.apply(claim("lives_in", "Rome", valid_from=at(2026, 2, 1),
                          valid_to=at(2026, 4, 1), sources=["ep_2"]), now=LATER)

    assert res.action == "add" and res.invalidated == []
    assert pieces_of(res) == [(at(2026, 3, 1), at(2026, 4, 1))]
    assert res.claim.sources == ["ep_2"]
    assert reinforced_by(res) == [first.id]
    kept = store.get_claim(first.id)
    assert (kept.valid_from, kept.valid_to) == (at(2026, 1, 1), at(2026, 3, 1)), (
        "no stored claim is rewritten")
    assert kept.observation_count == 2 and kept.sources == ["ep_1", "ep_2"]
    assert values_at(store, first, at(2026, 2, 15)) == ["Rome"]
    assert values_at(store, first, at(2026, 3, 15)) == ["Rome"]
    assert len(store.find_by_value("acme", first.value_key)) == 2


@pytest.mark.covers("inv:MM11")
def test_a_write_that_spans_both_sides_of_a_stored_period_is_stored_in_two_pieces(
        rec, store):
    """Rome is stored for March to May, and Rome is then written for January to July. The
    write holds two periods the store does not: January to March and May to July. Each is
    stored as a claim of its own, citing the write's sources, and the stored claim is
    reinforced for the overlap and left as it was."""
    held = stored(rec, "lives_in", "Rome", at(2026, 3, 1), at(2026, 5, 1))

    res = rec.apply(claim("lives_in", "Rome", valid_from=at(2026, 1, 1),
                          valid_to=at(2026, 7, 1), sources=["ep_2"]), now=LATER)

    assert res.action == "add"
    assert pieces_of(res) == [(at(2026, 1, 1), at(2026, 3, 1)),
                              (at(2026, 5, 1), at(2026, 7, 1))]
    parts = [res.claim] + [p.claim for p in res.also if p.action == "add"]
    assert len({p.id for p in parts}) == 2
    assert all(p.sources == ["ep_2"] for p in parts)
    assert all(store.get_claim(p.id) is not None for p in parts)
    assert reinforced_by(res) == [held.id]
    kept = store.get_claim(held.id)
    assert (kept.valid_from, kept.valid_to, kept.observation_count) == (
        at(2026, 3, 1), at(2026, 5, 1), 2)
    for month in (2, 4, 6):
        assert values_at(store, held, at(2026, month, 15)) == ["Rome"]


def test_a_write_over_several_stored_periods_is_stored_in_each_gap(rec, store):
    """More stored claims can mean more gaps: February to March and May to June are
    stored, so a write for January to July is stored for the three periods between."""
    a = stored(rec, "likes", "tea", at(2026, 2, 1), at(2026, 3, 1))
    b = stored(rec, "likes", "tea", at(2026, 5, 1), at(2026, 6, 1))

    res = rec.apply(claim("likes", "tea", valid_from=at(2026, 1, 1),
                          valid_to=at(2026, 7, 1), sources=["ep_2"]), now=LATER)

    assert pieces_of(res) == [(at(2026, 1, 1), at(2026, 2, 1)),
                              (at(2026, 3, 1), at(2026, 5, 1)),
                              (at(2026, 6, 1), at(2026, 7, 1))]
    assert reinforced_by(res) == [a.id, b.id]


def test_an_open_ended_write_over_a_closed_stored_period_begins_where_it_ends(rec, store):
    """Rome is stored for January to March, and Rome is then written from February with
    no end. The piece the store does not hold begins where the stored claim ends, and it
    is live now."""
    held = stored(rec, "lives_in", "Rome", at(2026, 1, 1), at(2026, 3, 1))

    res = rec.apply(claim("lives_in", "Rome", valid_from=at(2026, 2, 1),
                          sources=["ep_2"]), now=LATER)

    assert res.action == "add"
    assert pieces_of(res) == [(at(2026, 3, 1), None)]
    assert res.claim.is_live(LATER)
    assert reinforced_by(res) == [held.id]
    assert values_at(store, held, at(2026, 2, 15)) == ["Rome"]


def test_a_write_that_outlasts_a_live_claim_stores_the_period_after_it_ends(rec, store):
    """Tea is live now and stored to end in December. Tea written from now with no end
    holds a period the store does not: from December on. The live claim is reinforced and
    that period is stored. Before #435 the write was a repeat of the live claim, and the
    period after December was dropped."""
    held = stored(rec, "likes", "tea", at(2026, 1, 1), at(2026, 12, 1))

    res = rec.apply(claim("likes", "tea", valid_from=LATER, sources=["ep_2"]), now=LATER)

    assert res.action == "add"
    assert pieces_of(res) == [(at(2026, 12, 1), None)]
    assert reinforced_by(res) == [held.id]


def test_a_piece_beside_its_own_live_value_is_not_reported_as_an_accumulation(rec, store):
    """`collects` is declared by nobody, so a new value beside a live one is reported as
    an accumulation. The piece a write stores after its own live claim ends is not a
    second answer, so it is not reported, even with another value live in the slot."""
    stored(rec, "collects", "stamps", at(2026, 1, 1), at(2026, 12, 1))
    rec.apply(claim("collects", "vinyl", valid_from=at(2026, 1, 1)), now=LATER)

    res = rec.apply(claim("collects", "stamps", valid_from=LATER, sources=["ep_2"]),
                    now=LATER)

    assert res.action == "add" and pieces_of(res) == [(at(2026, 12, 1), None)]
    assert res.accumulated is None


def test_a_write_inside_a_stored_period_is_still_a_repeat(rec, store):
    """Rome is stored for January to June. February to April holds nothing new, so the
    write reinforces the stored claim and stores nothing (#351)."""
    held = stored(rec, "lives_in", "Rome", at(2026, 1, 1), at(2026, 6, 1))

    res = rec.apply(claim("lives_in", "Rome", valid_from=at(2026, 2, 1),
                          valid_to=at(2026, 4, 1), sources=["ep_2"]), now=LATER)

    assert res.action == "reinforce" and res.claim.id == held.id and res.also == []
    assert len(store.find_by_value("acme", held.value_key)) == 1


def test_a_write_that_two_stored_periods_cover_together_is_a_repeat_of_both(rec, store):
    """January to March and March to June are stored. No one claim holds February to
    April, but the two together do, so the write stores nothing and reinforces both."""
    a = stored(rec, "likes", "tea", at(2026, 1, 1), at(2026, 3, 1))
    b = stored(rec, "likes", "tea", at(2026, 3, 1), at(2026, 6, 1))

    res = rec.apply(claim("likes", "tea", valid_from=at(2026, 2, 1),
                          valid_to=at(2026, 4, 1), sources=["ep_2"]), now=LATER)

    assert res.action == "reinforce" and res.claim.id == a.id
    assert [(p.action, p.claim.id) for p in res.also] == [("reinforce", b.id)]
    assert [store.get_claim(c.id).observation_count for c in (a, b)] == [2, 2]
    assert len(store.find_by_value("acme", a.value_key)) == 2


def test_a_restatement_before_the_live_claim_that_two_stored_periods_hold_is_a_repeat(
        rec, store):
    """#283 with #435. Tea is live from June and stored for January to March and March to
    May. Tea for February to April starts before the live claim, so it is a restatement
    of an earlier period, and the caller's end, April, bounds that period. No one claim
    holds February to April, but the two stored ones do together, so the write stores
    nothing, reinforces both, and still names the live claim it restated."""
    live = stored(rec, "likes", "tea", at(2026, 6, 1))
    a = stored(rec, "likes", "tea", at(2026, 1, 1), at(2026, 3, 1))
    b = stored(rec, "likes", "tea", at(2026, 3, 1), at(2026, 5, 1))

    res = rec.apply(claim("likes", "tea", valid_from=at(2026, 2, 1),
                          valid_to=at(2026, 4, 1), sources=["ep_2"]), now=LATER)

    assert res.action == "reinforce" and res.claim.id == a.id
    assert [(p.action, p.claim.id) for p in res.also] == [("reinforce", b.id)]
    assert res.restated is not None and res.restated.id == live.id
    assert len(store.find_by_value("acme", a.value_key)) == 3


def test_a_write_three_stored_periods_hold_is_a_repeat_of_each(rec, store):
    """The stored claims can overlap each other, as claims stored directly or before #435
    can. January to March, March to May and 20 March to June hold February to April
    between them, the last two overlapping, so the write reinforces all three."""
    periods = [(at(2026, 1, 1), at(2026, 3, 1)), (at(2026, 3, 1), at(2026, 5, 1)),
               (at(2026, 3, 20), at(2026, 6, 1))]
    held = [claim("likes", "tea", valid_from=start, valid_to=end, recorded_at=start)
            for start, end in periods]
    for c in held:
        store.put_claim(c)

    res = rec.apply(claim("likes", "tea", valid_from=at(2026, 2, 1),
                          valid_to=at(2026, 4, 1), sources=["ep_2"]), now=LATER)

    assert res.action == "reinforce"
    assert [res.claim.id] + [p.claim.id for p in res.also] == [c.id for c in held]
    assert len(store.find_by_value("acme", held[0].value_key)) == 3


def test_a_write_of_no_length_is_stored_as_it_is(rec, store):
    """`remember()` refuses a period that ends where it begins, but `Reconciler.apply`
    takes whatever a caller builds. Such a period overlaps nothing, so it is stored as it
    was before #435 rather than being cut into pieces."""
    instant = at(2026, 2, 1)

    res = rec.apply(claim("likes", "tea", valid_from=instant, valid_to=instant),
                    now=LATER)

    assert res.action == "add" and res.also == []
    assert pieces_of(res) == [(instant, instant)]


def test_a_write_that_overlaps_no_stored_period_is_stored_whole(rec, store):
    held = stored(rec, "lives_in", "Rome", at(2026, 1, 1), at(2026, 3, 1))

    res = rec.apply(claim("lives_in", "Rome", valid_from=at(2026, 4, 1),
                          valid_to=at(2026, 5, 1), sources=["ep_2"]), now=LATER)

    assert res.action == "add" and res.also == []
    assert pieces_of(res) == [(at(2026, 4, 1), at(2026, 5, 1))]
    assert store.get_claim(held.id).observation_count == 1


def test_a_piece_ends_where_a_later_different_value_begins(rec, store):
    """Rome is stored for January to March, and Paris is live from April. Rome written
    from February ends where Paris begins, as a whole write would, so the piece is March
    to April. Paris begins after the write, so the write leaves it alone."""
    held = stored(rec, "lives_in", "Rome", at(2026, 1, 1), at(2026, 3, 1))
    paris = stored(rec, "lives_in", "Paris", at(2026, 4, 1))

    res = rec.apply(claim("lives_in", "Rome", valid_from=at(2026, 2, 1),
                          sources=["ep_2"]), now=LATER)

    assert res.action == "add" and res.invalidated == []
    assert pieces_of(res) == [(at(2026, 3, 1), at(2026, 4, 1))]
    assert reinforced_by(res) == [held.id]
    kept = store.get_claim(paris.id)
    assert kept.valid_to is None and kept.invalidated_at is None


def test_a_piece_supersedes_what_the_whole_write_would_and_at_the_same_instant(
        rec, store):
    """Rome is stored for January to March, and Paris is live from 15 January. Rome
    written from February, in a slot that holds one value, ends Paris where the write
    begins: February, not March where the stored piece begins. Rome holds from February,
    through the claim on record until March and the new piece after it, so ending Paris
    in March would leave two values true from February to March."""
    held = stored(rec, "lives_in", "Rome", at(2026, 1, 1), at(2026, 3, 1))
    paris = stored(rec, "lives_in", "Paris", at(2026, 1, 15))

    res = rec.apply(claim("lives_in", "Rome", valid_from=at(2026, 2, 1),
                          sources=["ep_2"]), now=LATER)

    assert res.action == "supersede"
    assert pieces_of(res) == [(at(2026, 3, 1), None)]
    assert [c.id for c in res.invalidated] == [paris.id]
    ended = store.get_claim(paris.id)
    assert ended.valid_to == at(2026, 2, 1) and ended.invalidated_at is None
    assert ended.invalidated_by == res.claim.id
    assert reinforced_by(res) == [held.id]
    for month in (2, 4):
        assert values_at(store, held, at(2026, month, 15)) == ["Rome"]


def test_a_write_the_store_already_holds_ends_nothing(rec, store):
    """A write whose whole period stored claims already hold is a repeat, and a repeat
    ends nothing, whether one claim holds the period or two do together (#351). Paris,
    live from 15 January, stays live."""
    stored(rec, "lives_in", "Rome", at(2026, 1, 1), at(2026, 3, 1))
    stored(rec, "lives_in", "Rome", at(2026, 3, 1), at(2026, 6, 1))
    paris = stored(rec, "lives_in", "Paris", at(2026, 1, 15))

    res = rec.apply(claim("lives_in", "Rome", valid_from=at(2026, 2, 1),
                          valid_to=at(2026, 4, 1), sources=["ep_2"]), now=LATER)

    assert res.action == "reinforce" and res.invalidated == []
    kept = store.get_claim(paris.id)
    assert kept.valid_to is None and kept.invalidated_at is None


@pytest.mark.parametrize("precision, start", [("month", at(2025, 6, 1)),
                                              ("year", at(2025, 1, 1))])
def test_a_piece_after_a_period_stated_by_month_or_year_begins_at_an_exact_instant(
        rec, store, precision, start):
    """Tea is stored from 15 June 2025 to March 2026. Tea "since June 2025", or "since
    2025", until April 2026 names a start the stored claim cannot be said to begin after
    (`_bounds`), so only March to April is new. The piece begins at the stored claim's
    exact end, so it carries no month or year precision: that precision described the
    write's own start."""
    held = stored(rec, "likes", "tea", at(2025, 6, 15), at(2026, 3, 1))

    res = rec.apply(claim("likes", "tea", valid_from=start, temporal_precision=precision,
                          valid_to=at(2026, 4, 1), sources=["ep_2"]), now=LATER)

    assert pieces_of(res) == [(at(2026, 3, 1), at(2026, 4, 1))]
    assert res.claim.temporal_precision is None
    assert reinforced_by(res) == [held.id]


def test_the_first_piece_keeps_the_precision_of_the_start_it_keeps(rec, store):
    held = stored(rec, "likes", "tea", at(2026, 3, 1), at(2026, 4, 1))

    res = rec.apply(claim("likes", "tea", valid_from=at(2026, 1, 1),
                          temporal_precision="month", valid_to=at(2026, 6, 1)),
                    now=LATER)

    assert pieces_of(res) == [(at(2026, 1, 1), at(2026, 3, 1)),
                              (at(2026, 4, 1), at(2026, 6, 1))]
    first, second = [res.claim] + [p.claim for p in res.also if p.action == "add"]
    assert (first.temporal_precision, second.temporal_precision) == ("month", None)
    assert reinforced_by(res) == [held.id]


def test_only_a_stored_period_the_writer_can_see_covers_part_of_a_write(rec, store):
    """Project A holds the value for January to March, and project B cannot read it, so
    B's write for February to April is stored whole. A user-wide claim is one B reads,
    so it does count."""
    elsewhere = database(PROJECT_A, valid_from=at(2026, 1, 1), valid_to=at(2026, 3, 1),
                         recorded_at=at(2026, 1, 1))
    store.put_claim(elsewhere)

    res = rec.apply(database(PROJECT_B, valid_from=at(2026, 2, 1),
                             valid_to=at(2026, 4, 1), sources=["ep_2"]), now=LATER)

    assert res.action == "add" and res.also == []
    assert pieces_of(res) == [(at(2026, 2, 1), at(2026, 4, 1))]
    assert store.get_claim(elsewhere.id).observation_count == 1

    wide = database(SCOPE, valid_from=at(2026, 5, 1), valid_to=at(2026, 7, 1),
                    recorded_at=at(2026, 5, 1))
    store.put_claim(wide)
    res = rec.apply(database(PROJECT_B, valid_from=at(2026, 6, 1),
                             valid_to=at(2026, 8, 1), sources=["ep_3"]), now=LATER)

    assert pieces_of(res) == [(at(2026, 7, 1), at(2026, 8, 1))]
    assert reinforced_by(res) == [wide.id]


@pytest.mark.parametrize("scope, counts", [(PROJECT_B, True), (SCOPE, False)],
                         ids=["stored in the writer's scope", "stored user-wide"])
def test_a_partial_write_that_names_an_expiry_counts_only_its_own_scope(
        rec, store, scope, counts):
    """A caller's repeat that names an expiry reinforces only a claim in exactly its own
    scope and moves the expiry onto it, so a fact the caller asked to have erased is
    erased. A write that overlaps part of such a claim does the same for the overlap, and
    its pieces carry the expiry. A user-wide claim is not in the project's own scope, so it does not count,
    the write is stored whole, and the user-wide claim keeps no expiry."""
    held = database(scope, valid_from=at(2026, 1, 1), valid_to=at(2026, 3, 1),
                    recorded_at=at(2026, 1, 1))
    store.put_claim(held)
    expiry = utcnow() + timedelta(days=30)

    res = rec.apply(database(PROJECT_B, valid_from=at(2026, 2, 1), valid_to=at(2026, 4, 1),
                             expires_at=expiry, derivation=Derivation.USER,
                             sources=["ep_2"]), now=LATER)

    start = at(2026, 3, 1) if counts else at(2026, 2, 1)
    assert pieces_of(res) == [(start, at(2026, 4, 1))]
    assert res.claim.expires_at == expiry
    assert reinforced_by(res) == ([held.id] if counts else [])
    assert store.get_claim(held.id).expires_at == (expiry if counts else None)


# --- a backdated write changes belief only from the time of the call (#436) ---

@pytest.mark.covers("inv:MM10")
def test_a_backdated_retraction_that_retires_does_so_at_the_time_of_the_call(rec, store):
    """#436. A retraction written with a past `recorded_at` and `close="retired"` retires
    the value at the time of the call, not at its `recorded_at`. The belief clock records
    what the store believed and when, and it is never rewritten, so a read of belief
    between the two instants still returns the value. Only the tombstone's own interval
    sits at its `recorded_at` (#317)."""
    now = utcnow()
    january, february = now - timedelta(days=60), now - timedelta(days=30)
    tea = rec.apply(claim("likes", "tea", valid_from=january, recorded_at=january),
                    now=now).claim

    res = rec.apply(claim("likes", "tea", polarity=-1, valid_from=february,
                          recorded_at=february, sources=["ep_2"]),
                    now=now, close="retired")

    assert [c.id for c in res.invalidated] == [tea.id]
    kept = store.get_claim(tea.id)
    assert kept.invalidated_at == now and kept.valid_to is None
    between = february + timedelta(days=10)
    assert [c.id for c in store.competing_claims(
        "acme", tea.fact_key, valid_at=between, known_at=between)] == [tea.id]
    assert store.competing_claims("acme", tea.fact_key, valid_at=between,
                                  known_at=now) == []
    assert store.get_claim(res.claim.id).invalidated_at == february


@pytest.mark.covers("inv:MM10")
def test_a_backdated_supersession_that_retires_does_so_at_the_time_of_the_call(
        rec, store):
    """#436. The same rule for a new value in a slot that holds one: Lisbon, written with
    a past `recorded_at` and `close="retired"`, retires Berlin at the time of the call.
    Before the call the store believed Berlin, and a read of that past still says so."""
    now = utcnow()
    long_ago, moved = now - timedelta(days=800), now - timedelta(days=30)
    berlin = rec.apply(claim("lives_in", "Berlin", valid_from=long_ago,
                             recorded_at=long_ago), now=long_ago).claim

    res = rec.apply(claim("lives_in", "Lisbon", valid_from=moved, recorded_at=moved,
                          sources=["ep_2"]), now=now, close="retired")

    assert res.action == "supersede" and [c.id for c in res.invalidated] == [berlin.id]
    kept = store.get_claim(berlin.id)
    assert kept.invalidated_at == now and kept.valid_to is None
    between = moved + timedelta(days=10)
    assert sorted(c.object for c in store.competing_claims(
        "acme", berlin.fact_key, valid_at=between, known_at=between)) == ["Berlin",
                                                                          "Lisbon"]


# --- quantity is not identity ------------------------------------------------

def test_quantity_does_not_open_a_second_slot(rec, store):
    """`amount` and `unit` describe the observation, not the fact. If they ever leak
    into `fact_key` or `value_key`, 70kg and 71kg become unrelated facts and a weight
    log accumulates forever instead of updating."""
    first = claim("prefers", "70", amount=70.0, unit="kilogram")
    second = claim("prefers", "70", amount=71.0, unit="kilogram")
    assert first.fact_key == second.fact_key
    assert first.value_key == second.value_key


def test_changing_the_triple_still_changes_identity(rec, store):
    """The converse, because the failure mode is someone adding the new fields to a
    serialisation tuple the identity functions read — which would break both directions
    at once and this asserts the half that keeps working."""
    base = claim("prefers", "tea")
    assert base.fact_key != claim("prefers", "tea", subject="bob").fact_key
    assert base.fact_key != claim("likes", "tea").fact_key
    assert base.value_key != claim("prefers", "coffee").value_key


def test_the_reverse_order_still_lets_the_later_statement_win(rec, store):
    """The converse, so the rule above is not just "precision always wins": stating a
    boundary first and an unqualified fact second must leave the second standing."""
    rec.apply(claim("lives_in", "Lisbon", valid_from=at(2026, 8, 1),
                    temporal_precision="month"), now=at(2026, 8, 15))
    rec.apply(claim("lives_in", "Berlin", valid_from=at(2026, 9, 3)), now=at(2026, 9, 3))
    assert live_objects(store, claim("lives_in", "x"),
                        as_of=at(2026, 9, 3)) == ["Berlin"]


def test_a_stated_boundary_never_retires_a_later_undated_fact(rec, store):
    """The invariant this module states at `_victims`, tested against the path that
    broke it: "a fact backfilled today but true from 2019 must not retire the 2026 fact
    that replaced it".

    An extracted "I moved to Lisbon in 2019" is that fact, arriving through extraction
    rather than through `remember(valid_from=...)`. A first attempt at the ordering rule
    let it retire a Berlin claim stated today, because a no-precision boundary was
    treated as never confidently after a stated one, whatever the distance between them.
    """
    rec.apply(claim("lives_in", "Berlin", valid_from=at(2026, 9, 3)), now=at(2026, 9, 3))
    rec.apply(claim("lives_in", "Lisbon", valid_from=at(2019, 1, 1),
                    temporal_precision="year"), now=at(2026, 9, 4))
    assert live_objects(store, claim("lives_in", "x"),
                        as_of=at(2026, 9, 4)) == ["Berlin"]


@pytest.mark.parametrize("precision, inside, outside", [
    ("day", at(2026, 5, 1, 23), at(2026, 5, 2, 1)),
    ("week", at(2026, 5, 6, 12), at(2026, 5, 12, 1)),
    ("season", at(2026, 6, 15, 12), at(2026, 9, 2, 1)),
])
def test_every_precision_covers_the_span_it_names(rec, store, precision, inside, outside):
    """Each precision denotes an interval, and ordering is decided by whether the two
    intervals overlap. A boundary inside the span cannot be ordered against it and falls
    to belief; one past the span's end is confidently later and stays live.

    Covers the `day`, `week` and `season` arms of `_bounds`, which the month and year
    cases above do not reach.
    """
    span_start = {"day": at(2026, 5, 1, 0), "week": at(2026, 5, 4, 0),
                  "season": at(2026, 6, 1, 0)}[precision]

    # A precise claim inside the span: incomparable, so the later statement wins.
    rec.apply(claim("lives_in", "Inside", valid_from=inside), now=inside)
    rec.apply(claim("lives_in", "Spanning", valid_from=span_start,
                    temporal_precision=precision), now=at(2026, 9, 30))
    assert live_objects(store, claim("lives_in", "x"),
                        as_of=at(2026, 9, 30)) == ["Spanning"]

    # A precise claim past the span's end: confidently later, so it survives a
    # subsequent statement about the span it already postdates.
    other = SQLiteStore(":memory:")
    rec2 = Reconciler(other, PredicateRegistry())
    rec2.apply(claim("lives_in", "After", valid_from=outside), now=outside)
    rec2.apply(claim("lives_in", "Spanning", valid_from=span_start,
                     temporal_precision=precision), now=at(2026, 9, 30))
    assert live_objects(other, claim("lives_in", "x"),
                        as_of=at(2026, 9, 30)) == ["After"]


def test_a_global_predicate_is_written_with_no_project():
    """One line in `_canonicalize`, and two consequences that are both wanted.

    A fact the vocabulary calls global is not project-relative, so it is written at the
    level above one. Its slot then has no project, which is what makes saying it in a
    second repository retire the first value rather than duplicate it; and it sits at user
    level, which visibility widens up into, so it is readable from inside every project.
    `fact_key_for` needs no special case for either.
    """
    from memvara import Memvara, NullLLM
    from memvara.embed import HashingEmbedder
    from memvara.schema import (BUILTIN_PREDICATES, Cardinality, PredicateRegistry,
                                PredicateSpec)
    from memvara.store import SQLiteStore

    registry = PredicateRegistry(BUILTIN_PREDICATES + (
        PredicateSpec(name="version", cardinality=Cardinality.ONE),))
    store = SQLiteStore(":memory:")
    cloud = Memvara(store=store, llm=NullLLM(), embedder=HashingEmbedder(dim=64),
                    registry=registry, user="alice", project="gh/o/cloud")
    web = Memvara(store=store, llm=NullLLM(), embedder=HashingEmbedder(dim=64),
                  registry=registry, user="alice", project="gh/o/web")
    try:
        # `version` is undeclared for project scoping, so it partitions: two repositories
        # running different versions are two true facts, not a contradiction.
        #
        # Each handle is asked about its own repository, and that is the point rather than
        # a detail of how the test is phrased. An earlier version of this test read both
        # values from one handle and passed, because the scope SQL was not yet comparing
        # the project — so it was reading the leak instead of the partition, and would
        # have gone on passing if the partition had never worked at all.
        cloud.remember("postgresql", "version", "17")
        web.remember("postgresql", "version", "16")

        def version_at(mem):
            return [c.object for c in mem.get_all(states=("live",))
                    if c.predicate == "version"]

        assert version_at(cloud) == ["17"]
        assert version_at(web) == ["16"], "neither value displaced the other"

        # `lives_in` is a builtin, so it is global: the second statement retires the
        # first even though it was made from a different repository.
        cloud.remember("user", "lives_in", "Berlin")
        web.remember("user", "lives_in", "Lisbon")
        live = [c.object for c in cloud.get_all(states=("live",))
                if c.predicate == "lives_in"]
        assert live == ["Lisbon"], "a global predicate holds one slot across projects"

        stored, = [c for c in cloud.get_all(states=("live",))
                   if c.predicate == "lives_in"]
        assert stored.scope.project is None
    finally:
        cloud.close()
        web.close()


def test_forget_and_history_find_a_global_predicate_written_from_a_project():
    """The probe has to be keyed the way the claim it is hunting was written.

    `_canonicalize` clears the project for a globally-declared predicate, so the stored
    `fact_key` has none. `forget()` and `history()` build a probe claim and look the slot
    up by its key, and while they built it from the caller's raw scope the two keys
    disagreed for exactly those predicates — which is all 23 builtins.

    Both failures were silent in the worst available way. `forget()` returned an empty
    list, which reads as "there was nothing to forget" rather than as a failure, and left
    the fact live. `history()` reported no history at all for a claim `get_all()` returns.

    The rule now has one definition, `PredicateRegistry.slot_scope`, and all three callers
    go through it. The partitioned predicate is here as the other half: it must keep its
    project, or the fix would have cured the probe by making every slot global.
    """
    from memvara import Memvara, NullLLM
    from memvara.embed import HashingEmbedder
    from memvara.schema import (BUILTIN_PREDICATES, Cardinality, PredicateRegistry,
                                PredicateSpec)
    from memvara.store import SQLiteStore

    registry = PredicateRegistry(BUILTIN_PREDICATES + (
        PredicateSpec(name="version", cardinality=Cardinality.ONE),))
    mem = Memvara(store=SQLiteStore(":memory:"), llm=NullLLM(),
                  embedder=HashingEmbedder(dim=64), registry=registry,
                  user="alice", project="gh/o/x")
    try:
        mem.remember("user", "lives_in", "Berlin")      # builtin, so global
        mem.remember("postgresql", "version", "17")     # undeclared, so partitioned

        assert [c.object for c in mem.history("user", "lives_in")] == ["Berlin"]
        assert [c.object for c in mem.history("postgresql", "version")] == ["17"]
        assert [c.object for c in mem.forget("user", "lives_in")] == ["Berlin"]
        assert [c.object for c in mem.forget("postgresql", "version")] == ["17"]
        assert mem.get_all(states=("live",)) == []
    finally:
        mem.close()
