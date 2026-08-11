"""Decay behaviour and the idempotency guarantee the scheduler depends on."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import timedelta

import pytest

from memvara.consolidate import SALIENCE_FLOOR, Consolidator
from memvara.embed.base import HashingEmbedder
from memvara.retrieve.scoring import recency_factor
from memvara.schema import PredicateRegistry
from memvara.store.sqlite import _MIN_TS, SQLiteStore, _dt
from memvara.telemetry import (
    CONSOLIDATE_CLAIMS_PER_SLOT,
    CONSOLIDATE_CROWDED_SLOTS,
    CONSOLIDATE_DECAYED,
    CONSOLIDATE_LATENCY_MS,
    CONSOLIDATE_MERGED,
    CONSOLIDATE_PROMOTED,
    CONSOLIDATE_ROWS_WRITTEN,
    CROWDED_SLOT,
    MemoryRecorder,
)
from memvara.consolidate.sweep import Sweep
from memvara.types import Claim, MemoryType, Scope, utcnow
from memvara.write.reconcile import Reconciler

NOW = utcnow()


@pytest.fixture
def consolidator():
    store = SQLiteStore(":memory:")
    yield Consolidator(store, HashingEmbedder(dim=64), PredicateRegistry())
    store.close()


def add(store, predicate: str, obj: str, *, age_days: float = 0.0, **kw) -> Claim:
    """A live claim whose fact became true `age_days` ago."""
    ts = NOW - timedelta(days=age_days)
    claim = Claim(
        subject="user",
        predicate=predicate,
        object=obj,
        scope=Scope(tenant="acme", user="u1"),
        valid_from=ts,
        recorded_at=ts,
        **kw,
    )
    store.put_claim(claim)
    return claim


def salience_of(store, claim_id: str) -> float:
    stored = store.get_claim(claim_id)
    assert stored is not None
    return stored.salience


# -- rates ------------------------------------------------------------------


def test_volatility_decays_at_measurably_different_rates(consolidator):
    store = consolidator.store
    fast = add(store, "working_on", "the migration", age_days=30)   # 7d half-life
    slow = add(store, "works_at", "acme", age_days=30)              # 730d half-life
    static = add(store, "born_in", "lisbon", age_days=30)           # 36500d half-life

    assert consolidator.decay(now=NOW) == 3

    s_fast = salience_of(store, fast.id)
    s_slow = salience_of(store, slow.id)
    s_static = salience_of(store, static.id)

    assert s_fast < s_slow < s_static
    # A month is four half-lives for a FAST predicate and a rounding error for a STATIC
    # one - that spread is the whole reason volatility is a schema property.
    assert s_fast == pytest.approx(0.5 ** (30 / 7), abs=1e-6)
    assert s_slow == pytest.approx(0.5 ** (30 / 730), abs=1e-6)
    assert s_static == pytest.approx(1.0, abs=1e-3)
    assert s_fast < 0.06 and s_slow > 0.95


def test_salience_floors_at_005_after_years_and_never_reaches_zero(consolidator):
    """Ages are capped to what the platform can store, which is not a formality.

    `_ts` clamps both ends, because a timestamp the C library cannot invert is a write
    that breaks every later read of its scope. Windows' floor is the epoch, so the 600
    years this test used to ask for arrived as 1970 — about 56 years, which does not
    bottom out a century half-life, and the assertion failed at 0.675 while decay was
    working correctly. Asking for an age the store can actually hold keeps the test about
    decay rather than about the clock."""
    store = consolidator.store
    oldest = _dt(_MIN_TS)
    assert oldest is not None
    limit_days = (NOW - oldest).days
    # Enough elapsed time to bottom out each half-life: a week, two years, a century.
    claims = [
        add(store, "working_on", "x", age_days=min(365 * 5, limit_days)),
        add(store, "works_at", "acme", age_days=min(365 * 50, limit_days)),
        add(store, "born_in", "lisbon", age_days=min(365 * 600, limit_days)),
    ]
    if limit_days < 365 * 600:
        pytest.skip(f"this platform stores at most {limit_days} days of history, which "
                    "cannot reach the floor of a century half-life")
    consolidator.decay(now=NOW)

    for claim in claims:
        value = salience_of(store, claim.id)
        assert value == SALIENCE_FLOOR
        assert value > 0.0


def test_decay_measured_from_valid_from_not_recorded_at(consolidator):
    """A fact back-dated on ingest must not arrive artificially fresh."""
    store = consolidator.store
    backdated = Claim(
        subject="user",
        predicate="working_on",
        object="the old project",
        scope=Scope(tenant="acme"),
        valid_from=NOW - timedelta(days=28),
        recorded_at=NOW,
    )
    store.put_claim(backdated)

    consolidator.decay(now=NOW)
    assert salience_of(store, backdated.id) == pytest.approx(0.5**4, abs=1e-6)


def test_future_dated_claim_does_not_gain_salience(consolidator):
    store = consolidator.store
    future = add(store, "works_at", "acme", age_days=-90)

    consolidator.decay(now=NOW)
    assert salience_of(store, future.id) == 1.0


# -- idempotency ------------------------------------------------------------


def test_decay_twice_at_same_instant_is_a_no_op(consolidator):
    store = consolidator.store
    claim = add(store, "works_at", "acme", age_days=730)  # exactly one SLOW half-life

    assert consolidator.decay(now=NOW) == 1
    first = salience_of(store, claim.id)
    assert first == pytest.approx(0.5, abs=1e-6)

    # Second pass finds nothing to write, and crucially does not halve 0.5 again.
    assert consolidator.decay(now=NOW) == 0
    assert salience_of(store, claim.id) == first


def test_run_twice_leaves_identical_state(consolidator):
    store = consolidator.store
    add(store, "works_at", "acme", age_days=730)
    add(store, "working_on", "the migration", age_days=14)
    add(store, "born_in", "lisbon", age_days=365 * 3)
    add(store, "goal", "ship v2", age_days=1, memory_type=MemoryType.EPISODIC,
        observation_count=4)
    add(store, "likes", "coffee", age_days=10)
    add(store, "likes", "Coffee", age_days=10, observation_count=2)

    first = consolidator.run()
    snapshot = {
        c.id: (round(c.salience, 6), c.observation_count, c.memory_type, c.invalidated_by)
        for c in store.iter_claims("acme", include_invalidated=True)
    }
    stats = store.stats()

    second = consolidator.run()

    assert second == {"decayed": 0, "merged": 0, "promoted": 0}
    assert first["merged"] == 1 and first["promoted"] == 1
    assert store.stats() == stats
    assert {
        c.id: (round(c.salience, 6), c.observation_count, c.memory_type, c.invalidated_by)
        for c in store.iter_claims("acme", include_invalidated=True)
    } == snapshot


def test_five_consecutive_runs_do_not_compound(consolidator):
    store = consolidator.store
    claim = add(store, "works_at", "acme", age_days=730)

    consolidator.decay(now=NOW)
    settled = salience_of(store, claim.id)
    for _ in range(5):
        assert consolidator.decay(now=NOW) == 0
    assert salience_of(store, claim.id) == settled
    # Compounding five more times would have driven a half-life-old claim to ~0.016.
    assert settled == pytest.approx(0.5, abs=1e-6)


def test_decay_advances_when_time_actually_passes(consolidator):
    """Idempotency must not mean 'frozen' - a later pass still decays further."""
    store = consolidator.store
    claim = add(store, "works_at", "acme", age_days=730)

    consolidator.decay(now=NOW)
    assert consolidator.decay(now=NOW + timedelta(days=730)) == 1
    assert salience_of(store, claim.id) == pytest.approx(0.25, abs=1e-6)


def test_reinforcement_between_passes_is_not_undone(consolidator):
    """A write-path bump raises the baseline instead of being erased by the next pass.

    At the *production* parameters - a claim at the default salience 1.0 and the
    pipeline's default bump of 0.25. The version of this test that used a bump of 0.5
    on a base of 0.4 passed against code for which the property was false: a bump only
    outran the recomputed curve while `age < 0.415 * half_life`, which is 2.9 days for
    a FAST predicate and 303 for a SLOW one, and past that it was erased on the next
    pass and every pass after it, because age only ever grows.
    """
    store = consolidator.store
    rec = Reconciler(store, consolidator.registry)
    claim = add(store, "works_at", "acme", age_days=730)  # one SLOW half-life

    consolidator.decay(now=NOW)
    assert salience_of(store, claim.id) == pytest.approx(0.5, abs=1e-6)

    rec.reinforce(store.get_claim(claim.id), ["ep_1"], NOW)
    # Half the trace had faded, so the observation buys most of the bump: 0.25 * (0.1 +
    # 0.9 * 0.5). It lands on the *base*, and the observation instant resets the clock.
    assert salience_of(store, claim.id) == pytest.approx(1.1375, abs=1e-6)

    assert consolidator.decay(now=NOW) == 0
    assert salience_of(store, claim.id) == pytest.approx(1.1375, abs=1e-6)


def test_a_fact_mentioned_every_day_ends_up_salient_not_at_the_floor(consolidator):
    """The headline failure, as one assertion.

    Measured on the shipped code: `observation_count=91`, `salience=0.05` (the floor),
    `recency_factor=1.35e-04`. Ninety days of daily reinforcement produced a claim
    ranked below one nobody had mentioned since it was written.
    """
    store = consolidator.store
    rec = Reconciler(store, consolidator.registry)
    start = NOW - timedelta(days=90)
    claim = add(store, "working_on", "the migration", age_days=90)  # FAST, 7d half-life

    for day in range(1, 91):
        at = start + timedelta(days=day)
        consolidator.decay(now=at)                       # the nightly scheduler
        rec.reinforce(store.get_claim(claim.id), [f"ep_{day}"], at)

    stored = store.get_claim(claim.id)
    assert stored.observation_count == 91
    assert stored.salience == pytest.approx(5.0)         # the ceiling, not the floor
    assert recency_factor(stored, consolidator.registry, NOW) == pytest.approx(1.0)


def test_spacing_beats_massing(consolidator):
    """Ten mentions spread over ten months must outrank ten in one conversation.

    Ebbinghaus's spacing effect (Cepeda et al. 2006), which the shipped rule had
    inverted: a flat bump on a value the next pass recomputed gave massed 3.25 and
    distributed 0.05 - a factor of 65 the wrong way.
    """
    store = consolidator.store
    rec = Reconciler(store, consolidator.registry)

    massed = add(store, "working_on", "the massed thing", age_days=0)
    for i in range(10):
        rec.reinforce(store.get_claim(massed.id), [f"m_{i}"], NOW)

    start = NOW - timedelta(days=300)
    spaced = add(store, "working_on", "the spaced thing", age_days=300)
    for i in range(10):
        at = start + timedelta(days=30 * i)
        consolidator.decay(now=at)
        rec.reinforce(store.get_claim(spaced.id), [f"s_{i}"], at)

    end = start + timedelta(days=270)
    assert salience_of(store, spaced.id) > 2 * salience_of(store, massed.id)
    assert recency_factor(store.get_claim(spaced.id), consolidator.registry, end) == \
        pytest.approx(1.0)


def test_massed_repetition_cannot_pin_a_claim_at_the_ceiling(consolidator):
    """The cheap poisoning vector: repeat a fact N times inside one `add()`.

    Under a flat bump, sixteen repetitions of one sentence pinned a claim at the cap
    permanently. A massed repetition now earns a tenth of the bump, so buying the cap
    costs 160 restatements of something nobody believes.
    """
    store = consolidator.store
    rec = Reconciler(store, consolidator.registry)
    claim = add(store, "likes", "the attacker's product")

    for i in range(16):
        rec.reinforce(store.get_claim(claim.id), [f"ep_{i}"], NOW)

    stored = store.get_claim(claim.id)
    assert stored.observation_count == 17
    assert stored.salience == pytest.approx(1.4, abs=1e-6)   # 1.0 + 16 * 0.025
    assert stored.salience < rec.max_salience


# -- degenerate inputs ------------------------------------------------------


def test_empty_store(consolidator):
    assert consolidator.decay(now=NOW) == 0
    assert consolidator.run() == {"decayed": 0, "merged": 0, "promoted": 0}


def test_single_claim_store(consolidator):
    claim = add(consolidator.store, "works_at", "acme", age_days=365)
    assert consolidator.run() == {"decayed": 1, "merged": 0, "promoted": 0}
    assert 0.0 < salience_of(consolidator.store, claim.id) < 1.0


def test_decay_is_scoped_to_one_tenant(consolidator):
    store = consolidator.store
    mine = add(store, "working_on", "x", age_days=90)
    theirs = Claim(
        subject="user", predicate="working_on", object="y",
        scope=Scope(tenant="other"), valid_from=NOW - timedelta(days=90),
        recorded_at=NOW - timedelta(days=90),
    )
    store.put_claim(theirs)

    assert consolidator.decay(tenant="acme", now=NOW) == 1
    assert salience_of(store, mine.id) == SALIENCE_FLOOR
    assert salience_of(store, theirs.id) == 1.0


def test_invalidated_claims_are_left_alone(consolidator):
    store = consolidator.store
    retired = add(store, "works_at", "oldco", age_days=400)
    store.invalidate(retired.id, at=NOW, by=None)

    assert consolidator.decay(now=NOW) == 0
    assert salience_of(store, retired.id) == 1.0


def test_unknown_predicate_decays_at_the_conservative_slow_rate(consolidator):
    store = consolidator.store
    claim = add(store, "invented_by_an_llm", "something", age_days=730)

    consolidator.decay(now=NOW)
    assert salience_of(store, claim.id) == pytest.approx(0.5, abs=1e-6)


def test_a_claim_nobody_restated_still_decays_from_valid_from(consolidator):
    """Re-observation is what moves the clock, and nothing else does.

    The counterweight to the tests above: an honest historical import must not become
    fresh just because the trace-origin rule exists.
    """
    store = consolidator.store
    backdated = add(store, "working_on", "the 2019 migration", age_days=28)
    backdated.recorded_at = NOW
    store.put_claim(backdated)

    consolidator.decay(now=NOW)
    assert salience_of(store, backdated.id) == pytest.approx(0.5**4, abs=1e-6)


# -- bounded transactions ---------------------------------------------------


class GappedStore:
    """A store that reports its transactions and runs a probe in each gap between them.

    The property under test is not "the sweep is fast", it is "the sweep is interruptible":
    holding one transaction for the whole pass hands an external writer a hard
    `database is locked` instead of backpressure, and the outage grows with the store.
    """

    def __init__(self, inner, probe):
        self._inner = inner
        self._probe = probe
        self.transactions = 0

    @contextmanager
    def batch(self):
        self.transactions += 1
        with self._inner.batch():
            yield
        self._probe()   # after the commit, so this runs with the write lock released

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_a_sweep_writes_in_bounded_transactions_an_outsider_can_write_between(tmp_path):
    path = str(tmp_path / "sweep.db")
    store = SQLiteStore(path)
    for i in range(10):
        add(store, "working_on", f"task {i}", age_days=30)

    outside = sqlite3.connect(path, timeout=0.25)
    landed: list[bool] = []

    def probe():
        try:
            outside.execute(
                "INSERT INTO episodes (id, tenant, role, content, ts, hash) "
                "VALUES (?,?,?,?,?,?)",
                (f"ext_{len(landed)}", "acme", "user", "hi", 0.0, "h"))
            outside.commit()
            landed.append(True)
        except sqlite3.OperationalError:  # pragma: no cover - the failure this prevents
            landed.append(False)

    gapped = GappedStore(store, probe)
    con = Consolidator(gapped, HashingEmbedder(dim=64), PredicateRegistry(), window=3)

    assert con.decay(now=NOW) == 10
    # Ten dirty rows at three per transaction: four transactions, four gaps, and an
    # outside writer that got the lock in every one of them.
    assert gapped.transactions == 4
    assert landed == [True] * 4

    outside.close()
    store.close()


def test_a_settled_store_opens_no_transaction_at_all(consolidator):
    """Nothing changed means nothing is written, so there is nothing to lock."""
    store = consolidator.store
    add(store, "works_at", "acme", age_days=365)
    consolidator.run()

    counted = GappedStore(store, lambda: None)
    con = Consolidator(counted, consolidator.embedder, consolidator.registry)
    assert con.run("acme") == {"decayed": 0, "merged": 0, "promoted": 0}
    assert counted.transactions == 0


# -- telemetry ---------------------------------------------------------------
#
# Consolidation is the subsystem whose numbers matter most over a year and the one
# nobody watches, because a scheduled pass that has silently stopped running looks
# exactly like a settled store.


def test_the_snapshot_reports_how_crowded_the_slots_are(consolidator):
    """Live claims per concept: the metric to build if only one gets built.

    Thirteen simultaneously-live answers to "where does the user work?", four of them
    different employers, produced no error and no log line across a 10,000-write
    simulation. This is the number that would have shown it in week one."""
    rec = MemoryRecorder()
    store = consolidator.store
    for employer in ("acme", "globex", "initech", "hooli"):
        # `worked_for` is unregistered, so it is MANY-cardinality and nothing retires
        # anything - exactly how thirteen live employers happened.
        add(store, "worked_for", employer)
    add(store, "lives_in", "berlin")
    Consolidator(store, consolidator.embedder, consolidator.registry,
                 telemetry=rec).decay(now=NOW)

    assert rec.values(CONSOLIDATE_CLAIMS_PER_SLOT) == [4.0]
    assert rec.values(CONSOLIDATE_CROWDED_SLOTS) == [1.0]   # only the employer slot
    assert 4 > CROWDED_SLOT


def test_a_slot_with_a_long_history_is_not_reported_as_crowded(consolidator):
    """The metric counts *live* claims per concept, and a claim that has ended is not one.

    Since supersession stopped closing transaction time, the belief-axis filter the
    snapshot used to rely on lets every superseded version through — so a slot with one
    current employer and five previous ones would report six, cross the crowding
    threshold, and fire the alarm whose entire value is that it only fires on the real
    thing. A metric that shouts at healthy stores gets muted, and then it is gone.

    The same filter keeps the pass bounded by the store's live size rather than by its
    whole history, which is the other reason it cannot be left to the belief axis alone.
    """
    rec = MemoryRecorder()
    store = consolidator.store
    reconciler = Reconciler(store, consolidator.registry)
    employers = ("acme", "globex", "initech", "hooli", "stark", "umbrella")
    for i, employer in enumerate(employers):
        # One job change a year, oldest first, so the chain builds the way it would have.
        at = NOW - timedelta(days=365 * (len(employers) - i))
        reconciler.apply(Claim(subject="user", predicate="works_at", object=employer,
                               scope=Scope(tenant="acme", user="u1"),
                               valid_from=at, recorded_at=at), now=at)

    sweep = Sweep(store, "acme", now=NOW, telemetry=rec)

    on_disk = list(store.iter_claims("acme", include_invalidated=True))
    assert len(on_disk) == len(employers), "every version is still on disk"
    assert [c.object for c in sweep.claims] == ["umbrella"]
    assert rec.values(CONSOLIDATE_CLAIMS_PER_SLOT) == [1.0]
    assert rec.values(CONSOLIDATE_CROWDED_SLOTS) == [0.0]


def test_an_empty_store_reports_a_maximum_of_zero_rather_than_nothing(consolidator):
    rec = MemoryRecorder()
    Consolidator(consolidator.store, consolidator.embedder, consolidator.registry,
                 telemetry=rec).decay(now=NOW)
    assert rec.values(CONSOLIDATE_CLAIMS_PER_SLOT) == [0.0]


def test_a_settled_pass_reports_zeroes_instead_of_going_quiet(consolidator):
    """"Decay has reported 0 for three months" is a settled store; "no decay series at
    all" is a scheduler nobody noticed had stopped. Only reporting the zero tells those
    two apart, so every stage emits even when it changed nothing."""
    rec = MemoryRecorder()
    store = consolidator.store
    add(store, "works_at", "acme", age_days=365)
    consolidator.run()

    con = Consolidator(store, consolidator.embedder, consolidator.registry,
                       telemetry=rec)
    assert con.run("acme") == {"decayed": 0, "merged": 0, "promoted": 0}
    assert con.store is store
    assert rec.total(CONSOLIDATE_DECAYED) == 0
    assert rec.total(CONSOLIDATE_MERGED) == 0
    assert rec.total(CONSOLIDATE_PROMOTED) == 0
    assert rec.total(CONSOLIDATE_ROWS_WRITTEN) == 0
    assert len(rec.values(CONSOLIDATE_LATENCY_MS)) == 1


def test_a_working_pass_reports_what_it_changed_and_what_it_wrote(consolidator):
    rec = MemoryRecorder()
    store = consolidator.store
    for i in range(3):
        add(store, "working_on", f"task {i}", age_days=30)
    con = Consolidator(store, consolidator.embedder, consolidator.registry,
                       telemetry=rec)
    assert con.decay(now=NOW) == 3
    assert rec.total(CONSOLIDATE_DECAYED) == 3
    assert rec.total(CONSOLIDATE_ROWS_WRITTEN) == 3


def test_each_stage_entry_point_carries_the_recorder_on_its_own(consolidator):
    """`run()` is the scheduled path, but a deployment that only calls one stage needs
    the slot metric just as much - so the recorder rides on the `Sweep`, which every
    entry point builds."""
    rec = MemoryRecorder()
    add(consolidator.store, "works_at", "acme")
    con = Consolidator(consolidator.store, consolidator.embedder,
                       consolidator.registry, telemetry=rec)
    con.decay(now=NOW)
    con.merge_duplicates("acme")
    con.promote("acme")
    assert len(rec.values(CONSOLIDATE_CLAIMS_PER_SLOT)) == 3
    assert rec.total(CONSOLIDATE_DECAYED) == 1
    assert rec.total(CONSOLIDATE_MERGED) == 0
    assert rec.total(CONSOLIDATE_PROMOTED) == 0
