"""Decay behaviour and the idempotency guarantee the scheduler depends on."""

from __future__ import annotations

from datetime import timedelta

import pytest

from engram.consolidate import SALIENCE_FLOOR, Consolidator
from engram.embed.base import HashingEmbedder
from engram.schema import PredicateRegistry
from engram.store.sqlite import SQLiteStore
from engram.types import Claim, MemoryType, Scope, utcnow

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
    store = consolidator.store
    # Enough elapsed time to bottom out each half-life: a week, two years, a century.
    claims = [
        add(store, "working_on", "x", age_days=365 * 5),
        add(store, "works_at", "acme", age_days=365 * 50),
        add(store, "born_in", "lisbon", age_days=365 * 600),
    ]
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
    """A write-path bump raises the baseline instead of being erased by the next pass."""
    store = consolidator.store
    claim = add(store, "works_at", "acme", age_days=730, salience=0.4)

    consolidator.decay(now=NOW)
    assert salience_of(store, claim.id) == pytest.approx(0.2, abs=1e-6)

    store.reinforce(claim.id, salience=0.9, observation_count=3, sources=["ep_1"])
    consolidator.decay(now=NOW)
    # Re-anchored on the bumped value (0.9 * 0.5), not snapped back to the old base.
    assert salience_of(store, claim.id) == pytest.approx(0.45, abs=1e-6)
    assert consolidator.decay(now=NOW) == 0


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
