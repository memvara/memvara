"""The reference model's reads reproduce the examples INTERNALS documents.

These run against the model alone. It has to reproduce the documented examples before it
is trusted to judge the store.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from harness.clock import EPOCH, moments_around
from harness.model import STATES, ReferenceStore, Row

NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)
MONTH = timedelta(days=31)


def row(rid: str, **kw: object) -> Row:
    values: dict[str, object] = dict(id=rid, user="u", subject="user", predicate="lives_in",
                                     obj=rid, polarity=1, confidence=1.0,
                                     valid_from=EPOCH, recorded_at=EPOCH)
    values.update(kw)
    return Row(**values)  # type: ignore[arg-type]


def test_the_stats_table_counts_four_rows_that_do_not_sum() -> None:
    """INTERNALS, "stats()": one live claim, one ended claim, one that ended and was
    later retired, and one recorded but not in force until next year."""
    m = ReferenceStore()
    m.add(row("live"))
    m.add(row("ended", valid_to=EPOCH + MONTH))
    m.add(row("retired", valid_to=EPOCH + MONTH, invalidated_at=EPOCH + 2 * MONTH))
    m.add(row("later", valid_from=NOW + timedelta(days=365)))
    assert m.stats(NOW) == {"claims": 4, "live_claims": 1, "ended_claims": 1,
                            "invalidated": 1}


def test_the_audit_view_is_the_belief_floor_alone() -> None:
    """INTERNALS, "The three states do not tile the store": a row recorded but not yet in
    force is in none of the three states, and asking for all three readmits it."""
    m = ReferenceStore()
    m.add(row("now"))
    m.add(row("later", valid_from=NOW + timedelta(days=365)))
    for states in (("live",), ("ended",), ("retired",), ("live", "ended"),
                   ("live", "retired"), ("ended", "retired")):
        assert "later" not in m.visible("u", states=states, now=NOW), states
    assert m.visible("u", states=STATES, now=NOW) == {"now", "later"}


def test_rome_and_berlin_leave_a_gap_as_of_the_fifteenth() -> None:
    """INTERNALS, "ask() reconstructs an ending the row cannot date"."""
    m = ReferenceStore()
    march = datetime(2026, 3, 1, tzinfo=timezone.utc)
    m.add(row("rome", valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
              recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc), valid_to=march))
    m.add(row("berlin", valid_from=march,
              recorded_at=datetime(2026, 3, 22, tzinfo=timezone.utc)))
    ides = datetime(2026, 3, 15, tzinfo=timezone.utc)
    assert m.visible("u", valid_at=ides, known_at=ides, now=NOW) == set()
    assert m.visible("u", now=NOW) == {"berlin"}


def test_a_collapsed_row_is_true_at_no_instant_and_stays_in_history() -> None:
    m = ReferenceStore()
    m.add(row("delhi", valid_from=EPOCH, valid_to=EPOCH))
    for t in moments_around(EPOCH):
        assert m.visible("u", valid_at=t, known_at=NOW, now=NOW) == set()
    assert m.history("u", "user", "lives_in", now=NOW) == ["delhi"]


def test_the_boundaries_are_inclusive_at_the_start_and_exclusive_at_the_end() -> None:
    m = ReferenceStore()
    m.add(row("rome", valid_to=EPOCH + MONTH))
    before, at, after = moments_around(EPOCH)
    assert m.visible("u", valid_at=before, known_at=NOW, now=NOW) == set()
    assert m.visible("u", valid_at=at, known_at=NOW, now=NOW) == {"rome"}
    end_before, end_at, _ = moments_around(EPOCH + MONTH)
    assert m.visible("u", valid_at=end_before, known_at=NOW, now=NOW) == {"rome"}
    assert m.visible("u", valid_at=end_at, known_at=NOW, now=NOW) == set()
    assert m.visible("u", known_at=before, now=NOW) == set()


def test_an_expired_row_vanishes_from_every_read_but_stays_counted_as_a_row() -> None:
    m = ReferenceStore()
    m.add(row("code", expires_at=NOW - timedelta(days=1)))
    for states in (("live",), STATES):
        assert m.visible("u", states=states, now=NOW) == set()
    assert m.history("u", "user", "lives_in", now=NOW) == []
    assert m.stats(NOW)["claims"] == 1 and m.stats(NOW)["live_claims"] == 0


def test_one_users_rows_are_never_another_users() -> None:
    m = ReferenceStore()
    m.add(row("mine"))
    m.add(row("theirs", user="v"))
    assert m.visible("u", states=STATES, now=NOW) == {"mine"}
    assert m.visible("v", states=STATES, now=NOW) == {"theirs"}


def test_history_is_the_slot_in_recording_order() -> None:
    m = ReferenceStore()
    m.add(row("b", recorded_at=EPOCH + MONTH))
    m.add(row("a", recorded_at=EPOCH + MONTH))
    m.add(row("c", recorded_at=EPOCH))
    m.add(row("other", predicate="works_at"))
    assert m.history("u", "user", "lives_in", now=NOW) == ["c", "a", "b"]


def test_the_pools_hold_one_predicate_of_each_kind() -> None:
    """The machine exercises supersession, a declared many-valued predicate and an
    accumulation only while the pools keep one predicate of each kind. The kinds come from
    memvara's own registry, so a change there cannot quietly remove a branch from every
    run."""
    from harness.model import DECLARED, FUNCTIONAL, POOLS

    assert FUNCTIONAL == {"lives_in"}
    assert DECLARED - FUNCTIONAL == {"likes"}
    assert set(POOLS) - DECLARED == {"collects"}
