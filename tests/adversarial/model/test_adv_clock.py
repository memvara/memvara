"""The fixed instants the reference model and its state machine write and read at."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from harness.clock import EPOCH, FAR_FUTURE, INSTANTS, within


def test_the_instants_are_utc_and_strictly_increasing() -> None:
    assert all(t.tzinfo is not None and t.utcoffset() == timedelta(0) for t in INSTANTS)
    assert list(INSTANTS) == sorted(set(INSTANTS))


def test_the_instants_lie_between_a_year_before_the_epoch_and_now() -> None:
    now = datetime.now(timezone.utc)
    assert all(EPOCH - timedelta(days=366) < t < now for t in INSTANTS)
    assert FAR_FUTURE > now


def test_within_includes_both_ends() -> None:
    assert within(EPOCH, EPOCH, EPOCH)
    assert not within(EPOCH, EPOCH + timedelta(microseconds=1), FAR_FUTURE)
