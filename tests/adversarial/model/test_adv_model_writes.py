"""Hand-written write cases, each run on a real store and on the reference model.

`Pair.apply` runs one operation on both, then compares every row and every read, so each
case below fails at the first operation where the two disagree. The cases cover each
branch of `Reconciler.apply` and each closing call once, before the state machine mixes
them at random.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from harness.clock import FAR_FUTURE, INSTANTS
from harness.drive import Pair
from harness.model import Delete, Erase, EraseExpired, Forget, Lapse, Remember

I0, I1, I2, I3, I4, I5 = INSTANTS


@pytest.fixture()
def pair() -> Iterator[Pair]:
    with Pair() as p:
        yield p


def test_a_new_value_ends_the_old_one_where_it_begins(pair: Pair) -> None:
    pair.apply(Remember("u1", "lives_in", "Berlin", valid_from=I0))
    e = pair.apply(Remember("u1", "lives_in", "Paris", valid_from=I2))
    assert e.closed == ["r1"] and pair.model.rows["r1"].valid_to == I2


def test_a_value_from_before_the_live_one_is_stored_as_history(pair: Pair) -> None:
    pair.apply(Remember("u1", "lives_in", "Paris", valid_from=I3))
    e = pair.apply(Remember("u1", "lives_in", "Rome", valid_from=I1))
    assert e.closed == [] and pair.model.rows["r2"].valid_to == I3


def test_a_value_from_the_same_instant_collapses_the_old_one(pair: Pair) -> None:
    pair.apply(Remember("u1", "lives_in", "Berlin", valid_from=I2))
    e = pair.apply(Remember("u1", "lives_in", "Paris", valid_from=I2))
    assert e.collapsed == ["r1"]


def test_a_much_less_confident_value_is_stored_beside_the_old_one(pair: Pair) -> None:
    pair.apply(Remember("u1", "lives_in", "Berlin", valid_from=I0))
    e = pair.apply(Remember("u1", "lives_in", "Paris", valid_from=I2, confidence=0.4))
    assert e.disputed == ["r1"] and e.closed == []


def test_a_new_value_can_retire_the_old_one_instead_of_ending_it(pair: Pair) -> None:
    pair.apply(Remember("u1", "lives_in", "Berlin", valid_from=I0))
    pair.apply(Remember("u1", "lives_in", "Paris", valid_from=I2, close="retired"))
    assert pair.model.rows["r1"].invalidated_at is not None
    assert pair.model.rows["r1"].valid_to is None


def test_a_repeat_reinforces_the_live_row(pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea"))
    e = pair.apply(Remember("u1", "likes", "tea"))
    assert e.reinforced == ["r1"] and pair.model.rows["r1"].observations == 2


def test_a_repeat_with_an_earlier_start_is_stored_for_the_earlier_period(pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea", valid_from=I3, recorded_at=I3))
    e = pair.apply(Remember("u1", "likes", "tea", valid_from=I0, recorded_at=I4))
    assert e.new == "r2" and e.added and e.reinforced == []
    assert pair.model.rows["r2"].valid_to == I3 and pair.model.rows["r1"].observations == 1


def test_restating_an_earlier_period_again_is_a_repeat_of_it(pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea", valid_from=I4, recorded_at=I4))
    pair.apply(Remember("u1", "likes", "tea", valid_from=I2, recorded_at=I4))
    for start in (I2, I3):      # the same start, then a later one inside the period
        e = pair.apply(Remember("u1", "likes", "tea", valid_from=start, recorded_at=I5))
        assert e.new is None and e.reinforced == ["r2"]
    assert pair.model.rows["r2"].observations == 3


def test_restating_from_an_even_earlier_start_adds_only_the_period_not_held(
        pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea", valid_from=I4, recorded_at=I4))
    pair.apply(Remember("u1", "likes", "tea", valid_from=I2, recorded_at=I4))
    e = pair.apply(Remember("u1", "likes", "tea", valid_from=I0, recorded_at=I5))
    assert e.new == "r3" and pair.model.rows["r3"].valid_to == I2


def test_a_restatement_covers_a_gap_between_two_stored_periods(pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea", valid_from=I1, valid_to=I2, recorded_at=I1))
    pair.apply(Remember("u1", "likes", "tea", valid_from=I4, recorded_at=I4))
    e = pair.apply(Remember("u1", "likes", "tea", valid_from=I0, recorded_at=I5))
    assert e.new == "r3" and pair.model.rows["r3"].valid_to == I4


def test_a_retired_earlier_period_is_stored_again_when_restated(pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea", valid_from=I4, recorded_at=I4))
    pair.apply(Remember("u1", "likes", "tea", valid_from=I2, recorded_at=I4))
    pair.apply(Delete("u1", "r2"))
    e = pair.apply(Remember("u1", "likes", "tea", valid_from=I2, recorded_at=I5))
    assert e.new == "r3" and pair.model.rows["r3"].valid_to == I4


def test_a_repeat_with_an_earlier_start_and_an_expiry_still_reinforces(pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea", valid_from=I3))
    e = pair.apply(Remember("u1", "likes", "tea", valid_from=I0, expires_at=FAR_FUTURE))
    assert e.reinforced == ["r1"] and pair.model.rows["r1"].expires_at == FAR_FUTURE


def test_a_retraction_ends_the_value_it_names_and_leaves_a_tombstone(pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea", valid_from=I0))
    e = pair.apply(Remember("u1", "likes", "tea", polarity=-1, valid_from=I2))
    assert e.new == "r2" and e.closed == ["r1"]
    assert pair.model.rows["r1"].valid_to == I2


def test_a_retraction_dated_in_the_future_leaves_a_tombstone_with_an_empty_interval(
        pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea"))
    e = pair.apply(Remember("u1", "likes", "tea", polarity=-1, valid_from=FAR_FUTURE))
    tombstone = pair.model.rows["r2"]
    assert e.closed == ["r1"] and pair.model.rows["r1"].valid_to == FAR_FUTURE
    assert tombstone.valid_from == tombstone.valid_to == FAR_FUTURE


def test_a_retraction_dated_in_the_future_can_retire_the_value_it_names(
        pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea"))
    pair.apply(Remember("u1", "likes", "tea", polarity=-1, valid_from=FAR_FUTURE,
                        close="retired"))
    tombstone = pair.model.rows["r2"]
    assert pair.model.rows["r1"].invalidated_at is not None
    assert tombstone.valid_from == tombstone.valid_to == FAR_FUTURE


def test_a_retraction_dated_in_the_future_sent_twice_agrees_with_the_model(
        pair: Pair) -> None:
    """The store and the model must agree on a repeat of a future-dated retraction. What
    that repeat should do is #349; this checks only that the two do the same thing."""
    pair.apply(Remember("u1", "likes", "tea"))
    pair.apply(Remember("u1", "likes", "tea", polarity=-1, valid_from=FAR_FUTURE))
    pair.apply(Remember("u1", "likes", "tea", polarity=-1, valid_from=FAR_FUTURE))


def test_a_repeated_retraction_reinforces_its_tombstone_and_reports_nothing(
        pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea", valid_from=I0))
    pair.apply(Remember("u1", "likes", "tea", polarity=-1, valid_from=I2))
    e = pair.apply(Remember("u1", "likes", "tea", polarity=-1, valid_from=I3))
    assert e.reinforced == ["r2"] and not e.reinforced_reported


def test_a_retraction_of_a_value_that_is_not_there_writes_nothing(pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea"))
    e = pair.apply(Remember("u1", "likes", "coffee", polarity=-1))
    assert e.new is None and e.closed == []


def test_a_retraction_in_an_empty_slot_still_leaves_a_tombstone(pair: Pair) -> None:
    e = pair.apply(Remember("u1", "collects", "stamps", polarity=-1))
    assert e.new == "r1" and pair.model.rows["r1"].state == "retired"


def test_a_second_value_nobody_declared_is_reported_as_an_accumulation(
        pair: Pair) -> None:
    pair.apply(Remember("u1", "collects", "stamps"))
    e = pair.apply(Remember("u1", "collects", "vinyl"))
    assert e.accumulated == 1


def test_forget_retires_every_live_row_in_the_slot(pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea"))
    pair.apply(Remember("u1", "likes", "coffee"))
    e = pair.apply(Forget("u1", "likes"))
    assert e.returned == ["r1", "r2"]


def test_forget_retires_a_row_scheduled_to_begin_later_and_leaves_an_ended_one(
        pair: Pair) -> None:
    pair.apply(Remember("u1", "lives_in", "Rome", valid_from=I0, valid_to=I1))
    pair.apply(Remember("u1", "lives_in", "Berlin", valid_from=I2))
    pair.apply(Remember("u1", "lives_in", "Paris", valid_from=FAR_FUTURE))
    e = pair.apply(Forget("u1", "lives_in"))
    assert e.returned == ["r2", "r3"] and pair.model.rows["r1"].state == "ended"


def test_delete_can_end_a_row_instead_of_retiring_it(pair: Pair) -> None:
    pair.apply(Remember("u1", "lives_in", "Berlin", valid_from=I0))
    pair.apply(Delete("u1", "r1", close="ended"))
    assert pair.model.rows["r1"].state == "ended"


def test_delete_ends_a_future_row_at_its_own_start(pair: Pair) -> None:
    pair.apply(Remember("u1", "lives_in", "Berlin", valid_from=FAR_FUTURE))
    pair.apply(Delete("u1", "r1", close="ended"))
    assert pair.model.rows["r1"].valid_to == FAR_FUTURE


def test_a_backdated_record_is_believed_from_its_own_recording(pair: Pair) -> None:
    pair.apply(Remember("u1", "lives_in", "Berlin", valid_from=I0, recorded_at=I1))
    pair.apply(Remember("u1", "lives_in", "Paris", valid_from=I3, recorded_at=I4))


def test_erase_removes_a_row_and_keeps_an_audit_record(pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea"))
    e = pair.apply(Erase("u1", "r1"))
    assert e.returned is True and pair.model.erased == {"r1"}


def test_an_expired_row_vanishes_and_the_sweep_erases_it(pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea", expires_at=FAR_FUTURE))
    pair.apply(Lapse("r1", pair.lapsed_instant()))
    e = pair.apply(EraseExpired())
    assert e.returned == ["r1"]


def test_one_users_writes_never_reach_the_other_user(pair: Pair) -> None:
    pair.apply(Remember("u1", "lives_in", "Berlin", valid_from=I0))
    pair.apply(Remember("u2", "lives_in", "Paris", valid_from=I1))
    e = pair.apply(Erase("u2", "r1"))
    assert e.returned is False


def test_delete_retires_the_callers_own_row(pair: Pair) -> None:
    pair.apply(Remember("u1", "likes", "tea"))
    e = pair.apply(Delete("u1", "r1"))
    assert e.returned is True and pair.model.rows["r1"].state == "retired"


def test_a_value_with_an_end_is_true_from_its_start_to_its_end(pair: Pair) -> None:
    pair.apply(Remember("u1", "lives_in", "Berlin", valid_from=I0, valid_to=I2))
    assert pair.model.rows["r1"].valid_to == I2


def test_an_end_at_or_before_the_start_is_refused(pair: Pair) -> None:
    e = pair.apply(Remember("u1", "lives_in", "Berlin", valid_from=I2, valid_to=I2))
    assert e.refused and e.new is None and pair.model.rows == {}


def test_an_end_is_brought_forward_to_where_a_later_value_begins(pair: Pair) -> None:
    pair.apply(Remember("u1", "lives_in", "Paris", valid_from=I3))
    pair.apply(Remember("u1", "lives_in", "Berlin", valid_from=I1, valid_to=I5))
    assert pair.model.rows["r2"].valid_to == I3
