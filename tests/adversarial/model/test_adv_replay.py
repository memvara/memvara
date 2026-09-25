"""A program that makes the store and the model disagree can be replayed as a test."""

from __future__ import annotations

import datetime

import pytest

from harness import model
from harness.clock import FAR_FUTURE, INSTANTS
from harness.drive import ModelDivergence, format_program, replay
from harness.model import Delete, EraseExpired, Expect, Forget, Lapse, Op, Remember

PROGRAM: list[Op] = [
    Remember("u1", "lives_in", "Berlin", valid_from=INSTANTS[0]),
    Remember("u1", "lives_in", "Paris", valid_from=INSTANTS[2], confidence=0.4),
    Remember("u1", "likes", "tea", expires_at=FAR_FUTURE),
    Forget("u1", "lives_in"),
    Delete("u2", "r1"),
    EraseExpired(),
]


def test_a_program_the_store_agrees_with_replays_cleanly() -> None:
    replay(PROGRAM)  # raises ModelDivergence at the first difference


def test_a_divergence_names_the_operation_where_the_two_parted(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that forgets nothing disagrees with the store at the `Forget`, which is
    the fourth operation."""
    monkeypatch.setattr(model.ReferenceStore, "forget",
                        lambda self, op, t: Expect(returned=[]))
    with pytest.raises(ModelDivergence) as caught:
        replay(PROGRAM)
    assert caught.value.index == 3 and isinstance(caught.value.op, Forget)


def test_a_printed_program_reads_back_as_the_same_operations() -> None:
    lapse = Lapse("r3", INSTANTS[1])
    ops: list[Op] = [*PROGRAM, lapse]
    names = {name: getattr(model, name) for name in (
        "Remember", "Forget", "Delete", "Erase", "Lapse", "EraseExpired")}
    names["datetime"] = datetime
    assert eval(format_program(ops), names) == ops  # noqa: S307 - our own output
    assert format_program([]) == "[]"
