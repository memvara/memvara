"""The driver's checks that do not use the reference model, each shown catching its fault.

Most of what `Pair.apply` checks is a comparison with the model. The checks pinned here
read only the store, so they still hold if the model and the store were ever wrong in the
same way. Each test makes the store misbehave in one way and checks that the driver names
that fault, rather than reporting a difference from the model.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator
from typing import Any

import pytest

from memvara import Memvara
from memvara.schema import Cardinality, PredicateSpec

from harness.clock import INSTANTS
from harness.drive import ModelDivergence, Pair
from harness.invariants import check_store_integrity
from harness.model import Erase, Remember

I0, I1, I2, I3, I4, I5 = INSTANTS


@pytest.fixture()
def pair() -> Iterator[Pair]:
    with Pair() as p:
        yield p


def test_a_pair_on_a_file_leaves_a_store_that_passes_the_integrity_checks(
        tmp_path: pathlib.Path) -> None:
    path = tmp_path / "store.db"
    with Pair(path) as pair:
        pair.apply(Remember("u1", "lives_in", "Berlin", valid_from=I0))
        pair.apply(Remember("u1", "lives_in", "Paris", valid_from=I2))
        pair.apply(Erase("u1", "r1"))
    assert path.exists()
    assert check_store_integrity(path) == []


def test_a_read_that_returns_a_row_recorded_after_its_known_at_is_reported(
        pair: Pair, monkeypatch: pytest.MonkeyPatch) -> None:
    pair.apply(Remember("u1", "lives_in", "Berlin", valid_from=I0, recorded_at=I4))
    original = Memvara.get_all

    def no_belief_floor(self: Memvara, *args: Any, known_at: Any = None,
                        **kwargs: Any) -> Any:
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Memvara, "get_all", no_belief_floor)
    with pytest.raises(AssertionError, match=r"u1: a read returned r1, recorded at .* "
                                             r"after the read's known_at"):
        pair.check()


def test_a_read_that_returns_another_users_row_is_reported(
        pair: Pair, monkeypatch: pytest.MonkeyPatch) -> None:
    pair.apply(Remember("u1", "likes", "tea"))
    pair.apply(Remember("u2", "likes", "coffee"))
    original = Memvara.get_all

    def both_users(self: Memvara, *args: Any, user: Any = None, **kwargs: Any) -> Any:
        return [*original(self, *args, user="u1", **kwargs),
                *original(self, *args, user="u2", **kwargs)]

    monkeypatch.setattr(Memvara, "get_all", both_users)
    with pytest.raises(AssertionError, match="u1: a read returned r2, which belongs to u2"):
        pair.check()


def test_an_erased_row_that_get_still_returns_is_reported(
        pair: Pair, monkeypatch: pytest.MonkeyPatch) -> None:
    pair.apply(Remember("u1", "likes", "tea"))
    kept = pair.mem.get(pair.real("r1"), user="u1")
    pair.apply(Erase("u1", "r1"))
    monkeypatch.setattr(Memvara, "get", lambda self, claim_id, **kwargs: kept)
    with pytest.raises(AssertionError, match="get\\(\\) still returns the erased row r1"):
        pair.check()


def test_a_live_row_the_writer_cannot_read_back_is_reported(
        pair: Pair, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Memvara.get_all

    def without_paris(self: Memvara, *args: Any, **kwargs: Any) -> Any:
        return [c for c in original(self, *args, **kwargs) if c.object != "Paris"]

    monkeypatch.setattr(Memvara, "get_all", without_paris)
    with pytest.raises(ModelDivergence, match="u1 cannot read back r1, a live row it has "
                                              "just written"):
        pair.apply(Remember("u1", "lives_in", "Paris", valid_from=I0))


def test_a_write_that_closes_a_row_under_an_undeclared_predicate_is_reported(
        pair: Pair) -> None:
    # The store is told that `collects` holds one value; the driver still treats it as
    # a predicate nobody declared, so the store closing the old value is the fault.
    pair.mem.registry.register(PredicateSpec("collects", Cardinality.ONE))
    pair.apply(Remember("u1", "collects", "stamps", valid_from=I0))
    with pytest.raises(ModelDivergence, match="a write under collects, which nobody "
                                              "declared, closed \\['r1'\\]"):
        pair.apply(Remember("u1", "collects", "vinyl", valid_from=I2))


def test_a_search_that_returns_one_row_twice_is_reported(
        pair: Pair, monkeypatch: pytest.MonkeyPatch) -> None:
    pair.apply(Remember("u1", "likes", "tea"))
    original = Memvara.search

    def twice(self: Memvara, *args: Any, **kwargs: Any) -> Any:
        hits = original(self, *args, **kwargs)
        return hits + hits

    monkeypatch.setattr(Memvara, "search", twice)
    with pytest.raises(AssertionError, match="returned r1 more than once"):
        pair.check()
