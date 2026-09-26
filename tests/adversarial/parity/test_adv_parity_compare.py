"""The comparison the parity tests rest on removes only what differs between two stores
for reasons that have nothing to do with the surface, and reports everything else.

A normaliser that removed too much would make every parity test pass whatever the
surfaces returned, so each rule here is paired with a check that a real difference
still shows.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from memvara.confirm import CONFIRM_TTL
from memvara.core import PROFILE_WINDOW
from memvara.types import (CLOSURE, Claim, Document, Episode, ForgetPreview, ForgetResult,
                           MemoryType, WriteReceipt)

from .compare import (PROFILE_STARTS, TOKEN_EXPIRES, WALL_CLOCK, Raised, Run, differences,
                      labels, normalise, normalise_text, text_labels)

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
THEN = datetime(2024, 1, 1, tzinfo=timezone.utc)
#: A run that took two seconds.
RUN = Run(start=NOW, end=NOW + timedelta(seconds=2))


def _claim(value: str, **fields: Any) -> Claim:
    return Claim(subject="user", predicate="lives_in", object=value, **fields)


def _at(value: str) -> dict[str, str]:
    """An instant as `normalise` writes one."""
    return {"__type__": "datetime", "value": value}


def test_the_same_claim_gets_the_same_label_whatever_id_its_store_drew() -> None:
    one, other = _claim("Lisbon"), _claim("Lisbon")
    assert one.id != other.id
    assert (normalise(one.id, labels(one), run=RUN)
            == normalise(other.id, labels(other), run=RUN)
            == "<claim user lives_in Lisbon>")


def test_a_turn_and_a_document_are_labelled_by_what_they_hold() -> None:
    turn = Episode(content="My name is Ada.")
    doc = Document(custom_id="docs/runbook")
    names = labels({"turn": turn, "docs": [doc]})
    assert names == {turn.id: "<turn My name is Ada.>", doc.id: "<document docs/runbook>"}


def test_two_ids_that_would_share_a_label_are_refused() -> None:
    with pytest.raises(ValueError, match="both <claim user lives_in Lisbon>"):
        labels([_claim("Lisbon"), _claim("Lisbon")])


def test_an_id_no_returned_object_names_is_numbered_in_the_order_it_appears() -> None:
    """Two such ids must not share one label, or swapping them would go unseen."""
    claim = _claim("Lisbon")
    names = labels({"sources": ["ep_ffffffffffffffffffff", "ep_0123456789abcdef0123"]},
                   claim)
    assert names == {claim.id: "<claim user lives_in Lisbon>",
                     "ep_ffffffffffffffffffff": "<ep 1>", "ep_0123456789abcdef0123": "<ep 2>"}


def test_an_id_nobody_labelled_is_marked_rather_than_kept() -> None:
    assert normalise("cl_0123456789abcdef0123", {}, run=RUN) == "<unlabelled id>"


def test_an_instant_taken_from_the_clock_during_the_run_is_replaced() -> None:
    """The tools write an instant to the minute, rounding down, so a minute before the
    run still counts."""
    for inside in (NOW, NOW + timedelta(seconds=2), NOW - timedelta(seconds=59)):
        assert normalise(inside, {}, run=RUN) == _at(WALL_CLOCK)


def test_the_token_s_expiry_and_the_profile_s_start_have_markers_of_their_own() -> None:
    assert normalise(NOW + timedelta(seconds=1) + CONFIRM_TTL, {}, run=RUN) == _at(
        TOKEN_EXPIRES)
    assert normalise(NOW + timedelta(seconds=1) - PROFILE_WINDOW, {}, run=RUN) == _at(
        PROFILE_STARTS)


def test_an_instant_off_by_any_other_amount_is_kept() -> None:
    """So a surface that stamps a retirement five hours early, or lets a confirm token
    live two days longer than the library does, fails the comparison."""
    for away in (NOW - timedelta(hours=5), NOW + CONFIRM_TTL + timedelta(days=2),
                 NOW - timedelta(minutes=2), THEN):
        assert normalise(away, {}, run=RUN) == _at(away.isoformat())


def test_an_instant_and_the_same_instant_as_text_are_a_difference() -> None:
    assert differences(normalise(THEN, {}, run=RUN), normalise(THEN.isoformat(), {}, run=RUN)) \
        == [": expected {'__type__': 'datetime', 'value': '2024-01-01T00:00:00+00:00'}, "
            "found '2024-01-01T00:00:00+00:00'"]


def test_an_enum_and_its_plain_value_are_a_difference() -> None:
    """A hosted client that handed back "procedural" where the library hands back
    `MemoryType.PROCEDURAL` would break every caller that compares with the enum."""
    assert differences(normalise(MemoryType.PROCEDURAL, {}, run=RUN),
                       normalise("procedural", {}, run=RUN)) == [
        ": expected {'__type__': 'MemoryType', 'value': 'procedural'}, found 'procedural'"]


def test_a_claim_is_compared_without_its_bookkeeping_and_with_its_state() -> None:
    claim = _claim("Lisbon", valid_from=THEN, recorded_at=NOW, valid_to=NOW)
    claim.meta["salience_base"] = 1.0
    claim.meta[CLOSURE] = [{"at": NOW.timestamp(), "close": "ended", "by": "api"}]
    out = normalise(claim, labels(claim), run=RUN)
    assert "salience_base" not in out["meta"]
    assert out["meta"][CLOSURE] == [{"at": _at(WALL_CLOCK), "close": "ended", "by": "api"}]
    assert (out["state"], out["salience_base"], out["valid_from"]) == (
        "ended", 1.0, _at(THEN.isoformat()))


def test_two_claims_that_differ_in_one_field_still_differ() -> None:
    one = _claim("Lisbon", valid_from=THEN, recorded_at=THEN)
    other = _claim("Lisbon", valid_from=THEN, recorded_at=THEN, confidence=0.5)
    found = differences(normalise(one, labels(one), run=RUN),
                        normalise(other, labels(other), run=RUN))
    assert found == [".confidence: expected 1.0, found 0.5"]


def test_a_receipt_is_compared_without_how_long_the_write_took() -> None:
    fast = normalise(WriteReceipt(latency_ms=0.2), {}, run=RUN)
    slow = normalise(WriteReceipt(latency_ms=9.0), {}, run=RUN)
    assert "latency_ms" not in fast and differences(fast, slow) == []


def test_a_preview_hides_its_token_and_a_result_sorts_what_it_closed() -> None:
    first, second = _claim("Berlin"), _claim("Lisbon")
    names = labels(first, second)
    preview = ForgetPreview(close="retired", matches={first.id: first.text},
                            confirm="signature", expires_at=NOW)
    assert normalise(preview, names, run=RUN)["confirm"] == "<token>"
    both_ways = [normalise(ForgetResult(close="retired", closed=order), names, run=RUN)
                 for order in ([first, second], [second, first])]
    assert differences(*both_ways) == []


@dataclasses.dataclass(frozen=True)
class _WiderPreview(ForgetPreview):
    note: str = ""


@dataclasses.dataclass(frozen=True)
class _WiderResult(ForgetResult):
    note: str = ""


def test_a_field_a_preview_or_a_result_gains_later_is_still_compared() -> None:
    """Only the token and the order of what was closed are set aside, so a field added
    to either class after this was written is compared like any other."""
    for one, other in ((_WiderPreview("retired", {}, "token", NOW, note="kept"),
                        _WiderPreview("retired", {}, "token", NOW, note="dropped")),
                       (_WiderResult("retired", [], note="kept"),
                        _WiderResult("retired", [], note="dropped"))):
        assert differences(normalise(one, {}, run=RUN), normalise(other, {}, run=RUN)) == [
            ".note: expected 'kept', found 'dropped'"]


def test_a_difference_names_its_path_and_both_values() -> None:
    found = differences({"added": [{"object": "Lisbon"}], "count": 1},
                        {"added": [{"object": "Porto"}], "count": 2})
    assert found == [".added[0].object: expected 'Lisbon', found 'Porto'",
                     ".count: expected 1, found 2"]


def test_floats_that_differ_only_in_the_ninth_decimal_are_equal() -> None:
    assert differences(0.548188936245, 0.548188930197) == []
    assert differences(0.5481, 0.5482) == [": expected 0.5481, found 0.5482"]


def test_a_value_of_another_type_is_a_difference_even_when_it_compares_equal() -> None:
    assert differences(1, True) == [": expected 1, found True"]
    assert differences(1, 1.0) == [": expected 1, found 1.0"]


def test_a_missing_key_and_a_missing_item_are_both_reported() -> None:
    assert differences({"a": 1}, {}) == [".a: only in the expected answer, 1"]
    assert differences([1, 2], [1]) == [": 2 item(s) expected, 1 found"]


def test_a_refusal_is_compared_by_its_class_and_its_message() -> None:
    one = normalise(Raised("KeyError", "no claim cl_0123456789abcdef0123"), {}, run=RUN)
    assert one == {"__type__": "Raised", "kind": "KeyError",
                   "message": "no claim <unlabelled id>"}


def test_tool_text_is_labelled_from_the_receipt_lines_and_the_order_of_appearance() -> None:
    texts = ["added 1\n+ [cl_0123456789abcdef0123] user likes jazz",
             "document doc_0123456789abcdef0123: done\nturn id(s): ep_0123456789abcdef0123",
             "Claim 'cl_ffffffffffffffffffff' is not visible here."]
    names = text_labels(texts)
    assert [normalise_text(text, names, run=RUN) for text in texts] == [
        "added 1\n+ [<user likes jazz>] user likes jazz",
        "document <doc 1>: done\nturn id(s): <ep 1>",
        "Claim '<cl 1>' is not visible here."]


def test_two_claims_with_the_same_text_in_tool_replies_are_refused() -> None:
    with pytest.raises(ValueError, match="both <user likes jazz>"):
        text_labels(["+ [cl_0123456789abcdef0123] user likes jazz",
                     "+ [cl_ffffffffffffffffffff] user likes jazz"])


def test_tool_text_marks_each_clock_reading_and_keeps_every_other_instant() -> None:
    text = ("recorded 2026-09-26 12:00Z true from 2024-01-01 00:00Z, as true on "
            "2024-01-31T00:00:00Z, erased at 2026-10-26 12:00Z, expires at "
            "2026-09-26 12:10Z, arrived since 2026-09-19 12:00Z, retired 2026-09-26 07:00Z"
            "\nconfirm: eyJabc.def")
    assert normalise_text(text, {}, run=RUN) == (
        f"recorded {WALL_CLOCK} true from 2024-01-01 00:00Z, as true on "
        f"2024-01-31T00:00:00Z, erased at 2026-10-26 12:00Z, expires at {TOKEN_EXPIRES}, "
        f"arrived since {PROFILE_STARTS}, retired 2026-09-26 07:00Z\nconfirm: <token>")
