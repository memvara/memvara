"""The comparison the parity tests rest on removes only what differs between two stores
for reasons that have nothing to do with the surface, and reports everything else.

A normaliser that removed too much would make every parity test pass whatever the
surfaces returned, so each rule here is paired with a check that a real difference
still shows.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from memvara.types import (CLOSURE, Claim, Document, Episode, ForgetPreview, ForgetResult,
                           WriteReceipt)

from .compare import (WALL_CLOCK, Raised, differences, labels, normalise, normalise_text,
                      text_labels)

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
THEN = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _claim(value: str, **fields: Any) -> Claim:
    return Claim(subject="user", predicate="lives_in", object=value, **fields)


def test_the_same_claim_gets_the_same_label_whatever_id_its_store_drew() -> None:
    one, other = _claim("Lisbon"), _claim("Lisbon")
    assert one.id != other.id
    assert (normalise(one.id, labels(one), near=NOW)
            == normalise(other.id, labels(other), near=NOW)
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
    assert normalise("cl_0123456789abcdef0123", {}, near=NOW) == "<unlabelled id>"


def test_an_instant_near_the_run_is_replaced_and_one_far_from_it_is_kept() -> None:
    assert normalise(NOW - timedelta(days=7), {}, near=NOW) == WALL_CLOCK
    assert normalise(NOW + timedelta(minutes=10), {}, near=NOW) == WALL_CLOCK
    assert normalise(THEN, {}, near=NOW) == THEN.isoformat()
    assert normalise(NOW + timedelta(days=30), {}, near=NOW) != WALL_CLOCK


def test_a_claim_is_compared_without_its_bookkeeping_and_with_its_state() -> None:
    claim = _claim("Lisbon", valid_from=THEN, recorded_at=NOW, valid_to=NOW)
    claim.meta["salience_base"] = 1.0
    claim.meta[CLOSURE] = [{"at": NOW.timestamp(), "close": "ended", "by": "api"}]
    out = normalise(claim, labels(claim), near=NOW)
    assert "salience_base" not in out["meta"]
    assert out["meta"][CLOSURE] == [{"at": WALL_CLOCK, "close": "ended", "by": "api"}]
    assert (out["state"], out["salience_base"], out["valid_from"]) == (
        "ended", 1.0, THEN.isoformat())


def test_two_claims_that_differ_in_one_field_still_differ() -> None:
    one = _claim("Lisbon", valid_from=THEN)
    other = _claim("Lisbon", valid_from=THEN, confidence=0.5)
    found = differences(normalise(one, labels(one), near=NOW),
                        normalise(other, labels(other), near=NOW))
    assert found == [".confidence: expected 1.0, found 0.5"]


def test_a_receipt_is_compared_without_how_long_the_write_took() -> None:
    fast = normalise(WriteReceipt(latency_ms=0.2), {}, near=NOW)
    slow = normalise(WriteReceipt(latency_ms=9.0), {}, near=NOW)
    assert "latency_ms" not in fast and differences(fast, slow) == []


def test_a_preview_hides_its_token_and_a_result_sorts_what_it_closed() -> None:
    first, second = _claim("Berlin"), _claim("Lisbon")
    names = labels(first, second)
    preview = ForgetPreview(close="retired", matches={first.id: first.text},
                            confirm="signature", expires_at=NOW)
    assert normalise(preview, names, near=NOW)["confirm"] == "<token>"
    both_ways = [normalise(ForgetResult(close="retired", closed=order), names, near=NOW)
                 for order in ([first, second], [second, first])]
    assert differences(*both_ways) == []


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
    one = normalise(Raised("KeyError", "no claim cl_0123456789abcdef0123"), {}, near=NOW)
    assert one == {"__type__": "Raised", "kind": "KeyError",
                   "message": "no claim <unlabelled id>"}


def test_tool_text_is_labelled_from_the_receipt_lines_and_the_order_of_appearance() -> None:
    texts = ["added 1\n+ [cl_0123456789abcdef0123] user likes jazz",
             "document doc_0123456789abcdef0123: done\nturn id(s): ep_0123456789abcdef0123",
             "Claim 'cl_ffffffffffffffffffff' is not visible here."]
    names = text_labels(texts)
    assert [normalise_text(text, names, near=NOW) for text in texts] == [
        "added 1\n+ [<user likes jazz>] user likes jazz",
        "document <doc 1>: done\nturn id(s): <ep 1>",
        "Claim '<cl 1>' is not visible here."]


def test_two_claims_with_the_same_text_in_tool_replies_are_refused() -> None:
    with pytest.raises(ValueError, match="both <user likes jazz>"):
        text_labels(["+ [cl_0123456789abcdef0123] user likes jazz",
                     "+ [cl_ffffffffffffffffffff] user likes jazz"])


def test_tool_text_loses_the_clock_and_the_token_and_keeps_every_other_instant() -> None:
    text = ("recorded 2026-09-26 11:59Z true from 2024-01-01 00:00Z, as true on "
            "2024-01-31T00:00:00Z, erased at 2026-10-26 12:00Z\nconfirm: eyJabc.def")
    assert normalise_text(text, {}, near=NOW) == (
        "recorded <wall clock> true from 2024-01-01 00:00Z, as true on "
        "2024-01-31T00:00:00Z, erased at 2026-10-26 12:00Z\nconfirm: <token>")
