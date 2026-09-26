"""Proposals from a model to retire or erase a claim that is already stored.

A model may report that the world changed, and so end a stored fact. It may not do more:
INTERNALS invariant 1 says "A model still cannot retire or erase anything, because a
proposed end is a retraction with `close="ended"`", and `docs/claude/write-pipeline.md`
says "a model can end a memory but never retire or erase one, and a replacement the
reconciler does not accept leaves the old memory live". Retired means the record was
wrong, and erased means the bytes are gone; both are the caller's decisions.

Every test checks what happened to each claim that was stored before the model spoke, with
`handles.fates`, whose own test shows it can see an end, a retirement and an erasure.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from harness import known_bugs
from memvara.store import SQLiteStore
from memvara.types import RefusedProposal, closure_reasons

from .handles import agentic, claim, fates, ledger, seed, with_model, without_model
from .scripted import Answer, Forever, ScriptedModel, Text

Make = Callable[..., ScriptedModel]

LEAVING_BERLIN = "Berlin is behind me now, the flat there is gone."
IN_LISBON = "These days home is Lisbon, and it has been for a while."

#: The six tools an agentic run is offered, in the order it is offered them.
TOOLS = ["search_memories", "get_claim", "propose_claim", "propose_end",
         "propose_supersede", "propose_link"]


def read(claim_id: str) -> tuple[str, dict[str, Any]]:
    return ("get_claim", {"claim_id": claim_id})


def end(claim_id: str, reason: str = "moved away") -> tuple[str, dict[str, Any]]:
    return ("propose_end", {"claim_id": claim_id, "reason": reason, "source_index": 0})


def supersede(claim_id: str, predicate: str, obj: str) -> tuple[str, dict[str, Any]]:
    return ("propose_supersede", {
        "claim_id": claim_id, "reason": "a new value", "subject": "user",
        "predicate": predicate, "object": obj, "source_index": 0, "confidence": 0.9,
        "memory_type": "semantic", "valid_from": None, "amount": None, "unit": None})


# -- single-call extraction ------------------------------------------------------------


@pytest.mark.parametrize("extra", [
    pytest.param({}, id="plain"),
    pytest.param({"close": "retired"}, id="asks-to-retire"),
    pytest.param({"state": "retired", "invalidated_at": "2020-01-01T00:00:00+00:00"},
                 id="claims-it-was-retired"),
    pytest.param({"erase": True, "purge": True}, id="asks-to-erase"),
])
def test_a_retraction_from_the_model_ends_the_fact_and_never_retires_it(
        scripted: Make, extra: dict[str, Any]) -> None:
    """INTERNALS, `write/reconcile.py`: the matches of a retraction "are **ended**, not
    retired". No key a model adds to its claim reaches the closure."""
    model = scripted(extract=[[claim("user", "lives_in", "Berlin", polarity=-1, **extra)]])
    mem = with_model(model)
    ids = seed(mem)
    before = ledger(mem)
    receipt = mem.add(LEAVING_BERLIN)
    assert fates(before, mem) == {ids["berlin"]: "ended", ids["tea"]: "unchanged",
                                  ids["acme"]: "unchanged"}
    assert [c.id for c in receipt.ended] == [ids["berlin"]] and receipt.retired == []
    assert ids["berlin"] in [c.id for c in mem.history("user", "lives_in")]


def test_a_new_value_from_the_model_ends_the_old_one_and_never_retires_it(
        scripted: Make) -> None:
    model = scripted(extract=[[claim("user", "lives_in", "Lisbon")]])
    mem = with_model(model)
    ids = seed(mem)
    before = ledger(mem)
    receipt = mem.add(IN_LISBON)
    assert fates(before, mem) == {ids["berlin"]: "ended", ids["tea"]: "unchanged",
                                  ids["acme"]: "unchanged"}
    assert [c.object for c in mem.get_all() if c.predicate == "lives_in"] == ["Lisbon"]
    assert receipt.retired == []


def test_a_model_cannot_clear_a_whole_slot_with_an_empty_retraction(scripted: Make) -> None:
    """To the reconciler, a retraction with no object clears the whole slot. From a model
    it never gets that far: `WritePipeline._claim_from_dict` drops a claim with no
    object."""
    model = scripted(extract=[[claim("user", "likes", "", polarity=-1)]])
    mem = with_model(model)
    seed(mem)
    before = ledger(mem)
    receipt = mem.add("None of my old tastes hold any more, honestly.")
    assert set(fates(before, mem).values()) == {"unchanged"}
    assert receipt.unextracted == 1


# -- agentic extraction ------------------------------------------------------------------


def test_an_end_the_model_proposes_after_reading_the_claim_ends_it_with_its_reason(
        scripted: Make) -> None:
    model = scripted()
    mem = agentic(model)
    ids = seed(mem)
    before = ledger(mem)
    model.queue("run_tools", Answer(calls=(read(ids["berlin"]),)),
                Answer(calls=(end(ids["berlin"], "moved to Lisbon"),)), Answer("done"))
    receipt = mem.add(LEAVING_BERLIN)
    assert fates(before, mem) == {ids["berlin"]: "ended", ids["tea"]: "unchanged",
                                  ids["acme"]: "unchanged"}
    assert closure_reasons(mem.store.get_claim(ids["berlin"])) == [
        ("ended", "moved to Lisbon")]
    assert receipt.proposals_refused == []
    assert receipt.llm_calls == model.count() == 3


def test_an_end_for_a_claim_the_model_never_read_is_refused(scripted: Make) -> None:
    model = scripted()
    mem = agentic(model)
    ids = seed(mem)
    before = ledger(mem)
    model.queue("run_tools", Answer(calls=(end(ids["berlin"]),)), Answer("done"))
    receipt = mem.add(LEAVING_BERLIN)
    assert receipt.proposals_refused == [
        RefusedProposal("propose_end", ids["berlin"], "not_read")]
    assert set(fates(before, mem).values()) == {"unchanged"}


def test_an_end_for_a_broader_scopes_claim_is_refused(scripted: Make) -> None:
    """A user-wide fact answers in every session, so a write inside one session may read
    it but not end it (`agentic._Session._closable`)."""
    model = scripted()
    mem = agentic(model)
    ids = seed(mem)
    before = ledger(mem)
    model.queue("run_tools", Answer(calls=(read(ids["berlin"]),)),
                Answer(calls=(end(ids["berlin"]),)), Answer("done"))
    receipt = mem.add(LEAVING_BERLIN, session="s1")
    assert receipt.proposals_refused == [
        RefusedProposal("propose_end", ids["berlin"], "broader_scope")]
    assert set(fates(before, mem).values()) == {"unchanged"}


def test_a_replacement_the_reconciler_does_not_accept_changes_nothing(
        scripted: Make) -> None:
    """INTERNALS: "a replacement whose new value closed nothing is reported as
    `not_applied` with the named memory left live"."""
    model = scripted()
    mem = agentic(model)
    ids = seed(mem)
    before = ledger(mem)
    model.queue("run_tools", Answer(calls=(read(ids["berlin"]),)),
                Answer(calls=(supersede(ids["berlin"], "likes", "Lisbon trams"),)),
                Answer("done"))
    receipt = mem.add("The Lisbon trams are the best part of living there now.")
    assert receipt.proposals_refused == [
        RefusedProposal("propose_supersede", ids["berlin"], "not_applied")]
    assert "Lisbon trams" in [c.object for c in mem.get_all()]
    assert set(fates(before, mem).values()) == {"unchanged"}


def test_a_replacement_the_reconciler_accepts_ends_the_named_claim(scripted: Make) -> None:
    model = scripted()
    mem = agentic(model)
    ids = seed(mem)
    before = ledger(mem)
    model.queue("run_tools", Answer(calls=(read(ids["berlin"]),)),
                Answer(calls=(supersede(ids["berlin"], "lives_in", "Lisbon"),)),
                Answer("done"))
    receipt = mem.add(IN_LISBON)
    assert receipt.proposals_refused == []
    assert fates(before, mem) == {ids["berlin"]: "ended", ids["tea"]: "unchanged",
                                  ids["acme"]: "unchanged"}
    assert [c.object for c in mem.get_all() if c.predicate == "lives_in"] == ["Lisbon"]


def test_an_agentic_model_learns_nothing_about_another_users_claim_and_cannot_end_it(
        scripted: Make) -> None:
    """Review Focus 4 of the plan. The answer for another user's id is the answer for an
    id that never existed, so the tool cannot be used to learn that an id exists."""
    model = scripted()
    mem = agentic(model)
    seed(mem)
    other = mem.remember("user", "lives_in", "Oslo", user="u2").added[0].id
    before = ledger(mem)
    model.queue("run_tools",
                Answer(calls=(read(other), read("cl_0000000000000000"))),
                Answer(calls=(end(other),)), Answer("done"))
    receipt = mem.add(LEAVING_BERLIN)
    hidden = "No stored memory with that id is visible to this write."
    assert model.tool_results[:2] == [hidden, hidden]
    assert receipt.proposals_refused == [RefusedProposal("propose_end", other, "not_read")]
    assert set(fates(before, mem).values()) == {"unchanged"}


def test_no_tool_retires_or_erases_and_calling_one_ends_the_run(scripted: Make) -> None:
    """A tool that was not offered is an answer the loop cannot use. Two in a row send
    the batch to single-call extraction, which here finds nothing."""
    model = scripted(extract=[[]])
    mem = agentic(model)
    ids = seed(mem)
    before = ledger(mem)
    model.queue("run_tools",
                Answer(calls=(("erase_claim", {"claim_id": ids["berlin"]}),)),
                Answer(calls=(("retire_claim", {"claim_id": ids["berlin"]}),)))
    receipt = mem.add(LEAVING_BERLIN)
    assert model.runs[0]["tools"] == TOOLS
    assert receipt.agentic_fallback == "malformed"
    assert receipt.llm_calls == model.count() == 3
    assert set(fates(before, mem).values()) == {"unchanged"}


# -- replacement advice, and the preview before a destructive write --------------------------

REPLACES = Text('{"same_thing": true, "same_property": true, "newer_value": true, '
                '"replaces": true}')


def test_replacement_advice_that_says_replace_closes_nothing(scripted: Make) -> None:
    """INTERNALS: replacement advice "closes nothing". The neighbours are written through
    a handle with no model, so that writing them asks no judge."""
    store = SQLiteStore(":memory:")
    plain = without_model(store)
    plain.remember("user", "drinks", "green tea every morning")
    plain.remember("user", "orders", "green tea at the cafe")
    model = scripted(judge=[Forever(REPLACES)])
    mem = with_model(model, store=store, advise_replacements=True)
    before = ledger(mem)
    receipt = mem.remember("user", "prefers", "green tea with lemon")
    assert sorted(c.object for c in receipt.may_replace) == [
        "green tea at the cafe", "green tea every morning"]
    assert set(fates(before, mem).values()) == {"unchanged"}
    assert receipt.llm_calls == model.count() == 2


def test_a_forget_preview_asks_no_model_so_a_model_cannot_widen_what_is_retired(
        scripted: Make) -> None:
    """`forget_matching` previews with a plain read (`**PLAIN_READ` in `core.py`), so a
    rewrite that would search for every other stored fact is never asked, and confirming
    retires only what the caller's own query matched."""
    rewrite = Text('{"queries": ["Berlin", "Acme", "works at"], "date_range": null}')
    model = scripted(chat=[Forever(rewrite)])
    mem = with_model(model)
    ids = seed(mem)
    before = ledger(mem)
    preview = mem.forget_matching("green tea", close="retired", k=1)
    assert list(preview.matches) == [ids["tea"]]
    mem.forget_matching("green tea", close="retired", k=1, confirm=preview.confirm)
    # The caller asked for this retirement; nothing else moved.
    assert fates(before, mem) == {ids["tea"]: "retired", ids["berlin"]: "unchanged",
                                  ids["acme"]: "unchanged"}
    assert model.count() == 0


# -- a bug these tests found, pinned until its fix lands -------------------------------------


@known_bugs.xfail("B30")
def test_a_model_retraction_below_half_the_incumbents_confidence_ends_nothing(
        scripted: Make) -> None:
    """#307. The reconciler closes an incumbent only for a candidate worth at least half of
    it (`reconcile.AUTHORITY_SHARE`), so a model's new value at 0.05 is stored beside the
    user's Berlin, asserted at 1.0. `Reconciler._retract` skips that rule, reasoning that
    every negative comes from the fast path or from `remember()`. The model tier produces
    negatives too, so a retraction at 0.05 ends Berlin."""
    model = scripted(extract=[[claim("user", "lives_in", "Berlin", polarity=-1,
                                     confidence=0.05)]])
    mem = with_model(model)
    ids = seed(mem)
    before = ledger(mem)
    receipt = mem.add(LEAVING_BERLIN)
    berlin = fates(before, mem)[ids["berlin"]]
    if berlin == "ended" and [c.id for c in receipt.ended] == [ids["berlin"]]:
        raise known_bugs.Reproduced(
            "B30: a model retraction at confidence 0.05 ended the Berlin the user asserted "
            "at 1.0")
    assert berlin == "unchanged"
