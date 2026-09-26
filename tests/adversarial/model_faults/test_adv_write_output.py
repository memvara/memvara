"""The write path against model output that is malformed, invented or enormous.

Whatever the model sends back, a write must keep every turn it was given, keep the facts
the fast path read from the same batch, and leave every claim already stored as it was.
The model is paid for as `docs/INTERNALS.md` says: one extraction call per batch, and one
acquisition call per new predicate spelling, once, until the learned-predicate cap.

A model claim here is a dict with all ten fields of `CLAIM_SCHEMA`. Most rows vary
`team based_in Porto`: `based_in` is an alias of the builtin `lives_in`, so the claim costs
no acquisition call, and its subject is `team`, so its slot is not the user's.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Callable

import pytest

from harness import known_bugs
from memvara.llm import _shape
from memvara.schema import DEFAULT_LEARNED_CAP
from memvara.types import Dispute, MemoryType
from memvara.write import pipeline, pollution

from .handles import (
    FAST_TURN, MODEL_TURN, fates, ledger, raised_in, seed, turns, with_model,
)
from .scripted import Forever, ScriptedModel, Text, Truncated

Make = Callable[..., ScriptedModel]

#: What an acquisition call answers when it reads a spelling as a new predicate that
#: holds many values.
NEW_MANY = Text('{"canonical": null, "cardinality": "many", "volatility": "slow", '
                '"memory_type": "semantic"}')


def claim(subject: str, predicate: str, obj: str, **changes: Any) -> dict[str, Any]:
    """A model claim citing turn 0, with every field of the claim schema."""
    return {"subject": subject, "predicate": predicate, "object": obj, "polarity": 1,
            "memory_type": "semantic", "confidence": 0.9, "source_index": 0,
            "when": None, "amount": None, "unit": None, **changes}


PORTO = claim("team", "based_in", "Porto")


def porto(**changes: Any) -> dict[str, Any]:
    return {**PORTO, **changes}


def without(field: str) -> dict[str, Any]:
    return {key: value for key, value in PORTO.items() if key != field}


def text(*claims: Any) -> Text:
    """The provider's answer holding `claims`, as a shipped backend would receive it."""
    return Text(json.dumps({"claims": list(claims)}))


def objects(mem: Any, *, predicate: str | None = None,
            extractor: str | None = None) -> list[str]:
    """The objects of the live claims under `predicate`, or written by `extractor`."""
    return sorted(c.object for c in mem.get_all()
                  if predicate in (None, c.predicate) and extractor in (None, c.extractor))


def model_claims(mem: Any) -> list[str]:
    return objects(mem, extractor=ScriptedModel.name)


def write(model: ScriptedModel, **options: Any) -> tuple[Any, dict[str, Any], Any]:
    """Seed a store with `model`, then write the fast turn and the model turn in one batch.
    Returns the handle, the ledger taken before the write, and the receipt."""
    mem = with_model(model, **options)
    seed(mem)
    before = ledger(mem)
    return mem, before, mem.add([FAST_TURN, MODEL_TURN])


# -- malformed output ---------------------------------------------------------------------

CUT = '{"claims": [{"subject": "team", "predicate": "based_in", "obj'

MALFORMED = [
    # Text a shipped backend would receive, validated by memvara's own shaping.
    pytest.param(Text("I could not find any durable facts in that turn, sorry."), False,
                 id="text-prose"),
    pytest.param(Text(""), False, id="text-empty"),
    pytest.param(Text(CUT), False, id="text-cut-off-with-no-stop-reason"),
    pytest.param(Truncated(CUT), True, id="text-cut-off-at-the-token-limit"),
    pytest.param(Text(json.dumps([PORTO])), False, id="text-a-top-level-list"),
    pytest.param(Text(json.dumps({"facts": [PORTO]})), False, id="text-no-claims-key"),
    pytest.param(Text(json.dumps({"claims": PORTO})), False, id="text-claims-an-object"),
    pytest.param(Text(json.dumps({"claims": [1, "team based_in Porto", None,
                                             ["team", "based_in", "Porto"]]})),
                 False, id="text-items-that-are-not-objects"),
    pytest.param(text(porto(source_index=7)), False, id="text-cites-turn-7"),
    pytest.param(text(porto(source_index="0")), False, id="text-cites-turn-string"),
    pytest.param(text(porto(source_index=True)), False, id="text-cites-turn-true"),
    pytest.param(text(porto(source_index=-1)), False, id="text-cites-turn-minus-1"),
    pytest.param(text(without("subject")), False, id="text-no-subject"),
    pytest.param(text(porto(predicate="")), False, id="text-empty-predicate"),
    pytest.param(text(porto(object="   ")), False, id="text-blank-object"),
    # Values a backend that does no validation of its own would return.
    pytest.param([], False, id="value-no-claims"),
    pytest.param([porto(source_index="0")], False, id="value-cites-turn-string"),
    pytest.param([porto(source_index=True)], False, id="value-cites-turn-true"),
    pytest.param([porto(source_index=-1)], False, id="value-cites-turn-minus-1"),
    pytest.param([porto(source_index=7)], False, id="value-cites-turn-7"),
    pytest.param([porto(predicate="")], False, id="value-empty-predicate"),
    pytest.param([without("predicate")], False, id="value-no-predicate"),
    pytest.param([porto(object="")], False, id="value-empty-object"),
    pytest.param([porto(object="   ")], False, id="value-blank-object"),
]


@pytest.mark.parametrize("reply, deferred", MALFORMED)
def test_a_reply_with_no_usable_claim_keeps_the_turns_and_touches_nothing(
        scripted: Make, reply: object, deferred: bool) -> None:
    """The turns are stored as given, the fast path's fact from the same batch is kept,
    no model claim is stored, every claim already stored is unchanged, and the write is
    billed one call. A reply the provider cut off at its token limit also marks the batch
    deferred, which is what tells a worker the turn is still owed an extraction."""
    model = scripted(extract=[reply])
    mem, before, receipt = write(model)
    assert turns(mem, receipt) == [FAST_TURN, MODEL_TURN]
    assert objects(mem, predicate="name") == ["Ada"]
    assert model_claims(mem) == []
    assert (receipt.deferred, receipt.unextracted) == (deferred, 1)
    assert receipt.llm_calls == model.count() == 1
    assert set(fates(before, mem).values()) == {"unchanged"}


def test_fields_the_backends_validation_repairs_are_stored_repaired(scripted: Make) -> None:
    """A confidence that overflows to infinity is read as unknown (0.5), a polarity that
    is not -1 is an assertion, an unknown memory type is semantic, and a time or amount
    that is not one is dropped. `write/pipeline.py` and `llm/_shape.py` state each rule."""
    reply = Text('{"claims": [{"subject": "team", "predicate": "based_in", '
                 '"object": "Porto", "polarity": "yes", "memory_type": "bogus", '
                 '"confidence": 1e400, "source_index": 0, "when": 42, "amount": NaN, '
                 '"unit": "kg"}]}')
    model = scripted(extract=[reply])
    mem, before, receipt = write(model)
    [stored] = [c for c in mem.get_all() if c.extractor == ScriptedModel.name]
    turn = mem.store.get_episode(receipt.episode_ids[1])
    assert (stored.object, stored.confidence, stored.polarity) == ("Porto", 0.5, 1)
    assert (stored.memory_type, stored.amount, stored.unit) == (MemoryType.SEMANTIC,
                                                                None, None)
    assert stored.valid_from == turn.ts
    assert receipt.llm_calls == model.count() == 1
    assert set(fates(before, mem).values()) == {"unchanged"}


@pytest.mark.parametrize("field, value, stored_as", [
    pytest.param("confidence", "high", ("confidence", 0.7), id="confidence-text"),
    pytest.param("polarity", "yes", ("polarity", 1), id="polarity-text"),
    pytest.param("memory_type", "bogus", ("memory_type", MemoryType.SEMANTIC),
                 id="memory-type-unknown"),
    pytest.param("amount", float("nan"), ("amount", None), id="amount-nan"),
])
def test_a_garbled_field_from_a_backend_with_no_validation_is_repaired(
        scripted: Make, field: str, value: object, stored_as: tuple[str, object]) -> None:
    """`WritePipeline._claim_from_dict` is the trust boundary for a backend that does not
    shape its output, and repairs each of these fields rather than storing it."""
    model = scripted(extract=[[porto(**{field: value})]])
    mem, before, receipt = write(model)
    [stored] = [c for c in mem.get_all() if c.extractor == ScriptedModel.name]
    attribute, expected = stored_as
    assert stored.object == "Porto" and getattr(stored, attribute) == expected
    assert receipt.llm_calls == model.count() == 1
    assert set(fates(before, mem).values()) == {"unchanged"}


def test_a_time_that_is_not_text_leaves_the_claim_starting_at_its_turn(
        scripted: Make) -> None:
    model = scripted(extract=[[porto(when=42)]])
    mem, _, receipt = write(model)
    [stored] = [c for c in mem.get_all() if c.extractor == ScriptedModel.name]
    assert stored.valid_from == mem.store.get_episode(receipt.episode_ids[1]).ts


# -- invented predicates ---------------------------------------------------------------

INVENTED_TURNS = (
    "Standup moved to Thursdays, office badges are teal now, and desks sit on floor three.",
    "Reminder for everyone: standup stays on Thursdays and badges stay teal on floor three.",
    "New starters: standup is on Thursdays, badges are teal, desks are on floor three.",
)

INVENTED = [claim("team", "zqx_standup_day", "Thursdays"),
            claim("team", "zqx_badge_colour", "teal"),
            claim("team", "zqx_desk_floor", "floor three")]


def test_an_invented_predicate_is_paid_for_once_even_across_a_restart(
        scripted: Make, tmp_path: pathlib.Path) -> None:
    """INTERNALS, `write/pipeline.py`: the answer is "learned, persisted through
    `store.put_spec()` and never asked again, including after a restart and including by
    another process"."""
    path = tmp_path / "s.db"
    model = scripted(extract=[INVENTED, INVENTED], resolve=[NEW_MANY] * 3)
    mem = with_model(model, path)
    first = mem.add(INVENTED_TURNS[0])
    assert first.llm_calls == model.count() == 4
    assert model.counts() == {"extract": 1, "resolve_predicate": 3}
    second = mem.add(INVENTED_TURNS[1])
    assert second.llm_calls == 1 and model.count("resolve_predicate") == 3
    mem.close()
    reopened_model = scripted(extract=[INVENTED])
    reopened = with_model(reopened_model, path)
    third = reopened.add(INVENTED_TURNS[2])
    assert third.llm_calls == reopened_model.count() == 1
    assert reopened_model.count("resolve_predicate") == 0
    reopened.close()


def test_a_model_cannot_end_the_users_value_by_renaming_a_slot(scripted: Make) -> None:
    """The model files Lisbon under a spelling of its own, and then answers that the
    spelling means `lives_in`. The merge is kept, but the claim is stored at 0.4, beside
    the user's Berlin, and Berlin stays live: `write/pollution.py` R4 stores a claim under
    a novel predicate at `min(confidence, 0.4)`, and the reconciler closes an incumbent
    only for a candidate worth at least half of it."""
    model = scripted(
        extract=[[claim("user", "home_base_city", "Lisbon", confidence=0.95)]],
        resolve=[Text('{"canonical": "lives_in", "cardinality": "one", '
                      '"volatility": "slow", "memory_type": "semantic"}')])
    mem = with_model(model)
    ids = seed(mem)
    before = ledger(mem)
    receipt = mem.add("These days home base is Lisbon, most weeks at least.")
    assert "lives_in" in model.calls[1].args["candidates"]
    assert mem.registry.normalize("home_base_city") == "lives_in"
    assert receipt.disputed == [Dispute(ids["berlin"], "user", "lives_in", "Berlin", 1.0,
                                        "Lisbon", 0.4)]
    assert objects(mem, predicate="lives_in") == ["Berlin", "Lisbon"]
    assert fates(before, mem)[ids["berlin"]] == "unchanged"
    assert receipt.llm_calls == model.count() == 2


def test_a_closed_vocabulary_pays_nothing_for_invented_predicates(scripted: Make) -> None:
    """INTERNALS, `closed_vocabulary`: a predicate the registry cannot resolve is refused,
    counted on `receipt.unregistered`, and "costs no model call"."""
    model = scripted(extract=[[*INVENTED, porto()]])
    mem = with_model(model, write_closed_vocabulary=True)
    seed(mem)
    before = ledger(mem)
    receipt = mem.add("The team works from Porto now: standup on Thursdays, badges teal, "
                      "desks on floor three.")
    assert model_claims(mem) == ["Porto"]
    assert receipt.unregistered == 3
    assert receipt.llm_calls == model.count() == 1
    assert set(fates(before, mem).values()) == {"unchanged"}


def test_past_the_learned_cap_an_invented_predicate_costs_no_call(scripted: Make) -> None:
    """One acquisition call per new spelling until the registry has learned
    `DEFAULT_LEARNED_CAP` predicates. After that, INTERNALS: "a novel form folds onto its
    nearest existing predicate ... it costs nothing"."""
    many = [claim("team", f"zqx_rel_{i}", f"item {i}")
            for i in range(DEFAULT_LEARNED_CAP + 5)]
    more = [claim("team", f"zqx_extra_{i}", f"item {i}") for i in range(5)]
    model = scripted(extract=[many, more], resolve=[Forever(NEW_MANY)])
    mem = with_model(model)
    first = mem.add("Every item on the release checklist has its own owner and region.")
    assert first.llm_calls == model.count() == 1 + DEFAULT_LEARNED_CAP
    assert model.count("resolve_predicate") == DEFAULT_LEARNED_CAP
    second = mem.add("Five more items joined the release checklist this morning.")
    assert second.llm_calls == 1
    assert model.count("resolve_predicate") == DEFAULT_LEARNED_CAP


# -- many claims in one reply --------------------------------------------------------------

SNACK_TURN = "The team keeps a snack list for the office kitchen, and it is very long."


def test_five_hundred_claims_in_one_reply_are_stored_from_one_call(scripted: Make) -> None:
    """Every claim is stored, and every one cites the turn it came from."""
    model = scripted(extract=[[claim("team", "likes", f"snack {i}") for i in range(500)]])
    mem = with_model(model)
    seed(mem)
    before = ledger(mem)
    receipt = mem.add(SNACK_TURN)
    [turn] = receipt.episode_ids
    assert len(receipt.added) == 500
    assert {tuple(c.sources) for c in receipt.added} == {(turn,)}
    assert receipt.llm_calls == model.count() == 1
    assert set(fates(before, mem).values()) == {"unchanged"}


def test_a_model_that_restates_one_fact_five_hundred_times_stores_it_once(
        scripted: Make) -> None:
    """Review Focus 1 of the plan: a runaway restatement is one fact, seen 500 times, and
    not 500 rows."""
    model = scripted(extract=[[claim("team", "likes", "snack list")] * 500])
    mem = with_model(model)
    receipt = mem.add(SNACK_TURN)
    assert (len(receipt.added), len(receipt.reinforced)) == (1, 499)
    [stored] = [c for c in mem.get_all() if c.object == "snack list"]
    assert mem.store.get_claim(stored.id).observation_count == 500
    assert receipt.llm_calls == model.count() == 1


# -- bugs these tests found, each pinned until its fix lands --------------------------------
#
# Each test states what the fixed code must do, and raises `known_bugs.Reproduced` only when
# it has seen its bug's own symptom. Any other failure fails the run.

#: What an acquisition call answers when it reads a spelling as a new predicate that holds
#: one value.
NEW_ONE = Text('{"canonical": null, "cardinality": "one", "volatility": "fast", '
               '"memory_type": "semantic"}')


def place(module: Any, function: str) -> tuple[str, str]:
    """A function of a memvara module, in the form `handles.raised_in` answers."""
    return os.path.realpath(module.__file__), function


#: Each item a backend with no validation of its own might return, with the exception
#: `add()` raises for it today, the place in memvara that raises it, and part of its message.
MALFORMED_ITEMS = [
    pytest.param([porto(source_index=[0])], TypeError, (pollution, "guard"),
                 "unhashable type: 'list'", id="source-index-a-list"),
    pytest.param([porto(source_index={"i": 0})], TypeError, (pollution, "guard"),
                 "unhashable type: 'dict'", id="source-index-an-object"),
    pytest.param([porto(polarity=float("inf"))], OverflowError,
                 (pipeline, "_claim_from_dict"), "cannot convert float infinity to integer",
                 id="polarity-infinite"),
    pytest.param([porto(polarity=float("-inf"))], OverflowError,
                 (pipeline, "_claim_from_dict"), "cannot convert float infinity to integer",
                 id="polarity-minus-infinite"),
    pytest.param([porto(confidence=10 ** 400)], OverflowError,
                 (pipeline, "_claim_from_dict"), "int too large to convert to float",
                 id="confidence-a-huge-integer"),
    pytest.param([claim("team", "zqx_office", "Porto", confidence=10 ** 400)], OverflowError,
                 (pollution, "guard"), "int too large to convert to float",
                 id="confidence-a-huge-integer-under-a-new-predicate"),
    pytest.param(["team based_in Porto"], AttributeError, (pollution, "guard"),
                 "'str' object has no attribute 'get'", id="an-item-that-is-text"),
    pytest.param([None], AttributeError, (pollution, "guard"),
                 "'NoneType' object has no attribute 'get'", id="an-item-that-is-null"),
    pytest.param({"claims": [PORTO]}, AttributeError, (pollution, "guard"),
                 "'str' object has no attribute 'get'", id="an-object-instead-of-a-list"),
    pytest.param(None, TypeError, (pollution, "guard"), "'NoneType' object is not iterable",
                 id="nothing-at-all"),
]


@pytest.mark.parametrize("reply, kind, where, words", MALFORMED_ITEMS)
@known_bugs.xfail("B26")
def test_a_malformed_item_is_dropped_and_the_write_returns_a_receipt(
        scripted: Make, reply: object, kind: type[Exception], where: tuple[Any, str],
        words: str) -> None:
    """#303. `WritePipeline._claim_from_dict` is the trust boundary for a backend that
    does not validate its own output, and it says anything malformed is dropped. These
    items reach code before it that assumes a well-formed item, so `add()` raises. The
    turns are stored by then, and the fast path's fact from the same batch is lost."""
    model = scripted(extract=[reply], resolve=[Forever(NEW_MANY)])
    mem = with_model(model)
    seed(mem)
    try:
        receipt = mem.add([FAST_TURN, MODEL_TURN])
    except kind as error:
        if raised_in(error) == place(*where) and words in str(error):
            raise known_bugs.Reproduced(
                f"B26: add() raised {kind.__name__} from {where[1]}: {error}") from error
        raise
    assert turns(mem, receipt) == [FAST_TURN, MODEL_TURN]
    assert objects(mem, predicate="name") == ["Ada"]
    assert receipt.llm_calls == model.count()


@known_bugs.xfail("B27")
def test_an_overflowing_confidence_costs_only_its_own_claim(scripted: Make) -> None:
    """#304. One claim's confidence is an integer of 401 digits in the provider's JSON.
    The backends' shaping raises on it, which the write path catches around the whole call,
    so every claim of the batch is lost and the batch is deferred. `finite_amount` in
    `llm/_shape.py` was hardened against this for `amount` and describes the failure."""
    huge = "1" + "0" * 400
    moved = json.dumps(claim("team", "moved_in", "summer", confidence=0.5)).replace(
        '"confidence": 0.5', f'"confidence": {huge}')
    reply = Text('{"claims": [' + json.dumps(PORTO) + ", " + moved + "]}")
    model = scripted(extract=[reply], resolve=[Forever(NEW_MANY)])
    mem = with_model(model)
    receipt = mem.add(MODEL_TURN)
    if receipt.deferred and model_claims(mem) == [] and len(model.failures) == 1:
        [(method, error)] = model.failures
        if (method, type(error)) == ("extract", OverflowError) and (
                raised_in(error) == place(_shape, "clamp_confidence")):
            raise known_bugs.Reproduced(
                f"B27: shaping raised OverflowError in clamp_confidence ({error}), and the "
                "whole batch was deferred")
    assert not receipt.deferred
    assert "Porto" in model_claims(mem)


@pytest.mark.parametrize("item", [
    pytest.param(claim("team", "zqx_office_hub", "Porto", source_index=5),
                 id="no-provenance"),
    pytest.param(claim("team", "zqx_office_hub", "Reykjavik harbour"), id="ungrounded"),
    pytest.param(claim("team", "zqx_office_hub", ""), id="empty-object"),
])
@known_bugs.xfail("B28")
def test_a_dropped_claim_costs_no_acquisition_and_teaches_nothing(
        scripted: Make, tmp_path: pathlib.Path, item: dict[str, Any]) -> None:
    """#305. The pollution guard and the closed vocabulary run before acquisition, so that
    a refused claim's spelling is never paid for or learned. A claim the trust boundary
    drops afterwards, for having no source turn, no grounding or no object, has its
    predicate acquired first: one model call, and a learned predicate kept in the store."""
    path = tmp_path / "s.db"
    model = scripted(extract=[[item]], resolve=[Forever(NEW_ONE)])
    mem = with_model(model, path)
    mem.add(MODEL_TURN)
    stored = [c.object for c in mem.get_all() if c.predicate == "zqx_office_hub"]
    mem.close()
    reopened = with_model(scripted(), path)
    kept = reopened.registry.known("zqx_office_hub")
    reopened.close()
    assert stored == []  # the claim itself is dropped, as it should be
    asked = [call.args["surface"] for call in model.calls
             if call.method == "resolve_predicate"]
    if asked == ["zqx_office_hub"] and kept:
        raise known_bugs.Reproduced(
            "B28: the dropped claim's predicate cost an acquisition call and is learned in "
            "the store")
    assert asked == []
    assert not kept


@known_bugs.xfail("B29")
def test_a_claim_with_no_subject_is_dropped_not_filed_under_the_user(scripted: Make) -> None:
    """#306. The backends' shaping drops a claim with no subject. The pipeline files one
    from a backend with no validation under the user, where it ends the user's own value."""
    item = {"predicate": "lives_in", "object": "Lisbon", "source_index": 0,
            "confidence": 0.9}
    model = scripted(extract=[[item]])
    mem = with_model(model)
    ids = seed(mem)
    before = ledger(mem)
    mem.add("These days home is Lisbon, and it has been for a while.")
    filed = [(c.subject, c.object) for c in mem.get_all() if c.predicate == "lives_in"]
    berlin = fates(before, mem)[ids["berlin"]]
    if filed == [("user", "Lisbon")] and berlin == "ended":
        raise known_bugs.Reproduced(
            "B29: a claim with no subject was filed under the user and ended the user's "
            "Berlin")
    assert berlin == "unchanged"
    assert ("user", "Lisbon") not in filed


@known_bugs.xfail("B29")
def test_an_object_that_is_not_text_is_dropped_not_stored_as_python_text(
        scripted: Make) -> None:
    """#306. An object that is a list is stored as the text of the list."""
    model = scripted(extract=[[porto(object=["Porto"])]])
    mem = with_model(model)
    mem.add(MODEL_TURN)
    if model_claims(mem) == ["['Porto']"]:
        raise known_bugs.Reproduced("B29: the list object was stored as the text ['Porto']")
    assert model_claims(mem) == []
