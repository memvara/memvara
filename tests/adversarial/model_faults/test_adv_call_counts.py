"""How many model calls each operation makes, against what `docs/INTERNALS.md` states.

The design's claim is that the model is called rarely, so, as INTERNALS' testing section
says, "a test that does not count calls does not test the design". Each row below sets a
store up without counting, runs one operation, and compares the calls that operation made,
by method, with the number INTERNALS gives. The row's docstring quotes the sentence it
checks. For a write, `receipt.llm_calls` must equal the calls counted.

The model answers every call properly here, so each count is the cost of an ordinary
operation, not of a failure.
"""

from __future__ import annotations

import pathlib
from typing import Any, Callable

import pytest

from memvara.store import SQLiteStore
from memvara.types import WriteReceipt

from .handles import (
    LONG, MODEL_TURN, NEW_MANY, PORTO, claim, with_model, without_model,
)
from .scripted import Answer, Forever, ScriptedModel, Text

Make = Callable[..., ScriptedModel]
Row = Callable[[Make, pathlib.Path], tuple[ScriptedModel, Callable[[], Any]]]

#: One reply that the query rewrite, the synthesis and the selector can each read.
CHAT = Text('{"queries": [], "date_range": null, "synthesis": "The notes mention Lisbon.", '
            '"kept": []}')

KEEPS = Text('{"same_thing": false, "same_property": false, "newer_value": false, '
             '"replaces": false}')

#: Turns the salience gate passes and the fast path does not read, so they reach the model.
OTHER_TURNS = ("The build now uses eight threads and finishes in about four minutes.",
               "Our deploy target moved to the Frankfurt cluster at the start of the month.")

INVENTED = claim("team", "zqx_office_hub", "Porto")


def answering(scripted: Make, extract: object = None) -> ScriptedModel:
    """A model that answers every call properly, each method with one reply, forever."""
    return scripted(extract=[Forever([PORTO] if extract is None else extract)],
                    resolve=[Forever(NEW_MANY)], chat=[Forever(CHAT)],
                    judge=[Forever(KEEPS)])


# -- the write path ---------------------------------------------------------------------


def add_a_turn_the_fast_path_reads(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS, tiers 0 and 1: "(no LLM)"; the fast path handles "I live in X"."""
    model = answering(scripted)
    mem = with_model(model)
    return model, lambda: mem.add("I live in Berlin.")


def add_a_turn_the_gate_drops(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS, tier 1: "`SalienceGate` drops turns carrying no durable fact"."""
    model = answering(scripted)
    mem = with_model(model)
    return model, lambda: mem.add("thanks!")


def add_a_turn_already_stored(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS, tier 0: "store the episode; skip content-hash duplicates"."""
    model = answering(scripted)
    mem = with_model(model)
    mem.add(MODEL_TURN)
    return model, lambda: mem.add(MODEL_TURN)


def add_three_turns_that_reach_the_model(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS, tier 2: the turns that survived tiers 0 and 1 "are batched into a
    single `llm.extract(...)` call"."""
    model = answering(scripted)
    mem = with_model(model)
    return model, lambda: mem.add([MODEL_TURN, *OTHER_TURNS])


def add_a_claim_under_a_new_predicate(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS, tier 2: "Unknown predicates trigger one `llm.resolve_predicate(...)` per
    *new surface form*"."""
    model = answering(scripted, [INVENTED])
    mem = with_model(model)
    return model, lambda: mem.add(MODEL_TURN)


def the_same_predicate_in_the_next_add(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS, tier 2: the answer is "cached via `registry.learn_alias` /
    `registry.learn` ... so it is never asked again"."""
    model = answering(scripted, [INVENTED])
    mem = with_model(model)
    mem.add(MODEL_TURN)
    return model, lambda: mem.add("The Porto office is where the team sits these days.")


def the_same_predicate_after_reopening_the_store(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS, tier 2: never asked again, "including after a restart, and including by
    another process"."""
    first = with_model(answering(scripted, [INVENTED]), tmp / "s.db")
    first.add(MODEL_TURN)
    first.close()
    model = answering(scripted, [INVENTED])
    mem = with_model(model, tmp / "s.db")
    return model, lambda: mem.add("The Porto office is where the team sits these days.")


def a_new_predicate_under_a_closed_vocabulary(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS, `closed_vocabulary`: a refused predicate "is never learned and costs no
    model call"."""
    model = answering(scripted, [INVENTED])
    mem = with_model(model, write_closed_vocabulary=True)
    return model, lambda: mem.add(MODEL_TURN)


def remember_a_fact(scripted: Make, tmp: pathlib.Path) -> Any:
    """`WritePipeline.assert_claim`: "Never consults a model, by construction.\""""
    model = answering(scripted)
    mem = with_model(model)
    return model, lambda: mem.remember("user", "lives_in", "Berlin")


def remember_with_replacement_advice(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS: the judge is asked "about up to `ADVISORY_CANDIDATES` of the nearest live
    claims in *other* slots", and `ADVISORY_CANDIDATES` is 3. Four such claims stand
    here, written through a handle with no model so that writing them asks nothing."""
    store = SQLiteStore(":memory:")
    plain = without_model(store)
    for predicate, obj in (("drinks", "green tea every morning"),
                           ("orders", "green tea at the cafe"),
                           ("brews", "green tea at home"),
                           ("buys", "green tea in bulk")):
        plain.remember("user", predicate, obj)
    model = answering(scripted)
    mem = with_model(model, store=store, advise_replacements=True)
    return model, lambda: mem.remember("user", "prefers", "green tea with lemon")


def reextract_a_stored_turn(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS: `reextract()` "is `add()` with tier 0 removed, for turns already in the
    store". The turn is stored by a handle with no model, so nothing read it."""
    store = SQLiteStore(":memory:")
    without_model(store).add(MODEL_TURN)
    model = answering(scripted)
    mem = with_model(model, store=store)
    return model, lambda: mem.reextract()


def reextract_the_same_turn_again(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS: "An episode that already has claims is skipped and counted on
    `receipt.already_extracted`"."""
    store = SQLiteStore(":memory:")
    without_model(store).add(MODEL_TURN)
    model = answering(scripted)
    mem = with_model(model, store=store)
    mem.reextract()
    return model, lambda: mem.reextract()


def a_long_turn_with_extraction_chunks(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS, `extraction_chunks`: "each piece of a long turn is a call of its own,
    so `llm_calls` counts one per piece"."""
    model = answering(scripted)
    mem = with_model(model, write_extraction_chunks=True)
    return model, lambda: mem.add(LONG)


def an_agentic_run_that_proposes_one_claim(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS, `agentic_extraction`: "Every request the run sent is billed in
    `llm_calls`". Two requests: one proposes the claim, one stops."""
    model = answering(scripted)
    proposal = {key: value for key, value in PORTO.items()
                if key not in ("polarity", "when")} | {"valid_from": None}
    model.queue("run_tools", Answer(calls=(("propose_claim", proposal),)), Answer("done"))
    mem = with_model(model, write_agentic_extraction=True)
    return model, lambda: mem.add(MODEL_TURN)


# -- the read path -------------------------------------------------------------------------


def stored(model: ScriptedModel, **options: Any) -> Any:
    """A handle with the model whose store already holds two facts and two turns, written
    with no model call."""
    mem = with_model(model, **options)
    mem.remember("user", "likes", "Lisbon trams")
    mem.remember("user", "visited", "Lisbon in spring")
    for day in range(2):
        mem.add(f"We talked about the Lisbon trip again on day {day}.", role="system")
    return mem


def search(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS invariant 1: `query_rewrite` is "one chat call before retrieval", on by
    default when the configured model can chat."""
    model = answering(scripted)
    mem = stored(model)
    return model, lambda: mem.search("Lisbon trip")


def search_without_a_rewrite(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS invariant 1: "`search(query_rewrite=False)` is the per-call opt-out"."""
    model = answering(scripted)
    mem = stored(model)
    return model, lambda: mem.search("Lisbon trip", query_rewrite=False)


def search_for_no_results(scripted: Make, tmp: pathlib.Path) -> Any:
    """`HybridRetriever.search`: "A plain read with `k <= 0` returns nothing, makes no call
    and reports no rewrite"."""
    model = answering(scripted)
    mem = stored(model)
    return model, lambda: mem.search("Lisbon trip", k=0)


def recall_with_a_summary(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS invariant 1: the rewrite's one call, and `synthesis`, "one chat call after
    `recall(synthesize=True)` has rendered its notes"."""
    model = answering(scripted)
    mem = stored(model)
    return model, lambda: mem.recall("Lisbon trip", synthesize=True)


def recall_with_a_summary_on_an_empty_store(scripted: Make, tmp: pathlib.Path) -> Any:
    """`Memvara.recall`: "A recall that found no notes makes no call and stays empty";
    the rewrite before the search is still one call."""
    model = answering(scripted)
    mem = with_model(model)
    return model, lambda: mem.recall("Lisbon trip", synthesize=True)


def a_ranked_recall(scripted: Make, tmp: pathlib.Path) -> Any:
    """INTERNALS invariant 1: `ranked` is "one chat call per read over the turns", beside
    the rewrite's one call."""
    model = answering(scripted)
    mem = stored(model, ranked=True)
    return model, lambda: mem.recall("Lisbon trip", ranked=True, include_episodes=True)


def a_forget_preview(scripted: Make, tmp: pathlib.Path) -> Any:
    """`docs/claude/retrieval.md`: a preview before a destructive write is a read that
    "passes `**memvara.select.PLAIN_READ`"."""
    model = answering(scripted)
    mem = stored(model)
    return model, lambda: mem.forget_matching("Lisbon trams", close="retired")


ROWS: list[Any] = [
    pytest.param(add_a_turn_the_fast_path_reads, {}, id="add-a-turn-the-fast-path-reads"),
    pytest.param(add_a_turn_the_gate_drops, {}, id="add-thanks"),
    pytest.param(add_a_turn_already_stored, {}, id="add-a-turn-already-stored"),
    pytest.param(add_three_turns_that_reach_the_model, {"extract": 1},
                 id="add-three-turns-that-reach-the-model"),
    pytest.param(add_a_claim_under_a_new_predicate,
                 {"extract": 1, "resolve_predicate": 1}, id="add-under-a-new-predicate"),
    pytest.param(the_same_predicate_in_the_next_add, {"extract": 1},
                 id="the-same-predicate-in-the-next-add"),
    pytest.param(the_same_predicate_after_reopening_the_store, {"extract": 1},
                 id="the-same-predicate-after-reopening"),
    pytest.param(a_new_predicate_under_a_closed_vocabulary, {"extract": 1},
                 id="a-new-predicate-under-a-closed-vocabulary"),
    pytest.param(remember_a_fact, {}, id="remember"),
    pytest.param(remember_with_replacement_advice, {"judge_replacement": 3},
                 id="remember-with-replacement-advice"),
    pytest.param(reextract_a_stored_turn, {"extract": 1}, id="reextract-a-stored-turn"),
    pytest.param(reextract_the_same_turn_again, {}, id="reextract-the-same-turn-again"),
    pytest.param(a_long_turn_with_extraction_chunks, {"extract": 3},
                 id="a-long-turn-with-extraction-chunks"),
    pytest.param(an_agentic_run_that_proposes_one_claim, {"run_tools": 2},
                 id="an-agentic-run-that-proposes-one-claim"),
    pytest.param(search, {"chat": 1}, id="search"),
    pytest.param(search_without_a_rewrite, {}, id="search-without-a-rewrite"),
    pytest.param(search_for_no_results, {}, id="search-for-no-results"),
    pytest.param(recall_with_a_summary, {"chat": 2}, id="recall-with-a-summary"),
    pytest.param(recall_with_a_summary_on_an_empty_store, {"chat": 1},
                 id="recall-with-a-summary-on-an-empty-store"),
    pytest.param(a_ranked_recall, {"chat": 2}, id="a-ranked-recall"),
    pytest.param(a_forget_preview, {}, id="a-forget-preview"),
]


@pytest.mark.parametrize("row, expected", ROWS)
def test_each_operation_makes_the_model_calls_internals_states(
        scripted: Make, tmp_path: pathlib.Path, row: Row,
        expected: dict[str, int]) -> None:
    model, operation = row(scripted, tmp_path)
    before = model.counts()
    result = operation()
    made = model.counts() - before
    assert dict(made) == expected
    if isinstance(result, WriteReceipt):
        assert result.llm_calls == sum(made.values())
