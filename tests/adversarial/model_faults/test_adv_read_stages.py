"""The read path when a model stage fails: query rewrite, synthesis and ranking.

INTERNALS invariant 1: each of the three read stages "records its outcome ... on the
result, and serves the plain read on every outcome but `applied`". So a read whose stage
failed must equal, byte for byte, the read a store with no model serves. Two lines are
the documented exceptions, both in `Memvara.recall`'s docstring: a `synthesize=True` block
starts with a line naming why no summary was written, and a `ranked=True` block ends with
a line naming why the model did not rank.

Every test reads one store through two handles: one whose model is scripted to fail, and
one with no model. A `search()` compared field by field passes `known_at`, one instant
after every write, because recency is measured at `known_at`, and two present-tense reads
a moment apart would otherwise differ in their scores' last digits.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import pytest

from memvara import EpisodeResult, Memvara
from memvara.server import MemvaraMCPServer
from memvara.store import SQLiteStore
from memvara.types import utcnow

from .handles import USER, rendered, tool_text, with_model, without_model
from .scripted import (
    APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, Late,
    RateLimitError, ScriptedModel, Text, Truncated,
)

Make = Callable[..., ScriptedModel]

QUERY = "Lisbon trip"

FACTS = [("lives_in", "Berlin"), ("likes", "Lisbon trams"), ("visited", "Lisbon in spring")]

TURN = "We talked about the Lisbon trip again on day {day}: the tram, the pastries, the tiles."


def pair(model: ScriptedModel) -> tuple[Memvara, Memvara]:
    """One store, written with no model, and two handles on it: the first has no model,
    the second has `model` for its rewrite, its synthesis and its selector."""
    store = SQLiteStore(":memory:")
    plain = without_model(store)
    for predicate, obj in FACTS:
        plain.remember("user", predicate, obj)
    for day in range(8):
        plain.add(TURN.format(day=day), role="system")
    return plain, with_model(model, store=store, ranked=True)


MALFORMED = ("fallback", "malformed", None)

REWRITE_FAILURES = [
    # Before its deadline an SDK timeout is an error: `timeout` means the call ended after
    # it (`memvara.select.chat.ChatFailed`).
    pytest.param(APITimeoutError(), ("fallback", "error", None), id="sdk-timeout-at-once"),
    pytest.param(Late(APITimeoutError(), 10.5), ("fallback", "timeout", None),
                 id="sdk-timeout-after-the-deadline"),
    pytest.param(TimeoutError("read timed out"), ("fallback", "timeout", None),
                 id="python-timeout"),
    pytest.param(RateLimitError(), ("fallback", "provider", 429), id="rate-limit-429"),
    pytest.param(APIStatusError("unavailable", status_code=503),
                 ("fallback", "provider", 503), id="unavailable-503"),
    pytest.param(AuthenticationError(), ("key_rejected", None, 401), id="rejected-key-401"),
    pytest.param(APIConnectionError("connection reset"), ("fallback", "error", None),
                 id="connection-reset"),
    pytest.param(Late(Text('{"queries": ["Lisbon holiday"], "date_range": null}'), 11.0),
                 ("fallback", "timeout", None), id="a-good-answer-after-the-deadline"),
    pytest.param(Text("Here are some other ways to ask about that."), MALFORMED,
                 id="prose"),
    pytest.param(Text(""), MALFORMED, id="empty"),
    pytest.param(Text('{"queries": ["Lisbon holi'), MALFORMED, id="cut-off-json"),
    pytest.param(Truncated('{"queries": ["Lisbon holi'), MALFORMED,
                 id="cut-off-at-the-limit"),
    pytest.param(Text('["Lisbon holiday"]'), MALFORMED, id="a-list-not-an-object"),
    pytest.param(Text('{"queries": "Lisbon holiday"}'), MALFORMED, id="queries-as-text"),
    pytest.param(Text('{"queries": [], "date_range": '
                      '{"from": "2026-03-10", "to": "2026-03-01"}}'), MALFORMED,
                 id="a-range-that-ends-before-it-starts"),
    pytest.param(Text('{"queries": [], "date_range": {"from": "last week", "to": "today"}}'),
                 MALFORMED, id="a-range-in-words"),
]


def outcome(stage: Any) -> tuple[Any, ...]:
    return (stage.outcome, stage.reason, stage.status)


# -- query rewrite -----------------------------------------------------------------------


@pytest.mark.parametrize("failure, expected", REWRITE_FAILURES)
def test_a_failed_rewrite_serves_the_recall_a_store_with_no_model_serves(
        scripted: Make, failure: object, expected: tuple[Any, ...]) -> None:
    model = scripted(chat=[failure])
    plain, mem = pair(model)
    got = mem.recall(QUERY, include_episodes=True, with_ids=True)
    assert got.text == plain.recall(QUERY, include_episodes=True)
    assert "Lisbon trams" in got.text and "day 3" in got.text  # facts and turns both
    assert outcome(got.rewrite) == expected
    assert model.count("chat") == 1


@pytest.mark.parametrize("failure, expected", REWRITE_FAILURES)
def test_a_failed_rewrite_serves_the_search_a_store_with_no_model_serves(
        scripted: Make, failure: object, expected: tuple[Any, ...]) -> None:
    model = scripted(chat=[failure])
    plain, mem = pair(model)
    now = utcnow()
    got = mem.search(QUERY, include_episodes=True, known_at=now)
    assert rendered(got) == rendered(plain.search(QUERY, include_episodes=True,
                                                  known_at=now))
    assert {type(r).__name__ for r in got} == {"Result", "EpisodeResult"}
    assert outcome(got.rewrite) == expected
    assert model.count("chat") == 1


@pytest.mark.parametrize("failure", [
    pytest.param(APITimeoutError(), id="sdk-timeout-at-once"),
    pytest.param(RateLimitError(), id="rate-limit-429"),
    pytest.param(Text("Here are some other ways to ask about that."), id="prose"),
])
def test_a_failed_rewrite_leaves_the_mcp_text_as_a_store_with_no_model_gives_it(
        scripted: Make, failure: object) -> None:
    """What an agent reads: the tool's text, over each handle, for each read tool."""
    model = scripted(chat=[failure, failure])
    plain, mem = pair(model)
    for tool in ("memory_recall", "memory_search"):
        got = tool_text(MemvaraMCPServer(mem, user=USER), tool, {"query": QUERY})
        want = tool_text(MemvaraMCPServer(plain, user=USER), tool, {"query": QUERY})
        assert got == want and got[1] is False
    assert model.count("chat") == 2


def test_a_rewrite_that_asks_for_too_much_costs_one_call_and_four_retrievals(
        scripted: Make, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review Focus 2 of the plan. `select/stages.py`: repeats of the question, in any
    case, are skipped, and only the first `MAX_QUERIES` (3) alternatives are kept, "so
    this caps a rewritten read at four retrievals"."""
    reply = {"queries": ["Lisbon trip", "LISBON TRIP", "a", "b", "c", "d", "e"],
             "date_range": None}
    model = scripted(chat=[Text(json.dumps(reply))])
    _, mem = pair(model)
    retrievals: list[str] = []
    once = mem.reader._search_once

    def counted(query: str, **options: Any) -> Any:
        retrievals.append(query)
        return once(query, **options)

    monkeypatch.setattr(mem.reader, "_search_once", counted)
    got = mem.search(QUERY, include_episodes=True)
    assert got.rewrite.outcome == "applied" and got.rewrite.queries == ("a", "b", "c")
    assert sorted(retrievals) == sorted([QUERY, "a", "b", "c"])
    assert model.count("chat") == 1


# -- synthesis -------------------------------------------------------------------------------

SYNTHESIS_FAILURES = [
    pytest.param(APITimeoutError(), "fallback: error", id="sdk-timeout-at-once"),
    pytest.param(Late(APITimeoutError(), 10.5), "fallback: timeout",
                 id="sdk-timeout-after-the-deadline"),
    pytest.param(RateLimitError(), "fallback: provider", id="rate-limit-429"),
    pytest.param(APIStatusError("unavailable", status_code=503), "fallback: provider",
                 id="unavailable-503"),
    pytest.param(AuthenticationError(), "key_rejected", id="rejected-key-401"),
    pytest.param(APIConnectionError("connection reset"), "fallback: error",
                 id="connection-reset"),
    pytest.param(Text("The notes say the trip was to Lisbon."), "fallback: malformed",
                 id="prose"),
    pytest.param(Text(""), "fallback: malformed", id="empty"),
    pytest.param(Truncated('{"synthesis": "The notes say'), "fallback: malformed",
                 id="cut-off-json"),
    pytest.param(Text('{"synthesis": ""}'), "fallback: malformed", id="an-empty-summary"),
    pytest.param(Text('{"synthesis": 42}'), "fallback: malformed", id="a-number"),
    pytest.param(Text('{"summary": "The trip was to Lisbon."}'), "fallback: malformed",
                 id="the-wrong-key"),
    pytest.param(Text('["The trip was to Lisbon."]'), "fallback: malformed",
                 id="a-list-not-an-object"),
]


@pytest.mark.parametrize("failure, why", SYNTHESIS_FAILURES)
def test_a_failed_synthesis_names_itself_and_keeps_every_note(
        scripted: Make, failure: object, why: str) -> None:
    model = scripted(chat=[failure])
    plain, mem = pair(model)
    asked = dict(include_episodes=True, synthesize=True, query_rewrite=False)
    got = mem.recall(QUERY, **asked).splitlines()
    no_model = plain.recall(QUERY, **asked).splitlines()
    notes = plain.recall(QUERY, include_episodes=True, query_rewrite=False).splitlines()
    assert got[0] == f"(summary not written — {why}.)"
    assert no_model[0] == "(summary not written — unconfigured.)"
    assert got[1:] == no_model[1:] == notes
    assert len(notes) > 4
    assert model.count("chat") == 1


def test_a_recall_that_finds_nothing_asks_for_no_summary(scripted: Make) -> None:
    """`Memvara.recall`: "A recall that found no notes makes no call and stays empty.\""""
    model = scripted()
    mem = with_model(model)
    assert mem.recall(QUERY, synthesize=True, query_rewrite=False) == ""
    assert model.count() == 0


# -- ranking -----------------------------------------------------------------------------------

RANKING_FAILURES = [
    pytest.param(APITimeoutError(), ("fallback", "error"), id="sdk-timeout-at-once"),
    pytest.param(RateLimitError(), ("fallback", "provider"), id="rate-limit-429"),
    pytest.param(AuthenticationError(), ("key_rejected", None), id="rejected-key-401"),
    pytest.param(APIConnectionError("connection reset"), ("fallback", "error"),
                 id="connection-reset"),
    pytest.param(Text("I would keep the second and the third excerpts."),
                 ("fallback", "malformed"), id="prose"),
    pytest.param(Text('{"kept": "all of them"}'), ("fallback", "malformed"),
                 id="kept-as-text"),
    pytest.param(Truncated('{"kept": [{"i": 1, "sp'), ("fallback", "malformed"),
                 id="cut-off-json"),
]


@pytest.mark.parametrize("failure, expected", RANKING_FAILURES)
def test_a_failed_ranking_selects_nothing_and_says_so_in_the_last_line(
        scripted: Make, failure: object, expected: tuple[str, str | None]) -> None:
    model = scripted(chat=[failure, failure])
    _, mem = pair(model)
    got = mem.recall(QUERY, include_episodes=True, ranked=True, query_rewrite=False,
                     with_ids=True)
    assert (got.selection.outcome, got.selection.reason) == expected
    assert got.text.splitlines()[-1] == (
        f"(model ranking not applied, showing the default order — {expected[0]}.)")
    results = mem.search(QUERY, include_episodes=True, ranked=True, query_rewrite=False)
    turns = [r for r in results if isinstance(r, EpisodeResult)]
    assert turns and all(r.explain.selected is None for r in turns)
    assert model.count("chat") == 2


def test_a_partly_readable_selector_reply_keeps_only_what_it_can_read(
        scripted: Make) -> None:
    """Review Focus 3 of the plan. `select/model.py` drops an entry it cannot read and
    keeps the rest: a repeated number, a number out of range, a number that is text or a
    boolean, an empty span and a missing span are all dropped."""
    kept = [{"i": 2, "span": "tram"}, {"i": 2, "span": "again"}, {"i": 99, "span": "x"},
            {"i": "1", "span": "y"}, {"i": True, "span": "z"}, {"i": 1, "span": ""},
            {"i": 3}]
    model = scripted(chat=[Text(json.dumps({"kept": kept}))])
    _, mem = pair(model)
    results = mem.search(QUERY, include_episodes=True, ranked=True, query_rewrite=False)
    assert results.selection.outcome == "applied"
    turns = [r for r in results if isinstance(r, EpisodeResult)]
    assert [r.explain.selected for r in turns] == [True] + [False] * (len(turns) - 1)
    numbered = re.search(r"^\[2\] \(\d{4}-\d{2}-\d{2} \d{2}:\d{2}\) (.*)$",
                         model.calls[0].args["prompt"], re.MULTILINE)
    assert numbered is not None
    assert (turns[0].episode.content, turns[0].explain.span) == (numbered.group(1), "tram")
