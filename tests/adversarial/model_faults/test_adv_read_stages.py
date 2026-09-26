"""The read path when a model stage fails: query rewrite, synthesis and ranking.

INTERNALS invariant 1: each of the three read stages "records its outcome ... on the
result, and serves the plain read on every outcome but `applied`". So a read whose stage
failed must equal, byte for byte, the read a store with no model serves. Two lines are
the documented exceptions, both in `Memvara.recall`'s docstring: a `synthesize=True` block
starts with a line naming why no summary was written, and a `ranked=True` block ends with
a line naming why the model did not rank.

Every test reads one store through two handles: one whose model is scripted to fail, and
one with no model. The store is written once for the whole module, because no test here
writes to it, and each test checks on its way out that the store is still as it was. A
`search()` compared field by field passes `known_at`, one instant after every write,
because recency is measured at `known_at`, and two present-tense reads a moment apart
would otherwise differ in their scores' last digits.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterator, TypeVar

import pytest

from harness import known_bugs
from memvara import EpisodeResult, Memvara
from memvara.select.base import StageOutcome
from memvara.server import MemvaraMCPServer
from memvara.store import SQLiteStore, Store
from memvara.types import SearchResults, utcnow

from .handles import USER, rendered, tool_text, with_model, without_model
from .scripted import (
    APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, Late,
    RateLimitError, ScriptedModel, Text, Truncated,
)

Make = Callable[..., ScriptedModel]
Pair = Callable[[ScriptedModel], tuple[Memvara, Memvara]]
Stage = TypeVar("Stage", bound=StageOutcome)

QUERY = "Lisbon trip"

FACTS = [("lives_in", "Berlin"), ("likes", "Lisbon trams"), ("visited", "Lisbon in spring")]

TURN = "We talked about the Lisbon trip again on day {day}: the tram, the pastries, the tiles."


@pytest.fixture(scope="module")
def notes() -> Iterator[SQLiteStore]:
    """The store every test in this module reads, written once with no model: three facts
    the caller asserted and eight turns."""
    store = SQLiteStore(":memory:")
    plain = without_model(store)
    for predicate, obj in FACTS:
        plain.remember("user", predicate, obj)
    for day in range(8):
        plain.add(TURN.format(day=day), role="system")
    yield store
    store.close()


def contents(store: Store) -> tuple[Any, ...]:
    """Every claim in every state, every turn, and the store's row counts."""
    return (list(store.iter_claims(states=("live", "ended", "retired"))),
            list(store.iter_episodes()), store.stats())


@pytest.fixture
def pair(notes: SQLiteStore) -> Iterator[Pair]:
    """Make two handles on the module's store for a scripted model: the first has no
    model, the second has `model` for its rewrite, its synthesis and its selector.

    The test fails if the store has changed by the time it ends, because every later test
    in this module reads the same store and expects to find only what the `notes`
    fixture wrote."""
    before = contents(notes)

    def handles(model: ScriptedModel) -> tuple[Memvara, Memvara]:
        return without_model(notes), with_model(model, store=notes, ranked=True)

    yield handles
    assert contents(notes) == before, "this test wrote to the store the whole module reads"


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


def recorded(stage: Stage | None) -> Stage:
    """`stage` itself. The test fails, saying so, if the read recorded no outcome for it."""
    assert stage is not None, "the read recorded no outcome for this stage"
    return stage


def outcome(stage: StageOutcome | None) -> tuple[Any, ...]:
    """A stage's outcome, reason and status."""
    stage = recorded(stage)
    return (stage.outcome, stage.reason, stage.status)


def searched(results: list[Any]) -> SearchResults:
    """`results` as the `SearchResults` that `search()` returns, which also carries the
    rewrite and selection records. The test fails, naming the type, if it is not one."""
    assert isinstance(results, SearchResults), (
        f"search() returned a {type(results).__name__}, not SearchResults")
    return results


# -- query rewrite -----------------------------------------------------------------------


@pytest.mark.parametrize("failure, expected", REWRITE_FAILURES)
def test_a_failed_rewrite_serves_the_recall_a_store_with_no_model_serves(
        scripted: Make, pair: Pair, failure: object, expected: tuple[Any, ...]) -> None:
    model = scripted(chat=[failure])
    plain, mem = pair(model)
    got = mem.recall(QUERY, include_episodes=True, with_ids=True)
    assert got.text == plain.recall(QUERY, include_episodes=True)
    assert "Lisbon trams" in got.text and "day 3" in got.text  # facts and turns both
    assert outcome(got.rewrite) == expected
    assert model.count("chat") == 1


@pytest.mark.parametrize("failure, expected", REWRITE_FAILURES)
def test_a_failed_rewrite_serves_the_search_a_store_with_no_model_serves(
        scripted: Make, pair: Pair, failure: object, expected: tuple[Any, ...]) -> None:
    model = scripted(chat=[failure])
    plain, mem = pair(model)
    now = utcnow()
    got = searched(mem.search(QUERY, include_episodes=True, known_at=now))
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
        scripted: Make, pair: Pair, failure: object) -> None:
    """What an agent reads: the tool's text, over each handle, for each read tool."""
    model = scripted(chat=[failure, failure])
    plain, mem = pair(model)
    for tool in ("memory_recall", "memory_search"):
        got = tool_text(MemvaraMCPServer(mem, user=USER), tool, {"query": QUERY})
        want = tool_text(MemvaraMCPServer(plain, user=USER), tool, {"query": QUERY})
        assert got == want and got[1] is False
    assert model.count("chat") == 2


def test_a_rewrite_that_asks_for_too_much_costs_one_call_and_four_retrievals(
        scripted: Make, pair: Pair, monkeypatch: pytest.MonkeyPatch) -> None:
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
    rewrite = recorded(searched(mem.search(QUERY, include_episodes=True)).rewrite)
    assert rewrite.outcome == "applied" and rewrite.queries == ("a", "b", "c")
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
        scripted: Make, pair: Pair, failure: object, why: str) -> None:
    model = scripted(chat=[failure])
    plain, mem = pair(model)
    got = mem.recall(QUERY, include_episodes=True, synthesize=True,
                     query_rewrite=False).splitlines()
    no_model = plain.recall(QUERY, include_episodes=True, synthesize=True,
                            query_rewrite=False).splitlines()
    notes = plain.recall(QUERY, include_episodes=True, query_rewrite=False).splitlines()
    assert got[0] == f"(summary not written — {why}.)"
    assert no_model[0] == "(summary not written — unconfigured.)"
    assert got[1:] == no_model[1:] == notes
    assert len(notes) > 4
    assert model.count("chat") == 1


def test_a_recall_that_finds_nothing_asks_for_no_summary(scripted: Make) -> None:
    """`Memvara.recall`: "A recall that found no notes makes no call and stays empty."

    This test reads a new, empty store rather than the module's store, because it needs a
    recall that finds nothing. It writes nothing either."""
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
        scripted: Make, pair: Pair, failure: object,
        expected: tuple[str, str | None]) -> None:
    model = scripted(chat=[failure, failure])
    _, mem = pair(model)
    got = mem.recall(QUERY, include_episodes=True, ranked=True, query_rewrite=False,
                     with_ids=True)
    selection = recorded(got.selection)
    assert (selection.outcome, selection.reason) == expected
    assert got.text.splitlines()[-1] == (
        f"(model ranking not applied, showing the default order — {expected[0]}.)")
    results = mem.search(QUERY, include_episodes=True, ranked=True, query_rewrite=False)
    turns = [r for r in results if isinstance(r, EpisodeResult)]
    assert turns and all(r.explain.selected is None for r in turns)
    assert model.count("chat") == 2


def test_a_partly_readable_selector_reply_keeps_only_what_it_can_read(
        scripted: Make, pair: Pair) -> None:
    """Review Focus 3 of the plan. `select/model.py` drops an entry it cannot read and
    keeps the rest: a repeated number, a number out of range, a number that is text or a
    boolean, an empty span and a missing span are all dropped."""
    kept = [{"i": 2, "span": "tram"}, {"i": 2, "span": "again"}, {"i": 99, "span": "x"},
            {"i": "1", "span": "y"}, {"i": True, "span": "z"}, {"i": 1, "span": ""},
            {"i": 3}]
    model = scripted(chat=[Text(json.dumps({"kept": kept}))])
    _, mem = pair(model)
    results = searched(mem.search(QUERY, include_episodes=True, ranked=True,
                                  query_rewrite=False))
    assert recorded(results.selection).outcome == "applied"
    turns = [r for r in results if isinstance(r, EpisodeResult)]
    assert [r.explain.selected for r in turns] == [True] + [False] * (len(turns) - 1)
    numbered = re.search(r"^\[2\] \(\d{4}-\d{2}-\d{2} \d{2}:\d{2}\) (.*)$",
                         model.calls[0].args["prompt"], re.MULTILINE)
    assert numbered is not None
    assert (turns[0].episode.content, turns[0].explain.span) == (numbered.group(1), "tram")


# -- a bug these tests found, pinned until its fix lands -------------------------------------


def kinds(results: Any) -> list[str]:
    """Each result's kind and id, in order."""
    return [f"{type(r).__name__} {(r.episode if isinstance(r, EpisodeResult) else r.claim).id}"
            for r in results]


@pytest.mark.parametrize("failure", [
    pytest.param(RateLimitError(), id="rate-limit-429"),
    pytest.param(Text("I would keep the second excerpt."), id="prose"),
])
@known_bugs.xfail("B31")
def test_a_failed_ranking_serves_the_plain_read(
        scripted: Make, pair: Pair, failure: object) -> None:
    """#308. INTERNALS invariant 1 says a failed stage serves the plain read. A ranked read
    gathers its turns at the reranker's depth whatever its outcome, so when the selector
    fails it interleaves more turns than a plain read takes, and they push out facts the
    plain read shows. Only the last line, which names the outcome, may differ."""
    model = scripted(chat=[failure, failure])
    plain, mem = pair(model)
    got = mem.recall(QUERY, include_episodes=True, ranked=True,
                     query_rewrite=False).splitlines()[:-1]
    want = plain.recall(QUERY, include_episodes=True, ranked=True,
                        query_rewrite=False).splitlines()[:-1]
    now = utcnow()
    got_read = kinds(mem.search(QUERY, include_episodes=True, ranked=True,
                                query_rewrite=False, known_at=now))
    want_read = kinds(plain.search(QUERY, include_episodes=True, ranked=True,
                                   query_rewrite=False, known_at=now))
    turns_got = sum(kind.startswith("EpisodeResult") for kind in got_read)
    turns_want = sum(kind.startswith("EpisodeResult") for kind in want_read)
    # A fact line starts "- ", and a turn's line starts "- [" with the day it was said.
    lost = [line for line in want
            if line.startswith("- ") and not line.startswith("- [") and line not in got]
    if turns_got > turns_want and lost:
        raise known_bugs.Reproduced(
            f"B31: the failed ranked read returned {turns_got} turns where the plain read "
            f"returns {turns_want}, and its block dropped {lost}")
    assert got == want
    assert got_read == want_read
