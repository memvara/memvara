"""`valid_during`: reading the claims that were true at any moment of a window (#234).

A question about March used to read the store at the last second of March, so a value that
held from 3 March to 10 March was not returned. These tests cover the window at each layer
that takes it: the store's three claim searches, the retriever, query rewrite, `search()`
and `recall()`, the MCP tools and the hosted client.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memvara.embed import HashingEmbedder
from memvara.store import SQLiteStore
from memvara.store.base import state_predicate
from memvara.types import Claim, Scope, time_window

SCOPE = Scope("acme", "alice")
RECORDED = datetime(2026, 1, 1, tzinfo=timezone.utc)
MAR_1 = datetime(2026, 3, 1, tzinfo=timezone.utc)
MAR_3 = datetime(2026, 3, 3, tzinfo=timezone.utc)
MAR_10 = datetime(2026, 3, 10, tzinfo=timezone.utc)
MAR_31 = datetime(2026, 3, 31, 23, 59, 59, tzinfo=timezone.utc)
APR_1 = datetime(2026, 4, 1, tzinfo=timezone.utc)
APR_30 = datetime(2026, 4, 30, 23, 59, 59, tzinfo=timezone.utc)
FEB_1 = datetime(2026, 2, 1, tzinfo=timezone.utc)
FEB_28 = datetime(2026, 2, 28, 23, 59, 59, tzinfo=timezone.utc)
MARCH = (MAR_1, MAR_31)


@pytest.fixture()
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def emb() -> HashingEmbedder:
    return HashingEmbedder(dim=64)


def put(store, emb, obj: str, *, valid_from=RECORDED, valid_to=None,
        predicate: str = "office_in", **kw) -> Claim:
    c = Claim(subject="team", predicate=predicate, object=obj, scope=SCOPE,
              recorded_at=RECORDED, valid_from=valid_from, valid_to=valid_to, **kw)
    store.put_claim(c)
    store.set_embedding(c.id, emb.encode([c.text])[0])
    return c


@pytest.fixture()
def march(store, emb) -> dict[str, Claim]:
    """Four values of one slot: one that ended before March, one that held only from 3 to
    10 March, one that started on 10 March and still holds, and one that starts in
    April."""
    return {
        "before": put(store, emb, "Leeds", valid_from=RECORDED, valid_to=MAR_1),
        "inside": put(store, emb, "Porto", valid_from=MAR_3, valid_to=MAR_10),
        "current": put(store, emb, "Lisbon", valid_from=MAR_10),
        "later": put(store, emb, "Madrid", valid_from=APR_1, predicate="next_office_in"),
    }


def _ids(store, emb, **kw) -> dict[str, set[str]]:
    """The claim ids each of the three claim searches returns for the same question."""
    qvec = emb.encode(["team office"])[0]
    return {
        "candidate_ids": set(store.candidate_ids([SCOPE], **kw)),
        "lexical_search": {i for i, _ in store.lexical_search("team office", [SCOPE],
                                                              10, **kw)},
        "vector_search": {i for i, _ in store.vector_search(qvec, [SCOPE], 10, **kw)},
    }


# --- The store ------------------------------------------------------------------------


def test_a_window_returns_every_value_that_held_during_it_on_all_three_searches(
        store, emb, march):
    """The value that held from 3 to 10 March is the one `valid_at` at the end of March
    could not see. A window over March returns it, the value that started on 10 March,
    and nothing that ended before March or starts after it."""
    want = {march["inside"].id, march["current"].id}
    for search, got in _ids(store, emb, valid_during=MARCH).items():
        assert got == want, search


def test_the_last_second_of_the_window_as_valid_at_misses_the_value_inside_it(
        store, emb, march):
    """The reading query rewrite used before #234, kept here as the reason for it."""
    for search, got in _ids(store, emb, valid_at=MAR_31).items():
        assert got == {march["current"].id}, search


def test_a_window_of_one_instant_answers_as_valid_at_does(store, emb, march):
    for at in (FEB_1, MAR_3, MAR_10, MAR_31, APR_1):
        assert _ids(store, emb, valid_during=(at, at)) == _ids(store, emb, valid_at=at), at


@pytest.mark.parametrize("window, names", [
    ((FEB_1, FEB_28), {"before"}),
    ((APR_1, APR_30), {"current", "later"}),
    # Both ends are included: a claim that stops at the window's start has stopped, and
    # one that starts at its end has started.
    ((MAR_10, MAR_10), {"current"}),
    ((MAR_1, MAR_1), set()),
    ((FEB_1, MAR_3), {"before", "inside"}),
])
def test_a_claim_counts_when_its_interval_overlaps_the_window(store, emb, march, window,
                                                              names):
    want = {march[n].id for n in names}
    for search, got in _ids(store, emb, valid_during=window).items():
        assert got == want, (search, window)


def test_a_window_moves_only_the_world_clock(store, emb):
    """`known_at` still bounds what we had heard: a value recorded in May about March is
    not returned to a read of what we believed in April about March."""
    late = Claim(subject="team", predicate="office_in", object="Porto", scope=SCOPE,
                 recorded_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
                 valid_from=MAR_3, valid_to=MAR_10)
    store.put_claim(late)
    store.set_embedding(late.id, emb.encode([late.text])[0])
    assert late.id in _ids(store, emb, valid_during=MARCH)["candidate_ids"]
    for search, got in _ids(store, emb, valid_during=MARCH, known_at=APR_1).items():
        assert late.id not in got, search


def test_a_window_and_the_ended_state(store, emb, march):
    """Ended at some moment of the window is ended by its end."""
    got = set(store.candidate_ids([SCOPE], valid_during=MARCH, states=["ended"]))
    assert got == {march["before"].id, march["inside"].id}


def test_a_retired_claim_is_not_returned_to_a_window(store, emb, march):
    store.invalidate(march["inside"].id, RECORDED, None)
    for search, got in _ids(store, emb, valid_during=MARCH).items():
        assert march["inside"].id not in got, search


def test_the_window_is_applied_inside_the_limit(store, emb):
    """Invariant 7: twenty values outside March rank above the one inside it for this
    query, and a limit of one still returns the one inside, because the window is in the
    query that applies the limit rather than a filter on what it returned."""
    for n in range(20):
        put(store, emb, f"office office office {n}", valid_from=APR_1,
            predicate=f"p{n}")
    inside = put(store, emb, "Porto", valid_from=MAR_3, valid_to=MAR_10)
    qvec = emb.encode(["team office"])[0]
    assert [i for i, _ in store.lexical_search("team office", [SCOPE], 1,
                                               valid_during=MARCH)] == [inside.id]
    assert [i for i, _ in store.vector_search(qvec, [SCOPE], 1,
                                              valid_during=MARCH)] == [inside.id]


def test_a_store_refuses_a_window_and_an_instant_together(store, emb, march):
    for search in (lambda **kw: store.candidate_ids([SCOPE], **kw),
                   lambda **kw: store.lexical_search("team", [SCOPE], 5, **kw),
                   lambda **kw: store.vector_search(emb.encode(["team"])[0], [SCOPE], 5,
                                                    **kw)):
        with pytest.raises(ValueError, match="valid_during cannot be combined"):
            search(valid_during=MARCH, valid_at=MAR_31)


def test_the_window_predicate_renames_only_the_marker_that_moves():
    assert state_predicate("?")[1] == ("known", "known", "valid", "valid")
    assert state_predicate("?", window=True)[1] == ("known", "known", "valid",
                                                    "valid_start")
    assert state_predicate("?", window=True)[0] == state_predicate("?")[0]
    assert state_predicate("?", states=["ended"], window=True)[1] == (
        "known", "known", "valid")


# --- Checking the argument ------------------------------------------------------------


@pytest.mark.parametrize("value, message", [
    ((MAR_31, MAR_1), "ends before it starts"),
    ((MAR_1,), "must be a pair"),
    ("2026-03", "must be a pair"),
    ((MAR_1, "2026-03-31"), "must be a pair"),
])
def test_a_window_that_is_not_a_pair_in_order_is_refused(value, message):
    with pytest.raises(ValueError, match=message):
        time_window(value)


def test_a_window_cannot_be_combined_with_valid_at_or_as_of():
    with pytest.raises(ValueError, match="with valid_at"):
        time_window(MARCH, valid_at=MAR_31)
    with pytest.raises(ValueError, match="with as_of"):
        time_window(MARCH, as_of=MAR_31)


def test_a_naive_window_is_read_as_utc():
    start, end = time_window((datetime(2026, 3, 1), datetime(2026, 3, 31)))
    assert start.tzinfo is timezone.utc and end.tzinfo is timezone.utc


# --- search() and recall() ------------------------------------------------------------


def _memory():
    from memvara import Memvara, NullLLM

    mem = Memvara(store=SQLiteStore(":memory:"), llm=NullLLM(), user="alice", embedder=HashingEmbedder(dim=512),
                  query_rewrite=False)
    mem.remember("team", "office_in", "Leeds", valid_from=RECORDED, valid_to=MAR_3)
    mem.remember("team", "office_in", "Porto", valid_from=MAR_3, valid_to=MAR_10)
    mem.remember("team", "office_in", "Lisbon", valid_from=MAR_10)
    return mem


def test_search_returns_every_value_that_held_during_the_window():
    mem = _memory()
    objects = {r.claim.object for r in mem.search("team office", valid_during=MARCH)}
    assert objects == {"Leeds", "Porto", "Lisbon"}
    assert [r.claim.object for r in mem.search("team office", valid_at=MAR_31)] == [
        "Lisbon"]
    assert {r.claim.object for r in mem.search(
        "team office", valid_during=(datetime(2026, 3, 4, tzinfo=timezone.utc),
                                     datetime(2026, 3, 9, tzinfo=timezone.utc)))
            } == {"Porto"}


@pytest.mark.parametrize("extra", [{"valid_at": MAR_31}, {"as_of": MAR_31}])
def test_search_refuses_a_window_with_another_world_clock(extra):
    with pytest.raises(ValueError, match="valid_during cannot be combined"):
        _memory().search("team office", valid_during=MARCH, **extra)


@pytest.mark.covers("inv:RT1")
def test_recall_names_the_period_and_lists_every_value_that_held_in_it():
    block = _memory().recall("team office", valid_during=MARCH)
    lines = block.splitlines()
    assert lines[0].startswith(
        "Known about the user at any time from 1 March 2026 to 31 March 2026, as far as "
        "we know today")
    assert {"- team office in Leeds", "- team office in Porto",
            "- team office in Lisbon"} <= set(lines)


def test_recall_history_over_a_window_is_what_had_ended_before_it():
    """A value that ended during the window is one of the facts, so the history tail
    lists only values that had ended before the window started."""
    mem = _memory()
    april = mem.recall("team office", valid_during=(APR_1, APR_30), include_history=True)
    facts, _, history = april.partition("No longer true")
    assert "Lisbon" in facts and "Porto" not in facts
    assert "Porto" in history and "Leeds" in history
    march = mem.recall("team office", valid_during=(MAR_3, MAR_31), include_history=True)
    facts, _, history = march.partition("No longer true")
    assert "Porto" in facts and "Porto" not in history
    # Leeds stopped at 3 March, the window's first moment, so it was not true during
    # the window: it is history, as a value that stops when another starts always is.
    assert "Leeds" in history and "Leeds" not in facts


def test_recall_refuses_a_window_with_valid_at():
    with pytest.raises(ValueError, match="valid_during cannot be combined"):
        _memory().recall("team office", valid_during=MARCH, valid_at=MAR_31)


def test_turns_read_the_windows_end_and_are_not_cut_at_its_start():
    """The window applies to the facts. A turn said before the window can still be the
    one that states what held during it, so the turns read the window's end as
    `valid_at`, as they did before #234."""
    from memvara import Memvara, NullLLM

    mem = Memvara(store=SQLiteStore(":memory:"), llm=NullLLM(), user="alice", embedder=HashingEmbedder(dim=512),
                  query_rewrite=False)
    mem.add([{"role": "user", "content": "the team office moves to Porto in March"}],
            ts=FEB_1)
    turns = [r for r in mem.search("team office Porto", valid_during=MARCH,
                                   include_episodes=True)
             if not hasattr(r, "claim")]
    assert [r.episode.content for r in turns] == [
        "the team office moves to Porto in March"]


def test_a_store_that_predates_the_window_refuses_a_windowed_read_by_name():
    """The window is passed only on a windowed read, so a store written before it still
    serves every other read, and a windowed one fails naming the argument rather than
    returning the wrong instant's rows."""
    from memvara import Memvara, NullLLM

    class Older(SQLiteStore):
        def lexical_search(self, query, scopes, limit, *, valid_at=None, known_at=None,
                           states=None, include_invalidated=None):
            return super().lexical_search(query, scopes, limit, valid_at=valid_at,
                                          known_at=known_at, states=states,
                                          include_invalidated=include_invalidated)

    mem = Memvara(store=Older(":memory:"), llm=NullLLM(), user="alice", embedder=HashingEmbedder(dim=512),
                  query_rewrite=False)
    mem.remember("team", "office_in", "Lisbon")
    assert [r.claim.object for r in mem.search("team office")] == ["Lisbon"]
    with pytest.raises(TypeError, match="valid_during"):
        mem.search("team office", valid_during=MARCH)


# --- the MCP tools --------------------------------------------------------------------


@pytest.fixture()
def srv():
    from memvara.server import MemvaraMCPServer

    server = MemvaraMCPServer(_memory(), user="alice")
    yield server
    server.close()


def _call(server, name, arguments):
    from test_server import call

    return call(server, name, arguments)


def test_memory_search_takes_a_period_and_names_it(srv):
    out, error = _call(srv, "memory_search", {
        "query": "team office", "valid_during": {"start": "2026-03-01", "end": "2026-03-31"}})
    assert not error, out
    assert out.splitlines()[0].startswith(
        "3 match(es) as true at any time from 2026-03-01 to 2026-03-31, as far as we know "
        "today.")
    assert {"Leeds", "Porto", "Lisbon"} <= {w.strip(".") for w in out.split()}


def test_an_end_given_as_a_date_includes_that_whole_day():
    """A value that starts at noon is found by a window whose end is that date, written
    as a date alone, because the end means the whole day."""
    from memvara.server import MemvaraMCPServer

    mem = _memory()
    mem.remember("team", "lunch_at", "noon cafe",
                 valid_from=datetime(2026, 4, 30, 12, tzinfo=timezone.utc))
    server = MemvaraMCPServer(mem, user="alice")
    try:
        out, error = _call(server, "memory_search", {
            "query": "team lunch",
            "valid_during": {"start": "2026-04-30", "end": "2026-04-30"}})
    finally:
        server.close()
    assert not error and "noon cafe" in out, out


def test_memory_recall_takes_a_period_and_names_both_days(srv):
    out, error = _call(srv, "memory_recall", {
        "query": "team office", "valid_during": {"start": "2026-03-01", "end": "2026-03-31"}})
    assert not error, out
    assert out.startswith("Known about the user at any time from 1 March 2026 to "
                          "31 March 2026")


@pytest.mark.parametrize("tool, extra, message", [
    ("memory_search", {"valid_at": "2026-03-31"}, "takes valid_during or one instant"),
    ("memory_search", {"as_of": "2026-03-31"}, "takes valid_during or one instant"),
    ("memory_recall", {"valid_at": "2026-03-31"}, "takes valid_during or valid_at"),
])
def test_a_period_with_an_instant_is_refused_with_the_reason(srv, tool, extra, message):
    out, error = _call(srv, tool, {"query": "team office", **extra,
                                   "valid_during": {"start": "2026-03-01",
                                                    "end": "2026-03-31"}})
    assert error and message in out, out


@pytest.mark.parametrize("window, message", [
    ({"start": "2026-03-01"}, "needs both start and end, and end is missing"),
    ({"end": "2026-03-31"}, "needs both start and end, and start is missing"),
    ({"start": "2026-03-31", "end": "2026-03-01"}, "ends before it starts"),
    ({"start": "March", "end": "2026-03-31"}, "valid_during.start must be an ISO-8601"),
    ({"begin": "2026-03-01", "end": "2026-03-31"}, "must be one of 'start', 'end'"),
])
@pytest.mark.parametrize("tool", ["memory_search", "memory_recall"])
def test_a_malformed_period_is_refused_with_the_reason(srv, tool, window, message):
    out, error = _call(srv, tool, {"query": "team office", "valid_during": window})
    assert error and message in out, out


def test_a_period_that_matches_nothing_says_so(srv):
    out, error = _call(srv, "memory_search", {
        "query": "team office", "valid_during": {"start": "2020-01-01", "end": "2020-01-31"}})
    assert not error
    assert "at any time from 2020-01-01 to 2020-01-31" in out
    assert "Nothing recorded held during that period" in out


# --- the hosted client ----------------------------------------------------------------


def _hosted(handler):
    import httpx

    from memvara.remote.api import RemoteMemvara

    mem = RemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http._client._transport = httpx.MockTransport(handler)
    return mem


def test_the_hosted_client_sends_the_window_only_when_given():
    import json

    import httpx

    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        if request.url.path == "/v1/recall":
            return httpx.Response(200, json={"text": "", "empty": True})
        return httpx.Response(200, json={"results": []})

    mem = _hosted(handler)
    mem.search("q", query_rewrite=False)
    mem.search("q", query_rewrite=False, valid_during=MARCH)
    mem.recall("q", query_rewrite=False, valid_during=MARCH)
    assert "valid_during" not in sent[0]
    window = {"start": "2026-03-01T00:00:00+00:00", "end": "2026-03-31T23:59:59+00:00"}
    assert sent[1]["valid_during"] == window and sent[2]["valid_during"] == window


def test_the_hosted_client_refuses_a_bad_window_before_sending_anything():
    sent = []
    mem = _hosted(lambda request: sent.append(request))
    with pytest.raises(ValueError, match="ends before it starts"):
        mem.search("q", valid_during=(MAR_31, MAR_1))
    with pytest.raises(ValueError, match="valid_during cannot be combined"):
        mem.recall("q", valid_during=MARCH, valid_at=MAR_31)
    assert sent == []


def test_a_deployment_before_the_window_refuses_it_rather_than_answering_for_now():
    """The field is not retried without, as `query_rewrite` is: an answer about the
    present to a question about March is a wrong answer, not a degraded one."""
    import httpx

    from memvara.remote.errors import InvalidRequest

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(422, json={"error": {"code": "invalid_request",
                                                   "message": "valid_during: extra"}})

    with pytest.raises(InvalidRequest):
        _hosted(handler).search("q", valid_during=MARCH)
    assert len(calls) == 1
    # A read that also opted out of query rewrite is sent once more without that field,
    # which is `query_rewrite`'s own retry, and the window is still in it.
    calls.clear()
    with pytest.raises(InvalidRequest):
        _hosted(handler).search("q", query_rewrite=False, valid_during=MARCH)
    assert [b"valid_during" in r.content for r in calls] == [True, True]


def test_a_rewrite_window_survives_the_wire():
    from memvara.remote import hydrate

    got = hydrate.rewrite({"outcome": "applied", "date_from": "2026-03-01",
                           "date_to": "2026-03-31", "valid_at": "2026-03-31T23:59:59Z",
                           "valid_during": {"start": "2026-03-01T00:00:00Z",
                                            "end": "2026-03-31T23:59:59Z"}})
    assert got.valid_during == MARCH
    assert hydrate.rewrite({"outcome": "applied", "date_from": "2026-03-01",
                            "date_to": "2026-03-31"}).valid_during is None
    with pytest.raises(ValueError):
        hydrate.rewrite({"outcome": "applied",
                         "valid_during": {"start": "2026-03-01T00:00:00Z",
                                          "end": "2026-03-31T23:59:59Z"}})


def test_the_fake_hosted_server_answers_a_windowed_read_as_the_library_does():
    from harness.fakes.fake_v1 import FakeV1

    with FakeV1() as fake:
        mem = fake.remote()
        mem.remember("team", "office_in", "Porto", valid_from=MAR_3, valid_to=MAR_10)
        mem.remember("team", "office_in", "Lisbon", valid_from=MAR_10)
        assert {r.claim.object for r in mem.search("team office", query_rewrite=False,
                                                   valid_during=MARCH)} == {"Porto",
                                                                            "Lisbon"}
        assert [r.claim.object for r in mem.search("team office", query_rewrite=False,
                                                   valid_at=MAR_31)] == ["Lisbon"]
