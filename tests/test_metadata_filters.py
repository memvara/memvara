"""Metadata filters and the file-path filter on `search()` and `recall()`.

The design claim these tests hold is design invariant 7 in `docs/INTERNALS.md`: whatever
narrows rows runs where the limit runs. A filter applied to a result that a store had
already cut to `limit` returns fewer than `k` matches while more exist, and says nothing.
So the central tests build a store where many rows that do **not** match outrank the rows
that do, and check two things together: an unfiltered read of the same size holds none of
the matching rows, which is what a post-filter would have had to work with, and the
filtered read still returns `k` of them.

Everything runs offline against `SQLiteStore`, `HashingEmbedder` and `NullLLM`.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pytest
from conftest import entity_registry

from memvara import AsyncMemvara, HashingEmbedder, Memvara, NullLLM, SQLiteStore
from memvara.filters import SearchFilter, meta_matches, search_filter
from memvara.remote.aio import AsyncRemoteMemvara
from memvara.remote.api import RemoteMemvara
from memvara.remote.errors import InvalidRequest
from memvara.retrieve import EpisodeResult
from memvara.server.config import FEATURES, ServerConfig, build_memvara
from memvara.server.mcp import MemvaraMCPServer
from memvara.types import Result, Scope

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
QUERY = "orbital cluster"


def make(**kw) -> Memvara:
    kw.setdefault("llm", NullLLM())
    kw.setdefault("user", "alice")
    return Memvara(embedder=HashingEmbedder(dim=64), **kw)


def scopes(mem: Memvara):
    return mem._scope(None, None, None, None).ancestors()


def crowded(mem: Memvara, *, many: int = 40, few: int = 5) -> tuple[set[str], set[str]]:
    """`many` claims tagged `infra` that match the query strongly, and `few` tagged `web`
    that match it weakly. Returns the two id sets.

    The strong rows are short and say only the query; the weak ones say it once inside a
    long sentence. BM25 and the hashing embedder both rank the short ones first, so every
    `web` row sits below every `infra` row.
    """
    strong, weak = set(), set()
    for i in range(many):
        receipt = mem.remember(f"node{i}", "uses", QUERY, team="infra")
        strong.update(c.id for c in receipt.added)
    for i in range(few):
        receipt = mem.remember(
            f"site{i}", "notes",
            f"the {QUERY} is mentioned once in this long note about quarterly planning "
            f"budgets hiring roadmaps offsites and many other unrelated topics {i}",
            team="web")
        weak.update(c.id for c in receipt.added)
    return strong, weak


def turns(mem: Memvara, *, many: int = 40, few: int = 5) -> tuple[set[str], set[str]]:
    """The same shape over raw turns. The weak turns are also the furthest in time from
    `T0`, so the time leg ranks them last as well."""
    strong_items = [{"content": f"{QUERY} {i}", "team": "infra",
                     "ts": T0 + timedelta(minutes=i)} for i in range(many)]
    weak_items = [{"content": f"the {QUERY} came up once in a long discussion about "
                              f"budgets, hiring, offsites and other matters {i}",
                   "team": "web", "ts": T0 + timedelta(days=30 + i)}
                  for i in range(few)]
    strong = set(mem.add(strong_items, role="system").episode_ids)
    weak = set(mem.add(weak_items, role="system").episode_ids)
    return strong, weak


WEB = search_filter({"team": "web"}, None)


# -- checking the arguments -------------------------------------------------------------


@pytest.mark.parametrize("filters, fragment", [
    ("team=web", "filters must be a mapping"),
    ({"a b": 1}, "filters key 'a b' is not allowed"),
    ({"": 1}, "filters key '' is not allowed"),
    ({"k" * 65: 1}, "is not allowed"),
    ({3: 1}, "filters key 3 is not allowed"),
    ({"team": []}, "is an empty list, which matches nothing"),
    ({"team": [{"a": 1}]}, "A list may hold only strings, numbers and booleans"),
    ({"team": {"a": 1}}, "A filter value is a string, a number, a boolean"),
    ({"team": None}, "A filter value is a string, a number, a boolean"),
])
def test_a_filter_the_rules_do_not_allow_is_refused_before_anything_is_read(filters,
                                                                            fragment):
    with pytest.raises(ValueError) as caught:
        search_filter(filters, None)
    assert fragment in str(caught.value)


def test_a_non_string_prefix_is_refused():
    with pytest.raises(ValueError, match="filepath_prefix must be a string"):
        search_filter(None, 3)  # type: ignore[arg-type]


def test_every_character_the_key_pattern_names_is_accepted():
    where = search_filter({"a.B-9_z": "x", "k" * 64: 1}, None)
    assert where is not None
    assert [key for key, _ in where.meta] == ["a.B-9_z", "k" * 64]


def test_no_filter_at_all_is_none_but_a_prefix_alone_is_a_filter():
    assert search_filter(None, None) is None
    assert search_filter({}, None) is None
    only_path = search_filter(None, "docs/")
    assert only_path == SearchFilter(meta=(), filepath_prefix="docs/")
    assert only_path.spec == "{}"


def test_the_wire_form_sends_one_value_bare_and_several_as_a_list():
    where = search_filter({"team": "web", "year": (2025, 2026)}, None)
    assert where.wire() == {"team": "web", "year": [2025, 2026]}


@pytest.mark.parametrize("stored, wanted, matches", [
    ("web", "web", True),
    ("Web", "web", False),
    (1, 1.0, True),
    (2.5, 2.5, True),
    ("1", 1, False),
    (1, "1", False),
    (True, True, True),
    (1, True, False),
    (True, 1, False),
    (0, False, False),
    (["web"], "web", False),
    ({"a": "web"}, "web", False),
])
def test_equality_keeps_json_types_apart(stored, wanted, matches):
    """`True == 1` in Python, and a caller asking for `archived: true` does not mean the
    count 1. A stored list is not searched for the value: the test is equality."""
    where = search_filter({"k": wanted}, None)
    assert where.matches({"k": stored}) is matches


def test_a_missing_key_does_not_match_and_every_key_must_match():
    where = search_filter({"team": "web", "tier": ["gold", "silver"]}, None)
    assert where.matches({"team": "web", "tier": "silver"})
    assert not where.matches({"team": "web"})
    assert not where.matches({"team": "web", "tier": "bronze"})


def test_a_stored_meta_column_that_is_not_an_object_matches_nothing_rather_than_raising():
    spec = WEB.spec
    assert meta_matches('{"team": "web"}', spec)
    assert not meta_matches("not json", spec)
    assert not meta_matches('["team"]', spec)
    assert not meta_matches("", spec)
    assert not meta_matches(None, spec)


# -- invariant 7: the limit applies after the filter, in every store method --------------


def test_a_filtered_search_returns_k_matches_when_many_non_matching_rows_rank_higher():
    """The assertion the invariant asks for, end to end.

    An unfiltered search of the same `k` returns no `web` row at all, and neither does an
    unfiltered one five times as deep, which is the pool the retriever over-fetches. A
    filter applied to either afterwards would have returned nothing. The filtered search
    returns three `web` rows."""
    mem = make()
    strong, weak = crowded(mem)
    plain = mem.search(QUERY, k=3)
    assert {r.claim.id for r in plain} <= strong
    deep = mem.search(QUERY, k=15)
    assert not {r.claim.id for r in deep} & weak

    filtered = mem.search(QUERY, k=3, filters={"team": "web"})
    assert len(filtered) == 3
    assert {r.claim.id for r in filtered} <= weak


def test_each_claim_leg_caps_after_filtering():
    mem = make()
    strong, weak = crowded(mem)
    where = search_filter({"team": "web"}, None)

    top = mem.store.lexical_search(QUERY, scopes(mem), 3)
    assert {cid for cid, _ in top} <= strong
    kept = mem.store.lexical_search(QUERY, scopes(mem), 3, where=where)
    assert len(kept) == 3 and {cid for cid, _ in kept} <= weak

    qvec = np.asarray(mem.embedder.encode([QUERY])[0], dtype=np.float32)
    top = mem.store.vector_search(qvec, scopes(mem), 3)
    assert {cid for cid, _ in top} <= strong
    kept = mem.store.vector_search(qvec, scopes(mem), 3, where=where)
    assert len(kept) == 3 and {cid for cid, _ in kept} <= weak

    assert set(mem.store.candidate_ids(scopes(mem), where=where)) == weak


def test_each_episode_leg_caps_after_filtering():
    mem = make()
    strong, weak = turns(mem)
    where = search_filter({"team": "web"}, None)
    store, sc = mem.store, scopes(mem)

    top = store.lexical_search_episodes(QUERY, sc, 3)
    assert {eid for eid, _ in top} <= strong
    kept = store.lexical_search_episodes(QUERY, sc, 3, where=where)
    assert len(kept) == 3 and {eid for eid, _ in kept} <= weak

    qvec = np.asarray(mem.embedder.encode([QUERY])[0], dtype=np.float32)
    top = store.vector_search_episodes(qvec, sc, 3)
    assert {eid for eid, _ in top} <= strong
    kept = store.vector_search_episodes(qvec, sc, 3, where=where)
    assert len(kept) == 3 and {eid for eid, _ in kept} <= weak

    top = store.episodes_near(T0, sc, 3)
    assert {eid for eid, _ in top} <= strong
    kept = store.episodes_near(T0, sc, 3, where=where)
    assert len(kept) == 3 and {eid for eid, _ in kept} <= weak

    assert set(store.episode_candidate_ids(sc, where=where)) == weak


def test_a_filtered_search_with_turns_returns_only_matching_turns_and_claims(
        monkeypatch):
    """All three episode legs run here, the time leg included, and each is handed the
    filter. The instant is a year after the turns so that every one of them had
    happened."""
    mem = make(read_w_temporal=1.0)
    _, weak_claims = crowded(mem)
    _, weak_turns = turns(mem)
    handed = []
    for leg in ("lexical_search_episodes", "vector_search_episodes", "episodes_near"):
        real = getattr(mem.store, leg)

        def spy(*args, _real=real, _leg=leg, **kwargs):
            handed.append((_leg, kwargs.get("where")))
            return _real(*args, **kwargs)

        monkeypatch.setattr(mem.store, leg, spy)
    hits = mem.search(QUERY, k=8, include_episodes=True, filters={"team": "web"},
                      valid_at=T0 + timedelta(days=365))
    assert sorted(leg for leg, _ in handed) == [
        "episodes_near", "lexical_search_episodes", "vector_search_episodes"]
    assert all(where == WEB for _, where in handed)
    assert hits
    for hit in hits:
        if isinstance(hit, EpisodeResult):
            assert hit.episode.id in weak_turns
        else:
            assert hit.claim.id in weak_claims


def test_a_snapshot_reader_connection_can_evaluate_the_filter(tmp_path):
    """A file-backed store reads through a second connection per thread, and a SQL
    function belongs to one connection. Without registering it there, every filtered
    read of a real file would fail with "no such function"."""
    mem = make(path=str(tmp_path / "m.db"))
    _, weak = crowded(mem, many=4, few=2)
    assert mem.store._reader() is not None
    assert {r.claim.id for r in mem.search(QUERY, filters={"team": "web"})} == weak
    mem.close()


# -- documents: their metadata and their file path ---------------------------------------


def documents(mem: Memvara):
    """Three documents that differ only in where they live and how they are labelled."""
    text = "The refund window is thirty days for every order placed online."
    kept = mem.add_document(text, filepath="policies/100%_done/refunds.md",
                            meta={"team": "support"}, custom_id="a")
    wildcard = mem.add_document(text + " Wildcard copy.",
                                filepath="policies/100xxdone/refunds.md",
                                meta={"team": "billing"}, custom_id="b")
    upper = mem.add_document(text + " Upper copy.", filepath="Policies/100%_done/r.md",
                             meta={"team": "billing"}, custom_id="c")
    return kept, wildcard, upper


def chunk_episodes(mem: Memvara, doc) -> set[str]:
    return {c.episode_id for c in mem.store.document_chunks("default", doc.id)}


def test_the_prefix_is_literal_so_percent_and_underscore_match_only_themselves():
    """`LIKE 'policies/100%_%'` would match all three paths: `%` and `_` are wildcards
    there, and ASCII case is ignored."""
    mem = make()
    kept, _, _ = documents(mem)
    hits = mem.search("refund window", include_episodes=True,
                      filepath_prefix="policies/100%_")
    assert hits
    assert {h.episode.id for h in hits} <= chunk_episodes(mem, kept)


def test_a_turn_or_claim_that_came_from_no_document_never_matches_a_prefix():
    mem = make()
    mem.add([{"content": "refund window notes from a call"}], role="system")
    mem.remember("user", "asked_about", "the refund window")
    assert mem.search("refund window", include_episodes=True, filepath_prefix="") == []


def test_a_document_chunk_matches_its_documents_metadata():
    mem = make()
    kept, _, _ = documents(mem)
    hits = mem.search("refund window", include_episodes=True,
                      filters={"team": "support"})
    assert hits
    assert {h.episode.id for h in hits} <= chunk_episodes(mem, kept)


def test_a_turns_own_metadata_includes_the_document_id_it_was_stored_with():
    mem = make()
    kept, wildcard, _ = documents(mem)
    hits = mem.search("refund window", include_episodes=True,
                      filters={"document_id": wildcard.id})
    assert {h.episode.id for h in hits} <= chunk_episodes(mem, wildcard)


def test_a_claim_matches_through_the_document_it_cites():
    """A claim came from a document when one of its sources is a chunk of it. Its own
    metadata has neither the team nor the path, so both matches go through the join."""
    mem = make()
    kept, wildcard, _ = documents(mem)
    cited = mem.remember("refunds", "window", "thirty days",
                         sources=sorted(chunk_episodes(mem, kept))).added
    other = mem.remember("refunds", "window_billing", "thirty days",
                         sources=sorted(chunk_episodes(mem, wildcard))).added
    loose = mem.remember("refunds", "window_note", "thirty days").added

    by_path = mem.search("refunds window", filepath_prefix="policies/100%_")
    assert [r.claim for r in by_path] == cited
    by_meta = mem.search("refunds window", filters={"team": "support"})
    assert [r.claim for r in by_meta] == cited
    both = mem.search("refunds window", filters={"team": "billing"},
                      filepath_prefix="policies/100xx")
    assert [r.claim for r in both] == other
    assert loose[0].id not in {r.claim.id for r in mem.search(
        "refunds window", filepath_prefix="policies/")}


def test_the_metadata_filter_and_the_prefix_must_both_match():
    mem = make()
    documents(mem)
    assert mem.search("refund window", include_episodes=True,
                      filters={"team": "support"},
                      filepath_prefix="policies/100xx") == []


# -- the retriever -----------------------------------------------------------------------


def test_the_graph_leg_does_not_run_on_a_filtered_search(monkeypatch):
    """`Store.adjacent` takes no filter, so a walk would step onto rows the filter
    excludes. The leg is given a zero weight instead."""
    mem = make(read_w_graph=1.0, registry=entity_registry())
    mem.remember("Alice", "reports_to", "Dana", team="web")
    mem.remember("Dana", "works_at", "Acme", team="infra")
    mem.remember("Acme", "headquartered_in", "Tallinn", team="infra")
    seen = []
    real = mem.reader._graph_search

    def spy(claims, fused, scope, limit, valid_at, known_at, states, w_graph, *rest):
        seen.append(w_graph)
        return real(claims, fused, scope, limit, valid_at, known_at, states, w_graph,
                    *rest)

    monkeypatch.setattr(mem.reader, "_graph_search", spy)
    query = "where does the employer of the person Alice reports to work"
    mem.search(query)
    assert seen[-1] > 0.0
    hits = mem.search(query, filters={"team": "web"})
    assert seen[-1] == 0.0
    assert [r.claim.object for r in hits] == ["Dana"]


def test_a_store_without_the_filter_argument_still_serves_unfiltered_reads():
    """The retriever passes `where` only when the caller filtered. A store written
    against the older protocol keeps working, and a filtered read against it fails
    naming the argument rather than returning rows the filter would have excluded."""

    class OlderStore(SQLiteStore):
        def lexical_search(self, query, scopes, limit, *, valid_at=None, known_at=None,
                           states=None, include_invalidated=None):
            return super().lexical_search(query, scopes, limit, valid_at=valid_at,
                                          known_at=known_at, states=states,
                                          include_invalidated=include_invalidated)

    mem = make(store=OlderStore())
    mem.remember("user", "prefers", "tabs", team="web")
    assert [r.claim.object for r in mem.search("prefers")] == ["tabs"]
    with pytest.raises(TypeError, match="where"):
        mem.search("prefers", filters={"team": "web"})


# -- the facade, the switch and the other facades ----------------------------------------


def test_recall_renders_only_matching_facts_and_turns():
    mem = make()
    mem.remember("user", "prefers", "tabs", team="infra")
    mem.remember("user", "prefers", "dark mode", team="web")
    mem.add([{"content": "we agreed dark mode ships first", "team": "web"},
             {"content": "we agreed tabs ship first", "team": "infra"}], role="system")
    block = mem.recall("prefers dark mode tabs ship", filters={"team": "web"},
                       include_episodes=True)
    assert "dark mode" in block
    assert "tabs" not in block


def test_a_switched_off_filter_is_refused_rather_than_ignored():
    mem = make(metadata_filters=False)
    mem.remember("user", "prefers", "tabs", team="infra")
    for kwargs in ({"filters": {"team": "web"}}, {"filepath_prefix": "docs/"}):
        with pytest.raises(ValueError, match="metadata_filters=False"):
            mem.search("prefers", **kwargs)
        with pytest.raises(ValueError, match="metadata_filters=False"):
            mem.recall("prefers", **kwargs)
    # An empty mapping narrows nothing, so there is nothing to refuse.
    assert [r.claim.object for r in mem.search("prefers", filters={})] == ["tabs"]


def test_the_switch_cannot_be_turned_off_against_a_hosted_deployment():
    with pytest.raises(TypeError, match="metadata_filters"):
        Memvara(api_key="k", base_url="https://example.test", metadata_filters=False)


def test_the_scoped_and_async_views_pass_the_filter_through():
    mem = make()
    mem.remember("user", "prefers", "tabs", team="infra")
    mem.remember("user", "prefers", "dark mode", team="web")
    view = mem.scope(user="alice")
    assert [r.claim.object for r in view.search("prefers", filters={"team": "web"})] \
        == ["dark mode"]
    assert "tabs" not in view.recall("prefers", filters={"team": "web"})

    amem = AsyncMemvara(mem)

    async def main():
        hits = await amem.search("prefers", filters={"team": "web"})
        block = await amem.recall("prefers", filters={"team": "web"})
        scoped = amem.scope(user="alice")
        scoped_hits = await scoped.search("prefers", filepath_prefix="nowhere/")
        scoped_block = await scoped.recall("prefers", filters={"team": "infra"})
        return hits, block, scoped_hits, scoped_block

    hits, block, scoped_hits, scoped_block = asyncio.run(main())
    assert [r.claim.object for r in hits] == ["dark mode"]
    assert "tabs" not in block
    assert scoped_hits == []
    assert "dark mode" not in scoped_block


# -- the hosted clients ------------------------------------------------------------------


def _remote(payload, calls, cls=RemoteMemvara, status=200):
    def handler(request):
        calls.append(request)
        return httpx.Response(status, json=payload)

    mem = cls(api_key="k", base_url="https://example.test", user="alice")
    mem._http._client._transport = httpx.MockTransport(handler)
    return mem


_EMPTY_SEARCH = {"as_of": None, "valid_at": None, "known_at": None, "states": ["live"],
                 "count": 0, "results": []}


def test_the_hosted_client_sends_the_filter_fields_only_when_set():
    calls: list = []
    mem = _remote(_EMPTY_SEARCH, calls)
    mem.search("q")
    assert "filters" not in json.loads(calls[-1].content)
    assert "filepath_prefix" not in json.loads(calls[-1].content)
    mem.search("q", filters={"team": ["web", "support"]}, filepath_prefix="docs/")
    body = json.loads(calls[-1].content)
    assert body["filters"] == {"team": ["web", "support"]}
    assert body["filepath_prefix"] == "docs/"
    mem.scope(agent="a1").search("q", filepath_prefix="docs/")
    body = json.loads(calls[-1].content)
    assert "filters" not in body and body["filepath_prefix"] == "docs/"

    mem = _remote({"text": "x", "empty": False}, calls)
    mem.recall("q", filters={"team": "web"})
    assert json.loads(calls[-1].content)["filters"] == {"team": "web"}
    mem.scope(agent="a1").recall("q", filepath_prefix="docs/")
    assert json.loads(calls[-1].content)["filepath_prefix"] == "docs/"


def test_the_hosted_client_refuses_a_bad_filter_without_a_request():
    calls: list = []
    mem = _remote(_EMPTY_SEARCH, calls)
    with pytest.raises(ValueError, match="is not allowed"):
        mem.search("q", filters={"a b": 1})
    assert calls == []


def test_a_deployment_that_does_not_know_the_fields_refuses_rather_than_ignores_them():
    """The facade's request models forbid unknown fields, so a deployment from before
    filters answers 422 and the client raises. It never returns unfiltered rows."""
    calls: list = []
    mem = _remote({"detail": [{"type": "extra_forbidden", "loc": ["body", "filters"]}]},
                  calls, status=422)
    with pytest.raises(InvalidRequest):
        mem.search("q", filters={"team": "web"})
    assert len(calls) == 1


def test_the_async_hosted_client_sends_the_filter_fields():
    calls: list = []

    async def main():
        mem = _remote(_EMPTY_SEARCH, calls, cls=AsyncRemoteMemvara)
        await mem.search("q", filters={"team": "web"})
        await mem.scope(agent="a1").search("q", filepath_prefix="docs/")
        mem = _remote({"text": "x", "empty": False}, calls, cls=AsyncRemoteMemvara)
        await mem.recall("q", filters={"team": "web"})
        await mem.scope(agent="a1").recall("q", filepath_prefix="docs/")

    asyncio.run(main())
    bodies = [json.loads(c.content) for c in calls]
    assert bodies[0]["filters"] == {"team": "web"}
    assert bodies[1]["filepath_prefix"] == "docs/"
    assert bodies[2]["filters"] == {"team": "web"}
    assert bodies[3]["filepath_prefix"] == "docs/"


# -- the MCP tools -----------------------------------------------------------------------


def _server(**kw):
    mem = make()
    mem.remember("user", "prefers", "tabs", team="infra")
    mem.remember("user", "prefers", "dark mode", team="web")
    return MemvaraMCPServer(mem, user="alice", **kw)


def _call(server, name, arguments):
    line = server.handle_line(json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments}}))
    result = json.loads(line)["result"]
    return result["content"][0]["text"], result["isError"]


def _schema(server, name):
    line = server.handle_line(json.dumps({"jsonrpc": "2.0", "id": 1,
                                          "method": "tools/list"}))
    tools = json.loads(line)["result"]["tools"]
    return next(t for t in tools if t["name"] == name)["inputSchema"]["properties"]


@pytest.mark.parametrize("tool", ["memory_search", "memory_recall"])
def test_both_read_tools_filter(tool):
    server = _server()
    text, is_error = _call(server, tool, {"query": "prefers", "filters": {"team": "web"}})
    assert not is_error and "dark mode" in text and "tabs" not in text
    text, is_error = _call(server, tool, {"query": "prefers", "filepath_prefix": "x/"})
    assert not is_error and "No stored memory matched" in text


def test_the_tool_schema_accepts_every_json_type_a_filter_value_may_have():
    server = _server()
    for value in (True, 1, 2.5, "web", ["web", 1, False]):
        text, is_error = _call(server, "memory_search",
                               {"query": "prefers", "filters": {"team": value}})
        assert not is_error, text


@pytest.mark.parametrize("arguments, fragment", [
    ({"filters": {"team": None}},
     "memory_search.filters.team must be a string, a number, a boolean or an array, "
     "got null"),
    ({"filters": {"a b": "web"}}, "memory_search.filters key 'a b' must match the pattern"),
    ({"filters": {"team": {"x": 1}}},
     "memory_search.filters.team must be a string, a number, a boolean or an array"),
    ({"filters": {"team": [["web"]]}},
     "memory_search.filters.team[0] must be a string, a number or a boolean"),
    ({"filters": "team=web"}, "memory_search.filters must be an object"),
    ({"filters": {"team": []}}, "memory_search.filters['team'] is an empty list"),
    ({"filepath_prefix": 3}, "memory_search.filepath_prefix must be a string"),
])
def test_the_tool_schema_refuses_a_bad_filter_with_the_argument_named(arguments,
                                                                      fragment):
    text, is_error = _call(_server(), "memory_search", {"query": "prefers", **arguments})
    assert is_error
    assert fragment in text


def test_a_server_with_the_switch_off_says_so_and_refuses_a_filtered_call():
    server = _server(features_off={"metadata_filters"})
    for tool in ("memory_search", "memory_recall"):
        schema = _schema(server, tool)
        assert "Not accepted on this server" in schema["filters"]["description"]
        assert "Not accepted on this server" in schema["filepath_prefix"]["description"]
        text, is_error = _call(server, tool, {"query": "prefers",
                                              "filters": {"team": "web"}})
        assert is_error
        assert "MEMVARA_FEATURE_METADATA_FILTERS=0" in text
        text, is_error = _call(server, tool, {"query": "prefers"})
        assert not is_error and "tabs" in text


def test_the_switch_is_on_by_default_and_reaches_the_engine(monkeypatch, tmp_path):
    assert "metadata_filters" in FEATURES
    base = {"MEMVARA_DB": str(tmp_path / "m.db"), "MEMVARA_EMBEDDER": "hashing"}
    on = build_memvara(ServerConfig.from_env(base))
    assert on.metadata_filters is True
    on.close()
    off = build_memvara(ServerConfig.from_env(
        {**base, "MEMVARA_FEATURE_METADATA_FILTERS": "0"}))
    assert off.metadata_filters is False
    off.close()


def test_the_scope_key_is_unchanged_by_a_filter():
    """A filter narrows rows inside the bound scope and never widens it: a claim in
    another user's scope with matching metadata is not returned."""
    mem = make()
    mem.remember("user", "prefers", "dark mode", team="web", user="bob")
    assert mem.search("prefers", filters={"team": "web"}) == []
    assert Scope("default", "alice").key() == mem._scope(None, None, None, None).key()


def test_results_are_the_ordinary_result_types():
    mem = make()
    mem.remember("user", "prefers", "dark mode", team="web")
    hits = mem.search("prefers", filters={"team": "web"})
    assert all(isinstance(h, Result) for h in hits)
