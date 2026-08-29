"""Every read method reaches the endpoint it claims to, with the scope it was bound to.

Two assertions per method, and the pair is the point. The path says the method went where
it says it goes; the returned type says it decoded the envelope that route actually
answers with. Either alone passes for the wrong reason — a method hitting the wrong
endpoint can still decode a fixture that happens to fit, and a method decoding nothing can
still hit the right path.

The payloads are written out rather than rendered through `memvara_cloud.rest.render`,
which `tests/test_remote_hydrate.py` uses. That module is the authority on the wire shape
and is the right fixture for a round trip, but it is an optional dependency: it is absent
here, so those tests skip. A skipped test for every row of the mapping table is not
coverage of the mapping table.
"""
import httpx
import pytest

from memvara.remote.api import RemoteMemvara
from memvara.retrieve import EpisodeResult, Path
from memvara.types import Answer, Claim, Delta, Provenance, Result


def _scope():
    return {"tenant": "t", "user": "alice", "agent": None, "session": None}


def _memory(id="cl_1", predicate="lives_in", obj="Berlin"):
    """One `Memory` as `/v1` renders it. Every field `hydrate.claim` indexes is here,
    because it indexes rather than `.get()`s and a missing one is a `KeyError`."""
    return {
        "id": id, "text": f"user {predicate} {obj}", "subject": "user",
        "predicate": predicate, "object": obj, "polarity": 1,
        "memory_type": "semantic", "scope": _scope(), "state": "live",
        "valid_time": {"valid_from": "2026-01-01T00:00:00+00:00", "valid_to": None},
        "transaction_time": {"recorded_at": "2026-01-01T00:00:00+00:00",
                             "invalidated_at": None, "invalidated_by": None},
        "confidence": 0.9, "salience": 1.0, "salience_base": None,
        "observation_count": 1, "last_observed": None, "derivation": "user",
        "extractor": "api", "source_ids": [], "metadata": {},
        "links": {"self": f"/v1/memories/{id}", "why": f"/v1/memories/{id}/why",
                  "history": "/v1/history?subject=user"},
    }


def _episode(id="ep_1"):
    return {"id": id, "role": "user", "ts": "2026-01-01T00:00:00+00:00",
            "content": "I live in Berlin", "scope": _scope(), "metadata": {}}


def _ranking(applicable=True):
    return {"vector_rank": 1, "vector_score": 0.8, "lexical_rank": 2,
            "lexical_score": 0.4, "fusion_score": 0.6,
            "recency": 0.9 if applicable else None,
            "confidence": 0.9 if applicable else None,
            "salience": 1.0 if applicable else None,
            "rerank_score": None, "raw_score": 0.7, "final_score": 0.7,
            "summary": "vector#1 bm25#2"}


@pytest.fixture
def recorded():
    """Build a real `RemoteMemvara` and swap only the transport underneath it.

    Replacing `_http`, or the `httpx.Client` inside it, would rebuild the base url, the
    bearer header and the timeout in the fixture instead of exercising what `__init__`
    produced — the defect `tests/test_remote_client.py` documents and was already fixed
    for once. Swapping `_transport` leaves everything the constructor set intact.
    """
    calls = []

    def build(payload, **kw):
        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=payload)

        mem = RemoteMemvara(api_key="k", base_url="https://example.test", **kw)
        mem._http._client._transport = httpx.MockTransport(handler)
        return mem

    build.calls = calls
    return build


# --- the scope, which rides on every call ------------------------------------


def test_the_bound_scope_is_sent_as_query_parameters(recorded):
    mem = recorded({"count": 0, "total": 0, "limit": 1, "offset": 0, "as_of": None,
                    "valid_at": None, "known_at": None, "states": ["live"],
                    "memories": []}, user="alice", agent="a1")
    mem.count()
    params = dict(recorded.calls[-1].url.params)
    assert params["user"] == "alice"
    assert params["agent"] == "a1"


def test_an_unbound_scope_field_is_omitted_rather_than_sent_empty(recorded):
    mem = recorded({"count": 0, "total": 0, "limit": 1, "offset": 0, "as_of": None,
                    "valid_at": None, "known_at": None, "states": ["live"],
                    "memories": []}, user="alice")
    mem.count()
    assert "agent" not in dict(recorded.calls[-1].url.params)


def test_the_tenant_is_never_sent_because_the_token_decides_it(recorded):
    mem = recorded({"scope": _scope(), "visible": 0, "tenant_counts": {"claims": 0},
                    "extractor": "fast-path-only", "read_only": False},
                   tenant="not-mine")
    mem.stats()
    assert "tenant" not in dict(recorded.calls[-1].url.params)


def test_the_constructor_makes_no_request(monkeypatch):
    """Building a client resolves a credential and opens a pool. Nothing else.

    Same rule `Memvara.__init__` follows for `llm=`: a constructor that dialled out would
    spend money, or fail, as a side effect of naming a deployment.

    The observation is armed **before** the constructor runs, which is the only ordering
    that can catch anything. An earlier version of this test installed a recording
    transport onto the client the constructor had already returned, so the list it
    asserted on was empty whatever the constructor did — a constructor that issued a
    request would have failed only incidentally, on DNS for `example.test`, and would
    have passed green behind a wildcard resolver or a corporate proxy.

    `httpx.Client.send` is the chokepoint every request goes through whatever transport
    is installed, so patching it catches a request the constructor made through a client
    this test never gets a handle on.
    """
    attempts = []

    def refuse(self, request, **kw):
        attempts.append(request)
        raise AssertionError(
            f"the constructor issued {request.method} {request.url}; it must open a "
            "pool and stop")

    monkeypatch.setattr(httpx.Client, "send", refuse)
    RemoteMemvara(api_key="k", base_url="https://example.test")
    assert attempts == []


# --- one per mapping-table row ------------------------------------------------


def test_search_reaches_the_search_endpoint_and_returns_results(recorded):
    mem = recorded({"as_of": None, "valid_at": None, "known_at": None,
                    "states": ["live"], "count": 1,
                    "results": [{"kind": "claim", "score": 0.7,
                                 "ranking": _ranking(), "memory": _memory()}]})
    hits = mem.search("where do I live")
    assert recorded.calls[-1].url.path == "/v1/search"
    assert recorded.calls[-1].method == "POST"
    assert isinstance(hits[0], Result) and hits[0].claim.object == "Berlin"


def test_search_decodes_an_episode_hit_as_an_episode_not_a_claim(recorded):
    """`include_episodes` mixes two kinds into one list, and they are not
    interchangeable: a claim has been extracted and reconciled, a turn is a verbatim
    thing somebody said once. The wire discriminates on `kind`, and an episode hit
    carries its turn under `episode` where a claim hit carries `memory`."""
    mem = recorded({"as_of": None, "valid_at": None, "known_at": None,
                    "states": ["live"], "count": 1,
                    "results": [{"kind": "episode", "score": 0.3,
                                 "ranking": _ranking(applicable=False),
                                 "episode": _episode()}]})
    hits = mem.search("berlin", include_episodes=True)
    assert isinstance(hits[0], EpisodeResult)
    assert hits[0].episode.content == "I live in Berlin"


def test_recall_reaches_the_recall_endpoint_and_returns_the_rendered_text(recorded):
    mem = recorded({"text": "Known about the user:\n- user lives in Berlin",
                    "empty": False})
    text = mem.recall("where do I live")
    assert recorded.calls[-1].url.path == "/v1/recall"
    assert isinstance(text, str) and "Berlin" in text


def test_recall_refuses_a_budget_rather_than_dropping_it(recorded):
    """The endpoint renders server-side and takes no budget. Ignoring one would ship an
    oversized prompt with nothing to notice it by."""
    mem = recorded({"text": "x", "empty": False})
    with pytest.raises(ValueError) as caught:
        mem.recall("q", budget=200)
    assert "budget" in str(caught.value)


def test_get_reaches_the_memory_endpoint_and_returns_a_claim(recorded):
    mem = recorded(_memory("cl_7"))
    claim = mem.get("cl_7")
    assert recorded.calls[-1].url.path == "/v1/memories/cl_7"
    assert isinstance(claim, Claim) and claim.id == "cl_7"


def test_get_returns_none_for_a_memory_that_is_not_there(recorded):
    """None rather than raising, for an id in another tenant as well as one that never
    existed: the facade answers 404 for both so this cannot become an existence oracle."""
    mem = RemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http._client._transport = httpx.MockTransport(
        lambda r: httpx.Response(404, json={"error": {"code": "not_found",
                                                      "message": "x"}}))
    assert mem.get("cl_missing") is None


def test_get_all_reaches_the_listing_endpoint_and_returns_claims(recorded):
    mem = recorded({"count": 1, "total": 1, "limit": 100, "offset": 0, "as_of": None,
                    "valid_at": None, "known_at": None, "states": ["live"],
                    "memories": [_memory()]})
    claims = mem.get_all(limit=25, offset=50)
    params = dict(recorded.calls[-1].url.params)
    assert recorded.calls[-1].url.path == "/v1/memories"
    assert params["limit"] == "25" and params["offset"] == "50"
    assert isinstance(claims[0], Claim)


def test_count_asks_for_one_row_and_reports_the_total(recorded):
    mem = recorded({"count": 1, "total": 47, "limit": 1, "offset": 0, "as_of": None,
                    "valid_at": None, "known_at": None, "states": ["live"],
                    "memories": [_memory()]})
    assert mem.count() == 47
    assert recorded.calls[-1].url.path == "/v1/memories"
    assert dict(recorded.calls[-1].url.params)["limit"] == "1"


def test_history_reaches_the_history_endpoint_and_reads_the_timeline(recorded):
    """The envelope's list is `timeline`, not `memories`. Reading the wrong key against a
    real deployment is a `KeyError` on every call, which is the good failure; reading it
    against a listing envelope that happens to have `memories` would be the bad one."""
    mem = recorded({"subject": "user", "predicate": "lives_in", "scope": _scope(),
                    "as_of": None, "valid_at": None, "known_at": None, "count": 1,
                    "timeline": [_memory()]})
    claims = mem.history("user", "lives_in")
    assert recorded.calls[-1].url.path == "/v1/history"
    assert isinstance(claims[0], Claim)


def test_why_reaches_the_provenance_endpoint_and_returns_provenance(recorded):
    mem = recorded({"memory": _memory("cl_2"), "derivation": "user", "extractor": "api",
                    "sources": [_episode()], "superseded": []})
    prov = mem.why("cl_2")
    assert recorded.calls[-1].url.path == "/v1/memories/cl_2/why"
    assert isinstance(prov, Provenance) and prov.claim.id == "cl_2"


def test_why_returns_none_for_a_memory_that_is_not_visible_here(recorded):
    """`Memvara.why` answers None, and `tools.py` branches on that. Raising instead would
    turn a missing id into an exception on the hosted path alone."""
    mem = RemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http._client._transport = httpx.MockTransport(
        lambda r: httpx.Response(404, json={"error": {"code": "not_found",
                                                      "message": "x"}}))
    assert mem.why("cl_missing") is None


def test_ask_reaches_the_ask_endpoint_and_returns_an_answer(recorded):
    mem = recorded({"question": "where did I live", "at": "2026-01-01T00:00:00+00:00",
                    "count": 1, "text": "Then and now agree.",
                    "readings": [{"subject": "user", "predicate": "lives_in",
                                  "now": [_memory()], "then": [_memory()],
                                  "stated": [_memory()], "diverged": False,
                                  "moved": False}]})
    answer = mem.ask("where did I live")
    assert recorded.calls[-1].url.path == "/v1/ask"
    assert isinstance(answer, Answer) and answer.readings[0].predicate == "lives_in"


def test_since_reaches_the_since_endpoint_and_returns_a_delta(recorded):
    from datetime import datetime, timezone

    mem = recorded({"since": "2026-01-01T00:00:00+00:00", "added": [_memory()],
                    "gone": []})
    delta = mem.since(datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert recorded.calls[-1].url.path == "/v1/since"
    assert "since" in dict(recorded.calls[-1].url.params)
    assert isinstance(delta, Delta) and len(delta.added) == 1


def test_produced_reaches_the_episode_endpoint_and_returns_claims(recorded):
    mem = recorded({"episode_id": "ep_1", "as_of": None, "valid_at": None,
                    "known_at": None, "count": 1, "memories": [_memory()]})
    claims = mem.produced("ep_1")
    assert recorded.calls[-1].url.path == "/v1/episodes/ep_1/produced"
    assert isinstance(claims[0], Claim)


def test_neighborhood_reaches_the_graph_endpoint_and_returns_paths(recorded):
    """The seed parameter is `entity`. FastAPI drops an unknown query parameter in
    silence, so a client sending `key=` would get a 422 for a missing `entity` at best
    and a walk from nowhere at worst."""
    mem = recorded({"as_of": None, "valid_at": None, "known_at": None, "count": 1,
                    "paths": [{"nodes": ["alice", "berlin"], "labels": ["lives_in"],
                               "hops": 1, "score": 0.5,
                               "edges": [{"memory": _memory(), "backward": False,
                                          "strength": 0.5}]}]})
    paths = mem.neighborhood("alice")
    assert recorded.calls[-1].url.path == "/v1/neighborhood"
    assert dict(recorded.calls[-1].url.params)["entity"] == "alice"
    assert isinstance(paths[0], Path)


def test_paths_between_reaches_the_paths_endpoint_and_returns_paths(recorded):
    mem = recorded({"as_of": None, "valid_at": None, "known_at": None, "count": 1,
                    "paths": [{"nodes": ["alice", "berlin"], "labels": ["lives_in"],
                               "hops": 1, "score": 0.5,
                               "edges": [{"memory": _memory(), "backward": False,
                                          "strength": 0.5}]}]})
    paths = mem.paths_between("alice", "berlin")
    params = dict(recorded.calls[-1].url.params)
    assert recorded.calls[-1].url.path == "/v1/paths"
    assert params["source"] == "alice" and params["target"] == "berlin"
    assert isinstance(paths[0], Path)


def test_standing_reaches_the_standing_endpoint_and_sends_k_as_limit(recorded):
    """The endpoint's parameter is `limit` and the library's argument is `k`. Task 11
    calls `standing(k=...)`, so the rename happens here or nowhere."""
    mem = recorded({"count": 1, "limit": 5, "truncated": False,
                    "memories": [_memory(predicate="prefers", obj="pytest")]})
    claims = mem.standing(k=5)
    assert recorded.calls[-1].url.path == "/v1/standing"
    assert dict(recorded.calls[-1].url.params)["limit"] == "5"
    assert isinstance(claims[0], Claim)


def test_stats_returns_the_tenant_counts_and_not_the_envelope(recorded):
    """`Memvara.stats()` answers with row counts, and `tools.py` reads `claims` and
    `live_claims` straight off it. The `/v1/stats` envelope nests those under
    `tenant_counts` beside `visible`, `extractor` and `read_only`, so returning the
    envelope would make the same expression a KeyError on one engine and a number on the
    other."""
    mem = recorded({"scope": _scope(), "visible": 2,
                    "tenant_counts": {"claims": 3, "episodes": 5},
                    "extractor": "fast-path-only", "read_only": False})
    counts = mem.stats()
    assert recorded.calls[-1].url.path == "/v1/stats"
    assert counts == {"claims": 3, "episodes": 5}


def test_service_returns_the_whole_envelope_that_stats_unwraps(recorded):
    """The twin of the test above, and the reason both methods exist.

    `stats()` unwraps to `tenant_counts` so that `stats()["claims"]` is a number against
    either engine. That drops `extractor` and `read_only`, which no local engine can
    answer for a hosted deployment and which a server needs at startup — `read_only` most
    of all, since a server that ignores it lists write tools the deployment refuses
    mid-conversation as a 403.
    """
    mem = recorded({"scope": _scope(), "visible": 2,
                    "tenant_counts": {"claims": 3, "episodes": 5},
                    "extractor": "fast-path-only", "read_only": True})
    body = mem.service()
    assert recorded.calls[-1].url.path == "/v1/stats"
    assert body["read_only"] is True
    assert body["extractor"] == "fast-path-only"
    assert body["tenant_counts"] == {"claims": 3, "episodes": 5}


def test_connectivity_is_empty_when_the_facade_does_not_report_joins(recorded):
    """`{}` is not a store with nothing in it. An empty store answers two zeros; a
    backend that cannot measure the join says nothing at all, and reading a missing key
    as zero would report a star nobody measured."""
    mem = recorded({"scope": _scope(), "visible": 0,
                    "tenant_counts": {"claims": 5},
                    "extractor": "fast-path-only", "read_only": False})
    assert mem.connectivity() == {}


def test_connectivity_reports_the_two_counts_when_present(recorded):
    mem = recorded({"scope": _scope(), "visible": 0,
                    "tenant_counts": {"claims": 10, "live_claims": 10,
                                      "joinable_claims": 4},
                    "extractor": "fast-path-only", "read_only": False})
    assert mem.connectivity() == {"live_claims": 10, "joinable_claims": 4}


def test_whoami_reaches_the_whoami_endpoint_and_returns_the_body(recorded):
    mem = recorded({"token_id": "tok_1", "scope": _scope(),
                    "granted_privilege": "write", "effective_privilege": "write",
                    "expires_at": None, "read_only": False})
    body = mem.whoami()
    assert recorded.calls[-1].url.path == "/v1/whoami"
    assert body["token_id"] == "tok_1"


def test_health_reaches_the_health_endpoint_and_returns_the_body(recorded):
    mem = recorded({"status": "ok", "memvara_version": "0.2.0"})
    body = mem.health()
    assert recorded.calls[-1].url.path == "/v1/health"
    assert body["status"] == "ok"
