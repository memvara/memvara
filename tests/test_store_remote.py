"""`RemoteStore`: the `Store` protocol methods that have a faithful REST mapping, and the
explicit `NotImplementedError` for every one that does not.

Offline throughout. `httpx.Client` is never let near a socket — every test hands
`RemoteStore` a `FakeTransport` (an `httpx.MockTransport`-shaped stand-in built on
`httpx.Client(transport=...)`, exactly the seam `httpx` itself documents for tests) so a
request is answered in-process by a Python function that inspects the method/path/body and
returns a canned `httpx.Response`. No test in this file opens a port.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import inspect

import httpx
import numpy as np
import pytest

from memvara import HashingEmbedder, Memvara, NullLLM
from memvara.store.base import Store
from memvara.store.remote import RemoteStore
from memvara.types import Claim, Derivation, Episode, MemoryType, Scope

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def memory_body(**overrides) -> dict:
    """The wire shape `GET /v1/memories/{id}` (and friends) hand back, matching what
    `RemoteStore._claim_from_memory` reads."""
    body = {
        "id": "clm_1",
        "subject": "user",
        "predicate": "lives_in",
        "object": "Lisbon",
        "scope": {"tenant": "acme", "user": "alice", "agent": None, "session": None},
        "text": "user lives_in Lisbon",
        "polarity": 1,
        "memory_type": "semantic",
        "valid_time": {"valid_from": "2024-01-01T00:00:00+00:00", "valid_to": None},
        "transaction_time": {"recorded_at": "2024-01-01T00:00:00+00:00",
                             "invalidated_at": None, "invalidated_by": None},
        "confidence": 0.9,
        "salience": 1.0,
        "observation_count": 1,
        "source_ids": [],
        "derivation": "user",
        "extractor": "fast-path",
        "metadata": {},
    }
    body.update(overrides)
    return body


class FakeTransport(httpx.BaseTransport):
    """Answers requests from a list of `(method, path_suffix) -> handler` rules, in
    registration order, recording every request it sees so a test can assert on what
    `RemoteStore` actually sent."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self._rules: list[tuple[str, str, Callable]] = []

    def on(self, method: str, path: str, handler) -> "FakeTransport":
        self._rules.append((method, path, handler))
        return self

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        path = request.url.path
        for method, rule_path, handler in self._rules:
            if method == request.method and path == rule_path:
                return handler(request)
        raise AssertionError(f"unhandled request: {request.method} {path}")


def make_store(transport: FakeTransport) -> RemoteStore:
    store = RemoteStore(base_url="https://app.memvara.dev", api_key="k-test")
    store._client = httpx.Client(base_url=store._base_url,
                                 headers={"Authorization": "Bearer k-test"},
                                 transport=transport)
    return store


def json_response(status: int, body) -> Callable:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)
    return handler


# -- construction ---------------------------------------------------------------------

def test_missing_httpx_is_a_clear_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "httpx":
            raise ImportError("no module named httpx")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="memvara\\[cloud\\]"):
        RemoteStore(base_url="https://app.memvara.dev", api_key="k")


def test_authorization_header_carries_the_bearer_key():
    store = RemoteStore(base_url="https://app.memvara.dev/", api_key="k-test")
    assert store._base_url == "https://app.memvara.dev"
    assert store._client.headers["authorization"] == "Bearer k-test"
    store.close()


# -- get_claim / get_claims ------------------------------------------------------------

def test_get_claim_maps_a_live_memory():
    transport = FakeTransport().on("GET", "/v1/memories/clm_1",
                                   json_response(200, memory_body()))
    store = make_store(transport)
    claim = store.get_claim("clm_1")
    assert isinstance(claim, Claim)
    assert claim.subject == "user" and claim.object == "Lisbon"
    assert claim.scope == Scope(tenant="acme", user="alice")
    assert claim.derivation == Derivation.USER
    assert claim.memory_type == MemoryType.SEMANTIC
    assert claim.sources == []
    store.close()


def test_get_claim_returns_none_on_404():
    transport = FakeTransport().on("GET", "/v1/memories/missing",
                                   json_response(404, {"error": "not_found"}))
    store = make_store(transport)
    assert store.get_claim("missing") is None
    store.close()


def test_get_claim_reraises_other_http_errors():
    transport = FakeTransport().on("GET", "/v1/memories/broken",
                                   json_response(500, {"error": "boom"}))
    store = make_store(transport)
    with pytest.raises(httpx.HTTPStatusError):
        store.get_claim("broken")
    store.close()


def test_get_claims_fetches_each_id_and_drops_missing_ones():
    transport = (FakeTransport()
                .on("GET", "/v1/memories/a", json_response(200, memory_body(id="a")))
                .on("GET", "/v1/memories/b", json_response(404, {"error": "not_found"}))
                .on("GET", "/v1/memories/c", json_response(200, memory_body(id="c"))))
    store = make_store(transport)
    out = store.get_claims(["a", "b", "c"])
    assert set(out) == {"a", "c"}
    store.close()


def test_get_claim_round_trips_a_fully_populated_body():
    body = memory_body(
        polarity=-1, confidence=0.4, salience=0.2, observation_count=3,
        source_ids=["ep_1", "ep_2"], derivation="consolidation", extractor=None,
        metadata={"k": "v"},
        valid_time={"valid_from": "2024-01-01T00:00:00+00:00",
                   "valid_to": "2024-06-01T00:00:00+00:00"},
        transaction_time={"recorded_at": "2024-01-01T00:00:00+00:00",
                          "invalidated_at": "2024-06-01T00:00:00+00:00",
                          "invalidated_by": "clm_2"},
    )
    transport = FakeTransport().on("GET", "/v1/memories/clm_1", json_response(200, body))
    store = make_store(transport)
    claim = store.get_claim("clm_1")
    assert claim.polarity == -1
    assert claim.confidence == 0.4
    assert claim.salience == 0.2
    assert claim.observation_count == 3
    assert claim.sources == ["ep_1", "ep_2"]
    assert claim.derivation == Derivation.CONSOLIDATION
    assert claim.extractor == ""
    assert claim.meta == {"k": "v"}
    assert claim.valid_to is not None
    assert claim.invalidated_at is not None
    assert claim.invalidated_by == "clm_2"
    store.close()


# -- batch: a passthrough, not a transaction --------------------------------------------

def test_batch_yields_self_and_is_not_atomic():
    store = RemoteStore(base_url="https://app.memvara.dev", api_key="k")
    with store.batch() as b:
        assert b is store
    store.close()


# -- purge / erase_claim: the two writes that do map --------------------------------

def test_purge_a_session_scope_sends_no_confirm_tenant():
    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        body = _json.loads(request.content)
        assert body == {"scope": {"user": "alice", "agent": None, "session": "s1"}}
        return httpx.Response(200, json={"counts": {"claims": 2, "episodes": 1,
                                                    "embeddings": 2, "entities": 0}})

    transport = FakeTransport().on("POST", "/v1/erasures", handler)
    store = make_store(transport)
    counts = store.purge(Scope(tenant="acme", user="alice", session="s1"))
    assert counts == {"claims": 2, "episodes": 1, "embeddings": 2, "entities": 0}
    store.close()


def test_purge_a_whole_tenant_sends_confirm_tenant():
    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        body = _json.loads(request.content)
        assert body["confirm_tenant"] == "acme"
        assert body["scope"] == {"user": None, "agent": None, "session": None}
        return httpx.Response(200, json={"counts": None})

    transport = FakeTransport().on("POST", "/v1/erasures", handler)
    store = make_store(transport)
    counts = store.purge(Scope(tenant="acme"))
    assert counts == {}
    store.close()


def test_erase_claim_found():
    transport = FakeTransport().on("POST", "/v1/erasures",
                                   json_response(200, {"erased": True}))
    store = make_store(transport)
    counts = store.erase_claim("clm_1")
    assert counts == {"claims": 1, "episodes": 0, "embeddings": 0, "entities": 0}
    store.close()


def test_erase_claim_not_found():
    transport = FakeTransport().on("POST", "/v1/erasures",
                                   json_response(200, {"erased": False}))
    store = make_store(transport)
    assert store.erase_claim("clm_missing") == {"claims": 0, "episodes": 0,
                                                "embeddings": 0, "entities": 0}
    store.close()


def test_erase_claim_passes_the_sources_flag():
    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        body = _json.loads(request.content)
        assert body == {"memory_id": "clm_1", "sources": True}
        return httpx.Response(200, json={"erased": True})

    transport = FakeTransport().on("POST", "/v1/erasures", handler)
    store = make_store(transport)
    store.erase_claim("clm_1", sources=True)
    store.close()


# -- stats --------------------------------------------------------------------------

def test_stats_returns_tenant_counts():
    transport = FakeTransport().on(
        "GET", "/v1/stats",
        json_response(200, {"scope": {"tenant": "acme"},
                            "tenant_counts": {"claims": 5, "episodes": 3}}))
    store = make_store(transport)
    assert store.stats() == {"claims": 5, "episodes": 3}
    assert store.stats(tenant="acme") == {"claims": 5, "episodes": 3}
    store.close()


def test_stats_for_a_different_tenant_is_rejected():
    transport = FakeTransport().on(
        "GET", "/v1/stats",
        json_response(200, {"scope": {"tenant": "acme"}, "tenant_counts": {}}))
    store = make_store(transport)
    with pytest.raises(ValueError, match="acme"):
        store.stats(tenant="other")
    store.close()


# -- _request: 204 / empty body -------------------------------------------------------

def test_request_returns_none_on_204():
    transport = FakeTransport().on("POST", "/v1/erasures",
                                   lambda r: httpx.Response(204))
    store = make_store(transport)
    result = store._request("POST", "/v1/erasures", json={})
    assert result is None
    store.close()


# -- every method with no REST equivalent raises NotImplementedError, with a message ----

def _store() -> RemoteStore:
    return RemoteStore(base_url="https://app.memvara.dev", api_key="k")


SCOPE = Scope(tenant="acme")
CLAIM = Claim(subject="user", predicate="lives_in", object="Lisbon", scope=SCOPE)


@pytest.mark.parametrize("call", [
    lambda s: s.add_episode(Episode(content="hi", scope=SCOPE)),
    lambda s: s.get_episode("ep_1"),
    lambda s: s.find_episode_by_hash("acme", "hash"),
    lambda s: s.get_episodes(["ep_1"]),
    lambda s: list(s.iter_episodes("acme")),
    lambda s: s.scope_episodes([SCOPE]),
    lambda s: s.put_claim(CLAIM),
    lambda s: s.competing_claims("acme", "fk"),
    lambda s: s.count_competing("acme", "fk"),
    lambda s: s.find_by_value("acme", "vk"),
    lambda s: s.claims_citing("acme", "ep_1"),
    lambda s: s.slot_history("acme", "fk"),
    lambda s: s.adjacent("acme", ["k1"]),
    lambda s: s.residue("clm_1"),
    lambda s: s.erasure_record("clm_1"),
    lambda s: s.invalidate("clm_1", T0, None),
    lambda s: s.set_valid_to("clm_1", None),
    lambda s: s.reinforce("clm_1", 1.0, 1, []),
    lambda s: s.set_embedding("clm_1", np.zeros(4)),
    lambda s: s.get_embedding("clm_1"),
    lambda s: s.set_episode_embedding("ep_1", np.zeros(4)),
    lambda s: s.get_episode_embedding("ep_1"),
    lambda s: s.clear_embeddings(),
    lambda s: s.candidate_ids([SCOPE]),
    lambda s: s.lexical_search("q", [SCOPE], 5),
    lambda s: s.vector_search(np.zeros(4), [SCOPE], 5),
    lambda s: s.episode_candidate_ids([SCOPE]),
    lambda s: s.lexical_search_episodes("q", [SCOPE], 5),
    lambda s: s.vector_search_episodes(np.zeros(4), [SCOPE], 5),
    lambda s: s.erase_episode("ep_1"),
    lambda s: s.put_spec(object(), "acme"),
    lambda s: s.all_specs("acme"),
    lambda s: s.put_entity("e1", "Lisbon", ("lisbon",)),
    lambda s: s.all_entities("acme"),
    lambda s: list(s.iter_claims("acme")),
])
def test_unmapped_methods_raise_with_an_explanatory_message(call):
    store = _store()
    with pytest.raises(NotImplementedError) as caught:
        call(store)
    message = str(caught.value)
    assert "no REST equivalent" in message
    store.close()



def test_the_wired_list_names_exactly_the_methods_that_do_not_raise():
    """`RemoteStore.WIRED` is a literal, and a literal drifts.

    It is read by `memvara.server.config.build_memvara`, which refuses to start a cloud
    server unless the engine's needs are in it — so a name that lands on this list without
    an endpoint behind it does not merely mislead a reader, it un-refuses a configuration
    that cannot work. And a name that falls *off* it keeps a working capability hidden.

    Derived from the source rather than restated: a method whose body raises
    `NotImplementedError` is unwired, and everything else on the protocol is wired. That
    is the same rule a reader applies, applied by a machine.
    """
    protocol = {name for name in dir(Store) if not name.startswith("_")}
    wired = set()
    for name in protocol:
        function = getattr(RemoteStore, name, None)
        if function is None:
            continue          # absent entirely, which is not "wired" either
        if "raise NotImplementedError" not in inspect.getsource(function):
            wired.add(name)
    assert RemoteStore.WIRED == wired, (
        f"WIRED says {sorted(RemoteStore.WIRED)} and the source says {sorted(wired)}"
    )


def test_a_memvara_can_still_be_constructed_over_a_remote_store():
    """The library tolerates it; only the *server* refuses to start one.

    Two constructor-time probes reach for methods `RemoteStore` does not have —
    `all_specs`, to rehydrate learned predicate schema, and the embedding-width probe in
    `memvara/embed/fingerprint.py`. Both are present on the object and raise, which no
    `getattr` guard sees, and both catch it and fall back to "nothing to rehydrate" and
    "unknown width". Without that, `Memvara(store=RemoteStore(...))` would be
    unconstructible and the object would have no honest way to exist at all.

    It is worth being able to construct: a caller wiring their own erasure or stats job
    against the cloud facade needs the four methods that *are* wired, and
    `memvara.server.config.build_memvara` refuses only the thing that cannot work —
    running the engine's read and write paths over it.
    """
    store = RemoteStore(base_url="https://example.invalid", api_key="k")
    memory = Memvara(store=store, llm=NullLLM(), embedder=HashingEmbedder(dim=32),
                     tenant="acme", user="alice")
    try:
        assert memory.store is store
        assert memory.default_scope.tenant == "acme"
    finally:
        memory.close()
