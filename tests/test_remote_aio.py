"""The async remote client uses a real async transport, not a thread.

Written with `asyncio.run` rather than `pytest-asyncio`, for `tests/test_aio.py`'s own
reason: a plugin with its own event-loop policy, fixture scoping and strict/auto modes is
a larger dependency than the thing it would be testing, and nothing here needs any of
those three features — a mock transport and one coroutine per test is enough. An earlier
version of this file used `pytest.mark.asyncio`; nothing about the transport under test
required it, so it was dropped rather than kept as a second, unexplained answer to a
question this repository had already settled.

Coverage of the surface is checked at three levels a name-only diff cannot tell apart.
`test_every_sync_method_has_an_async_twin_of_the_same_name` only diffs `dir()` output —
it would pass just as well if an async twin existed under the right name but took
different arguments, in a different order, or with a different default.
`test_async_twins_match_their_sync_signatures` closes that gap by comparing each pair's
full parameter list — name, kind and default, not just name — with `self` stripped from
both. `test_every_async_twin_is_actually_a_coroutine` closes the remaining one: a plain
`def` that happened to share its twin's name and signature would pass both checks above
and fail only when awaited, at the call site, in whichever request first reached it. All
three run again for `ScopedRemoteMemvara` against `AsyncScopedRemoteMemvara`, which the
first version of this file did not check at all.

`test_the_transport_is_a_real_async_client_and_not_a_thread_wrapper` asserts that the
mock injected at `mem._http._client` is an `httpx.AsyncClient`, which is what every
`await` in this file actually drives. It does not by itself prove the retry loop behind
that client — idempotency, retry-on-transport-error, attempt bounds — behaves like the
sync client's; `tests/test_remote_client.py`'s async section is what proves that, one
transport-level test at a time, against `AsyncHttpClient` directly rather than through
this facade. `test_every_write_method_carries_an_idempotency_key` below is the one test
in this file that reaches through the facade to the transport: it is what would have
caught a write method built with `write=False`, which no purely name/signature/coroutine
check can — that mistake produces a method with the right name, the right signature and
the right `async def`, and fails only in what it sends.
"""
import asyncio
import inspect
import json
from datetime import datetime, timezone

import httpx

from memvara.remote.aio import AsyncRemoteMemvara, AsyncScopedRemoteMemvara
from memvara.remote.api import RemoteMemvara, ScopedRemoteMemvara
from memvara.remote.errors import RemoteError
from memvara.types import Scope


def run(coro):
    return asyncio.run(coro)


def _client(handler):
    mem = AsyncRemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http._client = httpx.AsyncClient(base_url="https://example.test",
                                          transport=httpx.MockTransport(handler))
    return mem


def test_a_read_awaits_and_returns_the_decoded_body():
    # `/v1/stats` wraps its counts under `tenant_counts` — see `RemoteMemvara.stats`,
    # which this method mirrors. A body shaped `{"claims": 3}` was never a real response.
    mem = _client(lambda r: httpx.Response(200, json={"tenant_counts": {"claims": 3}}))

    async def main():
        assert (await mem.stats())["claims"] == 3
        await mem.aclose()

    run(main())


def test_the_transport_is_a_real_async_client_and_not_a_thread_wrapper():
    mem = _client(lambda r: httpx.Response(200, json={}))
    assert isinstance(mem._http._client, httpx.AsyncClient)
    run(mem.aclose())


def test_it_works_as_an_async_context_manager():
    mem = _client(lambda r: httpx.Response(200, json={}))

    async def main():
        async with mem as m:
            assert m is mem

    run(main())


def test_every_write_method_carries_an_idempotency_key():
    """Reaches through the facade, not just the transport — see the module docstring.

    A write method assembled with `write=False` would still have the right name, the
    right signature and the right `async def`; the only place that mistake shows up is
    in what actually goes out over the wire.
    """
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("idempotency-key"))
        path = request.url.path
        if path.endswith("/supersede") or path in ("/v1/memories", "/v1/facts"):
            return httpx.Response(200, json={
                "episode_ids": [], "added": [], "invalidated": [], "reinforced": [],
                "skipped": 0, "unextracted": 0, "llm_calls": 0, "latency_ms": 0,
                "deferred": False})
        if path.startswith("/v1/memories/") and request.method == "DELETE":
            return httpx.Response(200, json={"retired": False})
        if path == "/v1/forget":
            return httpx.Response(200, json={"retired": []})
        if path == "/v1/end":
            return httpx.Response(200, json={"ended": []})
        if path == "/v1/erasures":
            return httpx.Response(200, json={"erased": False, "counts": {}})
        return httpx.Response(200, json={})

    mem = _client(handler)

    async def main():
        await mem.add("hi")
        await mem.remember("user", "predicate_a", "value")
        await mem.supersede("clm_1", "user", "predicate_a", "value")
        await mem.forget("user", "predicate_a")
        await mem.delete("clm_1")
        await mem.end(claim_id="clm_1")
        await mem.erase("clm_1")
        await mem.purge()
        await mem.consolidate()
        await mem.aclose()

    run(main())
    assert seen and all(seen), f"a write went out with no Idempotency-Key: {seen}"


# -- surface parity: names, signatures, coroutine-ness ------------------------------


def _methods(cls: type) -> set[str]:
    """Real methods only. `dir()` also returns the `scope`/`memvara` properties on the
    scoped views, and a property has no signature to compare and is never a coroutine —
    comparing it as either would be comparing the wrong kind of thing."""
    return {n for n in dir(cls) if not n.startswith("_")
            and inspect.isfunction(getattr(cls, n))}


def _signature(fn: object) -> list[tuple[str, object, object]]:
    """`(name, kind, default)` per parameter, `self` dropped. Same names is not the same
    as same signatures — see the module docstring."""
    params = list(inspect.signature(fn).parameters.values())[1:]
    return [(p.name, p.kind, p.default) for p in params]


def _mismatched_signatures(sync_cls: type, async_cls: type,
                           names: set[str]) -> dict[str, tuple[object, object]]:
    return {name: (_signature(getattr(sync_cls, name)), _signature(getattr(async_cls, name)))
            for name in names
            if _signature(getattr(sync_cls, name)) != _signature(getattr(async_cls, name))}


def test_every_sync_method_has_an_async_twin_of_the_same_name():
    sync = _methods(RemoteMemvara)
    asyn = _methods(AsyncRemoteMemvara)
    missing = sync - asyn - {"close"}
    assert not missing, f"async client is missing: {sorted(missing)}"


def test_async_twins_match_their_sync_signatures():
    mismatched = _mismatched_signatures(RemoteMemvara, AsyncRemoteMemvara,
                                        _methods(RemoteMemvara) - {"close"})
    assert not mismatched, f"signature drift: {mismatched}"


def test_every_async_twin_is_actually_a_coroutine():
    """`scope()` is excused for `AsyncMemvara.scope()`'s own reason: it binds four
    strings and touches no store, so it stays synchronous on both sides on purpose."""
    sync = _methods(RemoteMemvara) - {"close", "scope"}
    not_coroutines = {name for name in sync
                      if not inspect.iscoroutinefunction(getattr(AsyncRemoteMemvara, name))}
    assert not not_coroutines, f"not async: {sorted(not_coroutines)}"


def test_every_scoped_method_has_an_async_twin_of_the_same_name():
    sync = _methods(ScopedRemoteMemvara)
    asyn = _methods(AsyncScopedRemoteMemvara)
    missing = sync - asyn
    assert not missing, f"async scoped view is missing: {sorted(missing)}"


def test_async_scoped_twins_match_their_sync_signatures():
    mismatched = _mismatched_signatures(ScopedRemoteMemvara, AsyncScopedRemoteMemvara,
                                        _methods(ScopedRemoteMemvara))
    assert not mismatched, f"signature drift: {mismatched}"


def test_every_async_scoped_twin_is_actually_a_coroutine():
    sync = _methods(ScopedRemoteMemvara)
    not_coroutines = {
        name for name in sync
        if not inspect.iscoroutinefunction(getattr(AsyncScopedRemoteMemvara, name))}
    assert not not_coroutines, f"not async: {sorted(not_coroutines)}"


# -- every method, driven --------------------------------------------------------------
#
# The three checks above compare names, signatures and coroutine-ness, and none of them
# calls anything: they pass for a method whose body posts to the wrong endpoint, reads the
# wrong key out of the envelope, or returns the wrong half of it. The sections below drive
# each method through a mock transport and assert the same two things
# `tests/test_remote_reads.py` asserts of the sync client — the path it reached and the
# type it decoded — because either alone passes for the wrong reason.
#
# This is not duplication of the sync tests. `AsyncRemoteMemvara`'s methods are separately
# written `async def`s that share no body with `RemoteMemvara`'s; only the small helpers in
# `remote.api` are common. A method that reached `/v1/forget` where its sync twin reaches
# `/v1/end` would leave the whole sync suite green.

import pytest

from memvara.redact import CLAIM_OBJECT, CLAIM_SUBJECT, CLAIM_TEXT, EPISODE
from memvara.remote.errors import NotFound
from memvara.retrieve import EpisodeResult, Path
from memvara.types import Answer, Claim, Delta, Episode, MemoryType, Provenance, Result

from test_remote_reads import _episode, _memory, _ranking, _scope
from test_remote_writes import _receipt


@pytest.fixture
def recorded():
    """`tests/test_remote_reads.py`'s fixture, on the async client.

    Swaps `_transport` and nothing else, for that file's reason: replacing `_http` or the
    `httpx.AsyncClient` inside it would rebuild the base url, the bearer header and the
    timeout in the fixture instead of exercising what `__init__` produced.
    """
    calls = []

    def build(payload=None, **kw):
        def handler(request):
            calls.append(request)
            return httpx.Response(200, json={} if payload is None else payload)

        mem = AsyncRemoteMemvara(api_key="k", base_url="https://example.test", **kw)
        mem._http._client._transport = httpx.MockTransport(handler)
        return mem

    build.calls = calls
    return build


_LISTING = {"count": 1, "total": 47, "limit": 100, "offset": 0, "as_of": None,
            "valid_at": None, "known_at": None, "states": ["live"],
            "memories": [_memory()]}
_PATHS = {"as_of": None, "valid_at": None, "known_at": None, "count": 1,
          "paths": [{"nodes": ["alice", "berlin"], "labels": ["lives_in"], "hops": 1,
                     "score": 0.5,
                     "edges": [{"memory": _memory(), "backward": False,
                                "strength": 0.5}]}]}
_ENVELOPE = {"scope": _scope(), "visible": 2,
             "tenant_counts": {"claims": 3, "live_claims": 3, "joinable_claims": 1},
             "extractor": "fast-path-only", "read_only": True}


#: `(id, payload, call, method, path, check)` — one row per read method. The `check` is
#: what stops this being a path assertion with a decorative call in front of it: it reads
#: a field off the decoded value, so a method returning the wrong half of the envelope
#: fails here even though it reached the right endpoint.
_READS = [
    ("health", {"status": "ok", "memvara_version": "0.2.0"},
     lambda m: m.health(), "GET", "/v1/health",
     lambda body: body["status"] == "ok"),
    ("whoami", {"token_id": "tok_1", "scope": _scope(), "granted_privilege": "write",
                "effective_privilege": "write", "expires_at": None, "read_only": False},
     lambda m: m.whoami(), "GET", "/v1/whoami",
     lambda body: body["token_id"] == "tok_1"),
    ("stats", _ENVELOPE, lambda m: m.stats(), "GET", "/v1/stats",
     lambda counts: counts["claims"] == 3 and "extractor" not in counts),
    ("service", _ENVELOPE, lambda m: m.service(), "GET", "/v1/stats",
     lambda body: body["read_only"] is True and body["extractor"] == "fast-path-only"),
    ("connectivity", _ENVELOPE, lambda m: m.connectivity(), "GET", "/v1/stats",
     lambda body: body == {"live_claims": 3, "joinable_claims": 1}),
    ("search", {"as_of": None, "valid_at": None, "known_at": None, "states": ["live"],
                "count": 1,
                "results": [{"kind": "claim", "score": 0.7, "ranking": _ranking(),
                             "memory": _memory()}]},
     lambda m: m.search("where do I live"), "POST", "/v1/search",
     lambda hits: isinstance(hits[0], Result) and hits[0].claim.object == "Berlin"),
    ("recall", {"text": "Known about the user:\n- user lives in Berlin", "empty": False},
     lambda m: m.recall("where do I live"), "POST", "/v1/recall",
     lambda text: "Berlin" in text),
    ("get", _memory("cl_7"), lambda m: m.get("cl_7"), "GET", "/v1/memories/cl_7",
     lambda claim: isinstance(claim, Claim) and claim.id == "cl_7"),
    ("get_all", _LISTING, lambda m: m.get_all(), "GET", "/v1/memories",
     lambda claims: isinstance(claims[0], Claim)),
    ("count", _LISTING, lambda m: m.count(), "GET", "/v1/memories",
     lambda total: total == 47),
    ("history", {"subject": "user", "predicate": "lives_in", "scope": _scope(),
                 "as_of": None, "valid_at": None, "known_at": None, "count": 1,
                 "timeline": [_memory()]},
     lambda m: m.history("user", "lives_in"), "GET", "/v1/history",
     lambda claims: isinstance(claims[0], Claim)),
    ("why", {"memory": _memory("cl_2"), "derivation": "user", "extractor": "api",
             "sources": [_episode()], "superseded": []},
     lambda m: m.why("cl_2"), "GET", "/v1/memories/cl_2/why",
     lambda prov: isinstance(prov, Provenance) and prov.claim.id == "cl_2"),
    ("ask", {"question": "where did I live", "at": "2026-01-01T00:00:00Z",
             "count": 1, "text": "Then and now agree.",
             "readings": [{"subject": "user", "predicate": "lives_in",
                           "now": [_memory()], "then": [_memory()],
                           "stated": [_memory()], "diverged": False, "moved": False}]},
     lambda m: m.ask("where did I live"), "POST", "/v1/ask",
     lambda answer: isinstance(answer, Answer)
     and answer.readings[0].predicate == "lives_in"),
    ("since", {"since": "2026-01-01T00:00:00Z", "added": [_memory()], "gone": []},
     lambda m: m.since(datetime(2026, 1, 1, tzinfo=timezone.utc)), "GET", "/v1/since",
     lambda delta: isinstance(delta, Delta) and len(delta.added) == 1),
    ("produced", {"episode_id": "ep_1", "as_of": None, "valid_at": None,
                  "known_at": None, "count": 1, "memories": [_memory()]},
     lambda m: m.produced("ep_1"), "GET", "/v1/episodes/ep_1/produced",
     lambda claims: isinstance(claims[0], Claim)),
    ("neighborhood", _PATHS, lambda m: m.neighborhood("alice"), "GET",
     "/v1/neighborhood", lambda paths: isinstance(paths[0], Path)),
    ("paths_between", _PATHS, lambda m: m.paths_between("alice", "berlin"), "GET",
     "/v1/paths", lambda paths: isinstance(paths[0], Path)),
    ("standing", {"count": 1, "limit": 5, "truncated": False,
                  "memories": [_memory(predicate="prefers", obj="pytest")]},
     lambda m: m.standing(k=5), "GET", "/v1/standing",
     lambda claims: isinstance(claims[0], Claim)),
]


#: The one route that carries no scope, because it carries no credential either: it
#: reports whether the deployment is answering and touches no store, so a `user=` on it
#: would be a scope sent where nothing reads one.
_UNSCOPED = {"health"}


@pytest.mark.parametrize("name, payload, call, method, path, check", _READS,
                         ids=[row[0] for row in _READS])
def test_each_read_reaches_its_endpoint_and_decodes_what_that_route_answers(
        recorded, name, payload, call, method, path, check):
    mem = recorded(payload, user="alice")

    async def main():
        value = await call(mem)
        await mem.aclose()
        return value

    value = run(main())
    sent = recorded.calls[-1]
    assert sent.method == method
    assert sent.url.path == path
    assert check(value), f"{path} decoded to {value!r}"
    # The scope rides on every call rather than being a constructor argument the methods
    # forget to use — the property `ScopedRemoteMemvara` rests on, asserted per route
    # because it is `_params()` at each call site that puts it there.
    assert dict(sent.url.params).get("user") == (None if name in _UNSCOPED else "alice")


def test_connectivity_is_empty_when_the_deployment_does_not_report_joins(recorded):
    """`{}` is not a store with nothing in it. An empty store answers two zeros; a backend
    that cannot measure the join says nothing at all, and reading a missing key as zero
    would report a star nobody measured."""
    mem = recorded({"scope": _scope(), "visible": 0, "tenant_counts": {"claims": 5},
                    "extractor": "fast-path-only", "read_only": False})

    async def main():
        value = await mem.connectivity()
        await mem.aclose()
        return value

    assert run(main()) == {}


def test_search_decodes_an_episode_hit_as_an_episode_not_a_claim(recorded):
    """`include_episodes` mixes two kinds into one list and they are not interchangeable.
    The wire discriminates on `kind`; handing an episode hit to the claim path raises on
    the missing `memory` key, which is what a caller sees the moment they ask for turns."""
    mem = recorded({"as_of": None, "valid_at": None, "known_at": None,
                    "states": ["live"], "count": 1,
                    "results": [{"kind": "episode", "score": 0.3,
                                 "ranking": _ranking(applicable=False),
                                 "episode": _episode()}]})

    async def main():
        hits = await mem.search("berlin", include_episodes=True)
        await mem.aclose()
        return hits

    hits = run(main())
    assert isinstance(hits[0], EpisodeResult)
    assert hits[0].episode.content == "I live in Berlin"


def test_ranked_reaches_the_wire_only_when_asked_for(recorded):
    """`AsyncRemoteMemvara`'s own request-building code, not shared with the sync
    client's — the same precedent `anchored`'s async test sets."""
    from memvara.types import SearchResults

    mem = recorded({"as_of": None, "valid_at": None, "known_at": None,
                    "states": ["live"], "count": 0, "results": [],
                    "selection": {"outcome": "applied", "candidates": 40, "kept": 5}})

    async def main():
        plain = await mem.search("q", include_episodes=True)
        ranked = await mem.search("q", include_episodes=True, ranked=True)
        await mem.aclose()
        return plain, ranked

    plain, ranked = run(main())
    sent_plain = json.loads(recorded.calls[-2].content)
    sent_ranked = json.loads(recorded.calls[-1].content)
    assert "ranked" not in sent_plain
    assert sent_ranked["ranked"] is True
    assert isinstance(ranked, SearchResults) and isinstance(plain, SearchResults)
    assert ranked.selection is not None and ranked.selection.outcome == "applied"


def test_recall_refuses_a_budget_rather_than_dropping_it(recorded):
    """`POST /v1/recall` renders server-side and takes no budget. A ceiling silently not
    applied is an oversized prompt with nothing to notice it by."""
    mem = recorded({"text": "x", "empty": False})

    async def main():
        with pytest.raises(ValueError, match="budget"):
            await mem.recall("q", budget=200)
        await mem.aclose()

    run(main())
    assert recorded.calls == []


@pytest.mark.parametrize("call", [
    lambda m: m.get("cl_missing"),
    lambda m: m.why("cl_missing"),
])
def test_a_missing_memory_reads_as_none_rather_than_raising(call):
    """`Memvara.get` and `Memvara.why` both answer None, and `tools.py` branches on that.
    Raising here would make a missing id an exception on the hosted path alone — and the
    facade answers 404 for an id in another tenant too, so this must not become a way to
    test whether an id exists elsewhere."""
    mem = AsyncRemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http._client._transport = httpx.MockTransport(
        lambda r: httpx.Response(404, json={"error": {"code": "not_found",
                                                      "message": "x"}}))

    async def main():
        value = await call(mem)
        await mem.aclose()
        return value

    assert run(main()) is None


def test_a_failure_that_is_not_a_404_still_reaches_the_caller(recorded):
    """The `except NotFound` in `get` and `why` is narrow on purpose. Swallowing every
    error there would turn an expired credential into "no such memory", and a caller would
    read an empty store where there is an authentication problem."""
    mem = AsyncRemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http._client._transport = httpx.MockTransport(
        lambda r: httpx.Response(403, json={"error": {"code": "forbidden_scope",
                                                      "message": "not yours"}}))

    async def main():
        with pytest.raises(RemoteError) as caught:
            await mem.get("cl_1")
        await mem.aclose()
        return caught.value

    assert not isinstance(run(main()), NotFound)


# -- the routing decision, on this side too --------------------------------------------
#
# `ended` and `retired` are different statements about whether a stored fact was ever
# true, they go to different endpoints, and `AsyncRemoteMemvara.forget` and `.delete` each
# carry their own copy of the branch that decides — shared with nothing in `api.py`. Until
# these tests existed, sending `close="ended"` to `DELETE /v1/memories/{id}` left the whole
# suite green, which is exactly the mistake `memvara/types.py` calls the one that cannot be
# found by reading the data afterwards.


def test_ending_goes_to_the_end_route_and_never_to_delete(recorded):
    mem = recorded({"memory_id": "cl_1", "subject": None, "predicate": None,
                    "count": 1, "ended": [_memory()], "erased": False})

    async def main():
        value = await mem.delete("cl_1", close="ended")
        await mem.aclose()
        return value

    assert run(main()) is True
    assert recorded.calls[-1].method == "POST"
    assert recorded.calls[-1].url.path == "/v1/end"


def test_retiring_goes_to_the_delete_route(recorded):
    mem = recorded({"id": "cl_1", "retired": True, "erased": False})

    async def main():
        value = await mem.delete("cl_1", close="retired")
        await mem.aclose()
        return value

    assert run(main()) is True
    assert recorded.calls[-1].method == "DELETE"
    assert recorded.calls[-1].url.path == "/v1/memories/cl_1"


def test_ending_a_slot_goes_to_the_end_route_and_never_to_forget(recorded):
    """`/v1/forget` has no `close` field, so a client posting there with `close="ended"`
    would file every ending as a retirement. `server/tools.py` makes exactly this call:
    `forget(..., close="ended")` is how `memory_end` closes a slot."""
    mem = recorded({"memory_id": None, "subject": "user", "predicate": "works_at",
                    "count": 1, "ended": [_memory(predicate="works_at", obj="Acme")],
                    "erased": False})

    async def main():
        claims = await mem.forget("user", "works_at", close="ended")
        await mem.aclose()
        return claims

    claims = run(main())
    assert recorded.calls[-1].url.path == "/v1/end"
    assert isinstance(claims[0], Claim)


def test_forgetting_a_slot_retires_it_through_the_forget_route(recorded):
    mem = recorded({"subject": "user", "predicate": "works_at", "count": 1,
                    "retired": [_memory(predicate="works_at", obj="Acme")],
                    "erased": False})

    async def main():
        claims = await mem.forget("user", "works_at")
        await mem.aclose()
        return claims

    claims = run(main())
    assert recorded.calls[-1].url.path == "/v1/forget"
    assert isinstance(claims[0], Claim)


@pytest.mark.parametrize("call", [
    lambda m: m.delete("cl_1", close="deleted"),
    lambda m: m.forget("user", "works_at", close="removed"),
    lambda m: m.supersede("cl_1", "user", "likes", "tea", close="replaced"),
])
def test_an_unknown_closure_reaches_no_endpoint_at_all(recorded, call):
    """Validated before the request rather than server-side: a closure checked after the
    fact would already have written something by the time the error came back."""
    mem = recorded()

    async def main():
        with pytest.raises(ValueError):
            await call(mem)
        await mem.aclose()

    run(main())
    assert recorded.calls == []


def test_end_needs_exactly_one_addressing_mode(recorded):
    """One memory against every current value of a slot: different blast radii, so a
    silent default on that choice is not a convenience."""
    mem = recorded()

    async def main():
        with pytest.raises(TypeError):
            await mem.end()
        with pytest.raises(TypeError):
            await mem.end(claim_id="cl_1", predicate="works_at")
        await mem.aclose()

    run(main())
    assert recorded.calls == []


def test_end_by_slot_sends_subject_and_predicate_and_no_id(recorded):
    mem = recorded({"memory_id": None, "subject": "user", "predicate": "works_at",
                    "count": 0, "ended": [], "erased": False})

    async def main():
        await mem.end(predicate="works_at")
        await mem.aclose()

    run(main())
    body = json.loads(recorded.calls[-1].read())
    assert body["subject"] == "user" and body["predicate"] == "works_at"
    assert "memory_id" not in body


# -- the remaining writes, and what each one sends -------------------------------------


def test_add_reaches_the_ingest_endpoint_and_returns_a_receipt(recorded):
    mem = recorded(_receipt())

    async def main():
        receipt = await mem.add("I moved to Berlin")
        await mem.aclose()
        return receipt

    receipt = run(main())
    assert recorded.calls[-1].url.path == "/v1/memories"
    assert receipt.episode_ids == ["ep_1"]


@pytest.mark.parametrize("messages, expected", [
    ([{"role": "user", "content": "hi", "channel": "slack"}],
     [{"role": "user", "content": "hi", "metadata": {"channel": "slack"}}]),
    (Episode(role="user", content="hi",
             ts=datetime(2026, 1, 1, tzinfo=timezone.utc)),
     [{"role": "user", "content": "hi", "ts": "2026-01-01T00:00:00+00:00",
       "metadata": {}}]),
    (["hi"], [{"content": "hi"}]),
    ("hi", "hi"),
], ids=["mapping-with-unknown-key", "episode", "string-in-a-list", "bare-string"])
def test_add_sends_each_message_shape_the_way_the_facade_spells_it(recorded, messages,
                                                                   expected):
    """Four input shapes, one wire format, and the first row is the one with teeth: the
    facade's request models are `extra="forbid"`, so a key beside `role` and `content` is
    a 422 rather than a field somebody quietly loses. It is folded into `metadata`, as
    `Memvara.add` folds it locally.

    The last two rows are one character apart and take different branches: a bare string
    is the whole conversation and goes out as `messages: "hi"`, while a string inside a
    sequence is one turn among others and goes out as a message object.
    """
    mem = recorded(_receipt())

    async def main():
        await mem.add(messages)
        await mem.aclose()

    run(main())
    sent = json.loads(recorded.calls[-1].read())
    assert sent["messages"] == expected


def test_remember_splits_cited_ids_from_turns_it_must_store(recorded):
    """One mixed sequence in, two fields out: over HTTP, citing a stored turn and writing
    a new one are different acts."""
    mem = recorded(_receipt())

    async def main():
        await mem.remember("user", "likes", "tea",
                           sources=["ep_9", {"content": "I like tea"}])
        await mem.aclose()

    run(main())
    body = json.loads(recorded.calls[-1].read())
    assert body["source_ids"] == ["ep_9"]
    assert body["sources"] == [{"content": "I like tea", "metadata": {}}]


def test_supersede_forwards_close_verbatim_rather_than_defaulting(recorded):
    """A mutation log records that a value changed and never which of the two it was, so
    restating the default here would file every correction as a world event."""
    mem = recorded(_receipt())

    async def main():
        await mem.supersede("cl_old", "user", "lives_in", "Berlin", close="retired")
        await mem.aclose()

    run(main())
    assert recorded.calls[-1].url.path == "/v1/memories/cl_old/supersede"
    assert json.loads(recorded.calls[-1].read())["close"] == "retired"


def test_purge_sends_the_bound_scope_and_returns_the_per_table_counts(recorded):
    """The counts are the evidence a deletion request has to be answered with, measured by
    the store rather than assembled by the facade."""
    mem = recorded({"target": "scope", "memory_id": None,
                    "scope": {"tenant": "t", "user": "alice", "agent": None,
                              "session": None},
                    "erased": True, "counts": {"claims": 4, "episodes": 9},
                    "sources_erased": None, "audit_subject_linkable": False},
                   user="alice")

    async def main():
        counts = await mem.purge()
        await mem.aclose()
        return counts

    counts = run(main())
    assert recorded.calls[-1].url.path == "/v1/erasures"
    assert json.loads(recorded.calls[-1].read())["scope"] == {"user": "alice"}
    assert counts == {"claims": 4, "episodes": 9}


def test_erase_reaches_the_erasure_endpoint_and_reports_whether_it_landed(recorded):
    mem = recorded({"target": "memory", "memory_id": "cl_1", "scope": None,
                    "erased": True, "counts": None, "sources_erased": False,
                    "audit_subject_linkable": None})

    async def main():
        value = await mem.erase("cl_1")
        await mem.aclose()
        return value

    assert run(main()) is True
    assert recorded.calls[-1].url.path == "/v1/erasures"


def test_consolidate_reaches_the_maintenance_endpoint_and_returns_a_job(recorded):
    """A job rather than counts: the endpoint answers 202 before the pass starts, so the
    outcome is on the job and in no status code."""
    mem = recorded({"id": "job_1", "kind": "consolidate", "tenant": "t",
                    "status": "queued", "created_at": "2026-01-01T00:00:00Z",
                    "started_at": None, "finished_at": None, "result": None,
                    "error": None, "links": {"self": "/v1/jobs/job_1"}})

    async def main():
        job = await mem.consolidate()
        await mem.aclose()
        return job

    assert run(main())["status"] == "queued"
    assert recorded.calls[-1].url.path == "/v1/maintenance/consolidate"


# -- redaction, which has to happen before the text leaves the process ------------------


def test_text_is_redacted_on_its_way_out_and_not_after_it_has_left(recorded):
    """Every field the policy is offered, checked on the wire.

    Running server-side would be the alternative and it is not redaction: the raw text has
    already crossed the network by then. The policy is handed the field name so a
    deployment can be aggressive on raw turns and conservative on claim objects, which is
    why the assertion below is per-field rather than "something was replaced".
    """
    class Loud:
        def redact(self, text, *, field, scope):
            return f"[{field}:{scope.user}]"

    mem = recorded(_receipt(), user="alice", redactor=Loud())

    async def main():
        await mem.add(Episode(role="user", content="I live at 12 Acacia Avenue"))
        await mem.remember("user", "lives_in", "12 Acacia Avenue",
                           text="user lives at 12 Acacia Avenue")
        await mem.aclose()

    run(main())
    fact = json.loads(recorded.calls[-1].read())
    assert fact["subject"] == f"[{CLAIM_SUBJECT}:alice]"
    assert fact["object"] == f"[{CLAIM_OBJECT}:alice]"
    assert fact["text"] == f"[{CLAIM_TEXT}:alice]"
    assert fact["predicate"] == "lives_in", "a predicate is a schema term, not free text"
    turn = json.loads(recorded.calls[0].read())["messages"][0]
    assert turn["content"] == f"[{EPISODE}:alice]"


# -- the scoped view -------------------------------------------------------------------


def test_the_scoped_view_narrows_and_leaves_the_client_it_came_from_alone():
    """A view is a second handle, not a mutation. A `scope()` that rebound the client
    would silently move every later call made through the original."""
    mem = AsyncRemoteMemvara(api_key="k", base_url="https://example.test", user="alice")
    view = mem.scope(agent="a1")
    assert isinstance(view, AsyncScopedRemoteMemvara)
    assert view.scope == Scope("default", "alice", "a1", None)
    assert view.memvara.default_scope == Scope("default", "alice", "a1", None)
    assert mem.default_scope.agent is None
    run(mem.aclose())


def test_the_view_shares_one_connection_pool_with_the_client_it_came_from():
    """Two handles on one deployment and one credential. A second pool would be two
    idempotency stores as well, which is what the transport's retry depends on."""
    mem = AsyncRemoteMemvara(api_key="k", base_url="https://example.test")
    assert mem.scope(user="alice").memvara._http is mem._http
    run(mem.aclose())


_SCOPED_CALLS = [
    (row[0], row[1], row[2], row[4]) for row in _READS
    # `service` is the one read absent from the scoped view: a server asks it once, at
    # startup, before it has narrowed to anything.
    if row[0] != "service"
] + [
    ("add", _receipt(), lambda v: v.add("hi"), "/v1/memories"),
    ("remember", _receipt(), lambda v: v.remember("user", "likes", "tea"), "/v1/facts"),
    ("supersede", _receipt(), lambda v: v.supersede("cl_1", "user", "likes", "tea"),
     "/v1/memories/cl_1/supersede"),
    ("forget", {"subject": "user", "predicate": "likes", "count": 0, "retired": [],
                "erased": False}, lambda v: v.forget("user", "likes"), "/v1/forget"),
    ("delete", {"id": "cl_1", "retired": True, "erased": False},
     lambda v: v.delete("cl_1"), "/v1/memories/cl_1"),
    ("end", {"memory_id": "cl_1", "subject": None, "predicate": None, "count": 1,
             "ended": [_memory()], "erased": False},
     lambda v: v.end(claim_id="cl_1"), "/v1/end"),
    ("erase", {"target": "memory", "memory_id": "cl_1", "scope": None, "erased": True,
               "counts": None, "sources_erased": False,
               "audit_subject_linkable": None},
     lambda v: v.erase("cl_1"), "/v1/erasures"),
    ("purge", {"target": "scope", "memory_id": None, "scope": None, "erased": True,
               "counts": {}, "sources_erased": None, "audit_subject_linkable": False},
     lambda v: v.purge(), "/v1/erasures"),
    ("consolidate", {"id": "job_1", "status": "queued"}, lambda v: v.consolidate(),
     "/v1/maintenance/consolidate"),
]


@pytest.mark.parametrize("name, payload, call, path", _SCOPED_CALLS,
                         ids=[row[0] for row in _SCOPED_CALLS])
def test_each_scoped_method_delegates_to_the_same_endpoint_at_the_bound_scope(
        recorded, name, payload, call, path):
    """Every method on the scoped view is a one-line forward, and a one-line forward is
    exactly where a wrong argument name or a dropped keyword hides: it type-checks, it
    reads correctly, and it reaches a different endpoint or a different scope.

    Both halves are asserted. The path says the delegation landed where the unscoped
    method lands; the query parameters say it ran at the scope the view was bound to, and
    not at the client's wider one.
    """
    mem = recorded(payload, user="alice")
    view = mem.scope(agent="a1")

    async def main():
        await call(view)
        await mem.aclose()

    run(main())
    params = dict(recorded.calls[-1].url.params)
    assert recorded.calls[-1].url.path == path
    if name in _UNSCOPED:
        assert "user" not in params
        return
    assert params["user"] == "alice" and params["agent"] == "a1"
    assert "session" not in params


def test_both_classes_say_which_scope_they_are_bound_to_when_printed():
    """`repr` is what a traceback and a debugger show, and a client whose scope is
    invisible there is one whose scope gets assumed."""
    mem = AsyncRemoteMemvara(api_key="k", base_url="https://example.test", user="alice")
    assert "alice" in repr(mem)
    assert "alice" in repr(mem.scope(agent="a1"))
    assert "a1" in repr(mem.scope(agent="a1"))
    run(mem.aclose())
