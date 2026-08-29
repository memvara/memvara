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

import httpx

from memvara.remote.aio import AsyncRemoteMemvara, AsyncScopedRemoteMemvara
from memvara.remote.api import RemoteMemvara, ScopedRemoteMemvara


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
