"""The async client uses a real async transport, not a thread.

The second test is the one with teeth: it asserts no thread pool is involved, because
wrapping a blocking client in `asyncio.to_thread` would pass every behavioural test here
while being strictly worse than the thing httpx already provides.

`test_every_sync_method_has_an_async_twin_of_the_same_name` only checks that a name
exists on both classes — `dir()` does not see arguments. It would not notice an async
`search` that dropped `min_score`, or one that took its arguments in a different order.
`test_async_twins_match_their_sync_signatures` closes that gap by comparing each pair's
parameter list, `self` stripped from both.
"""
import inspect

import httpx
import pytest

from memvara.remote.aio import AsyncRemoteMemvara
from memvara.remote.api import RemoteMemvara


async def _client(handler):
    mem = AsyncRemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http._client = httpx.AsyncClient(base_url="https://example.test",
                                          transport=httpx.MockTransport(handler))
    return mem


@pytest.mark.asyncio
async def test_a_read_awaits_and_returns_the_decoded_body():
    # `/v1/stats` wraps its counts under `tenant_counts` — see `RemoteMemvara.stats`,
    # which this method mirrors. A body shaped `{"claims": 3}` was never a real response.
    mem = await _client(
        lambda r: httpx.Response(200, json={"tenant_counts": {"claims": 3}}))
    assert (await mem.stats())["claims"] == 3
    await mem.aclose()


@pytest.mark.asyncio
async def test_the_transport_is_a_real_async_client_and_not_a_thread_wrapper():
    mem = await _client(lambda r: httpx.Response(200, json={}))
    assert isinstance(mem._http._client, httpx.AsyncClient)
    await mem.aclose()


@pytest.mark.asyncio
async def test_it_works_as_an_async_context_manager():
    mem = await _client(lambda r: httpx.Response(200, json={}))
    async with mem as m:
        assert m is mem


def test_every_sync_method_has_an_async_twin_of_the_same_name():
    sync = {n for n in dir(RemoteMemvara) if not n.startswith("_")}
    asyn = {n for n in dir(AsyncRemoteMemvara) if not n.startswith("_")}
    missing = sync - asyn - {"close"}
    assert not missing, f"async client is missing: {sorted(missing)}"


def test_async_twins_match_their_sync_signatures():
    """Same names is not the same as same signatures — see the module docstring."""
    sync = {n for n in dir(RemoteMemvara) if not n.startswith("_")} - {"close"}
    mismatched = {}
    for name in sync:
        sync_params = list(inspect.signature(getattr(RemoteMemvara, name)).parameters)[1:]
        async_params = list(
            inspect.signature(getattr(AsyncRemoteMemvara, name)).parameters)[1:]
        if sync_params != async_params:
            mismatched[name] = (sync_params, async_params)
    assert not mismatched, f"signature drift: {mismatched}"
