"""The hosted clients' side of closure reasons, query-addressed closure and typed links.

The routes themselves are built by the cloud repository. What is pinned here is what this
client sends and how it reads the answer, against a fake transport: every test asserts the
method and the path as well as the body, for `tests/test_remote_writes.py`'s reason. The
async client is exercised through the same cases, because its twin methods are separate
code and a signature check cannot see what they put on the wire.

The wire shapes this client commits the deployment to are:

* `reason` on `POST /v1/forget` and `POST /v1/end`, and in a JSON body on
  `DELETE /v1/memories/{id}` only when one is given;
* `until_reason`, `replaces` and `reason` on `POST /v1/facts`;
* `POST /v1/forget-matching` taking `query`, `close`, `k`, `reason` and `confirm`, and
  answering either a preview (`close`, `matches` of `memory_id` and `text`, `confirm`,
  `expires_at`) or a result (`close`, `closed` memories, `reason`), with a refused token
  as a 409;
* `POST /v1/links` answering one link, and `GET /v1/memories/{id}/links` answering
  `claim_links`, with 404 for an id the credential cannot see;
* `claim_links` on the `why` body.
"""
import asyncio
import json

import httpx
import pytest

from memvara import ConfirmationRefused, ForgetPreview, ForgetResult, Link
from memvara.remote.aio import AsyncRemoteMemvara
from memvara.remote.api import RemoteMemvara
from memvara.server.config import ConfigError, ServerConfig, build_memvara
from memvara.types import closure_reasons

from test_remote_reads import _episode, _memory
from test_remote_writes import _receipt

LINK = {"from_id": "cl_2", "to_id": "cl_1", "relation": "extends",
        "created_at": "2026-01-01T00:00:00Z", "by": "api"}
PREVIEW = {"close": "retired", "matches": [{"memory_id": "cl_1", "text": "user likes tea"}],
           "confirm": "tok.mac", "expires_at": "2026-01-01T00:10:00Z"}
RESULT = {"close": "retired", "closed": [_memory()], "reason": "misheard"}


def _route(request: httpx.Request) -> httpx.Response:
    path, method = request.url.path, request.method
    body = json.loads(request.read() or b"{}")
    if path == "/v1/forget-matching":
        if body.get("confirm") == "stale":
            return httpx.Response(409, json={"error": {
                "code": "conflict", "message": "this confirmation token expired",
                "retryable": False}})
        return httpx.Response(200, json=RESULT if "confirm" in body else PREVIEW)
    if path == "/v1/links":
        if body["to_id"] == "cl_hidden":
            return httpx.Response(404, json={"error": {
                "code": "not_found", "message": "no such memory", "retryable": False}})
        return httpx.Response(200, json=LINK)
    if path.endswith("/links"):
        if "cl_hidden" in path:
            return httpx.Response(404, json={"error": {
                "code": "not_found", "message": "no such memory", "retryable": False}})
        return httpx.Response(200, json={"claim_links": [LINK]})
    if path == "/v1/facts" or path.endswith("/supersede"):
        return httpx.Response(200, json=_receipt())
    if path == "/v1/forget":
        return httpx.Response(200, json={"retired": [_memory()]})
    if path == "/v1/end":
        return httpx.Response(200, json={"ended": [_memory()]})
    if method == "DELETE":
        return httpx.Response(200, json={"id": "cl_1", "retired": True, "erased": False})
    if path.endswith("/why"):
        return httpx.Response(200, json={
            "memory": _memory(), "derivation": "user", "extractor": "api",
            "sources": [_episode()], "superseded": [], "claim_links": [LINK]})
    raise AssertionError(f"unexpected {method} {path}")


class Recorded:
    """Both clients over one fake transport, and every request either one sent."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            return _route(request)

        self.sync = RemoteMemvara(api_key="k", base_url="https://example.test",
                                  user="alice")
        self.sync._http._client._transport = httpx.MockTransport(handler)
        self.aio = AsyncRemoteMemvara(api_key="k", base_url="https://example.test",
                                      user="alice")
        self.aio._http._client = httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(handler))

    def last(self) -> tuple[str, str, dict]:
        request = self.calls[-1]
        raw = request.read()
        return request.method, request.url.path, json.loads(raw) if raw else {}


@pytest.fixture()
def wire():
    return Recorded()


def both(wire: Recorded, name: str, *args, **kw):
    """Run one method on each client, returning both results and both last requests."""
    first = getattr(wire.sync, name)(*args, **kw)
    sent_sync = wire.last()
    second = asyncio.run(getattr(wire.aio, name)(*args, **kw))
    sent_async = wire.last()
    assert sent_sync == sent_async, f"{name}: the two clients sent different requests"
    return first, second, sent_sync


# --- reasons ------------------------------------------------------------------


def test_a_reason_on_delete_travels_in_a_body_and_never_in_the_query_string(wire):
    ok, again, (method, path, body) = both(wire, "delete", "cl_1", reason=" misheard ")
    assert ok is again is True
    assert (method, path, body) == ("DELETE", "/v1/memories/cl_1", {"reason": "misheard"})
    assert "reason" not in dict(wire.calls[-1].url.params)


def test_delete_without_a_reason_sends_no_body_at_all(wire):
    _, _, (method, _, body) = both(wire, "delete", "cl_1")
    assert method == "DELETE" and body == {}


def test_ending_by_id_or_slot_carries_the_reason_to_the_end_route(wire):
    _, _, sent = both(wire, "delete", "cl_1", close="ended", reason="left")
    assert sent == ("POST", "/v1/end", {"memory_id": "cl_1", "reason": "left"})
    _, _, sent = both(wire, "forget", "user", "works_at", close="ended", reason="left")
    assert sent == ("POST", "/v1/end",
                    {"subject": "user", "predicate": "works_at", "reason": "left"})
    _, _, sent = both(wire, "end", predicate="works_at", reason="left")
    assert sent[2]["reason"] == "left"


def test_retiring_a_slot_carries_the_reason_to_the_forget_route(wire):
    closed, _, sent = both(wire, "forget", "user", "likes", reason="misheard")
    assert sent == ("POST", "/v1/forget",
                    {"subject": "user", "predicate": "likes", "reason": "misheard"})
    assert [c.id for c in closed] == ["cl_1"]


def test_supersede_sends_the_reason_as_a_field_and_not_as_metadata(wire):
    """`supersede` takes `**meta`, so a `reason` it did not name would have been stored
    as metadata on the new memory instead of on the closed one."""
    _, _, (method, path, body) = both(wire, "supersede", "cl_1", "user", "likes",
                                      "coffee", reason="switched")
    assert (method, path) == ("POST", "/v1/memories/cl_1/supersede")
    assert body["reason"] == "switched" and "reason" not in body["metadata"]


def test_remember_sends_the_three_new_fields_only_when_given(wire):
    _, _, (_, path, body) = both(wire, "remember", "user", "likes", "coffee",
                                 replaces="cl_1", reason="switched",
                                 until_reason="for now")
    assert path == "/v1/facts"
    assert (body["replaces"], body["reason"], body["until_reason"]) == (
        "cl_1", "switched", "for now")
    _, _, (_, _, plain) = both(wire, "remember", "user", "likes", "coffee")
    assert not {"replaces", "reason", "until_reason"} & set(plain)


def test_a_bad_reason_is_refused_before_any_request(wire):
    with pytest.raises(ValueError, match="blank"):
        wire.sync.delete("cl_1", reason="  ")
    with pytest.raises(ValueError, match="limit is 500"):
        asyncio.run(wire.aio.forget("user", "likes", reason="x" * 501))
    assert wire.calls == []


def test_the_reason_on_a_hosted_claim_reads_back_from_its_metadata():
    """The wire carries `meta` as `metadata`, so the closure witness, reason included,
    comes back with the memory and `history`/`why` render it on either engine."""
    from memvara.remote import hydrate

    body = _memory()
    body["metadata"] = {"closure": [{"at": 1.0, "close": "ended", "by": None,
                                     "reason": "left"}]}
    assert closure_reasons(hydrate.claim(body)) == [("ended", "left")]


# --- forget_matching ------------------------------------------------------------


def test_a_preview_is_hydrated_and_sends_no_confirm(wire):
    preview, again, (method, path, body) = both(wire, "forget_matching", "tea",
                                                close="retired", k=5)
    assert (method, path) == ("POST", "/v1/forget-matching")
    assert body == {"query": "tea", "close": "retired", "k": 5}
    assert isinstance(preview, ForgetPreview) and preview == again
    assert preview.matches == {"cl_1": "user likes tea"}
    assert preview.confirm == "tok.mac"


def test_a_confirmed_call_returns_what_was_closed(wire):
    done, _, (_, _, body) = both(wire, "forget_matching", "tea", close="retired",
                                 reason="misheard", confirm="tok.mac")
    assert body["confirm"] == "tok.mac" and body["reason"] == "misheard"
    assert isinstance(done, ForgetResult)
    assert [c.id for c in done.closed] == ["cl_1"] and done.reason == "misheard"


def test_a_refused_token_is_the_same_exception_the_local_engine_raises(wire):
    with pytest.raises(ConfirmationRefused, match="expired"):
        wire.sync.forget_matching("tea", close="retired", confirm="stale")
    with pytest.raises(ConfirmationRefused, match="expired"):
        asyncio.run(wire.aio.forget_matching("tea", close="retired", confirm="stale"))


def test_erasure_is_refused_before_any_request(wire):
    with pytest.raises(ValueError, match="is not a closure"):
        wire.sync.forget_matching("tea", close="erased")
    assert wire.calls == []


# --- links --------------------------------------------------------------------


def test_link_posts_the_relation_and_returns_the_stored_link(wire):
    link, again, sent = both(wire, "link", "cl_2", "cl_1", "extends")
    assert sent == ("POST", "/v1/links", {"from_id": "cl_2", "to_id": "cl_1",
                                          "relation": "extends", "by": "api"})
    assert isinstance(link, Link) and link == again
    assert (link.from_id, link.to_id, link.relation) == ("cl_2", "cl_1", "extends")


def test_link_raises_key_error_for_an_id_the_credential_cannot_see(wire):
    with pytest.raises(KeyError):
        wire.sync.link("cl_2", "cl_hidden", "extends")
    with pytest.raises(KeyError):
        asyncio.run(wire.aio.link("cl_2", "cl_hidden", "extends"))


def test_link_refuses_an_unknown_relation_before_any_request(wire):
    with pytest.raises(ValueError, match="not a link relation"):
        wire.sync.link("cl_2", "cl_1", "supersedes")
    assert wire.calls == []


def test_links_reads_both_directions_and_is_empty_for_an_invisible_id(wire):
    found, again, (method, path, _) = both(wire, "links", "cl_1")
    assert (method, path) == ("GET", "/v1/memories/cl_1/links")
    assert found == again == [Link("cl_2", "cl_1", "extends", found[0].created_at)]
    assert wire.sync.links("cl_hidden") == []
    assert asyncio.run(wire.aio.links("cl_hidden")) == []


def test_the_async_scoped_view_forwards_the_three_new_methods(wire):
    async def main():
        view = wire.aio.scope(agent="a1")
        preview = await view.forget_matching("tea", close="retired", k=3)
        link = await view.link("cl_2", "cl_1", "extends")
        return preview, link, await view.links("cl_1")

    preview, link, links = asyncio.run(main())
    assert preview.confirm == "tok.mac" and links == [link]
    assert dict(wire.calls[-1].url.params)["agent"] == "a1"


def test_the_new_routes_carry_the_bound_project_as_a_header():
    """Every other route sends `Memvara-Project` when a project is bound, and these three
    go through the same `_request`, so the deployment narrows them the same way."""
    wire = Recorded()
    wire.sync.default_scope = wire.sync.default_scope.__class__(
        "default", "alice", project="github.com/acme/app")
    wire.sync.forget_matching("tea", close="retired")
    wire.sync.link("cl_2", "cl_1", "extends")
    wire.sync.links("cl_1")
    assert [r.headers.get("memvara-project") for r in wire.calls] == [
        "github.com/acme/app"] * 3


def test_why_reads_the_links_on_the_provenance_body(wire):
    prov = wire.sync.why("cl_1")
    assert [(k.from_id, k.relation) for k in prov.links] == [("cl_2", "extends")]


def test_why_from_a_server_that_predates_links_reports_none():
    from memvara.remote import hydrate

    prov = hydrate.provenance({"memory": _memory(), "derivation": "user",
                               "extractor": None, "sources": [], "superseded": []})
    assert prov.links == []


# --- the confirmation key in server configuration -----------------------------


def test_the_confirmation_key_is_read_from_the_environment_and_kept_out_of_repr():
    config = ServerConfig.from_env({"MEMVARA_DB": ":memory:",
                                    "MEMVARA_CONFIRM_SECRET": " s3cret "})
    assert config.confirm_secret == "s3cret"
    assert "s3cret" not in repr(config)
    assert ServerConfig.from_env({"MEMVARA_DB": ":memory:",
                                  "MEMVARA_CONFIRM_SECRET": "  "}).confirm_secret is None


def test_a_local_server_signs_tokens_with_the_configured_key():
    """Two servers built from the same environment accept each other's tokens, which is
    the property a multi-process deployment needs."""
    env = {"MEMVARA_DB": ":memory:", "MEMVARA_CONFIRM_SECRET": "shared",
           "MEMVARA_USER": "alice"}
    first = build_memvara(ServerConfig.from_env(env))
    second = build_memvara(ServerConfig.from_env(env))
    try:
        from memvara.types import utcnow

        token, _ = first._confirmer.issue(["cl_1"], "ended", now=utcnow())
        assert second._confirmer.check(token, "ended", now=utcnow()) == ["cl_1"]
    finally:
        first.close()
        second.close()


def test_cloud_mode_refuses_the_confirmation_key_without_echoing_it():
    cloud = ServerConfig.from_env({"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": "k",
                                   "MEMVARA_CONFIRM_SECRET": "s3cret"})
    with pytest.raises(ConfigError, match="MEMVARA_CONFIRM_SECRET") as caught:
        build_memvara(cloud)
    assert "s3cret" not in str(caught.value)
