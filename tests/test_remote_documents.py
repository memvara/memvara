"""The document methods of the hosted client, against a fake transport.

Each test asserts the method, the path and the body a call sends, as well as the value it
decodes, for the reason `tests/test_remote_reads.py` gives: a call that reached the wrong
route can still decode a fixture that happens to fit. The shapes are the ones the design
names (`POST/GET /v1/documents`, `GET/PATCH/DELETE /v1/documents/{id}`,
`POST /v1/documents/delete`, `GET /v1/documents/{id}/status`); the deployment's own
models are the authority once they exist, and `memvara/remote/hydrate.py` follows them.
"""
import asyncio
import base64
import json
from datetime import datetime, timezone

import httpx
import pytest

from memvara import PatternRedactor
from memvara.remote import hydrate
from memvara.remote.aio import AsyncRemoteMemvara
from memvara.remote.api import RemoteMemvara
from memvara.types import DeleteResult, Scope

DOCUMENT = {
    "id": "doc_1", "custom_id": "handbook",
    "scope": {"tenant": "prj_1", "user": "alice", "agent": None, "session": None,
              "project": None},
    "title": "Handbook", "filepath": "policies/handbook.md", "source_uri": None,
    "mime": "text/markdown", "content_hash": "abc", "status": "done", "error": None,
    "metadata": {"team": "support"}, "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-02T00:00:00Z", "chunks": 4,
}
STATUS = {"id": "doc_1", "status": "failed", "error": "provider unavailable", "chunks": 4,
          "updated_at": "2026-01-02T00:00:00Z"}
DELETED = {"id": "doc_1", "deleted": True, "custom_id": "handbook", "chunks": 4,
           "episodes": 4, "retired": ["cl_1"], "unlinked": ["cl_2"]}
NOT_FOUND = {"error": {"code": "not_found", "message": "no such document"}}


def _body(request):
    return json.loads(request.read()) if request.content else None


def _sync(answer, **kw):
    calls = []

    def handler(request):
        calls.append(request)
        status, payload = answer(request)
        return httpx.Response(status, json=payload)

    mem = RemoteMemvara(api_key="k", base_url="https://example.test", user="alice", **kw)
    mem._http._client._transport = httpx.MockTransport(handler)
    return mem, calls


def _async(answer, **kw):
    calls = []

    def handler(request):
        calls.append(request)
        status, payload = answer(request)
        return httpx.Response(status, json=payload)

    mem = AsyncRemoteMemvara(api_key="k", base_url="https://example.test", user="alice",
                             **kw)
    mem._http._client = httpx.AsyncClient(base_url="https://example.test",
                                          transport=httpx.MockTransport(handler))
    return mem, calls


def _ok(payload):
    return lambda request: (200, payload)


def _missing(request):
    return 404, NOT_FOUND


class _Both:
    """Runs one scenario against the sync client and its async twin, so the two cannot
    drift apart in what they send."""

    def __init__(self, kind):
        self.kind = kind

    def client(self, answer, **kw):
        mem, calls = (_sync if self.kind == "sync" else _async)(answer, **kw)
        return _Caller(mem, self.kind), calls


class _Caller:
    def __init__(self, mem, kind):
        self._mem, self._kind = mem, kind

    def __getattr__(self, name):
        method = getattr(self._mem, name)
        if self._kind == "sync":
            return method
        return lambda *a, **kw: asyncio.run(method(*a, **kw))


@pytest.fixture(params=["sync", "async"])
def both(request):
    return _Both(request.param)


# --- each route -----------------------------------------------------------------


def test_add_posts_the_text_and_decodes_the_document(both):
    mem, calls = both.client(_ok(DOCUMENT))
    doc = mem.add_document("Refunds take 14 days.", custom_id="handbook",
                           title="Handbook", filepath="policies/handbook.md",
                           mime="text/markdown", meta={"team": "support"})
    request = calls[-1]
    assert (request.method, request.url.path) == ("POST", "/v1/documents")
    assert request.url.params["user"] == "alice"
    assert request.headers.get("idempotency-key")
    assert _body(request) == {"content": "Refunds take 14 days.", "custom_id": "handbook",
                              "title": "Handbook", "filepath": "policies/handbook.md",
                              "mime": "text/markdown", "metadata": {"team": "support"},
                              "extract": True}
    assert (doc.id, doc.status, doc.chunks, doc.meta) == ("doc_1", "done", 4,
                                                          {"team": "support"})
    assert doc.scope == Scope("prj_1", "alice")
    assert doc.created_at == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_add_sends_a_url_and_bytes_as_their_own_fields(both):
    mem, calls = both.client(_ok(DOCUMENT))
    mem.add_document(url="https://example.com/refunds", extract=False)
    assert _body(calls[-1]) == {"url": "https://example.com/refunds", "extract": False}
    mem.add_document(b"%PDF-1.7", mime="application/pdf")
    assert _body(calls[-1]) == {
        "content_base64": base64.b64encode(b"%PDF-1.7").decode("ascii"),
        "mime": "application/pdf", "extract": True}


def test_add_needs_exactly_one_of_content_and_url_before_anything_is_sent(both):
    mem, calls = both.client(_ok(DOCUMENT))
    with pytest.raises(TypeError, match="exactly one of content and url"):
        mem.add_document()
    with pytest.raises(TypeError, match="exactly one of content and url"):
        mem.add_document("x", url="https://example.com")
    assert calls == []


def test_the_text_and_title_are_redacted_before_they_leave_the_process(both):
    mem, calls = both.client(_ok(DOCUMENT), redactor=PatternRedactor())
    mem.add_document("Mail alice@example.com.", title="alice@example.com")
    sent = json.dumps(_body(calls[-1]))
    assert "alice@example.com" not in sent
    mem.update_document("doc_1", content="Mail bob@example.com.")
    assert "bob@example.com" not in json.dumps(_body(calls[-1]))


def test_bytes_are_refused_when_a_redactor_is_configured(both):
    """Nothing here can read a PDF to redact it, and sending it unredacted is the one
    thing a configured redactor exists to prevent."""
    mem, calls = both.client(_ok(DOCUMENT), redactor=PatternRedactor())
    with pytest.raises(ValueError, match="bytes cannot be redacted"):
        mem.add_document(b"%PDF")
    assert calls == []


def test_get_escapes_a_custom_id_and_reads_a_404_as_none(both):
    mem, calls = both.client(_ok(DOCUMENT))
    assert mem.get_document("docs/handbook?v=2").id == "doc_1"
    assert calls[-1].url.raw_path.startswith(b"/v1/documents/docs%2Fhandbook%3Fv%3D2")
    mem, _ = both.client(_missing)
    assert mem.get_document("doc_1") is None


def test_list_sends_its_filters_as_query_parameters_and_decodes_the_page(both):
    mem, calls = both.client(_ok({"documents": [DOCUMENT], "next_cursor": "c2"}))
    page = mem.list_documents(filepath_prefix="policies/", status="done", limit=10,
                              cursor="c1")
    params = calls[-1].url.params
    assert (calls[-1].method, calls[-1].url.path) == ("GET", "/v1/documents")
    assert (params["filepath_prefix"], params["status"], params["limit"],
            params["cursor"]) == ("policies/", "done", "10", "c1")
    assert [d.id for d in page.items] == ["doc_1"] and page.next_cursor == "c2"


def test_update_patches_what_is_given_and_a_404_is_a_key_error(both):
    mem, calls = both.client(_ok(DOCUMENT))
    mem.update_document("doc_1", title="New", meta={"k": "v"}, filepath="a.md")
    assert (calls[-1].method, calls[-1].url.path) == ("PATCH", "/v1/documents/doc_1")
    assert _body(calls[-1]) == {"title": "New", "metadata": {"k": "v"},
                                "filepath": "a.md", "extract": True}
    mem.update_document("doc_1", content=b"%PDF")
    assert set(_body(calls[-1])) == {"content_base64", "extract"}
    mem, _ = both.client(_missing)
    with pytest.raises(KeyError, match="no document 'doc_1'"):
        mem.update_document("doc_1", title="x")


def test_delete_decodes_what_was_retired_and_a_404_deletes_nothing(both):
    mem, calls = both.client(_ok(DELETED))
    result = mem.delete_document("handbook")
    assert (calls[-1].method, calls[-1].url.path) == ("DELETE", "/v1/documents/handbook")
    assert calls[-1].headers.get("idempotency-key")
    assert result == DeleteResult("doc_1", True, "handbook", 4, 4, ("cl_1",), ("cl_2",))
    mem, _ = both.client(_missing)
    assert mem.delete_document("doc_9") == DeleteResult("doc_9", False)


def test_bulk_delete_posts_the_ids_in_order(both):
    mem, calls = both.client(_ok({"results": [DELETED, {**DELETED, "id": "doc_2",
                                                        "deleted": False}]}))
    results = mem.delete_documents(["doc_1", "doc_2"])
    assert (calls[-1].method, calls[-1].url.path) == ("POST", "/v1/documents/delete")
    assert _body(calls[-1]) == {"ids": ["doc_1", "doc_2"]}
    assert [(r.id, r.deleted) for r in results] == [("doc_1", True), ("doc_2", False)]


def test_status_decodes_and_a_404_is_a_key_error(both):
    mem, calls = both.client(_ok(STATUS))
    status = mem.document_status("doc_1")
    assert calls[-1].url.path == "/v1/documents/doc_1/status"
    assert (status.status, status.error, status.chunks) == ("failed",
                                                            "provider unavailable", 4)
    mem, _ = both.client(_missing)
    with pytest.raises(KeyError):
        mem.document_status("doc_1")


def test_a_status_the_library_has_no_word_for_is_refused():
    with pytest.raises(ValueError, match="'archived' is not one of"):
        hydrate.document({**DOCUMENT, "status": "archived"})


def test_the_scoped_views_forward_every_document_method(both):
    mem, calls = both.client(_ok(DOCUMENT))
    view = mem._mem.scope(agent="a1")

    def run(result):
        return asyncio.run(result) if both.kind == "async" else result

    run(view.add_document("x"))
    run(view.get_document("doc_1"))
    run(view.update_document("doc_1", title="t"))
    assert all(c.url.params["agent"] == "a1" for c in calls)
    for method, payload in (("list_documents", {"documents": [], "next_cursor": None}),
                            ("delete_document", DELETED),
                            ("delete_documents", {"results": [DELETED]}),
                            ("document_status", STATUS)):
        mem, calls = both.client(_ok(payload))
        view = mem._mem.scope(agent="a1")
        args = () if method == "list_documents" else (
            (["doc_1"],) if method == "delete_documents" else ("doc_1",))
        run(getattr(view, method)(*args))
        assert calls[-1].url.params["agent"] == "a1", method
