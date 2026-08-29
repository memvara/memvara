"""Writes, and the one routing decision that must not be got wrong.

`close="ended"` and `close="retired"` are different statements about whether a stored fact
was ever true. They go to different endpoints, and the tests that they do are the point of
this file: filing one as the other records a false reason for the change, and nothing
downstream can detect it, because both leave a closed memory and the row that says which
happened is the one being written.

Every test asserts the method and the path as well as the value, for the reason
`tests/test_remote_reads.py` gives: a write that reached the wrong endpoint can still
decode a fixture that happens to fit.
"""
import json
from datetime import datetime, timezone

import httpx
import pytest

from memvara.redact import CLAIM_OBJECT, CLAIM_SUBJECT, CLAIM_TEXT, EPISODE
from memvara.remote.api import RemoteMemvara
from memvara.types import Claim, Episode, MemoryType, WriteReceipt

from test_remote_reads import _memory


@pytest.fixture
def recorded():
    calls = []

    def build(payload=None, **kw):
        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=payload if payload is not None else {})

        mem = RemoteMemvara(api_key="k", base_url="https://example.test", **kw)
        mem._http._client._transport = httpx.MockTransport(handler)
        return mem

    build.calls = calls
    return build


def _receipt(**over):
    body = {"episode_ids": ["ep_1"], "added": [_memory()], "invalidated": [],
            "reinforced": [], "skipped": 0, "unextracted": 0, "llm_calls": 0,
            "latency_ms": 1.5, "deferred": False, "note": None}
    body.update(over)
    return body


def _sent(request):
    """The JSON body this request carried, decoded.

    Decoded rather than matched as a substring: a field is present or absent, and
    `"predicate"` appearing somewhere in a serialized blob is not the same claim.
    """
    return json.loads(request.read())


# --- the routing decision -----------------------------------------------------


def test_retiring_goes_to_the_delete_route(recorded):
    mem = recorded({"id": "cl_1", "retired": True, "erased": False})
    assert mem.delete("cl_1", close="retired") is True
    assert recorded.calls[-1].method == "DELETE"
    assert recorded.calls[-1].url.path == "/v1/memories/cl_1"


def test_ending_goes_to_the_end_route_and_never_to_delete(recorded):
    mem = recorded({"memory_id": "cl_1", "subject": None, "predicate": None,
                    "count": 1, "ended": [_memory()], "erased": False})
    assert mem.delete("cl_1", close="ended") is True
    assert recorded.calls[-1].url.path == "/v1/end"
    assert recorded.calls[-1].method == "POST"


def test_an_unknown_closure_raises_with_both_readings_named(recorded):
    """A typo would otherwise be indistinguishable from the default, and the default is
    one of the two answers the caller was choosing between."""
    mem = recorded()
    with pytest.raises(ValueError) as caught:
        mem.delete("cl_1", close="deleted")
    assert "ended" in str(caught.value) and "retired" in str(caught.value)


def test_an_unknown_closure_reaches_no_endpoint_at_all(recorded):
    """Validated before the request, not after it. A closure checked server-side would
    have already written something by the time the error came back."""
    mem = recorded()
    with pytest.raises(ValueError):
        mem.delete("cl_1", close="ended-ish")
    assert recorded.calls == []


def test_forgetting_a_slot_retires_it_through_the_forget_route(recorded):
    mem = recorded({"subject": "user", "predicate": "works_at", "count": 1,
                    "retired": [_memory(predicate="works_at", obj="Acme")],
                    "erased": False})
    claims = mem.forget("user", "works_at")
    assert recorded.calls[-1].url.path == "/v1/forget"
    assert isinstance(claims[0], Claim)


def test_ending_a_slot_goes_to_the_end_route_and_never_to_forget(recorded):
    """`/v1/forget` has no `close` field, so a client that posted there with
    `close="ended"` would file every ending as a retirement. `server/tools.py` calls
    exactly this: `forget(..., close="ended")` is how `memory_end` closes a slot."""
    mem = recorded({"memory_id": None, "subject": "user", "predicate": "works_at",
                    "count": 1, "ended": [_memory(predicate="works_at", obj="Acme")],
                    "erased": False})
    claims = mem.forget("user", "works_at", close="ended")
    assert recorded.calls[-1].url.path == "/v1/end"
    assert isinstance(claims[0], Claim)


def test_forget_refuses_an_unknown_closure_before_reaching_anything(recorded):
    mem = recorded()
    with pytest.raises(ValueError):
        mem.forget("user", "works_at", close="removed")
    assert recorded.calls == []


# --- addressing, on the route where two modes exist ---------------------------


def test_end_by_slot_sends_subject_and_predicate_and_no_id(recorded):
    mem = recorded({"memory_id": None, "subject": "user", "predicate": "works_at",
                    "count": 0, "ended": [], "erased": False})
    mem.end(subject="user", predicate="works_at")
    body = _sent(recorded.calls[-1])
    assert body["predicate"] == "works_at" and body["subject"] == "user"
    assert "memory_id" not in body


def test_end_by_id_sends_the_id_and_no_slot(recorded):
    mem = recorded({"memory_id": "cl_1", "subject": None, "predicate": None,
                    "count": 1, "ended": [_memory()], "erased": False})
    mem.end(claim_id="cl_1")
    body = _sent(recorded.calls[-1])
    assert body["memory_id"] == "cl_1"
    assert "predicate" not in body and "subject" not in body


def test_end_defaults_the_subject_but_never_the_addressing_mode(recorded):
    mem = recorded({"memory_id": None, "subject": "user", "predicate": "works_at",
                    "count": 0, "ended": [], "erased": False})
    mem.end(predicate="works_at")
    assert _sent(recorded.calls[-1])["subject"] == "user"


def test_end_needs_exactly_one_addressing_mode(recorded):
    """The two have different blast radii — one memory against every current value of a
    slot — so a silent default on that choice is not a convenience."""
    mem = recorded()
    with pytest.raises(TypeError):
        mem.end()
    with pytest.raises(TypeError):
        mem.end(claim_id="cl_1", predicate="works_at")
    assert recorded.calls == []


# --- one per remaining write --------------------------------------------------


def test_add_reaches_the_ingest_endpoint_and_returns_a_receipt(recorded):
    mem = recorded(_receipt())
    receipt = mem.add("I moved to Berlin")
    assert recorded.calls[-1].method == "POST"
    assert recorded.calls[-1].url.path == "/v1/memories"
    assert isinstance(receipt, WriteReceipt) and receipt.episode_ids == ["ep_1"]


def test_add_folds_an_unknown_message_key_into_metadata(recorded):
    """The facade's request models are `extra="forbid"`, so a stray key beside `role` and
    `content` is a 422 rather than a field somebody quietly loses."""
    mem = recorded(_receipt())
    mem.add([{"role": "user", "content": "hi", "channel": "slack"}])
    assert _sent(recorded.calls[-1])["messages"] == [
        {"role": "user", "content": "hi", "metadata": {"channel": "slack"}}]


def test_remember_reaches_the_facts_endpoint_and_returns_a_receipt(recorded):
    mem = recorded(_receipt())
    receipt = mem.remember("user", "likes", "tea")
    assert recorded.calls[-1].url.path == "/v1/facts"
    assert isinstance(receipt, WriteReceipt)


def test_remember_splits_cited_ids_from_turns_it_must_store(recorded):
    """`Memvara.remember` takes one mixed sequence; the facade takes two fields, because
    citing a stored turn and writing a new one are different acts over HTTP."""
    mem = recorded(_receipt())
    mem.remember("user", "likes", "tea", sources=["ep_9", {"content": "I like tea"}])
    body = _sent(recorded.calls[-1])
    assert body["source_ids"] == ["ep_9"]
    assert body["sources"] == [{"content": "I like tea", "metadata": {}}]


def test_supersede_forwards_close_verbatim_rather_than_defaulting(recorded):
    """A mutation log records that a value changed and never which of the two it was, so
    restating the default here would file every correction as a world event."""
    mem = recorded(_receipt())
    mem.supersede("cl_old", "user", "lives_in", "Berlin", close="retired")
    assert recorded.calls[-1].url.path == "/v1/memories/cl_old/supersede"
    assert _sent(recorded.calls[-1])["close"] == "retired"


def test_supersede_refuses_an_unknown_closure(recorded):
    mem = recorded(_receipt())
    with pytest.raises(ValueError):
        mem.supersede("cl_old", "user", "lives_in", "Berlin", close="replaced")
    assert recorded.calls == []


def test_erase_reaches_the_erasure_endpoint_and_reports_whether_it_landed(recorded):
    mem = recorded({"target": "memory", "memory_id": "cl_1", "scope": None,
                    "erased": True, "counts": None, "sources_erased": False,
                    "audit_subject_linkable": None})
    assert mem.erase("cl_1") is True
    assert recorded.calls[-1].method == "POST"
    assert recorded.calls[-1].url.path == "/v1/erasures"


def test_purge_sends_a_scope_and_returns_the_per_table_counts(recorded):
    """The counts are the evidence a deletion request has to be answered with, measured
    by the store rather than assembled by the facade."""
    mem = recorded({"target": "scope",
                    "memory_id": None,
                    "scope": {"tenant": "t", "user": "alice", "agent": None,
                              "session": None},
                    "erased": True, "counts": {"claims": 4, "episodes": 9},
                    "sources_erased": None, "audit_subject_linkable": False},
                   user="alice")
    counts = mem.purge()
    assert recorded.calls[-1].url.path == "/v1/erasures"
    assert _sent(recorded.calls[-1])["scope"] == {"user": "alice"}
    assert counts == {"claims": 4, "episodes": 9}


def test_consolidate_reaches_the_maintenance_endpoint_and_returns_a_job(recorded):
    """A job rather than counts: the endpoint answers 202 before the pass starts, so the
    outcome is on the job and in no status code."""
    mem = recorded({"id": "job_1", "kind": "consolidate", "tenant": "t",
                    "status": "queued", "created_at": "2026-01-01T00:00:00Z",
                    "started_at": None, "finished_at": None, "result": None,
                    "error": None, "links": {"self": "/v1/jobs/job_1"}})
    job = mem.consolidate()
    assert recorded.calls[-1].url.path == "/v1/maintenance/consolidate"
    assert job["status"] == "queued"


# --- the header that makes a retried write safe -------------------------------


@pytest.mark.parametrize("call", [
    lambda m: m.add("hello"),
    lambda m: m.remember("user", "likes", "tea"),
    lambda m: m.supersede("cl_1", "user", "likes", "tea"),
    lambda m: m.forget("user", "likes"),
    lambda m: m.forget("user", "likes", close="ended"),
    lambda m: m.delete("cl_1"),
    lambda m: m.delete("cl_1", close="ended"),
    lambda m: m.end(claim_id="cl_1"),
    lambda m: m.erase("cl_1"),
    lambda m: m.purge(),
    lambda m: m.consolidate(),
])
def test_every_write_carries_an_idempotency_key(recorded, call):
    """A write that fails after the request was sent may have committed, so repeating it
    can write the fact twice. The key is what makes the transport's retry safe, and a
    write that omitted it would be retried anyway."""
    mem = recorded(_receipt(**{"retired": [], "ended": [], "count": 0, "erased": False,
                               "counts": {}, "id": "cl_1", "status": "queued"}))
    call(mem)
    assert recorded.calls[-1].headers.get("idempotency-key")


# --- what a write sends, beyond where it sends it ------------------------------


@pytest.mark.parametrize("messages, expected", [
    (Episode(role="user", content="hi",
             ts=datetime(2026, 1, 1, tzinfo=timezone.utc)),
     [{"role": "user", "content": "hi", "ts": "2026-01-01T00:00:00+00:00",
       "metadata": {}}]),
    ({"role": "user", "content": "hi"}, [{"role": "user", "content": "hi",
                                          "metadata": {}}]),
    (["hi"], [{"content": "hi"}]),
    ("hi", "hi"),
], ids=["episode", "one-mapping", "string-in-a-list", "bare-string"])
def test_add_sends_each_message_shape_the_way_the_facade_spells_it(recorded, messages,
                                                                   expected):
    """A single turn and a sequence of them are different arguments and one wire format.

    The last two rows are one character apart and take different branches: a bare string
    is the whole conversation and goes out as `messages: "hi"`, while a string inside a
    sequence is one turn among others and goes out as a message object. A single `Episode`
    or mapping is wrapped rather than iterated — iterating a mapping yields its keys, so
    getting that wrong sends the field names as the conversation.
    """
    mem = recorded(_receipt())
    mem.add(messages)
    assert _sent(recorded.calls[-1])["messages"] == expected


def test_a_memory_type_goes_out_as_the_string_the_facade_spells(recorded):
    """`MemoryType` is a Python enum and the wire carries its `value`. A request model
    that lists the strings rejects the member itself, and `str(member)` is
    `MemoryType.PROCEDURAL` — neither is what the facade reads."""
    mem = recorded(_receipt())
    mem.remember("user", "prefers", "pytest", memory_type=MemoryType.PROCEDURAL)
    assert _sent(recorded.calls[-1])["memory_type"] == "procedural"


def test_an_unset_memory_type_is_omitted_so_the_predicate_decides(recorded):
    """Omitted rather than sent as null, and the difference is not cosmetic: the facade
    reads an absent `memory_type` as "use the type registered for this predicate", which
    is the whole point of registering one."""
    mem = recorded(_receipt())
    mem.remember("user", "prefers", "pytest")
    assert "memory_type" not in _sent(recorded.calls[-1])


def test_text_is_redacted_on_its_way_out_and_not_after_it_has_left(recorded):
    """Every field the policy is offered, checked on the wire.

    Redacting server-side would be the alternative and it is not redaction: the raw text
    has already crossed the network by then. The policy is handed the field name so a
    deployment can be aggressive on raw turns and conservative on claim objects, which is
    why this asserts per field rather than "something was replaced".
    """
    class Loud:
        def redact(self, text, *, field, scope):
            return f"[{field}:{scope.user}]"

    mem = recorded(_receipt(), user="alice", redactor=Loud())
    mem.add(Episode(role="user", content="I live at 12 Acacia Avenue"))
    mem.remember("user", "lives_in", "12 Acacia Avenue",
                 text="user lives at 12 Acacia Avenue")

    turn = _sent(recorded.calls[0])["messages"][0]
    assert turn["content"] == f"[{EPISODE}:alice]"
    fact = _sent(recorded.calls[-1])
    assert fact["subject"] == f"[{CLAIM_SUBJECT}:alice]"
    assert fact["object"] == f"[{CLAIM_OBJECT}:alice]"
    assert fact["text"] == f"[{CLAIM_TEXT}:alice]"
    assert fact["predicate"] == "lives_in", "a predicate is a schema term, not free text"
