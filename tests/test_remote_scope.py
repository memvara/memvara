"""A scoped view narrows and cannot widen.

`ScopedMemvara` exists so that an MCP handler has no way to address another tenant. The
remote twin has to hold that property the same way or cloud mode is a weaker server than
the local one.

The property is structural rather than behavioural, which is why these tests read
signatures: a handler cannot address another user if there is no parameter through which
to name one. A test that merely checked the right scope came out of the right call would
pass for a class that also accepted `user=`.
"""
import inspect
import json
from datetime import datetime, timezone

import httpx
import pytest

from memvara.remote.api import RemoteMemvara, ScopedRemoteMemvara
from memvara.types import Scope

from test_remote_writes import _receipt

SCOPE_ARGUMENTS = {"tenant", "user", "agent", "session"}


def _client(**kw):
    return RemoteMemvara(api_key="k", base_url="https://example.test", **kw)


def test_scope_returns_a_scoped_view():
    assert isinstance(_client().scope(user="alice"), ScopedRemoteMemvara)


def test_the_scoped_view_carries_the_narrowed_scope():
    mem = _client(user="alice")
    assert mem.scope(agent="a1")._scope.agent == "a1"
    assert mem.scope(agent="a1")._scope.user == "alice"


def test_the_narrowing_leaves_the_client_it_came_from_alone():
    """A view is a second handle, not a mutation. A `scope()` that rebound the client
    would silently move every later call made through the original."""
    mem = _client(user="alice")
    mem.scope(agent="a1")
    assert mem.default_scope.agent is None


@pytest.mark.parametrize("name", [
    "add", "remember", "recall", "search", "forget", "delete", "end", "get", "get_all",
    "history", "why", "ask", "since", "count", "stats",
])
def test_no_scoped_method_accepts_a_scope_argument(name):
    params = inspect.signature(getattr(ScopedRemoteMemvara, name)).parameters
    assert not SCOPE_ARGUMENTS & set(params)


def _public_methods():
    """Every public callable on the class, so these checks are derived from what is
    there rather than from a list somebody maintains."""
    for name in dir(ScopedRemoteMemvara):
        if name.startswith("_"):
            continue
        member = inspect.getattr_static(ScopedRemoteMemvara, name)
        if callable(member):
            yield name, inspect.signature(member)


def test_no_public_member_at_all_declares_a_scope_parameter():
    """The parametrized list above is a list somebody maintains. This is the same check
    derived from the class, so a method added later cannot slip past by not being on it.

    It reads parameter *names*, so it says nothing about a method that declares
    `**kwargs`: a scope name passed through one is invisible here. The two methods that
    do are pinned and then checked by calling, below.
    """
    offenders = {name: sorted(SCOPE_ARGUMENTS & set(sig.parameters))
                 for name, sig in _public_methods()
                 if SCOPE_ARGUMENTS & set(sig.parameters)}
    assert offenders == {}


def test_only_the_two_metadata_writers_take_kwargs_at_all():
    """`remember` and `supersede` forward `**kw`, as `ScopedMemvara` does, because a
    fact's metadata is caller-defined and cannot be enumerated. Every other method spells
    its parameters out.

    Pinned rather than merely allowed. A third forwarder appearing here fails this test
    and has to be justified, which is the point: the name-based guard above is blind to
    exactly this construct, so the set of methods it is blind to must not grow quietly.
    """
    with_kwargs = {name for name, sig in _public_methods()
                   if any(p.kind is inspect.Parameter.VAR_KEYWORD
                          for p in sig.parameters.values())}
    assert with_kwargs == {"remember", "supersede"}


@pytest.mark.parametrize("call", [
    lambda m: m.remember("user", "likes", "tea",
                         user="bob", agent="evil", session="s9"),
    lambda m: m.supersede("cl_1", "user", "likes", "tea",
                          user="bob", agent="evil", session="s9"),
])
def test_a_scope_name_passed_through_kwargs_never_reaches_the_query_string(call):
    """What the name-based guard cannot see, checked by calling instead.

    The invariant is structural and it is worth stating plainly: scope reaches the wire
    only through `_params()`, which reads the bound `default_scope` and nothing else.
    There is no path from a keyword argument to a scope query parameter, so a caller who
    passes `user=` gets a metadata key called `user` — recorded as data on the fact,
    which is honest — and the request still runs at the scope this view was bound to.
    """
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=_receipt())

    mem = _client(user="alice")
    mem._http._client._transport = httpx.MockTransport(handler)
    call(mem.scope(agent="a1"))

    params = dict(calls[-1].url.params)
    assert params["user"] == "alice" and params["agent"] == "a1"
    assert "session" not in params
    body = json.loads(calls[-1].read())
    assert body["metadata"] == {"user": "bob", "agent": "evil", "session": "s9"}


def test_the_bound_scope_is_what_actually_reaches_the_wire():
    """The signature check says a handler cannot name another user. This says the scope
    it *was* bound to is the one sent, so the narrowing is not decorative."""
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"count": 0, "total": 0, "limit": 1,
                                         "offset": 0, "as_of": None, "valid_at": None,
                                         "known_at": None, "states": ["live"],
                                         "memories": []})

    mem = _client(user="alice")
    mem._http._client._transport = httpx.MockTransport(handler)
    mem.scope(agent="a1").count()
    params = dict(calls[-1].url.params)
    assert params["user"] == "alice" and params["agent"] == "a1"
    assert "tenant" not in params


def test_the_scoped_view_exposes_the_client_underneath():
    """Parity with `ScopedMemvara`, which exposes the client it wraps the same way.

    Not something `server/tools.py` reaches for: no handler calls `ctx.memory.memvara`,
    and `MemoryAPI` deliberately does not declare the member, so reaching for it there is
    a type error rather than a scope escape somebody has to notice by reading — see
    `memvara/server/memory_api.py`. It is here because the two scoped views are meant to
    be interchangeable, and against a hosted deployment it gives nothing away: the
    credential binds the tenant, so what comes back cannot address another one either.
    """
    view = _client(user="alice").scope(agent="a1")
    assert isinstance(view.memvara, RemoteMemvara)
    assert view.memvara.default_scope == Scope("default", "alice", "a1", None)


def test_the_scope_is_readable_as_an_attribute():
    """`server/tools.py` reads `ctx.memory.scope` to report where it is, and `MemoryAPI`
    declares it as a member both implementations have."""
    view = _client(user="alice").scope()
    assert isinstance(view.scope, Scope)
    assert view.scope.user == "alice"


def test_the_view_shares_one_connection_pool_with_the_client_it_came_from():
    """Two handles on one deployment and one credential. A second pool would be two
    idempotency stores as well, which is the thing the transport's retry depends on."""
    mem = _client()
    assert mem.scope(user="alice").memvara._http is mem._http


# --- every method on the view, driven ----------------------------------------
#
# The checks above are structural: they read signatures and prove a handler has no
# parameter with which to name another scope. That says nothing about what the methods do.
# Every one of them is a single line forwarding to the client underneath, and a single
# forwarding line is exactly where a wrong keyword or a dropped argument hides — it type
# checks, it reads correctly, and it reaches a different endpoint or runs at a different
# scope. So each is called, and both halves are asserted: the path says the delegation
# landed where the unscoped method lands, and the query string says it ran at the scope
# the view was bound to rather than the wider one it came from.

from test_remote_reads import _episode, _memory, _ranking, _scope  # noqa: E402

#: The one route that carries no scope, because it carries no credential either.
UNSCOPED = {"health"}

_PATHS_BODY = {"as_of": None, "valid_at": None, "known_at": None, "count": 1,
               "paths": [{"nodes": ["alice", "berlin"], "labels": ["lives_in"],
                          "hops": 1, "score": 0.5,
                          "edges": [{"memory": _memory(), "backward": False,
                                     "strength": 0.5}]}]}
_LISTING_BODY = {"count": 1, "total": 1, "limit": 100, "offset": 0, "as_of": None,
                 "valid_at": None, "known_at": None, "states": ["live"],
                 "memories": [_memory()]}


def _answer(request):
    """One envelope per route, picked by path.

    A single handler rather than a payload beside every row, because these tests are about
    where a call goes and at what scope — the decoding is `tests/test_remote_reads.py`'s
    subject, and repeating twenty fixtures here to re-check it would make this file's
    failures ambiguous between the two.
    """
    path, method = request.url.path, request.method
    if path == "/v1/health":
        return httpx.Response(200, json={"status": "ok", "memvara_version": "0.2.0"})
    if path == "/v1/whoami":
        return httpx.Response(200, json={"token_id": "tok_1", "scope": _scope(),
                                         "granted_privilege": "write",
                                         "effective_privilege": "write",
                                         "expires_at": None, "read_only": False})
    if path == "/v1/stats":
        return httpx.Response(200, json={
            "scope": _scope(), "visible": 1,
            "tenant_counts": {"claims": 3, "live_claims": 3, "joinable_claims": 1},
            "extractor": "fast-path-only", "read_only": False})
    if path == "/v1/search":
        return httpx.Response(200, json={
            "as_of": None, "valid_at": None, "known_at": None, "states": ["live"],
            "count": 1, "results": [{"kind": "claim", "score": 0.7,
                                     "ranking": _ranking(), "memory": _memory()}]})
    if path == "/v1/recall":
        return httpx.Response(200, json={"text": "user lives in Berlin", "empty": False})
    if path == "/v1/history":
        return httpx.Response(200, json={"subject": "user", "predicate": "lives_in",
                                         "scope": _scope(), "as_of": None,
                                         "valid_at": None, "known_at": None, "count": 1,
                                         "timeline": [_memory()]})
    if path.endswith("/why"):
        return httpx.Response(200, json={"memory": _memory("cl_1"), "derivation": "user",
                                         "extractor": "api", "sources": [_episode()],
                                         "superseded": []})
    if path == "/v1/ask":
        return httpx.Response(200, json={
            "question": "q", "at": "2026-01-01T00:00:00+00:00", "count": 1,
            "text": "Then and now agree.",
            "readings": [{"subject": "user", "predicate": "lives_in",
                          "now": [_memory()], "then": [_memory()],
                          "stated": [_memory()], "diverged": False, "moved": False}]})
    if path == "/v1/since":
        return httpx.Response(200, json={"since": "2026-01-01T00:00:00+00:00",
                                         "added": [_memory()], "gone": []})
    if path.endswith("/produced") or path == "/v1/standing":
        return httpx.Response(200, json={"episode_id": "ep_1", "as_of": None,
                                         "valid_at": None, "known_at": None, "count": 1,
                                         "limit": 5, "truncated": False,
                                         "memories": [_memory()]})
    if path in ("/v1/neighborhood", "/v1/paths"):
        return httpx.Response(200, json=_PATHS_BODY)
    if path == "/v1/forget":
        return httpx.Response(200, json={"subject": "user", "predicate": "likes",
                                         "count": 0, "retired": [], "erased": False})
    if path == "/v1/end":
        return httpx.Response(200, json={"memory_id": "cl_1", "subject": None,
                                         "predicate": None, "count": 1,
                                         "ended": [_memory()], "erased": False})
    if path == "/v1/erasures":
        return httpx.Response(200, json={"target": "memory", "memory_id": "cl_1",
                                         "scope": None, "erased": True, "counts": {},
                                         "sources_erased": False,
                                         "audit_subject_linkable": None})
    if path == "/v1/maintenance/consolidate":
        return httpx.Response(200, json={"id": "job_1", "status": "queued"})
    if path == "/v1/facts" or path.endswith("/supersede"):
        return httpx.Response(200, json=_receipt())
    if path == "/v1/memories":
        if method == "POST":
            return httpx.Response(200, json=_receipt())
        return httpx.Response(200, json=_LISTING_BODY)
    if method == "DELETE":
        return httpx.Response(200, json={"id": "cl_1", "retired": True, "erased": False})
    return httpx.Response(200, json=_memory("cl_1"))


DELEGATIONS = [
    ("health", lambda v: v.health(), "/v1/health"),
    ("whoami", lambda v: v.whoami(), "/v1/whoami"),
    ("stats", lambda v: v.stats(), "/v1/stats"),
    ("connectivity", lambda v: v.connectivity(), "/v1/stats"),
    ("search", lambda v: v.search("q"), "/v1/search"),
    ("recall", lambda v: v.recall("q"), "/v1/recall"),
    ("get", lambda v: v.get("cl_1"), "/v1/memories/cl_1"),
    ("get_all", lambda v: v.get_all(), "/v1/memories"),
    ("count", lambda v: v.count(), "/v1/memories"),
    ("history", lambda v: v.history("user", "lives_in"), "/v1/history"),
    ("why", lambda v: v.why("cl_1"), "/v1/memories/cl_1/why"),
    ("ask", lambda v: v.ask("q"), "/v1/ask"),
    ("since", lambda v: v.since(datetime(2026, 1, 1, tzinfo=timezone.utc)), "/v1/since"),
    ("produced", lambda v: v.produced("ep_1"), "/v1/episodes/ep_1/produced"),
    ("neighborhood", lambda v: v.neighborhood("alice"), "/v1/neighborhood"),
    ("paths_between", lambda v: v.paths_between("alice", "berlin"), "/v1/paths"),
    ("standing", lambda v: v.standing(k=5), "/v1/standing"),
    ("add", lambda v: v.add("hi"), "/v1/memories"),
    ("remember", lambda v: v.remember("user", "likes", "tea"), "/v1/facts"),
    ("supersede", lambda v: v.supersede("cl_1", "user", "likes", "tea"),
     "/v1/memories/cl_1/supersede"),
    ("forget", lambda v: v.forget("user", "likes"), "/v1/forget"),
    ("delete", lambda v: v.delete("cl_1"), "/v1/memories/cl_1"),
    ("end", lambda v: v.end(claim_id="cl_1"), "/v1/end"),
    ("erase", lambda v: v.erase("cl_1"), "/v1/erasures"),
    ("purge", lambda v: v.purge(), "/v1/erasures"),
    ("consolidate", lambda v: v.consolidate(), "/v1/maintenance/consolidate"),
]


@pytest.mark.parametrize("name, call, path", DELEGATIONS,
                         ids=[row[0] for row in DELEGATIONS])
def test_each_scoped_method_reaches_the_same_endpoint_at_the_bound_scope(name, call,
                                                                        path):
    calls = []

    def handler(request):
        calls.append(request)
        return _answer(request)

    mem = _client(user="alice")
    mem._http._client._transport = httpx.MockTransport(handler)
    call(mem.scope(agent="a1"))

    params = dict(calls[-1].url.params)
    assert calls[-1].url.path == path
    if name in UNSCOPED:
        assert "user" not in params
        return
    assert params["user"] == "alice" and params["agent"] == "a1"
    assert "session" not in params


def test_every_public_method_on_the_view_is_covered_by_that_table():
    """The table is a list somebody maintains, so this is the check that it stays complete.

    A method added to `ScopedRemoteMemvara` without a row is a forwarding line nothing
    calls — which is how `search` came to return a different type here than on the client
    it forwards to, with every structural test in this file still green.
    """
    listed = {name for name, _, _ in DELEGATIONS} | {"memvara", "scope"}
    assert {name for name, _ in _public_methods()} - listed == set()
