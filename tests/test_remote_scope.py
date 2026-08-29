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
    """Required rather than a leak. `ScopedMemvara` has this property, `server/tools.py`
    calls `ctx.memory.memvara`, and the protocol in a later task declares it. Against a
    hosted deployment the credential binds the tenant, so what comes back cannot address
    another one either."""
    view = _client(user="alice").scope(agent="a1")
    assert isinstance(view.memvara, RemoteMemvara)
    assert view.memvara.default_scope == Scope("default", "alice", "a1", None)


def test_the_scope_is_readable_as_an_attribute():
    """`server/tools.py` reads `ctx.memory.scope` to report where it is, and the protocol
    in a later task declares it as a member both implementations have."""
    view = _client(user="alice").scope()
    assert isinstance(view.scope, Scope)
    assert view.scope.user == "alice"


def test_the_view_shares_one_connection_pool_with_the_client_it_came_from():
    """Two handles on one deployment and one credential. A second pool would be two
    idempotency stores as well, which is the thing the transport's retry depends on."""
    mem = _client()
    assert mem.scope(user="alice").memvara._http is mem._http
