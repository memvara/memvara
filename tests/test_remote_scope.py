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

import httpx
import pytest

from memvara.remote.api import RemoteMemvara, ScopedRemoteMemvara
from memvara.types import Scope

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


def test_no_public_member_at_all_accepts_a_scope_argument():
    """The parametrized list above is a list somebody maintains. This is the same check
    derived from the class, so a method added later cannot slip past by not being on it.
    """
    offenders = {}
    for name in dir(ScopedRemoteMemvara):
        if name.startswith("_"):
            continue
        member = inspect.getattr_static(ScopedRemoteMemvara, name)
        if not callable(member):
            continue
        found = SCOPE_ARGUMENTS & set(inspect.signature(member).parameters)
        if found:
            offenders[name] = sorted(found)
    assert offenders == {}


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
