"""Invariants of the hosted client that no older test checked.

`RemoteMemvara` talks to a memvara-cloud deployment over its `/v1` REST API. These tests
point the real client at `FakeV1`, a fake of that API answered by a local store, and read
every request the client sent. The invariants are in the "Invariants and assumptions"
section of `docs/claude/remote-and-cloud.md`.
"""

from __future__ import annotations

import inspect
import json

import pytest

from harness.fakes.fake_v1 import FakeV1
from memvara.remote.api import RemoteMemvara, ScopedRemoteMemvara
from memvara.types import Scope

#: The names a call could use to address a scope.
_SCOPE_NAMES = {"tenant", "user", "agent", "session", "project"}


def _everything_sent(fake: FakeV1) -> list[str]:
    """Each request the fake received, as one string of its path, query, headers and
    body, so a test can search all of it at once."""
    return [" ".join([request.method, request.path, json.dumps(request.query),
                      json.dumps(request.headers), request.body.decode("utf-8")])
            for request in fake.requests]


@pytest.mark.covers("inv:RC1")
def test_the_client_never_sends_the_tenant_it_holds_and_the_credential_decides() -> None:
    """docs/claude/remote-and-cloud.md: "The credential decides the tenant."
    `default_scope` holds a tenant for local bookkeeping and never sends it, because a
    tenant the caller could set would be a request to be trusted about identity.

    The client is built with a tenant that is not the credential's, and then writes and
    reads through several routes. No request may carry that tenant in its path, query,
    headers or body, and the memory it wrote must land in the credential's tenant.
    """
    with FakeV1(tenant="acme") as fake:
        client = fake.remote(tenant="claimed-tenant", user="alice")
        assert client.default_scope.tenant == "claimed-tenant"
        client.remember("user", "lives_in", "Lisbon")
        client.add("I work at Initech.")
        client.search("where does alice live")
        client.recall("where does alice live")
        client.get_all()
        client.count()
        client.history("user", "lives_in")
        client.stats()
        client.whoami()
        client.close()

        sent = _everything_sent(fake)
        assert len(sent) >= 9 and all(r.status == 200 for r in fake.requests)
        for request in fake.requests:
            assert "tenant" not in request.query, request.path
        assert not [line for line in sent if "claimed-tenant" in line]

        stored = fake.memvara.get_all(tenant="acme", user="alice")
        assert "Lisbon" in {c.object for c in stored}
        assert fake.memvara.get_all(tenant="claimed-tenant", user="alice") == []


@pytest.mark.covers("inv:RC2")
def test_a_narrowed_view_has_no_way_back_out() -> None:
    """docs/claude/remote-and-cloud.md: "Narrowing is one-way. `RemoteMemvara.scope()`
    returns a view with no way back out."

    A view bound to alice and one of her agents is narrowed again with every field left
    out or passed as None, and it keeps every bound field. No public member of the view
    takes a scope argument. Every request the view and the client under it send names
    alice and the agent, and nothing written in bob's scope is ever read through them.
    """
    with FakeV1(tenant="acme") as fake:
        bob = fake.remote(user="bob")
        bob.remember("user", "lives_in", "Oslo")
        bob.close()
        before = len(fake.requests)

        alice = fake.remote(user="alice")
        view = alice.scope(agent="a1")
        bound = Scope("default", "alice", "a1", None)
        assert view.scope == bound
        assert alice.scope(user=None, agent=None, session=None).scope.user == "alice"
        assert view.memvara.scope().scope == bound
        assert view.memvara.scope(user=None, agent=None, session=None).scope == bound

        takes_scope = {
            name for name in dir(ScopedRemoteMemvara) if not name.startswith("_")
            and callable(member := inspect.getattr_static(ScopedRemoteMemvara, name))
            and _SCOPE_NAMES & set(inspect.signature(member).parameters)}
        assert takes_scope == set(), takes_scope

        view.remember("user", "likes", "tea")
        read = {c.object for c in view.get_all()}
        read |= {r.claim.object for r in view.search("where does the user live")}
        read |= {c.object for c in view.memvara.scope().get_all()}
        assert "tea" in read and "Oslo" not in read, read

        for request in fake.requests[before:]:
            assert request.param("user") == "alice", request.path
            assert request.param("agent") == "a1", request.path
        alice.close()
        assert isinstance(view.memvara, RemoteMemvara)
