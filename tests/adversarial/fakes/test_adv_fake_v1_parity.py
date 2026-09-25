"""FakeV1 serves every route the remote clients call, and the real client reads back from
it the answers a local store gives for the same calls."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import re
from datetime import timedelta
from typing import Any, Callable

import pytest

from harness import stores
from harness.env import REPO
from harness.fakes.fake_v1 import FakeV1
from memvara import MemoryType
from memvara.schema import BUILTIN_PREDICATES, PredicateRegistry, PredicateSpec
from memvara.types import (ENTITY_REKEY, LAST_OBSERVED, OBJECT_ENTITY, SALIENCE_BASE,
                           SUBJECT_ENTITY, Claim, utcnow)

#: The two clients whose calls define the routes.
CLIENTS = (REPO / "memvara" / "remote" / "api.py", REPO / "memvara" / "remote" / "aio.py")

#: An id no store holds.
MISSING = "cl_00000000000000000000"


def _template(node: ast.expr) -> str | None:
    """A path argument as a route template with every interpolated part written `{}`, or
    None for a plain name, which is a helper forwarding a path its caller chose."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(part.value if isinstance(part, ast.Constant) else "{}"
                       for part in node.values)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "_document_path"):
        tail = node.args[1] if len(node.args) > 1 else ast.Constant("")
        assert isinstance(tail, ast.Constant), ast.dump(node)
        return "/v1/documents/{}" + str(tail.value)
    assert isinstance(node, ast.Name), f"a path this test cannot read: {ast.dump(node)}"
    return None


def client_routes() -> set[str]:
    """Every `METHOD /v1/...` route the two clients call, read from their source."""
    found: set[str] = set()
    for source in CLIENTS:
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr == "_request" and len(node.args) >= 2:
                method, path = node.args[0], node.args[1]
            elif node.func.attr == "_read" and node.args:
                # `_read` always POSTs: it is the helper for the two reads that carry a
                # query in their body.
                method, path = ast.Constant("POST"), node.args[0]
            else:
                continue
            template = _template(path)
            if template is not None and isinstance(method, ast.Constant):
                found.add(f"{method.value} {template}")
    return found


def test_the_fake_serves_exactly_the_routes_the_clients_call() -> None:
    called = client_routes()
    served = {re.sub(r"\{\w+\}", "{}", name) for name in FakeV1.ROUTES}
    assert called, "the scan found no calls, so it no longer reads the clients"
    assert sorted(called - served) == [], "routes the clients call that FakeV1 lacks"
    assert sorted(served - called) == [], "routes FakeV1 serves that no client calls"


def _shape(claim: Claim) -> tuple[Any, ...]:
    """What a claim says and its state, without its id and its instants, which differ
    between two stores that were told the same things."""
    state = ("retired" if claim.invalidated_at is not None
             else "ended" if claim.valid_to is not None else "live")
    return (claim.subject, claim.predicate, claim.object, claim.text, state,
            claim.memory_type, claim.polarity, claim.confidence, claim.derivation,
            claim.extractor, claim.scope)


def _program(mem: Any) -> dict[str, Any]:
    """remember, get, get_all, search, forget, delete and erase, on one small program,
    returning every answer in a form that two stores can be compared by."""
    out: dict[str, Any] = {}
    berlin = mem.remember("user", "lives_in", "Berlin").added[0]
    moved = mem.remember("user", "lives_in", "Lisbon")
    lisbon = moved.added[0]
    out["correction"] = ([_shape(c) for c in moved.added],
                         [_shape(c) for c in moved.closed])
    mem.remember("user", "prefers", "tabs for indentation",
                 memory_type=MemoryType.PROCEDURAL)
    out["get"] = [_shape(mem.get(lisbon.id)), _shape(mem.get(berlin.id)),
                  mem.get(MISSING)]
    out["get_all"] = sorted(_shape(c) for c in mem.get_all())
    out["get_all_every_state"] = sorted(
        _shape(c) for c in mem.get_all(states=["live", "ended", "retired"]))
    out["search"] = [(_shape(r.claim), round(r.score, 6))
                     for r in mem.search("where does the user live", k=5)]
    out["forget"] = sorted(_shape(c) for c in mem.forget("user", "prefers"))
    out["delete"] = [mem.delete(lisbon.id), mem.delete(lisbon.id), mem.delete(MISSING)]
    out["after_delete"] = _shape(mem.get(lisbon.id))
    out["erase"] = [mem.erase(berlin.id), mem.erase(berlin.id)]
    out["after_erase"] = mem.get(berlin.id)
    out["end"] = sorted(_shape(c) for c in mem.get_all(states=["live", "ended", "retired"]))
    return out


def test_the_remote_client_gets_the_answers_a_local_store_gives(fake_v1: FakeV1) -> None:
    local = stores.memory()
    try:
        assert _program(fake_v1.remote(user="alice")) == _program(local.scope(user="alice"))
    finally:
        local.close()


class _Blocking:
    """Runs an async client's calls to completion one at a time, so the same program can
    drive it."""

    def __init__(self, client: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._client = client
        self._loop = loop

    def __getattr__(self, name: str) -> Callable[..., Any]:
        method = getattr(self._client, name)
        return lambda *args, **kwargs: self._loop.run_until_complete(method(*args, **kwargs))


def test_the_async_client_gets_the_same_answers(fake_v1: FakeV1) -> None:
    loop = asyncio.new_event_loop()
    local = stores.memory()
    client = fake_v1.aremote(user="alice")
    try:
        assert _program(_Blocking(client, loop)) == _program(local.scope(user="alice"))
    finally:
        loop.run_until_complete(client.aclose())
        loop.close()
        local.close()


#: Fields `/v1` does not carry. The cloud's wire model has no place for them, so the
#: client fills in each one's default. The parity tests are where a difference would show.
_NOT_ON_THE_WIRE = {"temporal_precision", "object_kind", "amount", "unit"}

#: `Claim.meta` keys the wire does not carry as they are stored. Three are left out of
#: `metadata`, and two travel as the top-level `salience_base` and `last_observed`, whose
#: properties this test compares instead.
_BOOKKEEPING = {SALIENCE_BASE, LAST_OBSERVED, SUBJECT_ENTITY, OBJECT_ENTITY, ENTITY_REKEY}


def test_every_field_the_wire_carries_survives_the_round_trip(fake_v1: FakeV1) -> None:
    """The client's hydration is the inverse of the fake's rendering, field by field, for
    claims in all three states, with a restatement, sources, metadata, an expiry and a
    closure reason."""
    remote = fake_v1.remote(user="alice")
    remote.remember("user", "lives_in", "Berlin", valid_from=utcnow() - timedelta(days=30))
    remote.remember("user", "lives_in", "Berlin")
    moved = remote.remember("user", "lives_in", "Lisbon", topic="relocation",
                            sources=[{"role": "user", "content": "I moved to Lisbon"}])
    remote.remember("user", "prefers", "short answers", memory_type=MemoryType.PROCEDURAL,
                    expires_at=utcnow() + timedelta(days=3), expire_reason="a trial")
    remote.delete(moved.added[0].id, reason="it was Porto")
    local = fake_v1.memvara.scope(user="alice")
    hydrated = remote.get_all(states=["live", "ended", "retired"])
    assert sorted(c.object for c in hydrated) == ["Berlin", "Lisbon", "short answers"]
    assert max(c.observation_count for c in hydrated) == 2
    for claim in hydrated:
        stored = local.get(claim.id)
        assert stored is not None
        for field in dataclasses.fields(Claim):
            if field.name in _NOT_ON_THE_WIRE or field.name == "meta":
                continue
            assert getattr(claim, field.name) == getattr(stored, field.name), field.name
        assert claim.salience_base == stored.salience_base
        if stored.last_observed is None:
            assert claim.last_observed is None
        else:
            # Epoch seconds travel as an ISO instant, so they come back to the microsecond.
            assert claim.last_observed is not None
            assert claim.last_observed.timestamp() == pytest.approx(
                stored.last_observed.timestamp(), abs=1e-5)
        assert ({k: v for k, v in claim.meta.items() if k not in _BOOKKEEPING}
                == {k: v for k, v in stored.meta.items() if k not in _BOOKKEEPING})


def _walkable() -> PredicateRegistry:
    """The builtins plus one entity-valued predicate, so the store has a graph to walk."""
    return PredicateRegistry(BUILTIN_PREDICATES + (
        PredicateSpec("reports_to", object_type=("entity",), graph=True),))


def test_every_route_answers_the_client_that_calls_it() -> None:
    """One call through the real client for every route the fake serves, each answered
    with a 2xx that the client could read back into its own types."""
    with FakeV1(stores.memory(registry=_walkable())) as fake:
        remote = fake.remote(user="alice")
        assert remote.health()["status"] == "ok"
        assert remote.whoami()["scope"]["tenant"] == "default"

        added = remote.add("I moved to Lisbon last year.")
        assert [c.object for c in added.added] == ["Lisbon"]
        lisbon = added.added[0]
        assert remote.get(lisbon.id) == lisbon
        assert remote.stats()["claims"] == 1
        assert remote.service()["extractor"] == "fast-path-only"
        assert remote.connectivity()["live_claims"] == 1
        assert remote.count() == 1
        assert "Lisbon" in remote.recall("where does the user live")
        assert [c.id for c in remote.history("user", "lives_in")] == [lisbon.id]
        provenance = remote.why(lisbon.id)
        assert provenance is not None and provenance.episodes[0].content.startswith("I moved")
        assert [c.id for c in remote.produced(added.episode_ids[0])] == [lisbon.id]
        assert remote.ask("where does the user live").readings
        assert [c.id for c in remote.since(utcnow() - timedelta(hours=1)).added] == [lisbon.id]

        remote.remember("alice", "reports_to", "bob")
        remote.remember("bob", "reports_to", "carol")
        assert remote.neighborhood("alice", depth=2)
        assert remote.paths_between("alice", "carol")[0].nodes[-1] == "carol"

        rule = remote.remember("user", "prefers", "tabs", memory_type=MemoryType.PROCEDURAL)
        assert [c.object for c in remote.standing()] == ["tabs"]
        assert [r.text for r in remote.profile().standing] == [rule.added[0].text]
        newer = remote.supersede(rule.added[0].id, "user", "prefers", "spaces",
                                 memory_type=MemoryType.PROCEDURAL)
        assert [c.id for c in newer.closed] == [rule.added[0].id]
        assert remote.link(newer.added[0].id, lisbon.id, "extends").relation == "extends"
        assert [k.to_id for k in remote.links(newer.added[0].id)] == [lisbon.id]

        city = remote.remember("user", "works_at", "Acme").added[0]
        assert remote.end(claim_id=city.id)
        assert remote.delete(lisbon.id, close="ended")
        assert [c.object for c in remote.forget("user", "prefers", close="ended")] == ["spaces"]
        tea = remote.remember("user", "likes", "green tea").added[0]
        assert remote.delete(tea.id)
        remote.remember("user", "likes", "coffee")
        assert [c.object for c in remote.forget("user", "likes")] == ["coffee"]
        preview = remote.forget_matching("reports_to", close="retired", k=5)
        confirmed = remote.forget_matching("reports_to", close="retired", k=5,
                                           confirm=preview.confirm)
        assert sorted(c.id for c in confirmed.closed) == sorted(preview.matches)

        doc = remote.add_document("A runbook. Restart the service.", custom_id="docs/runbook",
                                  title="Runbook")
        assert remote.get_document("docs/runbook").id == doc.id
        assert [d.id for d in remote.list_documents().items] == [doc.id]
        assert remote.update_document(doc.id, title="The runbook").title == "The runbook"
        assert remote.document_status("docs/runbook").id == doc.id
        assert remote.delete_document("docs/runbook").deleted
        other = remote.add_document("Second note.", custom_id="docs/other")
        assert [r.deleted for r in remote.delete_documents([other.id, "docs/none"])] == [True, False]

        assert remote.erase(city.id) and remote.get(city.id) is None
        assert remote.purge()["claims"] > 0
        assert remote.consolidate()["status"] == "succeeded"
        assert remote.search("anything") == []

    answered = {r.route for r in fake.requests if r.status is not None and r.status < 300}
    assert sorted(set(FakeV1.ROUTES) - answered) == []
