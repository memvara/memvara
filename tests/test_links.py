"""Typed links between memories: `extends` and `derives`.

The acceptance tests from the design: links survive a round trip through the store,
erasing either end removes the row, and `memory_why` shows them. Around those, the
schema-13 migration from a real version-12 file, the scope rules on `link()` and
`links()`, and the `derives` links consolidation writes when it folds duplicates.
"""
import sqlite3
from datetime import datetime, timezone

import pytest

from memvara import Claim, HashingEmbedder, Link, Memvara, NullLLM, Scope, SQLiteStore
from memvara.consolidate import Consolidator
from memvara.server import MemvaraMCPServer
from memvara.store.sqlite import SCHEMA_VERSION

from test_server import call, text

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def mem(**kw) -> Memvara:
    kw.setdefault("user", "alice")
    return Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), **kw)


def two(m: Memvara) -> tuple[str, str]:
    a = m.remember("user", "migrating_to", "Postgres").added[0].id
    b = m.remember("user", "migration_plan", "three stages").added[0].id
    return a, b


# --- the store ----------------------------------------------------------------


def test_a_link_survives_a_round_trip_through_the_store(tmp_path):
    path = str(tmp_path / "links.db")
    store = SQLiteStore(path)
    stored = store.put_link("t", Link("cl_b", "cl_a", "extends", T0, "api"))
    assert stored == Link("cl_b", "cl_a", "extends", T0, "api")
    store.close()
    reopened = SQLiteStore(path)
    try:
        assert reopened.claim_links("t", "cl_a") == [
            Link("cl_b", "cl_a", "extends", T0, "api")]
        assert reopened.claim_links("t", "cl_b") == reopened.claim_links("t", "cl_a")
        assert reopened.claim_links("other-tenant", "cl_a") == []
    finally:
        reopened.close()


def test_recording_a_link_twice_keeps_the_first_row_and_returns_it():
    """The store hands back the row it kept, so a caller never reads it back, and a
    read-back is a second query that can see a different store than the write did."""
    store = SQLiteStore(":memory:")
    first = Link("cl_b", "cl_a", "derives", T0, "importer")
    assert store.put_link("t", first) == first
    later = Link("cl_b", "cl_a", "derives", datetime(2027, 1, 1, tzinfo=timezone.utc))
    assert store.put_link("t", later) == first
    assert store.claim_links("t", "cl_a") == [first]
    store.close()


def test_links_with_one_instant_come_back_in_a_stable_order():
    """Links written at one instant (an importer stamping one time) must still come back
    in one order, so the tie is broken on the far end's id rather than by the btree."""
    store = SQLiteStore(":memory:")
    for far in ("cl_z", "cl_a", "cl_m"):
        store.put_link("t", Link("cl_hub", far, "derives", T0))
    assert [k.to_id for k in store.claim_links("t", "cl_hub")] == ["cl_a", "cl_m", "cl_z"]
    store.close()


def test_the_table_refuses_a_relation_that_is_not_one_of_the_two():
    store = SQLiteStore(":memory:")
    with pytest.raises(sqlite3.IntegrityError):
        store.put_link("t", Link("a", "b", "supersedes", T0))  # type: ignore[arg-type]
    store.close()


def test_a_version_12_file_gains_the_link_table_and_keeps_its_claims(tmp_path):
    """A real upgrade: a file stamped 12 without the table, opened by this build."""
    path = str(tmp_path / "v12.db")
    m = mem(path=path)
    claim = m.remember("user", "lives_in", "Lisbon").added[0]
    m.close()
    raw = sqlite3.connect(path)
    raw.execute("DROP TABLE claim_links")
    raw.execute("PRAGMA user_version = 12")
    raw.commit()
    raw.close()

    upgraded = SQLiteStore(path)
    try:
        assert int(upgraded._db.execute("PRAGMA user_version").fetchone()[0]) == \
            SCHEMA_VERSION == 15
        tables = {r[0] for r in upgraded._db.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')")}
        assert {"claim_links", "cl_link_to"} <= tables
        assert upgraded.get_claim(claim.id).object == "Lisbon"
        assert upgraded._db.execute("SELECT COUNT(*) FROM claim_links").fetchone()[0] == 0
        upgraded._migrate_to_v13()                     # idempotent
        assert upgraded.put_link("default", Link(claim.id, "cl_x", "extends", T0))
    finally:
        upgraded.close()


# --- the library --------------------------------------------------------------


def test_link_records_a_relation_both_claims_can_see():
    m = mem()
    a, b = two(m)
    link = m.link(b, a, "extends")
    assert (link.from_id, link.to_id, link.relation, link.by) == (b, a, "extends", "api")
    assert m.links(a) == m.links(b) == [link]
    assert m.why(a).links == [link]


def test_linking_the_same_pair_twice_returns_the_stored_link():
    m = mem()
    a, b = two(m)
    first = m.link(b, a, "derives")
    assert m.link(b, a, "derives") == first
    assert len(m.links(a)) == 1


def test_a_repeated_link_does_not_read_the_row_back_through_a_scope_check(monkeypatch):
    """The duplicate path used to find the stored row by listing the claim's links, which
    filters by the far end's visibility. With the far end gone from view in between, the
    search found nothing and raised `StopIteration`, which the tool reported by that
    name. The row now comes straight back from the store's write."""
    m = mem()
    a, b = two(m)
    first = m.link(b, a, "extends")
    monkeypatch.setattr(m, "_links_of", lambda *_: pytest.fail("read back"))
    assert m.link(b, a, "extends") == first


def test_link_refuses_an_id_this_scope_cannot_see_and_a_self_link():
    m = mem()
    a, _ = two(m)
    bobs = m.remember("user", "likes", "tea", user="bob").added[0].id
    with pytest.raises(KeyError):
        m.link(a, bobs, "extends")
    with pytest.raises(KeyError):
        m.link("cl_nothing", a, "extends")
    with pytest.raises(ValueError, match="to itself"):
        m.link(a, a, "extends")
    with pytest.raises(ValueError, match="not a link relation"):
        m.link(a, bobs, "supersedes")
    assert m.links(a) == []


def test_links_hides_a_link_whose_far_end_this_scope_cannot_see():
    """The link exists; the id at its far end is somebody else's."""
    m = mem(user=None)
    shared = m.remember("user", "uses_tool", "ruff").added[0].id
    private = m.remember("user", "likes", "tea", user="bob").added[0].id
    m.link(private, shared, "extends", user="bob")
    assert [k.from_id for k in m.links(shared, user="bob")] == [private]
    assert m.links(shared) == [], "a tenant-level reader cannot see bob's claim"
    assert m.links(private, user="carol") == []


def test_a_store_without_links_refuses_to_record_one_and_reports_none():
    class NoLinks(SQLiteStore):
        put_link = None          # type: ignore[assignment]
        claim_links = None       # type: ignore[assignment]

    m = mem(store=NoLinks(":memory:"))
    a, b = two(m)
    with pytest.raises(NotImplementedError, match="cannot record links"):
        m.link(a, b, "extends")
    assert m.links(a) == [] and m.why(a).links == []


def test_why_dates_links_on_the_belief_clock():
    m = mem()
    a, b = two(m)
    link = m.link(b, a, "extends")
    assert m.why(a, known_at=link.created_at).links == [link]
    assert m.why(a, known_at=T0).links == []


def test_erasing_either_end_removes_the_link_in_the_same_transaction():
    m = mem()
    a, b = two(m)
    c = m.remember("user", "migration_owner", "platform team").added[0].id
    m.link(b, a, "extends")
    m.link(c, a, "extends")
    assert m.erase(b) is True
    assert m.store.residue(b)["claim_links"] == 0
    assert [k.from_id for k in m.links(a)] == [c]
    assert m.erase(a) is True
    assert m.links(c) == []
    assert m.store._db.execute("SELECT COUNT(*) FROM claim_links").fetchone()[0] == 0


def test_purging_a_scope_removes_every_link_that_touches_it():
    m = mem(user=None)
    shared = m.remember("user", "uses_tool", "ruff").added[0].id
    bobs = m.remember("user", "likes", "tea", user="bob").added[0].id
    m.link(bobs, shared, "derives", user="bob")
    m.purge(user="bob")
    assert m.links(shared) == []
    assert m.store._db.execute("SELECT COUNT(*) FROM claim_links").fetchone()[0] == 0


def test_the_scoped_and_async_views_link_and_list():
    import asyncio

    from memvara import AsyncMemvara

    m = mem()
    a, b = two(m)
    view = m.scope(user="alice")
    link = view.link(b, a, "extends")
    assert view.links(a) == [link]

    async def main():
        amem = AsyncMemvara(m)
        await amem.link(a, b, "derives", user="alice")
        scoped = amem.scope(user="alice")
        await scoped.link(b, a, "derives")
        return await amem.links(a, user="alice"), await scoped.links(b)

    via_facade, via_view = asyncio.run(main())
    assert len(via_facade) == 3 and len(via_view) == 3


# --- consolidation ------------------------------------------------------------


def _duplicates(m: Memvara) -> None:
    """Two spellings of one fact in one slot, written straight to the store as
    `tests/test_merge.py` does, because the write path would fold them on arrival."""
    for claim_id, obj, obs in (("cl_thin", "Acme", 1), ("cl_thick", "acme", 7)):
        m.store.put_claim(Claim(id=claim_id, subject="user", predicate="works_at",
                                object=obj, scope=Scope("default", "alice"),
                                observation_count=obs))


def test_a_consolidation_merge_is_recorded_as_a_supersession_and_writes_no_link():
    """A merge folds near-duplicates into one survivor. That is a supersession:
    `invalidated_by` records it and `why().superseded` reports it. A `derives` link would
    be a second record of the same fact, and `derives` is reserved for a claim inferred
    from other claims, which a merge does not create."""
    m = mem()
    _duplicates(m)
    assert Consolidator(m.store, m.embedder, m.registry).run()["merged"] == 1
    assert m.get("cl_thin").invalidated_by == "cl_thick"
    assert [c.id for c in m.why("cl_thick").superseded] == ["cl_thin"]
    assert m.links("cl_thick") == [] and m.why("cl_thick").links == []
    assert m.store._db.execute("SELECT COUNT(*) FROM claim_links").fetchone()[0] == 0


# --- the MCP surface ----------------------------------------------------------


@pytest.fixture()
def server():
    srv = MemvaraMCPServer(mem(), user="alice")
    yield srv
    srv.close()


def _ids(server):
    rows = {c.object: c.id for c in server._ctx.memory.get_all()}
    return rows["Postgres"], rows["three stages"]


def test_memory_link_records_and_memory_why_lists_it_from_both_ends(server):
    text(server, "memory_remember", {"predicate": "migrating_to", "object": "Postgres"})
    text(server, "memory_remember", {"predicate": "migration_plan",
                                     "object": "three stages"})
    a, b = _ids(server)
    body = text(server, "memory_link", {"from_id": b, "to_id": a,
                                        "relation": "extends"})
    assert body == (f"Linked: {b} extends {a}, meaning the first adds detail to the "
                    "second. memory_why on either one lists it.")
    why_a = text(server, "memory_why", {"claim_id": a})
    assert "Linked to 1 other memory(ies)." in why_a
    assert f"[{b} extends this] user migration plan three stages" in why_a
    why_b = text(server, "memory_why", {"claim_id": b})
    assert f"[this extends {a}] user migrating to Postgres" in why_b
    derived = text(server, "memory_link", {"from_id": a, "to_id": b,
                                           "relation": "derives"})
    assert "was inferred from the second" in derived


def test_memory_link_answers_an_invisible_id_without_saying_which(server):
    text(server, "memory_remember", {"predicate": "migrating_to", "object": "Postgres"})
    a = server._ctx.memory.get_all()[0].id
    body = text(server, "memory_link", {"from_id": a, "to_id": "cl_nothing",
                                        "relation": "extends"})
    assert body.startswith("Nothing linked: one of the two ids is not visible here.")


def test_memory_link_refuses_a_self_link_and_an_unknown_relation(server):
    text(server, "memory_remember", {"predicate": "migrating_to", "object": "Postgres"})
    a = server._ctx.memory.get_all()[0].id
    body, is_error = call(server, "memory_link", {"from_id": a, "to_id": a,
                                                  "relation": "extends"})
    assert is_error and "to itself" in body
    body, is_error = call(server, "memory_link", {"from_id": a, "to_id": a,
                                                  "relation": "supersedes"})
    assert is_error


def test_memory_why_says_so_when_a_linked_memory_vanished_between_two_reads(server,
                                                                            monkeypatch):
    """`why()` filters links to visible ends, so this needs a race: the far end erased
    after the links were read and before its text was."""
    text(server, "memory_remember", {"predicate": "migrating_to", "object": "Postgres"})
    text(server, "memory_remember", {"predicate": "migration_plan",
                                     "object": "three stages"})
    a, b = _ids(server)
    text(server, "memory_link", {"from_id": b, "to_id": a, "relation": "extends"})
    real_get = type(server._ctx.memory).get
    monkeypatch.setattr(type(server._ctx.memory), "get",
                        lambda self, cid: None if cid == b else real_get(self, cid))
    body = text(server, "memory_why", {"claim_id": a})
    assert f"[{b} extends this] (no longer visible)" in body
