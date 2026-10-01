"""`Store.batch(tenant=...)`: which tenant `Memvara` names when it opens a transaction.

A store may narrow its write lock to one tenant when the library tells it which one, so
what matters here is the tenant each operation passes, and that an operation which might
touch several tenants passes none. The first group of tests records the `tenant` every
batch is opened with. The second group checks the two kinds of store that cannot take the
keyword (SQLite ignores it, and a store written before it existed does not have it).

Runs fully offline against `SQLiteStore(":memory:")` and `HashingEmbedder`.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone

import pytest

from memvara import Memvara
from memvara.compat._notes import build_note, write_note
from memvara.consolidate import Consolidator
from memvara.embed import HashingEmbedder
from memvara.llm.base import NullLLM
from memvara.store import SQLiteStore, sole_tenant, transaction
from memvara.store.base import batch_takes_tenant
from memvara.types import Claim, Episode, Scope, utcnow
from memvara.write.reconcile import backfill_entities, split_entity

ACME = "acme"
GLOBEX = "globex"


class RecordingStore:
    """A `SQLiteStore` that records the `tenant` every batch is opened with."""

    def __init__(self) -> None:
        self.inner = SQLiteStore(":memory:")
        self.opened: list[str | None] = []

    @contextmanager
    def batch(self, *, tenant: str | None = None):
        self.opened.append(tenant)
        with self.inner.batch(tenant=tenant):
            yield self

    def __getattr__(self, name):
        return getattr(self.inner, name)


class LegacyStore:
    """A store written before `batch` took a keyword: `batch()` and nothing else."""

    def __init__(self) -> None:
        self.inner = SQLiteStore(":memory:")
        self.opened = 0

    @contextmanager
    def batch(self):
        self.opened += 1
        with self.inner.batch():
            yield self

    def __getattr__(self, name):
        return getattr(self.inner, name)


def handle(store, tenant: str = ACME) -> Memvara:
    return Memvara(store=store, tenant=tenant, user="alice", llm=NullLLM(),
                   embedder=HashingEmbedder(dim=512))


@pytest.fixture()
def store():
    s = RecordingStore()
    yield s
    s.inner.close()


@pytest.fixture()
def mem(store):
    return handle(store)


def opened_by(store: RecordingStore, operation) -> list[str | None]:
    """The tenants passed by every batch `operation` opens."""
    store.opened.clear()
    operation()
    return list(store.opened)


# --- every operation bound to one tenant passes that tenant -----------------------


def test_remember_passes_its_tenant(store, mem):
    seen = opened_by(store, lambda: mem.remember("user", "lives_in", "Berlin"))
    assert seen and set(seen) == {ACME}, "remember() opened a batch without its tenant"


def test_add_passes_its_tenant_for_the_turns_and_for_the_claims(store, mem):
    seen = opened_by(store, lambda: mem.add("I live in Lisbon."))
    # The turn commits first, then the claims, then the turn's vector: three batches.
    assert len(seen) >= 3 and set(seen) == {ACME}


def test_a_replacement_passes_its_tenant_through_both_batches(store, mem):
    old = mem.remember("user", "works_at", "Initech").added[0]
    seen = opened_by(store, lambda: mem.remember("user", "works_at", "Globex Corp",
                                                 replaces=old.id))
    assert seen and set(seen) == {ACME}


def test_supersede_passes_its_tenant(store, mem):
    old = mem.remember("user", "lives_in", "Berlin").added[0]
    new = Claim(subject="user", predicate="lives_in", object="Lisbon", scope=old.scope)
    seen = opened_by(store, lambda: mem.supersede(old.id, new))
    assert seen and set(seen) == {ACME}


def test_a_replacement_that_names_no_scope_adopts_the_callers_tenant(store, mem):
    old = mem.remember("user", "lives_in", "Berlin").added[0]
    new = Claim(subject="user", predicate="lives_in", object="Lisbon")   # Scope()
    seen = opened_by(store, lambda: mem.supersede(old.id, new))
    assert seen and set(seen) == {ACME}


def test_delete_passes_its_tenant(store, mem):
    claim = mem.remember("user", "lives_in", "Berlin").added[0]
    seen = opened_by(store, lambda: mem.delete(claim.id))
    assert seen and set(seen) == {ACME}


def test_forget_passes_its_tenant_for_the_slot_and_for_the_closing(store, mem):
    mem.remember("user", "lives_in", "Berlin")
    seen = opened_by(store, lambda: mem.forget("user", "lives_in"))
    assert len(seen) >= 2 and set(seen) == {ACME}


def test_forget_matching_passes_its_tenant_when_it_confirms(store, mem):
    mem.remember("user", "lives_in", "Berlin")
    preview = mem.forget_matching("Berlin", close="retired")
    seen = opened_by(store, lambda: mem.forget_matching(
        "Berlin", close="retired", confirm=preview.confirm))
    assert seen and set(seen) == {ACME}


def test_link_passes_its_tenant(store, mem):
    a = mem.remember("user", "lives_in", "Berlin").added[0]
    b = mem.remember("user", "works_at", "Initech").added[0]
    seen = opened_by(store, lambda: mem.link(a.id, b.id, "extends"))
    assert seen and set(seen) == {ACME}


@pytest.mark.parametrize("sources", [False, True])
def test_erase_passes_its_tenant(store, mem, sources):
    claim = mem.remember("user", "lives_in", "Berlin").added[0]
    seen = opened_by(store, lambda: mem.erase(claim.id, sources=sources))
    assert seen and set(seen) == {ACME}


def test_the_expiry_sweep_passes_the_tenant_of_each_claim_it_erases(store):
    soon = utcnow() + timedelta(days=1)
    handle(store, ACME).remember("user", "door_code", "4411", expires_at=soon)
    handle(store, GLOBEX).remember("user", "door_code", "7788", expires_at=soon)
    seen = opened_by(store, lambda: handle(store).erase_expired(
        now=utcnow() + timedelta(days=2)))
    assert sorted(seen) == [ACME, GLOBEX], (
        "each expired claim is erased in a batch for its own tenant, "
        "whichever tenant the sweeping handle is bound to")


def test_consolidation_passes_the_tenant_it_sweeps(store, mem):
    mem.remember("user", "lives_in", "Berlin")
    mem.remember("user", "works_at", "Initech")
    seen = opened_by(store, lambda: mem.consolidate())
    assert seen and set(seen) == {ACME}


def test_a_predicate_merge_passes_its_tenant(store, mem):
    mem.remember("user", "home_city", "Berlin")
    seen = opened_by(store, lambda: mem.merge_predicate("home_city", "lives_in",
                                                        dry_run=False))
    assert seen and set(seen) == {ACME}


def test_an_entity_backfill_passes_its_tenant(store, mem):
    mem.remember("user", "lives_in", "Berlin")
    seen = opened_by(store, lambda: backfill_entities(mem.writer.reconciler, ACME,
                                                      dry_run=False))
    assert seen == [ACME]


def test_an_entity_split_passes_its_tenant(store, mem):
    first = datetime(2018, 1, 1, tzinfo=timezone.utc)
    second = datetime(2026, 1, 1, tzinfo=timezone.utc)
    mem.remember("John Smith", "works_at", "Acme", valid_from=first, recorded_at=first)
    mem.remember("John Smith", "works_at", "Globex", valid_from=second, recorded_at=second)
    seen = opened_by(store, lambda: split_entity(
        mem.writer.reconciler, Scope(ACME, user="alice"), "John Smith",
        datetime(2020, 1, 1, tzinfo=timezone.utc), dry_run=False))
    assert seen == [ACME]


def test_a_document_passes_its_tenant_when_it_is_added_updated_and_deleted(store, mem):
    seen = opened_by(store, lambda: mem.add_document("The office is in Lisbon.",
                                                     custom_id="office"))
    assert seen and set(seen) == {ACME}, "add_document"
    seen = opened_by(store, lambda: mem.update_document(
        "office", content="The office is in Porto."))
    assert seen and set(seen) == {ACME}, "update_document"
    seen = opened_by(store, lambda: mem.update_document("office", title="Office"))
    assert seen and set(seen) == {ACME}, "update_document, metadata only"
    seen = opened_by(store, lambda: mem.delete_document("office"))
    assert seen and set(seen) == {ACME}, "delete_document"


def test_a_note_import_passes_the_tenant_of_the_note_and_the_note_it_replaces(store, mem):
    scope = Scope(ACME, user="alice")
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first, first_turn = build_note(memory_id="m1", text="Likes tea", scope=scope, ts=when)
    seen = opened_by(store, lambda: write_note(mem, first, first_turn))
    assert seen == [ACME]
    second, second_turn = build_note(memory_id="m1", text="Likes coffee", scope=scope,
                                     ts=when + timedelta(days=1))
    seen = opened_by(store, lambda: write_note(mem, second, second_turn, retire=first))
    assert seen == [ACME]


# --- an operation that may touch several tenants passes none ------------------------


def test_reembedding_passes_no_tenant_because_it_walks_every_tenant(store):
    handle(store, ACME).remember("user", "lives_in", "Berlin")
    handle(store, GLOBEX).remember("user", "lives_in", "Lisbon")
    seen = opened_by(store, lambda: handle(store).reembed())
    assert seen == [None]


def test_a_sweep_over_every_tenant_passes_no_tenant(store, mem):
    mem.remember("user", "lives_in", "Berlin")
    seen = opened_by(store, lambda: Consolidator(
        store, HashingEmbedder(dim=512), mem.registry).run(None))
    assert seen and set(seen) == {None}


def test_turns_in_two_tenants_open_batches_with_no_tenant(store, mem):
    turns = [Episode(content="I live in Lisbon.", scope=Scope(ACME, user="alice")),
             Episode(content="I work at Initech.", scope=Scope(GLOBEX, user="bob"))]
    seen = opened_by(store, lambda: mem.add(turns))
    assert seen and set(seen) == {None}


def test_a_replacement_filed_in_another_tenant_passes_no_tenant(store, mem):
    old = mem.remember("user", "lives_in", "Berlin").added[0]
    new = Claim(subject="user", predicate="lives_in", object="Lisbon",
                scope=Scope(GLOBEX, user="alice"))
    seen = opened_by(store, lambda: mem.supersede(old.id, new))
    assert seen and set(seen) == {None}


def test_a_claim_citing_a_turn_scoped_to_another_tenant_passes_no_tenant(store, mem):
    turn = Episode(content="Alice lives in Berlin.", scope=Scope(GLOBEX, user="alice"))
    seen = opened_by(store, lambda: mem.remember("user", "lives_in", "Berlin",
                                                 sources=[turn]))
    assert seen and set(seen) == {None}


def test_a_turn_that_names_no_scope_takes_the_claims_tenant(store, mem):
    turn = Episode(content="Alice lives in Berlin.")      # Scope(), tenant "default"
    seen = opened_by(store, lambda: mem.remember("user", "lives_in", "Berlin",
                                                 sources=[turn]))
    assert seen and set(seen) == {ACME}


def test_the_tenant_of_a_set_is_none_unless_it_holds_exactly_one():
    assert sole_tenant([ACME, ACME]) == ACME
    assert sole_tenant([ACME, GLOBEX]) is None
    assert sole_tenant([]) is None


# --- a store that cannot take the keyword -------------------------------------------


def test_a_store_whose_batch_has_no_tenant_keyword_still_works_everywhere():
    store = LegacyStore()
    mem = handle(store)
    old = mem.remember("user", "works_at", "Initech").added[0]
    mem.remember("user", "works_at", "Globex Corp", replaces=old.id)
    mem.add("I live in Lisbon.")
    claim = mem.remember("user", "likes", "tea").added[0]
    mem.forget("user", "likes")
    mem.erase(claim.id)
    mem.add_document("The office is in Lisbon.", custom_id="office")
    mem.consolidate()
    mem.reembed()
    assert store.opened > 0, "the library opened no batch at all"
    live = {c.object for c in mem.get_all()}
    assert {"Globex Corp", "Lisbon"} <= live and "Initech" not in live
    assert "tea" not in live
    store.inner.close()


def test_a_store_with_no_batch_at_all_still_works():
    store = SQLiteStore(":memory:")

    class NoBatch:
        def __getattr__(self, name):
            if name == "batch":
                raise AttributeError(name)
            return getattr(store, name)

    mem = handle(NoBatch())
    mem.remember("user", "lives_in", "Berlin")
    assert [r.claim.object for r in mem.search("Berlin", k=3)] == ["Berlin"]
    store.close()


def test_a_store_taking_kwargs_is_given_the_tenant():
    seen: list[dict] = []

    class Star:
        @contextmanager
        def batch(self, **kwargs):
            seen.append(kwargs)
            yield self

    with transaction(Star(), ACME):
        pass
    with transaction(Star(), None):
        pass
    assert seen == [{"tenant": ACME}, {}]


def test_the_keyword_is_decided_from_the_signature():
    assert not batch_takes_tenant(lambda: nullcontext())
    assert batch_takes_tenant(lambda *, tenant=None: nullcontext())
    assert batch_takes_tenant(lambda tenant=None: nullcontext())
    assert batch_takes_tenant(lambda **kw: nullcontext())
    assert not batch_takes_tenant(lambda tenant, /: nullcontext())


    class Unreadable:
        __signature__ = "not a signature"     # `inspect.signature` raises `TypeError`

        def __call__(self):  # pragma: no cover - never called
            return nullcontext()

    assert not batch_takes_tenant(Unreadable())


def test_a_type_error_inside_a_batch_that_takes_the_keyword_is_not_retried_without_it():
    """The reason the keyword is detected from the signature: catching `TypeError` would
    call the batch a second time without it, and hide a real fault in the first."""
    calls = []

    class Broken:
        def batch(self, *, tenant=None):
            calls.append(tenant)
            raise TypeError("a fault inside the store")

    with pytest.raises(TypeError, match="a fault inside"):
        transaction(Broken(), ACME)
    assert calls == [ACME]


def test_a_tenant_is_not_passed_when_there_is_none_to_pass():
    store = RecordingStore()
    with transaction(store):
        pass
    assert store.opened == [None]
    store.inner.close()


# --- SQLite behaves exactly as before ------------------------------------------------


def test_sqlite_takes_the_same_write_lock_whatever_tenant_a_batch_names(tmp_path):
    """The keyword is ignored: a batch for one tenant holds the whole file's write lock,
    so a batch for another tenant on a second handle cannot begin until it ends."""
    path = str(tmp_path / "m.db")
    first, second = SQLiteStore(path), SQLiteStore(path)
    second._db.execute("PRAGMA busy_timeout = 50")
    try:
        with first.batch(tenant=ACME):
            with pytest.raises(Exception, match="locked"):
                with second.batch(tenant=GLOBEX):
                    pass            # pragma: no cover - the lock refuses it first
        with second.batch(tenant=GLOBEX):    # the lock is free again
            pass
    finally:
        first.close()
        second.close()


def test_sqlite_commits_one_batch_for_a_tenant_exactly_as_one_for_none():
    s = SQLiteStore(":memory:")
    claim = Claim(subject="user", predicate="lives_in", object="Berlin",
                  scope=Scope(ACME, user="alice"))
    with pytest.raises(RuntimeError):
        with s.batch(tenant=ACME):
            s.put_claim(claim)
            raise RuntimeError("abandon the batch")
    assert s.get_claim(claim.id) is None, "a batch for a tenant still rolls back"
    with s.batch(tenant=ACME):
        s.put_claim(claim)
    assert s.get_claim(claim.id) is not None
    s.close()
