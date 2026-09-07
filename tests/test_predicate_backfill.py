"""`backfill_predicates` and `Memvara.merge_predicate`.

`PredicateRegistry.learn_alias` applies from the moment it is learned, so the claims
already filed under the old name stay in their old slot. These tests pin the repair:
moved claims change predicate and slot, the slots they land in are replayed so
duplicates fold and older values close, slots nothing moved into are left alone, a dry
run touches nothing, and each moved claim carries a dated note saying where it came from.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from memvara import Memvara, MergeReport, NullLLM, backfill_predicates
from memvara.embed import HashingEmbedder
from memvara.schema import Cardinality, PredicateRegistry
from memvara.store import SQLiteStore
from memvara.types import PREDICATE_REKEY, Claim, Scope, utcnow
from memvara.write import Reconciler

SCOPE = Scope("acme", "alice")


@pytest.fixture()
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def rec(store) -> Reconciler:
    return Reconciler(store, PredicateRegistry())


def _write(store, predicate: str, obj: str, minutes: int, *, text: str | None = None,
           subject: str = "user") -> Claim:
    """A claim recorded `minutes` ago under `predicate`, exactly as a session wrote it."""
    at = utcnow() - timedelta(minutes=minutes)
    c = Claim(subject=subject, predicate=predicate, object=obj, scope=SCOPE,
              valid_from=at, recorded_at=at, **({"text": text} if text else {}))
    store.put_claim(c)
    return c


# --- the pass itself ------------------------------------------------------------


def test_dry_run_reports_and_changes_nothing(rec, store):
    a = _write(store, "works_at", "Acme", 30)
    b = _write(store, "hired_by", "Globex", 20)
    report = backfill_predicates(rec, "acme", aliases={"hired_by": "works_at"})
    assert report.dry_run and report.scanned == 2
    assert report.moved == 1 and report.retired == 1 and report.written == 0
    assert store.get_claim(b.id).predicate == "hired_by"
    assert store.get_claim(a.id).valid_to is None
    # The registry was not taught anything either: the mapping came from the caller.
    assert rec.registry.normalize("hired_by") == "hired_by"


def test_a_moved_claim_changes_predicate_slot_text_and_carries_a_note(rec, store):
    moved = _write(store, "hired_by", "Globex", 20)
    old_key = moved.fact_key
    t = utcnow()
    backfill_predicates(rec, "acme", aliases={"hired_by": "works_at"}, dry_run=False,
                        now=t)
    got = store.get_claim(moved.id)
    assert got.predicate == "works_at"
    assert got.fact_key != old_key
    assert got.fact_key == Claim(subject="user", predicate="works_at", object="Globex",
                                 scope=SCOPE).fact_key
    # The automatic rendering follows the predicate, so search finds the new wording.
    assert got.text == "user works at Globex"
    assert got.meta[PREDICATE_REKEY] == [
        {"at": t.timestamp(), "from": "hired_by", "to": "works_at"}]


def test_a_caller_supplied_text_is_kept(rec, store):
    moved = _write(store, "hired_by", "Globex", 20, text="Alice joined Globex in May")
    backfill_predicates(rec, "acme", aliases={"hired_by": "works_at"}, dry_run=False)
    assert store.get_claim(moved.id).text == "Alice joined Globex in May"


def test_the_receiving_slot_is_replayed_so_the_older_value_closes(rec, store):
    older = _write(store, "works_at", "Acme", 30)
    newer = _write(store, "hired_by", "Globex", 20)
    report = backfill_predicates(rec, "acme", aliases={"hired_by": "works_at"},
                                 dry_run=False)
    assert report.moved == 1 and report.retired == 1 and report.merged == 0
    # Both rows were rewritten: the mover, and the claim the replay closed.
    assert report.written == 2
    closed = store.get_claim(older.id)
    assert closed.invalidated_by == newer.id
    # Valid time ends when the newer claim was recorded, which is when the slot would
    # have changed hands had both been filed under one name. Belief is not withdrawn.
    assert closed.valid_to == newer.recorded_at
    assert closed.invalidated_at is None
    assert store.get_claim(newer.id).valid_to is None


def test_the_same_value_under_both_names_folds_into_the_earlier_claim(rec, store):
    first = _write(store, "works_at", "Acme", 30)
    again = _write(store, "hired_by", "Acme", 20)
    report = backfill_predicates(rec, "acme", aliases={"hired_by": "works_at"},
                                 dry_run=False)
    assert report.merged == 1 and report.retired == 0
    folded = store.get_claim(again.id)
    assert folded.invalidated_by == first.id
    # A duplicate was never false, so only transaction time closes.
    assert folded.valid_to is None and folded.invalidated_at is not None
    assert store.get_claim(first.id).observation_count == 2
    notes = store.get_claim(first.id).meta[PREDICATE_REKEY]
    assert notes[-1]["reason"] == "absorbed" and notes[-1]["claim"] == again.id


def test_a_multi_valued_predicate_folds_duplicates_but_closes_nothing(rec, store):
    _write(store, "likes", "tea", 30)
    other = _write(store, "fond_of", "coffee", 20)
    dup = _write(store, "fond_of", "tea", 10)
    report = backfill_predicates(rec, "acme", aliases={"fond_of": "likes"}, dry_run=False)
    assert report.moved == 2 and report.merged == 1 and report.retired == 0
    assert store.get_claim(other.id).is_live(utcnow())
    assert not store.get_claim(dup.id).is_live(utcnow())


def test_slots_nothing_moved_into_are_left_alone(rec, store):
    # Two live values in one functional slot, a defect this pass did not cause and was
    # not asked to repair. Only slots that received a claim are replayed.
    a = _write(store, "works_at", "Acme", 30)
    b = _write(store, "works_at", "Globex", 20)
    _write(store, "fond_of", "tea", 10)
    report = backfill_predicates(rec, "acme", aliases={"fond_of": "likes"}, dry_run=False)
    assert report.moved == 1 and report.retired == 0 and report.written == 1
    assert store.get_claim(a.id).valid_to is None
    assert store.get_claim(b.id).valid_to is None


def test_retired_claims_move_with_their_slot(rec, store):
    gone = _write(store, "hired_by", "Initech", 40)
    gone.valid_to = utcnow() - timedelta(minutes=35)
    store.put_claim(gone)
    live = _write(store, "hired_by", "Globex", 20)
    report = backfill_predicates(rec, "acme", aliases={"hired_by": "works_at"},
                                 dry_run=False)
    assert report.moved == 2 and report.retired == 0
    assert store.get_claim(gone.id).predicate == "works_at"
    assert store.get_claim(live.id).predicate == "works_at"
    # The retired claim's history follows it; the replay did not reopen or re-close it.
    assert store.get_claim(gone.id).valid_to == gone.valid_to


def test_without_a_mapping_the_pass_applies_what_the_registry_resolves(rec, store):
    moved = _write(store, "hired_by", "Globex", 20)
    untouched = _write(store, "likes", "tea", 10)
    rec.registry.learn_alias("works_at", "hired_by")
    report = backfill_predicates(rec, "acme", dry_run=False)
    assert report.moved == 1
    assert store.get_claim(moved.id).predicate == "works_at"
    assert store.get_claim(untouched.id).predicate == "likes"


def test_a_mapping_moves_only_the_predicates_it_names(rec, store):
    # `payroll_at` was aliased earlier and one claim was written under the raw name
    # since. A pass for a different merge must not count or move it.
    rec.registry.learn_alias("works_at", "payroll_at")
    leftover = _write(store, "payroll_at", "Initech", 30)
    wanted = _write(store, "hired_by", "Globex", 20)
    report = backfill_predicates(rec, "acme", aliases={"hired_by": "works_at"},
                                 dry_run=False)
    assert report.moved == 1
    assert store.get_claim(wanted.id).predicate == "works_at"
    assert store.get_claim(leftover.id).predicate == "payroll_at"


def test_a_mapping_may_name_several_merges_at_once(rec, store):
    a = _write(store, "hired_by", "Globex", 30)
    b = _write(store, "fond_of", "tea", 20)
    report = backfill_predicates(rec, "acme",
                                 aliases={"hired_by": "works_at", "fond_of": "likes"},
                                 dry_run=False)
    assert report.moved == 2
    assert {store.get_claim(a.id).predicate, store.get_claim(b.id).predicate} == {
        "works_at", "likes"}


def test_a_mapping_is_normalized_before_it_is_applied(rec, store):
    moved = _write(store, "hired_by", "Globex", 20)
    backfill_predicates(rec, "acme", aliases={"hired_by": "Works At"}, dry_run=False)
    assert store.get_claim(moved.id).predicate == "works_at"


def test_the_report_reads_as_one_line(rec, store):
    _write(store, "hired_by", "Globex", 20)
    assert repr(backfill_predicates(rec, "acme", aliases={"hired_by": "works_at"})) == (
        "<MergeReport scanned=1 moved=1 written=0 merged=0 retired=0 dry-run>")
    assert str(backfill_predicates(rec, "acme", aliases={"hired_by": "works_at"},
                                   dry_run=False)) == (
        "<MergeReport scanned=1 moved=1 written=1 merged=0 retired=0>")


def test_a_store_without_batch_is_written_one_row_at_a_time(rec, store):
    moved = _write(store, "hired_by", "Globex", 20)

    class NoBatch:
        def __getattr__(self, name):
            if name == "batch":
                raise AttributeError(name)
            return getattr(store, name)

    rec.store = NoBatch()
    report = backfill_predicates(rec, "acme", aliases={"hired_by": "works_at"},
                                 dry_run=False)
    assert report.written == 1
    assert store.get_claim(moved.id).predicate == "works_at"


# --- Memvara.merge_predicate ----------------------------------------------------


@pytest.fixture()
def mem():
    m = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice")
    yield m
    m.close()


def test_merge_predicate_dry_run_teaches_nothing(mem):
    mem.remember("user", "hired_by", "Globex")
    report = mem.merge_predicate("hired_by", "works_at")
    assert isinstance(report, MergeReport) and report.dry_run and report.moved == 1
    assert mem.registry.normalize("hired_by") == "hired_by"
    assert mem._persisted_specs("acme") == [] or all(       # noqa: SLF001
        "hired_by" not in s.aliases for s in mem._persisted_specs(  # noqa: SLF001
            mem.default_scope.tenant))
    assert [c.predicate for c in mem.history("user", "hired_by")] == ["hired_by"]


def test_merge_predicate_moves_the_past_and_redirects_the_future(mem):
    old = mem.remember("user", "works_at", "Acme").added[0]
    new = mem.remember("user", "hired_by", "Globex").added[0]
    report = mem.merge_predicate("hired_by", "works_at", dry_run=False)
    assert not report.dry_run and report.moved == 1 and report.retired == 1
    # The alias is learned and persisted, so the next process resolves it too.
    assert mem.registry.normalize("hired_by") == "works_at"
    tenant = mem.default_scope.tenant
    spec = next(s for s in mem._persisted_specs(tenant) if s.name == "works_at")  # noqa: SLF001
    assert "hired_by" in spec.aliases
    # One slot, one history: the old employer closed when the new one was recorded.
    rows = {c.id: c for c in mem.history("user", "works_at")}
    assert rows[old.id].valid_to == rows[new.id].recorded_at
    assert rows[new.id].valid_to is None
    # A write under the old name now lands in the merged slot and supersedes.
    receipt = mem.remember("user", "hired_by", "Initech")
    assert [c.id for c in receipt.closed] == [new.id]
    assert receipt.added[0].predicate == "works_at"


def test_merge_predicate_registers_an_unknown_canonical_as_multi_valued(mem):
    mem.remember("user", "known_bug", "a leak in the pool")
    mem.remember("user", "known_defect", "a leak in the pool")
    assert not mem.registry.known("known_defect")
    report = mem.merge_predicate("known_bug", "known_defect", dry_run=False)
    assert report.moved == 1 and report.merged == 1 and report.retired == 0
    assert mem.registry.known("known_defect")
    assert mem.registry.spec("known_defect").cardinality is Cardinality.MANY
    assert mem.registry.normalize("known_bug") == "known_defect"


def test_merge_predicate_refuses_one_name_spelled_twice(mem):
    with pytest.raises(ValueError, match="are the same predicate name"):
        mem.merge_predicate("Works At", "works_at")
    with pytest.raises(ValueError, match="must both name"):
        mem.merge_predicate("", "works_at")


def test_merge_predicate_repairs_an_alias_the_registry_already_knows(mem):
    # The write path learns an alias from a model and moves nothing. The claim filed
    # under the raw spelling before that is exactly what the repair is for.
    stranded = mem.remember("user", "hired_by", "Globex").added[0]
    mem.registry.learn_alias("works_at", "hired_by")
    assert mem.registry.normalize("hired_by") == "works_at"
    report = mem.merge_predicate("hired_by", "works_at", dry_run=False)
    assert report.moved == 1
    assert mem.get(stranded.id).predicate == "works_at"


def test_merge_predicate_scopes_to_the_named_tenant(mem):
    mem.remember("user", "hired_by", "Globex")
    report = mem.merge_predicate("hired_by", "works_at", tenant="other")
    assert report.scanned == 0 and report.moved == 0


def test_the_async_wrapper_runs_the_merge_off_the_loop(mem):
    import asyncio

    from memvara import AsyncMemvara

    mem.remember("user", "hired_by", "Globex")
    report = asyncio.run(AsyncMemvara(mem).merge_predicate("hired_by", "works_at"))
    assert report.dry_run and report.moved == 1


def test_the_scoped_views_merge_within_their_tenant(mem):
    import asyncio

    from memvara import AsyncMemvara

    mem.remember("user", "hired_by", "Globex")
    view = mem.scope(tenant="other")
    assert view.merge_predicate("hired_by", "works_at").moved == 0
    assert mem.scope(user="alice").merge_predicate("hired_by", "works_at").moved == 1
    aview = AsyncMemvara(mem).scope(tenant="other")
    assert asyncio.run(aview.merge_predicate("hired_by", "works_at")).moved == 0
