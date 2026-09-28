"""Erasure, and the evidence for it.

`erase()` reported success from a return code. That proves the code took the branch it
thought it took, which is the same statement the return value already made and cannot
disagree with it — and "told the caller it deleted the memory while the text is still
readable" is the exact failure the method was added to remove.

Three properties, and each test here defends one:

1. **The proof is a physical re-query.** It has to be able to contradict the delete, so
   every test that asserts `proven` also asserts against a store where the rows are
   still there.
2. **It fails closed.** A store that cannot be asked yields `proven=False` with a reason,
   never `proven=True`. Unproven and proven-gone are different answers and only one of
   them is an erasure certificate.
3. **The audit row is written before the delete, in the same transaction.** A delete
   whose record failed to write is the state that cannot be detected afterwards, so the
   ordering is the guarantee — and the record holds nothing the erased fact could be read
   back out of.
"""

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pytest

from memvara import Episode, ErasureIncomplete, ErasureProof, Memvara, NullLLM, utcnow
from memvara.embed import HashingEmbedder
from memvara.store import SQLiteStore
from memvara.store.sqlite import SCHEMA_VERSION

DAY = timedelta(days=1)


@pytest.fixture()
def mem():
    with Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), tenant="acme",
                 user="alice") as m:
        yield m


def stored(mem) -> str:
    """One claim whose every word is distinct from the scope it lives in.

    The subject is not `alice`, deliberately: the scope key *is* `acme/alice/*/*` and the
    audit row records it on purpose (where the claim lived is not what it said), so a
    fixture whose subject is also "Alice" would make the leak assertion below unable to
    tell the two apart.
    """
    receipt = mem.remember("Dara Wray", "lives_in", "Lisbon",
                           sources=[Episode(content="Dara told me she lives in Lisbon.")])
    return receipt.added[0].id


# --- the proof is a re-query, not a restatement -------------------------------


def test_a_claim_that_is_still_there_cannot_be_proved_gone(mem):
    """The assertion that makes every other one in this file mean something."""
    claim_id = stored(mem)
    proof = mem.prove_erased(claim_id)
    assert proof.proven is False
    assert proof.surviving, "a stored claim must show residue in at least one table"
    assert "survived" in proof.reason


def test_an_erased_claim_is_proved_gone_in_every_table_it_could_survive_in(mem):
    """Five tables, because those are the five a claim can survive in: the row, the text
    index over it, its vector, its provenance edges, and the typed links that name it.
    And one file, the claim's row of the vector file, read from the file itself."""
    claim_id = stored(mem)
    assert mem.erase(claim_id) is True
    proof = mem.prove_erased(claim_id)
    assert proof.proven is True
    assert proof.surviving == {}
    assert set(proof.residue) == {"claims", "claims_fts", "embeddings", "claim_sources",
                                  "claim_links", "vector_file"}


def test_an_id_nothing_ever_stored_is_proved_gone_rather_than_raising(mem):
    """"Nothing of it is on disk" is the honest answer, and it is the one a re-check of a
    completed erasure request wants months later."""
    proof = mem.prove_erased("cl_never_existed")
    assert proof.proven is True and proof.residue["claims"] == 0


def test_the_audit_row_is_not_counted_as_residue(mem):
    """It is supposed to survive. Counting it would make every proof fail."""
    claim_id = stored(mem)
    mem.erase(claim_id)
    assert "erasures" not in mem.prove_erased(claim_id).residue
    assert mem.store.erasure_record(claim_id) is not None


# --- failing closed -----------------------------------------------------------


class _Deaf(SQLiteStore):
    """A store that can erase and cannot be asked whether it did.

    Two shapes in one, and the second is the one a `getattr` guard misses:
    `residue` is absent, `erasure_record` is present and raises — which is
    `RemoteStore`.
    """

    residue = None  # type: ignore[assignment]

    def erasure_record(self, claim_id: str):
        raise NotImplementedError("no audit endpoint")


def test_a_store_that_cannot_be_asked_yields_unproven_rather_than_proven():
    with Memvara(store=_Deaf(":memory:"), llm=NullLLM(),
                 embedder=HashingEmbedder(dim=64), user="alice") as mem:
        claim_id = stored(mem)
        proof = mem.prove_erased(claim_id)
        assert proof.proven is False
        assert proof.residue == {}, "no counts is not the same as counts of zero"
        assert "does not implement residue()" in proof.reason


def test_erase_refuses_to_report_success_it_cannot_support():
    """`ErasureIncomplete` rather than `False`, and rather than `True`.

    `False` already means "there was nothing to erase", so folding an unproven erasure
    into it would tell a caller acting on a legal request that the memory was never
    there. `True` is the failure this whole path exists to remove. An exception is the
    only answer that cannot be mistaken for either.
    """
    with Memvara(store=_Deaf(":memory:"), llm=NullLLM(),
                 embedder=HashingEmbedder(dim=64), user="alice") as mem:
        claim_id = stored(mem)
        with pytest.raises(ErasureIncomplete) as exc:
            mem.erase(claim_id)
        assert exc.value.proof.claim_id == claim_id
        assert "half-erased" in str(exc.value)


def test_a_store_whose_residue_raises_fails_closed_too():
    """`RemoteStore.residue` is present on the object and raises. Caught, not guarded."""
    class _Raises(SQLiteStore):
        def residue(self, claim_id: str) -> dict[str, int]:
            raise NotImplementedError("no endpoint")

    with Memvara(store=_Raises(":memory:"), llm=NullLLM(),
                 embedder=HashingEmbedder(dim=64), user="alice") as mem:
        assert mem.prove_erased(stored(mem)).proven is False


def test_erase_still_returns_false_for_an_id_it_could_not_see(mem):
    """Unknown, or another tenant's. Neither erases anything, so neither is unproven."""
    assert mem.erase("cl_not_here") is False


def test_an_erasure_that_lost_the_race_returns_false_rather_than_refusing(mem):
    """Two callers erasing one claim: the second deletes nothing.

    There is nothing to prove and nothing to refuse — the claim is gone and this call is
    not why. Raising here would turn an idempotent operation into an error the moment two
    requests for the same erasure arrive together, which is how they actually arrive.
    """
    claim_id = stored(mem)
    real = mem.store.erase_claim

    def race(cid, *, sources=False):
        real(cid, sources=sources)          # somebody else got here first
        return real(cid, sources=sources)   # ...so this one erases nothing

    mem.store.erase_claim = race
    assert mem.erase(claim_id) is False
    assert mem.prove_erased(claim_id).proven is True


def test_a_store_with_no_audit_trail_can_still_prove_the_rows_are_gone():
    """`record=None` is not a failed proof. A store that keeps no trail cannot say
    anything about *what recorded* the erasure, and that is a different question from
    whether the rows survived."""
    class _NoTrail(SQLiteStore):
        def erasure_record(self, claim_id: str):
            raise NotImplementedError("no audit endpoint")

    with Memvara(store=_NoTrail(":memory:"), llm=NullLLM(),
                 embedder=HashingEmbedder(dim=64), user="alice") as mem:
        claim_id = stored(mem)
        assert mem.erase(claim_id) is True
        proof = mem.prove_erased(claim_id)
        assert proof.proven is True and proof.record is None


# --- the audit row ------------------------------------------------------------


def test_the_record_says_what_happened_and_not_what_was_erased(mem):
    """An audit trail the erased fact can be read out of is a copy of it.

    This is the assertion that keeps the table honest: it names the fields that must be
    there *and* checks that the claim's own words are in none of them.
    """
    claim_id = stored(mem)
    before = datetime.now(timezone.utc)
    mem.erase(claim_id, sources=True)
    record = mem.store.erasure_record(claim_id)

    assert record is not None
    assert record["claim_id"] == claim_id
    assert record["tenant"] == "acme"
    assert record["scope"].startswith("acme")
    assert record["erased_at"] >= before
    assert record["sources"] == 1
    assert record["counts"]["claims"] == 1

    flat = repr(record).lower()
    for secret in ("lisbon", "lives_in", "dara", "told me"):
        assert secret not in flat, f"the audit row leaks {secret!r}"


def test_an_id_that_erased_nothing_writes_no_record(mem):
    """A row saying an erasure happened when it did not is the one entry this table must
    never hold."""
    mem.store.erase_claim("cl_not_here")
    assert mem.store.erasure_record("cl_not_here") is None


def test_a_failed_audit_write_leaves_the_claim_in_place(mem):
    """The ordering *is* the guarantee, so it is asserted by breaking the audit write.

    The other order — delete, then record — lets a delete succeed and its record fail,
    which is precisely the state nothing downstream can detect. Here the INSERT raises,
    the exception leaves `erase_claim` before any delete runs, and the claim survives.
    """
    claim_id = stored(mem)
    mem.store._db.execute("DROP TABLE erasures")
    with pytest.raises(Exception):
        mem.store.erase_claim(claim_id)
    assert mem.get(claim_id) is not None, "the delete ran without a record of it"


def test_the_erasures_table_is_schema_seven():
    """A store upgraded from an older file gets an empty table, and an empty table means
    "nothing erased since the upgrade" — never "nothing was ever erased here".

    The version assertion is a tripwire rather than a fact worth pinning: it fails on any
    schema bump so that whoever makes one has to come and decide whether the sentence
    above still holds. Version 9 added three nullable claim columns and no table, so it
    does. Version 10 added six columns to `predicates` — the graph declaration — and no
    table either; it does not touch `erasures`, does not write one, and cannot invent a
    record of an erasure that happened before the upgrade, so the sentence still holds.
    Version 11 added one nullable column to `claims`, `object_kind`, and no table; it
    neither reads nor writes `erasures`, so the sentence holds there too. Version 12 added
    a `project` column to `claims` and to `episodes` and rehashed every `fact_key`; it adds
    no table, does not read or write `erasures`, and rewriting a slot key cannot conjure a
    record of an erasure that happened before the upgrade, so it holds again. Version 13
    added a table, `claim_links`, and backfills nothing into it; it neither reads nor
    writes `erasures`, so the sentence holds. Version 14 added two tables, `documents`
    and `document_chunks`, and backfills nothing into them; it neither reads nor writes
    `erasures`, so the sentence holds. Version 15 added two nullable columns to `claims`,
    `expires_at` and `expire_reason`, and no table. The migration neither reads nor writes
    `erasures`; the expiry sweep writes a row there for each claim it erases, through the
    same `erase_claim` that `erase()` uses, and only after the upgrade. So an empty table
    still means "nothing erased since the upgrade", and the sentence holds. Version 16
    changed the entity fold and added no table or column. Its migration is the key
    re-derivation `_migrate_to_v12` already runs on every upgrade, which neither reads nor
    writes `erasures`, so the sentence holds. Version 17 added one nullable column to
    `erasures`, `vector_slot`, and backfills nothing into it; the migration adds no row
    and removes none, so an empty table still means "nothing erased since the upgrade",
    and the sentence holds.
    """
    assert SCHEMA_VERSION == 17
    store = SQLiteStore(":memory:")
    try:
        assert store.erasure_record("anything") is None
    finally:
        store.close()


def test_a_scoped_view_proves_the_same_erasure(mem):
    """It takes no scope, so the scoped view has nothing narrower to pass. It is on
    `ScopedMemvara` anyway, because a caller holding one should not have to reach through
    `.memvara` for the one call in the erasure path that answers "is it really gone"."""
    claim_id = stored(mem)
    scoped = mem.scope(user="alice")
    assert scoped.prove_erased(claim_id).proven is False
    assert scoped.erase(claim_id) is True
    assert scoped.prove_erased(claim_id).proven is True


def test_the_proof_carries_the_record_so_a_caller_need_not_go_looking(mem):
    claim_id = stored(mem)
    mem.erase(claim_id)
    proof = mem.prove_erased(claim_id)
    assert proof.record is not None and proof.record["claim_id"] == claim_id
    assert "gone" in repr(proof)
    assert "UNPROVEN" in repr(ErasureProof(claim_id="x", proven=False, reason="why"))


# --- failing closed, in every shape a store can fail in -----------------------


@pytest.mark.parametrize("residue, why", [
    (lambda self, cid: {}, "counted nothing"),
    (lambda self, cid: None, "counted nothing"),
    (lambda self, cid: "all gone", "counted nothing"),
    (lambda self, cid: {"claims": -1}, "not a row count"),
    (lambda self, cid: {"claims": "0"}, "not a row count"),
])
def test_a_residue_that_did_not_really_count_is_not_a_proof(residue, why):
    """The empty dict is the case this method exists to refuse, and the one the first
    version of it got wrong.

    `ErasureProof` says residue is "empty when nothing could be counted, which is a
    different thing from every count being zero" — and the code then treated them the
    same, because `all(n == 0 for n in {})` is vacuously true. A store that counts
    nothing, counts the wrong type, or hands back something that is not a mapping must
    not receive a certificate for a claim that is still sitting there.
    """
    store = type("Odd", (SQLiteStore,), {"residue": residue})(":memory:")
    with Memvara(store=store, llm=NullLLM(), embedder=HashingEmbedder(dim=64),
                 user="alice") as mem:
        claim_id = stored(mem)
        proof = mem.prove_erased(claim_id)
        assert proof.proven is False, "the claim is still stored"
        assert why in proof.reason


def test_a_residue_that_raises_anything_at_all_fails_closed():
    """Not just `NotImplementedError`. A locked database raises `OperationalError` and a
    third-party store can raise whatever it likes; narrowing the catch to the one type we
    thought of is how a check that did not run gets reported as a check that passed."""
    class Locked(SQLiteStore):
        def residue(self, claim_id):
            raise sqlite3.OperationalError("database is locked")

    with Memvara(store=Locked(":memory:"), llm=NullLLM(),
                 embedder=HashingEmbedder(dim=64), user="alice") as mem:
        claim_id = stored(mem)
        assert mem.prove_erased(claim_id).proven is False
        with pytest.raises(ErasureIncomplete):
            mem.erase(claim_id)


def test_an_unreadable_audit_trail_does_not_make_the_rows_less_gone():
    """`record` is a lookup, not evidence. Failing to read it must not flip a proof
    either way — and it used to raise straight out of a method whose whole job is to
    answer."""
    with Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), user="alice") as mem:
        claim_id = stored(mem)
        mem.erase(claim_id)
        mem.store._db.execute("DROP TABLE erasures")
        proof = mem.prove_erased(claim_id)
        assert proof.proven is True and proof.record is None


def test_the_shipped_stores_residue_names_every_table_a_claim_can_survive_in():
    """The one thing the generic contract cannot check.

    `Store.residue` lets an implementation name its own tables, so a store that counts
    only `claims` and forgets the text index gets a passing certificate — nothing here
    can tell an honest key set from a short one. That boundary is documented on the
    protocol; what is pinned here is the *shipped* store's set, so it cannot silently
    shrink.
    """
    store = SQLiteStore(":memory:")
    try:
        assert set(store.residue("cl_anything")) == {
            "claims", "claims_fts", "embeddings", "claim_sources", "claim_links",
            "vector_file"}
    finally:
        store.close()


# --- the proof reads the vector file ----------------------------------------------------


def on_disk(tmp_path) -> Memvara:
    return Memvara(str(tmp_path / "m.db"), llm=NullLLM(), embedder=HashingEmbedder(dim=64),
                   tenant="acme", user="alice")


def test_the_proof_reads_the_erased_claims_row_in_the_vector_file(tmp_path):
    """An erasure blanks the claim's row in `<db>.vecs`, and a proof that counted only the
    database's rows could not tell whether it had: a store whose erasure left the vector
    on disk still received a certificate. The proof now reads that row from the file. Put
    the vector back in the row, as such an erasure left it, and the proof refuses."""
    mem = on_disk(tmp_path)
    try:
        claim = mem.remember("user", "likes", "coffee with oat milk").added[0]
        slot = mem.store._db.execute(
            "SELECT slot FROM embeddings WHERE claim_id=?", (claim.id,)).fetchone()[0]
        raw = mem.store.get_embedding(claim.id).tobytes()
        assert mem.prove_erased(claim.id).residue["vector_file"] == 1
        assert mem.erase(claim.id) is True
        assert mem.prove_erased(claim.id).residue["vector_file"] == 0
        with open(str(tmp_path / "m.db.vecs"), "r+b") as fh:
            fh.seek(64 + slot * len(raw))
            fh.write(raw)
        proof = mem.prove_erased(claim.id)
        assert proof.proven is False, "a vector left in the file was certified as erased"
        assert proof.residue["vector_file"] == 1 and "vector_file" in (proof.reason or "")
    finally:
        mem.close()


def test_a_row_another_vector_holds_now_is_not_counted_against_the_erased_claim(tmp_path):
    """A freed row goes to the next vector written, and from then on it holds that
    vector, not the erased one."""
    mem = on_disk(tmp_path)
    try:
        tea = mem.remember("user", "likes", "green tea").added[0]
        held = "SELECT slot FROM embeddings WHERE claim_id=?"
        slot = mem.store._db.execute(held, (tea.id,)).fetchone()[0]
        assert mem.erase(tea.id) is True
        coffee = mem.remember("user", "likes", "coffee").added[0]
        assert mem.store._db.execute(held, (coffee.id,)).fetchone()[0] == slot
        assert mem.prove_erased(tea.id).proven is True
    finally:
        mem.close()


def test_an_erasure_recorded_before_schema_17_is_proved_from_the_database(tmp_path):
    """Schema 17 records which row of the vector file an erased claim's vector held, and
    backfills nothing: no earlier version recorded it. The proof of an erasure from
    before the upgrade cannot read the file, so it answers from the database, as it did
    before, rather than refusing every erasure a store already holds."""
    mem = on_disk(tmp_path)
    claim = mem.remember("user", "likes", "coffee with oat milk").added[0]
    assert mem.erase(claim.id) is True
    mem.store._db.execute("ALTER TABLE erasures DROP COLUMN vector_slot")
    mem.store._db.execute("PRAGMA user_version = 16")
    mem.store._db.commit()
    mem.close()
    reopened = on_disk(tmp_path)
    try:
        db = reopened.store._db
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 17
        assert db.execute("SELECT vector_slot FROM erasures WHERE claim_id=?",
                          (claim.id,)).fetchone()[0] is None
        proof = reopened.prove_erased(claim.id)
        assert proof.proven is True and proof.residue["vector_file"] == 0
    finally:
        reopened.close()


def test_a_vector_file_that_is_gone_or_cut_short_holds_nothing_for_the_proof(tmp_path):
    """The proof opens the vector file by its path, apart from this store's own handle on
    it. A file that is gone, or too short to say how wide its rows are, has no row that
    could hold the erased vector, so the proof counts 0 for it rather than raising."""
    mem = on_disk(tmp_path)
    vectors = mem.store._vec
    real = vectors.path
    try:
        claim = mem.remember("user", "likes", "coffee with oat milk").added[0]
        assert mem.erase(claim.id) is True
        vectors.path = str(tmp_path / "gone.vecs")
        proof = mem.prove_erased(claim.id)
        assert proof.proven is True and proof.residue["vector_file"] == 0
        (tmp_path / "short.vecs").write_bytes(b"MVEC")
        vectors.path = str(tmp_path / "short.vecs")
        proof = mem.prove_erased(claim.id)
        assert proof.proven is True and proof.residue["vector_file"] == 0
    finally:
        vectors.path = real
        mem.close()


# --- the audit row cannot outlive a failed delete -----------------------------


def test_a_delete_that_fails_after_the_audit_row_leaves_no_audit_row(tmp_path):
    """The converse of audit-before-delete, and the half that was missing.

    Damage the text index the way a truncated restore does, so the delete raises *after*
    the record is written. Before the fix the row sat pending in the open transaction and
    the next commit from anywhere — an ordinary `remember()`, or the standard FTS5 repair
    — made it durable: a trail asserting an erasure that never happened, about a claim
    still readable.
    """
    path = tmp_path / "m.db"
    mem = Memvara(str(path), llm=NullLLM(), embedder=HashingEmbedder(dim=64), user="d")
    try:
        claim_id = stored(mem)
        mem.store._db.execute("DELETE FROM claims_fts_data WHERE id > 1")
        with pytest.raises(sqlite3.Error):
            mem.erase(claim_id)
        # The repair a real operator runs next, which is what used to commit the lie.
        mem.store._db.execute("INSERT INTO claims_fts(claims_fts) VALUES('rebuild')")
        mem.store._db.commit()

        assert mem.get(claim_id) is not None, "nothing was erased"
        assert mem.store.erasure_record(claim_id) is None, (
            "an audit row survived a delete that never happened"
        )
    finally:
        mem.close()


def test_an_erasure_inside_an_abandoned_batch_still_rolls_back(tmp_path):
    """Why the compensation is a DELETE and not a SAVEPOINT.

    `RELEASE` commits a savepoint's work into the enclosing transaction, so wrapping the
    erase in one broke `batch()` — an erasure inside an abandoned batch stopped rolling
    back. Undoing the single row this method added leaves every transaction boundary
    where the caller put it.
    """
    path = tmp_path / "m.db"
    mem = Memvara(str(path), llm=NullLLM(), embedder=HashingEmbedder(dim=64), user="d")
    try:
        claim_id = stored(mem)
        with pytest.raises(RuntimeError):
            with mem.store.batch():
                mem.store.erase_claim(claim_id)
                raise RuntimeError("abandoned")
        assert mem.get(claim_id) is not None
        assert mem.store.erasure_record(claim_id) is None
    finally:
        mem.close()


def test_two_erasures_of_one_id_are_two_records(tmp_path):
    """A claim can be erased, restored from a backup, and erased again. Keyed on the id
    alone the second silently overwrote the first, so an append-only trail lost exactly
    the entry somebody would go looking for."""
    path = tmp_path / "m.db"
    mem = Memvara(str(path), llm=NullLLM(), embedder=HashingEmbedder(dim=64), user="d")
    try:
        claim = mem.remember("Dara Wray", "lives_in", "Lisbon").added[0]
        mem.erase(claim.id)
        first = mem.store.erasure_record(claim.id)
        mem.store.put_claim(claim)                    # restored from a backup
        mem.erase(claim.id)

        rows = mem.store._db.execute(
            "SELECT count(*) FROM erasures WHERE claim_id=?", (claim.id,)).fetchone()[0]
        assert rows == 2, "the first erasure was overwritten"
        assert mem.store.erasure_record(claim.id)["erased_at"] >= first["erased_at"], (
            "the lookup must return the most recent of the two"
        )
    finally:
        mem.close()



def test_the_erasure_record_names_the_project_the_claim_was_erased_from(tmp_path):
    """The audit row is the permanent record, so what it omits is unrecoverable.

    `erase_claim` deletes the row it is describing, and the `erasures` table deliberately
    holds no text, subject, predicate or object. The scope key is most of what is left, so
    a scope that reports every erasure as unscoped is not a cosmetic gap: after the delete
    there is nothing to go back to and read the project from.
    """
    from memvara import Memvara, NullLLM
    from memvara.embed import HashingEmbedder

    mem = Memvara(str(tmp_path / "m.db"), embedder=HashingEmbedder(dim=32), llm=NullLLM(),
                  user="alice", project="gh/o/x")
    try:
        mem.remember("postgresql", "version", "17")
        claim = mem.get_all()[0]
        assert claim.scope.project == "gh/o/x"
        mem.store.erase_claim(claim.id)
        recorded = [r["scope"] for r in mem.store._db.execute("SELECT scope FROM erasures")]
        assert recorded == [claim.scope.key()]
        assert "gh%2Fo%2Fx" in recorded[0]
    finally:
        mem.close()


# --- an erasure beside another write ---------------------------------------------------
#
# Two handles on one file stand for two processes. One handle opens a batch, which holds
# the database's write lock. The write under test starts through the other handle, on a
# thread, and has to wait for that lock. While it waits, the first handle erases or changes
# the claim, and then commits. The write must act on the store as the first handle left
# it: a claim erased while it waited stays erased, with its one erasure record.


@pytest.fixture()
def two(tmp_path):
    """Two handles on one store file."""
    path = str(tmp_path / "s.db")
    first = Memvara(path, llm=NullLLM(), embedder=HashingEmbedder(dim=64), tenant="acme",
                    user="alice")
    second = Memvara(path, llm=NullLLM(), embedder=HashingEmbedder(dim=64), tenant="acme",
                     user="alice")
    yield first, second
    first.close()
    second.close()


def behind(holder: Memvara, change: Callable[[], Any],
           call: Callable[[], Any]) -> tuple[Any, Any]:
    """Start `call` on a thread while `holder` holds the write lock, then make `change`
    through `holder` and commit it. Returns what `change` and `call` returned, and raises
    what `call` raised. `call` must still be waiting for the lock when `change` runs, and
    that is checked."""
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = call()
        except BaseException as exc:  # noqa: BLE001 - raised again below
            outcome["error"] = exc

    with holder.store.batch():
        worker = threading.Thread(target=run)
        worker.start()
        worker.join(0.2)
        assert worker.is_alive(), f"the write did not wait for the lock: {outcome}"
        changed = change()
    worker.join(10)
    assert not worker.is_alive(), "the write never finished"
    if "error" in outcome:
        raise outcome["error"]
    return changed, outcome["value"]


def test_a_delete_does_not_bring_back_a_claim_erased_while_it_waited(two):
    """The delete used to read the claim before it had the lock and write the whole row
    back after the erasure had committed, so the erased text was back on disk beside its
    own erasure record."""
    holder, deleter = two
    claim_id = stored(holder)
    erased, deleted = behind(holder, lambda: holder.erase(claim_id),
                             lambda: deleter.delete(claim_id))
    assert (erased, deleted) == (True, False)
    assert deleter.store.get_claim(claim_id) is None
    assert deleter.prove_erased(claim_id).proven


def test_a_second_erasure_of_one_claim_finds_nothing_and_records_nothing(two):
    holder, eraser = two
    claim_id = stored(holder)
    first, second = behind(holder, lambda: holder.erase(claim_id),
                           lambda: eraser.erase(claim_id))
    assert (first, second) == (True, False)
    rows = eraser.store._db.execute(
        "SELECT count(*) FROM erasures WHERE claim_id=?", (claim_id,)).fetchone()[0]
    assert rows == 1, "the second erasure recorded an erasure that erased nothing"


def test_a_link_to_a_claim_erased_while_it_waited_is_refused(two):
    holder, linker = two
    doomed = stored(holder)
    other = holder.remember("Dara Wray", "works_at", "Acme").added[0].id
    with pytest.raises(KeyError):
        behind(holder, lambda: holder.erase(doomed),
               lambda: linker.link(other, doomed, "extends"))
    assert linker.prove_erased(doomed).proven, "a link still names the erased claim"


def test_the_expiry_sweep_leaves_a_claim_whose_expiry_moved_while_it_waited(two):
    holder, sweeper = two
    code = holder.remember("user", "door_code", "4411", expires_at=utcnow() + DAY).added[0]
    _, swept = behind(
        holder,
        lambda: holder.remember("user", "door_code", "4411", expires_at=utcnow() + 10 * DAY),
        lambda: sweeper.erase_expired(now=utcnow() + 2 * DAY))
    assert swept == []
    assert sweeper.store.get_claim(code.id).expires_at > utcnow() + 2 * DAY


# --- the store's own erasure methods beside another write -------------------------------
#
# The same two handles, calling the store directly, as `purge()`, the documents service and
# any caller of the `Store` protocol do. Each erasure method reads what it is about to
# erase, and that read must happen under the write lock too.


def test_a_second_store_erasure_of_one_claim_finds_nothing_and_records_nothing(two):
    holder, eraser = two
    claim_id = stored(holder)
    first, second = behind(holder, lambda: holder.store.erase_claim(claim_id),
                           lambda: eraser.store.erase_claim(claim_id))
    assert (first["claims"], second["claims"]) == (1, 0)
    rows = eraser.store._db.execute(
        "SELECT count(*) FROM erasures WHERE claim_id=?", (claim_id,)).fetchone()[0]
    assert rows == 1, "the second erasure recorded an erasure that erased nothing"


def test_a_turn_a_claim_began_citing_while_its_erasure_waited_is_kept(two):
    """`erase_episode` erases only a turn nothing cites, and must decide that as the store
    stands once it holds the lock, or it erases a turn a claim has just come to cite."""
    holder, eraser = two
    turn = Episode(content="I like green tea.", role="user", scope=holder.default_scope)
    holder.store.add_episode(turn)
    _, erased = behind(
        holder, lambda: holder.remember("user", "likes", "green tea", sources=[turn.id]),
        lambda: eraser.store.erase_episode(turn.id))
    assert erased is False
    assert eraser.store.get_episode(turn.id) is not None


def test_a_purge_lists_the_vector_of_a_claim_written_while_it_waited(two):
    """`purge` deletes rows by scope, and blanks and counts only the vectors it listed
    first. The list must be taken under the lock, or a claim written while the purge
    waited loses its row while its vector is neither blanked nor counted."""
    holder, purger = two
    holder.remember("user", "likes", "tea")
    coffee, counts = behind(
        holder, lambda: holder.remember("user", "likes", "coffee with oat milk").added[0],
        lambda: purger.purge())
    assert (counts["claims"], counts["embeddings"]) == (2, 2)
    assert purger.store.get_claim(coffee.id) is None
