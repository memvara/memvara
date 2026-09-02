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
from datetime import datetime, timezone

import pytest

from memvara import ErasureIncomplete, ErasureProof, Memvara, NullLLM
from memvara.embed import HashingEmbedder
from memvara.store import SQLiteStore
from memvara.store.sqlite import SCHEMA_VERSION


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
                           sources=["Dara told me she lives in Lisbon."])
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
    """Four tables, because those are the four a claim's content can live in: the row,
    the text index over it, its vector, and its provenance edges."""
    claim_id = stored(mem)
    assert mem.erase(claim_id) is True
    proof = mem.prove_erased(claim_id)
    assert proof.proven is True
    assert proof.surviving == {}
    assert set(proof.residue) == {"claims", "claims_fts", "embeddings", "claim_sources"}


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
    does.
    """
    assert SCHEMA_VERSION == 9
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
            "claims", "claims_fts", "embeddings", "claim_sources"}
    finally:
        store.close()


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

