"""Erasure on expiry: the one case in which the engine deletes a claim on its own.

A caller may write a fact with `expires_at`. Once that instant passes, `erase_expired`
erases the claim through the same path `erase()` takes and proves it against the disk.
The tests here defend the edges of that exception to invariant 3 in `docs/INTERNALS.md`:

1. **Only a claim with an explicit `expires_at` is erased.** An ended claim, a superseded
   one and a retired one are kept whatever their `valid_to` says, because `valid_to` also
   closes history and erasing on it would erase history.
2. **Only after the instant passes, and always with a proof.** Every erasure returns an
   `ErasureProof` that re-queried the disk, and the store's erasure record is written.
3. **The switch stops the engine, not the record.** With `expiry_erasure` off the date is
   still stored and nothing is erased on its own.
"""
from __future__ import annotations

import asyncio
import io
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from memvara import (AsyncMemvara, ErasedClaim, ErasureIncomplete, HashingEmbedder,
                     Memvara, NullLLM)
from memvara.server import MemvaraMCPServer
from memvara.server.mcp import EXPIRY_INTERVAL, ExpirySweeper
from memvara.server.tools import TOOLS, without_expiry
from memvara.store import SQLiteStore
from memvara.store.sqlite import SCHEMA_VERSION
from memvara.types import utcnow

from test_server import call

DAY = timedelta(days=1)


def mem(path: str | None = None, **kw: Any) -> Memvara:
    return Memvara(path, llm=NullLLM(), embedder=HashingEmbedder(dim=64), user="alice",
                   **kw)


@pytest.fixture()
def m():
    with mem() as store:
        yield store


# --- writing an expiry ----------------------------------------------------------

def test_remember_stores_the_expiry_and_its_reason_and_they_survive_a_reopen(tmp_path):
    path = str(tmp_path / "m.db")
    when = (utcnow() + DAY).replace(microsecond=0)
    with mem(path) as first:
        claim = first.remember("user", "door_code", "4411", expires_at=when,
                               expire_reason="rental ends Friday").added[0]
        assert (claim.expires_at, claim.expire_reason) == (when, "rental ends Friday")
    with mem(path) as second:
        again = second.get(claim.id)
        assert again is not None
        assert (again.expires_at, again.expire_reason) == (when, "rental ends Friday")


def test_a_claim_written_without_an_expiry_has_none():
    with mem() as m:
        claim = m.remember("user", "lives_in", "Lisbon").added[0]
        assert claim.expires_at is None and claim.expire_reason is None
        assert m.get(claim.id).expires_at is None


def test_a_naive_expiry_is_read_as_utc(m):
    naive = (datetime.now(timezone.utc) + DAY).replace(tzinfo=None, microsecond=0)
    claim = m.remember("user", "door_code", "4411", expires_at=naive).added[0]
    assert claim.expires_at == naive.replace(tzinfo=timezone.utc)


@pytest.mark.parametrize("kw, message", [
    ({"expires_at": None, "expire_reason": "why"}, "no expires_at= was given"),
    ({"expires_at": "past"}, "is not in the future"),
    ({"expires_at": "future", "expire_reason": "  "}, "blank"),
    ({"expires_at": "future", "expire_reason": "x" * 501}, "500"),
])
def test_an_expiry_that_cannot_be_honoured_is_refused_and_nothing_is_written(m, kw, message):
    """A past instant is refused because the next sweep would erase the claim: storing it
    writes a fact only to delete it."""
    instants = {"past": utcnow() - timedelta(seconds=1), "future": utcnow() + DAY}
    if isinstance(kw.get("expires_at"), str):
        kw = {**kw, "expires_at": instants[kw["expires_at"]]}
    with pytest.raises(ValueError, match=message):
        m.remember("user", "door_code", "4411", **kw)
    assert m.count() == 0


def test_repeating_a_fact_with_an_expiry_puts_the_expiry_on_the_claim_on_record(m):
    """A repeat is a reinforcement, not a new row. Without this the expiry would be
    dropped with the candidate and a fact the caller asked to have erased would stay."""
    first = m.remember("user", "door_code", "4411").added[0]
    when = utcnow() + DAY
    receipt = m.remember("user", "door_code", "4411", expires_at=when,
                         expire_reason="temporary")
    assert [c.id for c in receipt.reinforced] == [first.id]
    assert m.get(first.id).expires_at == when
    m.remember("user", "door_code", "4411")
    assert m.get(first.id).expires_at == when, "a repeat naming no expiry keeps it"
    assert m.get(first.id).expire_reason == "temporary"


def test_a_repeat_with_an_expiry_and_an_earlier_start_still_puts_the_expiry_on_record(m):
    """A repeat that says the fact began earlier than the claim on record is normally
    stored as a claim of its own for that earlier period. One that names an expiry stays
    a repeat, so the expiry reaches the claim on record. Otherwise the claim for the
    earlier period would be erased and the fact the caller asked to have erased would
    stay."""
    now = utcnow()
    first = m.remember("user", "door_code", "4411", valid_from=now - DAY).added[0]
    when = now + DAY
    receipt = m.remember("user", "door_code", "4411", valid_from=now - 10 * DAY,
                         expires_at=when)
    assert [c.id for c in receipt.reinforced] == [first.id] and receipt.added == []
    assert m.get(first.id).expires_at == when


# --- the sweep ----------------------------------------------------------------

def test_nothing_is_erased_before_the_instant_and_everything_due_is_erased_after(m):
    code = m.remember("user", "door_code", "4411", expires_at=utcnow() + DAY,
                      expire_reason="rental").added[0]
    kept = m.remember("user", "lives_in", "Lisbon").added[0]
    assert m.erase_expired() == []
    assert m.erase_expired(now=code.expires_at - timedelta(seconds=1)) == []

    erased = m.erase_expired(now=code.expires_at)
    assert [e.claim_id for e in erased] == [code.id]
    report = erased[0]
    assert isinstance(report, ErasedClaim)
    assert (report.expires_at, report.expire_reason) == (code.expires_at, "rental")
    assert report.scope == code.scope
    assert report.proof.proven and report.proof.surviving == {}
    assert report.proof.record is not None and report.proof.record["claim_id"] == code.id
    assert m.get(code.id) is None
    assert m.history("user", "door_code") == [], "erased, not ended: history has a gap"
    assert m.get(kept.id) is not None
    assert m.erase_expired(now=code.expires_at + DAY) == [], "erasing twice erases once"


def test_the_report_carries_no_copy_of_the_erased_fact(m):
    m.remember("user", "door_code", "4411", expires_at=utcnow() + DAY)
    report = m.erase_expired(now=utcnow() + 2 * DAY)[0]
    assert "4411" not in repr(report)
    assert not {"subject", "predicate", "object", "text"} & set(ErasedClaim.__slots__)


def test_ended_superseded_and_retired_claims_are_kept_however_old_they_are(m):
    """Invariant 3: `valid_to` closes history and is never a reason to erase."""
    long_ago = utcnow() - timedelta(days=400)
    finished = m.remember("user", "worked_at", "Acme", valid_from=long_ago,
                          valid_to=long_ago + DAY).added[0]
    berlin = m.remember("user", "lives_in", "Berlin", valid_from=long_ago).added[0]
    m.remember("user", "lives_in", "Lisbon")
    wrong = m.remember("user", "speaks", "Klingon").added[0]
    m.delete(wrong.id)
    assert m.erase_expired(now=utcnow() + timedelta(days=36500)) == []
    for claim_id in (finished.id, berlin.id, wrong.id):
        assert m.store.get_claim(claim_id) is not None


def test_an_ended_claim_that_carries_an_expiry_is_still_erased_when_it_passes(m):
    """Ending a claim does not cancel the erasure the caller asked for."""
    berlin = m.remember("user", "lives_in", "Berlin", expires_at=utcnow() + DAY).added[0]
    lisbon = m.remember("user", "lives_in", "Lisbon").added[0]
    assert m.get(berlin.id).state == "ended"
    assert [e.claim_id for e in m.erase_expired(now=utcnow() + 2 * DAY)] == [berlin.id]
    assert m.get(lisbon.id) is not None


def test_the_sweep_reaches_every_tenant_in_the_store(tmp_path):
    """The expiry is a rule written on the claim, so it holds however the store is opened."""
    with mem() as m:
        other = m.remember("user", "door_code", "1", tenant="beta", user="bob",
                           expires_at=utcnow() + DAY).added[0]
        mine = m.remember("user", "door_code", "2", expires_at=utcnow() + DAY).added[0]
        erased = {e.claim_id for e in m.erase_expired(now=utcnow() + 2 * DAY)}
        assert erased == {other.id, mine.id}


def test_the_source_turns_are_kept_as_erase_keeps_them_by_default(m):
    receipt = m.add("my door code is 4411")
    turn = receipt.episode_ids[0]
    code = m.remember("user", "door_code", "4411", sources=[turn],
                      expires_at=utcnow() + DAY).added[0]
    m.erase_expired(now=utcnow() + 2 * DAY)
    assert m.get(code.id) is None
    assert m.store.get_episode(turn) is not None


def test_a_claim_whose_expiry_moved_after_the_listing_is_left_alone(m, monkeypatch):
    """Re-read just before the delete, so a later write that extended the expiry wins."""
    code = m.remember("user", "door_code", "4411", expires_at=utcnow() + DAY).added[0]
    stale = m.store.get_claim(code.id)
    gone = m.store.get_claim(code.id)
    gone.id = "cl_already_erased"
    m.remember("user", "door_code", "4411", expires_at=utcnow() + 10 * DAY)
    monkeypatch.setattr(m.store, "expired_claims", lambda now: [stale, gone])
    assert m.erase_expired(now=utcnow() + 2 * DAY) == []
    assert m.get(code.id) is not None


def test_a_claim_erased_by_someone_else_in_between_is_skipped(m, monkeypatch):
    code = m.remember("user", "door_code", "4411", expires_at=utcnow() + DAY).added[0]
    monkeypatch.setattr(m.store, "erase_claim",
                        lambda claim_id, sources=False: {"claims": 0})
    assert m.erase_expired(now=utcnow() + 2 * DAY) == []
    assert code.id


def test_a_failed_proof_raises_rather_than_reporting_an_erasure(m, monkeypatch):
    m.remember("user", "door_code", "4411", expires_at=utcnow() + DAY)
    monkeypatch.setattr(m.store, "residue", lambda claim_id: {"claims": 1})
    with pytest.raises(ErasureIncomplete):
        m.erase_expired(now=utcnow() + 2 * DAY)


class _NoExpiry:
    """A store proxy without `expired_claims`, as a third-party store may be."""

    def __init__(self, inner: SQLiteStore) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name == "expired_claims":
            raise AttributeError(name)
        return getattr(self._inner, name)


def test_a_store_that_cannot_list_expired_claims_is_skipped_at_open_and_refused_by_name():
    store = _NoExpiry(SQLiteStore(":memory:"))
    with Memvara(store=store, llm=NullLLM(), embedder=HashingEmbedder(dim=64)) as m:
        with pytest.raises(NotImplementedError, match="_NoExpiry does not implement"):
            m.erase_expired()


class _StubbedExpiry(_NoExpiry):
    """A store with the listing as a stub that raises, as `RemoteStore` has it."""

    def expired_claims(self, now: datetime) -> list[Any]:
        raise NotImplementedError("the deployment sweeps")


def test_a_store_with_a_stubbed_listing_opens_and_refuses_a_sweep_asked_for_by_name():
    store = _StubbedExpiry(SQLiteStore(":memory:"))
    with Memvara(store=store, llm=NullLLM(), embedder=HashingEmbedder(dim=64)) as m:
        with pytest.raises(NotImplementedError, match="the deployment sweeps"):
            m.erase_expired()


def test_the_remote_store_says_the_deployment_sweeps():
    from memvara.store.remote import RemoteStore

    with pytest.raises(NotImplementedError, match="erases expired claims itself"):
        RemoteStore(api_key="k", base_url="https://example.test").expired_claims(utcnow())


def test_async_erase_expired_is_the_same_sweep(m):
    code = m.remember("user", "door_code", "4411", expires_at=utcnow() + DAY).added[0]
    erased = asyncio.run(AsyncMemvara(m).erase_expired(utcnow() + 2 * DAY))
    assert [e.claim_id for e in erased] == [code.id]


# --- at open, and the switch ------------------------------------------------------

def _due(path: str) -> str:
    """Write a claim whose expiry has already passed, below `remember`, which refuses
    one. This is the state a store is in when it was closed across the instant."""
    with mem(path) as m:
        claim = m.remember("user", "door_code", "4411", expires_at=utcnow() + DAY).added[0]
        claim.expires_at = utcnow() - DAY
        m.store.put_claim(claim)
        return claim.id


def test_opening_a_store_erases_what_expired_while_it_was_closed(tmp_path):
    path = str(tmp_path / "m.db")
    claim_id = _due(path)
    with mem(path) as reopened:
        assert reopened.store.get_claim(claim_id) is None
        assert reopened.store.erasure_record(claim_id) is not None


def test_with_the_switch_off_the_expiry_is_kept_and_nothing_is_erased_at_open(tmp_path):
    path = str(tmp_path / "m.db")
    claim_id = _due(path)
    with mem(path, expiry_erasure=False) as reopened:
        assert reopened.expiry_erasure is False
        assert reopened.store.get_claim(claim_id).expires_at is not None
        # Called by name, it still erases: then the caller asked.
        assert [e.claim_id for e in reopened.erase_expired()] == [claim_id]


# --- the schema -----------------------------------------------------------------

def test_a_version_14_file_gains_the_expiry_columns_and_keeps_its_claims(tmp_path):
    """A real upgrade: a file stamped 14 without the columns, opened by this build."""
    path = str(tmp_path / "v14.db")
    with mem(path) as m:
        claim = m.remember("user", "lives_in", "Lisbon").added[0]
    raw = sqlite3.connect(path)
    # Rebuilt from its own CREATE statement without the two columns, rather than with
    # `DROP COLUMN`, which SQLite refuses when a comment sits before the last column.
    sql = raw.execute("SELECT sql FROM sqlite_master WHERE name = 'claims'").fetchone()[0]
    v14 = re.sub(r",\s*-- Version 15\..*?expire_reason\s+TEXT", "", sql, flags=re.S)
    assert "expires_at" not in v14 and v14 != sql
    raw.execute("ALTER TABLE claims RENAME TO claims_v15")
    raw.execute(v14)
    columns = ", ".join(r[1] for r in raw.execute("PRAGMA table_info(claims)"))
    raw.execute(f"INSERT INTO claims (rowid, {columns}) "
                f"SELECT rowid, {columns} FROM claims_v15")
    raw.execute("DROP TABLE claims_v15")
    raw.execute("PRAGMA user_version = 14")
    raw.commit()
    raw.close()

    with mem(path) as upgraded:
        db = upgraded.store._db
        assert int(db.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION == 16
        columns = {r["name"] for r in db.execute("PRAGMA table_info(claims)")}
        assert {"expires_at", "expire_reason"} <= columns
        indexes = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert "cl_expiry" in indexes
        old = upgraded.get(claim.id)
        assert old.object == "Lisbon" and old.expires_at is None
        upgraded.store._migrate_to_v15()                 # idempotent
        new = upgraded.remember("user", "door_code", "1",
                                expires_at=utcnow() + DAY).added[0]
        assert upgraded.get(new.id).expires_at is not None


def test_the_sweep_reads_the_partial_index():
    store = SQLiteStore(":memory:")
    plan = " ".join(str(tuple(r)) for r in store._db.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM claims WHERE expires_at IS NOT NULL "
        "AND expires_at <= ? ORDER BY expires_at, id", (0.0,)))
    assert "cl_expiry" in plan
    store.close()


# --- the MCP tool ---------------------------------------------------------------

def server(**kw: Any) -> MemvaraMCPServer:
    return MemvaraMCPServer(mem(), user="alice", **kw)


def _remember(srv: MemvaraMCPServer, **args: Any) -> tuple[str, bool]:
    return call(srv, "memory_remember", {"predicate": "door_code", "object": "4411", **args})


def test_the_tool_stores_the_expiry_and_says_the_fact_will_be_erased():
    srv = server()
    when = (utcnow() + DAY).replace(microsecond=0)
    body, is_error = _remember(srv, expires_at=when.isoformat(),
                               expire_reason="rental ends")
    assert not is_error, body
    assert "this fact will be erased at" in body and "cannot be brought back" in body
    assert "stops being returned at that instant" in body
    claim = srv._memory.get_all()[0]
    assert (claim.expires_at, claim.expire_reason) == (when, "rental ends")


@pytest.mark.parametrize("args, message", [
    ({"expire_reason": "why"}, "no expires_at was sent"),
    ({"expires_at": "2001-01-01T00:00:00Z"}, "is not in the future"),
    ({"expires_at": "next friday"}, "memory_remember.expires_at must be an ISO-8601"),
    ({"expires_at": "2999-01-01", "expire_reason": " "}, "expire_reason"),
])
def test_the_tool_refuses_an_expiry_it_cannot_honour_and_writes_nothing(args, message):
    srv = server()
    body, is_error = _remember(srv, **args)
    assert is_error and message in body, body
    assert srv._memory.count() == 0


def test_a_write_without_an_expiry_says_nothing_about_erasure():
    body, is_error = _remember(server())
    assert not is_error and "erased" not in body


def test_the_expiry_arguments_say_what_erasing_means_and_how_it_differs_from_ending():
    props = next(t for t in TOOLS if t.name == "memory_remember").properties
    description = props["expires_at"]["description"]
    for phrase in ("ERASED", "not ended and not retired", "true_until",
                   "stops returning the fact as soon as that instant passes",
                   "Refused when the instant is not in the future"):
        assert phrase in description
    assert props["expire_reason"]["maxLength"] == 500


def test_with_the_switch_off_the_arguments_stay_and_say_nothing_will_be_erased():
    srv = server(features_off={"expiry_erasure"})
    tools = {t["name"]: t for t in srv.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]}
    props = tools["memory_remember"]["inputSchema"]["properties"]
    assert "switched off" in props["expires_at"]["description"]
    assert "switched off" in props["expire_reason"]["description"]
    body, is_error = _remember(srv, expires_at=(utcnow() + DAY).isoformat())
    assert not is_error
    assert "expiry erasure is switched off on this server" in body
    assert srv._sweeper is None


def test_without_expiry_leaves_every_other_tool_unchanged():
    changed = [t.name for t, u in zip(TOOLS, without_expiry(TOOLS)) if t is not u]
    assert changed == ["memory_remember"]


def test_the_tool_sends_expiry_only_when_it_was_given():
    """A hosted deployment from before expiry refuses a field it does not know, so a
    write that sets none must send none."""
    seen: list[dict[str, Any]] = []

    class Spy:
        def __getattr__(self, name: str) -> Any:
            return getattr(inner, name)

        def remember(self, *args: Any, **kw: Any) -> Any:
            seen.append(kw)
            return inner.remember(*args, **kw)

    srv = server()
    inner = srv._ctx.memory
    object.__setattr__(srv._ctx, "memory", Spy())
    _remember(srv)
    _remember(srv, predicate="gate_code", expires_at=(utcnow() + DAY).isoformat())
    assert "expires_at" not in seen[0] and "expire_reason" not in seen[0]
    assert seen[1]["expires_at"] is not None


# --- the hourly sweep -------------------------------------------------------------

def test_the_server_sweeps_hourly_by_default_and_only_over_a_local_store():
    assert EXPIRY_INTERVAL == 3600
    assert server()._sweeper.interval == EXPIRY_INTERVAL


def test_one_sweep_erases_what_is_due_and_reports_the_count():
    with mem() as m:
        m.remember("user", "door_code", "4411", expires_at=utcnow() + DAY)
        sweeper = ExpirySweeper(m)
        assert sweeper.sweep() == 0
        claim = m.get_all()[0]
        claim.expires_at = utcnow() - DAY
        m.store.put_claim(claim)
        assert sweeper.sweep() == 1


def test_a_failed_sweep_warns_and_does_not_stop_the_next_one():
    class Broken:
        def erase_expired(self) -> list[ErasedClaim]:
            raise RuntimeError("disk full")

    sweeper = ExpirySweeper(Broken(), interval=60)  # type: ignore[arg-type]
    with pytest.warns(RuntimeWarning, match="expiry sweep failed .*disk full.* 60 seconds"):
        assert sweeper.sweep() == 0


def test_the_thread_sweeps_on_its_interval_until_stopped():
    """The wall clock is the thing under test here, so the interval is short and the
    test waits on an event rather than sleeping."""
    swept = threading.Event()

    class Counting:
        calls = 0

        def erase_expired(self) -> list[ErasedClaim]:
            Counting.calls += 1
            swept.set()
            return []

    sweeper = ExpirySweeper(Counting(), interval=0.01)  # type: ignore[arg-type]
    sweeper.start()
    assert swept.wait(5)
    sweeper.stop()
    after = Counting.calls
    assert after >= 1
    sweeper.stop()                                        # stopping twice is harmless
    assert Counting.calls == after


def test_serve_runs_the_sweep_beside_the_loop_and_stops_it_when_the_client_leaves():
    srv = server(expiry_interval=3600)
    out = io.StringIO()
    assert srv.serve(io.StringIO(""), out) == 0
    assert not srv._sweeper._thread.is_alive()
    srv.close()


def test_a_memvara_opened_with_the_switch_off_gets_no_sweeper():
    """The switch on the engine and the switch on the server both stop the sweep."""
    assert MemvaraMCPServer(mem(expiry_erasure=False))._sweeper is None


def test_a_store_that_cannot_list_expired_claims_gets_no_sweeper():
    store = _NoExpiry(SQLiteStore(":memory:"))
    m = Memvara(store=store, llm=NullLLM(), embedder=HashingEmbedder(dim=64))
    assert MemvaraMCPServer(m)._sweeper is None
    m.close()


# --- review fixes: scope of a repeat, reads after the instant, read-only servers --------

A, B = "github.com/acme/a", "github.com/acme/b"


def test_a_repeat_with_an_expiry_in_another_project_writes_its_own_claim():
    """Reinforcing across projects would put project A's expiry on project B's claim, and
    the sweep would then erase a fact B relies on. The repeat is written as its own claim
    in A instead, so the expiry is kept and B's claim is untouched."""
    with mem() as m:
        theirs = m.scope(project=B).remember("user", "door_code", "4411").added[0]
        receipt = m.scope(project=A).remember("user", "door_code", "4411",
                                              expires_at=utcnow() + DAY)
        assert receipt.reinforced == []
        mine = receipt.added[0]
        assert mine.scope.project == A and mine.expires_at is not None
        assert m.store.get_claim(theirs.id).expires_at is None
        assert [e.claim_id for e in m.erase_expired(now=utcnow() + 2 * DAY)] == [mine.id]
        assert m.store.get_claim(theirs.id) is not None


def test_a_repeat_with_an_expiry_in_another_session_leaves_that_claim_alone_too():
    """Agent and session are not in the value key either, so the same rule applies one
    level down, and the new claim beside the old is not reported as a pile-up."""
    with mem() as m:
        theirs = m.remember("user", "door_code", "4411", session="s1").added[0]
        receipt = m.remember("user", "door_code", "4411", session="s2",
                             expires_at=utcnow() + DAY)
        assert receipt.reinforced == [] and receipt.accumulated == []
        assert receipt.added[0].scope.session == "s2"
        assert m.store.get_claim(theirs.id).expires_at is None


def test_a_repeat_with_an_expiry_in_the_same_project_reinforces_the_claim_on_record():
    with mem() as m:
        scoped = m.scope(project=A)
        first = scoped.remember("user", "door_code", "4411").added[0]
        later = utcnow() + 3 * DAY
        other = m.scope(project=B).remember("user", "door_code", "4411",
                                            expires_at=later).added[0]
        assert other.id != first.id
        soon = utcnow() + DAY
        receipt = scoped.remember("user", "door_code", "4411", expires_at=soon)
        assert [c.id for c in receipt.reinforced] == [first.id]
        assert m.store.get_claim(first.id).expires_at == soon
        assert m.store.get_claim(other.id).expires_at == later


def _overdue(m: Memvara, **kw: Any) -> Any:
    """A claim whose expiry has passed and that no sweep has erased yet."""
    claim = m.remember("user", "door_code", "4411", expires_at=utcnow() + DAY,
                       **kw).added[0]
    claim.expires_at = utcnow() - timedelta(seconds=1)
    m.store.put_claim(claim)
    return claim


def test_a_claim_stops_answering_the_moment_its_expiry_passes_before_any_sweep(m):
    """The sweep only deletes. Reads leave the claim out as soon as the instant passes,
    inside the store query where the limit is applied, so `k` still counts."""
    kept = m.remember("user", "lives_in", "Lisbon").added[0]
    code = _overdue(m)
    assert m.store.get_claim(code.id) is not None, "not erased yet"
    assert m.get(code.id) is None and m.why(code.id) is None
    assert [c.id for c in m.get_all()] == [kept.id]
    assert [c.id for c in m.get_all(states=["live", "ended", "retired"])] == [kept.id]
    assert m.count() == 1
    assert all(r.claim.id != code.id for r in m.search("user door code 4411", k=5))
    assert "4411" not in str(m.recall("door code"))
    assert m.history("user", "door_code") == []
    assert m.store.candidate_ids([code.scope]) == [kept.id]
    assert m.store.lexical_search("4411", [code.scope], 5) == []


def test_an_expired_claim_is_not_reinforced_so_a_new_statement_survives_the_sweep(m):
    """Reinforcing the doomed claim would hand the new statement to the sweep."""
    old = _overdue(m)
    receipt = m.remember("user", "door_code", "4411")
    assert receipt.reinforced == [] and receipt.added[0].id != old.id
    assert [e.claim_id for e in m.erase_expired()] == [old.id]
    assert m.get(receipt.added[0].id) is not None


def test_a_retraction_repeated_after_the_first_expired_keeps_a_tombstone_of_its_own(m):
    """The retraction side of the test above. A repeated retraction is normally folded
    into the tombstone the first one left. Folded into a tombstone whose expiry has
    passed, it was erased with that tombstone by the next sweep, and a retraction written
    with no expiry left no record at all (#284)."""
    m.remember("user", "likes", "tea")
    m.remember("user", "likes", "tea", polarity=-1, expires_at=utcnow() + DAY)
    [first] = [c for c in m.store.iter_claims(None, True) if c.polarity < 0]
    first.expires_at = utcnow() - timedelta(seconds=1)
    m.store.put_claim(first)

    m.remember("user", "likes", "tea", polarity=-1)
    tombstones = [c for c in m.store.iter_claims(None, True) if c.polarity < 0]
    assert len(tombstones) == 2, "the repeat was folded into the expired tombstone"
    [second] = [c for c in tombstones if c.id != first.id]
    assert second.expires_at is None and second.state == "retired"
    assert [e.claim_id for e in m.erase_expired()] == [first.id]
    assert [c.id for c in m.store.iter_claims(None, True) if c.polarity < 0] == [second.id]


def test_erase_by_name_still_erases_a_claim_whose_expiry_has_passed(m):
    code = _overdue(m)
    assert m.erase(code.id) is True
    assert m.store.get_claim(code.id) is None


def test_with_the_switch_off_an_expired_claim_is_still_returned(tmp_path):
    """Off means the expiry does nothing: not erased, and not hidden either."""
    with mem(expiry_erasure=False) as m:
        code = _overdue(m)
        assert m.get(code.id) is not None
        assert [c.id for c in m.get_all()] == [code.id]


def test_a_read_only_server_does_not_erase_at_open_or_hourly(tmp_path):
    """Erasing is a write, and a read-only server writes nothing. The claim still stops
    answering, because reads leave it out whether or not it has been erased."""
    from memvara.server.config import ServerConfig, build_memvara

    path = str(tmp_path / "m.db")
    claim_id = _due(path)
    config = ServerConfig(path=path, read_only=True, embedder="hashing:64",
                          features_off=frozenset({"encryption"}))
    memory = build_memvara(config)
    try:
        assert memory.store.get_claim(claim_id) is not None
        assert memory.get(claim_id) is None
        assert MemvaraMCPServer(memory, read_only=True)._sweeper is None
    finally:
        memory.close()
