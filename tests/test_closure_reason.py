"""A reason recorded when a memory is ended, retired, set to end, or replaced.

The reason lives on the closure witness in `meta["closure"]`, beside the clock that
stopped, so it is written by the one function that ends any claim (`close_out`) and read
by the two that explain one (`history()` and `why()`). The acceptance test for this
feature is that a reason written through each surface reads back through both, and most
of this file is that test, once per surface.
"""
from datetime import datetime, timedelta, timezone

import pytest

from memvara import HashingEmbedder, Memvara, NullLLM
from memvara.server import MemvaraMCPServer
from memvara.types import (CLOSURE, REASON_CHARS, Claim, close_out, closure_reason,
                           closure_reasons, planned_end, utcnow)

from test_server import call, text

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2026, 3, 1, tzinfo=timezone.utc)


def mem() -> Memvara:
    return Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), user="alice")


@pytest.fixture()
def server():
    srv = MemvaraMCPServer(mem(), user="alice")
    yield srv
    srv.close()


# --- the record itself ------------------------------------------------------


def test_a_closure_without_a_reason_writes_exactly_the_record_it_always_did():
    """Existing rows must not change shape. A `reason: None` key on every supersession
    would rewrite the witness of every claim the engine closes from now on, for no
    information."""
    claim = Claim(subject="user", predicate="lives_in", object="Berlin", valid_from=T0)
    close_out(claim, T1, None, "ended")
    assert claim.meta[CLOSURE] == [{"at": T1.timestamp(), "close": "ended", "by": None}]


def test_a_reason_lands_on_the_witness_beside_the_clock_that_stopped():
    claim = Claim(subject="user", predicate="lives_in", object="Berlin", valid_from=T0)
    close_out(claim, T1, None, "ended", "moved to Lisbon")
    close_out(claim, T1 + timedelta(days=1), None, "retired", "it was never Berlin")
    assert closure_reasons(claim) == [("ended", "moved to Lisbon"),
                                      ("retired", "it was never Berlin")]


def test_a_blank_or_overlong_reason_is_refused_rather_than_stored():
    with pytest.raises(ValueError, match="blank"):
        closure_reason("   ")
    with pytest.raises(ValueError, match="limit is 500"):
        closure_reason("x" * (REASON_CHARS + 1))
    with pytest.raises(TypeError, match="must be a string"):
        closure_reason(5)  # type: ignore[arg-type]
    assert closure_reason("x" * REASON_CHARS) == "x" * REASON_CHARS


def test_a_planned_end_needs_an_end_to_explain():
    with pytest.raises(ValueError, match="no end date"):
        planned_end(Claim(subject="user", predicate="p", object="o"), "because")


def test_a_witness_that_is_not_a_record_is_skipped_when_reading_reasons():
    """`meta` is JSON a caller could once reach, so the reader takes what it can use and
    skips the rest rather than raising inside `history()`."""
    claim = Claim(subject="user", predicate="p", object="o",
                  meta={CLOSURE: ["garbage", {"close": "ended"}]})
    assert closure_reasons(claim) == []


# --- the library ------------------------------------------------------------


def test_a_reason_given_to_delete_reads_back_through_history_and_why():
    m = mem()
    claim = m.remember("user", "works_at", "Acme").added[0]
    assert m.delete(claim.id, close="ended", reason="left for Globex")
    [row] = m.history("user", "works_at")
    assert closure_reasons(row) == [("ended", "left for Globex")]
    prov = m.why(claim.id)
    assert prov is not None and closure_reasons(prov.claim) == [
        ("ended", "left for Globex")]


def test_a_reason_given_to_forget_is_recorded_on_every_value_it_closes():
    m = mem()
    m.remember("user", "likes", "tea")
    m.remember("user", "likes", "coffee")
    closed = m.forget("user", "likes", reason="misheard both")
    assert len(closed) == 2
    for row in m.history("user", "likes"):
        assert closure_reasons(row) == [("retired", "misheard both")]


def test_a_bad_reason_is_refused_before_anything_is_closed():
    """Validated first, so a refusal cannot leave half a slot closed."""
    m = mem()
    claim = m.remember("user", "works_at", "Acme").added[0]
    with pytest.raises(ValueError):
        m.delete(claim.id, reason=" ")
    with pytest.raises(ValueError):
        m.forget("user", "works_at", reason="x" * 501)
    assert m.get(claim.id).state == "live"


def test_a_planned_end_carries_its_reason_from_the_moment_it_is_written():
    m = mem()
    until = utcnow() + timedelta(days=30)
    claim = m.remember("user", "contract_with", "Acme", valid_to=until,
                       until_reason="the contract runs out").added[0]
    assert closure_reasons(claim) == [("ended", "the contract runs out")]
    assert claim.meta[CLOSURE][0]["at"] == until.timestamp()
    [row] = m.history("user", "contract_with")
    assert closure_reasons(row) == [("ended", "the contract runs out")]


def test_until_reason_without_an_end_is_refused():
    with pytest.raises(ValueError, match="no end date"):
        mem().remember("user", "contract_with", "Acme", until_reason="soon")


# --- replaces ---------------------------------------------------------------


def test_replaces_ends_one_value_of_a_many_valued_slot_and_records_the_lineage():
    """The case the argument exists for. `likes` accumulates, so writing "coffee" beside
    "tea" leaves both live; naming the replaced value is the only way to end it, and the
    lineage has to survive: `invalidated_by` on the old row, and `why()` on the new one
    naming it."""
    m = mem()
    tea = m.remember("user", "likes", "tea").added[0]
    m.remember("user", "likes", "jazz")
    receipt = m.remember("user", "likes", "coffee", replaces=tea.id,
                         reason="switched to coffee")
    [coffee] = receipt.added
    assert [c.id for c in receipt.closed] == [tea.id]
    old = m.get(tea.id)
    assert old.state == "ended" and old.invalidated_by == coffee.id
    assert closure_reasons(old) == [("ended", "switched to coffee")]
    assert [c.id for c in m.why(coffee.id).superseded] == [tea.id]
    live = sorted(c.object for c in m.get_all())
    assert live == ["coffee", "jazz"], "only the named value was replaced"


def test_replaces_ends_the_old_value_where_the_new_one_began():
    m = mem()
    old = m.remember("user", "likes", "tea", valid_from=T0).added[0]
    m.remember("user", "likes", "coffee", valid_from=T1, replaces=old.id)
    assert m.get(old.id).valid_to == T1


def test_replaces_under_retired_closes_belief_now_and_leaves_the_interval_alone():
    m = mem()
    old = m.remember("user", "likes", "tea", valid_from=T0).added[0]
    receipt = m.remember("user", "likes", "coffee", valid_from=T1, replaces=old.id,
                         close="retired", reason="it was never tea")
    row = m.get(old.id)
    assert row.state == "retired" and row.valid_to is None
    assert row.invalidated_at == receipt.added[0].recorded_at


def test_replaces_refuses_an_id_this_scope_cannot_see():
    m = mem()
    other = m.remember("user", "likes", "tea", user="bob").added[0]
    with pytest.raises(KeyError, match="no claim"):
        m.remember("user", "likes", "coffee", replaces=other.id)
    with pytest.raises(KeyError):
        m.remember("user", "likes", "coffee", replaces="cl_nothing")
    assert [c.object for c in m.get_all()] == [], "nothing was written"


def test_replaces_refuses_a_claim_that_is_no_longer_live():
    m = mem()
    old = m.remember("user", "likes", "tea").added[0]
    m.delete(old.id)
    with pytest.raises(ValueError, match="already retired"):
        m.remember("user", "likes", "coffee", replaces=old.id)
    assert [c.object for c in m.get_all()] == []


@pytest.mark.parametrize("close", ["ended", "retired"])
@pytest.mark.parametrize("dated", [True, False])
def test_remember_replaces_and_supersede_close_the_old_claim_identically(close, dated):
    """One implementation, so one result. `remember(replaces=)` used to carry its own
    copy of the supersession, and the two had drifted apart on the closure instant and
    on which claims they would accept."""
    closures = []
    for use_supersede in (False, True):
        m = mem()
        old = m.remember("user", "likes", "tea", valid_from=T0).added[0]
        began = {"valid_from": T1} if dated else {}
        if use_supersede:
            new = Claim(subject="user", predicate="likes", object="coffee",
                        scope=old.scope, valid_from=T1 if dated else utcnow())
            m.supersede(old.id, new, close=close, reason="switched")
        else:
            m.remember("user", "likes", "coffee", replaces=old.id, close=close,
                       reason="switched", **began)
        row = m.get(old.id)
        new_id = row.invalidated_by
        new_row = m.get(new_id)
        closures.append((row.state, closure_reasons(row),
                         row.valid_to == (new_row.valid_from if close == "ended" else None),
                         row.invalidated_at == (new_row.recorded_at
                                                if close == "retired" else None)))
    assert closures[0] == closures[1]
    assert closures[0] == (close, [(close, "switched")], True, True)


def test_supersede_ends_the_old_claim_where_the_new_one_begins_unless_told_otherwise():
    m = mem()
    old = m.remember("user", "likes", "tea", valid_from=T0).added[0]
    new = Claim(subject="user", predicate="likes", object="coffee", scope=old.scope,
                valid_from=T1)
    m.supersede(old.id, new)
    assert m.get(old.id).valid_to == T1
    other = m.remember("user", "likes", "jazz", valid_from=T0).added[0]
    at = T1 + timedelta(days=3)
    m.supersede(other.id, Claim(subject="user", predicate="likes", object="blues",
                                scope=old.scope, valid_from=T1), at=at)
    assert m.get(other.id).valid_to == at, "an explicit at still wins"


def test_supersede_refuses_to_end_a_claim_that_is_not_live_and_writes_nothing():
    m = mem()
    old = m.remember("user", "likes", "tea").added[0]
    m.delete(old.id, close="ended")
    before = len(m.get_all(include_invalidated=True))
    with pytest.raises(ValueError, match="already ended, so there is nothing left to end"):
        m.supersede(old.id, Claim(subject="user", predicate="likes", object="coffee"))
    assert len(m.get_all(include_invalidated=True)) == before


def test_supersede_refuses_a_retired_claim_under_either_closure():
    m = mem()
    old = m.remember("user", "likes", "tea").added[0]
    m.delete(old.id)
    for close in ("ended", "retired"):
        with pytest.raises(ValueError, match="already retired"):
            m.supersede(old.id, Claim(subject="user", predicate="likes", object="x"),
                        close=close)
    assert [c.object for c in m.get_all(include_invalidated=True)] == ["tea"]


def test_supersede_may_retire_a_claim_that_already_ended():
    """A value that finished and was later found never to have been true is a real
    correction, and a replayed mutation log contains exactly that sequence."""
    m = mem()
    old = m.remember("user", "likes", "tea").added[0]
    m.delete(old.id, close="ended", reason="stopped")
    m.supersede(old.id, Claim(subject="user", predicate="likes", object="coffee"),
                close="retired", reason="it was never tea")
    assert closure_reasons(m.get(old.id)) == [("ended", "stopped"),
                                              ("retired", "it was never tea")]


def test_supersede_records_a_reason_and_refuses_a_bad_one_before_writing():
    m = mem()
    old = m.remember("user", "likes", "tea").added[0]
    with pytest.raises(ValueError, match="blank"):
        m.supersede(old.id, Claim(subject="user", predicate="likes", object="x"),
                    reason=" ")
    assert m.get(old.id).state == "live"
    m.scope(user="alice").supersede(
        old.id, Claim(subject="user", predicate="likes", object="coffee"),
        reason="switched")
    assert closure_reasons(m.get(old.id)) == [("ended", "switched")]


def test_a_reason_without_replaces_is_refused():
    with pytest.raises(ValueError, match="no claim\nwas named|was named"):
        mem().remember("user", "likes", "coffee", reason="because")


# --- the MCP surface --------------------------------------------------------


def _id(server, predicate, obj):
    rows = server._ctx.memory.get_all(include_invalidated=True)
    return next(c.id for c in rows if c.predicate == predicate and c.object == obj)


def test_memory_end_with_a_reason_shows_it_in_history_and_why(server):
    text(server, "memory_remember", {"predicate": "works_at", "object": "Acme"})
    cid = _id(server, "works_at", "Acme")
    text(server, "memory_end", {"claim_id": cid, "reason": "left for Globex"})
    assert "ended because: left for Globex" in text(
        server, "memory_history", {"predicate": "works_at"})
    assert "ended because: left for Globex" in text(server, "memory_why",
                                                    {"claim_id": cid})


def test_memory_end_by_slot_records_the_reason_too(server):
    text(server, "memory_remember", {"predicate": "works_at", "object": "Acme"})
    text(server, "memory_end", {"predicate": "works_at", "reason": "company closed"})
    assert "ended because: company closed" in text(
        server, "memory_history", {"predicate": "works_at"})


def test_memory_forget_with_a_reason_shows_it_in_history(server):
    text(server, "memory_remember", {"predicate": "lives_in", "object": "Berlin"})
    cid = _id(server, "lives_in", "Berlin")
    text(server, "memory_forget", {"claim_id": cid, "reason": "misheard"})
    text(server, "memory_remember", {"predicate": "likes", "object": "tea"})
    text(server, "memory_forget", {"predicate": "likes", "reason": "about someone else"})
    assert "retired because: misheard" in text(server, "memory_history",
                                               {"predicate": "lives_in"})
    assert "retired because: about someone else" in text(server, "memory_history",
                                                         {"predicate": "likes"})


def test_a_blank_reason_is_an_argument_error_that_names_the_field(server):
    text(server, "memory_remember", {"predicate": "works_at", "object": "Acme"})
    body, is_error = call(server, "memory_end",
                          {"predicate": "works_at", "reason": "  "})
    assert is_error and body.startswith("memory_end.reason: reason is blank")
    body, is_error = call(server, "memory_end", {"predicate": "works_at",
                                                 "reason": "x" * 501})
    assert is_error and "at most 500" in body


def test_memory_remember_records_a_planned_end_with_its_reason(server):
    until = (utcnow() + timedelta(days=10)).isoformat()
    text(server, "memory_remember", {"predicate": "contract_with", "object": "Acme",
                                     "true_until": until,
                                     "until_reason": "the contract runs out"})
    assert "ended because: the contract runs out" in text(
        server, "memory_history", {"predicate": "contract_with"})


def test_memory_remember_refuses_until_reason_without_true_until(server):
    body, is_error = call(server, "memory_remember", {
        "predicate": "contract_with", "object": "Acme", "until_reason": "soon"})
    assert is_error and "no true_until was sent" in body


def test_memory_remember_replaces_one_value_and_records_why(server):
    # A predicate nobody declared holds many values, which is the case `replaces` is for.
    text(server, "memory_remember", {"predicate": "uses_library", "object": "Jest"})
    text(server, "memory_remember", {"predicate": "uses_library", "object": "ruff"})
    jest = _id(server, "uses_library", "Jest")
    body = text(server, "memory_remember", {
        "predicate": "uses_library", "object": "Vitest", "replaces": jest,
        "reason": "moved from Jest to Vitest"})
    assert "ended 1" in body
    history = text(server, "memory_history", {"predicate": "uses_library"})
    assert "ended because: moved from Jest to Vitest" in history
    live = sorted(c.object for c in server._ctx.memory.get_all())
    assert live == ["Vitest", "ruff"]


def test_memory_remember_refuses_a_reason_without_replaces(server):
    body, is_error = call(server, "memory_remember", {
        "predicate": "likes", "object": "tea", "reason": "because"})
    assert is_error and "replaces was not sent" in body


def test_memory_remember_refuses_a_replaces_id_it_cannot_see_and_writes_nothing(server):
    body, is_error = call(server, "memory_remember", {
        "predicate": "likes", "object": "tea", "replaces": "cl_nothing"})
    assert is_error and "names no fact visible here" in body
    assert server._ctx.memory.get_all() == []


def test_memory_remember_refuses_a_replaces_id_that_is_no_longer_live(server):
    text(server, "memory_remember", {"predicate": "likes", "object": "tea"})
    tea = _id(server, "likes", "tea")
    text(server, "memory_forget", {"claim_id": tea})
    body, is_error = call(server, "memory_remember", {
        "predicate": "likes", "object": "coffee", "replaces": tea})
    assert is_error and body.startswith("Nothing written:") and "already retired" in body
