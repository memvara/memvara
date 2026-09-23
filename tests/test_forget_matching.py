"""Ending or retiring every memory that matches a query, after seeing exactly which.

The acceptance tests from the design are the first three here: a preview changes
nothing, a confirmed call closes exactly the previewed ids, and a stale token closes
nothing. The rest pin the ways a token is refused, each of which must leave the store
exactly as it was, and the two MCP tools that split the closure by name.

Time is controlled by issuing tokens at an explicit instant through `Confirmer`, never by
patching the clock: the expiry is a belief-clock fact about a token, and a patched clock
would hide which instant the check actually read.
"""
from datetime import datetime, timedelta, timezone

import pytest

from memvara import (ConfirmationRefused, ForgetPreview, ForgetResult, HashingEmbedder,
                     Memvara, NullLLM)
from memvara.confirm import CONFIRM_TTL, Confirmer
from memvara.server import MemvaraMCPServer
from memvara.types import closure_reasons, utcnow

from test_server import call, text


def mem(**kw) -> Memvara:
    return Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), user="alice", **kw)


def seeded() -> tuple[Memvara, list[str]]:
    m = mem()
    ids = [m.remember("user", "maintains", obj).added[0].id
           for obj in ("staging cluster alpha", "staging cluster beta")]
    m.remember("user", "likes", "tea")
    return m, ids


def states(m: Memvara) -> dict[str, str]:
    return {c.id: c.state for c in m.get_all(include_invalidated=True)}


# --- the three acceptance tests ---------------------------------------------


def test_a_preview_changes_nothing():
    m, ids = seeded()
    before = states(m)
    preview = m.forget_matching("staging cluster", close="retired", k=2)
    assert isinstance(preview, ForgetPreview)
    assert set(preview.matches) == set(ids)
    assert preview.close == "retired"
    assert states(m) == before


def test_a_confirmed_call_closes_exactly_the_previewed_ids():
    """And nothing that started matching between the two calls. The query is not run
    again, so a third staging cluster written after the preview stays live."""
    m, ids = seeded()
    preview = m.forget_matching("staging cluster", close="ended", k=2)
    late = m.remember("user", "maintains", "staging cluster gamma").added[0]
    done = m.forget_matching("staging cluster", close="ended", k=2,
                             reason="the staging clusters were shut down",
                             confirm=preview.confirm)
    assert isinstance(done, ForgetResult)
    assert sorted(c.id for c in done.closed) == sorted(ids)
    assert {c.state for c in done.closed} == {"ended"}
    assert done.reason == "the staging clusters were shut down"
    after = states(m)
    assert all(after[i] == "ended" for i in ids)
    assert after[late.id] == "live"
    for i in ids:
        assert closure_reasons(m.get(i)) == [("ended",
                                              "the staging clusters were shut down")]


def test_a_stale_token_closes_nothing():
    m, ids = seeded()
    stale, _ = m._confirmer.issue(ids, "retired", now=utcnow() - CONFIRM_TTL
                                  - timedelta(seconds=1))
    before = states(m)
    with pytest.raises(ConfirmationRefused, match="expired"):
        m.forget_matching("staging cluster", close="retired", confirm=stale)
    assert states(m) == before


# --- every other refusal also changes nothing -------------------------------


def test_a_token_whose_claims_changed_since_the_preview_is_refused_whole():
    """All or nothing: closing the one that is still live would close a set the caller
    never saw."""
    m, ids = seeded()
    preview = m.forget_matching("staging cluster", close="retired", k=2)
    m.delete(ids[0], close="ended")
    before = states(m)
    with pytest.raises(ConfirmationRefused, match=f"{ids[0]} .* now ended"):
        m.forget_matching("staging cluster", close="retired", confirm=preview.confirm)
    assert states(m) == before


def test_a_token_naming_a_claim_that_was_erased_is_refused():
    m, ids = seeded()
    preview = m.forget_matching("staging cluster", close="retired", k=2)
    m.erase(ids[1])
    with pytest.raises(ConfirmationRefused, match="no longer visible"):
        m.forget_matching("staging cluster", close="retired", confirm=preview.confirm)
    assert m.get(ids[0]).state == "live"


def test_a_token_issued_for_the_other_closure_is_refused():
    m, ids = seeded()
    preview = m.forget_matching("staging cluster", close="ended", k=2)
    with pytest.raises(ConfirmationRefused, match="issued to end memories"):
        m.forget_matching("staging cluster", close="retired", confirm=preview.confirm)
    retire = m.forget_matching("staging cluster", close="retired", k=2)
    with pytest.raises(ConfirmationRefused, match="issued to retire memories"):
        m.forget_matching("staging cluster", close="ended", confirm=retire.confirm)
    assert {states(m)[i] for i in ids} == {"live"}


@pytest.mark.parametrize("token", [
    "", "no-dot-at-all", "!!!.abc", "eyJ9.0000", "e30.deadbeef", "a.bad-padding",
])
def test_a_token_this_key_did_not_issue_is_refused(token):
    m, ids = seeded()
    with pytest.raises(ConfirmationRefused, match="not issued by this memory server"):
        m.forget_matching("staging cluster", close="retired", confirm=token)
    assert {states(m)[i] for i in ids} == {"live"}


def test_an_altered_token_is_refused():
    """Swapping one id in the payload for another the caller can see is the attack the
    MAC exists for: without it the caller could confirm a set nobody previewed."""
    m, ids = seeded()
    other = m.remember("user", "likes", "coffee").added[0]
    preview = m.forget_matching("staging cluster", close="retired", k=2)
    body, mac = preview.confirm.rsplit(".", 1)
    forged_body, _ = Confirmer("attacker").issue([other.id], "retired", now=utcnow())
    forged = forged_body.rsplit(".", 1)[0] + "." + mac
    with pytest.raises(ConfirmationRefused, match="altered"):
        m.forget_matching("x", close="retired", confirm=forged)
    assert m.get(other.id).state == "live"


def test_a_token_from_another_scope_closes_nothing_there():
    """The key is shared by every scope in a process; visibility is what keeps a token
    minted for one user from closing another user's memories."""
    m, ids = seeded()
    preview = m.forget_matching("staging cluster", close="retired", k=2)
    with pytest.raises(ConfirmationRefused, match="no longer visible"):
        m.forget_matching("staging cluster", close="retired", confirm=preview.confirm,
                          user="bob")
    assert {states(m)[i] for i in ids} == {"live"}


# --- the key ----------------------------------------------------------------


def test_a_configured_key_lets_another_process_confirm_the_preview():
    """Two engines over one store with the same key stand in for two worker processes."""
    a = mem(confirm_secret="shared")
    claim = a.remember("user", "maintains", "staging cluster").added[0]
    preview = a.forget_matching("staging cluster", close="ended", k=1)
    b = Memvara(store=a.store, llm=NullLLM(), embedder=HashingEmbedder(dim=64),
                user="alice", confirm_secret=b"shared")
    done = b.forget_matching("staging cluster", close="ended", confirm=preview.confirm)
    assert [c.id for c in done.closed] == [claim.id]


def test_a_different_configured_key_refuses_the_token():
    a = mem(confirm_secret="one")
    a.remember("user", "maintains", "staging cluster")
    preview = a.forget_matching("staging cluster", close="ended", k=1)
    b = Memvara(store=a.store, llm=NullLLM(), embedder=HashingEmbedder(dim=64),
                user="alice", confirm_secret="two")
    with pytest.raises(ConfirmationRefused, match="not issued"):
        b.forget_matching("staging cluster", close="ended", confirm=preview.confirm)


def test_with_no_key_configured_one_process_shares_one_generated_key():
    a, b = mem(), mem()
    token, _ = a._confirmer.issue(["cl_1"], "ended", now=utcnow())
    assert b._confirmer.check(token, "ended", now=utcnow()) == ["cl_1"]


def test_an_empty_key_is_refused_rather_than_used():
    with pytest.raises(ValueError, match="empty"):
        Confirmer("")


def test_the_expiry_shown_is_the_expiry_the_token_carries():
    """The token encodes whole seconds. The instant handed back used to keep the
    fraction, so a caller confirming in the last second it was shown was refused."""
    c = Confirmer("k")
    at = datetime(2026, 9, 23, 12, 0, 0, 900_000, tzinfo=timezone.utc)
    token, expires = c.issue(["cl_1"], "ended", now=at)
    assert expires == datetime(2026, 9, 23, 12, 10, 0, tzinfo=timezone.utc)
    assert c.check(token, "ended", now=expires - timedelta(microseconds=1)) == ["cl_1"]
    with pytest.raises(ConfirmationRefused, match="expired"):
        c.check(token, "ended", now=expires)


def test_the_token_expires_ten_minutes_after_the_preview():
    c = Confirmer("k")
    at = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
    token, expires = c.issue(["cl_1"], "ended", now=at)
    assert expires == at + timedelta(minutes=10)
    assert c.check(token, "ended", now=expires - timedelta(seconds=1)) == ["cl_1"]
    with pytest.raises(ConfirmationRefused, match="expired"):
        c.check(token, "ended", now=expires)


# --- arguments --------------------------------------------------------------


def test_erasure_is_not_a_closure_this_accepts():
    m, _ = seeded()
    with pytest.raises(ValueError, match="is not a closure"):
        m.forget_matching("staging", close="erased")


@pytest.mark.parametrize("k", [0, 101])
def test_k_is_bounded(k):
    with pytest.raises(ValueError, match="out of range"):
        mem().forget_matching("x", close="ended", k=k)


def test_a_bad_reason_is_refused_before_the_preview():
    with pytest.raises(ValueError, match="blank"):
        mem().forget_matching("x", close="ended", reason=" ")


def test_the_scoped_and_async_views_forward_every_argument():
    import asyncio

    from memvara import AsyncMemvara

    m, ids = seeded()
    view = m.scope(user="alice")
    preview = view.forget_matching("staging cluster", close="retired", k=2)
    assert set(preview.matches) == set(ids)

    async def main():
        amem = AsyncMemvara(m)
        again = await amem.forget_matching("staging cluster", close="retired", k=2,
                                           user="alice")
        done = await amem.scope(user="alice").forget_matching(
            "staging cluster", close="retired", reason="misheard",
            confirm=again.confirm)
        return done

    done = asyncio.run(main())
    assert sorted(c.id for c in done.closed) == sorted(ids)


# --- the two MCP tools ------------------------------------------------------


@pytest.fixture()
def server():
    srv = MemvaraMCPServer(mem(), user="alice")
    yield srv
    srv.close()


def _confirm_token(body: str) -> str:
    return body.rsplit("confirm: ", 1)[1].strip()


def _seed(server):
    for obj in ("staging cluster alpha", "staging cluster beta"):
        text(server, "memory_remember", {"predicate": "maintains", "object": obj})


def test_memory_end_matching_previews_then_ends_exactly_what_it_listed(server):
    _seed(server)
    body = text(server, "memory_end_matching", {"query": "staging cluster", "k": 2})
    assert body.startswith("Preview: 2 live match(es). Nothing has changed yet.")
    assert all(c.state == "live" for c in server._ctx.memory.get_all())
    done = text(server, "memory_end_matching", {
        "query": "staging cluster", "k": 2, "reason": "shut down",
        "confirm": _confirm_token(body)})
    assert done.startswith("Ended 2 value(s), with the reason 'shut down'.")
    assert "marked ended." in done
    history = text(server, "memory_history", {"predicate": "maintains"})
    assert history.count("ended because: shut down") == 2


def test_memory_forget_matching_retires_and_says_so(server):
    _seed(server)
    body = text(server, "memory_forget_matching", {"query": "staging cluster", "k": 2})
    assert "retire them one at a time with memory_forget" in body or \
        "retire the right ones one at a time with memory_forget" in body
    done = text(server, "memory_forget_matching", {
        "query": "staging cluster", "k": 2, "confirm": _confirm_token(body)})
    assert done.startswith("Retired 2 value(s).") and "marked retired." in done
    assert {c.state for c in server._ctx.memory.get_all(include_invalidated=True)} == {
        "retired"}


def test_a_token_from_one_matching_tool_is_refused_by_the_other(server):
    _seed(server)
    body = text(server, "memory_end_matching", {"query": "staging cluster", "k": 2})
    refused, is_error = call(server, "memory_forget_matching", {
        "query": "staging cluster", "confirm": _confirm_token(body)})
    assert is_error and "issued to end memories" in refused
    assert all(c.state == "live" for c in server._ctx.memory.get_all())


def test_a_matching_tool_says_so_when_nothing_matches(server):
    body = text(server, "memory_end_matching", {"query": "anything at all"})
    assert body.startswith("Nothing matched 'anything at all', so there is nothing to "
                           "end.")


def test_a_confirmed_empty_preview_reports_that_nothing_was_closed(server):
    body = text(server, "memory_forget_matching", {"query": "anything"})
    assert "confirm" not in body
    token, _ = server._ctx.memory.memvara._confirmer.issue([], "retired", now=utcnow())
    done = text(server, "memory_forget_matching", {"query": "anything",
                                                   "confirm": token})
    assert done == "The preview listed nothing, so nothing was retired."


def test_a_blank_query_is_refused_on_the_preview(server):
    body, is_error = call(server, "memory_end_matching", {"query": "  "})
    assert is_error and "memory_end_matching.query is blank" in body
    body, is_error = call(server, "memory_end_matching", {})
    assert is_error and "memory_end_matching.query is blank" in body


def test_the_confirming_call_needs_no_query(server):
    """The confirmation closes what the token lists and never reads the query, so a
    missing or blank one there is not a reason to refuse a correct call."""
    _seed(server)
    body = text(server, "memory_end_matching", {"query": "staging cluster", "k": 2})
    done = text(server, "memory_end_matching", {"confirm": _confirm_token(body)})
    assert done.startswith("Ended 2 value(s).")


def test_neither_matching_tool_offers_a_closure_argument():
    """The closure is the tool's name. See the module docstring of `server/tools.py`."""
    from memvara.server.tools import BY_NAME

    for name in ("memory_end_matching", "memory_forget_matching"):
        assert "close" not in BY_NAME[name].properties
