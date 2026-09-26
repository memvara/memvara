"""Confirmed bugs, each pinned by a strict expected failure that cites its issue.

Each test states the behaviour the fix must produce. It raises known_bugs.Reproduced only
when it has seen that bug's own symptom, and the bug's marker accepts nothing else, so a
different failure in the same test fails loudly instead of passing for the known bug.
"""

from __future__ import annotations

import sys
from typing import Callable

import pytest

from harness import known_bugs, stores
from harness.hooks import HOOKS_DIR, HookRunner
from harness.stdio import McpProcess
from memvara import MemoryType
from memvara.server.tools import TOOLS

if str(HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(HOOKS_DIR))
import approve  # noqa: E402 - plugin/hooks is not a package; the path is set just above

LEVELS = ("session", "agent")

#: The two read-only tools that #267 is about.
B3_MISSING = {"memory_get_document", "memory_list_documents"}


def _live(scoped: object) -> list[str]:
    return sorted(claim.object for claim in scoped.get_all())  # type: ignore[attr-defined]


def _ended(scoped: object) -> list[str]:
    return sorted(claim.object  # type: ignore[attr-defined]
                  for claim in scoped.get_all(states=("ended",)))  # type: ignore[attr-defined]


def _read_only_tools() -> set[str]:
    return {tool.name for tool in TOOLS if not tool.writes}


def test_a_known_bug_marker_accepts_only_the_bugs_own_symptom() -> None:
    """Each marker absorbs only known_bugs.Reproduced, which a test raises after it has
    seen the bug's exact symptom. Any other failure in the same test fails loudly."""
    mark = known_bugs.xfail("B2").mark
    assert mark.kwargs["strict"] is True
    assert mark.kwargs["raises"] is known_bugs.Reproduced


# -- B2: a bound write ends the user-wide value --------------------------------------

@pytest.mark.parametrize("level", LEVELS)
@known_bugs.xfail("B2")
def test_a_bound_write_leaves_the_user_wide_value_live(level: str) -> None:
    mem = stores.memory()
    user = mem.scope(user="u")
    user.remember("user", "lives_in", "Berlin")
    mem.scope(user="u", **{level: "one"}).remember("user", "lives_in", "Paris")
    live = _live(user)
    if not live and _ended(user) == ["Berlin"]:
        raise known_bugs.Reproduced(f"B2: the {level}-bound write ended the user-wide value")
    assert live == ["Berlin"]


@pytest.mark.parametrize("level", LEVELS)
@known_bugs.xfail("B2")
def test_a_bound_write_leaves_its_siblings_on_the_user_wide_value(level: str) -> None:
    mem = stores.memory()
    user = mem.scope(user="u")
    user.remember("user", "lives_in", "Berlin")
    mem.scope(user="u", **{level: "one"}).remember("user", "lives_in", "Paris")
    sibling = _live(mem.scope(user="u", **{level: "two"}))
    if not sibling and _ended(user) == ["Berlin"]:
        raise known_bugs.Reproduced(f"B2: a sibling {level} lost the user-wide value")
    assert sibling == ["Berlin"]


@pytest.mark.parametrize("level", LEVELS)
def test_a_bound_scope_reads_its_own_value_in_place_of_the_user_wide_one(level: str) -> None:
    """Passes today and must keep passing after the B2 fix: the shadow side of the rule."""
    mem = stores.memory()
    mem.scope(user="u").remember("user", "lives_in", "Berlin")
    mem.scope(user="u", **{level: "one"}).remember("user", "lives_in", "Paris")
    assert _live(mem.scope(user="u", **{level: "one"})) == ["Paris"]


# -- B3: auto-approve misses two read-only tools --------------------------------------

def test_the_approve_hook_differs_from_the_server_only_by_the_known_bug() -> None:
    """Passes today and after the fix for #267. Any other difference between the hook's
    list and the server's read-only tools fails here at once."""
    allowed, read_only = set(approve.READ_ONLY), _read_only_tools()
    assert allowed <= read_only, allowed - read_only
    assert read_only - allowed <= B3_MISSING, read_only - allowed


@known_bugs.xfail("B3")
def test_the_approve_hook_allows_exactly_the_servers_read_only_tools() -> None:
    if _read_only_tools() - set(approve.READ_ONLY) == B3_MISSING:
        raise known_bugs.Reproduced("B3: the approve list misses the two document tools")
    assert set(approve.READ_ONLY) == _read_only_tools()


@pytest.mark.parametrize("tool", sorted(B3_MISSING))
@known_bugs.xfail("B3")
def test_the_approve_hook_allows_the_read_only_document_tools(
        hook_runner: Callable[..., HookRunner], tool: str) -> None:
    result = hook_runner("claude").run("approve", tool_name=f"mcp__memvara__{tool}")
    assert result.exit_code == 0
    if result.reply is None:
        raise known_bugs.Reproduced(f"B3: the approve hook printed no decision for {tool}")
    assert result.reply["hookSpecificOutput"]["permissionDecision"] == "allow"


# -- B4, fixed: deeply nested JSON no longer kills the server ---------------------------

def test_one_deeply_nested_request_does_not_kill_the_server(
        mcp: Callable[..., McpProcess]) -> None:
    """Over the real pipe: the reply is a parse error, and the server still answers."""
    server = mcp()
    server.initialize()
    depth = 100_000
    server.send_raw('{"jsonrpc":"2.0","id":99,"method":"ping","params":'
                    + "[" * depth + "]" * depth + "}")
    reply = server.recv(timeout=20)
    assert "error" in reply and reply.get("id") in (None, 99), reply
    assert server.request("ping") == {}


# -- B5, fixed: memory_standing refuses k below 1 --------------------------------------

def test_memory_standing_with_k_zero_never_reports_an_empty_store(
        mcp: Callable[..., McpProcess]) -> None:
    """Over the real pipe: `k=0` is refused by name, never answered as an empty store."""
    server = mcp()
    server.initialize()
    stored = server.call("memory_remember", subject="user", predicate="prefers",
                         object="tabs for indentation", memory_type="procedural")
    assert not stored.is_error, stored.text
    reply = server.call("memory_standing", k=0)
    assert reply.is_error and "memory_standing.k must be >= 1" in reply.text, reply.text


# -- B6, fixed: a string memory_type ----------------------------------------------------

def test_remember_takes_a_string_memory_type_or_refuses_it_by_name() -> None:
    user = stores.memory().scope(user="u")
    # A string on purpose: it is the spelling the MCP tool accepts.
    user.remember("user", "prefers", "tabs", memory_type="procedural")
    [claim] = user.get_all()
    assert claim.memory_type is MemoryType.PROCEDURAL


# -- B7: a session inside a project reading its own global fact -------------------------

@known_bugs.xfail("B7")
def test_a_session_inside_a_project_reads_the_global_facts_it_writes() -> None:
    """A global predicate clears only the project, so the claim lands at the session with
    no project, and the session's own ancestors never include that scope (#273)."""
    session = stores.memory().scope(user="u", project="github.com/o/a", session="s1")
    [claim] = session.add("I live in Berlin.").added
    seen = [c.id for c in session.get_all()]
    if not seen and (claim.scope.project, claim.scope.session) == (None, "s1"):
        raise known_bugs.Reproduced("B7: the claim sits where the session's reads never look")
    assert seen == [claim.id]
    assert session.why(claim.id) is not None


# -- B9, fixed: a future-dated retraction's tombstone no longer ends before it begins ---

def test_a_future_retraction_leaves_a_tombstone_that_does_not_end_before_it_begins(
        ) -> None:
    """The tombstone's world clock gets the clamp `close_out` gives every other closure,
    so it never ends before its own start (#275)."""
    from harness.clock import FAR_FUTURE

    m = stores.memory()
    user = m.scope(user="u")
    user.remember("user", "likes", "tea")
    user.remember("user", "likes", "tea", polarity=-1, valid_from=FAR_FUTURE)
    [tombstone] = [c for c in m.store.iter_claims(None, True) if c.polarity < 0]
    assert tombstone.valid_from == FAR_FUTURE and tombstone.valid_to is not None
    assert tombstone.valid_to == tombstone.valid_from


# -- B45: a backdated retraction's tombstone is live in reads of the past ----------------

@known_bugs.xfail("B45")
def test_a_backdated_retraction_leaves_a_tombstone_that_no_read_returns() -> None:
    """A tombstone is closed on both clocks at the instant its write is recorded, so it
    can never be live (docs/INTERNALS.md). A retraction backdated with `recorded_at` must
    close it at that instant too, not at the wall clock's (#317)."""
    from datetime import datetime, timezone

    jan, feb, mar = (datetime(2026, month, 1, tzinfo=timezone.utc) for month in (1, 2, 3))
    user = stores.memory().scope(user="u")
    user.remember("user", "likes", "tea", valid_from=jan, recorded_at=jan)
    user.remember("user", "likes", "tea", polarity=-1, valid_from=feb, recorded_at=feb)
    seen = [(c.object, c.polarity) for c in user.get_all(as_of=mar)]
    if seen == [("tea", -1)]:
        raise known_bugs.Reproduced(f"a read at March returned the tombstone: {seen}")
    assert seen == [], seen


# -- B69: a repeated future-dated retraction writes a new tombstone each time ------------

@known_bugs.xfail("B69")
def test_a_future_dated_retraction_sent_again_is_a_repeat() -> None:
    """A repeat of a retraction dated now reinforces its tombstone and reports nothing.
    A repeat of one dated in the future must do the same (#349)."""
    from harness.clock import FAR_FUTURE

    mem = stores.memory()
    user = mem.scope(user="u")
    user.remember("user", "likes", "tea")
    receipts = [user.remember("user", "likes", "tea", polarity=-1, valid_from=FAR_FUTURE)
                for _ in range(3)]
    tombstones = [c for c in mem.store.iter_claims(None, True) if c.polarity < 0]
    reported = [len(r.invalidated) for r in receipts]
    if len(tombstones) == 3 and reported == [1, 1, 1]:
        raise known_bugs.Reproduced(f"{len(tombstones)} tombstones; ended reported {reported}")
    assert len(tombstones) == 1 and reported == [1, 0, 0], (len(tombstones), reported)


# -- B16: forget leaves a scheduled value believed ---------------------------------------

@known_bugs.xfail("B16")
def test_forget_retires_a_scheduled_value_too() -> None:
    """`forget` retires everything the store currently believes in the slot, and a value
    scheduled to start later is believed (#282)."""
    from datetime import timedelta

    from harness.clock import FAR_FUTURE

    user = stores.memory().scope(user="u")
    user.remember("user", "lives_in", "Berlin")
    user.remember("user", "lives_in", "Paris", valid_from=FAR_FUTURE)
    user.forget("user", "lives_in")
    later = [c.object for c in user.get_all(valid_at=FAR_FUTURE + timedelta(days=1))]
    if later == ["Paris"]:
        raise known_bugs.Reproduced("B16: the scheduled value survived forget")
    assert later == []


# -- B17, fixed: a restatement with an earlier start keeps the earlier start -------------

def test_restating_a_fact_with_an_earlier_start_keeps_the_earlier_start() -> None:
    """The earlier start is new information. It must be kept, and what the store believed
    before the restatement must not change (#283)."""
    from harness.clock import INSTANTS

    user = stores.memory().scope(user="u")
    user.remember("user", "likes", "tea", valid_from=INSTANTS[3], recorded_at=INSTANTS[3])
    user.remember("user", "likes", "tea", valid_from=INSTANTS[0], recorded_at=INSTANTS[4])
    now_view = [c.object for c in user.get_all(valid_at=INSTANTS[1])]
    earlier_view = [c.object for c in user.get_all(valid_at=INSTANTS[1],
                                                   known_at=INSTANTS[3])]
    assert now_view == ["tea"]
    assert earlier_view == []


# -- B18: a repeated retraction is folded into an expired tombstone ----------------------

@known_bugs.xfail("B18")
def test_a_retraction_repeated_after_the_first_expired_keeps_its_own_record() -> None:
    """The positive path leaves expired claims out of its duplicate lookup. A retraction
    must too, or the sweep erases the repeat with the tombstone it was folded into
    (#284)."""
    from datetime import timedelta

    from memvara.types import utcnow

    mem = stores.memory()
    mem.remember("user", "likes", "tea", user="u")
    mem.remember("user", "likes", "tea", polarity=-1, user="u",
                 expires_at=utcnow() + timedelta(minutes=5))
    [tombstone] = [c for c in mem.store.iter_claims(None, True) if c.polarity < 0]
    tombstone.expires_at = utcnow() - timedelta(seconds=1)
    mem.store.put_claim(tombstone)
    mem.remember("user", "likes", "tea", polarity=-1, user="u")
    mem.erase_expired()
    left = [c for c in mem.store.iter_claims(None, True) if c.polarity < 0]
    if left == []:
        raise known_bugs.Reproduced("B18: the repeat was erased with the expired tombstone")
    assert len(left) == 1 and left[0].expires_at is None


# -- B47: search() does not return its results in order of score -----------------------

#: A small store in which a claim that matches "C++" is returned after claims that score
#: zero. Found by the upgrade tests in a store written fresh by today's code.
B47_FACTS = (("knows_language", "C"), ("knows_language", "C#"), ("knows_language", "C++"),
             ("visited", "Zurich"), ("visited", "Rome"), ("lives_in", "Lisbon"),
             ("works_at", "Acme Corp"), ("favorite_color", "blue"),
             ("likes_band", "the the band"), ("subscribes_to", "a jazz magazine"),
             ("visited", "Florence"))


@known_bugs.xfail("B47")
def test_search_returns_its_results_best_first() -> None:
    """`Result.score` is a normalized relevance, so the list search() returns is in that
    order, and a claim that matches cannot fall below claims that score zero (#327)."""
    from datetime import datetime, timedelta, timezone

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    mem = stores.memory()
    for day, (predicate, obj) in enumerate(B47_FACTS):
        when = start + timedelta(days=day)
        mem.remember("user", predicate, obj, user="u1", valid_from=when, recorded_at=when)
    results = mem.search("C++", k=10, user="u1")
    order = [(r.claim.object, round(r.score, 3)) for r in results]
    scores = [r.score for r in results]
    if scores != sorted(scores, reverse=True):
        raise known_bugs.Reproduced(f"results out of score order: {order}")
    assert sorted(r.claim.object for r in results[:3]) == ["C", "C#", "C++"], order

