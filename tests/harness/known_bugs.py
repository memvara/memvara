"""Bugs the adversarial suite has found and not yet fixed, one entry per GitHub issue.

A test that reproduces one of these carries `xfail("B2")`, a strict expected failure
that cites the issue. The test raises `Reproduced` only after it has seen that bug's own
symptom, and the marker accepts nothing else:

* A different failure in the same test is a real failure, even an AssertionError or an
  exception of the same type the bug raises, so a new bug cannot hide behind a known one.
* When the fix lands, the test passes, and strict mode fails the run until the fix PR
  removes the marker, so a marker cannot outlive its bug.

Security-class findings are not listed here. They follow SECURITY.md, and their tests
land together with their fixes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


class Reproduced(Exception):
    """A known bug's own symptom, seen by the test that pins it."""


@dataclass(frozen=True)
class KnownBug:
    id: str
    #: The issue's number in `repo`.
    issue: int
    title: str
    #: The repository the issue is in. The nightly run files its issues in the private
    #: memvara/build-health, so an entry it pins names that repository here; the entries
    #: filed by hand before that are in memvara/memvara.
    repo: str = "memvara/memvara"


KNOWN_BUGS: dict[str, KnownBug] = {bug.id: bug for bug in (
    KnownBug("B2", 266, "a session-bound or agent-bound write ends the user-wide value"),
    KnownBug("B3", 267, "auto-approve misses two read-only document tools"),
    KnownBug("B7", 273, "a session bound inside a project cannot read the global facts it "
             "writes"),
    KnownBug("B19", 295, "memory_recall's description names arguments that a feature "
             "switch removes"),
    KnownBug("B20", 296, "two tool descriptions state a default that the schema does not "
             "declare"),

    KnownBug("B23", 300, "a store refused for its embedder has already been migrated"),
    KnownBug("B24", 301, "a store refused as too new leaves its SQLite connection open"),

    KnownBug("B39", 316, "memory_recall reports ranked without include_episodes as a "
             "ValueError"),

    KnownBug("B26", 303, "malformed model output makes add() raise instead of dropping the "
             "item"),
    KnownBug("B28", 305, "a claim the trust boundary drops still costs a model call and "
             "leaves a learned predicate"),
    KnownBug("B29", 306, "a claim with no subject is filed under the user, and a list "
             "object is stored as Python text"),
    KnownBug("B30", 307, "a model retraction at low confidence ends a fact the user "
             "asserted"),
    KnownBug("B32", 309, "invented predicates past the learned cap make one write take "
             "quadratic time"),
    KnownBug("B45", 317, "a backdated retraction's tombstone is live in reads of the past"),
    KnownBug("B69", 349, "a repeated future-dated retraction writes a new tombstone each "
             "time"),
    KnownBug("B46", 318, "tier 0 of add() reinforces a restatement dated before the claim, "
             "and the earlier period is lost"),
    KnownBug("B70", 351, "a different value written twice for a period before the "
             "current one is stored twice"),
    KnownBug("B56", 338, 'on Codex, Copilot and OpenCode, "not configured" and "no matching '
             'memories" look identical'),
    KnownBug("B58", 340, 'the "_" approve separator never recovers a tool name on Cursor and '
             "OpenCode"),
    KnownBug("B59", 341, "the hooks never find a store configured in Codex's, Cursor's or "
             "OpenCode's own MCP config"),
    KnownBug("B64", 346, "a payload nested 100,000 levels deep crashes every hook body"),
    KnownBug("B52", 334, "a hosted write's receipt drops accumulated, disputed, collapsed "
             "and retyped, so cloud-mode memory_remember leaves out four notes"),
    KnownBug("B53", 335, "a hosted receipt always reports 0 for ungrounded and polluted"),
    KnownBug("B50", 332, "add() drops a turn repeated word for word after the value it "
             "stated has changed"),
    KnownBug("B73", 353, "memory_add says a turn was not stored when only no fact was "
             "extracted from it"),
    # Found by the framework tests against the real packages.
    KnownBug("B80", 359, "the mem0 shim's add() and delete_all() refuse the entity ids "
             "mem0 2.x takes there, and search() and get_all() refuse them with TypeError "
             "where mem0 raises ValueError"),
    KnownBug("B81", 360, "the mem0 shim lacks close(), the with statement, arguments mem0 "
             "2.x's methods take, and the score key of mem0's get() row"),
    KnownBug("B82", 361, "the mem0 shim's search() and get_all() default to top_k 10 and "
             "100, where mem0 2.x defaults to 20"),
)}


def xfail(bug_id: str) -> pytest.MarkDecorator:
    """The strict expected-failure marker for a registered bug."""
    bug = KNOWN_BUGS[bug_id]
    return pytest.mark.xfail(
        strict=True, raises=Reproduced,
        reason=f"{bug.id}, {bug.repo}#{bug.issue}: {bug.title}")
