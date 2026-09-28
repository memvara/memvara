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
    KnownBug("B7", 273, "a session bound inside a project cannot read the global facts it "
             "writes"),



    KnownBug("B58", 340, "on Cursor and OpenCode a read-only memvara tool is never approved: "
             "the names measured there do not name the server safely, and an approval path "
             "that does waits for an interactive measurement"),
    # Found by the framework tests against the real packages.
)}


def xfail(bug_id: str) -> pytest.MarkDecorator:
    """The strict expected-failure marker for a registered bug."""
    bug = KNOWN_BUGS[bug_id]
    return pytest.mark.xfail(
        strict=True, raises=Reproduced,
        reason=f"{bug.id}, {bug.repo}#{bug.issue}: {bug.title}")
