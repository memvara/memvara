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
    #: The issue number in memvara/memvara.
    issue: int
    title: str


KNOWN_BUGS: dict[str, KnownBug] = {bug.id: bug for bug in (
    KnownBug("B2", 266, "a session-bound or agent-bound write ends the user-wide value"),
    KnownBug("B3", 267, "auto-approve misses two read-only document tools"),
    KnownBug("B7", 273, "a session bound inside a project cannot read the global facts it "
             "writes"),
    KnownBug("B9", 275, "a retraction dated in the future stores a tombstone that ends "
             "before it begins"),
    KnownBug("B13", 280, "a damaged embedder record lets a same-width embedder change go "
             "unnoticed"),
    KnownBug("B14", 281, "two processes opening a new store at once: one can fail at "
             "startup"),
    KnownBug("B16", 282, "forget() leaves a scheduled value believed"),
    KnownBug("B17", 283, "a restatement with an earlier start loses the earlier start"),
    KnownBug("B18", 284, "a repeated retraction is folded into an expired tombstone"),
    KnownBug("B19", 295, "memory_recall's description names arguments that a feature "
             "switch removes"),
    KnownBug("B20", 296, "two tool descriptions state a default that the schema does not "
             "declare"),
    KnownBug("B21", 297, "the command-line help does not name --version"),

    KnownBug("B22", 299, "the MCP server exits with a traceback on a store from a newer "
             "version"),
    KnownBug("B23", 300, "a store refused for its embedder has already been migrated"),
    KnownBug("B24", 301, "a store refused as too new leaves its SQLite connection open"),
    KnownBug("B47", 327, "search() does not return its results in order of score"),
)}


def xfail(bug_id: str) -> pytest.MarkDecorator:
    """The strict expected-failure marker for a registered bug."""
    bug = KNOWN_BUGS[bug_id]
    return pytest.mark.xfail(
        strict=True, raises=Reproduced,
        reason=f"{bug.id}, memvara/memvara#{bug.issue}: {bug.title}")
