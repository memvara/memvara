"""The parity bugs filed as issues, each pinned by a strict expected failure that cites its
issue.

Each test states what the fix must produce. It raises `known_bugs.Reproduced` only after it
has seen its own bug's exact symptom, and the bug's marker accepts nothing else, so a
different failure in the same test fails the run. The step-by-step comparisons in
`test_adv_parity_library.py` and `test_adv_parity_mcp.py` pin #334 again where their
programs meet it; the tests here show each symptom on its own, in as few calls as it takes.
"""

from __future__ import annotations

import inspect
import re
from datetime import datetime, timezone
from typing import Any

import pytest

from harness import known_bugs, stores
from harness.env import REPO
from harness.fakes.fake_v1 import FakeV1, _receipt
from memvara import Memvara, MemoryType
from memvara.core import ScopedMemvara
from memvara.remote import hydrate
from memvara.remote.api import RemoteMemvara, ScopedRemoteMemvara
from memvara.types import WriteReceipt

from .compare import assert_same, labels, normalise, normalise_text, text_labels, timed
from .test_adv_parity_library import RECEIPT_GAP
from .test_adv_parity_mcp import reply_is_known_334, serving

#: The instant both values of the collapsing pair begin at.
SAME_START = datetime(2025, 6, 1, tzinfo=timezone.utc)


# -- B52, memvara/memvara#334: a hosted receipt drops four lists ------------------------

def _four_writes(mem: Any) -> tuple[list[Any], dict[str, Any]]:
    """Four pairs of writes. Returns every receipt, and the receipt of each pair's second
    write by the list it fills: a value added beside a live one in a slot with no
    cardinality, a weaker value stored beside a stronger one, a value closed at the
    instant it began, and a fact re-filed under another memory type."""
    first = [mem.remember("user", "tagged_with", "gardening"),
             mem.remember("user", "timezone", "Europe/Lisbon", confidence=1.0),
             mem.remember("user", "job_title", "engineer", valid_from=SAME_START),
             mem.remember("user", "likes", "jazz")]
    second = {
        "accumulated": mem.remember("user", "tagged_with", "chess"),
        "disputed": mem.remember("user", "timezone", "Europe/Berlin", confidence=0.1),
        "collapsed": mem.remember("user", "job_title", "manager", valid_from=SAME_START),
        "retyped": mem.remember("user", "likes", "jazz", memory_type=MemoryType.EPISODIC),
    }
    return first + list(second.values()), second


@pytest.fixture(scope="module")
def receipts() -> dict[str, tuple[Any, Any]]:
    """For each list, as the local library's receipt fills it and as the hosted client's
    receipt fills it, each normalised."""
    Written = tuple[list[Any], dict[str, Any]]

    def write_both() -> tuple[Written, Written]:
        local = stores.memory(user="alice")
        try:
            mine = _four_writes(local)
        finally:
            local.close()
        with FakeV1() as fake:
            remote = fake.remote(user="alice")
            try:
                return mine, _four_writes(remote)
            finally:
                remote.close()

    ((mine, mine_by_list), (theirs, theirs_by_list)), run = timed(write_both)
    mine_names, theirs_names = labels(*mine), labels(*theirs)
    return {kind: (normalise(getattr(mine_by_list[kind], kind), mine_names, run=run),
                   normalise(getattr(theirs_by_list[kind], kind), theirs_names, run=run))
            for kind in RECEIPT_GAP}


@pytest.mark.parametrize("kind", RECEIPT_GAP)
@known_bugs.xfail("B52")
def test_a_hosted_receipt_says_what_the_write_did(
        receipts: dict[str, tuple[Any, Any]], kind: str) -> None:
    local, hosted = receipts[kind]
    assert local, f"the local receipt no longer fills {kind}"
    if hosted == []:
        raise known_bugs.Reproduced(f"B52: the hosted receipt's {kind} is empty")
    assert_same(local, hosted, f"the hosted receipt's {kind}")


#: The words each note starts with, by the receipt list it is written from.
NOTES = {
    "accumulated": "note: 1 value(s) landed in a slot that already had live values",
    "disputed": "note: 1 value(s) were stored without replacing what was already there",
    "collapsed": "note: 1 value(s) were closed at the instant they began",
    "retyped": "note: 1 already-known fact(s) were re-filed under the memory_type",
}

#: The same four pairs of writes as `_four_writes`, as `memory_remember` calls. Each call
#: that ends in one of the four outcomes names the list it fills.
_CALLS: tuple[tuple[str | None, dict[str, Any]], ...] = (
    (None, {"predicate": "tagged_with", "object": "gardening"}),
    (None, {"predicate": "timezone", "object": "Europe/Lisbon", "confidence": 1.0}),
    (None, {"predicate": "job_title", "object": "engineer",
            "true_since": "2025-06-01T00:00:00Z"}),
    (None, {"predicate": "likes", "object": "jazz"}),
    ("accumulated", {"predicate": "tagged_with", "object": "chess"}),
    ("disputed", {"predicate": "timezone", "object": "Europe/Berlin", "confidence": 0.1}),
    ("collapsed", {"predicate": "job_title", "object": "manager",
                   "true_since": "2025-06-01T00:00:00Z"}),
    ("retyped", {"predicate": "likes", "object": "jazz", "memory_type": "episodic"}),
)


def _write_all(server: Any) -> dict[str, str]:
    """Every reply to `_CALLS`, by the list the call fills or by its position."""
    server.initialize()
    return {kind or f"write {index}": server.call("memory_remember", subject="user",
                                                  **arguments).text
            for index, (kind, arguments) in enumerate(_CALLS)}


@pytest.fixture(scope="module")
def notes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[list[str], list[str]]]:
    """For each list, the lines of the reply to the write that fills it, from a local
    server and from a server in cloud mode, each normalised."""
    def write_both() -> dict[str, dict[str, str]]:
        with serving(tmp_path_factory.mktemp("parity-notes"),
                     ("in-process", "stdio cloud")) as servers:
            return {surface: _write_all(server) for surface, server in servers.items()}

    replies, run = timed(write_both)
    lines: dict[str, dict[str, list[str]]] = {}
    for surface, texts in replies.items():
        names = text_labels(list(texts.values()))
        lines[surface] = {kind: normalise_text(text, names, run=run).split("\n")
                          for kind, text in texts.items()}
    return {kind: (lines["in-process"][kind], lines["stdio cloud"][kind]) for kind in NOTES}


@pytest.mark.parametrize("kind", sorted(NOTES))
@known_bugs.xfail("B52")
def test_cloud_mode_writes_the_note_a_local_server_writes(
        notes: dict[str, tuple[list[str], list[str]]], kind: str) -> None:
    local, cloud = notes[kind]
    assert any(row.startswith(NOTES[kind]) for row in local), (
        f"the local server no longer writes the {kind} note")
    if reply_is_known_334(local, cloud):
        raise known_bugs.Reproduced(f"B52: cloud mode leaves out the {kind} note")
    assert_same(local, cloud, f"the reply that fills {kind}")


# -- B53, memvara/memvara#335: a hosted receipt says 0 ungrounded and 0 polluted -----------

@known_bugs.xfail("B53")
def test_a_hosted_receipt_carries_the_ungrounded_and_polluted_counts() -> None:
    """A write that refused claims an extraction model proposed says how many: in
    `ungrounded`, those with no support in the turn they cite, and in `polluted`, a real
    value in the wrong slot. FakeV1 sends both counts, as memvara-cloud's renderer does,
    so the hosted client should read them back. Only a model proposes such claims, so this
    starts from a receipt rendered by the fake's own renderer."""
    sent = _receipt(WriteReceipt(ungrounded=2, polluted=1), "fast-path-only")
    assert (sent["ungrounded"], sent["polluted"]) == (2, 1), "the fake no longer sends them"
    read = hydrate.receipt(sent)
    if (read.ungrounded, read.polluted) == (0, 0):
        raise known_bugs.Reproduced("B53: the hosted receipt says 0 ungrounded, 0 polluted")
    assert (read.ungrounded, read.polluted) == (2, 1)


# -- B54, memvara/memvara#336: undocumented methods only one client has ----------------

#: The methods #336 is about: one client has them, and the section of docs/API.md about a
#: hosted deployment does not name them.
UNNAMED_336 = frozenset({"end", "health", "whoami", "bind", "merge_predicate"})


def _public(cls: type) -> set[str]:
    """The methods a caller can reach on an instance of `cls`, by name. A classmethod
    such as `Memvara.connect` is reached on the class whatever you hold, so it is left
    out."""
    return {name for name in dir(cls) if not name.startswith("_")
            and callable(getattr(cls, name, None))
            and not isinstance(inspect.getattr_static(cls, name, None), classmethod)}


def _one_sided() -> set[str]:
    """Every public method one client has and the other lacks, over the two pairs a caller
    can hold in each other's place: the two clients, and their scoped views."""
    pairs = ((Memvara, RemoteMemvara), (ScopedMemvara, ScopedRemoteMemvara))
    return set().union(*(_public(one) ^ _public(other) for one, other in pairs))


def _named_in_the_hosted_section(text: str | None = None) -> set[str]:
    """Every method docs/API.md names, as `name()`, in its section on a hosted deployment.
    `text` stands in for the file, for a test of how the section is found."""
    if text is None:
        text = (REPO / "docs" / "API.md").read_text(encoding="utf-8")
    after = text.split("### A hosted deployment", 1)[1]
    # The section ends at the next heading of its own level or a higher one; a level-four
    # heading is part of it.
    section = re.split(r"\n#{1,3} ", after, maxsplit=1)[0]
    return set(re.findall(r"`(\w+)\(\)`", section))


def test_the_hosted_section_ends_at_the_next_heading_of_its_level_or_higher() -> None:
    text = ("# API\n### A hosted deployment\n`one()`\n#### In depth\n`two()`\n"
            "## Something else\n`three()`\n### And more\n`four()`\n")
    assert _named_in_the_hosted_section(text) == {"one", "two"}


@known_bugs.xfail("B54")
def test_the_documentation_names_every_method_only_one_client_has() -> None:
    """docs/API.md says what a hosted deployment lacks and what it adds, so a caller can
    tell which calls to guard. A method only one client has, which that section does not
    name, fails a caller who swaps one client for the other with no warning."""
    unnamed = _one_sided() - _named_in_the_hosted_section()
    if unnamed == UNNAMED_336:
        raise known_bugs.Reproduced(f"B54: docs/API.md does not name {sorted(unnamed)}")
    assert unnamed == set(), sorted(unnamed)
