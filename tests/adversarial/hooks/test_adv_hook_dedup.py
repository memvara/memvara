"""Recall and capture do not repeat themselves within a session.

A memory injected on one turn is still in the conversation on the next, so recall keeps a
record per session of what it injected and does not inject it again
(plugin/hooks/recall.py, "It does not repeat itself"). `Stop` can fire more than once over
one reply, so capture records the size of each transcript it mined and skips one that has
not grown (plugin/hooks/capture.py, "It repeats").

Session start records the standing preferences it injects, so the first prompt of the
session does not inject them again (#343).
"""

from __future__ import annotations

import json
import pathlib
from typing import Callable

import pytest

from harness import known_bugs, stores
from harness.fakes.cli import FakeClis
from harness.fakes.hosted_mcp import FakeHostedMcp
from harness.hooks import HookRunner
from memvara import MemoryType

from . import support

Make = Callable[..., HookRunner]

#: A second turn, and the fake extractor's answer to it.
SECOND_TURN = ("Please remember that I work at Acme Robotics now.",
               "Noted: you work at Acme Robotics.")
SECOND_REPLY = json.dumps({"proposals": [
    {"kind": "fact", "subject": "user", "predicate": "works_at", "object": "Acme Robotics"}]})


@pytest.mark.parametrize("host", ("claude", "copilot"))
def test_recall_injects_a_memory_once_per_session(hooks: Make, store_env: dict[str, str],
                                                   host: str) -> None:
    runner = hooks(host, env=store_env)
    first = runner.run("recall", session="one", prompt=support.PROMPT)
    again = runner.run("recall", session="one", prompt=support.PROMPT)
    other = runner.run("recall", session="two", prompt=support.PROMPT)
    assert support.MEMORY in support.context_of(host, first.reply)
    assert support.context_of(host, again.reply) == ""
    assert support.MEMORY in support.context_of(host, other.reply)
    if host == "claude":
        assert support.status_of(host, again.reply) == "⋈ Memvara · 1 already in context"


def test_capture_mines_a_turn_once_however_often_stop_fires(
        hooks: Make, clis: FakeClis, tmp_path: pathlib.Path) -> None:
    db = support.make_store(tmp_path / "capture.db", memory=False)
    clis.script("claude", support.PROPOSALS_REPLY, SECOND_REPLY)
    runner = hooks("claude", stubs=clis, env=support.store_env(db))
    turn = (support.USER_TURN, support.ASSISTANT_TURN)
    transcript = support.write_transcript("claude", tmp_path / "t.jsonl", [turn])
    for _ in range(2):
        runner.run("capture", session="s", transcript_path=str(transcript))
    assert len(clis.calls("claude")) == 1
    support.write_transcript("claude", transcript, [turn, SECOND_TURN])
    runner.run("capture", session="s", transcript_path=str(transcript))
    assert len(clis.calls("claude")) == 2
    with stores.file(db) as mem:
        facts = sorted(claim.object for claim in mem.scope(user=support.USER).get_all())
    assert facts == ["Acme Robotics", "Lisbon"]


# -- known bugs --------------------------------------------------------------------------

#: A standing preference: a procedural memory, which session start injects.
PREFERENCE = "tabs for indentation in every file of every project"


@pytest.mark.parametrize("host", ("claude", "opencode"))
def test_the_standing_preferences_are_injected_once_when_a_session_opens(
        hooks: Make, tmp_path: pathlib.Path, host: str) -> None:
    """Recall re-checks the standing preferences every 15 minutes (recall.py,
    `_standing_refresh`), comparing a digest with the one recorded for the session. When
    session start recorded no digest, the first prompt of every session found the check due
    and the digest different, and injected the whole block again (#343)."""
    db = tmp_path / "standing.db"
    with stores.file(db) as mem:
        mem.scope(user=support.USER).remember("user", "prefers", PREFERENCE,
                                              memory_type=MemoryType.PROCEDURAL)
    runner = hooks(host, env=support.store_env(db))
    opened = support.context_of(host, runner.run("session_start", session="one").reply)
    assert support.STANDING_WORDS in opened and PREFERENCE in opened, opened
    first = runner.run("recall", session="one", prompt=support.UNRELATED)
    context = support.context_of(host, first.reply)
    if support.STANDING_WORDS in context and PREFERENCE in context:
        raise known_bugs.Reproduced(
            f"B61: the first prompt of a session on {host} injects the standing preferences "
            f"that session start injected")
    assert support.STANDING_WORDS not in context


def _recorded(runner: HookRunner, session: str) -> dict:
    """What session start recorded for `session` in the recall state, or {}."""
    path = runner.home / ".memvara" / ".hooks" / "recalled" / f"{session}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def test_session_start_records_nothing_on_a_host_without_recall(
        hooks: Make, tmp_path: pathlib.Path) -> None:
    """Cursor runs no recall, and recall is what prunes the state files, so a record session
    start wrote there would stay forever. Nothing on Cursor would ever read it."""
    db = tmp_path / "standing.db"
    with stores.file(db) as mem:
        mem.scope(user=support.USER).remember("user", "prefers", PREFERENCE,
                                              memory_type=MemoryType.PROCEDURAL)
    runner = hooks("cursor", env=support.store_env(db))
    opened = support.context_of("cursor", runner.run("session_start", session="one").reply)
    assert PREFERENCE in opened, opened
    assert _recorded(runner, "one") == {}


def test_a_standing_block_from_the_legacy_fallback_is_not_recorded(hooks: Make) -> None:
    """When memory_standing and memory_since both fail, session start falls back to a
    ranked read of the preferences, which recall's refresh never uses. Recording that
    block's digest would make the next refresh find the full block different and inject it
    again as "updated"."""
    with FakeHostedMcp() as fake:
        fake.memvara.scope(user=fake.user).remember("user", "prefers", PREFERENCE,
                                                    memory_type=MemoryType.PROCEDURAL)
        for route in ("tools/call memory_standing", "tools/call memory_since"):
            fake.fail(route, 500)
        runner = hooks("claude", env={"MEMVARA_API_KEY": fake.api_key,
                                      "MEMVARA_SERVER_URL": fake.serve()})
        opened = support.context_of("claude", runner.run("session_start", session="one").reply)
    assert PREFERENCE in opened, opened
    assert "standing" not in _recorded(runner, "one")
