"""Recall and capture do not repeat themselves within a session.

A memory injected on one turn is still in the conversation on the next, so recall keeps a
record per session of what it injected and does not inject it again
(plugin/hooks/recall.py, "It does not repeat itself"). `Stop` can fire more than once over
one reply, so capture records the size of each transcript it mined and skips one that has
not grown (plugin/hooks/capture.py, "It repeats").

The first prompt of a session injects the standing preferences that session start has
just injected, which B61 (#343) pins.
"""

from __future__ import annotations

import json
import pathlib
from typing import Callable

import pytest

from harness import known_bugs, stores
from harness.fakes.cli import FakeClis
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
@known_bugs.xfail("B61")
def test_the_standing_preferences_are_injected_once_when_a_session_opens(
        hooks: Make, tmp_path: pathlib.Path, host: str) -> None:
    """Recall re-checks the standing preferences every 15 minutes (recall.py,
    `_standing_refresh`), comparing a digest with the one it recorded for the session.
    Session start injects them and records no digest, so the first prompt of every session
    finds the check due and the digest different, and injects the whole block again."""
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
