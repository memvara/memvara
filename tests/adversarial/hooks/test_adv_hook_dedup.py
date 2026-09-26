"""Recall and capture do not repeat themselves within a session.

A memory injected on one turn is still in the conversation on the next, so recall keeps a
record per session of what it injected and does not inject it again
(plugin/hooks/recall.py, "It does not repeat itself"). `Stop` can fire more than once over
one reply, so capture records the size of each transcript it mined and skips one that has
not grown (plugin/hooks/capture.py, "It repeats").
"""

from __future__ import annotations

import json
import pathlib
from typing import Callable

import pytest

from harness import stores
from harness.fakes.cli import FakeClis
from harness.hooks import HookRunner

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
    runner = hooks("claude", stubs=clis,
                   env={"MEMVARA_DB": str(db), "MEMVARA_USER": support.USER})
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
