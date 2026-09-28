"""The bug the scripted sessions found, pinned by a strict expected failure that cites its
issue.

The test states the behaviour the fix must produce. It raises known_bugs.Reproduced only
when it has seen the bug's own symptom, so a different failure in the same test fails
loudly instead of passing for the known bug.
"""

from __future__ import annotations

from typing import Callable

from harness import known_bugs
from harness.stdio import McpProcess

#: A turn that none of the fast path's sentence forms matches. A server with no extraction
#: model therefore stores the turn and extracts no fact from it.
TURN = "The door code for our rental in Porto is 7731."


# -- B73: memory_add says a turn was not stored when only no fact was extracted ----------

@known_bugs.xfail("B73")
def test_memory_add_does_not_say_a_stored_turn_was_not_stored(
        mcp: Callable[..., McpProcess]) -> None:
    """Over the real pipe, with no extraction model: memory_add stores the turn and gives
    its id, and then a note says the turn was not stored. memory_recall with
    include_episodes returns the turn, so the note is wrong. Only the fact was not
    extracted (#353)."""
    server = mcp()
    server.initialize()
    added = server.call("memory_add", text=TURN)
    assert not added.is_error, added.text
    found = server.call("memory_recall", query="rental door code 7731",
                        include_episodes=True)
    assert not found.is_error, found.text
    if "not stored" in added.text and TURN in found.text:
        raise known_bugs.Reproduced(
            "B73: memory_add says the turn was not stored, and memory_recall returns it")
    assert "not stored" not in added.text, added.text
