"""The bug the scripted sessions found, #353, which is fixed. The test that pinned it now
checks the fix.
"""

from __future__ import annotations

from typing import Callable

from harness.stdio import McpProcess

#: A turn that none of the fast path's sentence forms matches. A server with no extraction
#: model therefore stores the turn and extracts no fact from it.
TURN = "The door code for our rental in Porto is 7731."


# -- B73, fixed: memory_add said a turn was not stored when only no fact was extracted ---

def test_memory_add_does_not_say_a_stored_turn_was_not_stored(
        mcp: Callable[..., McpProcess]) -> None:
    """Over the real pipe, with no extraction model: memory_add stores the turn and gives
    its id, and its note used to say the turn was not stored, although memory_recall with
    include_episodes returns it. Only the fact was not extracted (#353)."""
    server = mcp()
    server.initialize()
    added = server.call("memory_add", text=TURN)
    assert not added.is_error, added.text
    found = server.call("memory_recall", query="rental door code 7731",
                        include_episodes=True)
    assert not found.is_error, found.text
    assert TURN in found.text, found.text
    assert "not stored" not in added.text, added.text
    assert "were stored, but extraction recognised no fact in them" in added.text, added.text
