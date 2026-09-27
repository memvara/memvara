"""The recall hook reads at most `MAX_PROMPT_CHARS` of a prompt (#348).

The store's time grows with the query, so a pasted log file of a few megabytes ran past
the host's 10-second limit. The nightly timeout test runs the hook on a 16 MB prompt; these
check what the cap keeps.
"""

from __future__ import annotations

import pathlib
import sys

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import recall  # noqa: E402


def test_a_prompt_within_the_cap_is_read_whole() -> None:
    prompt = "x" * recall.MAX_PROMPT_CHARS
    assert recall._bounded(prompt) is prompt


def test_a_longer_prompt_keeps_its_first_and_last_half_of_the_cap() -> None:
    """A question is at the start or the end of what someone pastes, so both ends stay."""
    half = recall.MAX_PROMPT_CHARS // 2
    prompt = "where do I live? " + "log line\n" * 1_000_000 + " what did we decide?"
    bounded = recall._bounded(prompt)
    assert bounded.startswith("where do I live?")
    assert bounded.endswith("what did we decide?")
    assert len(bounded) == 2 * half + 1
