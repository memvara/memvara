"""A prompt that asks for no memory gets none from the recall hook.

People sometimes want one answer without their stored memory in it: "without using memory,
what is ...". The recall hook runs on every prompt whether or not anyone asked, so the
person needs a way to stand it down for one prompt. Supermemory measured the same need and
put a decision model in its hook (blog post of 2026-09-24); a short list of phrases does
the job here with no model call and no added latency.

The phrases name memory as the assistant's memory: "your memory", "memories", "memvara",
"using memory". A bare "memory" is not enough, because a coding prompt says "memory" about
RAM all the time, and "run the tests without memory limits" must still be answered with
what is stored. The end-to-end check, that an opted-out prompt sends no request at all and
says so on the status line, is in `test_hook_recall_requests.py`, beside the fake server.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import recall  # noqa: E402


@pytest.mark.parametrize("prompt", [
    "without using memory, what is the capital of France?",
    "Without using your memory, summarise this file",
    "answer without memvara",
    "answer this without your memories",
    "don't use memvara for this one",
    "Don’t use your memory here",
    "do not use memories: explain recursion",
    "ignore your memory and start fresh",
    "skip memvara this time",
    "without stored memories, how would you set this up?",
])
def test_a_prompt_that_asks_for_no_memory_is_opted_out(prompt: str) -> None:
    assert recall._opted_out(prompt)


@pytest.mark.parametrize("prompt", [
    "run the tests without memory limits",
    "why is there a memory leak in the daemon",
    "don't use memory-mapped files here",
    "without using memory-mapped IO, how would you read it?",
    "ignore memory warnings from the profiler",
    "what do you remember about my deploy",
    "add a without_memory flag to the bench",
    "",
])
def test_a_prompt_about_ram_or_about_memory_itself_is_not_opted_out(prompt: str) -> None:
    assert not recall._opted_out(prompt)
