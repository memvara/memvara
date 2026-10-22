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
    "Without memvara: what is a monad?",
    "Without your memories, how would you set this up?",
    "Don't use memvara for this one",
    "Don\u2019t use your memory here",
    "Do not use memories. Explain recursion",
    "Ignore your memory and start fresh",
    "Skip memvara this time",
    "Don't use my memory for this",
    "do not use any memories, explain recursion",
    "Ignoring your stored memories, what would you do?",
    "  without memory \u2014 what is 2+2?",
])
def test_a_prompt_that_opens_by_asking_for_no_memory_is_opted_out(prompt: str) -> None:
    assert recall._opted_out(prompt)


@pytest.mark.parametrize("prompt", [
    # RAM, at the start of a prompt, where the phrase would otherwise be read.
    "without memory limits, run the tests",
    "Without using memory allocation, rewrite this loop",
    "Don't use memory for caching here",
    "don't use your memory allocator in this module",
    "Ignore your memory model assumptions and check the atomics",
    "Don't check your memory usage, it is fine",
    "don't use memory-mapped files here",
    "Without using memory-mapped IO, how would you read it?",
    "ignore memory warnings from the profiler",
    # The memvara repository's own vocabulary: a table, a setup step, a package.
    "skip the memories table in the migration",
    "ignore memories older than 30 days",
    "Skip memvara setup and run the tests",
    "Without memvara installed, what happens?",
    # Asking for memory, or asking about it, in words that contain an opt-out phrase.
    "don't ignore your memory of the deploy, check it first",
    "why did you ignore your memory about the billing database?",
    "stop ignoring your memories and tell me what we decided",
    "never skip memvara on a prompt about deploys",
    "what happens if you ignore your memory?",
    # The phrase has to open the prompt, so text quoted or pasted below never switches
    # recall off, and neither does an instruction left at the end.
    "here is the code:\n// allocate this buffer without using memory from the pool",
    "what is a monad? answer without using memory",
    # Nothing to do with the opt-out.
    "why is there a memory leak in the daemon",
    "what do you remember about my deploy",
    "add a without_memory flag to the bench",
    "",
])
def test_a_prompt_that_does_not_open_by_asking_for_no_memory_is_not_opted_out(
        prompt: str) -> None:
    assert not recall._opted_out(prompt)
