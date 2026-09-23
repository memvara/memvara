"""The mark on every memory line the hooks inject: `⋈ ` at the start of the line.

A reader of the conversation, person or model, can then tell a recalled memory from
everything else in the context. The mark goes in front of the whole bullet, so a memory the
server renders as `- billing uses postgres` is injected as `⋈ - billing uses postgres`.
Headers and notes such as "3 further standing notes did not fit" are not marked; only the
memories are.

Two rules keep the mark from changing anything else:

- **Deduplication ignores it.** The recall hook hashes each line to avoid injecting it
  twice in one session, and it hashes the line *without* the mark. A session that was
  running before this change keeps its record of what it has already seen.
- **Capture drops it.** `lib.transcript` removes every line that starts with the mark
  before a turn is mined, so recalled memory is never extracted and stored a second time.
  That is the failure this exists to prevent: a recall block read back as conversation
  manufactures a duplicate of every fact in it.

The mark can be switched off with the `recall_mark` setting. Capture drops marked lines
whether or not the switch is on, because a transcript can hold blocks injected before the
switch changed.
"""

from __future__ import annotations

from .ipc import MARK as GLYPH
from .settings import enabled

#: The switch in `~/.memvara/settings.json` that turns the mark off.
FEATURE = "recall_mark"

#: What every injected memory line starts with.
MARK = f"{GLYPH} "

#: How the server and the local library render one memory.
BULLET = "- "


def on() -> bool:
    """Whether injected memory lines should carry the mark."""
    return enabled(FEATURE)


def unmarked(line: str) -> str:
    """`line` without a leading mark. Leaves an unmarked line alone."""
    return line[len(MARK):] if line.startswith(MARK) else line


def marked(line: str, mark: bool = True) -> str:
    """`line` with the mark in front, once. Returns `line` unchanged when `mark` is false."""
    if not mark or line.startswith(MARK):
        return line
    return MARK + line


def is_memory(line: str) -> bool:
    """Whether `line` is one injected memory, marked or not."""
    return unmarked(line).startswith(BULLET)


def mark_block(text: str, mark: bool = True) -> str:
    """Put the mark on every memory line of a block. Headers and notes are left alone.

    Uses `is_memory` to decide which lines are memories, the same test `count` and capture
    use, so the three cannot drift apart. Safe to apply twice, because `marked` does not
    mark a line that already carries the mark.
    """
    if not mark or not text:
        return text
    return "\n".join(marked(line) if is_memory(line) else line for line in text.split("\n"))


def unmark_block(text: str) -> str:
    """The block with the mark removed from every line, for hashing what it says."""
    return "\n".join(unmarked(line) for line in text.split("\n"))


def count(text: str) -> int:
    """How many memory lines a block holds."""
    return sum(1 for line in text.splitlines() if is_memory(line))
