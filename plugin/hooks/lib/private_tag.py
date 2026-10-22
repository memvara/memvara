"""Text a person wraps in `<private>...</private>` never reaches memvara through the hooks.

A person sometimes has to show the assistant something that must not be remembered: a
token, a customer's name, a health detail. Capture mines every turn and recall searches
with every prompt, so the tag is the person's way to keep one passage out of both.
`transcript.last_turn_with_context` strips every transcript entry before it is formatted,
cut or joined, and `recall.py` strips the prompt before anything else reads it. The model
the person is talking to still sees the text; only the hooks are bound by the tag, and a
memory tool the model calls itself is not.

The rules, each of which a real prompt needed:

* **A span is removed whole, tags included, and nothing is put in its place.** A
  placeholder would be worse than nothing: the fast extractor believes sentence forms, so
  "my name is [private]" could be stored as a name.
* **Tags nest.** A span ends at the closing tag that matches its opening one, so
  `<private>a <private>b</private> c</private>` hides all three.
* **An opening tag with no closing tag hides the rest of that text.** That fails closed:
  the other reading would store exactly what the person marked. Capture strips one
  transcript entry's strings at a time, so the rest of a prompt is lost, never the reply
  after it.
* **A closing tag with no opening tag is dropped and hides nothing**, since there is no
  telling where its span began.
* **A tag inside backticks, `` `<private>` ``, is a mention and not a tag**, so a person can
  write about the tag without hiding the rest of the prompt.
* **A hidden value repeated elsewhere is removed there too**, because the model reads the
  span and may echo it untagged in a reply or a command. Only values of three characters
  or more are hunted this way; removing every "42" would damage far more than it protects.
  A repeat is found only among the texts stripped together: one prompt, or the transcript
  window capture reads.
"""

from __future__ import annotations

import re
from typing import Any

#: An opening, closing or self-closing tag, with any attributes and spacing, in any case.
#: Not when a backtick stands right beside it: `` `<private>` `` is someone writing about
#: the tag.
_TAG = re.compile(r"(?<!`)<\s*(/)?\s*private\b[^>]*?(/)?\s*>(?!`)", re.IGNORECASE)

#: The shortest hidden value that is also removed where it appears untagged.
MIN_REPEAT_CHARS = 3


def split(text: str) -> "tuple[str, list[str]]":
    """`text` with every span removed, and the text of each span removed."""
    if "<" not in text:
        return text, []
    kept: list[str] = []
    hidden: list[str] = []
    depth = 0
    pos = 0
    span_start = 0
    for tag in _TAG.finditer(text):
        closing, self_closing = tag.group(1), tag.group(2)
        if depth == 0:
            kept.append(text[pos:tag.start()])
        if self_closing and not closing:
            pass  # `<private/>` holds nothing
        elif closing:
            if depth == 1:
                # Inner tags become line breaks, so each nested value is hunted on its own.
                hidden.append(_TAG.sub("\n", text[span_start:tag.start()]))
            depth = max(depth - 1, 0)
        else:
            if depth == 0:
                span_start = tag.end()
            depth += 1
        pos = tag.end()
    if depth:
        hidden.append(_TAG.sub("\n", text[span_start:]))
    else:
        kept.append(text[pos:])
    return "".join(kept), hidden


def scrub(text: str, hidden: "list[str]") -> str:
    """`text` with every repeat of a hidden value removed, ignoring case.

    The whole value and each of its lines are looked for, longest first, so a
    multi-line secret echoed one line at a time is caught too.
    """
    needles = set()
    for value in hidden:
        for piece in [value, *value.splitlines()]:
            piece = piece.strip()
            if len(piece) >= MIN_REPEAT_CHARS:
                needles.add(piece)
    for needle in sorted(needles, key=len, reverse=True):
        text = re.sub(re.escape(needle), "", text, flags=re.IGNORECASE)
    return text


def strip(text: str) -> str:
    """`text` with every span removed, and every repeat of a span's value in it."""
    kept, hidden = split(text)
    return scrub(kept, hidden) if hidden else kept


def strip_entry(value: Any, hidden: "list[str]") -> Any:
    """A copy of one parsed transcript entry with every string in it split.

    The values of the spans removed are appended to `hidden`, so the caller can scrub
    repeats from text assembled out of other entries. Every string is split on its own,
    so an unclosed tag in one field never reaches into the next.
    """
    if isinstance(value, str):
        kept, found = split(value)
        hidden.extend(found)
        return kept
    if isinstance(value, list):
        return [strip_entry(item, hidden) for item in value]
    if isinstance(value, dict):
        return {key: strip_entry(item, hidden) for key, item in value.items()}
    return value
