"""The part of a long turn a reader should see: the window that matches the question.

`recall()` shows a turn in at most `Memvara.RECALL_EPISODE_CHARS` characters, because a
turn is whatever someone said or pasted and an uncut one can fill the whole prompt. It
used to show the first 280 characters. In a long turn that is the part least likely to
hold the answer: an assistant's reply opens with a greeting and a restatement, and the
detail somebody later asks about is in the middle.

This module picks the window instead. The turn is split into sentences, each sentence is
scored by how many of the question's content words it contains, and the best one is shown
together with as many of its neighbours as fit — the ones after it first, because an
answer usually follows the sentence that names the topic. `bench/recall_window.py`
renders one search both ways for each of the 470 answerable LongMemEval-S questions, one
store per question. The gold answer's words reached the 4,000-character context for 44.9%
of the questions with the first 280 characters, and for 50.4% with this window.

Words are compared through `schema.word_stem`, the suffix stripper the intent classifier
already uses, so "adopted" in a question finds "adopt" in a turn. When no sentence shares a
word with the question, the result is exactly the old cut: the first `limit - 1` characters
and an ellipsis. So a turn is never rendered worse than before, only differently when the
question names something in it.

The function is pure: the same text, question and limit always give the same window.
"""

from __future__ import annotations

import re

from ..schema import word_stem
from .analyze import MIN_TERM_CHARS, analyze

__all__ = ["ELLIPSIS", "excerpt"]

#: Marks the side of a window where text was left out.
ELLIPSIS = "…"

#: A sentence ends at terminal punctuation followed by whitespace. The text arrives
#: flattened to one line (see `Memvara._safe_line`), so a line break is no longer there to
#: end one.
_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")

#: A run of the characters `str.isalnum` accepts, which is exactly what `\w` matches
#: less the underscore. The same words `analyze.tokenize` splits out, found by the
#: regular expression engine instead of that function's loop over every character, which
#: took half of the time spent on a long turn.
_WORD = re.compile(r"[^\W_]+")


def excerpt(text: str, query: str, limit: int) -> str:
    """The window of `text`, at most `limit` characters, that best matches `query`.

    `text` is returned unchanged when it fits. Otherwise the result is a run of whole
    sentences around the sentence that shares the most content words with `query`, with
    `ELLIPSIS` on each side where text was left out, and never longer than `limit`.

    >>> turn = ("Happy to help with the move! Packing tips first: label every box. "
    ...         "About your dog: greyhounds need a quiet room on moving day.")
    >>> excerpt(turn, "what breed is my dog", 80)
    '…About your dog: greyhounds need a quiet room on moving day.'

    A question that names nothing in the turn gets the head of it, as before:

    >>> excerpt(turn, "what is the capital of France", 40)
    'Happy to help with the move! Packing ti…'

    A sentence longer than `limit` on its own is cut around the first word it shares
    with the question:

    >>> long_one = "word " * 60 + "kayak rental receipt " + "word " * 60
    >>> excerpt(long_one, "the kayak receipt", 50)
    '…word word kayak rental receipt word word word…'
    """
    if len(text) <= limit:
        return text
    stems = {word_stem(t) for t in analyze(query).terms}
    if not stems:
        return _head(text, limit)
    spans = _sentences(text)
    folded: dict[str, str] = {}
    scores = [_shared(text[a:b], stems, folded) for a, b in spans]
    best = max(range(len(spans)), key=lambda i: (scores[i], -i))
    if scores[best] == 0:
        return _head(text, limit)
    a, b = spans[best]
    if _fits(text, a, b, limit):
        lo = hi = best
        grew = True
        while grew:
            grew = False
            if hi + 1 < len(spans) and _fits(text, spans[lo][0], spans[hi + 1][1], limit):
                hi += 1
                grew = True
            if lo > 0 and _fits(text, spans[lo - 1][0], spans[hi][1], limit):
                lo -= 1
                grew = True
        return _framed(text, spans[lo][0], spans[hi][1])
    return _around_first_hit(text, a, b, stems, limit)


def _sentences(text: str) -> list[tuple[int, int]]:
    """Start and end offsets of each sentence in `text`, in order, none empty."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(a, b) for a, b in spans if b > a]


def _shared(sentence: str, stems: set[str], folded: dict[str, str]) -> int:
    """How many of the question's word stems appear in `sentence`.

    `folded` maps each word already seen in this turn to its stem, so a word is stemmed
    once per turn rather than once per occurrence. With `_WORD`, that took a 1 MB turn
    from 200 ms to 76 ms, and a 100 KB one from 19 ms to 7.5 ms.
    """
    found: set[str] = set()
    for word in _WORD.findall(sentence.lower()):
        if len(word) >= MIN_TERM_CHARS:
            stem = folded.get(word)
            if stem is None:
                stem = folded[word] = word_stem(word)
            found.add(stem)
    return len(stems & found)


def _fits(text: str, a: int, b: int, limit: int) -> bool:
    """Whether `text[a:b]`, with an ellipsis on each cut side, is at most `limit`."""
    return (b - a) + (a > 0) + (b < len(text)) <= limit


def _framed(text: str, a: int, b: int) -> str:
    """`text[a:b]` with an ellipsis on each side where something was left out."""
    return (ELLIPSIS if a > 0 else "") + text[a:b] + (ELLIPSIS if b < len(text) else "")


def _head(text: str, limit: int) -> str:
    """The cut `recall()` has always made: the first `limit - 1` characters and `…`."""
    return text[:limit - 1].rstrip() + ELLIPSIS


def _around_first_hit(text: str, a: int, b: int, stems: set[str], limit: int) -> str:
    """A `limit`-character window of one long sentence, placed at its first match.

    A quarter of the window goes before the matching word, so the reader sees what led up
    to it and more of what follows. A window that the end of the sentence cuts short
    reaches back instead, so the room is used either way. Both ends move to a word
    boundary, and the window never leaves the sentence it was cut from.
    """
    hit = a
    for m in _WORD.finditer(text, a, b):
        if word_stem(m.group().lower()) in stems:
            hit = m.start()
            break
    room = limit - 2
    end = min(b, max(a, hit - room // 4) + room)
    start = max(a, end - room)
    if start > a and not text[start - 1].isspace():
        space = text.find(" ", start, hit)
        start = space + 1 if space >= 0 else start
    if end < b and not text[end].isspace():
        space = text.rfind(" ", start, end)
        end = space if space > start else end
    return _framed(text, start, end)
