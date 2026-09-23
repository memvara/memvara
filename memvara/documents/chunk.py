r"""Splitting a document into retrieval chunks.

A document is stored as chunks of about 1,000 characters, and each chunk becomes one
episode, so the episode text index and vectors can find the passage a question is about
rather than a whole document. This module decides where the chunks begin and end.

**Boundaries fall between sentences.** The text is split into sentences first, and a
chunk is a run of whole sentences. A sentence is cut only when it is longer than a chunk
on its own; it is then cut at word boundaries, and at the character limit only when a
single word is that long.

**A boundary depends on the text around it, not on its distance from the start.** A
chunker that packed sentences until each chunk was full would place every boundary by
counting from the top of the document, so a word added in the first paragraph would move
every boundary after it, and a re-ingest would find no chunk unchanged. Here a chunk ends
at a *cut point*: a sentence whose own digest falls below a threshold proportional to its
length, which on average puts one every `_MEAN_GAP` characters, and which comes at least
`_MIN_CHARS` after the previous cut point. Cut points are chosen from the sentences alone,
before any chunk is formed. Where a stretch between two cut points is longer than
`CHUNK_CHARS`, a chunk also ends before the sentence that would overflow it; such a
forced split moves no cut point, so its effect stays inside its own stretch. After an
edit, the old and the new text reach the same cut point within a sentence or two and are
chunked identically from there on. So an edit usually changes the chunks it touches and
the one after it. The one after it changes because it begins with the overlap described
next.

**Each chunk repeats the end of the one before it.** Up to `CHUNK_OVERLAP` characters of
whole sentences from the end of the previous chunk are repeated at the start of the
next, so a passage that straddles a boundary can be found from either side. When the
previous chunk's last sentence is longer than the overlap on its own, the overlap is the
end of that sentence, starting at a word.

>>> text = "Memvara stores facts. " * 100
>>> chunks = split(text)
>>> len(chunks) > 1 and all(len(c) <= CHUNK_CHARS + CHUNK_OVERLAP for c in chunks)
True
>>> split("One short sentence.")
['One short sentence.']
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

__all__ = ["CHUNK_CHARS", "CHUNK_OVERLAP", "normalise", "split"]

#: The most characters of its own text one chunk holds. The overlap repeated from the
#: chunk before it comes on top of this.
CHUNK_CHARS = 1000

#: The most characters repeated from the end of one chunk at the start of the next.
CHUNK_OVERLAP = 150

#: A chunk does not end at a cut point before it holds this many characters, so cut
#: points close together do not produce a run of tiny chunks.
_MIN_CHARS = 300

#: The average distance between cut points, in characters.
_MEAN_GAP = 600

#: A sentence ends after terminal punctuation, optionally followed by closing quotes or
#: brackets, when whitespace comes next. A line break always ends one, because a line of
#: a list, a heading or a table is a unit even without a full stop.
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s)|\n")


def normalise(text: str) -> str:
    """The form of the text a document is chunked and hashed from.

    Unicode NFC, so the same visible text typed on two systems hashes the same; every
    line ending as `\\n`; and no whitespace at either end. Nothing inside a line changes.

    >>> normalise("  caf\\u0065\\u0301\\r\\nnext line  ")
    'café\\nnext line'
    """
    text = unicodedata.normalize("NFC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _sentences(text: str) -> list[tuple[int, int]]:
    """`(start, end)` spans of the sentences in `text`, whitespace excluded, in order.

    A sentence longer than `CHUNK_CHARS` comes back as several spans, cut at the last
    whitespace before the limit, or at the limit itself when there is none.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        spans.extend(_trimmed(text, start, m.end()))
        start = m.end()
    spans.extend(_trimmed(text, start, len(text)))
    out: list[tuple[int, int]] = []
    for s, e in spans:
        while e - s > CHUNK_CHARS:
            cut = text.rfind(" ", s + 1, s + CHUNK_CHARS + 1)
            if cut <= s:
                cut = s + CHUNK_CHARS
            out.append((s, cut))
            s = cut
            while s < e and text[s].isspace():
                s += 1
        out.append((s, e))
    return out


def _trimmed(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """The span with surrounding whitespace removed, or nothing if it is all whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return [(start, end)] if end > start else []


def _is_cut_point(sentence: str) -> bool:
    """Whether a chunk may end after this sentence, decided by the sentence alone.

    The chance is proportional to the sentence's length, so the average distance between
    cut points is `_MEAN_GAP` characters whether the text is written in short sentences
    or long ones.
    """
    digest = hashlib.blake2b(sentence.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2 ** 64 < len(sentence) / _MEAN_GAP


def _overlap_start(text: str, spans: list[tuple[int, int]], first: int, last: int) -> int:
    """Where the overlap repeated from the chunk `spans[first:last + 1]` begins."""
    end = spans[last][1]
    start = end
    for i in range(last, first - 1, -1):
        if end - spans[i][0] > CHUNK_OVERLAP:
            break
        start = spans[i][0]
    if start < end:
        return start
    # The last sentence alone is longer than the overlap: take its end, from a word.
    cut = text.find(" ", end - CHUNK_OVERLAP, end)
    return cut + 1 if cut != -1 else end - CHUNK_OVERLAP


def _cut_points(text: str, spans: list[tuple[int, int]]) -> set[int]:
    """The sentences a chunk ends after, chosen from the sentences alone.

    A sentence is a cut point when `_is_cut_point` says so and at least `_MIN_CHARS`
    have passed since the previous cut point. Measured from the previous cut point and
    not from the start of the current chunk, so a forced split between two cut points
    changes neither of them; an edit therefore moves a cut point only through the
    sentences between it and the one before.
    """
    cuts: set[int] = set()
    since = 0
    for i, (start, end) in enumerate(spans):
        if end - spans[since][0] >= _MIN_CHARS and _is_cut_point(text[start:end]):
            cuts.add(i)
            since = i + 1
    return cuts


def split(text: str) -> list[str]:
    """Split normalised text into retrieval chunks. See the module docstring.

    Returns an empty list for text with nothing but whitespace in it.

    >>> split("") == []
    True
    """
    spans = _sentences(text)
    cuts = _cut_points(text, spans)
    groups: list[tuple[int, int]] = []
    first = 0
    for i, (start, end) in enumerate(spans):
        if i > first and end - spans[first][0] > CHUNK_CHARS:
            # A forced split, in a stretch with no cut point for a whole chunk. It
            # decides where this chunk ends and nothing else: the cut points were chosen
            # before it, so it cannot move one.
            groups.append((first, i - 1))
            first = i
        if i in cuts:
            groups.append((first, i))
            first = i + 1
    if first < len(spans):
        groups.append((first, len(spans) - 1))

    chunks: list[str] = []
    for n, (a, b) in enumerate(groups):
        begin = spans[a][0]
        if n:
            pa, pb = groups[n - 1]
            begin = _overlap_start(text, spans, pa, pb)
        chunks.append(text[begin:spans[b][1]])
    return chunks
