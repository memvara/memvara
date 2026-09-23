"""Cutting a long turn into pieces for extraction, at sentence boundaries.

`WritePipeline(extraction_chunks=True)` sends a turn longer than `EXTRACTION_CHUNK_CHARS`
to the model one piece at a time instead of in one call. This module decides where the
pieces start and end. It does nothing else: the claims each piece yields still cite the
whole episode, and the pipeline merges repeats across pieces before reconciliation.

A piece holds whole paragraphs where it can, and whole sentences otherwise. A paragraph
(text between blank lines) goes into a piece intact if it fits in one; a paragraph too
long for a piece is cut after a sentence (`.`, `!` or `?` followed by whitespace) or at a
line break, so a fact stated in one sentence is never split across two model calls. A
single sentence longer than a whole piece is the only thing cut mid-sentence, at the last
space that fits, or at the limit when it has no space at all.

This is a separate splitter from the one that cuts documents for retrieval. The two have
different jobs: a retrieval chunk is small and overlaps its neighbours so a search can
land on it, and an extraction piece is as large as the limit allows, with no overlap,
because an overlap would make the model state the facts in it twice.
"""

from __future__ import annotations

import re

#: A turn longer than this many characters is extracted in pieces when
#: `extraction_chunks` is on, and no piece is longer than this. The design names 6,000
#: characters for the threshold (`docs/superpowers/specs/
#: 2026-09-23-parity-phase-2-documents-and-retrieval-design.md`, section 4.3), and one
#: number serves as both the threshold and the piece size so that a turn just over the
#: line becomes two pieces rather than one full piece and a scrap.
EXTRACTION_CHUNK_CHARS = 6000

#: A blank line, with the whitespace around it. A paragraph ends at the end of a match.
_PARAGRAPH = re.compile(r"\n[ \t]*\n\s*")

#: The end of a sentence, with any closing quote or bracket, and the whitespace after it;
#: or a line break. A sentence ends at the end of a match.
_SENTENCE = re.compile(r"[.!?][\"')\]]*\s+|\n\s*")


def split_for_extraction(text: str, limit: int | None = None) -> list[str]:
    r"""Cut `text` into pieces of at most `limit` characters, at paragraph or sentence ends.

    `limit` defaults to `EXTRACTION_CHUNK_CHARS`, read when the function is called. A
    text no longer than the limit comes back whole and unchanged, as a list of one.
    Otherwise paragraphs, or the sentences of a paragraph too long for one piece, are
    packed into each piece in order until the next one would not fit, and each piece is
    returned with its surrounding whitespace removed.

    >>> split_for_extraction("Short turn.", limit=100)
    ['Short turn.']
    >>> split_for_extraction("I live in Porto. I work at Acme. My dog is Rex.", limit=34)
    ['I live in Porto. I work at Acme.', 'My dog is Rex.']

    A paragraph that fits in a piece is not split to fill the piece before it:

    >>> split_for_extraction("Alpha one. Alpha two.\n\nBeta one. Beta two.", limit=30)
    ['Alpha one. Alpha two.', 'Beta one. Beta two.']

    A sentence longer than the limit is cut at the last space that fits:

    >>> split_for_extraction("one two three four five six", limit=10)
    ['one two', 'three four', 'five six']
    """
    size = EXTRACTION_CHUNK_CHARS if limit is None else limit
    if size < 1:
        raise ValueError(f"limit must be at least 1 character, not {size}")
    if len(text) <= size:
        return [text]

    units: list[str] = []
    for paragraph in _spans(_PARAGRAPH, text):
        units.extend([paragraph] if len(paragraph) <= size
                     else _spans(_SENTENCE, paragraph))

    pieces: list[str] = []
    current = ""
    for unit in units:
        if len(current) + len(unit) <= size:
            current += unit
            continue
        if current:
            pieces.append(current)
            current = ""
        while len(unit) > size:
            # A space inside the first `size + 1` characters ends a word that fits,
            # because the space itself is dropped rather than carried into either piece.
            cut = unit.rfind(" ", 1, size + 1)
            cut = size if cut <= 0 else cut
            pieces.append(unit[:cut])
            unit = unit[cut:].lstrip()
        current = unit
    if current:
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


def _spans(boundary: re.Pattern[str], text: str) -> list[str]:
    """`text` cut after every match of `boundary`. Joined back, the spans are `text`."""
    spans: list[str] = []
    start = 0
    for match in boundary.finditer(text):
        spans.append(text[start:match.end()])
        start = match.end()
    if start < len(text):
        spans.append(text[start:])
    return spans
