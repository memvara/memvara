"""Every link in the shipped documentation resolves to something that exists.

Ten links did not, and none of them announced it. Two kinds, both silent:

**Anchors that outlived their heading.** `README.md` pointed three times at
`#open-core-and-exactly-where-the-line-is` and `#what-the-fast-path-does-not-catch-measured`.
Both headings had moved out to `docs/OPEN-CORE.md` and `docs/DESIGN.md`, and the links
stayed behind pointing at nothing — on GitHub as well as on PyPI. `docs/API.md`,
`docs/BENCHMARKS.md` and `docs/DESIGN.md` each had one of their own.

**Paths written from the wrong directory.** `docs/BENCHMARKS.md` linked to
`bench/baseline.py`, which from inside `docs/` means `docs/bench/baseline.py`. Same for
`demo/`, `SECURITY.md` and `memvara/store/base.py` — the repository-root spelling, used in
a file one level down.

A broken link is the failure mode `CLAUDE.md` describes for documentation generally: it
does not fail loudly, it sends a reader somewhere confidently and lets them act on it. The
person who pays is never the author. Nothing in the suite looked at a link before this, so
these accumulated across releases in the files most likely to be read first.

`README.md` is checked here rather than only its rendered form because it is also the body
of the PyPI project page. `docs/` is checked because a link that dead-ends is worth the
same whichever file holds it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: Repository root. `tests/` sits directly under it.
ROOT = Path(__file__).resolve().parent.parent

#: `README.md` first, then every shipped document. `CHANGELOG.md`, `CONTRIBUTING.md` and
#: `SECURITY.md` are deliberately included: they are linked *from* the README, so a reader
#: arriving from PyPI lands in them.
DOCS = [ROOT / name for name in ("README.md", "CHANGELOG.md", "CONTRIBUTING.md",
                                 "SECURITY.md")] + sorted((ROOT / "docs").glob("*.md"))

#: `[text](target)` or `[text](target#anchor)`. Bare autolinks and reference-style links
#: are not used in these files; if one is added this pattern simply does not see it, which
#: is a miss rather than a false alarm.
LINK = re.compile(r"\]\(([^)\s#]*)(?:#([^)\s]+))?\)")

#: An ATX heading. Setext headings are not used in these files.
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.MULTILINE)


def slug(heading: str) -> str:
    """GitHub's anchor derivation, to the extent these documents exercise it.

    Lowercase, drop everything that is not a word character, whitespace or a hyphen, then
    collapse whitespace runs to single hyphens. That covers the punctuation actually
    present in these headings — commas, parentheses, quotes, backticks and full stops.

    >>> slug('## Answer quality, end to end (an authored corpus)')
    'answer-quality-end-to-end-an-authored-corpus'
    >>> slug('Two meanings of "delete", kept apart')
    'two-meanings-of-delete-kept-apart'
    """
    text = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return re.sub(r"\s+", "-", text).strip("-")


def anchors(path: Path) -> set[str]:
    """Every in-page anchor `path` offers."""
    return {slug(m.group(1)) for m in HEADING.finditer(path.read_text(encoding="utf-8"))}


def links(path: Path) -> list[tuple[str, str | None, int]]:
    """Every relative link in `path`, as `(target, anchor, line)`.

    Absolute URLs are somebody else's uptime and are not checked here — a network call in
    the offline suite would trade a real guarantee for a flaky one.
    """
    text = path.read_text(encoding="utf-8")
    out = []
    for m in LINK.finditer(text):
        target, anchor = m.group(1), m.group(2)
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        out.append((target, anchor, text[:m.start()].count("\n") + 1))
    return out


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_relative_link_points_at_a_file_that_exists(doc: Path) -> None:
    """The `docs/BENCHMARKS.md -> bench/baseline.py` failure: correct from the repository
    root, wrong from the directory the file actually sits in."""
    broken = [(t, line) for t, _a, line in links(doc)
              if t and not (doc.parent / t).resolve().exists()]
    assert not broken, "\n".join(
        f"{doc.relative_to(ROOT)}:{line} -> {t} (resolves to "
        f"{(doc.parent / t).resolve().relative_to(ROOT)}, which does not exist)"
        for t, line in broken)


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_anchor_points_at_a_heading_that_exists(doc: Path) -> None:
    """The `#open-core-and-exactly-where-the-line-is` failure: the heading moved to another
    file and three links stayed behind. Covers same-page anchors and the anchor half of a
    cross-file link, since a heading can be renamed out from under either."""
    broken = []
    for target, anchor, line in links(doc):
        if anchor is None:
            continue
        dest = (doc.parent / target).resolve() if target else doc
        if dest.suffix != ".md" or not dest.exists():
            continue
        if anchor not in anchors(dest):
            broken.append((f"{target}#{anchor}" if target else f"#{anchor}", line))
    assert not broken, "\n".join(
        f"{doc.relative_to(ROOT)}:{line} -> {ref} (no heading with that slug)"
        for ref, line in broken)
