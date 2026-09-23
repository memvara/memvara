"""The readable text and the title of an HTML page, using only the standard library.

The parser keeps the text a person would read and drops the rest: scripts, styles,
navigation, forms and other page furniture. When the page marks its main content with a
`<main>` or `<article>` element, only the text inside those elements is kept, because the
text outside them is usually menus, footers and sidebars. When it does not, the whole body
is kept.

This is a heuristic. It does not run JavaScript, so a page that builds its content in the
browser comes back nearly empty, and the caller then gets a `no_text` error from
`memvara.ingest.extract`.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

__all__ = ["html_to_text"]

#: Elements whose whole content is dropped.
_DROP = frozenset({"script", "style", "noscript", "template", "svg", "canvas", "iframe",
                   "object", "nav", "header", "footer", "aside", "form", "button",
                   "select", "head"})

#: Elements that mark the main content of a page.
_MAIN = frozenset({"main", "article"})

#: Elements that start a new line of text. Everything else is inline.
_BLOCK = frozenset({"p", "div", "section", "br", "hr", "li", "ul", "ol", "dl", "dt", "dd",
                    "h1", "h2", "h3", "h4", "h5", "h6", "table", "tr", "td", "th",
                    "blockquote", "pre", "figure", "figcaption", "main", "article",
                    "body"})

#: Elements that never have an end tag, so they must not change the nesting counters.
_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
                   "meta", "source", "track", "wbr"})


class _Reader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.drop_depth = 0
        self.main_depth = 0
        self.in_title = False
        self.title_seen = False
        self.title: list[str] = []
        self.everything: list[str] = []
        self.main: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            # Only the first title counts: an inline SVG carries `<title>` elements of its
            # own, and they name the drawing, not the page.
            self.in_title = not self.title_seen
            self.title_seen = True
            return
        if tag in _VOID:
            if tag in _BLOCK:
                self._text("\n")
            return
        if tag in _DROP:
            self.drop_depth += 1
        elif tag in _MAIN:
            self.main_depth += 1
        if tag in _BLOCK:
            self._text("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
            return
        if tag in _DROP and self.drop_depth:
            self.drop_depth -= 1
        elif tag in _MAIN and self.main_depth:
            self.main_depth -= 1
        if tag in _BLOCK:
            self._text("\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title.append(data)
        else:
            self._text(data)

    def _text(self, data: str) -> None:
        if self.drop_depth:
            return
        self.everything.append(data)
        if self.main_depth:
            self.main.append(data)


def _tidy(parts: list[str]) -> str:
    """Join text pieces, collapse spaces inside lines, and keep at most one blank line."""
    lines = (re.sub(r"[ \t\r\f\v\xa0]+", " ", line).strip()
             for line in "".join(parts).split("\n"))
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_text(markup: str) -> tuple[str, str | None]:
    """The readable text of `markup` and its `<title>`, or `None` when it has no title.

    >>> text, title = html_to_text(
    ...     "<html><head><title> Release notes </title><style>p{}</style></head>"
    ...     "<body><nav>Home | Docs</nav><main><h1>0.2</h1><p>Adds &amp; fixes.</p>"
    ...     "</main><footer>(c) 2026</footer></body></html>")
    >>> title
    'Release notes'
    >>> print(text)
    0.2
    <BLANKLINE>
    Adds & fixes.
    """
    reader = _Reader()
    reader.feed(markup)
    reader.close()
    chosen = reader.main if "".join(reader.main).strip() else reader.everything
    title = " ".join("".join(reader.title).split()) or None
    return _tidy(chosen), title
