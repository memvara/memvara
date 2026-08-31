"""Claims the documentation makes about the code, checked against the code.

`tests/test_doc_links.py` next door answers "does this link resolve". These are the
claims a resolving link can still get wrong:

**The README's links are absolute, and nothing looked at them.** They have to be —
`README.md` is also the body of the PyPI project page, where a relative link resolves
against `pypi.org` and 404s — so `test_doc_links` skips every one of them by design. That
left the most-read file in the repository as the only one with unchecked links. These
tests resolve `https://github.com/memvara/memvara/blob/main/<path>` back to a path in this
checkout and check the file, and the heading, the same way.

**The MCP tool table.** `docs/integrations/mcp.md` lists the fourteen tools by name. The
equivalent list inside the packaged skill rotted exactly this way once — it said "the ten
tools" and listed ten, having never gained `memory_neighborhood` or `memory_paths` — and
`tests/test_init.py` guards that copy. This is the same guard for the copy in `docs/`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from memvara.server.tools import TOOLS

from test_doc_links import ROOT, anchors

#: A link into this repository, as the README has to spell it for PyPI's renderer.
GITHUB = re.compile(
    r"\]\(https://github\.com/memvara/memvara/(?:blob|tree)/main/([^)#\s]*)"
    r"(?:#([^)\s]+))?\)")

#: Also in HTML attributes: the badge row is raw HTML, so `LICENSE` sitting in an
#: `href=` is invisible to the markdown pattern above and just as broken on PyPI.
HTML_HREF = re.compile(r'href="([^"]+)"')

README = ROOT / "README.md"


def github_links() -> list[tuple[str, str | None, int]]:
    """Every in-repository link the README makes, as `(path, anchor, line)`."""
    text = README.read_text(encoding="utf-8")
    return [(m.group(1), m.group(2), text[:m.start()].count("\n") + 1)
            for m in GITHUB.finditer(text)]


def test_every_readme_link_into_this_repository_points_at_something_that_exists() -> None:
    """A renamed or moved file leaves the README pointing at a 404 on GitHub and on PyPI."""
    broken = [(path, line) for path, _a, line in github_links()
              if not (ROOT / path).exists()]
    assert not broken, "\n".join(
        f"README.md:{line} -> {path} (no such file in this checkout)"
        for path, line in broken)


def test_every_readme_anchor_points_at_a_heading_that_exists() -> None:
    """The other half: the file is right and the heading it names has been renamed."""
    broken = []
    for path, anchor, line in github_links():
        target = ROOT / path
        if anchor is None or target.suffix != ".md" or not target.exists():
            continue
        if anchor not in anchors(target):
            broken.append((f"{path}#{anchor}", line))
    assert not broken, "\n".join(
        f"README.md:{line} -> {ref} (no heading with that slug)" for ref, line in broken)


def test_the_readme_names_no_repository_path_relatively() -> None:
    """A relative link in this one file is a link that works on GitHub and 404s on PyPI.

    That is the worst shape a broken link can take, because the person who wrote it sees
    it working. Checked over raw HTML too: the licence badge sat in an `href="LICENSE"`,
    where no markdown pattern would ever have looked at it.
    """
    text = README.read_text(encoding="utf-8")
    offenders = [href for href in HTML_HREF.findall(text)
                 if not href.startswith(("http://", "https://", "mailto:", "#"))]
    assert not offenders, (
        f"relative href(s) in README.md: {offenders}. README.md is the PyPI project "
        "description, where a relative link resolves against pypi.org. Spell it "
        "https://github.com/memvara/memvara/blob/main/<path>.")


def test_the_docs_mcp_page_names_every_tool_the_server_serves() -> None:
    """`docs/integrations/mcp.md` states the count and lists the tools.

    Both halves are asserted, because a list one short of its own stated count agrees
    with itself perfectly. Order too, so the table reads in the order the server declares.
    """
    page = (ROOT / "docs" / "integrations" / "mcp.md").read_text(encoding="utf-8")
    words = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
             15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen"}
    word = words[len(TOOLS)]

    assert f"The {word} tools" in page, (
        f"the page must state the count as 'The {word} tools'; stating it positively is "
        "what makes a deleted sentence fail as loudly as a wrong one")

    seen: list[str] = []
    for name in re.findall(r"`(memory_[a-z_]+)`", page):
        if name not in seen:
            seen.append(name)
    assert seen == [t.name for t in TOOLS], (
        "the page must name every tool, once, in the order the server declares them; "
        f"got {seen}")


#: Documents that are not on a reader's path through the product, and are excluded from
#: the navigation rule for that reason rather than for convenience. `RELEASING.md` is a
#: runbook for whoever cuts a release; `WAVE4`, `WAVE5` and `WORKPLAN` are internal
#: planning records; `superpowers/` holds a plan and a spec for one piece of work. None of
#: them is somewhere a reader arrives and wonders where to go next, and giving each a
#: "Next:" line would be inventing a journey nobody takes.
NOT_ON_THE_READERS_PATH = {
    "docs/RELEASING.md", "docs/WAVE4.md", "docs/WAVE5.md", "docs/WORKPLAN.md",
}

#: Every page a reader can arrive at from the README or the documentation index.
READER_PAGES = [
    p for p in sorted((ROOT / "docs").rglob("*.md")) + sorted((ROOT / "examples").rglob("*.md"))
    if p.relative_to(ROOT).as_posix() not in NOT_ON_THE_READERS_PATH
    and "superpowers" not in p.parts
]


@pytest.mark.parametrize("page", READER_PAGES,
                         ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_page_says_where_to_go_next(page: Path) -> None:
    """A reader who finishes a page and has to guess is a reader who stops.

    The rule is one line: every document a reader can arrive at ends with a navigation
    footer naming at least one other page. It is asserted rather than trusted because it
    is exactly the kind of thing a new page gets added without, and because the failure
    is invisible to the author — who knows where the next page is.
    """
    tail = page.read_text(encoding="utf-8").rstrip().rsplit("\n", 1)[-1]
    assert tail.startswith(("Next:", "Previous:")), (
        f"{page.relative_to(ROOT)} does not end with a navigation footer. Finish it with "
        "a line starting 'Previous:' or 'Next:' naming where the reader goes.")
