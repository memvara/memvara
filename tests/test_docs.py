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

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from memvara.server.tools import TOOLS

from test_doc_links import DOCS, ROOT, anchors

#: A link into this repository, as the README has to spell it for PyPI's renderer.
GITHUB = re.compile(
    r"\]\(https://github\.com/memvara/memvara/(?:blob|tree)/main/([^)#\s]*)"
    r"(?:#([^)\s]+))?\)")

#: Also in HTML attributes: the badge row is raw HTML, so `LICENSE` sitting in an
#: `href=` is invisible to the markdown pattern above and just as broken on PyPI.
HTML_HREF = re.compile(r'href="([^"]+)"')

#: Any markdown link target, absolute or not. Needed because the relative-link check
#: below originally looked at `href=` alone, which meant an ordinary `[text](docs/API.md)`
#: — the far commoner way to write the mistake — passed it *and* passed `test_doc_links`,
#: whose "does this resolve" question a relative path answers correctly from the
#: repository root and wrongly from pypi.org.
MD_LINK = re.compile(r"\]\(([^)\s]+)\)")

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
    it working, and `test_doc_links` cannot see it either: a relative target resolves
    correctly from the repository root, which is the only thing that test asks.

    **Both syntaxes, deliberately.** The first version of this checked `href=` alone,
    because that is where the failure had actually been found — the licence badge sat in
    an `href="LICENSE"`. But the ordinary way to write the same mistake is a markdown
    link, and it went straight through. A guard that covers the rarer half of a failure
    and not the commoner half is the shape of guard that reads as protection and is not.
    """
    text = README.read_text(encoding="utf-8")
    relative = lambda t: not t.startswith(("http://", "https://", "mailto:", "#"))
    offenders = ([f'href="{h}"' for h in HTML_HREF.findall(text) if relative(h)]
                 + [f"]({t})" for t in MD_LINK.findall(text) if relative(t)])
    assert not offenders, (
        f"relative link(s) in README.md: {offenders}. README.md is the PyPI project "
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


#: Pages that present a complete, ordered sequence a reader is meant to follow and run,
#: as opposed to concept pages whose snippets are fragments around a `mem` defined
#: elsewhere. Only these are executed; widening the list means making the page runnable
#: first, which is the right order to do it in.
RUNNABLE_PAGES = ["docs/getting-started/quickstart.md",
                  "docs/getting-started/first-memory.md"]

#: A block a reader is not meant to run — a signature illustration, a placeholder — must
#: say so in the source, with a reason, on the line before its fence. The marker is an
#: HTML comment so it renders as nothing.
SKIP_MARKER = "<!-- runnable: no"

PY_BLOCK = re.compile(
    r"(?:(" + re.escape(SKIP_MARKER) + r"[^\n]*-->)\n)?```python\n(.*?)```", re.S)


def executable_source(page: Path) -> str:
    """Every runnable python block on `page`, concatenated in reading order."""
    kept = [body for marker, body in PY_BLOCK.findall(page.read_text(encoding="utf-8"))
            if not marker]
    return "import warnings; warnings.simplefilter('ignore')\n" + "\n".join(kept)


@pytest.mark.parametrize("relative", RUNNABLE_PAGES)
def test_the_getting_started_pages_actually_run(relative: str, tmp_path: Path) -> None:
    """A page that walks a reader through a sequence has to survive being walked through.

    Nothing executed the documentation before this, and one page did not survive it: an
    `added[0]` after a write that reinforced rather than added, a history printed in the
    wrong order because one write in the slot was undated, and two `forget()` variants
    shown as alternatives but written as a sequence. All three read perfectly and none of
    them ran.

    Run in a subprocess from `tmp_path`, so a page that names a database file writes it
    somewhere disposable rather than into the checkout, and so a page that calls
    `sys.exit` or leaks state cannot affect the rest of the suite.
    """
    script = tmp_path / "page.py"
    script.write_text(executable_source(ROOT / relative), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                          capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert proc.returncode == 0, (
        f"{relative} does not run as written:\n{proc.stderr}\n"
        "Every block on a page in RUNNABLE_PAGES executes in reading order. If a block is "
        "an illustration rather than a step, mark it with "
        "'<!-- runnable: no — <reason> -->' on the line before its fence.")


#: The `Memvara` facade methods each architecture diagram lists, read out of the diagrams
#: rather than restated, so the assertion cannot drift from the picture.
DIAGRAM_PAGES = ["README.md", "docs/reference/architecture.md"]

FACADE_NODE = re.compile(r"<b>Memvara</b> — memvara/core\.py<br/><i>(.*?)</i>", re.S)


@pytest.mark.parametrize("relative", DIAGRAM_PAGES)
def test_every_method_the_architecture_diagram_names_exists(relative: str) -> None:
    """A diagram is documentation, and an invented method on one is a fabricated API.

    Both of these listed `end` on the facade. There is no `Memvara.end` — ending a fact is
    `forget(close="ended")` or `delete(close="ended")` — on a page whose own first line
    says nothing in it is aspirational. Read out of the diagram text so that adding a
    method to the picture without adding it to the class fails here.
    """
    from memvara import Memvara

    text = (ROOT / relative).read_text(encoding="utf-8")
    node = FACADE_NODE.search(text)
    assert node is not None, f"{relative} has no Memvara facade node to check"

    # One diagram labels the node "the facade: add, remember, …"; drop a leading label.
    body = re.sub(r"^[^:·,]*:\s*", "", node.group(1).strip())
    names = [n.strip() for n in re.split(r"[·,]|<br/>", body) if n.strip()]
    missing = [n for n in names if not hasattr(Memvara, n)]
    assert not missing, (
        f"{relative} names {missing} on the Memvara facade, which does not have them. "
        "A diagram that invents a method is worse than one that omits it.")


#: The README's *Use cases* section opens on a transcript rather than a description, and
#: the transcript is a quotation. Fenced as `text` and starting with the first question,
#: which is what makes it findable here without pinning the prose around it.
TRANSCRIPT = re.compile(r"```text\n(Q\. What is checkout-service.*?)```", re.S)


def test_the_readme_quotes_the_coding_agent_example_verbatim() -> None:
    """A transcript in the README is a claim about what the program prints.

    Three of these lines are already pinned by `tests/test_examples.py`, which asserts on
    the timeline the example produces. The rest — the questions, and the blank line that
    separates each from its answer — are not, and a quotation that is right about the
    values and wrong about their shape is still a quotation nobody can reproduce. So the
    whole block is matched as a substring of the real output, rather than line by line:
    the reader copies the file and compares what they see with what is on the page.

    Run in a subprocess with `PYTHONPATH` removed, for the reason `tests/test_examples.py`
    gives: the examples import the installed package and nothing else.
    """
    quoted = TRANSCRIPT.search(README.read_text(encoding="utf-8"))
    assert quoted is not None, (
        "README.md no longer quotes the coding-agent transcript. If the section was "
        "rewritten deliberately, delete this test with it; if not, the quotation is gone.")

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run([sys.executable, str(ROOT / "examples" / "coding_agent.py")],
                          cwd=ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert quoted.group(1) in proc.stdout, (
        "the README quotes output that examples/coding_agent.py does not print:\n"
        f"--- README ---\n{quoted.group(1)}\n--- actual ---\n{proc.stdout}")


#: A link from the README back into its own headings, spelled absolutely because this file
#: is also the PyPI project description. `test_doc_links` skips it for being absolute, and
#: the `GITHUB` pattern above does not match it for having no `/blob/main/` path — so it
#: sat between the two guards, and a fragment 200s whatever the heading is called.
SELF_ANCHOR = re.compile(r"https://github\.com/memvara/memvara#([\w-]+)")


def test_every_readme_link_back_into_the_readme_names_a_heading_it_has() -> None:
    """The navigation row at the top is three of these, and a renamed heading breaks it.

    Silently, and in the place it costs most: the row is the first thing a reader clicks.
    GitHub answers a bad fragment with the page and no scroll, so nothing 404s and nobody
    finds out from the outside.
    """
    have = anchors(README)
    broken = sorted(a for a in SELF_ANCHOR.findall(README.read_text(encoding="utf-8"))
                    if a not in have)
    assert not broken, (
        f"README.md links to its own {broken}, which is not a heading in it. Either the "
        "heading was renamed or the link was mistyped; the anchors it does have are "
        f"{sorted(have)}.")


#: Every markdown file this repository ships, which is deliberately wider than
#: `test_doc_links.DOCS`. That set is `README.md`, `docs/` and `examples/` — the reader's
#: path — and a wording scan has to cover the copies *off* that path too: the packaged
#: skill under `memvara/skills/memvara/`, its mirror under `plugin/skills/`, and the
#: `README.md` files in `npm/`, `plugin/`, `demo/` and `release/`. The packaged skill is
#: the sharpest of those and this file's own docstring says why — its tool list rotted
#: exactly this way once, and it is vendored by sha into seven downstream repositories,
#: so a wrong sentence there is a wrong sentence in all of them.
#:
#: Directories that are ignored or generated are skipped by name rather than by asking
#: git, so this works from an unpacked sdist as well as from a checkout.
SHIPPED_MARKDOWN = sorted(
    p for p in ROOT.rglob("*.md")
    if not {".git", ".pytest_cache", "local", "node_modules", ".venv"} & set(p.parts))

#: `CHANGELOG.md` is not scanned. It quotes each wrong form inside the entry recording
#: that the form was corrected, so scanning it would go red on the evidence that the fix
#: happened. It is the one exclusion, and it is a property of what that file is for.
NOT_SCANNED_FOR_WORDING = {"CHANGELOG.md"}


def wrapped(phrase: str) -> re.Pattern[str]:
    """`phrase`, with every space allowed to be a line break.

    These files are hard-wrapped at about 95 columns, so a wording that sits on one line
    today lands across two the moment a word ahead of it changes. A pattern that only
    matches within a line stops working when the paragraph is re-flowed — silently, and
    in exactly the case it was written for.
    """
    return re.compile(phrase.replace(" ", r"\s+"))


#: Wordings this project has already paid to correct, as
#: `(slug, pattern, what is wrong with it, what to write instead)`.
#:
#: Both entries are here because the same sentence had to be corrected more than once, in
#: different files, after a reviewer found it rather than the suite. The author of a
#: correction knows which file they were editing, which is why the copy that survives is
#: never the one they were looking at.
#:
#: **The patterns are loose on purpose.** The first version of the second entry required
#: `read` and `takes` to be adjacent, and so matched neither *"Every read in the API
#: takes"* in `docs/concepts/bitemporal-memory.md` nor *"every read below takes"* in
#: `docs/API.md` — two live copies, in the commit that added the guard. A pattern that
#: only matches the exact sentence already fixed is a guard that reads as protection and
#: is not.
#:
#: **What this does not do is worth stating, because the name suggests otherwise.** It
#: catches a wording coming *back*. A fresh overstatement in fresh words passes it, and
#: nothing in the suite sees that. The list is a record of mistakes made, not a model of
#: the API, and the time to add an entry is after correcting a claim that turned out to
#: have copies — not guessed at in advance.
RETIRED_WORDINGS = [
    ("add-costs-a-single-call",
     wrapped(r"into (?:a single|one) (?:model )?call\b"),
     "`add()` makes one *extraction* call, and a predicate the registry has not seen "
     "before costs a second for acquisition. `WriteReceipt.llm_calls` counts both, so a "
     "reader who takes the sentence at its word and then measures finds two.",
     'write "a single extraction call"'),

    ("every-read-takes",
     wrapped(r"[Ee]very read\b(?:\s+\S+){0,3} takes\b"),
     "`recall()`, `get()` and `since()` take none of the three time keywords — "
     "deliberately, and `recall()`'s docstring says why — and `ask()` spells it `at=`. "
     "Exactly eight reads take them, so any form of \"every read\" sends a reader into a "
     "`TypeError`.",
     'write "eight reads take" and name them: `search`, `get_all`, `count`, `history`, '
     "`why`, `produced`, `neighborhood`, `paths_between`"),
]


@pytest.mark.parametrize("wording", RETIRED_WORDINGS,
                         ids=[w[0] for w in RETIRED_WORDINGS])
def test_no_document_brings_back_a_wording_this_project_has_corrected(
        wording: tuple[str, re.Pattern[str], str, str]) -> None:
    """A claim corrected in one file and left standing in another is the whole failure.

    Four copies, across two claims, none of them found by the suite. The sentence about
    what `add()` costs was fixed in `README.md` and left in `docs/FAQ.md`, four lines
    above that page's own promise that `WriteReceipt.llm_calls` reports the cost so the
    claim is checkable — an invitation to go and check it and find the page wrong. "Every
    read takes all three" was fixed in the README's feature strip and left in the README's
    own *Temporal memory* section, in `docs/concepts/temporal-retrieval.md`, in
    `docs/concepts/bitemporal-memory.md`, and in `docs/API.md`, where it is contradicted
    four lines later by the listing it introduces.

    Every shipped markdown file is scanned, rather than the pair that happened to
    disagree, because the copy that survives is never in the file you are looking at.
    """
    _slug, pattern, why, instead = wording
    hits = []
    for doc in SHIPPED_MARKDOWN:
        if doc.name in NOT_SCANNED_FOR_WORDING:
            continue
        text = doc.read_text(encoding="utf-8")
        hits += [f"  {doc.relative_to(ROOT)}:{text[:m.start()].count(chr(10)) + 1}: "
                 f"{' '.join(m.group(0).split())!r}" for m in pattern.finditer(text)]

    assert not hits, "\n".join([f"{why}\n\nSo {instead}. Found in:", *hits])
