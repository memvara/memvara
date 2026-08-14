#!/usr/bin/env python3
"""Read and rewrite `CHANGELOG.md`, so release notes are written once.

Two callers, one file:

* `bump_version.py` closes `## [Unreleased]` into `## [X.Y.Z] — YYYY-MM-DD`.
* `check_release.py` reads that section back out and hands it to
  `gh release edit --notes-file`, so the GitHub Release body **is** the changelog entry
  rather than a second copy of it that drifts.

The second half is the point of the first. Notes typed into the GitHub Release box are
invisible to anyone reading the repository, and notes written in the changelog are
invisible to anyone reading the release page; keeping both in step by hand lasts about two
releases. Deriving one from the other means the changelog is the thing that gets reviewed
in the bump pull request, and the release page cannot disagree with it.

## The format, as this file depends on it

Loosely Keep a Changelog. What matters here:

    ## [Unreleased]
    ## [0.1.0] — 2026-08-10

A level-2 heading, the version in square brackets, and for a released version an em dash
and an ISO date. Headings that are **not** in that shape are left alone — the tail of this
changelog has `## Wave 3`, `## Wave 2`, `## Wave 1` from before the project was versioned,
and a looser pattern would have read those as releases named "Wave".
"""

from __future__ import annotations

import re
from datetime import date as _date
from pathlib import Path

from release.versions import REPO, Refused

CHANGELOG = "CHANGELOG.md"

UNRELEASED = "Unreleased"

#: A version heading. The date separator is an em dash in this file; a plain hyphen is
#: accepted on read so that a hand-written entry is not rejected over punctuation, but
#: `open_release` always writes the em dash to match everything already there.
HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\](?:[ \t]*[—-][ \t]*(?P<date>\S+))?[ \t]*$",
                     re.MULTILINE)

#: Any level-2 heading at all, version-shaped or not. Section bodies end at the next one of
#: these, so `## Wave 3` terminates the section above it exactly as a release heading would.
ANY_H2 = re.compile(r"^## ", re.MULTILINE)


def read(repo: Path = REPO) -> str:
    path = repo / CHANGELOG
    if not path.is_file():
        raise Refused(f"no {CHANGELOG} under {repo}")
    return path.read_text(encoding="utf-8")


def _heading(text: str, version: str) -> re.Match[str]:
    for m in HEADING.finditer(text):
        if m.group("version") == version:
            return m
    listed = [m.group("version") for m in HEADING.finditer(text)]
    raise Refused(f"{CHANGELOG} has no `## [{version}]` heading "
                  f"(it has: {', '.join(listed) or 'none'})")


def section(text: str, version: str) -> str:
    """The body under one version heading, with surrounding blank lines stripped."""
    head = _heading(text, version)
    rest = text[head.end():]
    nxt = ANY_H2.search(rest)
    return rest[: nxt.start() if nxt else len(rest)].strip("\n")


def released_versions(text: str) -> list[str]:
    """Versions with a heading, newest first, excluding `[Unreleased]`."""
    return [m.group("version") for m in HEADING.finditer(text)
            if m.group("version") != UNRELEASED]


#: What an empty release says. A version heading with literally nothing under it would make
#: `section()` return "" and the release-notes lookup refuse, so "no entries" is spelled
#: out rather than left blank — and a reader of the changelog gets told that the absence is
#: deliberate instead of wondering whether the entry was lost.
NO_CHANGES = "No user-visible changes."


def open_release(text: str, version: str, when: _date, *, allow_empty: bool = False) -> str:
    """Close `[Unreleased]` into a dated release heading, leaving `[Unreleased]` empty.

    The whole body moves down; `[Unreleased]` stays at the top with nothing under it, ready
    for the next change. That ordering is deliberate — inserting the new heading *below*
    `[Unreleased]` rather than renaming it means a merge conflict in the bump pull request
    lands on the new heading line and not on the section every open branch is also editing.
    """
    if version in released_versions(text):
        raise Refused(f"{CHANGELOG} already has a `## [{version}]` heading. "
                      f"A released version is not re-cut; bump to a new one.")
    body = section(text, UNRELEASED)
    if not body.strip() and not allow_empty:
        raise Refused(
            f"`## [{UNRELEASED}]` in {CHANGELOG} is empty, so this release has no notes.\n"
            "  A release whose notes are blank is one nobody can tell apart from the "
            "previous one. Write the entry first, or pass --allow-empty-changelog if the "
            "release genuinely contains no user-visible change.")
    head = _heading(text, UNRELEASED)
    # Inserted immediately after the `[Unreleased]` heading line and *before* its trailing
    # newline, so the blank line that already separated that heading from its body now
    # separates the new heading from the same body. Appending a newline here instead would
    # leave two blank lines behind every release heading — cosmetic, and cosmetic drift in
    # a file this tool parses is how the parser stops matching later.
    inserted = f"\n\n## [{version}] — {when.isoformat()}"
    if not body.strip():
        inserted += f"\n\n{NO_CHANGES}"
    return text[: head.end()] + inserted + text[head.end():]


def write(text: str, repo: Path = REPO) -> None:
    (repo / CHANGELOG).write_text(text, encoding="utf-8")
