#!/usr/bin/env python3
"""Bump the version everywhere it is spelled, and close out the changelog.

    python3 -m release.bump_version --part minor
    python3 -m release.bump_version --version 0.2.0rc1
    python3 -m release.bump_version --part patch --npm-part patch    # the placeholder too

Run from the repository root. `.github/workflows/version-bump.yml` is the way this is
meant to be invoked — from GitHub, into a pull request, so a release is something a human
approves rather than something a laptop did.

## What it writes

* `pyproject.toml` and `memvara/__init__.py` — the two places the Python version is
  spelled, and nothing in the build keeps them equal.
* `CHANGELOG.md` — `## [Unreleased]` is closed into `## [X.Y.Z] — YYYY-MM-DD` and left
  empty above it. That section becomes the GitHub Release body verbatim at publish time,
  which is why this refuses to bump when it is empty: the notes are written once, here, or
  they are not written at all.
* `npm/memvara/package.json` — **only when asked.** The npm package is a name reservation
  on its own `0.0.x` line and an ordinary Python release does not touch it. The reasoning
  is in `release/versions.py`; the short version is that a matching number would assert an
  equivalence between a library and an empty object.

## What it refuses

* A version that does not go forwards. `--version 0.1.0` when 0.2.0 is out is a typo, and
  the only thing downstream of it that would notice is PyPI, permanently.
* A version already on PyPI. This is the one guard that has to live *here* rather than at
  publish time: at publish time the number is already in a tag, in a release and in a
  changelog heading, and the only fix is to bump again. Asked here, it costs one HTTP
  request and the answer arrives before anything is written.
* An empty `## [Unreleased]`. See above.

Nothing here commits, tags or pushes. It edits files; the workflow around it opens the
pull request.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):        # pragma: no cover - `python3 release/bump_version.py`
    # Running the file by path puts `release/` on sys.path, not the repository root, so
    # `release.versions` would not resolve. Supporting both invocations matters because the
    # two older scripts here are documented as `python3 release/publish_pypi.py`, and a
    # tool that works only when spelled one particular way is one that gets spelled the
    # other way at the worst moment.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from release import changelog
from release.versions import (
    NPM_SPELLING,
    NPM_VERSION,
    PARTS,
    PYTHON_SPELLINGS,
    REPO,
    Refused,
    bump,
    is_prerelease,
    npm_version,
    parse,
    python_version,
    rewrite,
    sorts_after,
)


def emit(**outputs: str) -> None:
    """Hand values to the surrounding workflow, if there is one.

    Written as `name=value` lines to `$GITHUB_OUTPUT`. No value here is a secret — they are
    version numbers and branch names — but the file is append-only and shared with every
    other step in the job, so this writes only what it names.
    """
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        for key, value in outputs.items():
            print(f"  {key}: {value}")
        return
    with open(path, "a", encoding="utf-8") as fh:
        for key, value in outputs.items():
            fh.write(f"{key}={value}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--part", choices=PARTS, help="bump this component of the version")
    what.add_argument("--version", help="set this exact version (e.g. 0.2.0rc1)")
    ap.add_argument("--npm-part", choices=PARTS, default=None,
                    help="also bump the npm placeholder. Off by default: see "
                         "release/versions.py for why the two numbers are not one number.")
    ap.add_argument("--npm-version", default=None,
                    help="set the npm placeholder to this exact version")
    ap.add_argument("--allow-empty-changelog", action="store_true",
                    help="release with no `## [Unreleased]` entries; for a re-cut only")
    ap.add_argument("--offline", action="store_true",
                    help="skip the registry check. Local rehearsals only — the check is "
                         "the one that cannot be usefully repeated later.")
    ap.add_argument("--repo", type=Path, default=REPO)
    args = ap.parse_args(argv)

    repo: Path = args.repo
    current = python_version(repo)
    new = args.version if args.version else bump(current, args.part)
    parse(new)                                   # refuses a version this project cannot ship

    if not sorts_after(new, current):
        raise Refused(f"{new} does not come after the current {current}. A version that "
                      f"goes backwards is a typo whose only reader is PyPI, permanently.")

    npm_current = npm_version(repo)
    npm_new = args.npm_version or (
        bump(npm_current, args.npm_part, pattern=NPM_VERSION) if args.npm_part else None)
    if npm_new is not None:
        parse(npm_new, pattern=NPM_VERSION)
        if not sorts_after(npm_new, npm_current, pattern=NPM_VERSION):
            raise Refused(f"the npm placeholder {npm_new} does not come after {npm_current}")

    if not args.offline:
        from release.registry import npm_versions, pypi_versions
        if new in pypi_versions("memvara"):
            raise Refused(
                f"memvara {new} is already on PyPI.\n"
                "  A version number cannot be reused there — not after a yank, not after "
                "deleting the release. Pick the next one.")
        if npm_new is not None and npm_new in npm_versions("memvara"):
            raise Refused(f"memvara {npm_new} is already on npm. `unpublish` is 72 hours "
                          f"and narrower than you think; do not plan around it.")

    # Everything that can refuse has refused by now. The changelog rewrite is computed
    # before any file is touched so that a bad `## [Unreleased]` cannot leave the two
    # version spellings bumped and the notes not written — a half-applied bump is the one
    # state that looks like a finished one.
    text = changelog.open_release(changelog.read(repo), new,
                                  datetime.now(timezone.utc).date(),
                                  allow_empty=args.allow_empty_changelog)

    for spelling in PYTHON_SPELLINGS:
        rewrite(spelling, current, new, repo)
        print(f"  {spelling.path}: {current} -> {new}")
    changelog.write(text, repo)
    print(f"  CHANGELOG.md: `## [Unreleased]` closed into `## [{new}]`")
    if npm_new is not None:
        rewrite(NPM_SPELLING, npm_current, npm_new, repo)
        print(f"  {NPM_SPELLING.path}: {npm_current} -> {npm_new}")

    emit(previous=current, version=new, tag=f"v{new}",
         prerelease="true" if is_prerelease(new) else "false",
         npm_version=npm_new or npm_current,
         npm_bumped="true" if npm_new else "false",
         branch=f"release/v{new}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Refused as exc:                       # pragma: no cover - CLI surface
        raise SystemExit(f"\n  REFUSED: {exc}\n")
