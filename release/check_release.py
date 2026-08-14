#!/usr/bin/env python3
"""Decide, before anything is uploaded, whether a GitHub Release may be published.

    python3 -m release.check_release --tag v0.2.0 --github-prerelease false

Run by the first job of `.github/workflows/release.yml`, on a checkout of the tag being
released. Everything it can refuse, it refuses here — before the build, before the test
matrix, and long before either registry has been touched.

## The failure this exists to make impossible

**A release tagged `v0.2.0` that ships `0.1.0`.** Nothing in the tooling connects a git tag
to a version: the tag is typed into GitHub's release form and the version lives in two
files. They agree by habit. So this compares them and refuses on a mismatch, which is the
only place in the pipeline where that comparison can be made before the number becomes
permanent.

## Idempotency, which is a decision and not an implementation detail

A published version can never be republished — absolutely on PyPI, and on npm past 72
hours. So the obvious guard is "refuse if the version is already on the registry", and
that guard is in the *bump* tool, where the answer is actionable.

Here it would be actively harmful. Consider the state this pipeline has to survive: PyPI
succeeded, npm failed on a misconfigured publisher, and the release is half done. The fix
is to re-run the workflow. A guard that refuses because PyPI already has the version turns
that re-run into a dead end, and the way out becomes publishing npm by hand — which is the
manual step this whole pipeline exists to remove.

So each registry is asked separately, and a registry that already has the version is
**skipped, loudly**, rather than being a failure:

* `pypi_needed=false` — PyPI already has it. The job is skipped and the run summary says
  so in as many words.
* `npm_needed=false` — likewise for the placeholder, which is skipped on every ordinary
  release because its version does not change.

Skipping is safe here for a reason worth stating rather than assuming: **two different
commits cannot share a tag.** A release for `v0.2.0` is always the same commit as any
earlier release for `v0.2.0`, so "PyPI already has 0.2.0" and "this exact release already
published" are the same fact. The one case that escapes it is deleting a tag and
re-creating it on a different commit; PyPI would reject that upload anyway, so nothing
wrong ships — what is lost is the loudness of the rejection, which is why the skip writes
a warning into the run summary rather than passing quietly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):        # pragma: no cover - `python3 release/check_release.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from release import changelog
from release.versions import (
    NPM_VERSION,
    REPO,
    Refused,
    is_prerelease,
    npm_version,
    parse,
    python_version,
)

#: The npm dist-tag a pre-release is allowed to take. `latest` is what a bare
#: `npm install memvara` resolves to, so putting a pre-release there hands it to everyone
#: who did not ask for one — silently, because nothing about the install says "candidate".
PRERELEASE_TAG = "next"
LATEST = "latest"


def version_from_tag(tag: str) -> str:
    """`v0.2.0` -> `0.2.0`, refusing anything else.

    The `v` prefix is required rather than tolerated. Accepting both `0.2.0` and `v0.2.0`
    would mean two tags could name one release, and the tag is what every downstream URL —
    the changelog link, the source archive, the `Repository` metadata — is built from.
    """
    if not tag.startswith("v"):
        raise Refused(f"the release tag {tag!r} does not start with `v`. "
                      f"This project tags releases `v<version>`, e.g. v0.2.0.")
    return tag[1:]


def dist_tag(npm_ver: str, *, github_prerelease: bool) -> str:
    """Which npm dist-tag this publish may use.

    Two independent reasons to withhold `latest`: the version itself is a pre-release, or
    the human marked the GitHub Release as one. Either is sufficient — the second exists
    because a release can be a candidate without its number saying so, and the person who
    ticked the box has stated an intent the number does not carry.
    """
    if is_prerelease(npm_ver, pattern=NPM_VERSION) or github_prerelease:
        return PRERELEASE_TAG
    return LATEST


def check_dist_tag(npm_ver: str, tag: str) -> None:
    """Refuse a pre-release aimed at `latest`.

    `dist_tag` above cannot currently produce that pairing, and this checks it anyway. The
    rule and the computation are two different things, and the one that matters is the
    rule: when a real JavaScript client exists and the tag becomes an input rather than a
    derivation, this is the line that still holds.
    """
    if tag == LATEST and is_prerelease(npm_ver, pattern=NPM_VERSION):
        raise Refused(f"{npm_ver} is a pre-release and cannot go to the `{LATEST}` "
                      f"dist-tag: that is what a bare `npm install memvara` resolves to.")


def emit(**outputs: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        for key, value in outputs.items():
            print(f"  {key}: {value}")
        return
    with open(path, "a", encoding="utf-8") as fh:
        for key, value in outputs.items():
            fh.write(f"{key}={value}\n")


def summary(markdown: str) -> None:
    """Append to the run summary, which is the only part of a job anyone reads later."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    print(markdown)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="the release tag, e.g. v0.2.0")
    ap.add_argument("--github-prerelease", default="false",
                    choices=["true", "false"],
                    help="whether the GitHub Release is marked as a pre-release")
    ap.add_argument("--notes-file", type=Path, default=None,
                    help="write the changelog section for this version here")
    ap.add_argument("--offline", action="store_true",
                    help="skip the registry queries; for tests and local rehearsals")
    ap.add_argument("--repo", type=Path, default=REPO)
    args = ap.parse_args(argv)

    repo: Path = args.repo
    gh_prerelease = args.github_prerelease == "true"

    # 1. The tag and the package version must be the same number.
    tagged = version_from_tag(args.tag)
    parse(tagged)
    version = python_version(repo)               # also refuses if the two files disagree
    if tagged != version:
        raise Refused(
            f"the release is tagged {args.tag} but the tree at that tag is version "
            f"{version}.\n"
            f"  Publishing would put {version} on PyPI under a release announcing "
            f"{tagged}, and the number cannot be taken back. Either the bump did not land "
            f"on this commit, or the tag was created on the wrong one.")

    # 2. A pre-release version must be marked as one on GitHub. The reverse is a warning:
    #    a final version under a pre-release label is misleading, but it does not put an
    #    unwanted build in front of anyone who typed `pip install memvara`.
    version_pre = is_prerelease(version)
    if version_pre and not gh_prerelease:
        raise Refused(
            f"{version} is a pre-release but the GitHub Release is not marked as one.\n"
            "  Tick 'Set as a pre-release' on the release and re-run. The label is what "
            "tells a reader of the releases page which builds are candidates, and it is "
            "also what keeps this pipeline off the npm `latest` dist-tag.")
    if gh_prerelease and not version_pre:
        summary(f"> **Note** — the GitHub Release is marked pre-release but `{version}` is "
                f"a final version. Publishing anyway; npm will use the "
                f"`{PRERELEASE_TAG}` dist-tag.")

    # 3. The release notes come from the changelog, so the changelog must have them.
    text = changelog.read(repo)
    notes = changelog.section(text, version)
    if not notes.strip():
        raise Refused(f"`## [{version}]` in CHANGELOG.md is empty. The GitHub Release body "
                      f"is generated from it, so there is nothing to publish as notes.")
    if args.notes_file:
        args.notes_file.write_text(notes + "\n", encoding="utf-8")

    npm_ver = npm_version(repo)
    npm_tag = dist_tag(npm_ver, github_prerelease=gh_prerelease)
    check_dist_tag(npm_ver, npm_tag)

    # 4. What each registry already has. See the module docstring for why "already there"
    #    is a skip and not a refusal.
    if args.offline:
        pypi_needed = npm_needed = True
    else:
        from release.registry import npm_versions, pypi_versions
        pypi_needed = version not in pypi_versions("memvara")
        npm_needed = npm_ver not in npm_versions("memvara")

    lines = [f"### memvara {version} (`{args.tag}`)", ""]
    lines.append(f"- PyPI `memvara=={version}`: "
                 + ("**will publish**" if pypi_needed else
                    "already on the index — **skipping**. If you expected a new upload, "
                    "the version was not bumped on this commit."))
    lines.append(f"- npm `memvara@{npm_ver}` (dist-tag `{npm_tag}`): "
                 + ("**will publish**" if npm_needed else
                    "already on the registry — **skipping**, which is the normal state: "
                    "the npm package is a name reservation on its own `0.0.x` line and an "
                    "ordinary Python release does not change it."))
    summary("\n".join(lines))

    emit(version=version, tag=args.tag,
         prerelease="true" if version_pre or gh_prerelease else "false",
         pypi_needed="true" if pypi_needed else "false",
         npm_version=npm_ver,
         npm_tag=npm_tag,
         npm_needed="true" if npm_needed else "false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Refused as exc:                       # pragma: no cover - CLI surface
        raise SystemExit(f"\n  REFUSED: {exc}\n")
