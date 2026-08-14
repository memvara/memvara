#!/usr/bin/env python3
"""Where the version is spelled, how it is bumped, and what the two packages promise.

Imported by `bump_version.py` and `check_release.py`; it holds no I/O policy of its own so
that both the bump and the release guard read the same definition of "the version".

## The two version lines, and why they are not one number

This project publishes two packages with the same name and they are **deliberately not
kept in step**:

* **`memvara` on PyPI** is the library. Its version is *the* release version: the git tag,
  the GitHub Release, `__version__`, and the changelog heading all carry it.
* **`memvara` on npm** is a name reservation. `npm/memvara/` exports
  `{implemented: false, ...}` and contains no client. It stays on a `0.0.x` line.

Locking them to one number was the obvious alternative and it is wrong here, for three
reasons that are worth having written down, because the question comes back:

1. **It would publish a lie.** `npm install memvara@0.4.0` alongside `pip install
   memvara==0.4.0` states that the two are the same software at the same version. One of
   them is an empty object. The number is the only thing a registry user has to go on.
2. **Every npm version is permanent and would be spent on nothing.** A year of ordinary
   Python releases would leave a dozen npm versions whose diff is one integer, and
   `unpublish` is 72 hours and narrower than people expect.
3. **The version grammars are not the same grammar.** PyPI takes PEP 440, npm takes
   semver, and they disagree exactly where releases get delicate: `0.2.0rc1` and
   `0.2.0-rc.1` are the same intent spelled two ways, and a coupled pipeline would need a
   translation layer whose bugs only appear on pre-releases — the releases least exercised
   and most likely to be cut in a hurry.

So the rule, enforced by `check_release.py` and repeated in `release/README.md`:

> **The npm package is published only when `npm/memvara/package.json` names a version the
> registry does not have.** An ordinary Python release leaves npm untouched. `0.0.x` is
> reserved for "there is no implementation"; a real JavaScript client, if one is ever
> written, starts its own semver line at `0.1.0` and the coupling question gets asked
> again then — with an actual client to reason about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: The repository, resolved from this file rather than configured — the same reasoning as
#: `publish_pypi.py`: a release procedure that ships with the package has no path to get
#: wrong on a different machine.
REPO = Path(__file__).resolve().parent.parent

#: PEP 440 as far as this project uses it: three numeric components, optionally a
#: pre-release. Deliberately narrower than PEP 440 permits. Epochs, post-releases and local
#: versions are all legal on PyPI and none of them have a sensible meaning for a library
#: whose whole release story is "bump one of three numbers", so accepting them here would
#: only widen the set of strings that can reach a tag comparison.
VERSION = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
                     r"(?P<pre>(?:a|b|rc)\d+)?$")

#: npm's grammar. Separate constant, not an alias: semver spells a pre-release
#: `0.2.0-rc.1` where PEP 440 spells it `0.2.0rc1`, and the two are not interchangeable in
#: either direction. This is the concrete form of the third argument in the module
#: docstring — a pipeline that kept the two packages on one number would need to translate
#: between these, and the translation is only exercised on pre-releases.
NPM_VERSION = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
                         r"(?:-(?P<pre>[0-9A-Za-z.-]+))?$")

PARTS = ("major", "minor", "patch")


class Refused(ValueError):
    """A refusal with a reason attached. Every one is cheaper than the alternative."""


@dataclass(frozen=True)
class Spelling:
    """One place a version is written, and the pattern that finds exactly that place.

    The pattern must match **once**. A version string is five characters long and appears
    in prose, in dependency pins and in changelog entries all over this repository; a
    rewrite that matched loosely would silently edit one of those. So each spelling carries
    an anchored pattern and the rewrite asserts a single hit rather than trusting it.

    `section` narrows the search to one TOML table, and it is not defensive padding: the
    first version of this file anchored on `^version = "..."$` and the test fixture's
    decoy — a `version` key under a different table — matched it. `[tool.poetry]`,
    `[tool.commitizen]` and `[tool.bumpversion]` all carry a key spelled exactly that way,
    and any of them landing in this project's `pyproject.toml` would have made the bump
    tool refuse with "2 lines" at best, or edit the wrong one at worst.
    """

    path: str
    #: A template taking `version`, compiled with re.MULTILINE against the file.
    pattern: str
    what: str
    #: A MULTILINE regex matching the line that opens the enclosing TOML table. The search
    #: runs from the end of that line to the next `[` at column zero.
    section: str | None = None

    def compiled(self, version: str) -> re.Pattern[str]:
        return re.compile(self.pattern.format(version=re.escape(version)), re.MULTILINE)

    def region(self, text: str) -> tuple[int, int]:
        """The slice of the file this spelling is allowed to match in."""
        if self.section is None:
            return 0, len(text)
        opener = re.search(self.section, text, re.MULTILINE)
        if opener is None:
            raise Refused(f"{self.path} has no {self.section!r} table, so there is nowhere "
                          f"for {self.what} to live")
        closer = re.compile(r"^\[", re.MULTILINE).search(text, opener.end())
        return opener.end(), closer.start() if closer else len(text)


#: Every file that states the Python package's version. Two, and nothing in the build
#: keeps them equal —
#: `tests/test_packaging.py::test_the_version_is_the_same_string_in_both_places_that_state_it`
#: is what does, which is why a one-sided bump fails the suite instead of shipping a wheel
#: whose `memvara-mcp --version` disagrees with `pip show`.
PYTHON_SPELLINGS = (
    Spelling("pyproject.toml", r'^version = "{version}"$', "the installer's version",
             section=r"^\[project\]$"),
    Spelling("memvara/__init__.py", r'^__version__ = "{version}"$', "the program's version"),
)

#: The npm placeholder's own, separate line. See the module docstring for why it is separate.
NPM_SPELLING = Spelling("npm/memvara/package.json", r'^  "version": "{version}",$',
                        "the npm placeholder's version")


def parse(version: str, *, pattern: re.Pattern[str] = VERSION) -> tuple[int, int, int, str]:
    m = pattern.match(version)
    if not m:
        raise Refused(f"{version!r} is not a version this project can release: expected "
                      f"MAJOR.MINOR.PATCH, optionally followed by a1/b1/rc1")
    g = m.groupdict()
    return int(g["major"]), int(g["minor"]), int(g["patch"]), g.get("pre") or ""


def is_prerelease(version: str, *, pattern: re.Pattern[str] = VERSION) -> bool:
    """True for `0.2.0rc1`, and for `0.2.0-rc.1` under `NPM_VERSION`.

    The npm `latest` dist-tag hangs off this, which is the reason it takes a pattern rather
    than assuming PEP 440: reading a semver pre-release with the PEP 440 pattern does not
    return "no pre-release", it fails to parse — and a guard that errors on the input it
    exists to catch is one somebody disables.
    """
    return bool(parse(version, pattern=pattern)[3])


def bump(current: str, part: str, *, pattern: re.Pattern[str] = VERSION) -> str:
    """The next version, dropping any pre-release suffix.

    `0.2.0rc1` + patch is `0.2.1` and not `0.2.0rc2`: bumping *out* of a pre-release is
    what a `part` bump means, and iterating a release candidate is an explicit
    `--version 0.2.0rc2`. Guessing between the two would be guessing about the thing the
    caller is least likely to want silently decided.
    """
    if part not in PARTS:
        raise Refused(f"{part!r} is not one of {', '.join(PARTS)}")
    major, minor, patch, pre = parse(current, pattern=pattern)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    # A patch bump off a pre-release would go backwards: 0.2.0rc1 -> 0.2.1 is right,
    # 0.2.0rc1 -> 0.2.0 is the *release* of that candidate and is `--version 0.2.0`.
    return f"{major}.{minor}.{patch + 1}" if not pre else f"{major}.{minor}.{patch}"


def sorts_after(new: str, current: str, *, pattern: re.Pattern[str] = VERSION) -> bool:
    """Does `new` come strictly after `current`?

    Pre-releases sort before their own final release, which is the only ordering question
    this project can actually hit: `0.2.0rc1 < 0.2.0`. The empty pre-release suffix has to
    sort *last* within a version triple, so it is mapped to a sentinel that compares above
    any real one rather than to `""`, which would sort first and make releasing a
    candidate look like a downgrade.
    """
    def key(v: str) -> tuple[int, int, int, str]:
        major, minor, patch, pre = parse(v, pattern=pattern)
        return (major, minor, patch, pre or "~")   # "~" > "a", "b", "rc"
    return key(new) > key(current)


def read(spelling: Spelling, repo: Path = REPO) -> str:
    """The version currently written in one file.

    Reads the literal rather than importing the package: this runs in the release guard,
    where importing `memvara` would mean installing numpy first, and where the whole point
    is to check the file on disk against a tag.
    """
    path = repo / spelling.path
    if not path.is_file():
        raise Refused(f"{spelling.path} does not exist under {repo}")
    text = path.read_text(encoding="utf-8")
    start, stop = spelling.region(text)
    generic = re.compile(spelling.pattern.format(version=r'(?P<v>[^"]+)'), re.MULTILINE)
    found = generic.findall(text[start:stop])
    if len(found) != 1:
        raise Refused(f"{spelling.path} has {len(found)} lines stating {spelling.what}; "
                      f"expected exactly one matching {generic.pattern}")
    return found[0]


def python_version(repo: Path = REPO) -> str:
    """The Python package's version, refusing if the two spellings disagree.

    They can only disagree via a hand-edit, and this is the earliest point at which that is
    catchable in a release: everything downstream — the tag check, the changelog lookup,
    the registry query — would otherwise be checking one of the two and shipping the other.
    """
    seen = {s.path: read(s, repo) for s in PYTHON_SPELLINGS}
    if len(set(seen.values())) != 1:
        detail = "\n".join(f"    {p}: {v}" for p, v in seen.items())
        raise Refused("the version is spelled two different ways:\n" + detail +
                      "\n  Nothing in the build keeps them equal. Fix by hand, or re-run "
                      "the bump workflow, which writes both.")
    return next(iter(seen.values()))


def npm_version(repo: Path = REPO) -> str:
    return read(NPM_SPELLING, repo)


def rewrite(spelling: Spelling, old: str, new: str, repo: Path = REPO) -> None:
    """Replace one version spelling in place, or refuse.

    `re.sub` with `count=1` would be the short way and would quietly do nothing when the
    pattern missed. This asserts the hit count first, because "the bump ran and changed
    nothing" is the failure that gets discovered by a mismatched wheel three steps later.
    """
    path = repo / spelling.path
    text = path.read_text(encoding="utf-8")
    start, stop = spelling.region(text)
    region = text[start:stop]
    hits = spelling.compiled(old).findall(region)
    if len(hits) != 1:
        raise Refused(f"expected exactly one {spelling.what} line in {spelling.path} "
                      f"holding {old}, found {len(hits)}")
    # A lambda rather than a replacement template: `new` is interpolated as data, so a
    # backslash in it can never be read as a group reference.
    region = spelling.compiled(old).sub(lambda m: m.group(0).replace(old, new),
                                        region, count=1)
    path.write_text(text[:start] + region + text[stop:], encoding="utf-8")
