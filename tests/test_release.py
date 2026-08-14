"""The release tooling, which is the only code here whose mistakes are permanent.

`release/` is not part of the installed package — the wheel target names `memvara` and
only `memvara` — so nothing in this file protects a user at runtime. What it protects is
the one action in this project that cannot be undone: a version number spent on the wrong
artifact. PyPI will not take a number back after a yank or a delete, and npm's `unpublish`
is 72 hours and narrower than people expect.

Two kinds of test are here and they fail for different reasons, which is worth knowing
when one of them goes red:

* **Against a synthetic repository** built in `tmp_path`. These test the logic — bumping,
  changelog surgery, the refusals — and a failure means the logic is wrong.
* **Against this repository.** `release/versions.py` finds each version by an anchored
  regex, so reformatting `pyproject.toml` or `package.json` can stop the bump tool from
  matching anything. A tool that runs, matches nothing and reports success is the failure
  mode; these tests are what turns it into a red suite instead.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import pytest

import memvara

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:                    # pragma: no cover - depends on invocation
    # `release` is not installed, and it must not be: it is release tooling, not library
    # code. Under `python -m pytest` from the repository root the root is already on the
    # path; under a bare `pytest` it is not, and the difference should not decide whether
    # these tests run.
    sys.path.insert(0, str(REPO))

from release import bump_version, changelog, check_release, versions  # noqa: E402
from release.versions import NPM_VERSION, Refused  # noqa: E402


# -- a synthetic repository ---------------------------------------------------------

PYPROJECT = """\
[project]
name = "memvara"
version = "{version}"
dependencies = ["numpy>=1.24"]

[tool.poetry]
# A decoy, and not a hypothetical one: `[tool.poetry]`, `[tool.commitizen]` and
# `[tool.bumpversion]` all carry a key spelled exactly `version = "..."`. The first
# version of the bump tool anchored on that line alone and this fixture caught it.
version = "9.9.9"
"""

INIT = '''\
"""A stand-in for the real package."""

SCHEMA_VERSION = 4
__version__ = "{version}"
'''

PACKAGE_JSON = """\
{{
  "name": "memvara",
  "version": "{version}",
  "license": "Apache-2.0"
}}
"""

CHANGELOG = """\
# Changelog

## [Unreleased]

### Added

- A thing worth telling someone about.

## [0.1.0] — 2026-08-10

First release.

## Wave 1

### Added

- The very beginning, from before this project had versions at all.
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with every file the release tooling writes, and nothing else."""
    build(tmp_path, python="0.1.0", npm="0.0.1")
    return tmp_path


def build(root: Path, *, python: str, npm: str, changelog_text: str = CHANGELOG) -> None:
    (root / "memvara").mkdir(exist_ok=True)
    (root / "npm" / "memvara").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(PYPROJECT.format(version=python))
    (root / "memvara" / "__init__.py").write_text(INIT.format(version=python))
    (root / "npm" / "memvara" / "package.json").write_text(PACKAGE_JSON.format(version=npm))
    (root / "CHANGELOG.md").write_text(changelog_text)


# -- reading and parsing versions ---------------------------------------------------


@pytest.mark.parametrize("text", ["0.1.0", "1.0.0", "0.2.0rc1", "10.20.30", "0.2.0b2"])
def test_the_versions_this_project_can_ship_parse(text: str) -> None:
    assert versions.parse(text)


@pytest.mark.parametrize("text", [
    "0.1", "v0.1.0", "0.1.0.post1", "1!0.1.0", "0.1.0+local", "0.2.0-rc.1", "", "latest",
])
def test_versions_outside_this_project_s_grammar_are_refused(text: str) -> None:
    """Deliberately narrower than PEP 440.

    Epochs, post-releases and local versions are all legal on PyPI and none of them mean
    anything for a library whose release story is "bump one of three numbers". Accepting
    them would only widen the set of strings that can reach a tag comparison. `0.2.0-rc.1`
    is in this list because it is semver, not PEP 440 — it is legal for the npm package
    and not for this one, which is the whole reason the two grammars are separate
    constants.
    """
    with pytest.raises(Refused):
        versions.parse(text)


def test_the_npm_grammar_takes_a_semver_prerelease_and_the_python_one_does_not() -> None:
    assert versions.parse("0.0.2-rc.1", pattern=NPM_VERSION)
    with pytest.raises(Refused):
        versions.parse("0.0.2rc1", pattern=NPM_VERSION)


@pytest.mark.parametrize("current,part,expected", [
    ("0.1.0", "patch", "0.1.1"),
    ("0.1.0", "minor", "0.2.0"),
    ("0.1.0", "major", "1.0.0"),
    ("0.1.9", "minor", "0.2.0"),
    ("1.4.2", "major", "2.0.0"),
])
def test_a_part_bump_moves_one_component_and_zeroes_the_ones_below_it(
        current: str, part: str, expected: str) -> None:
    assert versions.bump(current, part) == expected


def test_bumping_out_of_a_prerelease_drops_the_suffix() -> None:
    """`0.2.0rc1` + patch is `0.2.0`, not `0.2.1` and not `0.2.0rc2`.

    Releasing the candidate is what a patch bump off a candidate means. Iterating it — to
    `rc2` — is a different intent and needs saying explicitly with `--version`, because
    guessing between the two would be guessing about the thing the caller is least likely
    to want silently decided.
    """
    assert versions.bump("0.2.0rc1", "patch") == "0.2.0"
    assert versions.bump("0.2.0rc1", "minor") == "0.3.0"
    assert versions.bump("0.2.0rc1", "major") == "1.0.0"


def test_an_unknown_part_is_refused() -> None:
    with pytest.raises(Refused):
        versions.bump("0.1.0", "epoch")


@pytest.mark.parametrize("new,current", [
    ("0.1.1", "0.1.0"), ("0.2.0", "0.1.9"), ("1.0.0", "0.99.0"),
    ("0.2.0", "0.2.0rc1"), ("0.2.0rc2", "0.2.0rc1"), ("0.2.0b1", "0.2.0a9"),
])
def test_these_versions_go_forwards(new: str, current: str) -> None:
    assert versions.sorts_after(new, current)


@pytest.mark.parametrize("new,current", [
    ("0.1.0", "0.1.0"), ("0.1.0", "0.2.0"), ("0.2.0rc1", "0.2.0"), ("0.2.0rc1", "0.2.0rc1"),
])
def test_these_do_not(new: str, current: str) -> None:
    """`0.2.0rc1` after `0.2.0` is the case worth having a test for.

    A pre-release sorts *before* its own final release, so releasing a candidate for a
    version that already shipped is a step backwards. Spelling the empty suffix as `""`
    would have sorted it first and made exactly this look like progress.
    """
    assert not versions.sorts_after(new, current)


def test_prerelease_detection_reads_the_right_grammar() -> None:
    assert versions.is_prerelease("0.2.0rc1")
    assert not versions.is_prerelease("0.2.0")
    assert versions.is_prerelease("0.0.2-rc.1", pattern=NPM_VERSION)
    assert not versions.is_prerelease("0.0.2", pattern=NPM_VERSION)


# -- rewriting the files ------------------------------------------------------------


def test_a_bump_rewrites_the_version_and_leaves_lookalikes_alone(repo: Path) -> None:
    """The decoy `version = "9.9.9"` under `[tool.something]` must survive.

    A version string is a handful of characters and turns up in prose, in pins and in
    changelog entries; a loose rewrite edits one of those and nothing notices until an
    installer does.
    """
    for spelling in versions.PYTHON_SPELLINGS:
        versions.rewrite(spelling, "0.1.0", "0.2.0", repo)
    assert 'version = "0.2.0"' in (repo / "pyproject.toml").read_text()
    assert 'version = "9.9.9"' in (repo / "pyproject.toml").read_text()
    assert '__version__ = "0.2.0"' in (repo / "memvara" / "__init__.py").read_text()
    assert "SCHEMA_VERSION = 4" in (repo / "memvara" / "__init__.py").read_text()


def test_a_rewrite_that_would_match_nothing_refuses_instead_of_succeeding(repo: Path) -> None:
    """The failure this prevents is a bump that runs, changes nothing and exits 0.

    `re.sub` with `count=1` is the short way to write the rewrite and it does exactly that
    when the pattern misses, which is discovered three steps later by a wheel whose
    metadata does not match its tag.
    """
    with pytest.raises(Refused):
        versions.rewrite(versions.PYTHON_SPELLINGS[0], "0.9.9", "1.0.0", repo)


def test_reading_a_version_from_a_file_that_states_it_twice_refuses(repo: Path) -> None:
    path = repo / "memvara" / "__init__.py"
    path.write_text(path.read_text() + '\n__version__ = "0.1.0"\n')
    with pytest.raises(Refused):
        versions.python_version(repo)


def test_two_files_that_disagree_about_the_version_refuse(repo: Path) -> None:
    """The earliest catchable point for a hand-edited half-bump.

    Everything downstream — the tag check, the changelog lookup, the registry query —
    would otherwise check one of the two spellings and ship the other.
    """
    versions.rewrite(versions.PYTHON_SPELLINGS[0], "0.1.0", "0.2.0", repo)
    with pytest.raises(Refused) as caught:
        versions.python_version(repo)
    assert "two different ways" in str(caught.value)


def test_the_search_is_confined_to_the_project_table(repo: Path) -> None:
    """Not defensive padding — see the decoy in the fixture above.

    Reading the whole file would find two `version = "..."` lines the moment any tool that
    keeps its own version key is configured in `pyproject.toml`, and the bump would either
    refuse for a reason nobody could act on or edit the wrong one.
    """
    path = repo / "pyproject.toml"
    assert versions.read(versions.PYTHON_SPELLINGS[0], repo) == "0.1.0"
    path.write_text(path.read_text().replace("[project]", "[project-ish]"))
    with pytest.raises(Refused) as caught:
        versions.read(versions.PYTHON_SPELLINGS[0], repo)
    assert "table" in str(caught.value)


def test_a_missing_file_refuses_by_name(repo: Path) -> None:
    (repo / "npm" / "memvara" / "package.json").unlink()
    with pytest.raises(Refused) as caught:
        versions.npm_version(repo)
    assert "package.json" in str(caught.value)


# -- the changelog ------------------------------------------------------------------


def test_a_section_ends_at_the_next_heading_whatever_that_heading_is() -> None:
    """`## Wave 1` has to terminate the section above it as firmly as a version does.

    The tail of the real changelog predates versioning and is full of those. A pattern
    that only knew about `## [x.y.z]` headings would swallow them into the release above.
    """
    assert changelog.section(CHANGELOG, "0.1.0") == "First release."
    assert "Wave" not in changelog.section(CHANGELOG, "0.1.0")


def test_only_bracketed_headings_count_as_versions() -> None:
    assert changelog.released_versions(CHANGELOG) == ["0.1.0"]


def test_asking_for_a_version_with_no_heading_names_the_ones_that_exist() -> None:
    with pytest.raises(Refused) as caught:
        changelog.section(CHANGELOG, "0.9.0")
    assert "0.1.0" in str(caught.value)


def test_closing_a_release_moves_the_body_down_and_leaves_unreleased_empty() -> None:
    out = changelog.open_release(CHANGELOG, "0.2.0", date(2026, 8, 14))
    assert changelog.section(out, changelog.UNRELEASED) == ""
    assert "A thing worth telling someone about." in changelog.section(out, "0.2.0")
    assert "## [0.2.0] — 2026-08-14" in out
    # The older release is untouched and still below the new one.
    assert out.index("## [0.2.0]") < out.index("## [0.1.0]")


def test_an_empty_unreleased_section_refuses() -> None:
    """A release with no notes is one nobody can tell apart from the previous one."""
    empty = CHANGELOG.replace("### Added\n\n- A thing worth telling someone about.\n", "")
    with pytest.raises(Refused) as caught:
        changelog.open_release(empty, "0.2.0", date(2026, 8, 14))
    assert "empty" in str(caught.value)


def test_an_empty_section_can_be_released_deliberately_and_says_so() -> None:
    empty = CHANGELOG.replace("### Added\n\n- A thing worth telling someone about.\n", "")
    out = changelog.open_release(empty, "0.2.0", date(2026, 8, 14), allow_empty=True)
    assert changelog.section(out, "0.2.0") == changelog.NO_CHANGES


def test_re_cutting_a_version_that_already_has_a_heading_refuses() -> None:
    with pytest.raises(Refused):
        changelog.open_release(CHANGELOG, "0.1.0", date(2026, 8, 14))


def test_a_changelog_with_no_unreleased_section_refuses() -> None:
    with pytest.raises(Refused):
        changelog.open_release("# Changelog\n\n## [0.1.0] — 2026-08-10\n\nx\n",
                               "0.2.0", date(2026, 8, 14))


def test_a_missing_changelog_refuses(tmp_path: Path) -> None:
    with pytest.raises(Refused):
        changelog.read(tmp_path)


# -- the bump, end to end -----------------------------------------------------------


def test_a_minor_bump_writes_both_versions_and_the_changelog(repo: Path) -> None:
    assert bump_version.main(["--part", "minor", "--offline", "--repo", str(repo)]) == 0
    assert versions.python_version(repo) == "0.2.0"
    text = (repo / "CHANGELOG.md").read_text()
    assert "## [0.2.0] —" in text
    assert changelog.section(text, changelog.UNRELEASED) == ""


def test_an_ordinary_python_release_does_not_touch_the_npm_placeholder(repo: Path) -> None:
    """The version relationship, as a test rather than as a paragraph.

    The npm package is a name reservation with no implementation. Publishing a matching
    number for it on every Python release would assert an equivalence between a library
    and an empty object, and would spend a permanent npm version on nothing.
    """
    bump_version.main(["--part", "minor", "--offline", "--repo", str(repo)])
    assert versions.npm_version(repo) == "0.0.1"


def test_the_npm_placeholder_moves_only_when_asked_and_on_its_own_line(repo: Path) -> None:
    bump_version.main(["--part", "minor", "--npm-part", "patch", "--offline",
                       "--repo", str(repo)])
    assert versions.python_version(repo) == "0.2.0"
    assert versions.npm_version(repo) == "0.0.2"
    # The rewrite went through the JSON without reformatting the rest of it.
    assert json.loads((repo / "npm" / "memvara" / "package.json").read_text()) == {
        "name": "memvara", "version": "0.0.2", "license": "Apache-2.0"}


def test_an_explicit_version_is_taken_as_given(repo: Path) -> None:
    bump_version.main(["--version", "0.2.0rc1", "--offline", "--repo", str(repo)])
    assert versions.python_version(repo) == "0.2.0rc1"


def test_a_version_that_goes_backwards_is_refused_before_anything_is_written(
        repo: Path) -> None:
    before = (repo / "pyproject.toml").read_text()
    with pytest.raises(Refused):
        bump_version.main(["--version", "0.0.9", "--offline", "--repo", str(repo)])
    assert (repo / "pyproject.toml").read_text() == before


def test_a_bad_changelog_leaves_the_versions_alone(repo: Path) -> None:
    """A half-applied bump is the one state that looks like a finished one.

    The changelog rewrite is computed before any file is touched precisely so that a
    refusal here cannot leave `pyproject.toml` bumped and the notes unwritten.
    """
    (repo / "CHANGELOG.md").write_text("# Changelog\n\n## [0.1.0] — 2026-08-10\n\nx\n")
    with pytest.raises(Refused):
        bump_version.main(["--part", "minor", "--offline", "--repo", str(repo)])
    assert versions.python_version(repo) == "0.1.0"


def test_an_npm_version_that_goes_backwards_is_refused(repo: Path) -> None:
    with pytest.raises(Refused):
        bump_version.main(["--part", "patch", "--npm-version", "0.0.0", "--offline",
                           "--repo", str(repo)])


def test_the_bump_hands_the_workflow_what_it_needs(repo: Path, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "outputs"
    out.write_text("")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    bump_version.main(["--part", "minor", "--offline", "--repo", str(repo)])
    written = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert written["version"] == "0.2.0"
    assert written["tag"] == "v0.2.0"
    assert written["branch"] == "release/v0.2.0"
    assert written["prerelease"] == "false"
    assert written["npm_bumped"] == "false"


def test_a_prerelease_bump_says_so_on_the_way_out(repo: Path, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "outputs"
    out.write_text("")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    bump_version.main(["--version", "0.2.0rc1", "--offline", "--repo", str(repo)])
    written = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert written["prerelease"] == "true"


# -- the release guard --------------------------------------------------------------


def test_a_tag_without_the_v_prefix_is_refused() -> None:
    """One release, one tag spelling. Every downstream URL is built from it."""
    assert check_release.version_from_tag("v0.2.0") == "0.2.0"
    with pytest.raises(Refused):
        check_release.version_from_tag("0.2.0")


@pytest.mark.parametrize("npm_ver,gh_pre,expected", [
    ("0.0.2", False, "latest"),
    ("0.0.2", True, "next"),
    ("0.0.2-rc.1", False, "next"),
    ("0.0.2-rc.1", True, "next"),
])
def test_a_prerelease_never_reaches_the_latest_dist_tag(npm_ver: str, gh_pre: bool,
                                                        expected: str) -> None:
    """`latest` is what a bare `npm install memvara` resolves to.

    Either signal is enough to withhold it: a version that says it is a candidate, or a
    human who ticked the box. The second exists because a release can be a candidate
    without its number saying so.
    """
    assert check_release.dist_tag(npm_ver, github_prerelease=gh_pre) == expected


def test_the_dist_tag_rule_holds_even_if_the_computation_is_bypassed() -> None:
    check_release.check_dist_tag("0.0.2", "latest")
    with pytest.raises(Refused):
        check_release.check_dist_tag("0.0.2-rc.1", "latest")


def test_the_tag_and_the_shipped_version_must_be_the_same_number(repo: Path) -> None:
    """The failure this whole file exists for: a release tagged v0.2.0 that ships 0.1.0.

    Nothing else connects the tag to the version. The tag is typed into GitHub's release
    form; the version lives in two files. They agree by habit until they do not.
    """
    with pytest.raises(Refused) as caught:
        check_release.main(["--tag", "v0.2.0", "--offline", "--repo", str(repo)])
    assert "0.1.0" in str(caught.value) and "v0.2.0" in str(caught.value)


def test_a_matching_tag_passes_and_reports_what_it_will_publish(
        repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "outputs"
    out.write_text("")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    notes = tmp_path / "notes.md"
    assert check_release.main(["--tag", "v0.1.0", "--offline", "--repo", str(repo),
                               "--notes-file", str(notes)]) == 0
    written = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert written["version"] == "0.1.0"
    assert written["npm_version"] == "0.0.1"
    assert written["npm_tag"] == "latest"
    assert written["prerelease"] == "false"
    assert notes.read_text().strip() == "First release."


def test_the_release_notes_are_the_changelog_section_and_not_a_second_copy(
        repo: Path, tmp_path: Path) -> None:
    notes = tmp_path / "notes.md"
    check_release.main(["--tag", "v0.1.0", "--offline", "--repo", str(repo),
                        "--notes-file", str(notes)])
    assert notes.read_text().strip() == changelog.section(CHANGELOG, "0.1.0")


def test_a_version_with_no_changelog_section_is_refused(repo: Path) -> None:
    build(repo, python="0.2.0", npm="0.0.1")
    with pytest.raises(Refused) as caught:
        check_release.main(["--tag", "v0.2.0", "--offline", "--repo", str(repo)])
    assert "CHANGELOG" in str(caught.value)


def test_a_prerelease_version_must_be_marked_as_one_on_github(repo: Path) -> None:
    build(repo, python="0.2.0rc1", npm="0.0.1",
          changelog_text=CHANGELOG.replace("## [Unreleased]",
                                           "## [Unreleased]\n\n## [0.2.0rc1] — 2026-08-14"))
    with pytest.raises(Refused) as caught:
        check_release.main(["--tag", "v0.2.0rc1", "--offline", "--repo", str(repo)])
    assert "pre-release" in str(caught.value)

    assert check_release.main(["--tag", "v0.2.0rc1", "--github-prerelease", "true",
                               "--offline", "--repo", str(repo)]) == 0


def test_a_final_version_under_a_prerelease_label_is_allowed_but_noted(
        repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The asymmetry is deliberate.

    A candidate published as final puts an untested build in front of everyone who typed
    `pip install memvara`. A final published under a candidate label is only misleading,
    and the npm dist-tag still moves to `next`, so nothing is handed to anyone by default.
    """
    out = tmp_path / "outputs"
    out.write_text("")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert check_release.main(["--tag", "v0.1.0", "--github-prerelease", "true",
                               "--offline", "--repo", str(repo)]) == 0
    written = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert written["npm_tag"] == "next"
    assert written["prerelease"] == "true"


def test_registry_answers_decide_which_jobs_run(repo: Path, tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """Already-published is a skip, not a refusal, and this is why.

    A run that published to PyPI and then failed on npm has to be re-runnable. A guard
    that refuses because PyPI already holds the version turns the re-run into a dead end,
    and the way out becomes publishing npm by hand — the manual step the pipeline exists
    to remove. Two commits cannot share a tag, so "PyPI already has 0.1.0" and "this exact
    release already published" are the same fact.
    """
    import release.registry as registry
    monkeypatch.setattr(registry, "pypi_versions", lambda name: {"0.1.0"})
    monkeypatch.setattr(registry, "npm_versions", lambda name: set())
    out = tmp_path / "outputs"
    out.write_text("")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert check_release.main(["--tag", "v0.1.0", "--repo", str(repo)]) == 0
    written = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert written["pypi_needed"] == "false"
    assert written["npm_needed"] == "true"


def test_an_unreachable_registry_stops_the_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """"The registry did not answer" must never be read as "nothing is published".

    The check exists because the action it guards cannot be undone.
    """
    import release.registry as registry

    class Boom:
        class RequestException(Exception):
            pass

        @staticmethod
        def get(url: str, timeout: int) -> None:
            raise Boom.RequestException("no route to host")

    monkeypatch.setitem(sys.modules, "requests", Boom)
    with pytest.raises(Refused):
        registry.pypi_versions("memvara")


def test_a_registry_404_means_the_project_does_not_exist_yet(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import release.registry as registry

    class Resp:
        status_code = 404

        @staticmethod
        def json() -> dict:                      # pragma: no cover - never reached on 404
            raise AssertionError("a 404 body must not be parsed")

    class Fake:
        class RequestException(Exception):
            pass

        @staticmethod
        def get(url: str, timeout: int) -> Resp:
            return Resp()

    monkeypatch.setitem(sys.modules, "requests", Fake)
    assert registry.pypi_versions("memvara") == set()
    assert registry.npm_versions("memvara") == set()


# -- against this repository, not a synthetic one -----------------------------------


def test_the_bump_tool_can_still_find_the_version_in_the_real_files() -> None:
    """If this fails, the bump tool would run, match nothing, and report success.

    Both files are edited by hand often enough that a reformat is plausible — a different
    quote style in `pyproject.toml`, a re-indented `package.json` — and neither would
    break anything else in this repository.
    """
    assert versions.python_version(REPO) == memvara.__version__
    assert versions.parse(versions.npm_version(REPO), pattern=NPM_VERSION)


def test_the_npm_placeholder_is_on_its_own_zero_zero_line() -> None:
    """The version relationship between the two packages, asserted rather than described.

    `0.0.x` is reserved for "there is no implementation". A real JavaScript client would
    start its own semver line at `0.1.0`, and at that point the question of whether to
    couple the two numbers gets asked again — with an actual client to reason about. This
    test is what makes that a decision instead of a drift.
    """
    assert versions.npm_version(REPO).startswith("0.0."), (
        "npm/memvara is a name reservation with no implementation; see release/versions.py")


def test_the_real_changelog_is_shaped_the_way_the_release_tooling_reads_it() -> None:
    """`## [Unreleased]` is what the bump closes and the release notes come out of.

    Renaming or dropping it breaks the pipeline in the bump job, which is the good place
    for it to break — but only if something notices before then.
    """
    text = changelog.read(REPO)
    assert changelog.UNRELEASED in changelog.HEADING.findall(text)[0]
    assert memvara.__version__ in changelog.released_versions(text)


def test_every_release_heading_in_the_real_changelog_is_a_version_this_project_can_ship(
) -> None:
    for version in changelog.released_versions(changelog.read(REPO)):
        assert versions.parse(version)


def test_the_release_workflows_exist_and_name_the_scripts_they_run() -> None:
    """The scripts above are only reachable through the workflows that call them.

    A rename here is silent: a workflow referring to a module that no longer exists fails
    at release time, on the one run where a red job costs the most.
    """
    workflows = REPO / ".github" / "workflows"
    bump = (workflows / "version-bump.yml").read_text()
    release = (workflows / "release.yml").read_text()
    assert "release.bump_version" in bump
    assert "release.check_release" in release
    # The publish jobs must ask GitHub for an OIDC token; without this they fall back to
    # looking for a stored credential that deliberately does not exist.
    assert release.count("id-token: write") >= 2


def test_no_workflow_carries_a_literal_credential() -> None:
    """A grep, kept as a test, for the class of mistake this repository has already had.

    The pipeline's whole point is that no registry token exists to leak. A hard-coded one
    would defeat it silently — the workflow would work, which is exactly why nobody would
    look.
    """
    shapes = re.compile(r"pypi-[A-Za-z0-9_\-]{20,}|npm_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}")
    for path in sorted((REPO / ".github" / "workflows").glob("*.yml")):
        assert not shapes.search(path.read_text()), f"{path.name} contains a literal token"
