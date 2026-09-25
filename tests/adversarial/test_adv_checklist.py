"""The coverage checklist and its ratchet (tests/harness/checklist.py).

The first tests check this repository: every gap is in the baseline, every baseline line
is still a gap, every covers mark can be read and names an item, and a new switch shows
up as a gap. The rest check how the checklist reads the code and the tests, most of them
on small inputs written for the test.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import textwrap

import pytest

from harness import checklist, tiers
from memvara.server import config as server_config
from memvara.server import mcp as mcp_module
from memvara.server.tools import BY_NAME


@dataclasses.dataclass(frozen=True)
class Repository:
    """This checkout's checklist, the covers marks of its tests, and its baseline."""

    items: set[str]
    scan: checklist.Scan
    baseline: set[str]

    @property
    def gaps(self) -> set[str]:
        return checklist.gaps(self.items, self.scan.covered)


@pytest.fixture(scope="module")
def repo() -> Repository:
    return Repository(checklist.items(), checklist.scan(), checklist.read_baseline())


def _listed(lines: list[str]) -> str:
    return "".join(f"\n  {line}" for line in lines)


def test_every_gap_is_listed_in_the_baseline(repo: Repository) -> None:
    """An item with no test fails here, for example a tool added without one. The fix is
    a test that covers it, not a new line in the baseline."""
    new = sorted(repo.gaps - repo.baseline)
    assert not new, (
        "No test covers the items below, and tests/harness/checklist_baseline.txt does "
        "not list them. Write a test for each one and mark it with "
        "@pytest.mark.covers(...), instead of adding it to the baseline:" + _listed(new))


def test_every_line_of_the_baseline_is_still_a_gap(repo: Repository) -> None:
    """The baseline only shrinks. A line whose item a test now covers, or whose item no
    longer exists, must go, or it would hide that item if its test were later lost."""
    stale = sorted(repo.baseline - repo.gaps)
    assert not stale, (
        "The lines below are in tests/harness/checklist_baseline.txt, but they are no "
        "longer gaps: a test covers each one now, or its item no longer exists. Delete "
        "them from the baseline:" + _listed(stale))


def test_every_covers_mark_can_be_read(repo: Repository) -> None:
    assert not repo.scan.problems, _listed(sorted(repo.scan.problems))


def test_every_covers_mark_names_an_item_on_the_checklist(repo: Repository) -> None:
    wrong = checklist.misdeclared(repo.scan, repo.items)
    assert not wrong, _listed(wrong)


def test_the_covers_marker_is_registered(request: pytest.FixtureRequest) -> None:
    assert any(line.startswith("covers(") for line in request.config.getini("markers"))


def test_a_new_feature_switch_is_a_gap_the_baseline_does_not_list(
        repo: Repository, monkeypatch: pytest.MonkeyPatch) -> None:
    """The design's proof for this checklist: a switch added to FEATURES with no test
    fails the fast tier, and the failure names the switch."""
    monkeypatch.setattr(server_config, "FEATURES",
                        (*server_config.FEATURES, "brand_new_switch"))
    assert checklist.gaps(checklist.items(), repo.scan.covered) - repo.baseline == {
        "switch:brand_new_switch"}


def test_an_exempt_item_is_neither_a_gap_nor_in_the_baseline(repo: Repository) -> None:
    """A rule that no test can check stays on the checklist but is never a gap, so the
    baseline can reach empty."""
    for item in checklist.EXEMPT:
        assert item not in repo.gaps, item
        assert item not in repo.baseline, item


def test_every_exemption_names_an_item_on_the_checklist(repo: Repository) -> None:
    """An exemption whose rule was removed or renamed would outlive it unnoticed."""
    assert set(checklist.EXEMPT) <= repo.items, sorted(set(checklist.EXEMPT) - repo.items)


def test_no_test_covers_an_exempt_item(repo: Repository) -> None:
    """An item a test can cover is not exempt: its exemption must go."""
    assert not set(checklist.EXEMPT) & repo.scan.covered


def test_every_exemption_gives_its_reason() -> None:
    for item, reason in checklist.EXEMPT.items():
        assert item.startswith("inv:") and len(reason.split()) >= 8, (item, reason)


def test_the_baseline_leaves_out_comments_and_blank_lines(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "baseline.txt"
    path.write_text("# a comment\n\ntool:memory_recall\n  env:MEMVARA_DB  \n",
                    encoding="utf-8")
    assert checklist.read_baseline(path) == {"tool:memory_recall", "env:MEMVARA_DB"}


def test_every_tool_and_every_switch_is_an_item() -> None:
    assert "tool:memory_recall" in checklist.tool_items()
    switches = checklist.switch_items()
    assert {"switch:documents", "switch:read_only", "switch:anchored"} <= set(switches)


def test_a_pair_is_a_tool_whose_listing_a_switch_changes() -> None:
    """documents hides the four document tools and leaves memory_stats alone. A switch
    that belongs to the plugin, such as index_command, changes no tool."""
    pairs = set(checklist.tool_switch_items())
    assert {pair for pair in pairs if pair.endswith("/documents")} == {
        "tool-switch:memory_add_document/documents",
        "tool-switch:memory_delete_document/documents",
        "tool-switch:memory_get_document/documents",
        "tool-switch:memory_list_documents/documents"}
    assert "tool-switch:memory_add/read_only" in pairs
    assert "tool-switch:memory_recall/anchored" in pairs
    assert not any(pair.endswith("/index_command") for pair in pairs)


def test_a_switch_that_reveals_a_tool_makes_a_pair_too(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool that belongs to a feature which is off by default is listed only when the
    feature is switched on. Its pair must be found all the same."""
    probe = dataclasses.replace(BY_NAME["memory_stats"], name="memory_probe",
                                feature="extraction_chunks")
    monkeypatch.setattr(mcp_module, "TOOLS", (*mcp_module.TOOLS, probe))
    assert "tool-switch:memory_probe/extraction_chunks" in checklist.tool_switch_items()


def test_the_default_listing_follows_the_features_that_are_off_by_default(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The listing that every switch is compared against must read the features that are
    off by default when it runs, as each flipped listing does. Read once at import, it
    kept the document tools while the flipped listing showed them too, and the four
    document pairs vanished."""
    monkeypatch.setattr(server_config, "FEATURES_OFF_BY_DEFAULT",
                        server_config.FEATURES_OFF_BY_DEFAULT | {"documents"})
    assert {pair for pair in checklist.tool_switch_items()
            if pair.endswith("/documents")} == {
        "tool-switch:memory_add_document/documents",
        "tool-switch:memory_delete_document/documents",
        "tool-switch:memory_get_document/documents",
        "tool-switch:memory_list_documents/documents"}


def test_every_kind_of_environment_read_is_found(tmp_path: pathlib.Path) -> None:
    """A read by .get(), by getenv() or by subscript counts, with the name written out
    or held in a constant, including one imported from another module. A variable that
    is only mentioned counts for nothing."""
    (tmp_path / "keys.py").write_text('KEY = "MEMVARA_FROM_IMPORT"\n', encoding="utf-8")
    source = tmp_path / "config.py"
    source.write_text(textwrap.dedent('''
        import os
        from os import getenv
        from .keys import KEY

        LOCAL = "MEMVARA_FROM_CONSTANT"


        def read(env):
            env.get("MEMVARA_BY_GET")
            getenv("MEMVARA_BY_GETENV")
            os.environ["MEMVARA_BY_SUBSCRIPT"]
            env.get(KEY)
            env.get(LOCAL)
            env.get("OTHER_VARIABLE")
            print("MEMVARA_ONLY_MENTIONED")
        '''), encoding="utf-8")
    assert checklist.env_items(source) == [
        "env:MEMVARA_BY_GET", "env:MEMVARA_BY_GETENV", "env:MEMVARA_BY_SUBSCRIPT",
        "env:MEMVARA_FROM_CONSTANT", "env:MEMVARA_FROM_IMPORT"]


def test_config_reads_the_store_key_through_an_imported_name() -> None:
    """config.py reads MEMVARA_DB_KEY as env.get(KEY_ENV), a name it imports from
    memvara/store/encryption.py."""
    found = checklist.env_items()
    assert {"env:MEMVARA_DB", "env:MEMVARA_DB_KEY", "env:MEMVARA_READ_ONLY"} <= set(found)
    assert not any(item.startswith("env:MEMVARA_FEATURE_") for item in found)


_TELEMETRY = '''"""Counters.

The failures, and the series that catches each:

============  ==========
failure       signal
============  ==========
first mode    ``a.b``,
              ``a.c``
second mode   ``d.e``
that wraps
============  ==========

**A third arrived with the new
seam**, and it is described here.
"""
'''


def test_a_host_has_an_item_only_for_the_hooks_it_fires() -> None:
    hooks = set(checklist.hook_items())
    assert {"hook:claude/recall", "hook:codex/capture", "hook:cursor/approve"} <= hooks
    assert "hook:cursor/recall" not in hooks  # Cursor has no prompt event


def test_every_open_known_bug_is_an_item() -> None:
    assert "bug:B2" in checklist.bug_items()


def test_silent_failure_modes_come_from_the_table_and_from_an_announcement(
        tmp_path: pathlib.Path) -> None:
    path = tmp_path / "telemetry.py"
    path.write_text(_TELEMETRY, encoding="utf-8")
    assert checklist.silent_items(path) == [
        "silent:first-mode", "silent:second-mode-that-wraps", "silent:new-seam"]


def test_the_real_telemetry_docstring_gives_the_wrapped_row_and_the_seventh() -> None:
    found = checklist.silent_items()
    assert "silent:poisoning-a-retraction-that-retires-nothing" in found
    assert "silent:redaction-seam" in found


def test_a_telemetry_docstring_without_its_table_is_an_error(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "telemetry.py"
    path.write_text('"""Counters, and no table."""\n', encoding="utf-8")
    with pytest.raises(checklist.ChecklistError, match="silent failure modes"):
        checklist.silent_items(path)


def test_an_indented_telemetry_table_is_read_the_same(tmp_path: pathlib.Path) -> None:
    """A table set in by a few spaces, as a quoted block would be, lists the same modes.
    Reading its columns from the wrong place would lose every row in silence."""
    lines = _TELEMETRY.splitlines(keepends=True)
    borders = [i for i, line in enumerate(lines) if line.startswith("====")]
    first, last = borders[0], borders[-1]
    indented = lines[:first] + [f"    {line}" for line in lines[first:last + 1]]
    path = tmp_path / "telemetry.py"
    path.write_text("".join(indented + lines[last + 1:]), encoding="utf-8")
    assert checklist.silent_items(path) == [
        "silent:first-mode", "silent:second-mode-that-wraps", "silent:new-seam"]


def test_a_telemetry_table_with_no_rows_is_an_error(tmp_path: pathlib.Path) -> None:
    """The announced mode must not hide a table that lost its rows."""
    path = tmp_path / "telemetry.py"
    path.write_text('"""Counters.\n\n=======  ======\nfailure  signal\n=======  ======\n'
                    '=======  ======\n\n**A third arrived with the new seam**.\n"""\n',
                    encoding="utf-8")
    with pytest.raises(checklist.ChecklistError, match="no rows"):
        checklist.silent_items(path)


_EXAMPLE_TABLE = """\
A row is written like this:

====  =====
key   value
====  =====
the   rest
====  =====

"""


def test_an_earlier_table_does_not_stand_in_for_the_failure_table(
        tmp_path: pathlib.Path) -> None:
    """Only the table whose header names the failure and signal columns lists the modes.
    An example table earlier in the docstring must not be read in its place."""
    path = tmp_path / "telemetry.py"
    path.write_text(_TELEMETRY.replace("The failures, and", _EXAMPLE_TABLE + "The failures, and"),
                    encoding="utf-8")
    assert checklist.silent_items(path) == [
        "silent:first-mode", "silent:second-mode-that-wraps", "silent:new-seam"]


def test_a_table_without_the_failure_and_signal_columns_is_an_error(
        tmp_path: pathlib.Path) -> None:
    path = tmp_path / "telemetry.py"
    path.write_text(_TELEMETRY.replace("failure       signal", "mode          series"),
                    encoding="utf-8")
    with pytest.raises(checklist.ChecklistError, match="failure and signal"):
        checklist.silent_items(path)


_INTERNALS = """\
# Internals

## Design invariants (do not violate)

Each one is stated as **Claim**, and this line is not an invariant.

1. **The first rule.** It holds.

2. **The second rule
   spans two lines.**

   > **Claim.** An indented line belongs to the entry above it.

---

## Something else

3. **Not an invariant, because it is in another section.**
"""

_PAGE = """\
# A page

## Invariants and assumptions

- **A page rule.** More text.
- **A restated rule.** This is invariant 2 in INTERNALS.md.

## Read next
"""

_IDS = {
    "docs/INTERNALS.md": {"I1": "The first rule.", "I2": "The second rule spans two lines."},
    "docs/claude/page.md": {"PG1": "A page rule.", "I2": "A restated rule."},
}


def _documents(tmp_path: pathlib.Path, ids: dict[str, dict[str, str]],
               page: str = _PAGE) -> pathlib.Path:
    """A repository with an INTERNALS, one docs/claude page, and an ids file."""
    (tmp_path / "docs" / "claude").mkdir(parents=True)
    (tmp_path / "docs" / "INTERNALS.md").write_text(_INTERNALS, encoding="utf-8")
    (tmp_path / "docs" / "claude" / "page.md").write_text(page, encoding="utf-8")
    path = tmp_path / "ids.json"
    path.write_text(json.dumps({"documents": ids}), encoding="utf-8")
    return path


def _with(document: str, **changes: str | None) -> dict[str, dict[str, str]]:
    """_IDS with some ids of one document changed, or removed where the value is None."""
    ids = {name: dict(entries) for name, entries in _IDS.items()}
    for key, lead in changes.items():
        if lead is None:
            del ids[document][key]
        else:
            ids[document][key] = lead
    return ids


def _one_problem(problems: list[str], *parts: str) -> None:
    """There is exactly one problem, and it names each of `parts`."""
    assert len(problems) == 1, problems
    for part in parts:
        assert part in problems[0], (part, problems[0])


def test_each_invariant_gets_the_id_recorded_for_its_sentence(tmp_path: pathlib.Path) -> None:
    assert checklist.invariant_ids(tmp_path, _documents(tmp_path, _IDS)) == (
        {"I1", "I2", "PG1"}, [])


def test_a_bullet_without_an_id_is_named(tmp_path: pathlib.Path) -> None:
    ids, problems = checklist.invariant_ids(
        tmp_path, _documents(tmp_path, _with("docs/claude/page.md", PG1=None)))
    assert ids == {"I1", "I2"}
    _one_problem(problems, "docs/claude/page.md", "'A page rule.'", "no id")


def test_an_id_whose_sentence_is_gone_is_named(tmp_path: pathlib.Path) -> None:
    _, problems = checklist.invariant_ids(tmp_path, _documents(
        tmp_path, _with("docs/claude/page.md", PG2="A rule that was removed.")))
    _one_problem(problems, "PG2", "docs/claude/page.md", "'A rule that was removed.'",
                 "no longer states")


def test_a_reworded_internals_invariant_keeps_its_id_and_is_named(
        tmp_path: pathlib.Path) -> None:
    ids, problems = checklist.invariant_ids(tmp_path, _documents(
        tmp_path, _with("docs/INTERNALS.md", I1="The old first rule.")))
    assert "I1" in ids
    _one_problem(problems, "docs/INTERNALS.md", "invariant 1", "'The first rule.'",
                 "'The old first rule.'")


def test_a_bullet_that_restates_an_invariant_must_carry_its_id(
        tmp_path: pathlib.Path) -> None:
    _, problems = checklist.invariant_ids(tmp_path, _documents(
        tmp_path, _with("docs/claude/page.md", I2=None, PG2="A restated rule.")))
    _one_problem(problems, "'A restated rule.'", "restates I2", "PG2")


def test_a_bullet_that_opens_with_no_bold_sentence_is_named(tmp_path: pathlib.Path) -> None:
    page = _PAGE.replace("- **A page rule.** More text.", "- A page rule, not in bold.")
    _, problems = checklist.invariant_ids(
        tmp_path, _documents(tmp_path, _with("docs/claude/page.md", PG1=None), page))
    _one_problem(problems, "docs/claude/page.md", "'A page rule, not in bold.'",
                 "bold sentence")


def test_two_bullets_that_open_with_the_same_sentence_are_named(
        tmp_path: pathlib.Path) -> None:
    """An id is attached to its invariant's opening sentence, so two bullets on one page
    that open alike cannot each have their own id. That is the problem to report, not
    that the second id names a sentence the page no longer states."""
    page = _PAGE.replace("- **A restated rule.**",
                         "- **A page rule.** Said a second time.\n- **A restated rule.**")
    _, problems = checklist.invariant_ids(tmp_path, _documents(
        tmp_path, _with("docs/claude/page.md", PG2="A page rule."), page))
    _one_problem(problems, "docs/claude/page.md", "'A page rule.'", "same sentence")
    assert "no longer" not in problems[0]


def test_two_ids_that_record_one_sentence_are_named(tmp_path: pathlib.Path) -> None:
    """Two ids for one opening sentence cannot both name its invariant. The problem names
    both, instead of calling one of them stale."""
    _, problems = checklist.invariant_ids(tmp_path, _documents(
        tmp_path, _with("docs/claude/page.md", PG2="A page rule.")))
    _one_problem(problems, "docs/claude/page.md", "'A page rule.'", "PG1", "PG2")
    assert "no longer" not in problems[0]


def test_internals_without_its_invariants_heading_is_an_error(tmp_path: pathlib.Path) -> None:
    path = _documents(tmp_path, _IDS)
    (tmp_path / "docs" / "INTERNALS.md").write_text("# Internals\n", encoding="utf-8")
    with pytest.raises(checklist.ChecklistError, match="Design invariants"):
        checklist.invariant_ids(tmp_path, path)


def test_every_documented_invariant_has_a_stable_id() -> None:
    ids, problems = checklist.invariant_ids()
    assert not problems, "\n".join(problems)
    assert {"I1", "I8", "MM1", "MS1"} <= ids


def test_the_checklist_holds_an_item_of_every_kind() -> None:
    found = checklist.items()
    for item in ("tool:memory_recall", "switch:read_only",
                 "tool-switch:memory_add_document/documents", "env:MEMVARA_DB",
                 "hook:claude/recall", "inv:I8", "silent:predicate-explosion", "bug:B2"):
        assert item in found, item


def test_a_source_that_yields_nothing_stops_the_checklist(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A source that yields no items has changed shape. Reading nothing from it would
    drop its items from the checklist, and nothing would report it."""
    monkeypatch.setattr(checklist, "env_items", lambda: [])
    with pytest.raises(checklist.ChecklistError, match="environment variables"):
        checklist.items()


def test_no_open_known_bug_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every bug being fixed is the goal, not a source that changed shape."""
    monkeypatch.setattr(checklist, "bug_items", lambda: [])
    assert not any(item.startswith("bug:") for item in checklist.items())


def _tests(tmp_path: pathlib.Path, files: dict[str, str]) -> checklist.Scan:
    """Scan a folder of test files written for one test."""
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return checklist.scan(tmp_path)


def _places(problems: set[str]) -> set[tuple[str, int]]:
    """The file name and line number each problem starts with."""
    places = set()
    for problem in problems:
        where = problem.split(": ", 1)[0]
        path, line = where.rsplit(":", 1)
        places.add((pathlib.PurePath(path).name, int(line)))
    return places


def test_a_covers_mark_on_a_test_covers_its_items(tmp_path: pathlib.Path) -> None:
    found = _tests(tmp_path, {"test_one.py": '''
        import pytest

        @pytest.mark.covers("tool:memory_recall", "inv:I3")
        async def test_recall():
            pass
        '''})
    assert found.covered == {"tool:memory_recall", "inv:I3"}
    assert {item for _, item in found.declared} == {"tool:memory_recall", "inv:I3"}
    assert all(where.endswith("test_one.py:4") for where, _ in found.declared)
    assert not found.problems


def test_a_test_expected_to_fail_covers_nothing(tmp_path: pathlib.Path) -> None:
    found = _tests(tmp_path, {"test_one.py": '''
        import pytest

        @pytest.mark.covers("tool:memory_recall")
        @pytest.mark.xfail(strict=True, reason="a known bug")
        def test_recall():
            pass
        '''})
    assert found.covered == set()
    assert {item for _, item in found.declared} == {"tool:memory_recall"}


def test_a_known_bugs_marker_covers_its_bug_and_nothing_else(tmp_path: pathlib.Path) -> None:
    found = _tests(tmp_path, {"test_one.py": '''
        import pytest
        from harness import known_bugs

        @pytest.mark.covers("tool:memory_standing")
        @known_bugs.xfail("B5")
        def test_standing():
            pass
        '''})
    assert found.covered == {"bug:B5"}


def test_marks_on_a_class_and_in_pytestmark_reach_the_tests_they_apply_to(
        tmp_path: pathlib.Path) -> None:
    found = _tests(tmp_path, {"test_one.py": '''
        import pytest

        pytestmark = [pytest.mark.covers("env:MEMVARA_DB")]

        @pytest.mark.covers("switch:documents")
        class TestDocuments:
            pytestmark = pytest.mark.covers("tool:memory_add_document")

            def test_add(self):
                pass

        @pytest.mark.xfail(strict=True)
        class TestBroken:
            @pytest.mark.covers("tool:memory_recall")
            def test_recall(self):
                pass
        '''})
    assert found.covered == {"env:MEMVARA_DB", "switch:documents",
                             "tool:memory_add_document"}
    assert not found.problems


def test_a_test_skipped_without_a_condition_covers_nothing(tmp_path: pathlib.Path) -> None:
    found = _tests(tmp_path, {"test_one.py": '''
        import sys
        import pytest

        @pytest.mark.covers("tool:memory_recall")
        @pytest.mark.skip(reason="never runs")
        def test_skipped():
            pass

        @pytest.mark.covers("tool:memory_search")
        @pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
        def test_sometimes_skipped():
            pass
        '''})
    assert found.covered == {"tool:memory_search"}


def test_a_quarantined_test_covers_nothing(tmp_path: pathlib.Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tiers, "TESTS", tmp_path.resolve())
    found = _tests(tmp_path, {
        "adversarial/quarantine/test_flaky.py": '''
            import pytest
            from harness import known_bugs

            @pytest.mark.covers("tool:memory_recall")
            def test_flaky():
                pass

            @known_bugs.xfail("B4")
            def test_pinned():
                pass
            ''',
        "adversarial/nightly/test_slow.py": '''
            import pytest

            @pytest.mark.covers("tool:memory_search")
            def test_slow():
                pass
            '''})
    assert found.covered == {"tool:memory_search"}


def test_a_covers_mark_the_scan_cannot_read_is_a_problem(tmp_path: pathlib.Path) -> None:
    found = _tests(tmp_path, {"test_one.py": '''
        import pytest

        ITEMS = ("tool:memory_recall",)

        @pytest.mark.covers(*ITEMS)
        def test_from_a_variable():
            pass

        @pytest.mark.covers(item="tool:memory_recall")
        def test_by_keyword():
            pass

        @pytest.mark.covers()
        def test_empty():
            pass

        @pytest.mark.covers
        def test_bare():
            pass
        '''})
    assert found.covered == set()
    assert _places(found.problems) == {("test_one.py", 6), ("test_one.py", 10),
                                       ("test_one.py", 14), ("test_one.py", 18)}


def test_a_covers_mark_anywhere_but_on_a_test_is_a_problem(tmp_path: pathlib.Path) -> None:
    found = _tests(tmp_path, {
        "test_one.py": '''
            import pytest

            @pytest.mark.covers("tool:memory_recall")
            def helper():
                pass

            @pytest.fixture
            @pytest.mark.covers("tool:memory_search")
            def server():
                pass

            @pytest.mark.parametrize(
                "n", [pytest.param(1, marks=pytest.mark.covers("tool:memory_ask"))])
            def test_param(n):
                pass
            ''',
        "support.py": '''
            import pytest

            COVERS = pytest.mark.covers("tool:memory_since")
            '''})
    assert found.covered == set()
    assert _places(found.problems) == {("test_one.py", 4), ("test_one.py", 9),
                                       ("test_one.py", 14), ("support.py", 4)}


def test_a_problem_names_a_file_outside_the_checkout_by_its_resolved_path(
        tmp_path: pathlib.Path) -> None:
    """The checklist shows a path the way the tier report does: from the repository root,
    or resolved and in full when the file is outside the checkout. A second copy of that
    function showed the path exactly as it was given, so one file had two names."""
    (tmp_path / "sub").mkdir()
    found = _tests(tmp_path / "sub" / "..", {"test_one.py": '''
        import pytest

        @pytest.mark.covers()
        def test_empty():
            pass
        '''})
    [problem] = found.problems
    assert problem.startswith(f"{(tmp_path / 'test_one.py').resolve()}:4: "), problem


def test_a_covers_id_that_names_no_item_is_reported_with_its_place() -> None:
    found = checklist.Scan(declared={("tests/x.py:3", "tool:memory_recal"),
                                     ("tests/x.py:4", "bug:B2"),
                                     ("tests/x.py:5", "tool:memory_recall")})
    wrong = checklist.misdeclared(found, {"tool:memory_recall", "bug:B2"})
    assert len(wrong) == 2, wrong
    assert wrong[0].startswith("tests/x.py:3: ") and "tool:memory_recal" in wrong[0]
    assert "tool:memory_recall" not in wrong[0]
    assert wrong[1].startswith("tests/x.py:4: ") and "known_bugs.xfail('B2')" in wrong[1]
