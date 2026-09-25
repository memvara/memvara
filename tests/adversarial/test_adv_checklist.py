"""The coverage checklist and its ratchet (tests/harness/checklist.py)."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import textwrap

import pytest

from harness import checklist
from memvara.server import mcp as mcp_module
from memvara.server.tools import BY_NAME


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
    drop its items from the checklist without a word."""
    monkeypatch.setattr(checklist, "env_items", lambda: [])
    with pytest.raises(checklist.ChecklistError, match="environment variables"):
        checklist.items()


def test_no_open_known_bug_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every bug being fixed is the goal, not a source that changed shape."""
    monkeypatch.setattr(checklist, "bug_items", lambda: [])
    assert not any(item.startswith("bug:") for item in checklist.items())
