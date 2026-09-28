# The coverage checklist and its ratchet: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build workstream F5 of the adversarial suite: a checklist of everything the suite must test, read from the code, plus a fast-tier test that fails when a new gap appears and when a recorded gap closes.

**Architecture:** `tests/harness/checklist.py` reads eight sources and turns each into item ids of the form `kind:name`. It reads the `@pytest.mark.covers(...)` marks from the source of every file under `tests/`, by walking the syntax tree rather than importing. `tests/harness/checklist_baseline.txt` records today's gaps, and `tests/adversarial/test_adv_checklist.py` compares the gaps with the baseline in both directions, so the two stay equal. That the baseline only shrinks is kept by review, because a change can add a line to it. `tests/harness/invariant_ids.json` gives each documented invariant an id that survives rewording.

**Tech Stack:** Python 3.10 to 3.13, pytest, the standard library's `ast`, `json` and `re`, and memvara's own `MemvaraMCPServer`, run in the test process.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`: the F5 row of the Phase 0 table, "What the checklist enumerates", "How a test gets counted", done criterion 1, and the F5 proof under "Verification". The workstream brief adds the file list and the rule that existing tests get `covers` marks without changing what they do.

## Global Constraints

- Items are taken from the code, not from lists kept by hand. The one hand-kept file is `invariant_ids.json`, because an id that survives rewording cannot be derived from the text.
- The scan of `tests/` reads source through `ast`. It never imports a test module.
- A test marked as an expected failure covers nothing.
- The fast tier fails when a new gap appears, and when a baseline line is no longer a gap.
- The fast tier must stay within its budget: the new test file runs in a few seconds.
- Python 3.10 to 3.13, on Linux, macOS and Windows.
- Every module in `tests/harness/` imports with no side effects, because `--doctest-modules` imports each one while pytest collects.
- Work on branch `claude/adversarial-checklist`. Commit files by name. Never push. No AI attribution anywhere.
- Prose follows "How to write" and "Write plainly" in `CLAUDE.md`.
- Existing tests get `covers` marks only; what they do does not change.
- Files owned: `tests/harness/checklist.py`, `tests/harness/invariant_ids.json`, `tests/harness/checklist_baseline.txt`, `tests/adversarial/test_adv_checklist.py`, the marker registration in `tests/adversarial/conftest.py`, `covers` marks in existing files under `tests/adversarial/`, this plan, and a section of `docs/claude/testing.md`.

## Review Focus

1. **A `covers` mark written in a form the scan cannot read** (a variable, a keyword, a `pytest.param`, a helper module). A person expects the run to fail and name the file and line, not to count nothing in silence. Task 6 tests every form.
2. **A reworded invariant.** A person expects the failure to say which sentence lost its id and which recorded sentence went stale, so they can keep the id. Task 4 tests rewording in INTERNALS and on a page.
3. **A source that changes shape**, such as a renamed INTERNALS heading or a telemetry docstring without its table. A person expects a loud error, not a baseline that quietly loses lines. Tasks 3, 4 and 5 test it.
4. **A switch that reveals a tool rather than hiding one**, as a default-off feature that owns a tool would. A person expects its pair to be found. Task 1 tests it with a probe tool.
5. **A newly registered known bug.** A person expects no new gap when the bug has its strict expected failure, and a gap when it has none. Tasks 6 and 7 test both.

## Decisions this plan makes where the design is silent or needs reading

- **Switches are flipped from their default**, not turned off. Two features (`extraction_chunks`, `agentic_extraction`) and both modes are off by default, so turning them on is the only change there is. Today the flip finds 31 pairs.
- **The two modes take part in the pairs**: `read_only` hides nine write tools, and `anchored` rewrites the schemas of three read tools.
- **The pairs are found in the test process.** `python -m memvara.server` builds the same `MemvaraMCPServer` from the same three settings (`memvara/server/cli.py`), so the `tools/list` reply is the same, and 25 servers take 0.2 seconds instead of 25 process starts.
- **A known bug is covered by its strict expected failure.** "An xfail test covers nothing" and done criterion 1 ("every known bug has an issue and a strict xfail") would otherwise contradict each other, since an open bug's only test is expected to fail. A `covers` mark cannot name a bug.
- **Besides xfail, a test skipped without a condition and a quarantined test cover nothing**, for the same reason: neither shows that anything works.
- **The seventh silent failure mode is read from its own sentence.** The telemetry docstring tabulates six and announces the seventh in prose ("**A seventh arrived with the redaction seam**").
- **A bullet that restates an INTERNALS invariant shares its id.** The fast tier enforces this where the bullet says so ("This is invariant 3"). `memory-model.md`'s "An unknown predicate defaults to `Cardinality.MANY`" restates invariant 2 without saying so, and is mapped to `I2` by hand.

## File Structure

- `tests/harness/checklist.py` (create): the eight sources, the scan, the baseline reader. One responsibility: say what the checklist holds and what the tests declare.
- `tests/harness/invariant_ids.json` (create): the stable id of each documented invariant.
- `tests/harness/checklist_baseline.txt` (create): today's gaps.
- `tests/adversarial/test_adv_checklist.py` (create): the ratchet tests on this repository, and unit tests of each reader on small inputs.
- `tests/adversarial/conftest.py` (modify): register the `covers` marker.
- `tests/adversarial/test_adv_stdio.py`, `tests/adversarial/test_adv_hooks.py` (modify): add `covers` marks.
- `docs/claude/testing.md` (modify): a section before the final `Next:` line.

Every command below runs from the worktree root with the CI virtual environment, and with `TMPDIR` set to a directory of your own under `/private/tmp`:

```bash
PY=/Applications/workstation/agent-memory/.claude/worktrees/friendly-einstein-53c8da/local/venv-ci/bin/python
PYTHONPATH=$PWD TMPDIR=/private/tmp/adversarial-checklist $PY -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_checklist.py
```

---

### Task 1: Tools, switches and the tool-switch pairs

**Files:**
- Create: `tests/harness/checklist.py`
- Create: `tests/adversarial/test_adv_checklist.py`

**Interfaces:**
- Produces: `checklist.MODES: tuple[str, ...]`, `checklist.ChecklistError`, `checklist.tool_items() -> list[str]`, `checklist.switch_items() -> list[str]`, `checklist.listing(*, read_only=False, anchored=False, features_off=FEATURES_OFF_BY_DEFAULT) -> dict[str, Any]`, `checklist.tool_switch_items() -> list[str]`.

- [ ] **Step 1: Write the failing tests**

```python
"""The coverage checklist and its ratchet (tests/harness/checklist.py)."""

from __future__ import annotations

import dataclasses

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
```

- [ ] **Step 2: Run them and see them fail**

Run: `... -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_checklist.py`
Expected: collection error, `ImportError: cannot import name 'checklist' from 'harness'`.

- [ ] **Step 3: Write the module and the three readers**

```python
"""The coverage checklist: every item the adversarial suite must test, and the tests
that cover each one. (The full docstring is written in Task 5.)"""

from __future__ import annotations

import json
from typing import Any, Collection

from memvara.server import config as server_config
from memvara.server.mcp import MemvaraMCPServer
from memvara.server.tools import TOOLS

from . import stores

#: The server's two modes. Each is a switch like a feature, but not one of FEATURES.
MODES = ("read_only", "anchored")


class ChecklistError(Exception):
    """The checklist could not read one of its sources, so some items are unknown."""


def tool_items() -> list[str]:
    return [f"tool:{tool.name}" for tool in TOOLS]


def switch_items() -> list[str]:
    return [f"switch:{name}" for name in _switches()]


def _switches() -> tuple[str, ...]:
    # Read when called, so that a test can add a switch to FEATURES and see it here.
    return (*server_config.FEATURES, *MODES)


_TOOLS_LIST = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})


def listing(*, read_only: bool = False, anchored: bool = False,
            features_off: Collection[str] = server_config.FEATURES_OFF_BY_DEFAULT,
            ) -> dict[str, Any]:
    """The `tools/list` reply of a server with these settings, keyed by tool name.

    The server runs in this process, over an in-memory store. `python -m memvara.server`
    builds the same `MemvaraMCPServer` from the same three settings
    (memvara/server/cli.py), so this is the list the real process sends.
    """
    server = MemvaraMCPServer(stores.memory(), user="checklist", read_only=read_only,
                              anchored=anchored, features_off=features_off)
    try:
        line = server.handle_line(_TOOLS_LIST)
    finally:
        server.close()
    if line is None:
        raise ChecklistError("the server sent no reply to tools/list")
    return {spec["name"]: spec for spec in json.loads(line)["result"]["tools"]}


def tool_switch_items() -> list[str]:
    """One item for each tool whose `tools/list` entry changes when a switch is flipped:
    the tool disappears or appears, or its schema or description changes.

    The design says to turn each switch off. Two features and both modes are off by
    default, so turning them on is the only change there is, and every switch is flipped
    from its default instead.
    """
    default = listing()
    found = []
    for switch in _switches():
        if switch in MODES:
            flipped = listing(read_only=switch == "read_only",
                              anchored=switch == "anchored")
        else:
            flipped = listing(
                features_off=set(server_config.FEATURES_OFF_BY_DEFAULT) ^ {switch})
        found += [f"tool-switch:{tool}/{switch}" for tool in sorted({*default, *flipped})
                  if default.get(tool) != flipped.get(tool)]
    return found
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `... -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_checklist.py`
Expected: `3 passed`.

---

### Task 2: The environment variables config.py reads

**Files:**
- Modify: `tests/harness/checklist.py`
- Modify: `tests/adversarial/test_adv_checklist.py`

**Interfaces:**
- Produces: `checklist.CONFIG: pathlib.Path`, `checklist.env_items(path: pathlib.Path = CONFIG) -> list[str]`.

- [ ] **Step 1: Write the failing tests** (add `import pathlib` and `import textwrap` to the test file's imports)

```python
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
```

- [ ] **Step 2: Run them and see them fail**

Expected: `AttributeError: module 'harness.checklist' has no attribute 'env_items'`.

- [ ] **Step 3: Implement**

```python
import ast
import pathlib

from .env import REPO

CONFIG = REPO / "memvara" / "server" / "config.py"


def env_items(path: pathlib.Path = CONFIG) -> list[str]:
    """One item for each `MEMVARA_*` variable that the module at `path` reads.

    A read is `<mapping>.get(NAME)`, `getenv(NAME)` or `<mapping>[NAME]`. NAME is a string
    literal, or a module-level name bound to one, either in this module or in a module of
    this checkout that it imports the name from. That second case is how config.py reads
    MEMVARA_DB_KEY. The `MEMVARA_FEATURE_<NAME>` variables are read by prefix, and they
    are the `switch:` items already.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    keys = [key for node in ast.walk(tree) if (key := _read_key(node)) is not None]
    names = _strings(tree, path, {key.id for key in keys if isinstance(key, ast.Name)})
    found = set()
    for key in keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            value: str | None = key.value
        else:
            value = names.get(key.id) if isinstance(key, ast.Name) else None
        if value is not None and value.startswith("MEMVARA_"):
            found.add(f"env:{value}")
    return sorted(found)


def _read_key(node: ast.AST) -> ast.expr | None:
    """The key an environment read looks up, or None when `node` is not a read."""
    if isinstance(node, ast.Subscript):
        return node.slice
    if isinstance(node, ast.Call) and node.args and _callee(node) in ("get", "getenv"):
        return node.args[0]
    return None


def _callee(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _strings(tree: ast.Module, path: pathlib.Path, wanted: set[str]) -> dict[str, str]:
    """The string literals that module-level names are bound to, including the names in
    `wanted` that the module imports from another module of this checkout."""
    found = _bound_strings(tree)
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        for alias in node.names:
            name = alias.asname or alias.name
            source = _module_file(path, node.level, node.module) if name in wanted else None
            if source is not None:
                bound = _bound_strings(ast.parse(source.read_text(encoding="utf-8")))
                if alias.name in bound:
                    found[name] = bound[alias.name]
    return found


def _bound_strings(tree: ast.Module) -> dict[str, str]:
    """Module-level names bound to a string literal, as in `KEY_ENV = "MEMVARA_DB_KEY"`."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            found.update({target.id: node.value.value for target in targets
                          if isinstance(target, ast.Name)})
    return found


def _module_file(path: pathlib.Path, level: int, module: str) -> pathlib.Path | None:
    """The source of the module that `from <level dots><module> import ...`, written in
    `path`, names, when that module is part of this checkout."""
    base = path.parents[level - 1] if level else REPO
    target = base.joinpath(*module.split("."))
    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None
```

- [ ] **Step 4: Run the tests and see them pass**

Expected: `5 passed`.

---

### Task 3: Hooks, known bugs and silent failure modes

**Files:**
- Modify: `tests/harness/checklist.py`
- Modify: `tests/adversarial/test_adv_checklist.py`

**Interfaces:**
- Produces: `checklist.hook_items() -> list[str]`, `checklist.bug_items() -> list[str]`, `checklist.TELEMETRY: pathlib.Path`, `checklist.silent_items(path: pathlib.Path = TELEMETRY) -> list[str]`, and the private helper `_shown(path) -> str` that later tasks use in messages.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run them and see them fail**

Expected: `AttributeError: module 'harness.checklist' has no attribute 'hook_items'` (and the same for the others).

- [ ] **Step 3: Implement**

```python
import re

from .hooks import HOOKS_DIR, host_record
from .known_bugs import KNOWN_BUGS

TELEMETRY = REPO / "memvara" / "telemetry.py"


def hook_items() -> list[str]:
    """One item for each hook that each host fires. A host record lists only the hooks
    its client has an event for, so Cursor, which has no prompt event, has no recall
    item."""
    found = []
    for path in sorted((HOOKS_DIR / "hosts").glob("*.py")):
        if path.stem != "__init__":
            record = host_record(path.stem)
            found += [f"hook:{record.id}/{hook}" for hook in record.events]
    return found


def bug_items() -> list[str]:
    return [f"bug:{bug_id}" for bug_id in KNOWN_BUGS]


_BORDER = re.compile(r"=+(?: +=+)+")
_ANNOUNCED = re.compile(r"\*\*An? \w+ arrived with the (.+?)\*\*", re.S)


def silent_items(path: pathlib.Path = TELEMETRY) -> list[str]:
    """One item for each silent failure mode that the docstring of `path` lists.

    Most are rows of a table whose first column names the failure. A line with text in
    both columns starts a row, and a line with text in one column continues the row above
    it, which is how "poisoning / a retraction that retires nothing" spans two lines. A
    later mode can be announced in a sentence of its own instead, as the seventh is:
    "**A seventh arrived with the redaction seam**". Those sentences are read too.
    """
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    lines = doc.splitlines()
    borders = [i for i, line in enumerate(lines) if _BORDER.fullmatch(line.strip())]
    if len(borders) < 3:
        raise ChecklistError(
            f"the table of silent failure modes in {_shown(path)} was not found")
    border = lines[borders[0]]
    column = border.index("=", border.index(" "))
    rows: list[str] = []
    for line in lines[borders[1] + 1:borders[2]]:
        first, second = line[:column].strip(), line[column:].strip()
        if first and second:
            rows.append(first)
        elif first and rows:
            rows[-1] += f" {first}"
    names = rows + [" ".join(name.split()) for name in _ANNOUNCED.findall(doc)]
    return [f"silent:{_slug(name)}" for name in names]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _shown(path: pathlib.Path) -> str:
    """`path` from the repository root, or in full when it is outside the checkout."""
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)
```

- [ ] **Step 4: Run the tests and see them pass**

Expected: `10 passed`.

---

### Task 4: Invariants and their stable ids

**Files:**
- Modify: `tests/harness/checklist.py`
- Create: `tests/harness/invariant_ids.json`
- Modify: `tests/adversarial/test_adv_checklist.py`

**Interfaces:**
- Produces: `checklist.INVARIANT_IDS: pathlib.Path`, `checklist.Invariant` (frozen dataclass: `document: str`, `lead: str`, `text: str`, `number: int | None = None`), `checklist.documented_invariants(repo: pathlib.Path = REPO) -> list[Invariant]`, `checklist.invariant_ids(repo: pathlib.Path = REPO, ids_path: pathlib.Path = INVARIANT_IDS) -> tuple[set[str], list[str]]` (the ids of the stated invariants, and the problems).

- [ ] **Step 1: Write the failing tests** (add `import json` to the test file's imports)

```python
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
    """A repository with an INTERNALS, one docs/claude page and an ids file."""
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


def test_each_invariant_gets_the_id_recorded_for_its_sentence(tmp_path: pathlib.Path) -> None:
    assert checklist.invariant_ids(tmp_path, _documents(tmp_path, _IDS)) == (
        {"I1", "I2", "PG1"}, [])


def test_a_bullet_without_an_id_is_named(tmp_path: pathlib.Path) -> None:
    ids, problems = checklist.invariant_ids(
        tmp_path, _documents(tmp_path, _with("docs/claude/page.md", PG1=None)))
    assert ids == {"I1", "I2"}
    assert problems == ["docs/claude/page.md: 'A page rule.' has no id in ids.json. Add "
                        "one, or, if the bullet was reworded, update the sentence of its "
                        "old id."]


def test_an_id_whose_sentence_is_gone_is_named(tmp_path: pathlib.Path) -> None:
    _, problems = checklist.invariant_ids(tmp_path, _documents(
        tmp_path, _with("docs/claude/page.md", PG2="A rule that was removed.")))
    assert problems == ["ids.json gives PG2 to docs/claude/page.md: 'A rule that was "
                        "removed.', which that document no longer states. Update the "
                        "sentence, or remove the id."]


def test_a_reworded_internals_invariant_keeps_its_id_and_is_named(
        tmp_path: pathlib.Path) -> None:
    ids, problems = checklist.invariant_ids(tmp_path, _documents(
        tmp_path, _with("docs/INTERNALS.md", I1="The old first rule.")))
    assert "I1" in ids
    assert problems == ["docs/INTERNALS.md: 'The first rule.' is invariant 1, and "
                        "ids.json records I1 as 'The old first rule.'. If it was "
                        "reworded, record the new sentence."]


def test_a_bullet_that_restates_an_invariant_must_carry_its_id(
        tmp_path: pathlib.Path) -> None:
    _, problems = checklist.invariant_ids(tmp_path, _documents(
        tmp_path, _with("docs/claude/page.md", I2=None, PG2="A restated rule.")))
    assert problems == ["docs/claude/page.md: 'A restated rule.' restates I2, so its id "
                        "must be I2, not PG2."]


def test_a_bullet_that_opens_with_no_bold_sentence_is_named(tmp_path: pathlib.Path) -> None:
    page = _PAGE.replace("- **A page rule.** More text.", "- A page rule, not in bold.")
    _, problems = checklist.invariant_ids(
        tmp_path, _documents(tmp_path, _with("docs/claude/page.md", PG1=None), page))
    assert problems == ["docs/claude/page.md: 'A page rule, not in bold.' does not open "
                        "with a bold sentence, and its id is attached to that sentence."]


def test_internals_without_its_invariants_heading_is_an_error(tmp_path: pathlib.Path) -> None:
    path = _documents(tmp_path, _IDS)
    (tmp_path / "docs" / "INTERNALS.md").write_text("# Internals\n", encoding="utf-8")
    with pytest.raises(checklist.ChecklistError, match="Design invariants"):
        checklist.invariant_ids(tmp_path, path)


def test_every_documented_invariant_has_a_stable_id() -> None:
    ids, problems = checklist.invariant_ids()
    assert not problems, "\n".join(problems)
    assert {"I1", "I8", "MM1", "MS1"} <= ids
```

- [ ] **Step 2: Run them and see them fail**

Expected: `AttributeError: module 'harness.checklist' has no attribute 'invariant_ids'`.

- [ ] **Step 3: Implement the parser and the checks**

```python
from dataclasses import dataclass

_HERE = pathlib.Path(__file__).resolve().parent
INVARIANT_IDS = _HERE / "invariant_ids.json"

_INTERNALS = "docs/INTERNALS.md"
_INTERNALS_HEADING = "## Design invariants (do not violate)"
_PAGE_HEADING = "## Invariants and assumptions"
_NUMBERED = re.compile(r"(\d+)\. ")
_BULLET = re.compile(r"- ")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CITED = re.compile(r"\binvariant (\d+)\b")


@dataclass(frozen=True)
class Invariant:
    """One invariant, as a document states it."""

    #: The document's path from the repository root, such as docs/INTERNALS.md.
    document: str
    #: The bold sentence the invariant opens with, which is what its id is attached to.
    #: Empty when the invariant opens with no bold sentence.
    lead: str
    #: The whole entry, on one line.
    text: str
    #: The number INTERNALS gives it, or None for a bullet on a docs/claude page.
    number: int | None = None


def documented_invariants(repo: pathlib.Path = REPO) -> list[Invariant]:
    """Each numbered invariant in INTERNALS, and each bullet under "Invariants and
    assumptions" on a docs/claude page."""
    internals = _section(repo / _INTERNALS, _INTERNALS_HEADING)
    if internals is None:
        raise ChecklistError(f"{_INTERNALS} has no section headed {_INTERNALS_HEADING!r}")
    found = [Invariant(_INTERNALS, _lead(text), text, int(match.group(1)))
             for match, text in _entries(internals, _NUMBERED)]
    for page in sorted((repo / "docs" / "claude").glob("*.md")):
        document = page.relative_to(repo).as_posix()
        found += [Invariant(document, _lead(text), text)
                  for _, text in _entries(_section(page, _PAGE_HEADING) or [], _BULLET)]
    return found


def _section(path: pathlib.Path, heading: str) -> list[str] | None:
    """The lines of `path` under `heading`, up to the next heading of the same level."""
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
    if heading not in lines:
        return None
    start = lines.index(heading) + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")),
               len(lines))
    return lines[start:end]


def _entries(lines: list[str],
             marker: re.Pattern[str]) -> list[tuple[re.Match[str], str]]:
    """The list entries in `lines`, each as its marker and its text on one line.

    An entry starts at a line that begins with `marker`, and runs on through the indented
    and blank lines after it. The first line that is neither ends it.
    """
    entries: list[tuple[re.Match[str], list[str]]] = []
    current: list[str] | None = None
    for line in lines:
        match = marker.match(line)
        if match:
            current = [line[match.end():]]
            entries.append((match, current))
        elif current is not None and (not line or line[0].isspace()):
            current.append(line)
        else:
            current = None
    return [(match, " ".join(" ".join(body).split())) for match, body in entries]


def _lead(text: str) -> str:
    match = _BOLD.match(text)
    return match.group(1) if match else ""


def invariant_ids(repo: pathlib.Path = REPO, ids_path: pathlib.Path = INVARIANT_IDS,
                  ) -> tuple[set[str], list[str]]:
    """The ids of the invariants the documents state, and every problem with those ids.

    `ids_path` maps each document to the ids of its invariants, and each id to the bold
    sentence its invariant opens with. The sentence is how an id finds its invariant.
    INTERNALS's numbered invariants take their number as their id, I1 to I8. A bullet
    that restates one of them ("This is invariant 3") must carry that invariant's id, so
    a test of the invariant covers both.
    """
    recorded: dict[str, dict[str, str]] = json.loads(
        ids_path.read_text(encoding="utf-8"))["documents"]
    name = ids_path.name
    ids: set[str] = set()
    matched: set[tuple[str, str]] = set()
    problems: list[str] = []
    for invariant in documented_invariants(repo):
        entries = recorded.get(invariant.document, {})
        where = f"{invariant.document}: {invariant.lead or invariant.text[:80]!r}"
        if not invariant.lead:
            problems.append(f"{where} does not open with a bold sentence, and its id is "
                            "attached to that sentence.")
            continue
        if invariant.number is not None:
            key = f"I{invariant.number}"
            lead = entries.get(key)
            if lead is None:
                problems.append(f"{where} is invariant {invariant.number}, and {name} has "
                                f"no {key}. Add it.")
            elif lead != invariant.lead:
                problems.append(f"{where} is invariant {invariant.number}, and {name} "
                                f"records {key} as {lead!r}. If it was reworded, record "
                                "the new sentence.")
        else:
            found = [k for k, lead in entries.items() if lead == invariant.lead]
            if not found:
                problems.append(f"{where} has no id in {name}. Add one, or, if the "
                                "bullet was reworded, update the sentence of its old id.")
                continue
            key = found[0]
            cited = sorted({f"I{number}" for number in _CITED.findall(invariant.text)})
            if cited and key not in cited:
                problems.append(f"{where} restates {' or '.join(cited)}, so its id must "
                                f"be {cited[0]}, not {key}.")
        ids.add(key)
        matched.add((invariant.document, key))
    for document, entries in recorded.items():
        for key, lead in entries.items():
            if (document, key) not in matched:
                problems.append(f"{name} gives {key} to {document}: {lead!r}, which that "
                                "document no longer states. Update the sentence, or "
                                "remove the id.")
    return ids, problems
```

- [ ] **Step 4: Write `tests/harness/invariant_ids.json`**

Each document maps an id to its invariant's opening sentence, copied from the document with runs of whitespace collapsed. INTERNALS takes `I1` to `I8`. Each page has its own prefix: `CG` consolidation-and-graph, `MS` mcp-server, `MM` memory-model, `RP` release-and-plugins, `RC` remote-and-cloud, `RT` retrieval, `TB` telemetry-and-benchmarks, `WP` write-pipeline. The bullets that cite invariants 1, 3, 7 and 8 carry `I1`, `I3`, `I7` and `I8`, and memory-model's restatement of invariant 2 carries `I2`.

```json
{
  "about": "Stable ids for the invariants on the coverage checklist. Each document maps an id to the bold sentence its invariant opens with, which is how the id finds the invariant. INTERNALS's numbered invariants are I1 to I8, and a bullet that restates one carries its id. When you reword an invariant, change its sentence here and keep its id. docs/claude/testing.md explains the checklist.",
  "documents": {
    "docs/INTERNALS.md": {
      "I1": "Contradiction resolution is decided by the deterministic reconciler, and a model is reached only through named stages.",
      "I2": "Unknown predicates default to `Cardinality.MANY`.",
      "I3": "The engine hard-deletes only claims that carry an explicit `expires_at`, only after it passes, and always with a proof record; ending and superseding never delete. And end-of-life moves exactly one clock.",
      "I4": "Every claim carries provenance.",
      "I5": "The library must run with no API key and no network.",
      "I6": "A multi-hop answer is evaluated at one clock pair.",
      "I7": "A filter and a limit may not live in different layers.",
      "I8": "No MCP client can backdate the transaction clock."
    },
    "docs/claude/consolidation-and-graph.md": {
      "CG1": "Consolidation is deterministic and calls no model.",
      "CG2": "`Sweep` reads its snapshot once and writes back in bounded transactions.",
      "CG3": "`now` is read once for the whole pass.",
      "CG4": "The graph leg is closed when the store provably has no joins.",
      "CG5": "An empty connectivity result keeps the leg on.",
      "CG6": "An unregistered predicate accumulates rather than superseding, and decays slowly."
    },
    "docs/claude/mcp-server.md": {
      "MS1": "The scope is bound at startup and no tool call can change it.",
      "I8": "No MCP client can backdate the transaction clock.",
      "MS2": "Cloud mode refuses `MEMVARA_LLM` and `MEMVARA_EMBEDDER`.",
      "MS3": "A configuration error is a refusal that names the fix.",
      "MS4": "The documented example and the written entry may not drift.",
      "MS5": "Tool text and the packaged skill do not repeat each other."
    },
    "docs/claude/memory-model.md": {
      "I1": "Deterministic paths never call a model.",
      "I2": "An unknown predicate defaults to `Cardinality.MANY`.",
      "MM1": "`resolve_states()` is the single place either spelling of a state filter is interpreted.",
      "MM2": "The three states do not tile the store.",
      "I3": "The engine deletes a row only when a claim's explicit expiry has passed, and no write closes both clocks.",
      "I7": "A filter and a limit may not live in different layers.",
      "MM3": "A sort that sits above a limit ends in a content key, never an id.",
      "MM4": "Scope is bound at startup and cannot be widened by a call.",
      "MM5": "Each text index row sits at the rowid of the row it indexes.",
      "MM6": "Every commit `SQLiteStore` makes that can change a turn goes through `_maybe_commit`.",
      "MM7": "A claim's state changes with the clock only at one of its five time columns.",
      "MM8": "A change to the entity fold bumps `SCHEMA_VERSION`."
    },
    "docs/claude/release-and-plugins.md": {
      "RP1": "A published version is final.",
      "RP2": "The build directory is rebuilt from scratch on every run.",
      "RP3": "The vendored skill is not edited downstream.",
      "RP4": "`plugin/hooks/` has no sanctioned transform at all.",
      "RP5": "The hooks manifest under `plugin/hooks/` is generated, not vendored.",
      "RP6": "`plugin-claude.md` opens with its exact first line and carries exactly one `@@LOCAL@@` marker.",
      "RP7": "The seven names in `plugin-repos.txt` are pinned by a test."
    },
    "docs/claude/remote-and-cloud.md": {
      "RC1": "The credential decides the tenant.",
      "RC2": "Narrowing is one-way.",
      "RC3": "Redaction runs client-side.",
      "RC4": "An unmapped operation raises rather than approximating.",
      "RC5": "This repository does not implement the server."
    },
    "docs/claude/retrieval.md": {
      "RT1": "`recall()` takes `valid_at` and no other time keyword.",
      "RT2": "Eight reads take the time keywords",
      "RT3": "The recall header names the text as data.",
      "RT4": "A leg abstains rather than contributing noise.",
      "RT5": "The graph leg seeds on content, never on ids.",
      "I7": "A filter and a limit may not live in different layers",
      "RT6": "A document's passages are episodes.",
      "RT7": "A store can only be opened by the embedder that wrote it.",
      "RT8": "A cosine threshold belongs to an embedding space.",
      "RT9": "A leg on another thread sees exactly what the calling thread would."
    },
    "docs/claude/telemetry-and-benchmarks.md": {
      "TB1": "\"Verify\" means comparing an output, never that a command exited 0.",
      "TB2": "A number goes into the results document with its caveat attached.",
      "TB3": "The offline run repeats exactly.",
      "TB4": "The benchmark harnesses do not exercise ingestion.",
      "TB5": "Telemetry adds no required dependency and no background thread.",
      "TB6": "Every read-path model call is counted, whichever stage made it.",
      "TB7": "Benchmark reads are plain.",
      "TB8": "A graph harness declares its corpus's relations, or it measures a store with no edges."
    },
    "docs/claude/write-pipeline.md": {
      "WP1": "A document chunk passes the role check.",
      "WP2": "The gate biases toward recall.",
      "WP3": "A closed vocabulary refuses what nothing declared.",
      "WP4": "Guidance adds to the extraction prompt and never replaces it.",
      "WP5": "The fast path chooses precision over recall.",
      "WP6": "`reject_ungrounded` defaults to `\"auto\"`.",
      "WP7": "A truncated model response fails the write.",
      "WP8": "`WriteReceipt` must be fully populated, including `llm_calls`.",
      "WP9": "Extraction that cannot run is reported, and how depends on the deployment.",
      "WP10": "A model proposes and the reconciler applies.",
      "WP11": "A long turn is extracted whole or not at all."
    }
  }
}
```

That is 8 INTERNALS ids and 59 page ids: 67 `inv:` items from 73 stated invariants.

- [ ] **Step 5: Run the tests and see them pass**

Expected: `18 passed`.

---

### Task 5: The whole checklist, the empty-source guard, and the first documentation

**Files:**
- Modify: `tests/harness/checklist.py` (the module docstring and `items()`)
- Modify: `tests/adversarial/test_adv_checklist.py`
- Modify: `docs/claude/testing.md`

**Interfaces:**
- Produces: `checklist.items() -> set[str]`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run them and see them fail**

Expected: `AttributeError: module 'harness.checklist' has no attribute 'items'`.

- [ ] **Step 3: Implement**

```python
def items() -> set[str]:
    """Every item on the checklist.

    Every source except the known bugs must yield at least one item. One that yields
    none has changed shape, a renamed heading for example, and reading nothing from it
    would drop its items from the checklist without a word. The known bugs may run out,
    because that is the goal.
    """
    sources = {
        "tools": tool_items(),
        "switches": switch_items(),
        "tool-switch pairs": tool_switch_items(),
        "environment variables": env_items(),
        "hooks": hook_items(),
        "invariants": [f"inv:{key}" for key in invariant_ids()[0]],
        "silent failure modes": silent_items(),
    }
    empty = [kind for kind, found in sources.items() if not found]
    if empty:
        raise ChecklistError(
            f"no {' or '.join(empty)} were found, so tests/harness/checklist.py can no "
            "longer read the code they come from")
    return {item for found in sources.values() for item in found} | set(bug_items())
```

Write the module docstring (what the checklist is, the eight kinds, and how coverage is counted), and add the section "The coverage checklist" to `docs/claude/testing.md` just before its `Next:` line, covering what is on the checklist and the invariant ids.

- [ ] **Step 4: Run the file and see it pass**

Expected: `21 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/harness/checklist.py tests/harness/invariant_ids.json tests/adversarial/test_adv_checklist.py docs/claude/testing.md
git commit -m "Add a checklist of everything the adversarial suite must test, read from the code"
```

---

### Task 6: Reading the covers marks

**Files:**
- Modify: `tests/harness/checklist.py`
- Modify: `tests/adversarial/test_adv_checklist.py`

**Interfaces:**
- Consumes: `harness.tiers.TESTS`, `harness.tiers.tier_of`.
- Produces: `checklist.Scan` (dataclass: `covered: set[str]`, `declared: set[tuple[str, str]]` of `(where, item)`, `problems: set[str]`), `checklist.scan(root: pathlib.Path = tiers.TESTS) -> Scan`, `checklist.misdeclared(scan: Scan, items: Collection[str]) -> list[str]`.

- [ ] **Step 1: Write the failing tests** (import `tiers` beside `checklist`: `from harness import checklist, tiers`)

```python
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


def test_a_covers_id_that_names_no_item_is_reported_with_its_place() -> None:
    found = checklist.Scan(declared={("tests/x.py:3", "tool:memory_recal"),
                                     ("tests/x.py:4", "bug:B2"),
                                     ("tests/x.py:5", "tool:memory_recall")})
    assert checklist.misdeclared(found, {"tool:memory_recall", "bug:B2"}) == [
        "tests/x.py:3: covers names tool:memory_recal, which is not on the checklist.",
        "tests/x.py:4: covers names bug:B2. A known bug is covered by the test that "
        "carries its strict expected failure, known_bugs.xfail('B2'), not by covers."]
```

- [ ] **Step 2: Run them and see them fail**

Expected: `AttributeError: module 'harness.checklist' has no attribute 'scan'`.

- [ ] **Step 3: Implement**

```python
from dataclasses import field

from . import stores, tiers


@dataclass
class Scan:
    """What the tests under one folder declare, read from their source."""

    #: The items covered by tests that are expected to pass, and `bug:<id>` for each
    #: known bug that a test pins with its strict expected failure.
    covered: set[str] = field(default_factory=set)
    #: Each id that a covers mark names, with the file and line of the mark, including
    #: the marks on tests that cover nothing.
    declared: set[tuple[str, str]] = field(default_factory=set)
    #: The covers marks the scan cannot read, each with its file and line.
    problems: set[str] = field(default_factory=set)


def scan(root: pathlib.Path = tiers.TESTS) -> Scan:
    """Read the covers marks and known-bug markers of every test under `root`."""
    result = Scan()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            _scan_file(path, result)
    return result


def _scan_file(path: pathlib.Path, result: Scan) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    read: set[int] = set()
    if path.name.startswith("test_") or path.name.endswith("_test.py"):
        runs = tiers.tier_of(path) != "quarantine"
        _scan_body(path, tree.body, _pytestmark(tree.body), runs, result, read)
    # A covers mark the walk above did not read is one pytest would apply somewhere the
    # scan does not look, or not at all. Either way the test would cover nothing without
    # saying so, so it is a problem instead.
    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for node in ast.walk(tree):
        if (id(node) not in read and id(node) not in called
                and _mark_name(node) == ("covers", True)):
            result.problems.add(
                f"{_where(path, node)}: a covers mark is read only as a decorator of a "
                "test function or test class, or in pytestmark")


def _scan_body(path: pathlib.Path, body: list[ast.stmt], marks: list[ast.expr],
               runs: bool, result: Scan, read: set[int]) -> None:
    for node in body:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test")):
            _scan_test(path, [*marks, *node.decorator_list], runs, result, read)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            _scan_body(path, node.body,
                       [*marks, *node.decorator_list, *_pytestmark(node.body)],
                       runs, result, read)


def _pytestmark(body: list[ast.stmt]) -> list[ast.expr]:
    """The marks a module or class applies to all its tests through `pytestmark`."""
    for node in body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets):
            value = node.value
            return list(value.elts) if isinstance(value, (ast.List, ast.Tuple)) else [value]
    return []


def _scan_test(path: pathlib.Path, marks: list[ast.expr], runs: bool, result: Scan,
               read: set[int]) -> None:
    covers: list[str] = []
    bugs: list[str] = []
    passes = True
    for mark in marks:
        name, is_pytest_mark = _mark_name(mark)
        if name == "covers" and is_pytest_mark:
            read.add(id(mark))
            ids = _literals(mark, path, "covers", result)
            covers += ids
            result.declared.update((_where(path, mark), item) for item in ids)
        elif name == "xfail":
            passes = False
            if not is_pytest_mark:  # known_bugs.xfail("B2"), this suite's own marker
                bugs += _literals(mark, path, "known_bugs.xfail", result)[:1]
        elif name == "skip" and is_pytest_mark:
            runs = False
    if runs:
        result.covered.update(f"bug:{bug}" for bug in bugs)
        if passes:
            result.covered.update(covers)


def _mark_name(node: ast.AST) -> tuple[str | None, bool]:
    """The name of the mark `node` applies, and whether it is one of pytest's marks.

    `pytest.mark.xfail(...)` and `mark.skip` are pytest's own. `known_bugs.xfail("B2")`
    is not: it is this suite's marker for a registered bug.
    """
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Attribute):
        base = target.value
        return target.attr, ((isinstance(base, ast.Attribute) and base.attr == "mark")
                             or (isinstance(base, ast.Name) and base.id == "mark"))
    if isinstance(target, ast.Name):
        return target.id, False
    return None, False


def _literals(mark: ast.expr, path: pathlib.Path, what: str, result: Scan) -> list[str]:
    """The string literals that `mark` is called with. Anything else is a problem, since
    the scan cannot know its value without running the test module."""
    args = mark.args if isinstance(mark, ast.Call) else []
    keywords = mark.keywords if isinstance(mark, ast.Call) else []
    values = [arg.value for arg in args
              if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
    if len(values) < len(args) or keywords:
        result.problems.add(f"{_where(path, mark)}: {what} takes string literals only, "
                            "so that the checklist can read it without importing the test")
    elif not values:
        result.problems.add(f"{_where(path, mark)}: {what} names nothing")
    return values


def _where(path: pathlib.Path, node: ast.AST) -> str:
    return f"{_shown(path)}:{getattr(node, 'lineno', 0)}"


def misdeclared(scan: Scan, items: Collection[str]) -> list[str]:
    """Each id that a covers mark names and a test cannot cover, with its place."""
    wrong = []
    for where, item in sorted(scan.declared):
        if item.startswith("bug:"):
            wrong.append(f"{where}: covers names {item}. A known bug is covered by the "
                         "test that carries its strict expected failure, "
                         f"known_bugs.xfail({item[4:]!r}), not by covers.")
        elif item not in items:
            wrong.append(f"{where}: covers names {item}, which is not on the checklist.")
    return wrong
```

- [ ] **Step 4: Run the file and see it pass**

Expected: `30 passed`.

---

### Task 7: The baseline, the ratchet and the marker

**Files:**
- Modify: `tests/harness/checklist.py` (`BASELINE`, `read_baseline`)
- Create: `tests/harness/checklist_baseline.txt`
- Modify: `tests/adversarial/conftest.py`
- Modify: `tests/adversarial/test_adv_checklist.py`
- Modify: `docs/claude/testing.md`

**Interfaces:**
- Produces: `checklist.BASELINE: pathlib.Path`, `checklist.read_baseline(path: pathlib.Path = BASELINE) -> set[str]`.

- [ ] **Step 1: Write the failing tests** (add `from memvara.server import config as server_config` to the test file's imports)

```python
@dataclasses.dataclass(frozen=True)
class Repository:
    """This checkout's checklist, the covers marks of its tests, and its baseline."""

    items: set[str]
    scan: checklist.Scan
    baseline: set[str]

    @property
    def gaps(self) -> set[str]:
        return self.items - self.scan.covered


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
        f"{len(new)} checklist items have no test, and "
        "tests/harness/checklist_baseline.txt does not list them. Write a test for each "
        "one, and mark it with @pytest.mark.covers(...):" + _listed(new))


def test_every_line_of_the_baseline_is_still_a_gap(repo: Repository) -> None:
    """The baseline only shrinks. A line whose item a test now covers, or whose item no
    longer exists, must go, or it would hide that item if its test were later lost."""
    stale = sorted(repo.baseline - repo.gaps)
    assert not stale, (
        f"{len(stale)} lines of tests/harness/checklist_baseline.txt are no longer "
        "gaps, because a test covers each one now or the item no longer exists. Delete "
        "them:" + _listed(stale))


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
    assert checklist.items() - repo.scan.covered - repo.baseline == {
        "switch:brand_new_switch"}


def test_the_baseline_leaves_out_comments_and_blank_lines(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "baseline.txt"
    path.write_text("# a comment\n\ntool:memory_recall\n  env:MEMVARA_DB  \n",
                    encoding="utf-8")
    assert checklist.read_baseline(path) == {"tool:memory_recall", "env:MEMVARA_DB"}
```

- [ ] **Step 2: Run them and see them fail**

Expected: errors in the `repo` fixture, `AttributeError: module 'harness.checklist' has no attribute 'read_baseline'`, and `test_the_covers_marker_is_registered` failing its assertion.

- [ ] **Step 3: Implement `read_baseline` and register the marker**

```python
#: The gaps that exist today, one item per line.
BASELINE = _HERE / "checklist_baseline.txt"


def read_baseline(path: pathlib.Path = BASELINE) -> set[str]:
    """The items the baseline lists, leaving out its comments and blank lines."""
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {line for line in lines if line and not line.startswith("#")}
```

In `tests/adversarial/conftest.py`:

```python
def pytest_configure(config: pytest.Config) -> None:
    """Register the covers mark, which tests/harness/checklist.py reads from each test's
    source. docs/claude/testing.md explains it."""
    config.addinivalue_line(
        "markers",
        "covers(*items): the checklist items this test covers, such as "
        "'tool:memory_recall' or 'inv:I3'. See docs/claude/testing.md.")
```

- [ ] **Step 4: Write the baseline**

Compute `sorted(checklist.items() - checklist.scan().covered)` once, in a throwaway script outside the repository, and write it to `tests/harness/checklist_baseline.txt` under a header comment that says what the file is, that a covered item's line must be deleted, and that a new item needs a test rather than a line.

- [ ] **Step 5: Run the file and see it pass**

Expected: `37 passed`.

- [ ] **Step 6: Document the covers mark and the baseline**

Extend the section in `docs/claude/testing.md`: how to declare coverage, what covers nothing, how a known bug is covered, the baseline's two failures and what to do about each, and the proof test.

- [ ] **Step 7: Commit**

```bash
git add tests/harness/checklist.py tests/harness/checklist_baseline.txt tests/adversarial/conftest.py tests/adversarial/test_adv_checklist.py docs/claude/testing.md
git commit -m "Read the covers marks in tests/ and fail the fast tier when a gap appears outside the baseline"
```

---

### Task 8: Mark the existing adversarial tests and shrink the baseline

**Files:**
- Modify: `tests/adversarial/test_adv_stdio.py`
- Modify: `tests/adversarial/test_adv_hooks.py`
- Modify: `tests/harness/checklist_baseline.txt`

A test gets an item only when its assertions check that item. The marks:

| Test | Marks |
|---|---|
| `test_adv_stdio.py::test_a_real_server_process_remembers_and_recalls_over_its_pipe` | `tool:memory_remember`, `tool:memory_recall` |
| `test_adv_stdio.py::test_the_store_outlives_the_server_process` | `tool:memory_history`, `env:MEMVARA_DB` |
| `test_adv_stdio.py::test_a_switched_off_feature_hides_its_tools` | `tool-switch:memory_add_document/documents` |
| `test_adv_stdio.py::test_a_read_only_server_lists_only_read_only_tools` | `switch:read_only`, `env:MEMVARA_READ_ONLY` |
| `test_adv_hooks.py::test_session_start_without_a_store_says_not_configured` | `hook:claude/session_start` |
| `test_adv_hooks.py::test_session_start_reads_the_store_the_client_config_names` | `hook:claude/session_start` |
| `test_adv_hooks.py::test_the_approve_hook_allows_a_read_only_memvara_tool` | `hook:claude/approve` |
| `test_adv_hooks.py::test_the_approve_hook_says_nothing_about_a_write_tool` | `hook:claude/approve` |

Not marked, and why: the read-only test asserts on annotations rather than naming a tool, so it does not mark the nine read-only pairs; the documents test names one of the four document tools, so it does not mark `switch:documents`; the harness self-tests check the harness, not memvara; the known-bug tests already cover their bugs through their markers.

- [ ] **Step 1: Add the marks**

For example:

```python
@pytest.mark.covers("tool:memory_remember", "tool:memory_recall")
def test_a_real_server_process_remembers_and_recalls_over_its_pipe(mcp: Start) -> None:
```

- [ ] **Step 2: Run the checklist and see the ratchet fail for the right reason**

Run: `... -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_checklist.py`
Expected: `test_every_line_of_the_baseline_is_still_a_gap` fails and lists the nine items the marks now cover.

- [ ] **Step 3: Delete those nine lines from the baseline**

- [ ] **Step 4: Run the checklist and the marked files, and see them pass**

Run: `... -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_checklist.py tests/adversarial/test_adv_stdio.py tests/adversarial/test_adv_hooks.py`
Expected: all pass, with the same counts for the two marked files as before.

- [ ] **Step 5: Commit**

```bash
git add tests/adversarial/test_adv_stdio.py tests/adversarial/test_adv_hooks.py tests/harness/checklist_baseline.txt
git commit -m "Mark the adversarial tests that cover checklist items, and shrink the baseline to match"
```

---

### Task 9: Verification

- [ ] **Step 1: Run the new test file 20 times in a row**, and record each run's result line. Expected: 20 identical `N passed` lines.
- [ ] **Step 2: The design's proof on a scratch copy.** Export the branch to a directory under `/private/tmp`, add `"brand_new_switch": True` to `FEATURE_DEFAULTS` in that copy's `memvara/server/config.py`, and run the checklist test there with `PYTHONPATH` set to the copy. Expected: `test_every_gap_is_listed_in_the_baseline` fails and its message lists `switch:brand_new_switch`. Record the command and the output for the report.
- [ ] **Step 3: The full gate, with a private coverage file:**

```bash
COVERAGE_FILE=$PWD/local/cov/.coverage.checklist $PY -m coverage run -m pytest -q -p no:cacheprovider
COVERAGE_FILE=$PWD/local/cov/.coverage.checklist $PY -m coverage report
```

Expected: the suite passes and `memvara/` stays at 100%.

- [ ] **Step 4: Types:** `$PY -m mypy -p memvara`, `$PY -m mypy tests/harness`, and `$PY -m mypy tests/harness --ignore-missing-imports`. Expected: `Success` from all three.
