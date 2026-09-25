"""The coverage checklist: every item the adversarial suite must test, and the tests
that cover each one.

Every item is read from the code, never from a list kept by hand, so a new tool, switch
or invariant joins the checklist as soon as it exists. An item's id has the form
`kind:name`:

* `tool:<name>` for each tool in `memvara.server.tools.TOOLS`.
* `switch:<name>` for each feature switch in `memvara.server.config.FEATURES`, and
  `switch:read_only` and `switch:anchored` for the server's two modes.
* `tool-switch:<tool>/<switch>` for each tool whose `tools/list` entry changes when that
  switch is flipped from its default.
* `env:<VARIABLE>` for each `MEMVARA_*` variable that `memvara/server/config.py` reads,
  found by walking its syntax tree.
* `hook:<host>/<hook>` for each hook that a host in `plugin/hooks/hosts/` fires.
* `inv:<id>` for each numbered invariant in `docs/INTERNALS.md`, and each bullet under
  "Invariants and assumptions" on a `docs/claude/` page. `invariant_ids.json` gives each
  one an id that survives rewording.
* `silent:<name>` for each silent failure mode that the `memvara/telemetry.py` docstring
  lists.
* `bug:<id>` for each open bug in `known_bugs.KNOWN_BUGS`.

docs/claude/testing.md says how a test declares what it covers, and how the baseline of
today's gaps is kept.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Collection

from memvara.server import config as server_config
from memvara.server.mcp import MemvaraMCPServer
from memvara.server.tools import TOOLS

from . import stores, tiers
from .env import REPO
from .hooks import HOOKS_DIR, host_record
from .known_bugs import KNOWN_BUGS

_HERE = pathlib.Path(__file__).resolve().parent
#: The gaps that exist today, one item per line.
BASELINE = _HERE / "checklist_baseline.txt"
#: The stable id of every documented invariant.
INVARIANT_IDS = _HERE / "invariant_ids.json"
CONFIG = REPO / "memvara" / "server" / "config.py"
TELEMETRY = REPO / "memvara" / "telemetry.py"

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
            features_off: Collection[str] | None = None) -> dict[str, Any]:
    """The `tools/list` reply of a server with these settings, keyed by tool name.

    The server runs in this process, over an in-memory store. `python -m memvara.server`
    builds the same `MemvaraMCPServer` from the same three settings
    (memvara/server/cli.py), so this is the list the real process sends. `features_off`
    defaults to the features that are off by default, read when this runs, for the reason
    `_switches` reads FEATURES then.
    """
    if features_off is None:
        features_off = server_config.FEATURES_OFF_BY_DEFAULT
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
    # The second column starts where the border's second run of "=" does. The search
    # starts after any indent, so that a table set in by a few spaces reads the same.
    border = lines[borders[0]]
    indent = len(border) - len(border.lstrip())
    column = border.index("=", border.index(" ", indent))
    rows: list[str] = []
    for line in lines[borders[1] + 1:borders[2]]:
        first, second = line[:column].strip(), line[column:].strip()
        if first and second:
            rows.append(first)
        elif first and rows:
            rows[-1] += f" {first}"
    if not rows:
        raise ChecklistError(
            f"the table of silent failure modes in {_shown(path)} has no rows")
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
    that a test of the invariant covers both.
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


def items() -> set[str]:
    """Every item on the checklist.

    Every source except the known bugs must yield at least one item. One that yields
    none has changed shape, a renamed heading for example, and reading nothing from it
    would drop its items from the checklist, and nothing would report it. The known bugs
    may run out, because that is the goal.
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


#: Items that no test can check, each with the reason. They are rules for people, listed
#: among a page's invariants, rather than behaviour of memvara. They stay on the
#: checklist, so that a reworded rule is still noticed, but they are never gaps.
EXEMPT: dict[str, str] = {
    "inv:TB1": "A rule for how a person checks work: an output is compared, not an exit "
               "status. It describes a method, not anything memvara does.",
    "inv:TB2": "A rule for how a person reports a benchmark number, with its caveat. It "
               "governs a document, not code.",
    "inv:RC5": "A commercial boundary: the hosted server lives in another repository, so "
               "there is no code here for a test to run.",
    "inv:RP1": "A rule of the release process, which the suite leaves out of scope: the "
               "package index refuses to replace a published version.",
}


def gaps(items: Collection[str], covered: Collection[str]) -> set[str]:
    """The items that no test covers and that are not exempt."""
    return set(items) - set(covered) - set(EXEMPT)


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
    """Read the covers marks and known-bug markers of every test under `root`.

    The files are parsed, never imported, because importing a nightly or local test can
    need Docker or a package that this machine does not have. A test covers nothing when
    it is not expected to pass: when it is marked xfail, marked skip with no condition,
    or sits in the quarantine tier. A known bug is covered by the test that carries its
    strict expected failure, `known_bugs.xfail("B2")`.
    """
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
    # A covers mark the walk above did not read is one that pytest applies somewhere the
    # scan does not look, or not at all. Either way its test would cover nothing without
    # saying so, so it is reported instead.
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
    """The marks that a module or class applies to all its tests through `pytestmark`."""
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
    """The name of the mark that `node` applies, and whether it is one of pytest's marks.

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
    """The string literals that `mark` is called with. Anything else is a problem,
    because the scan cannot know its value without running the test module."""
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


def read_baseline(path: pathlib.Path = BASELINE) -> set[str]:
    """The items the baseline lists, leaving out its comments and blank lines."""
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def misdeclared(scan: Scan, items: Collection[str]) -> list[str]:
    """Each id that a covers mark names and that no test can cover, with its place."""
    wrong = []
    for where, item in sorted(scan.declared):
        if item.startswith("bug:"):
            wrong.append(f"{where}: covers names {item}. A known bug is covered by the "
                         "test that carries its strict expected failure, "
                         f"known_bugs.xfail({item[4:]!r}), not by covers.")
        elif item not in items:
            wrong.append(f"{where}: covers names {item}, which is not on the checklist.")
    return wrong
