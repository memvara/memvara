"""The scripted layer: scenarios played by a deterministic agent over the real pipe.

A scenario (tests/scenarios/schema.json) describes a few sessions of a user talking to an
agent that has memvara, and the gold that must hold afterwards. In the scripted layer every
turn carries a script: the tool calls and hook runs a careful agent would make for that
turn. This module checks scenario files, plays their scripts against a real
`python -m memvara.server` process and the plugin's real hook scripts, and checks each
gold item against what happened. docs/claude/testing.md explains the format and how to
add a scenario.

Importing this module runs nothing, because `--doctest-modules` imports every module under
tests/ while pytest collects.
"""

from __future__ import annotations

import functools
import json
import pathlib
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pytest

from harness import known_bugs, stores, tiers
from harness.env import REPO, feature_env
from harness.hooks import HookRunner, host_record
from harness.stdio import PROTOCOL, McpProcess
from memvara import MemoryType
from memvara.server.config import FEATURES
from memvara.server.mcp import SUPPORTED_PROTOCOLS
from memvara.server.tools import BY_NAME

if str(REPO) not in sys.path:  # `benchmarks` is not an installed package
    sys.path.insert(0, str(REPO))
from benchmarks.agent_memory.normalization import normalize, phrase_in  # noqa: E402

#: Where the scenario files and their schema live, and the scripted ones this module plays.
SCENARIOS = REPO / "tests" / "scenarios"
SCHEMA = SCENARIOS / "schema.json"
SCRIPTED = SCENARIOS / "scripted"

#: What the scripted layer can give a scenario. `hooks.capture` is left out on purpose:
#: capture starts an agent CLI to mine the turn, HookRunner refuses it unless it is given
#: stub CLIs, and the scripted layer gives it none. A scenario that needs it belongs to
#: the real-agent layer.
PROVIDES = frozenset({"tools", "hooks.session_start", "hooks.recall", "hooks.approve"})

#: The tiers a scripted scenario can belong to. The scenario tests live in a fast-tier
#: folder, so only a run that selects the fast tier collects them, and these are the tiers
#: such runs select. A run with --tier local or --tier quarantine never collects them.
RUN_TIERS = frozenset().union(*(wanted for wanted in tiers.SELECTS.values() if "fast" in wanted))

#: The least time, in seconds, that a mark used as an `expires_at` must leave before the
#: fact expires. The turn that checks the fact before it expires makes round trips after
#: the mark, and on a loaded machine one round trip can take more than a second.
EXPIRY_WINDOW = 4.0

#: The JSON Schema keywords `schema_errors` implements, annotations included.
KEYWORDS = frozenset({
    "$schema", "$id", "$defs", "$ref", "title", "description", "type", "properties",
    "required", "additionalProperties", "items", "enum", "const", "minItems",
    "minLength", "minimum", "pattern", "oneOf"})
_ANNOTATIONS = frozenset({"title", "description"})

#: A placeholder in a step's arguments: `{name}` for a captured value or a marked instant,
#: and `{file:<path>}` for a workspace file. Only a whole string is a placeholder, so text
#: with braces in it is never changed.
_VALUE = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_FILE = re.compile(r"\{file:([^{}]+)\}")


# -- the format --------------------------------------------------------------------------

class ScenarioError(ValueError):
    """A scenario file that does not follow the format. The message lists every problem."""


@functools.cache
def schema() -> dict[str, Any]:
    """The scenario format, as a JSON Schema. Read once, because it does not change while
    the tests run; do not modify what it returns."""
    loaded: dict[str, Any] = json.loads(SCHEMA.read_text(encoding="utf-8"))
    return loaded


def schema_errors(instance: Any, rules: Mapping[str, Any], *,
                  root: Mapping[str, Any] | None = None, path: str = "$") -> list[str]:
    """Every way `instance` breaks the JSON Schema `rules`, or an empty list.

    A small validator for the keywords tests/scenarios/schema.json uses, because the
    jsonschema package is not a dependency of this repository. A keyword it does not
    implement raises ValueError instead of being skipped, so the schema cannot state a rule
    that nothing checks.

    >>> schema_errors({"id": 3}, {"type": "object", "required": ["id", "tier"],
    ...                            "properties": {"id": {"type": "string"}}})
    ["$: missing required field 'tier'", '$.id: expected string, got integer']
    """
    root = rules if root is None else root
    unknown = set(rules) - KEYWORDS
    if unknown:
        raise ValueError(f"{path}: the schema uses {sorted(unknown)}, which this validator "
                         "does not implement")
    if "$ref" in rules:
        if set(rules) - _ANNOTATIONS - {"$ref"}:
            raise ValueError(f"{path}: a $ref must stand alone, apart from annotations")
        return schema_errors(instance, _resolve(root, rules["$ref"]), root=root, path=path)
    if "oneOf" in rules:
        return _one_of(instance, rules["oneOf"], root, path)
    if "type" in rules and not _has_type(instance, rules["type"]):
        return [f"{path}: expected {_names(rules['type'])}, got {_type_of(instance)}"]
    errors: list[str] = []
    if "const" in rules and instance != rules["const"]:
        errors.append(f"{path}: must be {rules['const']!r}")
    if "enum" in rules and instance not in rules["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {rules['enum']}")
    if isinstance(instance, str):
        if len(instance) < rules.get("minLength", 0):
            errors.append(f"{path}: must be at least {rules['minLength']} character(s) long")
        if "pattern" in rules and re.search(rules["pattern"], instance) is None:
            errors.append(f"{path}: {instance!r} does not match {rules['pattern']}")
    if "minimum" in rules and _has_type(instance, "number") and instance < rules["minimum"]:
        errors.append(f"{path}: must be at least {rules['minimum']}")
    if isinstance(instance, list):
        if len(instance) < rules.get("minItems", 0):
            errors.append(f"{path}: needs at least {rules['minItems']} item(s)")
        for index, item in enumerate(instance):
            errors += schema_errors(item, rules.get("items", {}), root=root,
                                    path=f"{path}[{index}]")
    if isinstance(instance, dict):
        errors += [f"{path}: missing required field {name!r}"
                   for name in rules.get("required", ()) if name not in instance]
        known = rules.get("properties", {})
        extra = rules.get("additionalProperties", True)
        for name, value in instance.items():
            if name in known:
                errors += schema_errors(value, known[name], root=root, path=f"{path}.{name}")
            elif extra is False:
                errors.append(f"{path}: unknown field {name!r}")
            elif isinstance(extra, Mapping):
                errors += schema_errors(value, extra, root=root, path=f"{path}.{name}")
    return errors


def _one_of(instance: Any, options: Sequence[Mapping[str, Any]], root: Mapping[str, Any],
            path: str) -> list[str]:
    """Exactly one option must match. When none does, the errors of the option that came
    closest are reported, because a step with a typo is closest to the shape its author
    meant."""
    results = [schema_errors(instance, option, root=root, path=path) for option in options]
    matched = sum(1 for errors in results if not errors)
    if matched == 1:
        return []
    if matched > 1:
        return [f"{path}: matches {matched} of the allowed shapes, and must match exactly one"]
    closest = min(results, key=len)
    return ([f"{path}: matches none of the {len(options)} allowed shapes; the closest one "
             "fails with:"] + [f"  {error}" for error in closest])


def _resolve(root: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"only references inside the schema are supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    found: Mapping[str, Any] = node
    return found


_PYTHON_TYPES: Mapping[str, Any] = {
    "object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
    "integer": int, "number": (int, float)}


def _has_type(instance: Any, wanted: str | Sequence[str]) -> bool:
    """JSON's types, in which a boolean is not a number although Python's bool is an int."""
    names = [wanted] if isinstance(wanted, str) else list(wanted)
    return any(isinstance(instance, _PYTHON_TYPES[name])
               and not (isinstance(instance, bool) and name in ("integer", "number"))
               for name in names)


def _names(wanted: str | Sequence[str]) -> str:
    return wanted if isinstance(wanted, str) else " or ".join(wanted)


def _type_of(instance: Any) -> str:
    return next((name for name in ("boolean", "integer", "number", "string", "array",
                                   "object", "null") if _has_type(instance, name)),
                type(instance).__name__)


def problems(scenario: Mapping[str, Any], *, path: pathlib.Path | None = None) -> list[str]:
    """What is wrong with a scenario that its schema cannot express, or an empty list.

    Call it only on a scenario that already matches the schema: every check here relies on
    that shape. `path`, when given, is the file the scenario came from.
    """
    found: list[str] = []
    if path is not None and path.stem != scenario["id"]:
        found.append(f"the file is {path.name}, and it must be named after the scenario's "
                     f"id: {scenario['id']}.json")
    if scenario["tier"] not in RUN_TIERS:
        *first, last = sorted(RUN_TIERS)
        found.append(f"tier {scenario['tier']!r}: the scripted layer runs only in the "
                     f"{', '.join(first)} and {last} tiers, so a {scenario['tier']} scenario "
                     "would never run")
    found += [f"forbidden names {rule['tool']}, and memvara has no tool with that name, so "
              "the rule could never match" for rule in scenario.get("forbidden", [])
              if rule["tool"] not in BY_NAME]
    found += _gold_problems(scenario)
    found += _env_problems(scenario["env"], "env")
    for number, session in enumerate(scenario["sessions"], 1):
        found += _env_problems(session.get("env", {}), f"session {number} env")
        if "user" in session.get("env", {}):
            found.append(f"session {number} env: a session cannot change the user, because "
                         "store gold reads every claim at the scenario's user; set the user "
                         "in the scenario's env")
    found += _script_problems(scenario)
    return found


def _repeated(values: Sequence[str]) -> list[str]:
    return sorted({value for value in values if values.count(value) > 1})


def _reads(turn: Mapping[str, Any]) -> bool:
    """Whether a turn has a step that can show the agent something: a tool call or a hook."""
    return any("tool" in step or "hook" in step for step in turn.get("script", []))


def _gold_problems(scenario: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    store, answers = scenario["store_gold"], scenario["answer_gold"]
    ids = [item["id"] for item in [*store, *answers]]
    if not ids:
        found.append("the scenario has no gold, so it checks nothing")
    found += [f"gold id {gold_id!r} is used more than once" for gold_id in _repeated(ids)]
    turns = [turn for session in scenario["sessions"] for turn in session["turns"]]
    turn_ids = [turn["id"] for turn in turns if "id" in turn]
    found += [f"turn id {turn_id!r} is used more than once" for turn_id in _repeated(turn_ids)]
    reads = {turn["id"]: _reads(turn) for turn in turns if "id" in turn}
    for item in store:
        if item["state"] == "absent" and {"count", "memory_type"} & set(item):
            found.append(f"store gold {item['id']!r}: a claim that is absent has no count "
                         "and no memory_type")
        elif item.get("count") == 0 and "memory_type" in item:
            found.append(f"store gold {item['id']!r}: memory_type needs at least one claim")
    for item in answers:
        if "turn" in item and item["turn"] not in turn_ids:
            found.append(f"answer gold {item['id']!r} names turn {item['turn']!r}, and no "
                         "turn has that id")
        elif not (reads[item["turn"]] if "turn" in item else _reads(turns[-1])):
            where = f"turn {item['turn']!r}" if "turn" in item else "the last turn"
            found.append(f"answer gold {item['id']!r} checks {where}, which has no tool or "
                         "hook step, so its answer is always empty and the check proves "
                         "nothing")
        for key in ("must_contain", "must_not_contain"):
            if key in item and not normalize(item[key]):
                found.append(f"answer gold {item['id']!r}: {key} has no words left once "
                             "case and punctuation are removed, so it would match every "
                             "answer or none")
        if "must_not_match" in item:
            try:
                re.compile(item["must_not_match"])
            except re.error as exc:
                found.append(f"answer gold {item['id']!r}: must_not_match is not a regular "
                             f"expression: {exc}")
    store_ids = {item["id"] for item in store}
    for gold_id, entry in scenario.get("known_bugs", {}).items():
        if gold_id not in ids:
            found.append(f"known_bugs names {gold_id!r}, which is not a gold id")
            continue
        if entry["bug"] not in known_bugs.KNOWN_BUGS:
            found.append(f"known_bugs[{gold_id!r}] names {entry['bug']}, which "
                         "tests/harness/known_bugs.py does not register")
        if ("states" in entry["symptom"]) != (gold_id in store_ids):
            found.append(f"known_bugs[{gold_id!r}]: a store gold item takes a states "
                         "symptom, and an answer gold item takes answer_contains")
    return found


def _env_problems(env: Mapping[str, Any], where: str) -> list[str]:
    found = [f"{where}: memvara has no feature named {name!r}"
             for name in env.get("features", {}) if name not in FEATURES]
    if "protocol" in env and env["protocol"] not in SUPPORTED_PROTOCOLS:
        found.append(f"{where}: the server does not speak protocol {env['protocol']!r}; it "
                     f"speaks {', '.join(SUPPORTED_PROTOCOLS)}")
    return found


def _steps(scenario: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Every script step in the order it plays, with where it is, for messages."""
    for s, session in enumerate(scenario["sessions"], 1):
        for t, turn in enumerate(session["turns"], 1):
            for n, step in enumerate(turn.get("script", []), 1):
                yield f"session {s}, turn {t}, step {n}", step


def _references(value: Any) -> Iterator[tuple[str, str]]:
    """The placeholders inside a value: ("value", name) or ("file", path)."""
    if isinstance(value, str):
        found = _VALUE.fullmatch(value)
        if found:
            yield "value", found.group(1)
        found = _FILE.fullmatch(value)
        if found:
            yield "file", found.group(1)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _inside(name: str) -> bool:
    """Whether a workspace path stays inside the workspace on every platform."""
    parts = pathlib.PurePosixPath(name).parts
    return (bool(parts) and not name.startswith("/") and "\\" not in name
            and ":" not in name and ".." not in parts)


def _script_problems(scenario: Mapping[str, Any]) -> list[str]:
    """Steps that name a tool memvara lacks, a value no earlier step sets, a workspace file
    that is not there, or a capability the scenario does not declare."""
    files = set(scenario.get("workspace", {}).get("files", {}))
    found = [f"workspace file {name!r} must be a relative path inside the workspace"
             for name in sorted(files) if not _inside(name)]
    known: set[str] = set()
    used: set[str] = set()
    # Each mark's offset, to check a mark used as an expiry against EXPIRY_WINDOW.
    offsets: dict[str, float] = {}
    for where, step in _steps(scenario):
        for key in ("args", "fields", "claim_id"):
            for kind, name in _references(step.get(key)):
                if kind == "file" and name not in files:
                    found.append(f"{where}: {{file:{name}}} names no workspace file")
                elif kind == "value" and name not in known:
                    found.append(f"{where}: {{{name}}} is used before any step sets it")
        expiry = _VALUE.fullmatch(str(step.get("args", {}).get("expires_at", "")))
        offset = offsets.get(expiry.group(1)) if expiry else None
        if expiry and offset is not None and offset < EXPIRY_WINDOW:
            found.append(f"{where}: expires_at uses the mark {expiry.group(1)!r}, which is only "
                         f"{offset:g} seconds ahead; make it at least {EXPIRY_WINDOW:g}, so "
                         "the steps before the expiry cannot race it on a slow machine")
        if "mark" in step:
            offsets[step["mark"]] = step.get("offset_seconds", 0.0)
        if "tool" in step:
            used.add("tools")
            if step["tool"] not in BY_NAME:
                found.append(f"{where}: memvara has no tool named {step['tool']!r}")
        if "hook" in step:
            used.add(f"hooks.{step['hook']}")
        if "wait_until" in step and step["wait_until"] not in known:
            found.append(f"{where}: wait_until names {step['wait_until']!r}, which no earlier "
                         "mark sets")
        for name, pattern in step.get("capture", {}).items():
            try:
                if re.compile(pattern).groups > 1:
                    found.append(f"{where}: the capture {name!r} has more than one group")
            except re.error as exc:
                found.append(f"{where}: the capture {name!r} is not a regular expression: "
                             f"{exc}")
        for name in [*step.get("capture", {}), *([step["mark"]] if "mark" in step else [])]:
            if name in known:
                found.append(f"{where}: {name!r} is set a second time")
            known.add(name)
    surfaces, requires = set(scenario["surfaces"]), set(scenario["requires"])
    if "tools" in used and "stdio" not in surfaces:
        found.append("a step calls a tool, so surfaces must list stdio")
    if any(need.startswith("hooks.") for need in used) and "hooks" not in surfaces:
        found.append("a step runs a hook, so surfaces must list hooks")
    found += [f"the script uses {need}, and requires does not list it"
              for need in sorted(used - requires)]
    found += [f"requires lists {need}, which the scripted layer cannot provide; a scenario "
              "that needs it belongs to the real-agent layer"
              for need in sorted(requires - PROVIDES)]
    return found


def load(path: pathlib.Path) -> dict[str, Any]:
    """One scenario file, checked against the schema and then against `problems`.

    Read as UTF-8, with a byte-order mark allowed, because editors on Windows add one.
    Raises ScenarioError listing everything that is wrong.
    """
    try:
        scenario = json.loads(path.read_text(encoding="utf-8-sig"))
    except ValueError as exc:
        raise ScenarioError(f"{path.name} is not JSON: {exc}") from None
    found = schema_errors(scenario, schema())
    if not found:
        found = problems(scenario, path=path)
    if found:
        raise ScenarioError(f"{path.name} does not follow tests/scenarios/schema.json:\n"
                            + "\n".join(f"- {problem}" for problem in found))
    loaded: dict[str, Any] = scenario
    return loaded


@functools.cache
def load_all(directory: pathlib.Path = SCRIPTED) -> tuple[dict[str, Any], ...]:
    """Every scenario in `directory` that loads, read once per run, because the files do
    not change while the tests run.

    A file that does not load is left out here and fails its own format test, so it is
    reported rather than silently dropped.
    """
    found = []
    for path in sorted(directory.glob("*.json")):
        try:
            found.append(load(path))
        except ScenarioError:
            continue
    return tuple(found)


def selected(scenarios: Iterable[Mapping[str, Any]], tier: str) -> list[Mapping[str, Any]]:
    """The scenarios a run with `--tier tier` includes, by each scenario's own tier."""
    wanted = tiers.SELECTS[tier]
    return [scenario for scenario in scenarios if scenario["tier"] in wanted]


# -- running -----------------------------------------------------------------------------

class RunError(RuntimeError):
    """A script that cannot go on: a placeholder with no value, or a capture that found
    nothing. The scenario stops there, and every test of it reports this error."""


#: The server settings a scenario gets for anything its `env` leaves out.
DEFAULT_ENV: Mapping[str, Any] = {
    "user": "tester", "project": None, "features": {}, "read_only": False,
    "protocol": PROTOCOL}

#: The three states a stored claim can be in. A snapshot reads all of them.
STATES = ("live", "ended", "retired")

#: A `mark` step sleeps this long on each side of the instant it records, so the steps
#: before and after it cannot share that instant, even on a clock with coarse resolution.
MARK_GAP = 0.03

#: `wait_until` sleeps this long past the marked instant.
WAIT_MARGIN = 0.05

#: The step kinds, keyed by the field that names each one.
_KINDS = (("tool", "tool"), ("hook", "hook"), ("op", "op"), ("mark", "mark"),
          ("wait_until", "wait"))

#: How each read tool begins a reply that found nothing. Such a reply shows the agent no
#: memory, although it repeats the query it was asked. `test_adv_runner.py` checks that each
#: opening is still the tool's own wording.
NOTHING_FOUND: Mapping[str, str] = {
    "memory_recall": "No stored memory matched",
    "memory_search": "No stored memory matched",
    "memory_history": "Nothing has ever been recorded for",
    "memory_standing": "No standing preferences are stored",
    "memory_ask": "Nothing in this scope matches",
    "memory_list_documents": "No documents are stored here",
    "memory_get_document": "No document with that id or custom_id is visible here",
}

#: The line the session-start hook injects first: the scope the store is bound to and how
#: many claims are visible there, as `_binding_line` in plugin/hooks/session_start.py
#: writes it. It says where memory is, not what it holds, so it is not memory shown.
SCOPE_LINE = re.compile(
    r"Memvara scope: \S+ \(tenant/user/project/agent/session; '\*' means unbound\), "
    r".+ visible\.( Session segment is bound — memory written now will NOT carry over to "
    r"other sessions\.)?")


@dataclass
class Step:
    """What one script step did, as the agent saw it."""

    kind: str
    name: str
    #: The tool's text, or the context a hook put in front of the model. Empty otherwise.
    text: str = ""
    #: False when the step did not run, as when memvara is switched off.
    ran: bool = True
    is_error: bool = False

    @property
    def shown(self) -> str:
        """The part of this step's output that shows the agent memory.

        A read that found nothing shows none. Its reply repeats the query, and the query's
        words are not memory: counted, they would let must_contain pass on a turn that
        found nothing. The session-start hook's scope line is not memory either, so that
        hook shows only what follows it.
        """
        opening = NOTHING_FOUND.get(self.name) if self.kind == "tool" else None
        if opening is not None and self.text.startswith(opening):
            return ""
        if self.kind == "hook" and self.name == "session_start":
            return "\n".join(line for line in self.text.splitlines()
                             if not SCOPE_LINE.fullmatch(line)).strip()
        return self.text


@dataclass
class Turn:
    """One user turn and the steps the scripted agent took for it."""

    session: int
    index: int
    id: str | None
    user: str
    steps: list[Step] = field(default_factory=list)

    @property
    def answer(self) -> str:
        """Every memory memvara showed the agent in this turn, in order.

        The scripted agent answers from this and from nothing else, so this is the text
        answer gold is checked on. A read that found nothing adds nothing to it (see
        `Step.shown`).
        """
        return "\n\n".join(step.shown for step in self.steps if step.shown)


@dataclass(frozen=True)
class Row:
    """One stored claim in a snapshot. Gold compares its text and state, never its id."""

    text: str
    state: str
    memory_type: str


@dataclass
class Outcome:
    """Everything one play of a scenario did, for its gold to be checked against."""

    scenario: str
    memvara: bool
    env: Mapping[str, Any]
    turns: list[Turn] = field(default_factory=list)
    #: What a reader at each project sees after the last session, in every state, keyed by
    #: project. None is user level.
    rows: dict[str | None, list[Row]] = field(default_factory=dict)
    #: Every tool call the script made, with its arguments after substitution.
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    #: Steps that did not behave the way the script said they would.
    problems: list[str] = field(default_factory=list)

    def turn(self, turn_id: str | None = None) -> Turn:
        """The turn with this id, or the last turn when no id is given."""
        if turn_id is None:
            return self.turns[-1]
        return next(turn for turn in self.turns if turn.id == turn_id)


def substitute(value: Any, values: Mapping[str, str], workspace: pathlib.Path) -> Any:
    """`value` with every placeholder replaced: `{name}` by a captured value or a marked
    instant, and `{file:<path>}` by that workspace file's contents.

    Only a string that is a placeholder and nothing else is replaced, so text that happens
    to contain braces is passed on unchanged.
    """
    if isinstance(value, str):
        found = _VALUE.fullmatch(value)
        if found:
            if found.group(1) not in values:
                raise RunError(f"{value} has no value yet; a capture or a mark earlier in "
                               "the script must set it")
            return values[found.group(1)]
        found = _FILE.fullmatch(value)
        if found:
            return (workspace / found.group(1)).read_text(encoding="utf-8")
        return value
    if isinstance(value, Mapping):
        return {key: substitute(item, values, workspace) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute(item, values, workspace) for item in value]
    return value


def mark(values: dict[str, str], name: str, offset: float = 0.0) -> None:
    """Record now plus `offset` seconds under `name`, as ISO-8601 in UTC."""
    time.sleep(MARK_GAP)
    values[name] = (datetime.now(timezone.utc) + timedelta(seconds=offset)).isoformat()
    time.sleep(MARK_GAP)


def wait_until(values: Mapping[str, str], name: str) -> None:
    """Sleep until the instant recorded under `name` has passed, by WAIT_MARGIN."""
    remaining = (datetime.fromisoformat(values[name])
                 - datetime.now(timezone.utc)).total_seconds() + WAIT_MARGIN
    if remaining > 0:
        time.sleep(remaining)


def run(scenario: Mapping[str, Any], workdir: pathlib.Path) -> Outcome:
    """Play `scenario` in `workdir` against a real server, and record what happened.

    The seed is written through the library first. Then each session starts its own
    server process on the same store file, the way a client starts one per conversation,
    and plays its turns in order. After the last session the store is read once more, with
    expiry switched off so the read neither erases nor hides an expired claim: the rows are
    what the server left on disk.
    """
    env = _env(scenario["env"])
    outcome = Outcome(scenario["id"], True, env)
    home, work, db = workdir / "home", workdir / "work", workdir / "memory.db"
    home.mkdir(parents=True)
    work.mkdir()
    for name, text in scenario.get("workspace", {}).get("files", {}).items():
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _seed(db, env, scenario.get("seed", []))
    values: dict[str, str] = {}
    for number, session in enumerate(scenario["sessions"], 1):
        _Session(f"{scenario['id']}-{number}", number,
                 _env(scenario["env"], session.get("env", {})),
                 db, home, work, values, outcome).play(session["turns"])
    outcome.rows = _snapshot(db, env["user"], _projects(scenario, env))
    return outcome


def without_memvara(scenario: Mapping[str, Any]) -> Outcome:
    """The same scenario for an agent with no memory at all: no seed, no server, no hook.

    Every step is recorded as not run, so every answer is empty and the store holds
    nothing. This is the negative control.
    """
    env = _env(scenario["env"])
    outcome = Outcome(scenario["id"], False, env)
    for number, session in enumerate(scenario["sessions"], 1):
        for index, turn in enumerate(session["turns"], 1):
            steps = [Step(_kind(step), _label(step), ran=False)
                     for step in turn.get("script", [])]
            outcome.turns.append(Turn(number, index, turn.get("id"), turn["user"], steps))
    outcome.rows = {project: [] for project in _projects(scenario, env)}
    return outcome


def _env(*layers: Mapping[str, Any]) -> dict[str, Any]:
    """DEFAULT_ENV with each layer on top. Feature switches merge one by one."""
    merged = dict(DEFAULT_ENV)
    features: dict[str, bool] = {}
    for layer in layers:
        features.update(layer.get("features", {}))
        merged.update((key, value) for key, value in layer.items() if key != "features")
    merged["features"] = features
    return merged


def _project_of(item: Mapping[str, Any], env: Mapping[str, Any]) -> str | None:
    """The project a store gold item reads at: the one it names, or the scenario's own.
    An item naming null reads at user level."""
    return item["project"] if "project" in item else env["project"]


def _projects(scenario: Mapping[str, Any], env: Mapping[str, Any]) -> list[str | None]:
    """The projects store gold reads at: the scenario's own, and any an item names."""
    found: list[str | None] = [env["project"]]
    for item in scenario["store_gold"]:
        project = _project_of(item, env)
        if project not in found:
            found.append(project)
    return found


def _kind(step: Mapping[str, Any]) -> str:
    return next(kind for key, kind in _KINDS if key in step)


def _label(step: Mapping[str, Any]) -> str:
    return str(next(step[key] for key, _ in _KINDS if key in step))


def _seed(db: pathlib.Path, env: Mapping[str, Any], ops: Sequence[Mapping[str, Any]]) -> None:
    """Write the seed through the library: memory from conversations before this one."""
    if not ops:
        return
    with stores.file(db) as mem:
        scoped = mem.scope(user=env["user"], project=env["project"])
        for op in ops:
            kind = op.get("memory_type")
            scoped.remember(op.get("subject", "user"), op["predicate"], op["object"],
                            memory_type=MemoryType(kind) if kind else None)


def _snapshot(db: pathlib.Path, user: str,
              projects: Sequence[str | None]) -> dict[str | None, list[Row]]:
    """Every claim a reader at each project sees, in every state.

    Opened with expiry switched off, so this read neither erases an expired claim nor
    hides it: the rows are what the server left on disk.
    """
    with stores.file(db, expiry_erasure=False, sweep_expired=False) as mem:
        return {project: [Row(claim.text, claim.state, claim.memory_type.value)
                          for claim in mem.scope(user=user, project=project)
                          .get_all(states=STATES)]
                for project in projects}


def _context(host: str, reply: Mapping[str, Any] | None) -> str:
    """The text a hook's reply puts in front of the model, in either envelope shape."""
    key = host_record(host).context_key
    if not reply or not key:
        return ""
    nested = reply.get("hookSpecificOutput")
    if isinstance(nested, Mapping) and key in nested:
        return str(nested[key])
    return str(reply.get(key, ""))


class _Session:
    """One conversation: its own server process, the hook runners it needs, and its turns."""

    def __init__(self, session_id: str, number: int, env: Mapping[str, Any],
                 db: pathlib.Path, home: pathlib.Path, work: pathlib.Path,
                 values: dict[str, str], outcome: Outcome) -> None:
        self.id = session_id
        self.number = number
        self.env = env
        self.db, self.home, self.work = db, home, work
        self.values = values
        self.outcome = outcome
        self.hooks: dict[str, HookRunner] = {}
        self.server = McpProcess(
            db, home=home, user=env["user"], features=env["features"],
            read_only=env["read_only"], cwd=work,
            scope={"project": env["project"]} if env["project"] else None)

    def play(self, turns: Sequence[Mapping[str, Any]]) -> None:
        """Play every turn, then close the server as a client does when the session ends.

        When a step raises, the server is killed before the error goes on, so no process
        outlives a failed run. Either way, every hook runner the session made is closed,
        which stops any recall daemon or capture child its hooks left running.
        """
        try:
            self._play(turns)
        finally:
            for runner in self.hooks.values():
                runner.close()

    def _play(self, turns: Sequence[Mapping[str, Any]]) -> None:
        try:
            agreed = self.server.initialize(self.env["protocol"]).get("protocolVersion")
            if agreed != self.env["protocol"]:
                self.outcome.problems.append(
                    f"session {self.number}: asked for protocol {self.env['protocol']} and "
                    f"the server answered {agreed}")
            for index, turn in enumerate(turns, 1):
                record = Turn(self.number, index, turn.get("id"), turn["user"])
                for n, step in enumerate(turn.get("script", []), 1):
                    where = f"session {self.number}, turn {index}, step {n}"
                    record.steps.append(self._step(step, turn, where))
                self.outcome.turns.append(record)
        except BaseException:
            self.server.kill()
            raise
        code = self.server.close()
        if code != 0:
            self.outcome.problems.append(
                f"session {self.number}: the server exited with code {code}; its stderr "
                f"ends: {self.server.stderr_text()[-300:]!r}")

    def _step(self, step: Mapping[str, Any], turn: Mapping[str, Any], where: str) -> Step:
        """Play one step with the handler for its kind, as `_kind` names it."""
        handlers = {"tool": self._tool, "hook": self._hook, "op": self._erase,
                    "mark": self._mark, "wait": self._wait}
        return handlers[_kind(step)](step, turn, where)

    def _mark(self, step: Mapping[str, Any], turn: Mapping[str, Any], where: str) -> Step:
        mark(self.values, step["mark"], step.get("offset_seconds", 0.0))
        return Step("mark", step["mark"])

    def _wait(self, step: Mapping[str, Any], turn: Mapping[str, Any], where: str) -> Step:
        wait_until(self.values, step["wait_until"])
        return Step("wait", step["wait_until"])

    def _tool(self, step: Mapping[str, Any], turn: Mapping[str, Any], where: str) -> Step:
        name = step["tool"]
        args = substitute(step.get("args", {}), self.values, self.work)
        self.outcome.calls.append((name, args))
        result = self.server.call(name, **args)
        expected = bool(step.get("expect_error", False))
        if result.is_error != expected:
            wanted = "failed" if expected else "succeeded"
            self.outcome.problems.append(
                f"{where}: {name} should have {wanted}, and it returned: {result.text[:300]!r}")
        self._capture(step, result.text, where)
        return Step("tool", name, result.text, is_error=result.is_error)

    def _hook(self, step: Mapping[str, Any], turn: Mapping[str, Any], where: str) -> Step:
        host = step.get("host", "claude")
        runner = self.hooks.get(host)
        if runner is None:
            runner = self.hooks[host] = HookRunner(host, home=self.home, cwd=self.work,
                                                   server_env=self._server_env())
        fields: dict[str, Any] = {"session": self.id}
        if step["hook"] == "recall":
            fields["prompt"] = turn["user"]
        fields.update(substitute(step.get("fields", {}), self.values, self.work))
        result = runner.run(step["hook"], **fields)
        if result.exit_code != 0:
            self.outcome.problems.append(
                f"{where}: the {step['hook']} hook exited with code {result.exit_code}; its "
                f"stderr ends: {result.stderr[-300:]!r}")
        text = _context(host, result.reply)
        self._capture(step, text, where)
        return Step("hook", step["hook"], text)

    def _erase(self, step: Mapping[str, Any], turn: Mapping[str, Any], where: str) -> Step:
        """The operator erases one claim and nothing else.

        Opened without the expiry sweep: a plain library open also erases every expired
        claim, which would do the server's expiry work for it and let a scenario that
        checks that work pass for the wrong reason.
        """
        claim_id = substitute(step["claim_id"], self.values, self.work)
        with stores.file(self.db, sweep_expired=False) as mem:
            erased = mem.scope(user=self.env["user"], project=self.env["project"]).erase(
                claim_id, sources=bool(step.get("sources", False)))
        if not erased:
            self.outcome.problems.append(
                f"{where}: the operator's erase of {claim_id} found nothing to erase")
        return Step("op", "erase")

    def _capture(self, step: Mapping[str, Any], text: str, where: str) -> None:
        for name, pattern in step.get("capture", {}).items():
            found = re.search(pattern, text, re.MULTILINE)
            if found is None:
                raise RunError(f"{where}: the capture {name!r} found nothing for "
                               f"{pattern!r} in: {text[:300]!r}")
            self.values[name] = found.group(1) if found.re.groups else found.group(0)

    def _server_env(self) -> dict[str, str]:
        """The client config's env block the hooks read to find this session's store."""
        env = {"MEMVARA_DB": str(self.db), "MEMVARA_USER": self.env["user"]}
        if self.env["project"]:
            env["MEMVARA_PROJECT"] = self.env["project"]
        if self.env["read_only"]:
            env["MEMVARA_READ_ONLY"] = "1"
        env.update(feature_env(self.env["features"]))
        return env


# -- gold --------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Gold:
    """One gold item: a claim the store must or must not hold, or a check on an answer."""

    scenario: Mapping[str, Any]
    id: str
    #: "store" or "answer".
    kind: str
    spec: Mapping[str, Any]
    #: The scenario's known_bugs entry for this item, when a known bug breaks it.
    known_bug: Mapping[str, Any] | None = None

    @property
    def test_id(self) -> str:
        return f"{self.scenario['id']}/{self.id}"


@dataclass(frozen=True)
class Verdict:
    """Whether one gold item held, what was expected and found, and the observation a known
    bug's symptom is compared with."""

    passed: bool
    detail: str
    observed: Mapping[str, Any]


def gold_items(scenario: Mapping[str, Any]) -> list[Gold]:
    """The scenario's gold items, store gold first, each with its known bug if it has one."""
    bugs = scenario.get("known_bugs", {})
    return [Gold(scenario, item["id"], kind, item, bugs.get(item["id"]))
            for kind, key in (("store", "store_gold"), ("answer", "answer_gold"))
            for item in scenario[key]]


def gold_params(scenarios: Iterable[Mapping[str, Any]]) -> list[Any]:
    """One pytest parameter per gold item. An item that a known bug breaks carries that
    bug's strict expected-failure marker, and no other item does."""
    return [pytest.param(gold, id=gold.test_id,
                         marks=[known_bugs.xfail(gold.known_bug["bug"])] if gold.known_bug
                         else [])
            for scenario in scenarios for gold in gold_items(scenario)]


def abstained(turn: Turn) -> bool:
    """Whether memvara showed the agent nothing in this turn: its answer is empty.

    That is true when every tool step replied with its tool's "nothing found" opening
    (NOTHING_FOUND) and no hook step injected anything, and when the steps did not run at
    all, as when memvara is switched off. A write receipt, or any stored memory, makes it
    false.
    """
    return not turn.answer


def check(gold: Gold, outcome: Outcome) -> Verdict:
    """Whether one gold item holds for one play of its scenario."""
    if gold.kind == "store":
        return _check_store(gold.spec, outcome)
    return _check_answer(gold.spec, outcome)


def _check_store(item: Mapping[str, Any], outcome: Outcome) -> Verdict:
    project = _project_of(item, outcome.env)
    rows = [row for row in outcome.rows[project] if row.text == item["text"]]
    if item["state"] == "absent":
        passed, wanted = not rows, "no claim in any state"
    else:
        matching = [row for row in rows if row.state == item["state"]]
        if "count" in item:
            passed = len(matching) == item["count"]
            wanted = f"exactly {item['count']} {item['state']}"
        else:
            passed, wanted = bool(matching), f"at least one {item['state']}"
        if "memory_type" in item:
            passed = passed and {row.memory_type for row in matching} == {item["memory_type"]}
            wanted += f", filed as {item['memory_type']}"
    where = f"project {project}" if project else "user level"
    found = ", ".join(f"{row.state} {row.memory_type}" for row in rows) or "nothing"
    return Verdict(passed, f"{item['text']!r} read at {where}: wanted {wanted}, found {found}",
                   {"states": sorted(row.state for row in rows)})


def _check_answer(item: Mapping[str, Any], outcome: Outcome) -> Verdict:
    """must_contain and must_not_contain use the benchmark's own token rule, `phrase_in`:
    whole words, ignoring case and punctuation. They leave out the length ceiling and the
    competitor check that `matches_value` adds for a short answer, because memvara's
    replies are long by design and a history reply names every value a slot has held."""
    turn = outcome.turn(item.get("turn"))
    text = turn.answer
    if "must_contain" in item:
        passed, wanted = phrase_in(text, item["must_contain"]), \
            f"contain {item['must_contain']!r}"
    elif "must_not_contain" in item:
        passed, wanted = not phrase_in(text, item["must_not_contain"]), \
            f"not contain {item['must_not_contain']!r}"
    elif "must_not_match" in item:
        passed = re.search(item["must_not_match"], text, re.MULTILINE) is None
        wanted = f"have no match for {item['must_not_match']!r}"
    else:
        passed, wanted = abstained(turn), "show that nothing is stored"
    return Verdict(passed, f"the answer to session {turn.session}, turn {turn.index} should "
                           f"{wanted}, and it was: {text!r}", {"answer": text})


def symptom_seen(symptom: Mapping[str, Any], verdict: Verdict) -> bool:
    """Whether a failed check shows exactly the symptom a known bug is recorded with."""
    if "states" in symptom:
        return verdict.observed.get("states") == sorted(symptom["states"])
    return str(symptom["answer_contains"]) in str(verdict.observed.get("answer", ""))


def judge(gold: Gold, outcome: Outcome) -> None:
    """Return when one gold item holds, and raise when it does not.

    The failure is known_bugs.Reproduced only when the item names a known bug and the
    failure shows that bug's own symptom. Any other failure is an AssertionError, which a
    known bug's strict marker does not absorb, so a new bug cannot hide behind a known one.
    """
    verdict = check(gold, outcome)
    if verdict.passed:
        return
    if gold.known_bug is not None and symptom_seen(gold.known_bug["symptom"], verdict):
        raise known_bugs.Reproduced(f"{gold.known_bug['bug']}: {verdict.detail}")
    raise AssertionError(verdict.detail)


def forbidden_calls(rules: Sequence[Mapping[str, Any]],
                    calls: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    """The calls that match a forbidden rule: the same tool, with every argument the rule
    names set to the rule's value."""
    return [f"{tool} with {dict(args)}" for rule in rules for tool, args in calls
            if tool == rule["tool"]
            and all(args.get(key) == value for key, value in rule.get("args", {}).items())]


def fails_without_memvara(scenario: Mapping[str, Any]) -> list[str]:
    """The gold ids that fail for an agent with no memory at all.

    An empty list means the gold cannot tell memvara working from memvara absent.
    """
    outcome = without_memvara(scenario)
    return [gold.id for gold in gold_items(scenario) if not check(gold, outcome).passed]
