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

import json
import pathlib
import re
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence

from harness import known_bugs, tiers
from harness.env import REPO
from memvara.server.config import FEATURES
from memvara.server.mcp import SUPPORTED_PROTOCOLS
from memvara.server.tools import BY_NAME

if str(REPO) not in sys.path:  # `benchmarks` is not an installed package
    sys.path.insert(0, str(REPO))
from benchmarks.agent_memory.normalization import normalize  # noqa: E402

#: Where the scenario files and their schema live, and the scripted ones this module plays.
SCENARIOS = REPO / "tests" / "scenarios"
SCHEMA = SCENARIOS / "schema.json"
SCRIPTED = SCENARIOS / "scripted"

#: What the scripted layer can give a scenario. `hooks.capture` is left out on purpose:
#: HookRunner refuses capture until stub agent CLIs exist, because capture can start the
#: real agent CLI. A scenario that needs it belongs to the real-agent layer.
PROVIDES = frozenset({"tools", "hooks.session_start", "hooks.recall", "hooks.approve"})

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


class ScenarioError(ValueError):
    """A scenario file that does not follow the format. The message lists every problem."""


def schema() -> dict[str, Any]:
    """The scenario format, as a JSON Schema."""
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
    found += _gold_problems(scenario)
    found += _env_problems(scenario["env"], "env")
    for number, session in enumerate(scenario["sessions"], 1):
        found += _env_problems(session.get("env", {}), f"session {number} env")
    found += _script_problems(scenario)
    return found


def _repeated(values: Sequence[str]) -> list[str]:
    return sorted({value for value in values if values.count(value) > 1})


def _gold_problems(scenario: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    store, answers = scenario["store_gold"], scenario["answer_gold"]
    ids = [item["id"] for item in [*store, *answers]]
    if not ids:
        found.append("the scenario has no gold, so it checks nothing")
    found += [f"gold id {gold_id!r} is used more than once" for gold_id in _repeated(ids)]
    turn_ids = [turn["id"] for session in scenario["sessions"]
                for turn in session["turns"] if "id" in turn]
    found += [f"turn id {turn_id!r} is used more than once" for turn_id in _repeated(turn_ids)]
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
    for where, step in _steps(scenario):
        for key in ("args", "fields", "claim_id"):
            for kind, name in _references(step.get(key)):
                if kind == "file" and name not in files:
                    found.append(f"{where}: {{file:{name}}} names no workspace file")
                elif kind == "value" and name not in known:
                    found.append(f"{where}: {{{name}}} is used before any step sets it")
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


def load_all(directory: pathlib.Path = SCRIPTED) -> list[dict[str, Any]]:
    """Every scenario in `directory` that loads.

    A file that does not load is left out here and fails its own format test, so it is
    reported rather than silently dropped.
    """
    found = []
    for path in sorted(directory.glob("*.json")):
        try:
            found.append(load(path))
        except ScenarioError:
            continue
    return found


def selected(scenarios: Iterable[Mapping[str, Any]], tier: str) -> list[Mapping[str, Any]]:
    """The scenarios a run with `--tier tier` includes, by each scenario's own tier."""
    wanted = tiers.SELECTS[tier]
    return [scenario for scenario in scenarios if scenario["tier"] in wanted]
