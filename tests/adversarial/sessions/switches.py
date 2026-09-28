"""The settings that change the MCP server's tool surface, what a server started with any
combination of them must serve, and the combinations each tier starts real servers with.

A server's `tools/list` answer depends on eleven settings: nine feature switches, read-only
mode and anchored mode. This module names them, predicts the answer a server started with
any combination of them must give (the oracle), and checks a server against it, either in
this process or over the real pipe. docs/claude/testing.md explains how the tests use it.

The oracle takes its facts from the tables in the code, not from a copy written by hand:
the tools, their arguments, their descriptions and which ones write come from
`memvara.server.tools.TOOLS`, the arguments a feature removes from `FEATURE_ARGUMENTS`, and
the replacement descriptions from the constants `tools.py` writes them with. What this
module states itself are the rules that combine those facts, because the rules are what
the tests check.

Importing this module starts nothing and opens nothing.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import itertools
import pathlib
import shutil
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from harness import stores
from harness.env import child_env, feature_env
from harness.stdio import McpProcess, ToolResult
from memvara import Memvara
from memvara.server import tools as server_tools
from memvara.server.config import FEATURE_DEFAULTS, ServerConfig, build_memvara
from memvara.server.mcp import MemvaraMCPServer
from memvara.server.tools import FEATURE_ARGUMENTS, TOOLS, Tool
from memvara.types import Claim

# -- the settings ----------------------------------------------------------------------

#: Features that own a tool: with the feature off, the server does not list the tool.
TOOL_FEATURES: tuple[str, ...] = tuple(sorted({t.feature for t in TOOLS if t.feature}))

#: Features that own arguments and remove them when switched off.
ARGUMENT_FEATURES: tuple[str, ...] = tuple(sorted(FEATURE_ARGUMENTS))


@dataclass(frozen=True)
class Redescribed:
    """A feature that, when it is switched off, keeps its arguments and changes their
    descriptions to `description`."""

    feature: str
    arguments: tuple[str, ...]
    description: str

    def governs(self, tool: Tool) -> bool:
        """Whether this rule applies to `tool`: the tool takes every argument the rule
        names. Taking one is not enough. memory_list_documents takes `filepath_prefix`
        alone, to list one folder of documents, and metadata_filters leaves it alone."""
        return set(self.arguments) <= set(tool.properties)


#: The two features that keep their arguments when off. `metadata_filters` rewrites the
#: two filter arguments of memory_search and memory_recall (config.py: it "decides whether
#: memory_search and memory_recall accept filters and filepath_prefix"); `expiry_erasure`
#: rewrites memory_remember's two expiry arguments. The texts are tools.py's own.
REDESCRIBED: tuple[Redescribed, ...] = (
    Redescribed("metadata_filters", ("filters", "filepath_prefix"),
                server_tools._FILTERS_OFF),
    Redescribed("expiry_erasure", server_tools._EXPIRY_ARGUMENTS,
                server_tools._EXPIRES_AT_OFF),
)

#: What each `anchored` argument becomes on a server that anchors by default: the schema a
#: tool declares, paired with the one that replaces it. These are tools.py's own four dicts.
ANCHORED_ON: tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...] = (
    (server_tools._ANCHORED, server_tools._ANCHORED_ON),
    (server_tools._ANCHORED_ASK, server_tools._ANCHORED_ASK_ON),
)

#: Every feature that changes what the server lists.
FEATURE_SWITCHES: tuple[str, ...] = (
    TOOL_FEATURES + ARGUMENT_FEATURES + tuple(rule.feature for rule in REDESCRIBED))

#: The eleven settings, in the order the arrays below give them columns.
SWITCHES: tuple[str, ...] = FEATURE_SWITCHES + ("read_only", "anchored")

_SETTINGS = frozenset(SWITCHES)


@dataclass(frozen=True)
class Combination:
    """One way to start a server: the settings that are moved away from their defaults.

    Every feature in SWITCHES is on by default today, so moving one switches it off.
    Read-only mode and anchored mode are off by default, so moving one switches it on.
    """

    moved: frozenset[str]

    def __post_init__(self) -> None:
        unknown = self.moved - _SETTINGS
        if unknown:
            raise ValueError(f"not a setting that changes the tool list: {sorted(unknown)}")

    @classmethod
    def of(cls, *moved: str) -> Combination:
        return cls(frozenset(moved))

    @property
    def read_only(self) -> bool:
        return "read_only" in self.moved

    @property
    def anchored(self) -> bool:
        return "anchored" in self.moved

    def feature_on(self, feature: str) -> bool:
        return FEATURE_DEFAULTS[feature] != (feature in self.moved)

    def env(self) -> dict[str, str]:
        """The variables that start a server this way, one for every setting."""
        found = feature_env({name: self.feature_on(name) for name in FEATURE_SWITCHES})
        found["MEMVARA_READ_ONLY"] = "1" if self.read_only else "0"
        found["MEMVARA_ANCHORED"] = "1" if self.anchored else "0"
        return found

    @property
    def label(self) -> str:
        """The moved settings in SWITCHES order, or `defaults`, for test ids."""
        return "+".join(s for s in SWITCHES if s in self.moved) or "defaults"


# -- the oracle ------------------------------------------------------------------------

def unavailable(tool: Tool, combination: Combination) -> str | None:
    """Why a server started this way does not list `tool`, or None when it does.

    The feature comes first: mcp.py checks it before read-only mode, so a switched-off
    write tool on a read-only server is refused for its feature.
    """
    if tool.feature is not None and not combination.feature_on(tool.feature):
        return "feature"
    if combination.read_only and tool.writes:
        return "read_only"
    return None


def removed(combination: Combination) -> frozenset[str]:
    """The arguments no tool offers on a server started this way."""
    return frozenset(argument for feature in ARGUMENT_FEATURES
                     if not combination.feature_on(feature)
                     for argument in FEATURE_ARGUMENTS[feature])


def _argument(tool: Tool, name: str, schema: Mapping[str, Any],
              combination: Combination) -> Mapping[str, Any]:
    """One argument's schema on a server started with `combination`."""
    if name == "anchored" and combination.anchored:
        replacements = [on for off, on in ANCHORED_ON if off == schema]
        if not replacements:
            raise LookupError(f"{tool.name} takes `anchored`, and ANCHORED_ON does not pair "
                              "its schema with a replacement; add the pair")
        if len(replacements) > 1:
            raise LookupError(f"{tool.name} takes `anchored`, and ANCHORED_ON pairs its "
                              f"schema with {len(replacements)} replacements, so which one "
                              "the server uses cannot be told; keep one pair for each schema")
        [schema] = replacements
    for rule in REDESCRIBED:
        if (not combination.feature_on(rule.feature) and name in rule.arguments
                and rule.governs(tool)):
            schema = {**schema, "description": rule.description}
    return schema


def _description(tool: Tool, served: Mapping[str, Any]) -> str:
    """One tool's description on a server that serves `served` of its arguments.

    A tool whose description names arguments a switch can remove is described for the
    ones it keeps (`DESCRIBED_BY_ARGUMENTS` in memvara/server/tools.py, #295); every other
    tool keeps its description whatever is removed."""
    describe = server_tools.DESCRIBED_BY_ARGUMENTS.get(tool.name)
    if describe is None or set(served) == set(tool.properties):
        return tool.description
    return describe(set(served))


@dataclass(frozen=True)
class Expected:
    """What the oracle predicts for a server started with one combination. It is worked
    out once for each combination, and the three checks below share it."""

    combination: Combination
    #: Each tool the server must not list, with the reason: "feature" or "read_only".
    unavailable: Mapping[str, str]
    #: The arguments no tool may offer.
    removed: frozenset[str]
    #: Whether the server's memory is a hosted deployment. Such a server describes
    #: `memory_forget` and `memory_end` by what every hosted release closes, as
    #: `for_a_hosted_deployment` in memvara/server/tools.py sets out.
    hosted: bool = False

    @classmethod
    def of(cls, combination: Combination, *, hosted: bool = False) -> Expected:
        reasons = {tool.name: unavailable(tool, combination) for tool in TOOLS}
        return cls(combination,
                   {name: reason for name, reason in reasons.items() if reason is not None},
                   removed(combination), hosted)

    @property
    def listed(self) -> tuple[Tool, ...]:
        """The tools the server must list, in the order it must list them."""
        tools = server_tools.for_a_hosted_deployment(TOOLS) if self.hosted else TOOLS
        return tuple(tool for tool in tools if tool.name not in self.unavailable)

    def served(self) -> list[dict[str, Any]]:
        """The `tools/list` answer the server must give."""
        answer = []
        for tool in self.listed:
            properties = {name: dict(_argument(tool, name, schema, self.combination))
                          for name, schema in tool.properties.items()
                          if name not in self.removed}
            answer.append({
                "name": tool.name,
                "description": _description(tool, properties),
                "inputSchema": {"type": "object", "properties": properties,
                                "required": [r for r in tool.required if r in properties],
                                "additionalProperties": False},
                "annotations": {"readOnlyHint": not tool.writes,
                                "destructiveHint": tool.destructive,
                                "openWorldHint": False},
            })
        return answer


def served(combination: Combination) -> list[dict[str, Any]]:
    """The `tools/list` answer a server started with `combination` must give, in order."""
    return Expected.of(combination).served()


# -- the minimal calls -----------------------------------------------------------------

@dataclass(frozen=True)
class Ids:
    """The ids a minimal call names: two facts to link, one to retire, one to end, and a
    document."""

    linked_from: str
    linked_to: str
    forgettable: str
    endable: str
    document: str


#: Ids that name nothing, for calls that are refused before anything runs.
NOWHERE = Ids("cl_00000000000000000000", "cl_00000000000000000001",
              "cl_00000000000000000002", "cl_00000000000000000003",
              "doc_00000000000000000000")


def minimal_calls(ids: Ids) -> dict[str, dict[str, Any]]:
    """For every tool, the least a caller must send for the call to run.

    That is the tool's required arguments, plus one more for the five tools whose schema
    requires nothing that a call can run without: memory_forget and memory_end need a claim
    to close, memory_end_matching and memory_forget_matching need a query for their
    preview, and memory_add_document needs its content.
    """
    return {
        "memory_recall": {"query": "where does the user live"},
        "memory_search": {"query": "where does the user live"},
        "memory_neighborhood": {"entity": "Oslo"},
        "memory_paths": {"source": "user", "target": "Oslo"},
        "memory_ask": {"question": "where does the user live"},
        "memory_since": {"since": "2024-01-01"},
        "memory_standing": {},
        "memory_profile": {},
        "memory_add": {"text": "Thanks, that is all for today."},
        "memory_remember": {"predicate": "likes", "object": "chess"},
        "memory_forget": {"claim_id": ids.forgettable},
        "memory_end": {"claim_id": ids.endable},
        "memory_end_matching": {"query": "chess club"},
        "memory_forget_matching": {"query": "chess club"},
        "memory_link": {"from_id": ids.linked_from, "to_id": ids.linked_to,
                        "relation": "extends"},
        "memory_history": {"predicate": "lives_in"},
        "memory_why": {"claim_id": ids.linked_from},
        "memory_stats": {},
        "memory_add_document": {"content": "Notes from the chess club meeting."},
        "memory_get_document": {"id": ids.document},
        "memory_list_documents": {},
        "memory_delete_document": {"id": ids.document},
    }


#: The value each argument a feature can remove is sent with, to check it is refused.
PROBES: Mapping[str, Any] = {"reason": "a reason", "until_reason": "a reason",
                             "query_rewrite": False, "synthesize": False}

#: The rule for the filter arguments: metadata_filters governs every one of them.
[FILTERS] = [rule for rule in REDESCRIBED if rule.feature == "metadata_filters"]

#: What a filter argument is sent with to probe it, by its JSON type: `filters` takes an
#: object and `filepath_prefix` a string.
FILTER_VALUES: Mapping[str, Any] = {"object": {"team": "support"}, "string": "policies/"}


def seed(memory: Memvara) -> Ids:
    """Store four facts and a document for the minimal calls to name, and return their
    ids."""
    scoped = memory.scope(user=USER)
    home = scoped.remember("user", "lives_in", "Oslo").added[0]
    work = scoped.remember("user", "works_at", "Contoso").added[0]
    hobby = scoped.remember("user", "likes", "chess").added[0]
    language = scoped.remember("user", "speaks", "Norwegian").added[0]
    document = scoped.add_document("Notes about the office move to Oslo.",
                                   custom_id="notes/office.md")
    return Ids(linked_from=work.id, linked_to=home.id, forgettable=hobby.id,
               endable=language.id, document=str(document.custom_id))


# -- talking to a server ---------------------------------------------------------------

#: The user every server here is bound to, and the file name of its store.
USER = "tester"
STORE = "memory.db"


class Client(Protocol):
    """A server as these checks talk to it: over the real pipe (`McpProcess`) or in this
    process (`InProcess`)."""

    def list_tools(self) -> list[dict[str, Any]]: ...

    def call(self, name: str, /, **arguments: Any) -> ToolResult: ...


class InProcess:
    """A server in this test's own process, reached through `handle_message`: the method
    its stdio loop calls for every line it reads."""

    def __init__(self, server: MemvaraMCPServer) -> None:
        self.server = server
        self._next_id = 0

    def request(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            message["params"] = dict(params)
        reply = self.server.handle_message(message)
        if reply is None or "error" in reply:
            raise AssertionError(f"{method} was not answered with a result: {reply!r}")
        result: dict[str, Any] = reply["result"]
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self.request("tools/list")["tools"])

    def call(self, name: str, /, **arguments: Any) -> ToolResult:
        return ToolResult.parse(
            self.request("tools/call", {"name": name, "arguments": arguments}))


def base_env(home: pathlib.Path) -> dict[str, str]:
    """The MEMVARA_ variables every child process of the suite starts with
    (`harness.env.child_env`), so that a server built in this process is configured like
    one started over the pipe. `home` is only what child_env asks for."""
    return {name: value for name, value in child_env(home).items()
            if name.startswith("MEMVARA_")}


def _config(combination: Combination, base: Mapping[str, str]) -> ServerConfig:
    """The configuration `python -m memvara.server` reads from `base` and the
    combination's variables, for an in-memory store."""
    return ServerConfig.from_env({**base, "MEMVARA_DB": ":memory:", "MEMVARA_USER": USER,
                                  **combination.env()})


def _server(memory: Memvara, config: ServerConfig) -> MemvaraMCPServer:
    """The server over `memory`, built from `config` as `python -m memvara.server` builds
    it."""
    return MemvaraMCPServer(memory, read_only=config.read_only, anchored=config.anchored,
                            features_off=config.features_off, **config.scope_kwargs)


def in_process(memory: Memvara, combination: Combination,
               base: Mapping[str, str]) -> InProcess:
    """A server over a shared engine, `memory`, configured the way `python -m
    memvara.server` configures itself: `ServerConfig.from_env` reads `base`, then the
    combination's variables.

    One engine is shared by every server, which keeps 2,048 of them cheap. A server with
    metadata filters off switches them off on the engine it is given, so they are
    switched back on before each server is built.
    """
    memory.metadata_filters = True
    return InProcess(_server(memory, _config(combination, base)))


@contextlib.contextmanager
def seeded_in_process(combination: Combination,
                      base: Mapping[str, str]) -> Iterator[tuple[InProcess, Ids]]:
    """A server in this process with an engine of its own, both built the way `python -m
    memvara.server` builds them, over a new in-memory store that `seed` fills. It is what
    a check that runs the minimal calls needs, because those calls write."""
    config = _config(combination, base)
    memory = build_memvara(config)
    assert isinstance(memory, Memvara), "a local configuration builds a local engine"
    try:
        ids = seed(memory)
        yield InProcess(_server(memory, config)), ids
    finally:
        memory.close()


# -- the checks ------------------------------------------------------------------------

def listing_problems(client: Client, expected: Expected) -> list[str]:
    """How the tool list a server gives differs from the one the oracle predicts."""
    got, wanted = client.list_tools(), expected.served()
    problems = []
    names, wanted_names = [t.get("name") for t in got], [t["name"] for t in wanted]
    if names != wanted_names:
        problems.append(f"lists {names}, and the oracle expects {wanted_names}")
    by_name = {tool["name"]: tool for tool in wanted}
    for tool in got:
        if tool.get("name") in by_name and tool != by_name[tool["name"]]:
            problems.append(f"{tool['name']} differs in "
                            f"{_differences(tool, by_name[tool['name']])}")
    return problems


def _differences(got: Mapping[str, Any], wanted: Mapping[str, Any]) -> str:
    """Which parts of one listed tool differ from the oracle's, for a failure message."""
    parts = [key for key in ("description", "annotations") if got.get(key) != wanted[key]]
    got_schema, wanted_schema = got.get("inputSchema", {}), wanted["inputSchema"]
    got_args, wanted_args = got_schema.get("properties", {}), wanted_schema["properties"]
    for label, names in (("arguments it should not offer", set(got_args) - set(wanted_args)),
                         ("arguments it should offer", set(wanted_args) - set(got_args)),
                         ("arguments with another schema",
                          {n for n in set(got_args) & set(wanted_args)
                           if got_args[n] != wanted_args[n]})):
        if names:
            parts.append(f"{label}: {sorted(names)}")
    parts += [f"inputSchema.{key}" for key in ("type", "required", "additionalProperties")
              if got_schema.get(key) != wanted_schema[key]]
    parts += [f"field {key!r}" for key in sorted(set(got) ^ set(wanted))]
    return "; ".join(parts)


#: One filter probe: the tool's name, the filter argument it was sent, and the answer.
Probe = tuple[str, str, ToolResult]


def _filter_probes(client: Client, expected: Expected,
                   calls: Mapping[str, Mapping[str, Any]]) -> Iterator[Probe]:
    """Each listed tool that metadata_filters governs, called once with each filter
    argument added to its minimal call. refusal_problems and call_problems both judge
    these."""
    for tool in expected.listed:
        if FILTERS.governs(tool):
            for argument in FILTERS.arguments:
                value = FILTER_VALUES[tool.properties[argument]["type"]]
                yield tool.name, argument, client.call(
                    tool.name, **{**calls[tool.name], argument: value})


def refusal_problems(client: Client, expected: Expected,
                     calls: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """What is wrong with how a server refuses what it does not offer.

    A tool it does not list must be refused by name, with the reason: the feature and its
    variable for a switched-off tool, or read-only mode for a write tool. An argument a
    switch removed must be refused as unknown. On a server without metadata filters, a
    read that carries either filter argument must be refused, naming the switch. None of
    these calls reaches the store, because each is refused before anything runs.
    """
    problems = []
    for name, reason in expected.unavailable.items():
        feature = server_tools.BY_NAME[name].feature
        needs = (f"MEMVARA_FEATURE_{str(feature).upper()}=0" if reason == "feature"
                 else "this memory server is read-only")
        result = client.call(name, **calls[name])
        if not (result.is_error and result.text.startswith(f"{name} is unavailable: ")
                and needs in result.text):
            problems.append(f"{name} should be refused naming {needs!r}, and the "
                            f"server answered: {result.text[:200]!r}")
    for tool in expected.listed:
        for argument in sorted(expected.removed & set(tool.properties)):
            result = client.call(tool.name, **{**calls[tool.name], argument: PROBES[argument]})
            needs = f"{tool.name}: unknown argument(s) '{argument}'"
            if not (result.is_error and result.text.startswith(needs)):
                problems.append(f"{tool.name} should refuse {argument!r} as unknown, and the "
                                f"server answered: {result.text[:200]!r}")
    if not expected.combination.feature_on(FILTERS.feature):
        for name, argument, result in _filter_probes(client, expected, calls):
            if not (result.is_error and "MEMVARA_FEATURE_METADATA_FILTERS=0" in result.text):
                problems.append(f"{name} should refuse {argument!r}, naming the switch, "
                                f"and the server answered: {result.text[:200]!r}")
    return problems


def call_problems(client: Client, expected: Expected,
                  calls: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Each tool a server lists must run with its minimal call, and with metadata filters
    on, a read with each filter argument must run too."""
    problems = []
    for tool in expected.listed:
        result = client.call(tool.name, **calls[tool.name])
        if result.is_error:
            problems.append(f"{tool.name} with {dict(calls[tool.name])} failed: "
                            f"{result.text[:300]!r}")
    if expected.combination.feature_on(FILTERS.feature):
        for name, argument, result in _filter_probes(client, expected, calls):
            if result.is_error:
                problems.append(f"{name} with {argument!r} failed: {result.text[:300]!r}")
    return problems


def _problems(client: Client, expected: Expected, calls: Mapping[str, Mapping[str, Any]],
              *, refusals: bool, run: bool) -> list[str]:
    """The list check, then the refusal check if `refusals`, then the call check if
    `run`."""
    problems = listing_problems(client, expected)
    if refusals:
        problems += refusal_problems(client, expected, calls)
    if run:
        problems += call_problems(client, expected, calls)
    return problems


def in_process_failures(combinations: Sequence[Combination], base: Mapping[str, str], *,
                        refusals: bool = True,
                        calls: bool = False) -> dict[Combination, list[str]]:
    """Each combination whose server, built in this process, differs from what the oracle
    predicts, with its problems.

    The list is always checked, and with `refusals` what the server refuses. None of that
    reaches the store, so one engine serves every server. With `calls`, every tool a
    server lists also runs its minimal call. Calls write, so each server then gets an
    engine and a seeded store of its own (`seeded_in_process`).
    """
    found: dict[Combination, list[str]] = {}
    if calls:
        for combination in combinations:
            with seeded_in_process(combination, base) as (client, ids):
                problems = _problems(client, Expected.of(combination), minimal_calls(ids),
                                     refusals=refusals, run=True)
            if problems:
                found[combination] = problems
        return found
    memory = stores.memory()
    try:
        nowhere = minimal_calls(NOWHERE)
        for combination in combinations:
            problems = _problems(in_process(memory, combination, base),
                                 Expected.of(combination), nowhere,
                                 refusals=refusals, run=False)
            if problems:
                found[combination] = problems
    finally:
        memory.close()
    return found


def summary(failures: Mapping[Combination, Sequence[str]], total: int) -> str:
    """A failure message: how many combinations failed, and the problems of the first
    three."""
    first = "\n".join(f"{combination.label}: {list(problems)}"
                      for combination, problems in list(failures.items())[:3])
    return f"{len(failures)} of {total} combinations failed. The first:\n{first}"


# -- the arrays of combinations --------------------------------------------------------

#: The first row of the 12-run Plackett-Burman design, one sign per setting, `+` for a
#: moved setting. Rows 2 to 11 shift it right one more place each; row 12 moves nothing.
_PLACKETT_BURMAN_12 = "++-+++---+-"


def _combination(moved: Sequence[bool]) -> Combination:
    return Combination(frozenset(s for s, move in zip(SWITCHES, moved) if move))


@functools.cache
def fast_runs() -> tuple[Combination, ...]:
    """The 12-run Plackett-Burman design, an orthogonal array of strength 2: each two
    settings are seen in each of their four combinations exactly three times. It also
    covers all eight combinations of every three settings."""
    if len(SWITCHES) != len(_PLACKETT_BURMAN_12):
        raise ValueError(
            f"the 12-run array has a column for each of 11 settings, and there are "
            f"{len(SWITCHES)}; more settings need a larger array, such as the 20-run "
            "Plackett-Burman design")
    rows = [_PLACKETT_BURMAN_12[-i:] + _PLACKETT_BURMAN_12[:-i] for i in range(11)]
    rows.append("-" * len(_PLACKETT_BURMAN_12))
    return tuple(_combination([sign == "+" for sign in row]) for row in rows)


@functools.cache
def nightly_runs() -> tuple[Combination, ...]:
    """The fold-over of `fast_runs`: each run with every setting reversed. It covers all
    eight combinations of every three settings on its own, and together with `fast_runs`
    it is an orthogonal array of strength 3, in which each combination of three settings
    appears exactly three times."""
    return tuple(Combination(_SETTINGS - run.moved) for run in fast_runs())


@functools.cache
def every_combination() -> tuple[Combination, ...]:
    """All 2 ** 11 = 2,048 combinations, the defaults first."""
    return tuple(_combination(moved)
                 for moved in itertools.product((False, True), repeat=len(SWITCHES)))


def interaction_counts(runs: Sequence[Combination],
                       strength: int) -> dict[tuple[str, ...], Counter[tuple[bool, ...]]]:
    """For each group of `strength` settings, how many runs move each subset of them."""
    return {group: Counter(tuple(s in run.moved for s in group) for run in runs)
            for group in itertools.combinations(SWITCHES, strength)}


# -- real servers ----------------------------------------------------------------------

#: How long after it is written the template's door code expires. The template waits for
#: it, so every server starts on a store that holds an expired fact, which a read-only
#: server must hide and must not erase.
EXPIRES_AFTER = timedelta(seconds=1)

#: Everything a store holds: every row of every table as SQL, then a SHA-256 of the vector
#: file and one of the embedder record.
Dump = tuple[tuple[str, ...], str, str]


def _digest(path: pathlib.Path) -> str:
    """The SHA-256 of the file at `path`, or "absent" when there is no such file."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "absent"


def store_dump(path: pathlib.Path) -> Dump:
    """Everything the store at `path` holds, to tell whether anything wrote to it.

    The database is read with sqlite3 alone, opened read-only, so reading it changes
    nothing. Every row of every table is compared, including rows no read returns, such as
    links, turns and the records of erasures. The two files beside the database are
    compared by their hashes.

    This is not the upgrade tests' `golden.snapshot()`, because that one also records
    SQLite's write-ahead log and shared-memory file. The writer that seeds the template
    leaves an empty log and a shared-memory file beside it, and a read-only server removes
    both when it closes, without changing a row, so `golden.changes()` would report every
    read-only run as a change. `golden.snapshot()` also opens the database for writing,
    which removes those two files from the store it reads.
    """
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        lines = tuple(sorted(connection.iterdump()))
    finally:
        connection.close()
    return (lines, _digest(path.with_name(path.name + ".vecs")),
            _digest(path.with_name(path.name + ".embedder.json")))


def read_claims(path: pathlib.Path) -> list[Claim]:
    """Every claim the user has, in every state, read with expiry off, so the read neither
    erases nor hides a fact whose expiry has passed."""
    with stores.file(path, expiry_erasure=False, sweep_expired=False) as memory:
        return list(memory.scope(user=USER).get_all(states=("live", "ended", "retired")))


@dataclass(frozen=True)
class Template:
    """A seeded store in `directory`, copied for each server so that every run starts from
    the same memory, and a dump of what it holds."""

    directory: pathlib.Path
    ids: Ids
    dump: Dump

    @classmethod
    def build(cls, directory: pathlib.Path) -> Template:
        """Store what `seed` stores, and a door code that has expired by the time this
        returns."""
        directory.mkdir(parents=True, exist_ok=True)
        with stores.file(directory / STORE) as memory:
            ids = seed(memory)
            expires = datetime.now(timezone.utc) + EXPIRES_AFTER
            memory.scope(user=USER).remember("user", "door_code", "4417", expires_at=expires)
        time.sleep(max(0.0, (expires - datetime.now(timezone.utc)).total_seconds()) + 0.05)
        return cls(directory, ids, store_dump(directory / STORE))

    def copy(self, target: pathlib.Path) -> pathlib.Path:
        """The store copied into `target`, with every file beside it whose name starts
        with the store's."""
        target.mkdir(parents=True, exist_ok=True)
        for source in sorted(self.directory.glob(STORE + "*")):
            shutil.copy2(source, target / source.name)
        return target / STORE


def over_the_pipe(start: Callable[..., McpProcess], combination: Combination,
                  template: Template, workdir: pathlib.Path) -> list[str]:
    """Start a real server this way on a copy of the template store, and check it end to
    end: the list it gives, how it refuses what it does not offer, every listed tool
    called once with its minimal call, a clean exit, and, on a read-only server, a store
    left exactly as it was, down to every row and both files beside it. `start` is the
    `mcp` fixture's function."""
    db = template.copy(workdir)
    server = start(db, user=USER, env=combination.env())
    server.initialize()
    problems = _problems(server, Expected.of(combination), minimal_calls(template.ids),
                         refusals=True, run=True)
    code = server.close()
    if code != 0:
        problems.append(f"the server exited with code {code}; its stderr ends: "
                        f"{server.stderr_text()[-300:]!r}")
    if combination.read_only:
        after = store_dump(db)
        if after != template.dump:
            rows = sorted(set(after[0]) ^ set(template.dump[0]))[:3]
            problems.append(f"a read-only server changed the store; the first rows that "
                            f"differ are {rows}, and the files beside it match: "
                            f"{after[1:] == template.dump[1:]}")
    return problems
