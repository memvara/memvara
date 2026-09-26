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

import hashlib
import itertools
import pathlib
import shutil
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

from harness import stores
from harness.env import child_env
from harness.stdio import McpProcess, ToolResult
from memvara import Memvara
from memvara.server import tools as server_tools
from memvara.server.config import FEATURE_DEFAULTS, ServerConfig
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
    descriptions to `description`.

    Only a tool that takes the argument named `marker` has its `arguments` changed. Taking
    that argument is what shows that the feature applies to the tool.
    """

    feature: str
    arguments: tuple[str, ...]
    marker: str
    description: str


#: The two features that keep their arguments when off. `metadata_filters` rewrites the
#: filter arguments of the tools that take `filters` (config.py: it "decides whether
#: memory_search and memory_recall accept filters and filepath_prefix"); `expiry_erasure`
#: rewrites memory_remember's two expiry arguments. The texts are tools.py's own.
REDESCRIBED: tuple[Redescribed, ...] = (
    Redescribed("metadata_filters", ("filters", "filepath_prefix"), "filters",
                server_tools._FILTERS_OFF),
    Redescribed("expiry_erasure", server_tools._EXPIRY_ARGUMENTS, "expires_at",
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


@dataclass(frozen=True)
class Combination:
    """One way to start a server: the settings that are moved away from their defaults.

    Every feature in SWITCHES is on by default today, so moving one switches it off.
    Read-only mode and anchored mode are off by default, so moving one switches it on.
    """

    moved: frozenset[str]

    def __post_init__(self) -> None:
        unknown = self.moved - set(SWITCHES)
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
        found = {f"MEMVARA_FEATURE_{f.upper()}": "1" if self.feature_on(f) else "0"
                 for f in FEATURE_SWITCHES}
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
            raise LookupError(f"{tool.name} takes `anchored` with a schema that ANCHORED_ON "
                              "does not pair with a replacement; add the pair")
        schema = replacements[0]
    for rule in REDESCRIBED:
        if (not combination.feature_on(rule.feature) and rule.marker in tool.properties
                and name in rule.arguments):
            schema = {**schema, "description": rule.description}
    return schema


def served(combination: Combination) -> list[dict[str, Any]]:
    """The `tools/list` answer a server started with `combination` must give, in order."""
    gone = removed(combination)
    answer = []
    for tool in TOOLS:
        if unavailable(tool, combination) is not None:
            continue
        properties = {name: dict(_argument(tool, name, schema, combination))
                      for name, schema in tool.properties.items() if name not in gone}
        answer.append({
            "name": tool.name,
            "description": tool.description,
            "inputSchema": {"type": "object", "properties": properties,
                            "required": [r for r in tool.required if r in properties],
                            "additionalProperties": False},
            "annotations": {"readOnlyHint": not tool.writes,
                            "destructiveHint": tool.destructive,
                            "openWorldHint": False},
        })
    return answer


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

#: The tools that take a metadata filter, and the value each filter argument is sent with.
#: Each argument is sent on its own call, because metadata_filters governs both.
FILTERING: tuple[str, ...] = tuple(tool.name for tool in TOOLS if "filters" in tool.properties)
FILTER_PROBES: Mapping[str, Any] = {"filters": {"team": "support"},
                                    "filepath_prefix": "policies/"}


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
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        text = "".join(str(block.get("text", "")) for block in result.get("content", []))
        return ToolResult(text=text, is_error=bool(result.get("isError")), raw=result)


def base_env(home: pathlib.Path) -> dict[str, str]:
    """The MEMVARA_ variables every child process of the suite starts with
    (`harness.env.child_env`), so that a server built in this process is configured like
    one started over the pipe. `home` is only what child_env asks for."""
    return {name: value for name, value in child_env(home).items()
            if name.startswith("MEMVARA_")}


def in_process(memory: Memvara, combination: Combination,
               base: Mapping[str, str]) -> InProcess:
    """A server over `memory`, configured the way `python -m memvara.server` configures
    itself: `ServerConfig.from_env` reads `base`, then the combination's variables.

    One engine is shared by every server, which keeps 2,048 of them cheap. A server with
    metadata filters off switches them off on the engine it is given, so they are
    switched back on before each server is built.
    """
    config = ServerConfig.from_env({**base, "MEMVARA_DB": ":memory:", "MEMVARA_USER": USER,
                                    **combination.env()})
    memory.metadata_filters = True
    return InProcess(MemvaraMCPServer(
        memory, read_only=config.read_only, anchored=config.anchored,
        features_off=config.features_off, **config.scope_kwargs))


# -- the checks ------------------------------------------------------------------------

def listing_problems(client: Client, combination: Combination) -> list[str]:
    """How the tool list a server gives differs from `served(combination)`."""
    got, wanted = client.list_tools(), served(combination)
    problems = []
    names, wanted_names = [t.get("name") for t in got], [t["name"] for t in wanted]
    if names != wanted_names:
        problems.append(f"lists {names}, and the oracle expects {wanted_names}")
    expected = {tool["name"]: tool for tool in wanted}
    for tool in got:
        if tool.get("name") in expected and tool != expected[tool["name"]]:
            problems.append(f"{tool['name']} differs in "
                            f"{_differences(tool, expected[tool['name']])}")
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


def refusal_problems(client: Client, combination: Combination,
                     calls: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """What is wrong with how a server refuses what it does not offer.

    A tool it does not list must be refused by name, with the reason: the feature and its
    variable for a switched-off tool, or read-only mode for a write tool. An argument a
    switch removed must be refused as unknown. On a server without metadata filters, a
    read that carries either filter argument must be refused, naming the switch. None of
    these calls reaches the store, because each is refused before anything runs.
    """
    problems = []
    for tool in TOOLS:
        reason = unavailable(tool, combination)
        if reason is None:
            continue
        needs = (f"MEMVARA_FEATURE_{str(tool.feature).upper()}=0" if reason == "feature"
                 else "this memory server is read-only")
        result = client.call(tool.name, **calls[tool.name])
        if not (result.is_error and result.text.startswith(f"{tool.name} is unavailable: ")
                and needs in result.text):
            problems.append(f"{tool.name} should be refused naming {needs!r}, and the "
                            f"server answered: {result.text[:200]!r}")
    gone = removed(combination)
    for tool in TOOLS:
        if unavailable(tool, combination) is not None:
            continue
        for argument in sorted(gone & set(tool.properties)):
            result = client.call(tool.name, **{**calls[tool.name], argument: PROBES[argument]})
            needs = f"{tool.name}: unknown argument(s) '{argument}'"
            if not (result.is_error and result.text.startswith(needs)):
                problems.append(f"{tool.name} should refuse {argument!r} as unknown, and the "
                                f"server answered: {result.text[:200]!r}")
    if not combination.feature_on("metadata_filters"):
        for name in FILTERING:
            for argument, value in FILTER_PROBES.items():
                result = client.call(name, **{**calls[name], argument: value})
                if not (result.is_error
                        and "MEMVARA_FEATURE_METADATA_FILTERS=0" in result.text):
                    problems.append(f"{name} should refuse {argument!r}, naming the switch, "
                                    f"and the server answered: {result.text[:200]!r}")
    return problems


def call_problems(client: Client, combination: Combination,
                  calls: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Each tool a server lists must run with its minimal call, and with metadata filters
    on, a read with each filter argument must run too."""
    problems = []
    for tool in TOOLS:
        if unavailable(tool, combination) is not None:
            continue
        result = client.call(tool.name, **calls[tool.name])
        if result.is_error:
            problems.append(f"{tool.name} with {dict(calls[tool.name])} failed: "
                            f"{result.text[:300]!r}")
    if combination.feature_on("metadata_filters"):
        for name in FILTERING:
            for argument, value in FILTER_PROBES.items():
                result = client.call(name, **{**calls[name], argument: value})
                if result.is_error:
                    problems.append(f"{name} with {argument!r} failed: "
                                    f"{result.text[:300]!r}")
    return problems


def in_process_failures(combinations: Sequence[Combination], base: Mapping[str, str], *,
                        refusals: bool = True) -> dict[Combination, list[str]]:
    """Each combination whose server, built in this process, lists anything other than the
    oracle predicts, with its problems. With `refusals`, what the server refuses is
    checked too; none of it reaches the store, so one engine serves every server."""
    memory = stores.memory()
    calls = minimal_calls(NOWHERE)
    found: dict[Combination, list[str]] = {}
    try:
        for combination in combinations:
            client = in_process(memory, combination, base)
            problems = listing_problems(client, combination)
            if refusals:
                problems += refusal_problems(client, combination, calls)
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


def nightly_runs() -> tuple[Combination, ...]:
    """The fold-over of `fast_runs`: each run with every setting reversed. It covers all
    eight combinations of every three settings on its own, and together with `fast_runs`
    it is an orthogonal array of strength 3, in which each combination of three settings
    appears exactly three times."""
    return tuple(Combination(frozenset(SWITCHES) - run.moved) for run in fast_runs())


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
        """Store four facts and a document for the minimal calls to name, and a door code
        that has expired by the time this returns."""
        directory.mkdir(parents=True, exist_ok=True)
        expires = datetime.now(timezone.utc) + EXPIRES_AFTER
        with stores.file(directory / STORE) as memory:
            scoped = memory.scope(user=USER)
            home = scoped.remember("user", "lives_in", "Oslo").added[0]
            work = scoped.remember("user", "works_at", "Contoso").added[0]
            hobby = scoped.remember("user", "likes", "chess").added[0]
            language = scoped.remember("user", "speaks", "Norwegian").added[0]
            scoped.remember("user", "door_code", "4417", expires_at=expires)
            document = scoped.add_document("Notes about the office move to Oslo.",
                                           custom_id="notes/office.md")
        time.sleep(max(0.0, (expires - datetime.now(timezone.utc)).total_seconds()) + 0.05)
        ids = Ids(linked_from=work.id, linked_to=home.id, forgettable=hobby.id,
                  endable=language.id, document=str(document.custom_id))
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
    calls = minimal_calls(template.ids)
    problems = (listing_problems(server, combination)
                + refusal_problems(server, combination, calls)
                + call_problems(server, combination, calls))
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
