"""Protocol and validator fuzzing: the real stdio server, and the validator that every
tool call passes through, fed input that a correct client would never send.

`docs/claude/testing.md` explains these tests. This folder holds only test files, so the
helpers they share live here:

* `request_line`, `notification_line` and `call_line` spell one JSON-RPC message as the
  line a client writes.
* `exchange` sends raw lines to a server and returns every message the server wrote in
  reply, which is how a test counts replies.
* `rows` reads everything a store file holds, and `changed` compares two readings, so a
  test can show that a refused call changed nothing.
* `seed` gives a store something to lose, and `shared_server` is one seeded server for
  every test in a module that imports it.
* `conforming`, `arguments`, `violating` and `broken_arguments` build Hypothesis
  strategies from a tool's input schema, and `json_values` draws any JSON value at all.

Importing this module has no side effects, because `--doctest-modules` imports it while
pytest collects.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import pathlib
import re
import sqlite3
from typing import Any, Collection, Iterator, Mapping, Sequence

import pytest
from hypothesis import strategies as st

from harness.stdio import McpProcess, McpProcessError

# -- lines on the wire -------------------------------------------------------------


def request_line(request_id: Any, method: str,
                 params: Mapping[str, Any] | None = None) -> str:
    """One JSON-RPC request, as the line a client writes.

    The id goes in exactly as given, so a test can send an id of any JSON type, null
    included. Python writes a NaN or an infinite float as the bare token `NaN` or
    `Infinity`, which is how a test sends one of those as an argument.
    """
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = dict(params)
    return json.dumps(message)


def notification_line(method: str, params: Mapping[str, Any] | None = None) -> str:
    """One JSON-RPC notification: a message with no id, which gets no reply."""
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = dict(params)
    return json.dumps(message)


def call_line(request_id: Any, tool: str, arguments: Any) -> str:
    """A tools/call request. `arguments` goes in as given, even when it is not an object."""
    return request_line(request_id, "tools/call", {"name": tool, "arguments": arguments})


_FENCES = itertools.count(1)


def exchange(server: McpProcess, *lines: str | bytes,
             timeout: float | None = None) -> list[dict[str, Any]]:
    """Send `lines` exactly as given, then a ping, and return every message the server
    wrote before it answered the ping.

    The ping works as a fence. The server handles its input one line at a time, in
    order, so everything it writes about `lines` arrives before the ping's reply. A
    request that got no reply, or two, therefore shows in the list this returns. The
    fence's own reply is checked as well, so every call also checks that the server
    still answers a ping after the input it was given.
    """
    for line in lines:
        server.send_raw(line)
    fence = f"fence-{next(_FENCES)}"
    server.send_raw(request_line(fence, "ping"))
    seen: list[dict[str, Any]] = []
    while True:
        message = server.recv(timeout=timeout)
        if message.get("id") == fence and "method" not in message:
            if message.get("result") != {}:
                raise McpProcessError(
                    f"the fence ping got {message!r} instead of an empty result"[:500])
            return seen
        seen.append(message)


def text_of(reply: Mapping[str, Any]) -> str:
    """The text of a tools/call reply, joined from its content blocks."""
    return "".join(str(block.get("text", "")) for block in reply["result"]["content"])


# -- the store -----------------------------------------------------------------------


def rows(db: str | pathlib.Path) -> dict[str, Any]:
    """Everything the store file `db` holds: the rows of every table, and a digest of its
    vector file, which lives beside it as `<db>.vecs`.

    A second, read-only connection reads the file while the server keeps it open, which
    SQLite allows because the store keeps a write-ahead log. Each table's rows are sorted
    by their repr, because a table has no order of its own. `changed` compares two
    readings.
    """
    path = pathlib.Path(db).resolve()
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        tables = [name for (name,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        held: dict[str, Any] = {
            table: sorted(repr(row) for row in connection.execute(f'SELECT * FROM "{table}"'))
            for table in tables}
    finally:
        connection.close()
    vectors = path.with_name(path.name + ".vecs")
    held[".vecs"] = (hashlib.sha256(vectors.read_bytes()).hexdigest()
                     if vectors.exists() else None)
    return held


def changed(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """The names of the tables, and of `.vecs`, whose contents differ between two readings
    taken by `rows`."""
    return sorted(name for name in set(before) | set(after)
                  if before.get(name) != after.get(name))


#: What `seed` stores: two facts, a turn and a document, so that a refused call has
#: something it could wrongly end, retire or erase.
SEED: tuple[tuple[str, dict[str, Any]], ...] = (
    ("memory_remember", {"predicate": "lives_in", "object": "Berlin"}),
    ("memory_remember", {"predicate": "likes", "object": "green tea"}),
    ("memory_add", {"text": "I work at Acme and I like hiking."}),
    ("memory_add_document", {"content": "Refunds are allowed for thirty days.",
                             "custom_id": "handbook", "metadata": {"team": "support"}}),
)


def seed(server: McpProcess) -> None:
    """Store `SEED` through `server`."""
    for tool, arguments in SEED:
        result = server.call(tool, **arguments)
        if result.is_error:
            raise AssertionError(f"seeding the store with {tool} failed: {result.text}")


@pytest.fixture(scope="module")
def shared_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[McpProcess]:
    """One seeded server for every test in a module that sends it only input it must
    survive. A test module imports this fixture to use it.

    Starting a server costs about a quarter of a second, and this workstream's whole fast
    tier has about twelve, so tests that each send a few lines share one server.
    """
    home = tmp_path_factory.mktemp("fuzz-home")
    server = McpProcess(tmp_path_factory.mktemp("fuzz-store") / "memory.db", home=home)
    try:
        server.initialize()
        seed(server)
        yield server
    finally:
        server.kill()


# -- strategies from a tool's input schema ------------------------------------------

#: The schema keywords the strategies below understand: the subset of JSON Schema that
#: `memvara/server/validate.py` lists, plus `description`. test_adv_fuzz_helpers.py fails
#: when a tool starts using a keyword outside it.
SCHEMA_KEYWORDS = frozenset({
    "type", "description", "default", "enum", "minimum", "maximum", "maxLength",
    "pattern", "items", "additionalProperties", "propertyNames"})

#: Text that a schema's `string` allows: any character except a surrogate, which is half
#: of a character and cannot be encoded.
TEXT = st.text(st.characters(exclude_categories=["Cs"]), max_size=40)

#: Text that may also hold lone surrogates, which a JSON string can carry as escapes.
ANY_TEXT = st.text(st.characters(exclude_categories=["Cs"]) | st.characters(categories=["Cs"]),
                   max_size=40)

#: A strategy for each JSON type, for drawing a value of the wrong type.
_OF_TYPE: dict[str, st.SearchStrategy[Any]] = {
    "string": TEXT,
    "integer": st.integers(),
    "number": st.floats(allow_nan=False, allow_infinity=False),
    "boolean": st.booleans(),
    "array": st.lists(st.integers(), max_size=2),
    "object": st.dictionaries(TEXT, st.integers(), max_size=2),
    "null": st.none(),
}


def _types(spec: Mapping[str, Any]) -> list[str]:
    kind = spec["type"]
    return list(kind) if isinstance(kind, list) else [kind]


def _keys(spec: Mapping[str, Any]) -> st.SearchStrategy[str]:
    names = spec.get("propertyNames")
    return conforming({"type": "string", **names}) if names else TEXT


def conforming(spec: Mapping[str, Any]) -> st.SearchStrategy[Any]:
    """Values that `spec` allows, which the validator must accept and pass on unchanged.

    `number` never draws NaN or an infinity, because JSON has no such numbers.
    """
    kinds = _types(spec)
    if len(kinds) > 1:
        return st.one_of([conforming({**spec, "type": kind}) for kind in kinds])
    kind = kinds[0]
    if "enum" in spec:
        return st.sampled_from(list(spec["enum"]))
    if kind == "string":
        longest = spec.get("maxLength", 40)
        if "pattern" in spec:
            return st.from_regex(spec["pattern"], fullmatch=True).filter(
                lambda text: len(text) <= longest)
        return st.text(st.characters(exclude_categories=["Cs"]), max_size=min(longest, 40))
    if kind == "integer":
        return st.integers(spec.get("minimum"), spec.get("maximum"))
    if kind == "number":
        low, high = spec.get("minimum"), spec.get("maximum")
        numbers = st.floats(low, high, allow_nan=False, allow_infinity=False)
        first = None if low is None else math.ceil(low)
        last = None if high is None else math.floor(high)
        if first is not None and last is not None and first > last:
            return numbers      # a range such as 0.1 to 0.2 holds no whole number
        return numbers | st.integers(first, last)
    if kind == "boolean":
        return st.booleans()
    if kind == "array":
        return st.lists(conforming(spec["items"]), max_size=4)
    if kind == "object":
        return st.dictionaries(_keys(spec), conforming(spec["additionalProperties"]),
                               max_size=4)
    raise ValueError(f"no strategy for the schema type {kind!r}")


def violating(spec: Mapping[str, Any]) -> st.SearchStrategy[Any]:
    """Values that break one rule of `spec`, which the validator must refuse.

    A value of the wrong type breaks a rule, and so does NaN or an infinity where an
    integer goes. For a string: a lone surrogate, a value outside `enum`, one that
    `pattern` does not match, and one longer than `maxLength`. For a number: one outside
    `minimum` or `maximum`, the infinities included. An array breaks its schema through
    one bad item, and an object through one bad value or one bad key. When the schema
    allows several types, each allowed type's own rules are broken as well, and a near
    miss is left out when another allowed type accepts it, such as "false" where a string
    is also allowed.
    """
    kinds = _types(spec)
    allowed = set(kinds) | ({"integer"} if "number" in kinds else set())
    options = [st.one_of([strategy for kind, strategy in _OF_TYPE.items()
                          if kind not in allowed])]
    # The near misses a model actually sends, which validate.py's docstring lists: a
    # boolean or a number written as a string. They get their own option so that they
    # are drawn often, not as one wrong type among six.
    if kinds == ["boolean"]:
        options.append(st.sampled_from(["false", "true", "False", "0", "1", 0, 1]))
    if ("integer" in kinds or "number" in kinds) and "string" not in kinds:
        options.append(st.integers(-10, 100).map(str))
    if "integer" in kinds and "number" not in kinds:
        options.append(st.sampled_from([math.nan, math.inf, -math.inf]))
    if "string" in kinds:
        options.append(st.tuples(TEXT, st.characters(categories=["Cs"]), TEXT).map("".join))
        if "enum" in spec:
            options.append(TEXT.filter(lambda text: text not in spec["enum"]))
        if "pattern" in spec:
            options.append(TEXT.filter(lambda text: re.search(spec["pattern"], text) is None))
        if "maxLength" in spec:
            longest = spec["maxLength"]
            options.append(st.text(st.characters(exclude_categories=["Cs"]),
                                   min_size=longest + 1, max_size=longest + 8))
    low, high = spec.get("minimum"), spec.get("maximum")
    if "integer" in kinds:
        if low is not None:
            options.append(st.integers(max_value=low - 1))
        if high is not None:
            options.append(st.integers(min_value=high + 1))
    if "number" in kinds:
        if low is not None:
            options.append(st.floats(max_value=low, exclude_max=True, allow_nan=False))
        if high is not None:
            options.append(st.floats(min_value=high, exclude_min=True, allow_nan=False))
    if "array" in kinds:
        options.append(st.tuples(st.lists(conforming(spec["items"]), max_size=3),
                                 violating(spec["items"]), st.integers(0, 3)).map(
            lambda drawn: [*drawn[0][:drawn[2]], drawn[1], *drawn[0][drawn[2]:]]))
    if "object" in kinds:
        good = st.dictionaries(_keys(spec), conforming(spec["additionalProperties"]),
                               max_size=3)
        options.append(st.tuples(good, TEXT, violating(spec["additionalProperties"])).map(
            lambda drawn: {**drawn[0], drawn[1]: drawn[2]}))
        if "propertyNames" in spec:
            bad_key = violating({"type": "string", **spec["propertyNames"]}).filter(
                lambda key: isinstance(key, str))
            options.append(st.tuples(good, bad_key,
                                     conforming(spec["additionalProperties"])).map(
                lambda drawn: {**drawn[0], drawn[1]: drawn[2]}))
    return st.one_of(options)


def arguments(properties: Mapping[str, Mapping[str, Any]], required: Sequence[str], *,
              leave_out: Collection[str] = ()) -> st.SearchStrategy[dict[str, Any]]:
    """Arguments that follow a tool's schema: every required one, and any of the others.

    An argument named in `leave_out` is never drawn. That is for an argument a test must
    not send, such as a URL the server would fetch. Naming a required argument there is
    refused, because a required argument is always drawn.
    """
    both = sorted(set(leave_out) & set(required))
    if both:
        raise ValueError(f"cannot leave out a required argument: {both}")
    return st.fixed_dictionaries(
        {name: conforming(properties[name]) for name in required},
        optional={name: conforming(spec) for name, spec in properties.items()
                  if name not in required and name not in leave_out})


def broken_arguments(tool: str, properties: Mapping[str, Mapping[str, Any]],
                     required: Sequence[str], *, leave_out: Collection[str] = (),
                     ) -> st.SearchStrategy[tuple[Any, str]]:
    """Arguments that break one rule of a tool's schema, with the text that the
    validator's refusal must start with.

    The rule broken is one of these: an argument's value (see `violating`), an argument
    name the schema does not declare, a required argument left out, or arguments that
    are not an object at all.
    """
    base = arguments(properties, required, leave_out=leave_out)

    def with_unknown(drawn: tuple[dict[str, Any], str, int]) -> tuple[Any, str]:
        good, key, value = drawn
        return {**good, key: value}, f"{tool}: unknown argument(s) {key!r}"

    ways: list[st.SearchStrategy[tuple[Any, str]]] = [
        st.tuples(base, TEXT.filter(lambda key: key not in properties),
                  st.integers()).map(with_unknown),
        st.one_of(_OF_TYPE["array"], _OF_TYPE["string"], _OF_TYPE["integer"],
                  _OF_TYPE["boolean"], _OF_TYPE["null"]).map(
            lambda value: (value, f"{tool}: arguments must be a JSON object")),
    ]
    if required:
        def without(drawn: tuple[dict[str, Any], str]) -> tuple[Any, str]:
            good, gone = drawn
            return ({key: value for key, value in good.items() if key != gone},
                    f"{tool}: missing required argument(s) {gone!r}.")

        ways.append(st.tuples(base, st.sampled_from(list(required))).map(without))
    others = st.one_of(ways)
    names = sorted(name for name in properties if name not in leave_out)
    if not names:
        return others

    def one_bad_value(name: str) -> st.SearchStrategy[tuple[Any, str]]:
        return st.tuples(base, violating(properties[name])).map(
            lambda drawn: ({**drawn[0], name: drawn[1]}, f"{tool}.{name}"))

    # Half of the draws break an argument's value, because checking values is most of
    # what the validator does. The other three ways share the other half.
    values = st.sampled_from(names).flatmap(one_bad_value)
    return st.booleans().flatmap(lambda pick_a_value: values if pick_a_value else others)


def json_values(*, max_leaves: int = 12) -> st.SearchStrategy[Any]:
    """Any JSON value. NaN and the infinities are included, because the server's JSON
    parser accepts them, and so are lone surrogates, which a JSON string can carry as
    escapes."""
    scalars = st.none() | st.booleans() | st.integers() | st.floats() | ANY_TEXT
    return st.recursive(
        scalars,
        lambda inner: st.lists(inner, max_size=4) | st.dictionaries(ANY_TEXT, inner,
                                                                    max_size=4),
        max_leaves=max_leaves)
