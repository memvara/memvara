# Protocol and validator fuzzing (A3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed the real stdio MCP server, and the validator every tool call passes through, input that a correct client would never send, and show that every request with an id still gets exactly one reply with that id, that a refused call changes nothing in the store, and that the server still answers `ping` afterwards.

**Architecture:**
- Every test talks to a real `python -m memvara.server` through `harness.stdio.McpProcess`, except the Hypothesis properties that call `memvara.server.validate.validate` in the test process.
- `exchange(server, *lines)` sends raw lines and then a ping, and returns every message the server wrote before the ping's reply. The server handles one line at a time and in order, so that list counts the replies, and the ping's own reply checks that the server carries on.
- `rows(db)` reads every table of the store through a second, read-only SQLite connection, plus a digest of the `.vecs` vector file, while the server keeps running. A reading before and after a refused call must be equal.
- Hypothesis strategies are built from each tool's own input schema (`Tool.properties` in-process, `tools/list` over the pipe), so a new argument is fuzzed without anyone editing this suite.
- The folder may hold only `test_adv_*.py` files and `__init__.py`, so the shared helpers live in `tests/adversarial/fuzz/__init__.py`.

**Tech Stack:** Python 3.10–3.13, pytest, Hypothesis (the tier profiles in `tests/harness/tiers.py`), `sqlite3`, `harness.stdio.McpProcess`, `harness.stores`, the `mcp` fixture in `tests/adversarial/conftest.py`.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`, the A3 row of "Phase 2: The agent-facing deterministic tiers", and the "First targets" rows for the validator and the protocol.

## Global Constraints

- Everything runs offline with no API key. The `url` argument of `memory_add_document` is never sent, because the server would fetch it.
- Every child process gets its environment from `harness.env.child_env`. `McpProcess` already does this.
- The fast tier of this workstream takes about 12 seconds on a laptop. Anything slower goes in `tests/adversarial/fuzz/nightly/`.
- Files this plan may create or change: `tests/adversarial/fuzz/__init__.py`, `tests/adversarial/fuzz/test_adv_*.py`, `tests/adversarial/fuzz/nightly/__init__.py`, `tests/adversarial/fuzz/nightly/test_adv_*.py`, this plan, and a new section in `docs/claude/testing.md` just before its final line, which starts with `Next:`.
- Not edited: `tests/harness/known_bugs.py`, `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`. The maintainer's session pins confirmed bugs and updates the counts and the changelog.
- `tests/harness/checklist.py` does not exist on `origin/main`, so no test carries a `covers` mark.
- A test that meets documented behaviour asserts it and cites where it is documented. Here that is: batches are refused (`MemvaraMCPServer.handle_message`, and the design's list of documented behaviour), a null id is treated as a notification (`handle_message`), blank lines are skipped (`iter_messages` in `memvara/server/protocol.py`), and a deeply nested line gets a parse error (the #268 fix in `decode`).
- A bug found here is classified against the "In scope" section of `SECURITY.md` first. A security-class bug is described only in the final report, never in a committed file, and the test that shows it is left out. Any other bug is also reported with an offline reproduction, and its failing test is left out of the commits and kept under `local/a3-fuzz-findings/`, which git ignores.
- No skip is added, so the skip ledger needs no new rule.
- Commits name their files, and carry no trailer or footer naming the tool that wrote them. Write plainly: every sentence must be understood on its first reading.

## The three predictions

The design's "First targets" section predicts three failures for this workstream. Task 2 writes a test for each one before any other test in this plan, and runs it to see whether it fails for the predicted reason:

1. NaN passes the validator's bounds on `confidence`.
2. A refusal quotes the whole invalid value, with no length cap.
3. Invalid UTF-8 on standard input ends the server's line loop. On Windows, standard input is decoded with the locale's code page.

A prediction that is confirmed is reported, and its failing test is not committed. A prediction that is refuted keeps its test, which then pins the correct behaviour.

## Review Focus

1. **CRLF line endings, and blank or whitespace-only lines between requests.** A client on Windows, or one that pads its stream, must get replies only to its requests. Test in Task 3.
2. **Two requests written on one line.** A client that forgot a newline must get one parse error, and neither request may run. Test in Task 3.
3. **The deepest nesting that still parses.** Just inside the decoder's limit, the server must still build a reply, including a refusal that describes the whole nested value. The limit depends on the Python version, so the test searches for it. Test in Task 3.
4. **An integer past Python's 4,300-digit conversion limit**, as the id or as an argument. It must come back as a parse error with a null id, and the server must carry on. Test in Task 3.
5. **A write refused after it would have ended another value.** A NaN confidence on a slot that already holds a value, or together with `replaces`, must leave the old value live and write nothing. Test in Task 4.

---

### Task 1: The helpers, and proof that each catches its fault

**Files:**
- Create: `tests/adversarial/fuzz/__init__.py`
- Create: `tests/adversarial/fuzz/test_adv_fuzz_helpers.py`
- Modify: `docs/claude/testing.md` (a new section before the final `Next:` line)

**Interfaces:**
- Produces, in `tests/adversarial/fuzz/__init__.py`:
  - `request_line(request_id: Any, method: str, params: Mapping[str, Any] | None = None) -> str`
  - `notification_line(method: str, params: Mapping[str, Any] | None = None) -> str`
  - `call_line(request_id: Any, tool: str, arguments: Any) -> str`
  - `exchange(server: McpProcess, *lines: str | bytes, timeout: float | None = None) -> list[dict[str, Any]]`
  - `text_of(reply: Mapping[str, Any]) -> str`
  - `rows(db: str | pathlib.Path) -> dict[str, Any]` and `changed(before, after) -> list[str]`
  - `SEED`, `seed(server: McpProcess) -> None`, and the module-scoped fixture `shared_server`
  - `SCHEMA_KEYWORDS: frozenset[str]`, `TEXT`, `ANY_TEXT`
  - `conforming(spec) -> SearchStrategy[Any]`, `violating(spec) -> SearchStrategy[Any]`
  - `arguments(properties, required, *, leave_out=()) -> SearchStrategy[dict[str, Any]]`
  - `broken_arguments(tool, properties, required, *, leave_out=()) -> SearchStrategy[tuple[Any, str]]`: the arguments, and the text the refusal must start with
  - `json_values(*, max_leaves: int = 12) -> SearchStrategy[Any]`

- [ ] **Step 1: Write the failing tests**

```python
"""The helpers in this folder catch what they exist to catch.

A helper that cannot see a fault makes every test built on it pass for the wrong reason.
So each helper is shown catching its fault here, without a server wherever one is not
needed.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from harness import stores
from harness.stdio import McpProcessError
from memvara.server.tools import TOOLS

from . import SCHEMA_KEYWORDS, changed, exchange, request_line, rows


class Scripted:
    """A stand-in for McpProcess that answers each line with the replies a test gave it,
    and answers the fence ping that `exchange` sends last."""

    def __init__(self, replies: dict[str, list[dict[str, Any]]],
                 fence_result: Any = None) -> None:
        self.replies = replies
        self.fence_result = {} if fence_result is None else fence_result
        self.waiting: list[dict[str, Any]] = []

    def send_raw(self, data: str | bytes) -> None:
        line = data.decode() if isinstance(data, bytes) else data
        message = json.loads(line)
        if str(message.get("id", "")).startswith("fence-"):
            self.waiting.append({"jsonrpc": "2.0", "id": message["id"],
                                 "result": self.fence_result})
        else:
            self.waiting.extend(self.replies.get(line, []))

    def recv(self, timeout: float | None = None) -> dict[str, Any]:
        return self.waiting.pop(0)


def test_exchange_returns_both_replies_when_one_request_gets_two() -> None:
    line = request_line(1, "ping")
    twice = [{"jsonrpc": "2.0", "id": 1, "result": {}}] * 2
    assert exchange(Scripted({line: twice}), line) == twice  # type: ignore[arg-type]


def test_exchange_returns_nothing_for_a_request_that_got_no_reply() -> None:
    assert exchange(Scripted({}), request_line(1, "ping")) == []  # type: ignore[arg-type]


def test_exchange_fails_when_the_fence_ping_is_not_answered_with_an_empty_result() -> None:
    with pytest.raises(McpProcessError, match="fence ping"):
        exchange(Scripted({}, fence_result={"surprise": 1}),  # type: ignore[arg-type]
                 request_line(1, "ping"))


def test_rows_sees_a_write_in_the_tables_and_in_the_vector_file(
        tmp_path: pathlib.Path) -> None:
    db = tmp_path / "memory.db"
    memory = stores.file(db)
    try:
        user = memory.scope(user="tester")
        user.remember("user", "lives_in", "Berlin")
        before = rows(db)
        assert changed(before, rows(db)) == []
        user.remember("user", "likes", "green tea")
        after = rows(db)
    finally:
        memory.close()
    assert {"claims", "embeddings", ".vecs"} <= set(changed(before, after))


def test_every_schema_keyword_a_tool_uses_is_one_the_strategies_understand() -> None:
    """A keyword the strategies did not know would be ignored while values are drawn, so
    the fuzzer would send values the schema forbids as if they were allowed, and never
    try the ones it forbids."""
    used: set[str] = set()

    def walk(spec: dict[str, Any]) -> None:
        used.update(spec)
        for key in ("items", "additionalProperties", "propertyNames"):
            if isinstance(spec.get(key), dict):
                walk(spec[key])

    for tool in TOOLS:
        for spec in tool.properties.values():
            walk(spec)
    assert used <= SCHEMA_KEYWORDS, sorted(used - SCHEMA_KEYWORDS)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `PYTHONPATH=$PWD TMPDIR=/private/tmp/a3-fuzz-tmp python -m pytest -q -p no:cacheprovider tests/adversarial/fuzz/test_adv_fuzz_helpers.py`
Expected: FAIL at collection, with an ImportError for `SCHEMA_KEYWORDS`, because `tests/adversarial/fuzz/__init__.py` does not define the helpers yet.

- [ ] **Step 3: Write the helpers**

`tests/adversarial/fuzz/__init__.py`:

```python
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
TEXT = st.text(st.characters(exclude_categories=("Cs",)), max_size=40)

#: Text that may also hold lone surrogates, which a JSON string can carry as escapes.
ANY_TEXT = st.text(st.characters(exclude_categories=("Cs",)) | st.characters(categories=("Cs",)),
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
        return st.text(st.characters(exclude_categories=("Cs",)), max_size=min(longest, 40))
    if kind == "integer":
        return st.integers(spec.get("minimum"), spec.get("maximum"))
    if kind == "number":
        low, high = spec.get("minimum"), spec.get("maximum")
        whole = st.integers(None if low is None else math.ceil(low),
                            None if high is None else math.floor(high))
        return st.floats(low, high, allow_nan=False, allow_infinity=False) | whole
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
    one bad item, and an object through one bad value or one bad key.
    """
    kinds = _types(spec)
    allowed = set(kinds) | ({"integer"} if "number" in kinds else set())
    options = [st.one_of([strategy for kind, strategy in _OF_TYPE.items()
                          if kind not in allowed])]
    if "integer" in kinds and "number" not in kinds:
        options.append(st.sampled_from([math.nan, math.inf, -math.inf]))
    if kinds == ["string"]:
        options.append(st.tuples(TEXT, st.characters(categories=("Cs",)), TEXT).map("".join))
        if "enum" in spec:
            options.append(TEXT.filter(lambda text: text not in spec["enum"]))
        if "pattern" in spec:
            options.append(TEXT.filter(lambda text: re.search(spec["pattern"], text) is None))
        if "maxLength" in spec:
            longest = spec["maxLength"]
            options.append(st.text(st.characters(exclude_categories=("Cs",)),
                                   min_size=longest + 1, max_size=longest + 8))
    low, high = spec.get("minimum"), spec.get("maximum")
    if kinds == ["integer"]:
        if low is not None:
            options.append(st.integers(max_value=low - 1))
        if high is not None:
            options.append(st.integers(min_value=high + 1))
    if kinds == ["number"]:
        if low is not None:
            options.append(st.floats(max_value=low, exclude_max=True, allow_nan=False))
        if high is not None:
            options.append(st.floats(min_value=high, exclude_min=True, allow_nan=False))
    if kinds == ["array"]:
        options.append(st.tuples(st.lists(conforming(spec["items"]), max_size=3),
                                 violating(spec["items"]), st.integers(0, 3)).map(
            lambda drawn: [*drawn[0][:drawn[2]], drawn[1], *drawn[0][drawn[2]:]]))
    if kinds == ["object"]:
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
    not send, such as a URL the server would fetch.
    """
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
    names = sorted(name for name in properties if name not in leave_out)
    if names:
        def one_bad_value(name: str) -> st.SearchStrategy[tuple[Any, str]]:
            return st.tuples(base, violating(properties[name])).map(
                lambda drawn: ({**drawn[0], name: drawn[1]}, f"{tool}.{name}"))

        ways.append(st.sampled_from(names).flatmap(one_bad_value))
    if required:
        def without(drawn: tuple[dict[str, Any], str]) -> tuple[Any, str]:
            good, gone = drawn
            return ({key: value for key, value in good.items() if key != gone},
                    f"{tool}: missing required argument(s) {gone!r}.")

        ways.append(st.tuples(base, st.sampled_from(list(required))).map(without))
    return st.one_of(ways)


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
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `PYTHONPATH=$PWD TMPDIR=/private/tmp/a3-fuzz-tmp python -m pytest -q -p no:cacheprovider tests/adversarial/fuzz/test_adv_fuzz_helpers.py`
Expected: `5 passed`.

- [ ] **Step 5: Start the section in `docs/claude/testing.md`**

Insert before the final line, which starts with `Next:`:

```markdown
## Protocol and validator fuzzing

`tests/adversarial/fuzz/` sends the MCP server input that a correct client would never send. After each input it checks three things: every request that has an id gets exactly one reply with that id, a call the server refuses changes nothing in the store, and the server still answers a ping. The plan is `docs/superpowers/plans/2026-09-26-adversarial-fuzz.md`.

- **How replies are counted.** `exchange(server, *lines)` sends the lines exactly as given, then a ping, and returns every message the server wrote before it answered the ping. The server handles one line at a time and in order, so a request that got no reply, or two, shows in that list. The ping's own reply is checked too, which is how every test checks that the server carries on.
- **How "changed nothing" is checked.** `rows(db)` reads every row of every table through a second, read-only SQLite connection, and a digest of the `.vecs` vector file, while the server keeps running. A test takes one reading before a refused call and one after, and `changed` names any table that differs.
- **Where the helpers live.** This folder may hold only test files, so the helpers are in `tests/adversarial/fuzz/__init__.py`. `test_adv_fuzz_helpers.py` shows each one catching the fault it exists to catch.
```

- [ ] **Step 6: Commit**

```bash
git add tests/adversarial/fuzz/__init__.py tests/adversarial/fuzz/test_adv_fuzz_helpers.py docs/claude/testing.md
git commit -m "Add the fuzzing helpers that count replies and read every row of a store"
```

---

### Task 2: The three predictions, each tested before anything else

**Files:**
- Test, written first in the file it belongs to, and moved to `local/a3-fuzz-findings/` if it fails: the protocol predictions in `tests/adversarial/fuzz/test_adv_wire.py`, the validator predictions in `tests/adversarial/fuzz/test_adv_arguments.py`

**Interfaces:**
- Consumes: `call_line`, `exchange`, `text_of`, `rows`, `changed`, `shared_server` from Task 1; the `mcp` fixture.

- [ ] **Step 1: Write the protocol prediction tests at the top of `test_adv_wire.py`**

```python
from __future__ import annotations

import json
from typing import Callable

import pytest

from harness.stdio import McpProcess

from . import call_line, exchange, text_of

Start = Callable[..., McpProcess]

LOCALES = {"a UTF-8 locale": {"LC_ALL": "en_US.UTF-8"},
           "strict UTF-8 input": {"PYTHONIOENCODING": "utf-8:strict"}}


@pytest.mark.parametrize("env", list(LOCALES.values()), ids=list(LOCALES))
def test_invalid_utf8_on_standard_input_gets_a_parse_error_and_the_server_carries_on(
        mcp: Start, env: dict[str, str]) -> None:
    """A byte that is not UTF-8 is a malformed line, like any other. It must get one
    reply, and the server must answer the next request."""
    server = mcp(env=env)
    server.initialize()
    replies = exchange(server, b'{"jsonrpc":"2.0","id":7,"method":"ping","params":{"x":"\xff"}}')
    assert len(replies) == 1, replies


def test_the_server_reads_its_input_as_utf8_whatever_the_locale(mcp: Start) -> None:
    """MCP's stdio transport is UTF-8. A Node client writes "Zürich" as UTF-8 bytes, and
    the server must store "Zürich" even when its locale names another encoding, as the
    ANSI code page does on Windows."""
    server = mcp(env={"LC_ALL": "en_US.ISO8859-1"})
    server.initialize()
    line = call_line(3, "memory_remember", {"predicate": "lives_in", "object": "Zürich"})
    replies = exchange(server, json.dumps(json.loads(line), ensure_ascii=False).encode("utf-8"))
    assert "Zürich" in text_of(replies[0])
```

- [ ] **Step 2: Write the validator prediction tests at the top of `test_adv_arguments.py`**

```python
from __future__ import annotations

import math
import re
from typing import Any

import pytest

from harness.stdio import McpProcess
from memvara.server.tools import TOOLS, Tool
from memvara.server.validate import ToolError, validate

from . import call_line, changed, exchange, rows, text_of
from . import shared_server  # noqa: F401 - a fixture; importing it lets pytest find it here

NUMBERS = [(tool, name) for tool in TOOLS for name, spec in tool.properties.items()
           if spec["type"] == "number"]


def _minimal(tool: Tool) -> dict[str, Any]:
    return {name: "tea" for name in tool.required}


def _label(pair: tuple[Tool, str]) -> str:
    return f"{pair[0].name}.{pair[1]}"


def _refused(server: McpProcess, tool: str, arguments: Any) -> str:
    before = rows(server.db)
    replies = exchange(server, call_line(1, tool, arguments))
    assert len(replies) == 1 and replies[0].get("id") == 1, replies
    assert replies[0]["result"]["isError"] is True, replies[0]
    assert changed(before, rows(server.db)) == []
    return text_of(replies[0])


@pytest.mark.parametrize("pair", NUMBERS, ids=_label)
def test_nan_is_refused_by_the_bounds_of_every_number_argument(
        pair: tuple[Tool, str]) -> None:
    """NaN is not between 0 and 1. Every comparison with it is false, so a bound written
    as `value < low` or `value > high` lets it through."""
    tool, name = pair
    with pytest.raises(ToolError, match=re.escape(f"{tool.name}.{name}")):
        validate(tool.properties, tool.required, {**_minimal(tool), name: math.nan},
                 tool=tool.name)


def test_a_refusal_quotes_at_most_a_short_part_of_the_value_it_refuses(
        shared_server: McpProcess) -> None:
    """A refusal is read by a model, so a value of 100,000 characters must not come back
    whole inside it. `safe_detail` caps a failure's detail at 300 characters."""
    text = _refused(shared_server, "memory_recall", {"query": "tea", "k": "y" * 100_000})
    assert len(text) < 2_000, len(text)
```

Task 4 replaces `_minimal` with a version that gives each required argument a value its tool accepts.

- [ ] **Step 3: Run them, and read why each fails**

Run: `PYTHONPATH=$PWD TMPDIR=/private/tmp/a3-fuzz-tmp python -m pytest -q -p no:cacheprovider tests/adversarial/fuzz/test_adv_wire.py tests/adversarial/fuzz/test_adv_arguments.py`
Expected, if the predictions hold:
- the two invalid-UTF-8 tests fail with `McpProcessError` quoting a `UnicodeDecodeError` from `iter_messages` and exit code 1;
- the locale test fails because the stored value reads "ZÃ¼rich";
- the NaN test fails with "DID NOT RAISE" for `memory_recall.min_score`, `memory_search.min_score` and `memory_remember.confidence`;
- the length test fails with a reply of more than 100,000 characters.

- [ ] **Step 4: Classify, record and move**

Classify each failure against the "In scope" section of `SECURITY.md`. Move the failing tests, with the imports they need, to `local/a3-fuzz-findings/test_adv_stdin_encoding.py` and `local/a3-fuzz-findings/test_adv_validator_predictions.py`, which git ignores, and write each reproduction into the final report. A prediction that is refuted keeps its test in the file it was written in. No commit in this task: the files it wrote to are committed in Tasks 3 and 4, without the failing tests.

### Task 3: The wire: pipelining, ids, batches, framing and nesting

**Files:**
- Create: `tests/adversarial/fuzz/test_adv_wire.py` (it keeps the protocol prediction tests of Task 2 only if they passed)
- Modify: `docs/claude/testing.md` (one bullet in the section)

**Interfaces:**
- Consumes: `request_line`, `notification_line`, `call_line`, `exchange`, `text_of`, `rows`, `changed`, `shared_server` from Task 1; the `mcp` fixture.

- [ ] **Step 1: Write the tests**

```python
"""The stdio server's framing and request ids, over the real pipe.

Each test writes raw lines to a real `python -m memvara.server` and counts what comes
back with `exchange`, which ends every send with a ping. So every test also checks that
the server still answers after the input it was given.
"""

from __future__ import annotations

import collections
import json
import random
import sqlite3
import sys
from typing import Any, Callable

import pytest

from harness.stdio import McpProcess

from . import call_line, changed, exchange, notification_line, request_line, rows, text_of
from . import shared_server  # noqa: F401 - a fixture; importing it lets pytest find it here

Start = Callable[..., McpProcess]

#: The one reply to any line that is JSON but not a single JSON-RPC request object.
NOT_ONE_OBJECT = {"jsonrpc": "2.0", "id": None, "error": {
    "code": -32600, "message": "expected a single JSON-RPC request object per line"}}

#: Deeper than any Python version's JSON decoder allows.
DEEP = 100_000


def _same(first: Any, second: Any) -> bool:
    """Equal as JSON, so that true is not 1 and 1.0 is not 1."""
    return json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def _nested(depth: int) -> str:
    return "[" * depth + "]" * depth


def _stored_objects(server: McpProcess, predicate: str) -> list[str]:
    connection = sqlite3.connect(f"{server.db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return sorted(value for (value,) in connection.execute(
            "SELECT object FROM claims WHERE predicate = ?", (predicate,)))
    finally:
        connection.close()


# -- many requests at once ------------------------------------------------------------

def test_500_pipelined_requests_with_shuffled_ids_each_get_exactly_one_reply(
        mcp: Start) -> None:
    """A client may write many requests before it reads any reply. Each request with an
    id gets exactly one reply with that id, whatever order the ids come in and whatever
    the request asks for. A notification between them gets no reply, and every write in
    the stream is stored exactly once."""
    server = mcp()
    server.initialize()
    rng = random.Random(20260926)
    ids: list[Any] = [*rng.sample(range(1, 2**53), 350), *(f"r{n}" for n in range(150))]
    rng.shuffle(ids)
    lines, kinds, written = [], {}, []
    for n, request_id in enumerate(ids):
        kind = n % 5
        kinds[request_id] = kind
        if kind == 0:
            lines.append(request_line(request_id, "ping"))
        elif kind == 1:
            written.append(f"pipelined value {n}")
            lines.append(call_line(request_id, "memory_remember",
                                   {"predicate": "likes", "object": written[-1]}))
        elif kind == 2:
            lines.append(call_line(request_id, "memory_search",
                                   {"query": f"value {n}", "k": 3}))
        elif kind == 3:
            lines.append(call_line(request_id, "memory_recall", {"query": "tea", "k": "8"}))
        else:
            lines.append(request_line(request_id, "no/such/method"))
        if n % 7 == 0:
            lines.append(notification_line("notifications/initialized"))
    replies = exchange(server, "\n".join(lines))
    assert collections.Counter(reply.get("id") for reply in replies) == \
        collections.Counter(ids)
    for reply in replies:
        kind = kinds[reply["id"]]
        if kind == 4:
            assert reply["error"]["code"] == -32601, reply
        else:
            assert reply["result"].get("isError", False) is (kind == 3), reply
    assert _stored_objects(server, "likes") == sorted(written)


# -- ids ------------------------------------------------------------------------------

#: An id of every JSON type. JSON-RPC allows a string or a number, and MCP narrows that
#: to a string or an integer, but the server does not police the type: it answers with
#: whatever id the request carried, which lets any client match its reply.
ODD_IDS: dict[str, Any] = {
    "a float": 1.5,
    "true": True,
    "false": False,
    "zero": 0,
    "a negative integer": -7,
    "an object": {"a": [1, None]},
    "an array": [1, "two"],
    "an integer past 64 bits": 10**30,
    "an integer of 4000 digits": int("9" * 4000),
    "an empty string": "",
    "a 10000-character string": "x" * 10_000,
    "a string outside ASCII": "é\U0001f600",
}


@pytest.mark.parametrize("request_id", list(ODD_IDS.values()), ids=list(ODD_IDS))
def test_an_id_of_any_json_type_comes_back_unchanged_on_exactly_one_reply(
        shared_server: McpProcess, request_id: Any) -> None:
    replies = exchange(shared_server, request_line(request_id, "ping"))
    assert len(replies) == 1, replies
    assert _same(replies[0].get("id"), request_id)
    assert replies[0].get("result") == {}


def test_a_null_id_is_treated_as_a_notification_and_gets_no_reply(
        shared_server: McpProcess) -> None:
    """`MemvaraMCPServer.handle_message` treats a missing id as a notification, and an
    explicit null one too, because a reply addressed to null could not be matched to
    anything. So a ping with a null id gets no reply."""
    assert exchange(shared_server, request_line(None, "ping")) == []


@pytest.mark.parametrize("method", ["notifications/initialized", "notifications/cancelled",
                                    "ping", "no/such/method"])
def test_a_notification_gets_no_reply_whatever_its_method(
        shared_server: McpProcess, method: str) -> None:
    """JSON-RPC forbids answering a notification, and the server ignores one it does not
    know rather than failing on it (`MemvaraMCPServer._dispatch`)."""
    assert exchange(shared_server, notification_line(method)) == []


# -- lines that are not one request ----------------------------------------------------

BATCHES: dict[str, list[Any]] = {
    "requests": [json.loads(request_line(1, "ping")),
                 json.loads(call_line(2, "memory_remember",
                                      {"predicate": "likes", "object": "batched"})),
                 json.loads(call_line(3, "memory_forget", {"predicate": "lives_in"}))],
    "empty": [],
    "notifications": [json.loads(notification_line("notifications/initialized"))],
    "nested": [[json.loads(request_line(4, "ping"))]],
}


@pytest.mark.parametrize("batch", list(BATCHES.values()), ids=list(BATCHES))
def test_a_batch_is_refused_whole_and_nothing_in_it_runs(
        shared_server: McpProcess, batch: list[Any]) -> None:
    """MCP removed JSON-RPC batches in its 2025-06-18 revision, and the server refuses a
    batch outright rather than half-implementing it (`MemvaraMCPServer.handle_message`;
    the design's list of documented behaviour says "Batches are refused"). So the line
    gets one Invalid Request error with a null id, no request inside it is answered, and
    none of them runs: the write and the retirement in the first batch change nothing."""
    before = rows(shared_server.db)
    assert exchange(shared_server, json.dumps(batch)) == [NOT_ONE_OBJECT]
    assert changed(before, rows(shared_server.db)) == []


@pytest.mark.parametrize("line", ["1", "3.5", '"ping"', "true", "null"])
def test_a_line_that_is_json_but_not_an_object_is_refused_as_an_invalid_request(
        shared_server: McpProcess, line: str) -> None:
    assert exchange(shared_server, line) == [NOT_ONE_OBJECT]


@pytest.mark.parametrize("gap", ["", " "], ids=["touching", "a space apart"])
def test_two_requests_on_one_line_get_one_parse_error_and_neither_runs(
        shared_server: McpProcess, gap: str) -> None:
    """The newline is what frames a message, so two requests on one line are one line
    that does not parse. It gets one parse error with a null id, because no id can be
    read from a line that did not parse, and neither write runs."""
    before = rows(shared_server.db)
    line = (call_line(1, "memory_remember", {"predicate": "likes", "object": "first"})
            + gap + call_line(2, "memory_remember", {"predicate": "likes", "object": "second"}))
    replies = exchange(shared_server, line)
    assert len(replies) == 1, replies
    assert replies[0]["id"] is None and replies[0]["error"]["code"] == -32700
    assert changed(before, rows(shared_server.db)) == []


def test_blank_lines_and_crlf_line_endings_are_framing_and_get_no_reply(
        shared_server: McpProcess) -> None:
    """`iter_messages` in `memvara/server/protocol.py` skips blank lines and strips the
    whitespace around a message, so a client that ends its lines with CRLF, or pads
    between messages, is not failed for it."""
    replies = exchange(shared_server, "", "   ", "\t", "\r", request_line(7, "ping") + "\r")
    assert replies == [{"jsonrpc": "2.0", "id": 7, "result": {}}]


# -- nesting ----------------------------------------------------------------------------

PLACES: dict[str, Callable[[int], str]] = {
    "params": lambda depth: ('{"jsonrpc":"2.0","id":5,"method":"ping","params":{"x":'
                             + _nested(depth) + "}}"),
    "an argument": lambda depth: (
        '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"memory_remember",'
        '"arguments":{"predicate":"likes","object":' + _nested(depth) + "}}}"),
    "the id": lambda depth: '{"jsonrpc":"2.0","id":' + _nested(depth) + ',"method":"ping"}',
    "the top level": _nested,
    "objects": lambda depth: '{"a":' * depth + "1" + "}" * depth,
}


@pytest.mark.parametrize("place", list(PLACES))
def test_a_line_nested_too_deeply_gets_one_parse_error_and_nothing_runs(
        shared_server: McpProcess, place: str) -> None:
    """The fix for #268: Python's JSON decoder raises RecursionError, not ValueError, on
    nesting deeper than the interpreter allows, and `decode` in
    `memvara/server/protocol.py` turns it into a parse error. Its id is null, because no
    id can be read from a line that did not parse. Wherever the nesting sits, the server
    answers once, runs nothing and carries on."""
    before = rows(shared_server.db)
    replies = exchange(shared_server, PLACES[place](DEEP))
    assert len(replies) == 1, replies
    assert replies[0]["id"] is None
    assert replies[0]["error"]["code"] == -32700
    assert "nested too deeply" in replies[0]["error"]["message"]
    assert changed(before, rows(shared_server.db)) == []


def test_the_deepest_nesting_that_still_parses_is_answered_and_the_server_carries_on(
        shared_server: McpProcess) -> None:
    """Just inside the decoder's limit, a line parses, and the server must still build a
    reply from it: here, a refusal that describes the whole nested value it was sent
    where an integer goes. That reply is built by recursion too, so this is the depth at
    which a line could parse and then fail on the way out. The limit depends on the
    Python version, so the test searches for the deepest nesting that is not a parse
    error, and checks it and the three depths just inside it."""
    def reply_to(depth: int) -> dict[str, Any]:
        line = ('{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":'
                '"memory_recall","arguments":{"query":"tea","k":' + _nested(depth) + "}}}")
        replies = exchange(shared_server, line)
        assert len(replies) == 1, replies
        return replies[0]

    low, high = 1, DEEP
    assert "error" not in reply_to(low) and "error" in reply_to(high)
    while high - low > 1:
        middle = (low + high) // 2
        if "error" in reply_to(middle):
            high = middle
        else:
            low = middle
    for depth in range(max(1, low - 3), low + 1):
        reply = reply_to(depth)
        assert reply["id"] == 6 and reply["result"]["isError"] is True, reply
        assert text_of(reply).startswith("memory_recall.k must be an integer, got an array")


def test_an_integer_past_pythons_digit_limit_gets_a_parse_error_and_the_server_carries_on(
        shared_server: McpProcess) -> None:
    """Python 3.11, and 3.10 from 3.10.7, refuse to parse an integer of more than 4,300
    digits, which guards against a conversion that takes quadratic time. The server
    reports that as a parse error with a null id, whether the integer is the id or an
    argument. An interpreter without the limit parses it, and then the ping is answered
    as usual."""
    huge = "9" * 5000
    limited = hasattr(sys, "get_int_max_str_digits")
    for line in ('{"jsonrpc":"2.0","id":' + huge + ',"method":"ping"}',
                 '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":'
                 '"memory_recall","arguments":{"query":"tea","budget":' + huge + "}}}"):
        replies = exchange(shared_server, line)
        assert len(replies) == 1, replies
        if limited:
            assert replies[0]["id"] is None and replies[0]["error"]["code"] == -32700
        else:
            assert "result" in replies[0], replies[0]
```

- [ ] **Step 2: Run them**

Run: `PYTHONPATH=$PWD TMPDIR=/private/tmp/a3-fuzz-tmp python -m pytest -q -p no:cacheprovider tests/adversarial/fuzz/test_adv_wire.py --durations=5`
Expected: all pass, in about three seconds. These tests check behaviour that exists, so there is no production code to write. Step 3 shows that they fail when that behaviour breaks.

- [ ] **Step 3: Show that the tests catch each fault they are for**

Save this script as `local/a3-fuzz-findings/mutate.py`. It copies the checkout to a scratch folder, applies one fault, and runs the named test file there. Each child process imports the copy, because `harness.env.child_env` points `PYTHONPATH` at the checkout the harness lives in.

```python
"""Run one fuzz test file against a copy of the checkout with one deliberate fault."""
import os
import pathlib
import shutil
import subprocess
import sys

CHECKOUT = pathlib.Path(__file__).resolve().parents[2]
FAULTS = {
    "batch": ("memvara/server/mcp.py", "        if not isinstance(message, Mapping):\n",
              "        if isinstance(message, list) and message and isinstance(message[0], dict):\n"
              "            message = message[0]\n"
              "        if not isinstance(message, Mapping):\n"),
    "recursion": ("memvara/server/protocol.py", "    except RecursionError:",
                  "    except ZeroDivisionError:"),
    "blank-lines": ("memvara/server/protocol.py", "        if line:\n            yield line",
                    "        yield line"),
    "null-id": ("memvara/server/mcp.py", "is_request = request_id is not None",
                "is_request = \"id\" in message"),
    "string-boolean": ("memvara/server/validate.py", "        if not isinstance(value, bool):",
                       "        if not isinstance(value, (bool, str)):"),
    "surrogate": ("memvara/server/validate.py", 'value.encode("utf-8")',
                  'value.encode("utf-8", "surrogatepass")'),
    "infinite-maximum": ("memvara/server/validate.py",
                         "if high is not None and value > high:",
                         "if high is not None and value > high and value != float(\"inf\"):"),
    "inclusive-maximum": ("memvara/server/validate.py",
                          "if high is not None and value > high:",
                          "if high is not None and value >= high:"),
}

fault, test_file = sys.argv[1], sys.argv[2]
path, old, new = FAULTS[fault]
copy = pathlib.Path("/private/tmp/a3-fuzz-tmp/mutants") / fault
shutil.rmtree(copy, ignore_errors=True)
for part in ("memvara", "tests", "plugin"):
    shutil.copytree(CHECKOUT / part, copy / part,
                    ignore=shutil.ignore_patterns("__pycache__", ".hypothesis"))
shutil.copy2(CHECKOUT / "pyproject.toml", copy / "pyproject.toml")
target = copy / path
source = target.read_text()
assert source.count(old) == 1, (fault, source.count(old))
target.write_text(source.replace(old, new))
env = {"PYTHONPATH": str(copy), "TMPDIR": "/private/tmp/a3-fuzz-tmp"}
result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                         "-x", test_file], cwd=copy, env={**os.environ, **env})
print(f"fault {fault}: pytest exited {result.returncode}")
```

Run, for example: `python local/a3-fuzz-findings/mutate.py batch tests/adversarial/fuzz/test_adv_wire.py`. The faults and the file that must fail with each: `batch`, `recursion`, `blank-lines` and `null-id` with `test_adv_wire.py`. Record the exit code and the failing test of each for the final report. Nothing from this step is committed.

- [ ] **Step 4: Add the bullet to `docs/claude/testing.md`**

Append to the section's list:

```markdown
- **The wire** (`test_adv_wire.py`). 500 requests are written before any reply is read, with shuffled ids and a mix of methods, and every write among them must be stored exactly once. An id of every JSON type comes back unchanged. A null id is treated as a notification and gets no reply. A batch is refused whole, and nothing in it runs. Two requests on one line get one parse error. Blank lines and CRLF endings get no reply. A line nested 100,000 deep gets a parse error, wherever the nesting sits, and the deepest nesting that still parses is found by a search and answered. An integer of more than 4,300 digits gets a parse error.
```

- [ ] **Step 5: Commit**

```bash
git add tests/adversarial/fuzz/test_adv_wire.py docs/claude/testing.md
git commit -m "Check the stdio server's replies to pipelined, batched, nested and oddly framed input"
```

---

### Task 4: Arguments a schema forbids, for every tool, over the pipe

**Files:**
- Create: `tests/adversarial/fuzz/test_adv_arguments.py` (it keeps the validator prediction tests of Task 2 only if they passed)
- Modify: `docs/claude/testing.md` (one bullet)

**Interfaces:**
- Consumes: `call_line`, `exchange`, `text_of`, `rows`, `changed`, `shared_server` from Task 1.

- [ ] **Step 1: Write the tests**

```python
"""Argument values a tool's schema does not allow, sent over the real pipe to every tool.

The validator in `memvara/server/validate.py` refuses each of these before the tool runs.
So each test checks the three things a refusal must do: come back as one tool error that
names the argument, change nothing in the store, and leave the server answering.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pytest

from harness.stdio import McpProcess
from memvara.server.tools import TOOLS, Tool

from . import call_line, changed, exchange, rows, text_of
from . import shared_server  # noqa: F401 - a fixture; importing it lets pytest find it here

#: A valid value for each required argument, so that the one argument a test breaks is
#: the only thing wrong with the call. A tool that gains a required argument fails here
#: with a KeyError until it is added.
REQUIRED: dict[str, Any] = {
    "query": "tea", "question": "what do I like", "entity": "user", "source": "user",
    "target": "tea", "since": "2024-01-01", "text": "I like tea.", "predicate": "likes",
    "object": "tea", "from_id": "cl_0", "to_id": "cl_1", "relation": "extends",
    "claim_id": "cl_0", "id": "doc_0",
}

#: A lone surrogate: half of a character, which cannot be encoded as UTF-8.
LONE = "a\ud800b"


def _minimal(tool: Tool) -> dict[str, Any]:
    return {name: REQUIRED[name] for name in tool.required}


def _types(spec: dict[str, Any]) -> list[str]:
    return spec["type"] if isinstance(spec["type"], list) else [spec["type"]]


def _of_type(kind: str) -> list[tuple[Tool, str]]:
    return [(tool, name) for tool in TOOLS for name, spec in tool.properties.items()
            if _types(spec) == [kind]]


def _label(pair: tuple[Tool, str]) -> str:
    return f"{pair[0].name}.{pair[1]}"


def _refused(server: McpProcess, tool: str, arguments: Any) -> str:
    """Send one call, check that it got exactly one reply, that the reply is a tool error,
    and that the store did not change, and return the reply's text."""
    before = rows(server.db)
    replies = exchange(server, call_line(1, tool, arguments))
    assert len(replies) == 1 and replies[0].get("id") == 1, replies
    assert replies[0]["result"]["isError"] is True, replies[0]
    assert changed(before, rows(server.db)) == []
    return text_of(replies[0])


BOOLEANS = _of_type("boolean")
INTEGERS = _of_type("integer")
NUMBERS = _of_type("number")


@pytest.mark.parametrize("value", ["false", "true", 0, 1], ids=repr)
@pytest.mark.parametrize("pair", BOOLEANS, ids=_label)
def test_a_string_or_a_number_where_a_boolean_goes_is_refused_by_name(
        shared_server: McpProcess, pair: tuple[Tool, str], value: Any) -> None:
    """Every handler reads its flags through `bool(...)`, where the string "false" is
    True, so the validator accepts only a real boolean. Its docstring tells how "false"
    once reached production as True."""
    tool, name = pair
    text = _refused(shared_server, tool.name, {**_minimal(tool), name: value})
    assert text.startswith(f"{tool.name}.{name} must be a boolean, got "), text
    assert text.endswith(f"({value!r})"), text


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf],
                         ids=["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("pair", INTEGERS, ids=_label)
def test_nan_and_the_infinities_are_refused_where_an_integer_goes(
        shared_server: McpProcess, pair: tuple[Tool, str], value: float) -> None:
    """The server's JSON parser accepts the tokens NaN, Infinity and -Infinity, so they
    reach the validator as floats, and a float is not an integer."""
    tool, name = pair
    text = _refused(shared_server, tool.name, {**_minimal(tool), name: value})
    assert text == f"{tool.name}.{name} must be an integer, got a number ({value!r})"


@pytest.mark.parametrize("value", [math.inf, -math.inf], ids=["Infinity", "-Infinity"])
@pytest.mark.parametrize("pair", NUMBERS, ids=_label)
def test_an_infinite_number_is_refused_by_the_bound_it_breaks(
        shared_server: McpProcess, pair: tuple[Tool, str], value: float) -> None:
    tool, name = pair
    spec = tool.properties[name]
    bound = f"<= {spec['maximum']}" if value > 0 else f">= {spec['minimum']}"
    text = _refused(shared_server, tool.name, {**_minimal(tool), name: value})
    assert text == f"{tool.name}.{name} must be {bound}, got {value!r}"


@pytest.mark.parametrize("case", ["a new slot", "a slot that holds a value",
                                  "a named replacement"])
def test_a_nan_confidence_is_refused_and_leaves_the_slot_as_it_was(
        shared_server: McpProcess, case: str) -> None:
    """A NaN confidence must never be stored. A write refused for one must leave the store
    as it was, including the value it would otherwise have ended: confidence decides
    whether a write may end the value already in its slot, and every comparison with NaN
    is false."""
    arguments: dict[str, Any] = {"predicate": "lives_in", "object": "Oslo",
                                 "confidence": math.nan}
    if case == "a new slot":
        arguments["predicate"] = "works_in"
    elif case == "a named replacement":
        found = shared_server.call("memory_search", query="Berlin").text
        berlin = re.search(r"id=(cl_[0-9a-f]+)[^\n]*lives in Berlin", found)
        assert berlin is not None, found
        arguments["replaces"] = berlin.group(1)
    _refused(shared_server, "memory_remember", arguments)


def _surrogate_cases() -> list[Any]:
    """Each place in a tool's arguments where its schema says a string goes, holding a
    lone surrogate: a string argument, a string item of an array, a string value of an
    object, and an object's key where the schema declares `propertyNames` for its keys.
    Each case carries the label the refusal must start with."""
    cases = []
    for tool in TOOLS:
        for name, spec in tool.properties.items():
            label = f"{tool.name}.{name}"
            kinds = _types(spec)
            if "string" in kinds:
                cases.append(pytest.param(tool, name, LONE, label, id=label))
            if "array" in kinds and "string" in _types(spec["items"]):
                cases.append(pytest.param(tool, name, ["tea", LONE], f"{label}[1]",
                                          id=f"{label} item"))
            if "object" in kinds:
                values = spec["additionalProperties"]
                if "string" in _types(values):
                    cases.append(pytest.param(tool, name, {"team": LONE}, f"{label}.team",
                                              id=f"{label} value"))
                elif "array" in _types(values):
                    cases.append(pytest.param(tool, name, {"team": [LONE]},
                                              f"{label}.team[0]", id=f"{label} value item"))
                if "propertyNames" in spec:
                    cases.append(pytest.param(tool, name, {LONE: "tea"},
                                              f"{label} key {LONE!r}", id=f"{label} key"))
    return cases


@pytest.mark.parametrize(("tool", "name", "value", "label"), _surrogate_cases())
def test_a_lone_surrogate_is_refused_wherever_the_schema_says_a_string_goes(
        shared_server: McpProcess, tool: Tool, name: str, value: Any, label: str) -> None:
    """Python's JSON parser turns the escape \\ud800 into half of a character, which no
    store can encode. The validator refuses it wherever the schema says a string goes,
    and names the argument, rather than the exception a store would raise later."""
    text = _refused(shared_server, tool.name, {**_minimal(tool), name: value})
    assert text.startswith(label), text
    assert "unpaired surrogate" in text, text


def test_a_lone_surrogate_outside_the_arguments_is_answered_by_the_protocol(
        shared_server: McpProcess) -> None:
    """In an argument's name, it is an unknown argument. In a tool's name or a method's,
    it is an unknown tool or an unknown method, answered with the request's id."""
    assert _refused(shared_server, "memory_search", {"query": "tea", LONE: 1}).startswith(
        "memory_search: unknown argument(s) 'a\\ud800b'")
    replies = exchange(shared_server, call_line(2, LONE, {}))
    assert [(reply["id"], reply["error"]["code"]) for reply in replies] == [(2, -32602)]
    replies = exchange(shared_server, '{"jsonrpc":"2.0","id":3,"method":"a\\ud800b"}')
    assert [(reply["id"], reply["error"]["code"]) for reply in replies] == [(3, -32601)]
```

- [ ] **Step 2: Run them**

Run: `PYTHONPATH=$PWD TMPDIR=/private/tmp/a3-fuzz-tmp python -m pytest -q -p no:cacheprovider tests/adversarial/fuzz/test_adv_arguments.py --durations=5`
Expected: all pass. Then run `local/a3-fuzz-findings/mutate.py` from Task 3 with each of the faults `string-boolean`, `surrogate` and `infinite-maximum` against this file. Each must make it fail. Record the results for the final report.

- [ ] **Step 3: Add the bullet to `docs/claude/testing.md`**

```markdown
- **Arguments** (`test_adv_arguments.py`), for every tool that takes each kind of argument: the strings "false" and "true" and the numbers 0 and 1 where a boolean goes; NaN, Infinity and -Infinity where an integer goes; an infinity past a number's bounds; a NaN confidence, which must leave the slot as it was even when the write would have ended the value there; and a lone surrogate in every place the schema says a string goes. Each must be refused with a message that names the argument, and change nothing.
```

- [ ] **Step 4: Commit**

```bash
git add tests/adversarial/fuzz/test_adv_arguments.py docs/claude/testing.md
git commit -m "Check that the server refuses booleans, numbers and strings a tool's schema forbids"
```

---

### Task 5: Generated calls, from every tool's own schema

**Files:**
- Create: `tests/adversarial/fuzz/test_adv_schema_fuzz.py`
- Modify: `docs/claude/testing.md` (one bullet)

**Interfaces:**
- Consumes: `arguments`, `broken_arguments`, `json_values`, `ANY_TEXT`, `call_line`, `exchange`, `text_of`, `rows`, `changed`, `seed` from Task 1; the `mcp` fixture.
- Produces: `PER_TOOL` and `NEVER_SENT`, which Task 6 imports.

- [ ] **Step 1: Write the tests**

```python
"""Arguments drawn by Hypothesis from each tool's own input schema.

In the test process, every tool's validator must accept every call that follows its
schema, refuse every call that breaks one rule of it with a message naming the rule, and
never raise anything but a refusal, whatever JSON it is handed. Over the real pipe, a
sample of the same calls must each get one reply, a refusal must be the validator's own
message and change nothing in the store, and the server must still answer a ping.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from harness.stdio import McpProcess
from memvara.server.tools import TOOLS, Tool
from memvara.server.validate import ToolError, validate

from . import (ANY_TEXT, arguments, broken_arguments, call_line, changed, exchange,
               json_values, rows, seed, text_of)

Start = Callable[..., McpProcess]

#: Examples per tool. The tier's profile sets a count for a whole test, and each property
#: here runs once for every one of the 22 tools, so the nightly and weekly tiers take a
#: tenth of their count per tool (300 and 2,000), and the fast tier keeps its 30.
PER_TOOL = settings(max_examples=max(30, settings.default.max_examples // 10))

#: Arguments never sent to a real server. memory_add_document fetches a url, and this
#: suite does not reach the network.
NEVER_SENT = frozenset({"url"})


def _as_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
@PER_TOOL
@given(data=st.data())
def test_a_call_that_follows_the_schema_is_accepted_and_its_defaults_filled_in(
        tool: Tool, data: st.DataObject) -> None:
    """Every argument the caller sent comes back unchanged, every argument it left out
    that has a default comes back as that default, and nothing else is added. Running the
    result through the validator again changes nothing."""
    sent = data.draw(arguments(tool.properties, tool.required))
    out = validate(tool.properties, tool.required, sent, tool=tool.name)
    expected = {name: spec["default"] for name, spec in tool.properties.items()
                if "default" in spec}
    expected.update(sent)
    assert _as_json(out) == _as_json(expected)
    again = validate(tool.properties, tool.required, out, tool=tool.name)
    assert _as_json(again) == _as_json(out)


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
@PER_TOOL
@given(data=st.data())
def test_a_call_that_breaks_one_rule_is_refused_with_one_line_that_names_it(
        tool: Tool, data: st.DataObject) -> None:
    """The refusal starts with the tool's name and the argument at fault, which is what
    lets a model correct its call, and it is one line, because a value quoted in it
    must not be able to start a line of its own."""
    sent, start = data.draw(broken_arguments(tool.name, tool.properties, tool.required))
    with pytest.raises(ToolError) as refused:
        validate(tool.properties, tool.required, sent, tool=tool.name)
    message = str(refused.value)
    assert message.startswith(start), (message[:300], start)
    assert message.splitlines() == [message], message[:300]


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
@PER_TOOL
@given(data=st.data())
def test_any_json_as_arguments_is_accepted_or_refused_and_nothing_else_is_raised(
        tool: Tool, data: st.DataObject) -> None:
    """Whatever JSON arrives, the validator answers with arguments or with a refusal. Any
    other exception would reach the model as a Python error instead of something it can
    act on, which `validate.py`'s docstring tells happened once for booleans."""
    names = st.sampled_from(sorted(tool.properties)) | ANY_TEXT if tool.properties \
        else ANY_TEXT
    sent = data.draw(st.dictionaries(names, json_values(), max_size=6) | json_values())
    try:
        validate(tool.properties, tool.required, sent, tool=tool.name)
    except ToolError:
        pass


def test_a_sample_of_generated_calls_each_get_one_reply_and_a_refusal_changes_nothing(
        mcp: Start) -> None:
    """Calls drawn from the schemas the server itself lists, some following them and some
    breaking one rule, sent over the real pipe:

    * each call gets exactly one reply, with its id, and the server then answers a ping;
    * a call the validator refuses gets the validator's own message, word for word;
    * a call that comes back as an error has changed nothing in the store.
    """
    server = mcp()
    server.initialize()
    seed(server)
    schemas = {spec["name"]: spec["inputSchema"] for spec in server.list_tools()}
    ids = itertools.count(1)

    @given(data=st.data())
    def check(data: st.DataObject) -> None:
        name = data.draw(st.sampled_from(sorted(schemas)), label="tool")
        properties, required = schemas[name]["properties"], schemas[name]["required"]
        drawn = data.draw(st.one_of(
            arguments(properties, required, leave_out=NEVER_SENT),
            broken_arguments(name, properties, required, leave_out=NEVER_SENT).map(
                lambda pair: pair[0])), label="arguments")
        request_id = next(ids)
        line = call_line(request_id, name, drawn)
        received = json.loads(line)["params"]["arguments"]  # what the server will parse
        before = rows(server.db)
        replies = exchange(server, line)
        assert [reply.get("id") for reply in replies] == [request_id], replies
        result = replies[0]["result"]
        try:
            validate(properties, required, received, tool=name)
        except ToolError as refusal:
            assert result["isError"] is True and text_of(replies[0]) == str(refusal)
        if result["isError"]:
            assert changed(before, rows(server.db)) == []

    check()
```

- [ ] **Step 2: Run them**

Run: `PYTHONPATH=$PWD TMPDIR=/private/tmp/a3-fuzz-tmp python -m pytest -q -p no:cacheprovider tests/adversarial/fuzz/test_adv_schema_fuzz.py --durations=5`
Expected: all pass, in about five seconds.

- [ ] **Step 3: Show that the properties catch a fault**

Run `local/a3-fuzz-findings/mutate.py` from Task 3 with the faults `string-boolean` and `inclusive-maximum` against this file. Each must make a property fail. Record the results for the final report.

- [ ] **Step 4: Add the bullet to `docs/claude/testing.md`**

```markdown
- **Generated calls** (`test_adv_schema_fuzz.py`). Hypothesis draws arguments from each tool's own input schema, so a new argument is fuzzed without anyone editing the suite. In the test process, a call that follows the schema must be accepted with its defaults filled in, a call that breaks one rule must be refused with one line that names the rule, and any JSON at all must be accepted or refused, never met with another exception. Over the real pipe, a sample of such calls must each get one reply, a refusal must be the validator's own message word for word, and a refused call must change nothing. Each property runs once per tool, so the nightly and weekly tiers give it a tenth of their example count per tool, 300 and 2,000, and the fast tier keeps 30.
```

- [ ] **Step 5: Commit**

```bash
git add tests/adversarial/fuzz/test_adv_schema_fuzz.py docs/claude/testing.md
git commit -m "Fuzz every tool's arguments from its own schema, in the validator and over the pipe"
```

---

### Task 6: The nightly tier: 20 MB lines, and servers configured other ways

**Files:**
- Create: `tests/adversarial/fuzz/nightly/__init__.py`
- Create: `tests/adversarial/fuzz/nightly/test_adv_long_lines_nightly.py`
- Create: `tests/adversarial/fuzz/nightly/test_adv_server_configs_nightly.py`
- Modify: `docs/claude/testing.md` (one bullet)

**Interfaces:**
- Consumes: `exchange`, `call_line`, `request_line`, `rows`, `changed`, `text_of`, `seed`, `arguments`, `broken_arguments` from Task 1; `NEVER_SENT` from Task 5.

- [ ] **Step 1: Write the tests**

`tests/adversarial/fuzz/nightly/__init__.py`:

```python
"""Tests in this folder run only when --tier selects it. See docs/claude/testing.md."""
```

`tests/adversarial/fuzz/nightly/test_adv_long_lines_nightly.py`:

```python
"""Lines of 20 MB, over the real pipe.

A client can write a line of any length, and the server reads a whole line before it
parses it. Each test checks that a 20 MB line gets exactly one reply, or none when it is
blank, and that the server still answers a ping afterwards. Storing a 20 MB fact takes
about a minute on a laptop, which is why these tests run nightly.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from harness.stdio import McpProcess

from .. import call_line, changed, exchange, rows

Start = Callable[..., McpProcess]

MB_20 = 20 * 1024 * 1024

#: Seconds to wait for one reply. The slowest line here takes about a minute.
PATIENCE = 600.0


def _server(mcp: Start) -> McpProcess:
    server = mcp(timeout=PATIENCE)
    server.initialize()
    return server


def _one_reply(server: McpProcess, line: str) -> dict[str, Any]:
    replies = exchange(server, line, timeout=PATIENCE)
    assert len(replies) == 1, [str(reply)[:300] for reply in replies]
    return replies[0]


def test_a_request_padded_with_20_mb_of_whitespace_is_answered_once(mcp: Start) -> None:
    server = _server(mcp)
    reply = _one_reply(server, '{"jsonrpc":"2.0","id":1,' + " " * MB_20 + '"method":"ping"}')
    assert reply == {"jsonrpc": "2.0", "id": 1, "result": {}}


def test_a_20_mb_line_that_is_not_json_gets_one_parse_error(mcp: Start) -> None:
    server = _server(mcp)
    reply = _one_reply(server, "x" * MB_20)
    assert reply["id"] is None and reply["error"]["code"] == -32700


def test_a_20_mb_line_of_spaces_is_a_blank_line_and_gets_no_reply(mcp: Start) -> None:
    """`iter_messages` strips a line before it looks at it, so this is a blank line."""
    server = _server(mcp)
    assert exchange(server, " " * MB_20, timeout=PATIENCE) == []


def test_20_mb_where_an_integer_goes_is_refused_and_changes_nothing(mcp: Start) -> None:
    server = _server(mcp)
    before = rows(server.db)
    reply = _one_reply(server, call_line(2, "memory_recall", {"query": "tea", "k": "9" * MB_20}))
    assert reply["id"] == 2 and reply["result"]["isError"] is True
    assert changed(before, rows(server.db)) == []


def test_a_20_mb_argument_name_is_refused_and_changes_nothing(mcp: Start) -> None:
    server = _server(mcp)
    before = rows(server.db)
    reply = _one_reply(server, call_line(3, "memory_recall", {"query": "tea", "z" * MB_20: 1}))
    assert reply["id"] == 3 and reply["result"]["isError"] is True
    assert changed(before, rows(server.db)) == []


def test_a_20_mb_fact_is_stored_whole_and_the_server_carries_on(mcp: Start) -> None:
    """`object` has no length cap on purpose: it is the fact itself (see `_SUBJECT_CHARS`
    in `memvara/server/tools.py`). So a 20 MB fact is stored whole."""
    server = _server(mcp)
    value = "word " * (MB_20 // 5)
    reply = _one_reply(server, call_line(4, "memory_remember",
                                         {"predicate": "notes", "object": value}))
    assert reply["id"] == 4 and reply["result"]["isError"] is False
    connection = sqlite3.connect(f"{server.db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        lengths = [length for (length,) in connection.execute(
            "SELECT length(object) FROM claims WHERE predicate = 'notes'")]
    finally:
        connection.close()
    assert lengths == [len(value)]


def test_a_20_mb_query_is_answered_once(mcp: Start) -> None:
    server = _server(mcp)
    reply = _one_reply(server, call_line(5, "memory_search", {"query": "word " * (MB_20 // 5)}))
    assert reply["id"] == 5 and "result" in reply
```

`tests/adversarial/fuzz/nightly/test_adv_server_configs_nightly.py`:

```python
"""Generated calls against servers configured in the ways that change which tools are
listed and what their schemas say: read-only, anchored by default, and with every
feature that owns a tool or an argument switched off.

A call to a tool the server does not list must be refused with a reason and change
nothing. Otherwise the properties are the fast tier's (test_adv_schema_fuzz.py): one
reply per call, a validator refusal word for word, and no change after any refusal.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from harness.stdio import McpProcess
from memvara.server.tools import TOOLS
from memvara.server.validate import ToolError, validate

from .. import arguments, broken_arguments, call_line, changed, exchange, rows, seed, text_of
from ..test_adv_schema_fuzz import NEVER_SENT

Start = Callable[..., McpProcess]

#: Every feature that owns a tool or an argument.
OWNING = ("profile", "forget_matching", "end_reason", "links", "documents",
          "query_rewrite", "synthesis", "metadata_filters", "expiry_erasure")

CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "read-only": {"read_only": True},
    "anchored": {"env": {"MEMVARA_ANCHORED": "1"}},
    "features off": {"features": {name: False for name in OWNING}},
}

#: Each configuration takes a tenth of the tier's example count.
EACH = settings(max_examples=max(30, settings.default.max_examples // 10))


@pytest.mark.parametrize("options", list(CONFIGURATIONS.values()), ids=list(CONFIGURATIONS))
def test_generated_calls_to_a_server_configured_another_way_follow_the_same_rules(
        mcp: Start, options: dict[str, Any]) -> None:
    seeder = mcp()
    seeder.initialize()
    seed(seeder)
    seeder.close()
    server = mcp(seeder.db, **options)
    server.initialize()
    schemas = {spec["name"]: spec["inputSchema"] for spec in server.list_tools()}
    ids = itertools.count(1)

    @EACH
    @given(data=st.data())
    def check(data: st.DataObject) -> None:
        name = data.draw(st.sampled_from(sorted(tool.name for tool in TOOLS)), label="tool")
        before = rows(server.db)
        if name not in schemas:
            reply = exchange(server, call_line(next(ids), name, {}))
            assert len(reply) == 1 and reply[0]["result"]["isError"] is True, reply
            assert f"{name} is unavailable" in text_of(reply[0])
            assert changed(before, rows(server.db)) == []
            return
        properties, required = schemas[name]["properties"], schemas[name]["required"]
        drawn = data.draw(st.one_of(
            arguments(properties, required, leave_out=NEVER_SENT),
            broken_arguments(name, properties, required, leave_out=NEVER_SENT).map(
                lambda pair: pair[0])), label="arguments")
        request_id = next(ids)
        line = call_line(request_id, name, drawn)
        replies = exchange(server, line)
        assert [reply.get("id") for reply in replies] == [request_id], replies
        result = replies[0]["result"]
        try:
            validate(properties, required, json.loads(line)["params"]["arguments"], tool=name)
        except ToolError as refusal:
            assert result["isError"] is True and text_of(replies[0]) == str(refusal)
        if result["isError"]:
            assert changed(before, rows(server.db)) == []

    check()
```

- [ ] **Step 2: Run the nightly tier of this folder**

Run: `PYTHONPATH=$PWD TMPDIR=/private/tmp/a3-fuzz-tmp python -m pytest -q -p no:cacheprovider tests/adversarial/fuzz --tier nightly --durations=10`
Expected: all pass. The 20 MB fact takes about a minute.

- [ ] **Step 3: Add the bullet to `docs/claude/testing.md`**

```markdown
- **Nightly** (`nightly/`). Lines of 20 MB: a request padded with whitespace, a line that is not JSON, a line of spaces, 20 MB where an integer goes, a 20 MB argument name, a 20 MB fact, which is stored whole and takes about a minute, and a 20 MB query. And generated calls against a read-only server, a server that anchors by default, and a server with every feature that owns a tool or an argument switched off, where a call to a tool the server does not list must be refused and change nothing.
```

- [ ] **Step 4: Commit**

```bash
git add tests/adversarial/fuzz/nightly/__init__.py tests/adversarial/fuzz/nightly/test_adv_long_lines_nightly.py tests/adversarial/fuzz/nightly/test_adv_server_configs_nightly.py docs/claude/testing.md
git commit -m "Add nightly fuzzing: 20 MB lines, and generated calls to servers configured other ways"
```

---

### Task 7: Verification

- [ ] **Step 1: The fast tier of this folder, with timings**

Run: `PYTHONPATH=$PWD TMPDIR=/private/tmp/a3-fuzz-tmp python -m pytest -q -p no:cacheprovider tests/adversarial/fuzz --durations=20`
Expected: all pass, in about 12 seconds.

- [ ] **Step 2: Twenty runs of each new test file in a row**

Run each of the five fast files and the two nightly files (with `--tier nightly`) twenty times, stopping at the first failure. Record how many of the twenty passed for each.

- [ ] **Step 3: The whole gate, with a private coverage file**

Run: `PYTHONPATH=$PWD TMPDIR=/private/tmp/a3-fuzz-tmp COVERAGE_FILE=$PWD/local/cov/.coverage.a3 python -m coverage run -m pytest -q -p no:cacheprovider`
Then: `PYTHONPATH=$PWD COVERAGE_FILE=$PWD/local/cov/.coverage.a3 python -m coverage report | tail -3`
Expected: no failures, and a coverage total of 100%.

- [ ] **Step 4: The type checks**

Run: `python -m mypy -p memvara`, `python -m mypy tests/harness`, and `python -m mypy tests/harness --ignore-missing-imports`.
Expected: `Success: no issues found` from each.

- [ ] **Step 5: The report**

Write the final report the common brief describes: the branch, each commit's short sha and subject, the files, the pass counts, the flake result, the gate's result line and coverage total, the mypy results, every bug with its classification and reproduction, and every departure from this plan or from the design, with its reason.
