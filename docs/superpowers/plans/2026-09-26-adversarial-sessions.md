# Scripted agent sessions and the tool surface (A1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the scripted agent workflows the design lists for workstream A1, played over the real stdio pipe, and check the server's tool list, its input schemas and its refusals under every combination of the settings that change them, in this process for all 2,048 combinations and over the real pipe across one array per tier.

**Architecture:**

- **Workflows** are scenario files in the existing format (`tests/scenarios/schema.json`), played by the existing runner. Two workflows cannot be expressed without two more `env` fields, `confirm_secret` and `predicates`, so the runner gains those, with a skip on Python 3.10 for a scenario that loads a predicate vocabulary.
- **The tool surface** is checked by `tests/adversarial/sessions/switches.py`, a support module. It names the eleven settings that change `tools/list`, predicts the whole answer for any combination of them from the tables in `memvara/server/tools.py` (the oracle), and checks a server against it, in this process or over the pipe. The fast tier starts real servers for a 12-run orthogonal array, the nightly tier for its fold-over, and the weekly tier for all 2,048 combinations.
- **Protocol versions** and **read-only cloud credentials** get one test file each, over the real pipe, the second against `harness.fakes.FakeV1`.

**Tech Stack:** Python 3.10 to 3.13, pytest, the harness (`harness.stdio.McpProcess`, `harness.env.child_env`, `harness.stores`, `harness.fakes.FakeV1`, `harness.skips`), the scenario runner, and memvara's in-process `MemvaraMCPServer`.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`, the "A1 sessions" row of the Phase 2 table.

## Global Constraints

- **Platforms.** Python 3.10 to 3.13 on Linux, macOS and Windows. Nothing may need a feature newer than 3.10, except a test that skips on 3.10 with a reason the skip ledger explains.
- **Offline.** No test reaches the network or a model. The cloud tests use `FakeV1` on 127.0.0.1.
- **Child processes** get their environment from `harness.env.child_env`, which `McpProcess` and `HookRunner` already do.
- **Budget.** "Fast-tier budget: about 25 seconds for everything you add."
- **Files owned.** New scenario files under `tests/scenarios/scripted/`, new files under `tests/adversarial/sessions/` (with `nightly/` and `weekly/`), this plan, and a new section in `docs/claude/testing.md` just before its line that starts with `Next:`. `runner.py` and `schema.json` change only where a scenario cannot be expressed otherwise, in a small tested change. Do not edit `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md` or `tests/harness/known_bugs.py`.
- **Checklist marks.** `tests/harness/checklist.py` does not exist on `origin/main`, so no test carries a `covers` mark.
- **Known bugs.** A confirmed bug is reported, not pinned here. A bug that matches `SECURITY.md`'s "In scope" list is written into no committed file.
- **Documented behaviour** from the design's list is asserted, with a citation, and never reported as a bug.
- **Data.** Made-up people and data only.
- **Git.** Branch `test/adversarial-sessions`, no upstream. Commit files by name. No AI attribution anywhere. No path containing the word "claude" in a commit message; write "the testing guide".
- **Prose.** Plain sentences that a reader with no context understands on the first read.

Commands below use these names:

```bash
PY=/Applications/workstation/agent-memory/.claude/worktrees/friendly-einstein-53c8da/local/venv-ci/bin/python
WT=/Applications/workstation/agent-memory/.claude/worktrees/agent-a1f11d7e630bf7554
TMP=/private/tmp/a1-sessions-tmp
# run from $WT:
PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider <paths>
```

## Review Focus

1. **A tool that a switch hides but that still runs when called by name.** Hiding would then be cosmetic, and a model that remembers the name could still call it. Pinned by `refusal_problems`, which every in-process and real-server check runs (Tasks 3 and 4).
2. **Two switches that rewrite the same tool's schema.** `anchored` and `metadata_filters` both rewrite `memory_search` and `memory_recall`. If the second rewrite started from the table, it would undo the first. Pinned by the planted fault `filters-rewritten-from-the-table` (Task 3).
3. **A switch that removes an argument a handler still reads with `args[...]`.** The call would fail with a KeyError on that server only. Pinned by the minimal call of every listed tool in every real-server run (Task 4).
4. **A confirmation token whose listed facts changed after the preview.** Confirming it must close nothing, not just the facts still live. Pinned by `stale-confirm-closed-nothing` in `bulk-end-tokens` (Task 9).
5. **A read-only server that writes.** An expired fact must be hidden, not erased, and nothing may change on disk. Pinned by the store check of every read-only real-server run (Task 4) and by `code-still-on-disk` in `expiry-off-and-read-only` (Task 11).

---

## File structure

| File | Responsibility |
|---|---|
| `tests/scenarios/schema.json` | Two more `env` fields, `confirm_secret` and `predicates`. |
| `tests/adversarial/sessions/runner.py` | Passes the two fields to each server and to the hooks' client config; skips a scenario that loads a vocabulary on Python 3.10. |
| `tests/adversarial/sessions/test_adv_scenarios.py` | Gives the skip to the tests that play a scenario. |
| `tests/adversarial/sessions/test_adv_runner.py` | Tests of the two fields and the skip. |
| `tests/adversarial/sessions/switches.py` | The settings, the oracle, the checks, the arrays and the real-server run. Not a test file. |
| `tests/adversarial/sessions/conftest.py` | The seeded store that every real server of the surface tests starts from. |
| `tests/adversarial/sessions/test_adv_switches.py` | The oracle against all 2,048 combinations in this process, the planted faults, the minimal calls and the arrays. |
| `tests/adversarial/sessions/test_adv_switches_pipe.py` | The fast tier's 12 real servers. |
| `tests/adversarial/sessions/nightly/test_adv_switches_nightly.py` | The nightly tier's 12 real servers. |
| `tests/adversarial/sessions/weekly/test_adv_switches_weekly.py` | The weekly tier's 2,048 real servers. |
| `tests/adversarial/sessions/test_adv_protocols.py` | The three protocol versions over the real pipe. |
| `tests/adversarial/sessions/test_adv_cloud_read_only.py` | A cloud-mode server with a read-only key, against `FakeV1`. |
| `tests/scenarios/scripted/*.json` | Ten new scenarios. |
| `docs/claude/testing.md` | One sentence in "How a scenario plays", and a new section, "Scripted sessions and the tool surface". |

---

### Task 1: The plan

**Files:**
- Create: `docs/superpowers/plans/2026-09-26-adversarial-sessions.md`

- [ ] **Step 1: Commit the plan**

```bash
git add docs/superpowers/plans/2026-09-26-adversarial-sessions.md
git commit -m "Plan the adversarial suite's scripted sessions and tool-surface tests"
```

---

### Task 2: A confirmation key and predicate vocabularies in a scenario's env

Two workflows need a server variable the format cannot set. An expired confirmation token can only be made by a server that holds a known key (the alternative is a ten-minute wait), and the graph tools walk only relations a vocabulary declares as edges, which no built-in predicate is.

**Files:**
- Modify: `tests/scenarios/schema.json` (the `env` definition)
- Modify: `tests/adversarial/sessions/runner.py`
- Modify: `tests/adversarial/sessions/test_adv_scenarios.py` (`pytest_generate_tests`)
- Test: `tests/adversarial/sessions/test_adv_runner.py`
- Modify: `docs/claude/testing.md`

**Interfaces:**
- Produces: `runner.VARIABLES: Mapping[str, str]`, `runner.variables(env) -> dict[str, str]`, `runner.TOMLLIB_SKIP: str`, `runner.marks(scenario) -> list[pytest.MarkDecorator]`; `DEFAULT_ENV` gains `"confirm_secret": None, "predicates": None`.

- [ ] **Step 1: Write the failing tests** at the end of `test_adv_runner.py`, with `import sys`, `from harness import skips` and `from memvara.confirm import Confirmer` added to its imports:

```python
# -- the confirmation key and the predicate vocabularies ---------------------------------

def test_a_confirm_secret_and_predicates_can_be_set_and_must_not_be_empty() -> None:
    scenario = sample(env={"user": "tester", "confirm_secret": "k", "predicates": "engineering"})
    assert errors(scenario) == []
    scenario["sessions"][1]["env"] = {"confirm_secret": ""}
    assert ("$.sessions[1].env.confirm_secret: must be at least 1 character(s) long"
            in errors(scenario))


def test_the_two_fields_become_server_variables() -> None:
    env = {**runner.DEFAULT_ENV, "confirm_secret": "k", "predicates": "engineering"}
    assert runner.variables(env) == {"MEMVARA_CONFIRM_SECRET": "k",
                                     "MEMVARA_PREDICATES": "engineering"}
    assert runner.variables(runner.DEFAULT_ENV) == {}


def test_a_scenario_that_loads_predicates_skips_below_python_3_11() -> None:
    """A session's env counts as much as the scenario's. The reason must be one the skip
    ledger explains below Python 3.11 and nowhere else, or the skip would fail the run."""
    assert runner.marks(sample()) == []
    in_a_session = sample()
    in_a_session["sessions"][1]["env"] = {"predicates": "engineering"}
    for scenario in (sample(env={"user": "tester", "predicates": "engineering"}),
                     in_a_session):
        [mark] = runner.marks(scenario)
        assert mark.name == "skipif" and mark.args == (sys.version_info < (3, 11),)
        assert mark.kwargs["reason"] == runner.TOMLLIB_SKIP
    assert skips.explained(runner.TOMLLIB_SKIP, version=(3, 10))
    assert not skips.explained(runner.TOMLLIB_SKIP, version=(3, 11))


def expired_token(secret: str) -> str:
    """A token that `secret` signed correctly and that expired in August 2026."""
    token, _ = Confirmer(secret).issue(["cl_0f0f0f0f0f0f0f0f0f0f"], "ended",
                                       now=datetime(2026, 8, 20, 9, tzinfo=timezone.utc))
    return token


def test_the_confirm_secret_reaches_the_server_on_a_real_server(
        tmp_path: pathlib.Path) -> None:
    """With the secret, the server holds the key that signed the token, so it checks the
    token as far as its expiry. Without it, the server's own key refuses the signature."""
    def answer(env: dict[str, Any], where: str) -> str:
        scenario = sample(env=env)
        del scenario["sessions"][0]
        script(scenario)[:] = [{"tool": "memory_end_matching", "expect_error": True,
                                "args": {"confirm": expired_token("k")}}]
        outcome = runner.run(scenario, tmp_path / where)
        assert outcome.problems == []
        return outcome.turn().answer

    assert "this confirmation token expired at" in answer(
        {"user": "tester", "confirm_secret": "k"}, "with")
    assert "was not issued by this memory server" in answer({"user": "tester"}, "without")


@pytest.mark.skipif(sys.version_info < (3, 11), reason=runner.TOMLLIB_SKIP)
def test_predicates_reach_the_server_on_a_real_server(tmp_path: pathlib.Path) -> None:
    """With the engineering vocabulary, runs_on is another spelling of current_host, and
    the receipt says so. With the built-in predicates alone, runs_on is a new predicate."""
    scenario = sample(env={"user": "tester", "predicates": "engineering"})
    del scenario["sessions"][0]
    script(scenario)[:] = [{"tool": "memory_remember", "args": {
        "subject": "payments-api", "predicate": "runs_on", "object": "host-7"}}]
    outcome = runner.run(scenario, tmp_path)
    assert outcome.problems == []
    assert "'runs_on' is another spelling of 'current_host'" in outcome.turn().answer
```

- [ ] **Step 2: Run them and see them fail for the right reason**

Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions/test_adv_runner.py -k "confirm_secret or two_fields or predicates"`
Expected: FAIL. The format test fails with "unknown field 'confirm_secret'", and the others with `AttributeError: module ... has no attribute 'variables'` (or `'marks'`, `'TOMLLIB_SKIP'`).

- [ ] **Step 3: Add the two fields to the schema.** In `tests/scenarios/schema.json`, the `env` definition becomes:

```json
    "env": {
      "type": "object", "additionalProperties": false,
      "description": "How the server starts. A field left out takes its default: user tester, no project, every feature at its default, writes enabled, protocol 2025-06-18, a confirmation key each server makes for itself, and the built-in predicates alone. A session's own env overrides the scenario's, field by field, except the user: a session cannot change it, because store gold reads every claim at the scenario's user.",
      "properties": {
        "user": {"type": "string", "minLength": 1, "description": "MEMVARA_USER."},
        "project": {"type": "string", "minLength": 1, "description": "MEMVARA_PROJECT, as host/owner/repo."},
        "features": {"type": "object", "additionalProperties": {"type": "boolean"}, "description": "Feature name to on or off, as MEMVARA_FEATURE_<NAME>."},
        "read_only": {"type": "boolean", "description": "MEMVARA_READ_ONLY."},
        "protocol": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$", "description": "The MCP protocol version the client asks for."},
        "confirm_secret": {"type": "string", "minLength": 1, "description": "MEMVARA_CONFIRM_SECRET: the key that signs the confirmation tokens of memory_end_matching and memory_forget_matching. Servers that share it accept each other's tokens."},
        "predicates": {"type": "string", "minLength": 1, "description": "MEMVARA_PREDICATES: the predicate vocabularies to load, such as engineering. Loading one needs Python 3.11 or later, so every test that plays the scenario skips on Python 3.10."}
      }
    },
```

- [ ] **Step 4: Teach the runner the two fields.** In `runner.py`:

After `selected`, add the skip:

```python
#: Why a scenario that loads a predicate vocabulary does not play on Python 3.10. It is
#: worded to match the skip ledger's rule for tomllib in tests/harness/skips.py.
TOMLLIB_SKIP = "tomllib arrives in 3.11, and this scenario loads a predicate vocabulary with it"


def marks(scenario: Mapping[str, Any]) -> list[pytest.MarkDecorator]:
    """The marks every test that plays `scenario` carries.

    A scenario whose env, or any session's env, names `predicates` skips on Python 3.10,
    because loading a vocabulary needs `tomllib` and the server would refuse to start.
    Any other scenario carries none.
    """
    envs = [scenario["env"], *(session.get("env", {}) for session in scenario["sessions"])]
    if any(env.get("predicates") for env in envs):
        return [pytest.mark.skipif(sys.version_info < (3, 11), reason=TOMLLIB_SKIP)]
    return []
```

Replace `DEFAULT_ENV` and add the variables:

```python
#: The server settings a scenario gets for anything its `env` leaves out.
DEFAULT_ENV: Mapping[str, Any] = {
    "user": "tester", "project": None, "features": {}, "read_only": False,
    "protocol": PROTOCOL, "confirm_secret": None, "predicates": None}

#: The env fields that each set one server variable, for the server and for the client
#: config the hooks read.
VARIABLES: Mapping[str, str] = {"confirm_secret": "MEMVARA_CONFIRM_SECRET",
                                "predicates": "MEMVARA_PREDICATES"}


def variables(env: Mapping[str, Any]) -> dict[str, str]:
    """The variables an env's `confirm_secret` and `predicates` fields set, for the
    fields that have a value.

    >>> variables({"confirm_secret": "k", "predicates": None})
    {'MEMVARA_CONFIRM_SECRET': 'k'}
    """
    return {name: str(env[field]) for field, name in VARIABLES.items() if env.get(field)}
```

In `gold_params`, each parameter also carries its scenario's marks:

```python
def gold_params(scenarios: Iterable[Mapping[str, Any]]) -> list[Any]:
    """One pytest parameter per gold item. An item that a known bug breaks carries that
    bug's strict expected-failure marker, and no other item does. Every item also carries
    its scenario's `marks`."""
    return [pytest.param(gold, id=gold.test_id,
                         marks=[*marks(scenario),
                                *([known_bugs.xfail(gold.known_bug["bug"])]
                                  if gold.known_bug else [])])
            for scenario in scenarios for gold in gold_items(scenario)]
```

In `_Session.__init__`, pass the variables to the server:

```python
        self.server = McpProcess(
            db, home=home, user=env["user"], features=env["features"],
            read_only=env["read_only"], cwd=work, env=variables(env),
            scope={"project": env["project"]} if env["project"] else None)
```

In `_Session._server_env`, before the feature switches:

```python
        env.update(variables(self.env))
```

- [ ] **Step 5: Give the skip to the tests that play a scenario.** In `test_adv_scenarios.py`, the end of `pytest_generate_tests` becomes:

```python
    if name == "test_no_forbidden_tool_was_called":
        scenarios = [scenario for scenario in scenarios if scenario.get("forbidden")]
    elif name == "test_the_gold_fails_without_memvara":
        scenarios = [scenario for scenario in scenarios if scenario["negative_control"]]
    # The negative control starts no server, so it needs none of the marks playing needs.
    plays = name != "test_the_gold_fails_without_memvara"
    metafunc.parametrize("scenario", [
        pytest.param(scenario, id=scenario["id"], marks=runner.marks(scenario) if plays else [])
        for scenario in scenarios])
```

- [ ] **Step 6: Run the runner's tests and every scenario**

Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions`
Expected: PASS, the 14 existing scenarios included. `$PY -m pytest -q -p no:cacheprovider --doctest-modules tests/adversarial/sessions/runner.py` passes the new doctest.

- [ ] **Step 7: Document it.** In `docs/claude/testing.md`, section "How a scenario plays", the sentence "`env` sets how the server starts: the user, the project, the feature switches, read-only mode and the protocol version." becomes "`env` sets how the server starts: the user, the project, the feature switches, read-only mode, the protocol version, the confirmation key and the predicate vocabularies. The last two are described in "Scripted sessions and the tool surface" below." Then add the new section just before the line that starts with `Next:`:

```markdown
## Scripted sessions and the tool surface

This section covers workstream A1 of the design: more scripted scenarios, and tests of the tool list, the input schemas and the refusals the server gives under every combination of the settings that change them. The plan is `docs/superpowers/plans/2026-09-26-adversarial-sessions.md`.

### Two more fields in a scenario's env

`env` can also set two server variables, because two workflows cannot be written without them:

- `confirm_secret` sets `MEMVARA_CONFIRM_SECRET`, the key that signs the confirmation tokens of `memory_end_matching` and `memory_forget_matching`. Servers that share it accept each other's tokens, so a preview from one session can be confirmed in the next. It also lets a scenario hold a token that was signed correctly and has expired; the only other way to get one is to wait ten minutes.
- `predicates` sets `MEMVARA_PREDICATES`, the predicate vocabularies the server loads. The graph tools walk only relations a vocabulary declares as graph edges, and no built-in predicate is one, so without a vocabulary `memory_neighborhood` and `memory_paths` can only answer that nothing is connected. Loading a vocabulary needs Python 3.11, where `tomllib` arrives. So every test that plays such a scenario skips on Python 3.10, with a reason the skip ledger's `tomllib` rule explains.

The runner passes both to each session's server, and writes them into the client config the hooks read, like the other env fields.
```

- [ ] **Step 8: Commit**

```bash
git add tests/scenarios/schema.json tests/adversarial/sessions/runner.py \
  tests/adversarial/sessions/test_adv_scenarios.py tests/adversarial/sessions/test_adv_runner.py \
  docs/claude/testing.md
git commit -m "Let a scenario set the confirmation key and the predicate vocabularies" -m "An expired confirmation token can only come from a server that holds a known key, and the graph tools only walk relations that a loaded vocabulary declares as edges. Loading one needs tomllib, so a scenario that does skips on Python 3.10 with the skip ledger's existing tomllib reason."
```

---

### Task 3: The oracle, checked in this process against all 2,048 combinations

**Files:**
- Create: `tests/adversarial/sessions/switches.py` (everything except the section "real servers", which Task 4 adds)
- Test: `tests/adversarial/sessions/test_adv_switches.py`
- Modify: `docs/claude/testing.md`

**Interfaces:**
- Consumes: `memvara.server.tools.TOOLS`, `FEATURE_ARGUMENTS`, `_FILTERS_OFF`, `_EXPIRY_ARGUMENTS`, `_EXPIRES_AT_OFF`, `_ANCHORED`, `_ANCHORED_ON`, `_ANCHORED_ASK`, `_ANCHORED_ASK_ON`; `memvara.server.config.FEATURE_DEFAULTS`, `ServerConfig.from_env`; `harness.env.child_env`; `harness.stdio.ToolResult`.
- Produces: `switches.SWITCHES`, `Combination` (`.of`, `.moved`, `.read_only`, `.anchored`, `.feature_on`, `.env()`, `.label`), `unavailable`, `removed`, `served`, `Ids`, `NOWHERE`, `minimal_calls`, `PROBES`, `FILTERING`, `FILTER`, `USER`, `STORE`, `Client`, `InProcess`, `base_env`, `in_process`, `listing_problems`, `refusal_problems`, `call_problems`, `in_process_failures`, `summary`, `fast_runs`, `nightly_runs`, `every_combination`, `interaction_counts`.

- [ ] **Step 1: Write the failing test file** `tests/adversarial/sessions/test_adv_switches.py`:

```python
"""The tool surface in this process: the oracle in switches.py against a server built for
each of the 2,048 combinations of the settings that change it, and the arrays of
combinations the real-server tests start.

The section "Scripted sessions and the tool surface" in docs/claude/testing.md explains
the oracle and the arrays.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Callable

import pytest

from harness import stores
from memvara.server import mcp
from memvara.server.config import FEATURES, FEATURES_OFF_BY_DEFAULT
from memvara.server.mcp import MemvaraMCPServer
from memvara.server.tools import BY_NAME, FEATURE_ARGUMENTS, TOOLS, without_filters
from memvara.server.validate import validate

from . import switches

Combination = switches.Combination


@pytest.fixture
def base(tmp_path: pathlib.Path) -> dict[str, str]:
    return switches.base_env(tmp_path)


# -- the settings and the oracle -------------------------------------------------------

def test_the_switches_are_the_settings_that_change_the_tool_list() -> None:
    """Every feature is switched away from its default one at a time, and read-only and
    anchored mode are switched on. The ones that change the list must be exactly the
    eleven in SWITCHES, so a new setting that changes it cannot go unchecked."""
    memory = stores.memory()

    def listing(**options: Any) -> list[dict[str, Any]]:
        memory.metadata_filters = True
        return switches.InProcess(MemvaraMCPServer(memory, user=switches.USER,
                                                   **options)).list_tools()

    try:
        default = listing()
        changed = {feature for feature in FEATURES
                   if listing(features_off=FEATURES_OFF_BY_DEFAULT ^ {feature}) != default}
        changed |= {mode for mode in ("read_only", "anchored")
                    if listing(**{mode: True}) != default}
    finally:
        memory.close()
    assert changed == set(switches.SWITCHES)


def test_a_combination_sets_one_variable_for_each_setting() -> None:
    env = Combination.of("documents", "read_only").env()
    assert len(env) == len(switches.SWITCHES)
    assert env["MEMVARA_FEATURE_DOCUMENTS"] == "0" and env["MEMVARA_FEATURE_LINKS"] == "1"
    assert env["MEMVARA_READ_ONLY"] == "1" and env["MEMVARA_ANCHORED"] == "0"


def test_a_combination_refuses_a_setting_that_does_not_change_the_list() -> None:
    with pytest.raises(ValueError, match="encryption"):
        Combination.of("encryption")


def test_every_combination_lists_and_refuses_what_the_oracle_predicts(
        base: dict[str, str]) -> None:
    runs = switches.every_combination()
    failures = switches.in_process_failures(runs, base)
    assert not failures, switches.summary(failures, len(runs))


# -- the check finds the faults it is for ----------------------------------------------

def _keeps_synthesize(patch: pytest.MonkeyPatch) -> None:
    """The server forgets that `synthesis` owns an argument, so `synthesize` stays."""
    patch.setattr(mcp, "FEATURE_ARGUMENTS",
                  {k: v for k, v in FEATURE_ARGUMENTS.items() if k != "synthesis"})


def _keeps_filter_descriptions(patch: pytest.MonkeyPatch) -> None:
    """With metadata filters off, the filter arguments keep their usual descriptions."""
    patch.setattr(mcp, "without_filters", lambda tools: tools)


def _rewrites_filters_from_the_table(patch: pytest.MonkeyPatch) -> None:
    """The filter rewrite starts from each tool's entry in TOOLS instead of from the tool
    it was given, so it undoes the anchored rewrite and brings removed arguments back."""
    patch.setattr(mcp, "without_filters", lambda tools: without_filters(tuple(
        BY_NAME[tool.name] if "filters" in tool.properties else tool for tool in tools)))


def _marks_a_write_tool_read_only(patch: pytest.MonkeyPatch) -> None:
    """memory_link is marked as a tool that does not write. Its annotation is then wrong
    wherever it is listed, and a read-only server lists it."""
    patch.setattr(mcp, "TOOLS", tuple(dataclasses.replace(tool, writes=False)
                                      if tool.name == "memory_link" else tool
                                      for tool in TOOLS))


FAULTS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None],
                        Callable[[Combination], bool]]] = {
    "synthesize-is-kept": (_keeps_synthesize, lambda c: not c.feature_on("synthesis")),
    "filter-descriptions-are-kept": (
        _keeps_filter_descriptions, lambda c: not c.feature_on("metadata_filters")),
    "filters-rewritten-from-the-table": (
        _rewrites_filters_from_the_table,
        lambda c: not c.feature_on("metadata_filters") and (
            c.anchored or not c.feature_on("query_rewrite")
            or not c.feature_on("synthesis"))),
    "a-write-tool-marked-read-only": (_marks_a_write_tool_read_only,
                                      lambda c: c.feature_on("links")),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_the_check_finds_a_planted_fault_on_exactly_the_servers_it_affects(
        fault: str, base: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Each fault is planted in the server's composition code in memvara/server/mcp.py,
    and the listing check must fail on every combination the fault touches and on no
    other. The oracle reads memvara/server/tools.py, which stays as it is."""
    plant, affected = FAULTS[fault]
    plant(monkeypatch)
    runs = switches.every_combination()
    found = set(switches.in_process_failures(runs, base, refusals=False))
    assert found == {combination for combination in runs if affected(combination)}


# -- the minimal calls -----------------------------------------------------------------

def test_every_tool_has_a_minimal_call_that_its_schema_accepts() -> None:
    """Every call goes through this validator, so a minimal call it refused would fail on
    every server. No minimal call may send an argument that a switch removes, and each
    such argument needs a value to be probed with."""
    calls = switches.minimal_calls(switches.NOWHERE)
    assert sorted(calls) == sorted(BY_NAME)
    for tool in TOOLS:
        validate(tool.properties, tool.required, calls[tool.name], tool=tool.name)
        assert not set(calls[tool.name]) & set(switches.PROBES), tool.name
    assert set(switches.PROBES) == {argument for arguments in FEATURE_ARGUMENTS.values()
                                    for argument in arguments}


# -- the arrays ------------------------------------------------------------------------

def test_the_fast_array_is_orthogonal_and_covers_every_three_settings() -> None:
    runs = switches.fast_runs()
    assert len(runs) == len(set(runs)) == 12
    assert Combination.of() in runs
    pairs = switches.interaction_counts(runs, 2)
    assert all(len(seen) == 4 and set(seen.values()) == {3} for seen in pairs.values())
    triples = switches.interaction_counts(runs, 3)
    assert all(len(seen) == 8 for seen in triples.values())


def test_the_nightly_array_covers_every_three_settings_and_completes_strength_three() -> None:
    nightly = switches.nightly_runs()
    assert len(nightly) == len(set(nightly)) == 12
    assert not set(nightly) & set(switches.fast_runs())
    assert all(len(seen) == 8 for seen in switches.interaction_counts(nightly, 3).values())
    both = switches.interaction_counts(switches.fast_runs() + nightly, 3)
    assert all(len(seen) == 8 and set(seen.values()) == {3} for seen in both.values())


def test_the_weekly_array_is_every_combination() -> None:
    runs = switches.every_combination()
    assert len(runs) == len(set(runs)) == 2 ** len(switches.SWITCHES) == 2048
```

- [ ] **Step 2: Run it and see it fail for the right reason**

Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions/test_adv_switches.py`
Expected: FAIL at collection with `ImportError: cannot import name 'switches'`.

- [ ] **Step 3: Write `tests/adversarial/sessions/switches.py`**, every section up to and including "the arrays of combinations". The complete module, with Task 4's section at its end, is:

```python
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

import itertools
import pathlib
import shutil
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from harness import stores
from harness.env import child_env
from harness.stdio import McpProcess, ToolResult
from memvara import Memvara
from memvara.server import tools as server_tools
from memvara.server.config import FEATURE_DEFAULTS, ServerConfig
from memvara.server.mcp import MemvaraMCPServer
from memvara.server.tools import FEATURE_ARGUMENTS, TOOLS, Tool

# -- the settings ----------------------------------------------------------------------

#: Features that own a tool: with the feature off, the server does not list the tool.
TOOL_FEATURES: tuple[str, ...] = tuple(sorted({t.feature for t in TOOLS if t.feature}))

#: Features that own arguments and remove them when switched off.
ARGUMENT_FEATURES: tuple[str, ...] = tuple(sorted(FEATURE_ARGUMENTS))


@dataclass(frozen=True)
class Redescribed:
    """A feature that keeps its arguments when switched off and rewrites their descriptions.

    `arguments` are rewritten only on a tool that takes `marker`, because taking it is what
    makes a tool one the feature governs.
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

    That is the tool's required arguments, plus one more for the four tools whose schema
    requires nothing that a call can run without: memory_forget and memory_end need a claim
    to close, the two matching tools need a query for their preview, and
    memory_add_document needs its content.
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

#: The tools that take a metadata filter, and the filter they are sent.
FILTERING: tuple[str, ...] = tuple(tool.name for tool in TOOLS if "filters" in tool.properties)
FILTER: Mapping[str, str] = {"team": "support"}


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
    switch removed must be refused as unknown. A filtered read on a server without
    metadata filters must be refused, naming the switch. None of these calls reaches the
    store, because each is refused before anything runs.
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
            result = client.call(name, **{**calls[name], "filters": dict(FILTER)})
            if not (result.is_error and "MEMVARA_FEATURE_METADATA_FILTERS=0" in result.text):
                problems.append(f"{name} should refuse a filter, naming the switch, and the "
                                f"server answered: {result.text[:200]!r}")
    return problems


def call_problems(client: Client, combination: Combination,
                  calls: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Each tool a server lists must run with its minimal call, and with metadata filters
    on, a filtered read must run too."""
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
            result = client.call(name, **{**calls[name], "filters": dict(FILTER)})
            if result.is_error:
                problems.append(f"{name} with a filter failed: {result.text[:300]!r}")
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

Rows = list[tuple[str, str]]


def read_rows(path: pathlib.Path) -> Rows:
    """Every claim the user has, in every state, and every document, as stored on disk.
    Read with expiry off, so the read changes nothing."""
    with stores.file(path, expiry_erasure=False, sweep_expired=False) as memory:
        scoped = memory.scope(user=USER)
        claims = sorted((c.text, c.state)
                        for c in scoped.get_all(states=("live", "ended", "retired")))
        documents = sorted(("document", str(d.custom_id))
                           for d in scoped.list_documents(limit=100).items)
    return claims + documents


@dataclass(frozen=True)
class Template:
    """A seeded store in `directory`, copied for each server so that every run starts from
    the same memory, and the rows it holds."""

    directory: pathlib.Path
    ids: Ids
    rows: Rows

    @classmethod
    def build(cls, directory: pathlib.Path) -> Template:
        """Store four facts and a document for the minimal calls to name."""
        directory.mkdir(parents=True, exist_ok=True)
        with stores.file(directory / STORE) as memory:
            scoped = memory.scope(user=USER)
            home = scoped.remember("user", "lives_in", "Oslo").added[0]
            work = scoped.remember("user", "works_at", "Contoso").added[0]
            hobby = scoped.remember("user", "likes", "chess").added[0]
            language = scoped.remember("user", "speaks", "Norwegian").added[0]
            document = scoped.add_document("Notes about the office move to Oslo.",
                                           custom_id="notes/office.md")
        ids = Ids(linked_from=work.id, linked_to=home.id, forgettable=hobby.id,
                  endable=language.id, document=str(document.custom_id))
        return cls(directory, ids, read_rows(directory / STORE))

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
    left exactly as it was. `start` is the `mcp` fixture's function."""
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
    if combination.read_only and read_rows(db) != template.rows:
        problems.append("a read-only server changed the store")
    return problems
```

In this task, write the module without the last section ("real servers"), and leave out the imports only that section uses (`shutil`, `Callable`, `McpProcess`).

- [ ] **Step 4: Run the tests and see them pass**

Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions/test_adv_switches.py --durations=5`
Expected: PASS, 12 tests, in about 2 seconds. The whole in-process check took 0.45 s on a laptop and each planted fault 0.2 s.

- [ ] **Step 5: Document it.** Add to the new section of `docs/claude/testing.md`:

```markdown
### The tool surface

`tests/adversarial/sessions/switches.py` predicts what a server must list, and how it must refuse what it does not list, for every combination of the settings that change it.

- **Which settings count.** Eleven settings change what `tools/list` returns: the four features that own a tool (`documents`, `forget_matching`, `links` and `profile`), the three that remove arguments (`end_reason`, `query_rewrite` and `synthesis`), the two that keep their arguments and rewrite their descriptions (`metadata_filters` and `expiry_erasure`), read-only mode and anchored mode. The first seven are read from `Tool.feature` and `FEATURE_ARGUMENTS`. A guard test switches every feature away from its default, one at a time, and fails if a feature outside the eleven changes the list or one inside them does not, so a new switch cannot go unchecked.
- **The oracle.** `switches.served(combination)` builds the whole `tools/list` answer from the tables in `memvara/server/tools.py`: the tools, their arguments and descriptions, and which ones write. The rules that combine them are written out in the module, because the rules are what the tests check. A tool is listed unless its feature is off, or the server is read-only and the tool writes. An argument that a switched-off feature owns is not offered. Anchored mode gives each `anchored` argument its "default true" schema. With metadata filters or expiry erasure off, their arguments stay, with the descriptions `tools.py` writes for that case. A tool that is not listed must still be refused by name when it is called: for its feature first, naming the variable, and otherwise because the server is read-only. An argument that a switch removed must be refused as unknown.
- **Every combination, in this process.** `test_adv_switches.py` builds a server for each of the 2,048 combinations from the environment, the way `python -m memvara.server` builds itself, and compares its list and its refusals with the oracle. This takes about half a second. Four faults planted in the server's own composition code in `memvara/server/mcp.py` are each found on exactly the combinations they affect. One of them makes the filter rewrite start from the table, which undoes the anchored rewrite; only a check across combinations of switches can see that.
```

- [ ] **Step 6: Commit**

```bash
git add tests/adversarial/sessions/switches.py tests/adversarial/sessions/test_adv_switches.py \
  docs/claude/testing.md
git commit -m "Check the tool list and refusals against an oracle for all 2,048 switch combinations" -m "The oracle builds the tools/list answer from the tables in the server's tool module, and the check runs in-process for every combination of the eleven settings that change the list. Four faults planted in the server's composition code are each found on exactly the combinations they affect."
```

---

### Task 4: Real servers across one array per tier

**Files:**
- Modify: `tests/adversarial/sessions/switches.py` (add the section "real servers")
- Create: `tests/adversarial/sessions/conftest.py`
- Test: `tests/adversarial/sessions/test_adv_switches_pipe.py`
- Create: `tests/adversarial/sessions/nightly/__init__.py`, `tests/adversarial/sessions/nightly/test_adv_switches_nightly.py`
- Create: `tests/adversarial/sessions/weekly/__init__.py`, `tests/adversarial/sessions/weekly/test_adv_switches_weekly.py`
- Modify: `docs/claude/testing.md`

**Interfaces:**
- Consumes: Task 3's `switches` module; the `mcp` fixture from `tests/adversarial/conftest.py`.
- Produces: `switches.Rows`, `read_rows`, `Template` (`build`, `copy`, `ids`, `rows`), `over_the_pipe(start, combination, template, workdir) -> list[str]`; the session fixture `surface_template`.

- [ ] **Step 1: Write the failing test and the fixture.** `tests/adversarial/sessions/conftest.py`:

```python
"""Fixtures for the scripted sessions and the tool-surface tests."""

from __future__ import annotations

import pytest

from . import switches


@pytest.fixture(scope="session")
def surface_template(tmp_path_factory: pytest.TempPathFactory) -> switches.Template:
    """The seeded store every real server of the tool-surface tests starts from. It is
    built once per run and copied for each server, so no run sees another's writes."""
    return switches.Template.build(tmp_path_factory.mktemp("surface-template"))
```

`tests/adversarial/sessions/test_adv_switches_pipe.py`:

```python
"""Real servers, started with the switch combinations of the fast tier's array, each
checked against the oracle in switches.py and made to run every tool it lists.

The array is the 12-run Plackett-Burman design (`switches.fast_runs`). The nightly and
weekly tiers start larger arrays with the same check, in nightly/ and weekly/.
"""

from __future__ import annotations

import pathlib
from typing import Callable

import pytest

from harness.stdio import McpProcess

from . import switches


@pytest.mark.parametrize("combination", switches.fast_runs(), ids=lambda c: c.label)
def test_a_real_server_serves_what_the_oracle_predicts(
        combination: switches.Combination, mcp: Callable[..., McpProcess],
        surface_template: switches.Template, tmp_path: pathlib.Path) -> None:
    assert switches.over_the_pipe(mcp, combination, surface_template, tmp_path) == []
```

- [ ] **Step 2: Run it and see it fail for the right reason**

Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions/test_adv_switches_pipe.py`
Expected: ERROR in the fixture, `AttributeError: module ... has no attribute 'Template'`.

- [ ] **Step 3: Add the section "real servers" to `switches.py`,** as Task 3 shows it, with the imports it uses (`shutil`, `Callable`, `McpProcess`).

- [ ] **Step 4: Run the fast runs and see them pass**

Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions/test_adv_switches_pipe.py --durations=3`
Expected: PASS, 12 tests, about 0.25 s each on a laptop.

- [ ] **Step 5: Add the nightly and weekly arrays.** `tests/adversarial/sessions/nightly/__init__.py` holds `"""Tool-surface runs too many for every pull request. See docs/claude/testing.md."""`, and `weekly/__init__.py` the same sentence. `nightly/test_adv_switches_nightly.py`:

```python
"""The nightly tier's real servers: the fold-over of the fast tier's array, the same 12
runs with every setting reversed (`switches.nightly_runs`). The nightly run also collects
the fast tier's runs, and with them every combination of every three settings is started
exactly three times."""

from __future__ import annotations

import pathlib
from typing import Callable

import pytest

from harness.stdio import McpProcess

from .. import switches


@pytest.mark.parametrize("combination", switches.nightly_runs(), ids=lambda c: c.label)
def test_a_real_server_serves_what_the_oracle_predicts_nightly(
        combination: switches.Combination, mcp: Callable[..., McpProcess],
        surface_template: switches.Template, tmp_path: pathlib.Path) -> None:
    assert switches.over_the_pipe(mcp, combination, surface_template, tmp_path) == []
```

`weekly/test_adv_switches_weekly.py`:

```python
"""The weekly tier's real servers: every one of the 2,048 combinations
(`switches.every_combination`). A run takes about a quarter of a second on a laptop, so
the file takes about nine minutes."""

from __future__ import annotations

import pathlib
from typing import Callable

import pytest

from harness.stdio import McpProcess

from .. import switches


@pytest.mark.parametrize("combination", switches.every_combination(),
                         ids=lambda c: c.label)
def test_every_combination_on_a_real_server(
        combination: switches.Combination, mcp: Callable[..., McpProcess],
        surface_template: switches.Template, tmp_path: pathlib.Path) -> None:
    assert switches.over_the_pipe(mcp, combination, surface_template, tmp_path) == []
```

- [ ] **Step 6: Run the nightly array, and the weekly one once**

Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions/nightly --tier nightly`
Expected: PASS, 12 tests.
Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions/weekly --tier weekly`
Expected: PASS, 2,048 tests, in about nine minutes.
Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_tiers.py`
Expected: PASS: both new folders have an `__init__.py` and neither sits inside another tier folder.

- [ ] **Step 7: Document it.** Add to "The tool surface" in `docs/claude/testing.md`:

```markdown
- **Real servers.** Each run copies a small seeded store, starts `python -m memvara.server` with the combination's variables, checks its list and its refusals against the oracle, calls every listed tool once with its minimal arguments, and checks the exit code. A read-only run must also leave the store exactly as it found it. A tool's minimal call sends its required arguments, plus the one argument four tools need at run time: a claim for `memory_forget` and `memory_end`, a query for the two matching tools' previews, and content for `memory_add_document`. The fast tier runs a 12-run orthogonal array, the Plackett-Burman design: every two settings appear in each of their four combinations exactly three times, and every three settings in all eight of theirs. The nightly tier runs its fold-over, the same 12 runs with every setting reversed. It covers all eight combinations of every three settings on its own, and with the fast runs, which a nightly run also collects, each of those combinations is started exactly three times. The weekly tier runs all 2,048, which takes about nine minutes.
```

- [ ] **Step 8: Commit**

```bash
git add tests/adversarial/sessions/switches.py tests/adversarial/sessions/conftest.py \
  tests/adversarial/sessions/test_adv_switches_pipe.py \
  tests/adversarial/sessions/nightly/__init__.py \
  tests/adversarial/sessions/nightly/test_adv_switches_nightly.py \
  tests/adversarial/sessions/weekly/__init__.py \
  tests/adversarial/sessions/weekly/test_adv_switches_weekly.py docs/claude/testing.md
git commit -m "Start real servers across arrays of switch combinations and call every listed tool" -m "The fast tier starts the 12-run Plackett-Burman design, the nightly tier its fold-over, and the weekly tier all 2,048 combinations. Each server is checked against the oracle, refuses what it does not list, runs every tool it lists once, and on a read-only server leaves the store unchanged."
```

---

### Task 5: The three protocol versions

The server behaviour exists already, so these tests have no implementation step. Each assertion compares an exact value, so a change in the behaviour fails it.

**Files:**
- Test: `tests/adversarial/sessions/test_adv_protocols.py`
- Modify: `docs/claude/testing.md`

- [ ] **Step 1: Write the test file:**

```python
"""The three MCP protocol versions the server speaks, each agreed over a real pipe.

The server agrees to whichever supported version a client asks for, and answers every
version the same way otherwise. A version it does not speak gets its newest, as the MCP
lifecycle rule asks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from harness import stores
from harness.stdio import McpProcess
from memvara.server.mcp import PROTOCOL_VERSION, SUPPORTED_PROTOCOLS, MemvaraMCPServer


@dataclass(frozen=True)
class Handshake:
    """What one server answered, in the order a client asks."""

    agreed: dict[str, Any]
    tools: list[dict[str, Any]]
    stats: str
    batch: dict[str, Any]
    exit_code: int


@pytest.fixture(scope="module")
def handshakes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Handshake]:
    """One real server for each supported version, each on a new store and asked the
    same things in the same order."""
    home = tmp_path_factory.mktemp("protocol-home")
    found = {}
    for version in SUPPORTED_PROTOCOLS:
        server = McpProcess(tmp_path_factory.mktemp("protocol") / "memory.db", home=home)
        try:
            agreed = server.initialize(version)
            tools = server.list_tools()
            stats = server.call("memory_stats").text
            server.send_raw(json.dumps([{"jsonrpc": "2.0", "id": 90, "method": "ping"}]))
            batch = server.recv()
            server.request("ping")
            code = server.close()
        finally:
            server.kill()
        found[version] = Handshake(agreed, tools, stats, batch, code)
    return found


@pytest.mark.parametrize("version", SUPPORTED_PROTOCOLS)
def test_the_server_agrees_to_each_version_it_supports(
        version: str, handshakes: dict[str, Handshake]) -> None:
    assert handshakes[version].agreed["protocolVersion"] == version
    assert handshakes[version].exit_code == 0


def test_the_versions_differ_only_in_the_version_agreed(
        handshakes: dict[str, Handshake]) -> None:
    """The same capabilities, server information and instructions, the same tools and the
    same tool result under every version: a client on any of the three sees one server."""
    def rest(handshake: Handshake) -> dict[str, Any]:
        return {k: v for k, v in handshake.agreed.items() if k != "protocolVersion"}

    first, *others = SUPPORTED_PROTOCOLS
    for version in others:
        assert rest(handshakes[version]) == rest(handshakes[first])
        assert handshakes[version].tools == handshakes[first].tools
        assert handshakes[version].stats == handshakes[first].stats


@pytest.mark.parametrize("version", SUPPORTED_PROTOCOLS)
def test_a_batch_is_refused_whichever_version_was_agreed(
        version: str, handshakes: dict[str, Handshake]) -> None:
    """JSON-RPC batches belong to revision 2025-03-26 and were removed in 2025-06-18. The
    server refuses a batch under every version, which the design lists as documented
    behaviour ("Batches are refused", docs/superpowers/specs/
    2026-09-25-adversarial-test-suite-design.md). The server keeps answering afterwards,
    because the fixture's ping after the batch was answered."""
    assert handshakes[version].batch == {
        "jsonrpc": "2.0", "id": None,
        "error": {"code": -32600,
                  "message": "expected a single JSON-RPC request object per line"}}


@pytest.mark.parametrize("asked", ["2099-01-01", "2024-10-07", 20250618, None])
def test_a_version_the_server_does_not_speak_gets_its_newest(asked: Any) -> None:
    """MCP's lifecycle rule: a server that does not support the version a client asks for
    answers with one it does, and should answer with its newest. A version that is not a
    string, or no version at all, is treated the same way."""
    params: dict[str, Any] = {"capabilities": {},
                              "clientInfo": {"name": "adversarial", "version": "0"}}
    if asked is not None:
        params["protocolVersion"] = asked
    server = MemvaraMCPServer(stores.memory(), user="tester")
    try:
        reply = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                       "params": params})
    finally:
        server.close()
    assert reply is not None
    assert reply["result"]["protocolVersion"] == PROTOCOL_VERSION == SUPPORTED_PROTOCOLS[0]
```

- [ ] **Step 2: Run it and see it pass**

Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions/test_adv_protocols.py`
Expected: PASS, 11 tests, in about a second.

- [ ] **Step 3: Document it.** Add to "The tool surface":

```markdown
- **The three protocol versions.** `test_adv_protocols.py` agrees each version the server supports, 2025-06-18, 2025-03-26 and 2024-11-05, over a real pipe. The server echoes the version it was asked for, and nothing else differs: the capabilities, the instructions, the tools and a tool's result are the same under all three. A JSON-RPC batch is refused whichever version was agreed, which the design lists as documented behaviour, although batches belong to 2025-03-26. A version the server does not speak, or none at all, gets its newest.
```

The commit comes after Task 6.

---

### Task 6: Read-only cloud credentials

**Files:**
- Test: `tests/adversarial/sessions/test_adv_cloud_read_only.py`
- Modify: `docs/claude/testing.md`

- [ ] **Step 1: Write the test file:**

```python
"""A cloud-mode server whose API key may only read, started over a real pipe against the
FakeV1 fake of the hosted API.

The server asks the deployment about its credential once, at startup, with GET /v1/stats,
and a read-only answer hides every write tool as MEMVARA_READ_ONLY would
(`_service_facts` in memvara/server/mcp.py). A write asked for by name is refused by the
server itself, so the deployment never receives it.
"""

from __future__ import annotations

from typing import Callable

from harness.fakes.fake_v1 import FakeV1
from harness.stdio import McpProcess

from . import switches


def cloud(fake: FakeV1) -> dict[str, str]:
    """The variables that point a server at the fake in cloud mode. The mcp fixture also
    sets MEMVARA_DB, which cloud mode ignores."""
    return {"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": fake.api_key,
            "MEMVARA_SERVER_URL": fake.serve()}


def test_a_read_only_key_lists_only_read_tools_and_refuses_every_write_by_name(
        mcp: Callable[..., McpProcess]) -> None:
    read_only = switches.Combination.of("read_only")
    with FakeV1(read_only=True) as fake:
        server = mcp(env=cloud(fake))
        server.initialize()
        problems = (switches.listing_problems(server, read_only)
                    + switches.refusal_problems(server, read_only,
                                                switches.minimal_calls(switches.NOWHERE)))
        found = server.call("memory_search", query="where does the user live")
        stats = server.call("memory_stats").text
        assert server.close() == 0
    assert problems == []
    assert not found.is_error, found.text
    assert "writes: disabled — this server is read-only" in stats
    routes = {request.route for request in fake.requests}
    assert {"GET /v1/stats", "POST /v1/search"} <= routes
    assert [request.route for request in fake.requests if request.status != 200] == []


def test_a_writable_key_lists_every_tool(mcp: Callable[..., McpProcess]) -> None:
    """The control for the test above: with a key that may write, the same server lists
    the write tools, so the difference comes from the credential."""
    with FakeV1() as fake:
        server = mcp(env=cloud(fake))
        server.initialize()
        problems = switches.listing_problems(server, switches.Combination.of())
        assert server.close() == 0
    assert problems == []
```

- [ ] **Step 2: Run it and see it pass**

Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions/test_adv_cloud_read_only.py`
Expected: PASS, 2 tests, in under a second.

- [ ] **Step 3: Document it.** Add to "The tool surface":

```markdown
- **Read-only cloud credentials.** `test_adv_cloud_read_only.py` starts a server in cloud mode against `FakeV1`. With a key that may only read, the server lists only the read tools, refuses each write tool by name without sending anything to the deployment, and still answers reads. With a key that may write, it lists every tool, which shows that the difference comes from the credential.
```

- [ ] **Step 4: Commit Tasks 5 and 6**

```bash
git add tests/adversarial/sessions/test_adv_protocols.py \
  tests/adversarial/sessions/test_adv_cloud_read_only.py docs/claude/testing.md
git commit -m "Check the three protocol versions and a read-only cloud key over the real pipe" -m "Each supported version is agreed and answered the same way, a batch is refused under every version, and an unsupported version gets the newest. A cloud-mode server whose key may only read lists only the read tools and refuses every write by name without reaching the deployment."
```

---

### Task 7: The session scenario

Scenarios test behaviour that already exists, so they have no implementation step. Their negative control shows that the gold can fail: each file's `test_the_gold_fails_without_memvara` plays it for an agent with no memory. A gold item that fails with memvara is either a mistake in the scenario, fixed in the scenario, or a bug, handled as the global constraints say.

**Files:**
- Create: `tests/scenarios/scripted/session-recall-every-prompt.json`
- Modify: `docs/claude/testing.md`

- [ ] **Step 1: Write the scenario**

```json
{
  "id": "session-recall-every-prompt",
  "description": "An agent's host runs the session-start hook when a conversation opens and the recall hook on every prompt. The user gives a rental door code, which the fast path cannot read, so the agent stores the turn with memory_add and then the exact fact with memory_remember, citing that turn. The user also says where they now work, which memory_add stores on its own. A later prompt in the same session asks for the code, and the recall hook brings it; asked again, the hook does not repeat it. memory_why quotes the sentence the code came from, and the next session opens with both facts. A sentence that names another city does not change where the user lives.",
  "tier": "fast",
  "surfaces": ["stdio", "hooks"],
  "env": {"user": "tester"},
  "seed": [
    {"op": "remember", "predicate": "prefers", "object": "short answers", "memory_type": "procedural"},
    {"op": "remember", "predicate": "lives_in", "object": "Lisbon"}
  ],
  "sessions": [
    {"turns": [
      {"id": "opens", "user": "Morning! Can you help me plan my week?",
       "script": [{"hook": "session_start"}, {"hook": "recall"}]},
      {"id": "door-code", "user": "The door code for our rental in Porto is 7731.",
       "script": [
         {"hook": "recall"},
         {"tool": "memory_add", "args": {"text": "The door code for our rental in Porto is 7731."}, "capture": {"code_turn": "turn id\\(s\\): (ep_[0-9a-f]+)"}},
         {"tool": "memory_remember", "args": {"predicate": "rental_door_code", "object": "7731", "sources": ["{code_turn}"]}, "capture": {"code_id": "\\[(cl_[0-9a-f]+)\\] user rental door code 7731"}}
       ]},
      {"id": "new-job", "user": "Also, I work at Northwind Traders now.",
       "script": [
         {"hook": "recall"},
         {"tool": "memory_add", "args": {"text": "Also, I work at Northwind Traders now."}}
       ]},
      {"id": "asks-code", "user": "What's the rental door code again?",
       "script": [{"hook": "recall"}]},
      {"id": "asks-again", "user": "Sorry, what was the rental door code?",
       "script": [{"hook": "recall"}]},
      {"id": "why", "user": "How do you know the door code?",
       "script": [{"hook": "recall"}, {"tool": "memory_why", "args": {"claim_id": "{code_id}"}}]}
    ]},
    {"turns": [
      {"id": "next-morning", "user": "Morning again. What's on today?",
       "script": [{"hook": "session_start"}]}
    ]}
  ],
  "store_gold": [
    {"id": "employer-stored-by-add", "text": "user works at Northwind Traders", "state": "live", "count": 1},
    {"id": "code-stored-by-remember", "text": "user rental door code 7731", "state": "live", "count": 1},
    {"id": "home-unchanged", "text": "user lives in Lisbon", "state": "live", "count": 1},
    {"id": "no-home-from-the-code-sentence", "text": "user lives in Porto", "state": "absent"}
  ],
  "answer_gold": [
    {"id": "session-start-brings-the-preference", "turn": "opens", "must_contain": "user prefers short answers"},
    {"id": "session-start-brings-the-home", "turn": "opens", "must_contain": "user lives in Lisbon"},
    {"id": "add-reads-the-employer", "turn": "new-job", "must_contain": "user works at Northwind Traders"},
    {"id": "recall-hook-brings-this-sessions-write", "turn": "asks-code", "must_contain": "user rental door code 7731"},
    {"id": "recall-hook-does-not-repeat-it", "turn": "asks-again", "must_not_contain": "7731"},
    {"id": "why-quotes-the-source-turn", "turn": "why", "must_contain": "The door code for our rental in Porto is 7731"},
    {"id": "next-session-brings-the-employer", "turn": "next-morning", "must_contain": "user works at Northwind Traders"},
    {"id": "next-session-brings-the-code", "turn": "next-morning", "must_contain": "user rental door code 7731"}
  ],
  "requires": ["tools", "hooks.session_start", "hooks.recall"],
  "negative_control": true
}
```

- [ ] **Step 2: Play it**

Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions/test_adv_scenarios.py -k session-recall-every-prompt`
Expected: PASS, 15 tests (format, 12 gold, script, negative control).

- [ ] **Step 3: Document it.** Add to the new section of `docs/claude/testing.md`:

```markdown
### The workflow scenarios

Ten scenarios cover the workflows the design lists for this workstream. Each file's `description` says what it checks and why.

| Scenario | What it adds to the fourteen before it |
|---|---|
| `session-recall-every-prompt` | The session-start hook, the recall hook on every prompt, and `memory_add` beside `memory_remember` with `sources` |
```

- [ ] **Step 4: Commit** together with Task 8.

---

### Task 8: The three kinds of correction, in the forms not yet covered

**Files:**
- Create: `tests/scenarios/scripted/correction-end-forms.json`, `correction-retire-forms.json`, `correction-erased-with-sources.json`
- Modify: `docs/claude/testing.md`

- [ ] **Step 1: Write `correction-end-forms.json`**

```json
{
  "id": "correction-end-forms",
  "description": "Four ways to end a fact that was true, in the forms correction-ended does not use. One of two pets dies, and memory_end closes that one value by its claim id, at the day it happened, leaving the other pet and the past alone. The user's taste moves from jazz to blues, and memory_remember with replaces ends the old value in the same write, because likes holds several values at once. A job on a contract that runs out later is written with true_until, and a gym membership is ended at a future instant with memory_end. Both still answer today and no longer answer after their end, and a claim with an end instant set is in the state ended even before that instant arrives (Claim.state in memvara/types.py). memory_history keeps every reason. None of these says the record was wrong, so nothing is retired.",
  "tier": "fast",
  "surfaces": ["stdio"],
  "env": {"user": "tester"},
  "seed": [
    {"op": "remember", "predicate": "member_of", "object": "FitHub gym"}
  ],
  "sessions": [
    {"turns": [
      {"user": "I have a cat called Miso, who I've had since June 2019, and a dog called Pepper, who I've had since March 2021. I like jazz and hiking.",
       "script": [
         {"tool": "memory_remember", "args": {"predicate": "owns_pet", "object": "Miso the cat", "true_since": "2019-06-01"}},
         {"tool": "memory_remember", "args": {"predicate": "owns_pet", "object": "Pepper the dog", "true_since": "2021-03-01"}},
         {"tool": "memory_remember", "args": {"predicate": "likes", "object": "jazz"}},
         {"tool": "memory_remember", "args": {"predicate": "likes", "object": "hiking"}}
       ]}
    ]},
    {"turns": [
      {"user": "Sad news: Pepper died on 10 April.",
       "script": [
         {"tool": "memory_search", "args": {"query": "what pets does the user own"}, "capture": {"pepper_id": "id=(cl_[0-9a-f]+) [^\\]]*\\] user owns pet Pepper the dog"}},
         {"tool": "memory_end", "args": {"claim_id": "{pepper_id}", "at": "2026-04-10", "reason": "Pepper died"}}
       ]},
      {"id": "pets-now", "user": "Which pets do I have now?",
       "script": [{"tool": "memory_recall", "args": {"query": "what pets does the user own"}}]},
      {"id": "pets-2025", "user": "And which pets did I have in June 2025?",
       "script": [{"tool": "memory_search", "args": {"query": "what pets does the user own", "valid_at": "2025-06-01"}}]},
      {"user": "I've gone off jazz. These days I listen to blues instead.",
       "script": [
         {"tool": "memory_search", "args": {"query": "what music does the user like"}, "capture": {"jazz_id": "id=(cl_[0-9a-f]+) [^\\]]*\\] user likes jazz"}},
         {"tool": "memory_remember", "args": {"predicate": "likes", "object": "blues", "replaces": "{jazz_id}", "reason": "the user moved from jazz to blues"}}
       ]},
      {"id": "likes-now", "user": "What do I like these days?",
       "script": [{"tool": "memory_recall", "args": {"query": "what does the user like"}}]},
      {"id": "planned", "user": "I'm working at Fabrikam on a contract that runs out in two months. And I've cancelled FitHub: the membership stops at the end of next month.",
       "script": [
         {"mark": "gym_end", "offset_seconds": 2592000},
         {"mark": "contract_end", "offset_seconds": 5184000},
         {"mark": "after_both", "offset_seconds": 10368000},
         {"tool": "memory_remember", "args": {"predicate": "works_at", "object": "Fabrikam", "true_until": "{contract_end}", "until_reason": "the contract runs out"}},
         {"tool": "memory_end", "args": {"predicate": "member_of", "at": "{gym_end}", "reason": "the user cancelled the membership"}}
       ]},
      {"id": "now", "user": "Where do I work, and which gym am I a member of?",
       "script": [{"tool": "memory_recall", "args": {"query": "where does the user work and which gym is the user a member of"}}]},
      {"id": "later", "user": "Looking four months ahead, what will still hold?",
       "script": [{"tool": "memory_search", "args": {"query": "where does the user work and which gym is the user a member of", "valid_at": "{after_both}"}}]},
      {"id": "history", "user": "Remind me what happened with Pepper, with jazz and with the contract.",
       "script": [
         {"tool": "memory_history", "args": {"predicate": "owns_pet"}},
         {"tool": "memory_history", "args": {"predicate": "likes"}},
         {"tool": "memory_history", "args": {"predicate": "works_at"}}
       ]}
    ]}
  ],
  "store_gold": [
    {"id": "pepper-ended", "text": "user owns pet Pepper the dog", "state": "ended", "count": 1},
    {"id": "pepper-not-retired", "text": "user owns pet Pepper the dog", "state": "retired", "count": 0},
    {"id": "miso-live", "text": "user owns pet Miso the cat", "state": "live", "count": 1},
    {"id": "jazz-ended", "text": "user likes jazz", "state": "ended", "count": 1},
    {"id": "jazz-not-retired", "text": "user likes jazz", "state": "retired", "count": 0},
    {"id": "blues-live", "text": "user likes blues", "state": "live", "count": 1},
    {"id": "hiking-live", "text": "user likes hiking", "state": "live", "count": 1},
    {"id": "contract-ends-later", "text": "user works at Fabrikam", "state": "ended", "count": 1},
    {"id": "gym-ends-later", "text": "user member of FitHub gym", "state": "ended", "count": 1}
  ],
  "answer_gold": [
    {"id": "miso-now", "turn": "pets-now", "must_contain": "Miso the cat"},
    {"id": "pepper-not-now", "turn": "pets-now", "must_not_contain": "Pepper"},
    {"id": "pepper-in-2025", "turn": "pets-2025", "must_contain": "Pepper the dog"},
    {"id": "blues-now", "turn": "likes-now", "must_contain": "blues"},
    {"id": "hiking-now", "turn": "likes-now", "must_contain": "hiking"},
    {"id": "jazz-not-now", "turn": "likes-now", "must_not_contain": "jazz"},
    {"id": "future-end-is-explained", "turn": "planned", "must_contain": "end at an instant still in the future"},
    {"id": "contract-holds-today", "turn": "now", "must_contain": "Fabrikam"},
    {"id": "gym-holds-today", "turn": "now", "must_contain": "FitHub gym"},
    {"id": "contract-over-later", "turn": "later", "must_not_contain": "Fabrikam"},
    {"id": "gym-over-later", "turn": "later", "must_not_contain": "FitHub"},
    {"id": "pepper-reason-kept", "turn": "history", "must_contain": "ended because: Pepper died"},
    {"id": "jazz-reason-kept", "turn": "history", "must_contain": "ended because: the user moved from jazz to blues"},
    {"id": "contract-reason-kept", "turn": "history", "must_contain": "ended because: the contract runs out"}
  ],
  "forbidden": [{"tool": "memory_forget"}],
  "requires": ["tools"],
  "negative_control": true
}
```

- [ ] **Step 2: Write `correction-retire-forms.json`**

```json
{
  "id": "correction-retire-forms",
  "description": "Retiring, the correction that says a record was wrong, in the forms correction-retired does not use. memory_forget with a predicate retires every current value of that fact at once: here two allergies that were the user's sister's, not the user's. With a subject as well, it retires a third party's value and leaves the user's own alone. With a claim id, it retires one value of a fact that holds several and leaves the others. A call that names both a claim id and a predicate is refused, and an id that is not visible here retires nothing. memory_history keeps every retired value with its reason, and recall shows only what is still believed.",
  "tier": "fast",
  "surfaces": ["stdio"],
  "env": {"user": "tester"},
  "seed": [
    {"op": "remember", "predicate": "allergic_to", "object": "shellfish"},
    {"op": "remember", "predicate": "allergic_to", "object": "pollen"},
    {"op": "remember", "predicate": "lives_in", "object": "Braga"},
    {"op": "remember", "subject": "Mei", "predicate": "lives_in", "object": "Porto"}
  ],
  "sessions": [
    {"turns": [
      {"id": "not-mine", "user": "Those allergies you have for me are wrong. Shellfish and pollen are my sister Mei's allergies. I'm not allergic to anything.",
       "script": [
         {"tool": "memory_forget", "args": {"predicate": "allergic_to", "reason": "they are the user's sister Mei's allergies"}},
         {"tool": "memory_remember", "args": {"subject": "Mei", "predicate": "allergic_to", "object": "shellfish"}},
         {"tool": "memory_remember", "args": {"subject": "Mei", "predicate": "allergic_to", "object": "pollen"}}
       ]},
      {"id": "mine-now", "user": "So what am I allergic to?",
       "script": [{"tool": "memory_history", "args": {"predicate": "allergic_to"}}]}
    ]},
    {"turns": [
      {"id": "mei-fix", "user": "One more fix: Mei was never allergic to pollen. It's walnuts.",
       "script": [
         {"tool": "memory_search", "args": {"query": "what is Mei allergic to"}, "capture": {"pollen_id": "id=(cl_[0-9a-f]+) [^\\]]*\\] Mei allergic to pollen"}},
         {"tool": "memory_forget", "args": {"claim_id": "{pollen_id}", "predicate": "allergic_to"}, "expect_error": true},
         {"tool": "memory_forget", "args": {"claim_id": "{pollen_id}", "reason": "Mei's allergy is walnuts, not pollen"}},
         {"tool": "memory_remember", "args": {"subject": "Mei", "predicate": "allergic_to", "object": "walnuts"}}
       ]},
      {"id": "mei-city", "user": "And Mei doesn't live in Porto. You've mixed her up with someone else.",
       "script": [{"tool": "memory_forget", "args": {"subject": "Mei", "predicate": "lives_in", "reason": "mixed up with someone else"}}]},
      {"id": "unknown-id", "user": "Forget the other allergy note too, the one with that old id.",
       "script": [{"tool": "memory_forget", "args": {"claim_id": "cl_00000000000000000000"}}]},
      {"id": "mei-now", "user": "What is Mei allergic to?",
       "script": [{"tool": "memory_history", "args": {"subject": "Mei", "predicate": "allergic_to"}}]},
      {"id": "party", "user": "What allergies should I plan for at Mei's party?",
       "script": [{"tool": "memory_recall", "args": {"query": "Mei allergic to"}}]},
      {"id": "where", "user": "Where do I live, and where does Mei live?",
       "script": [{"tool": "memory_recall", "args": {"query": "where does the user live and where does Mei live"}}]}
    ]}
  ],
  "store_gold": [
    {"id": "user-shellfish-retired", "text": "user allergic to shellfish", "state": "retired", "count": 1},
    {"id": "user-pollen-retired", "text": "user allergic to pollen", "state": "retired", "count": 1},
    {"id": "user-allergies-not-ended", "text": "user allergic to shellfish", "state": "ended", "count": 0},
    {"id": "mei-shellfish-live", "text": "Mei allergic to shellfish", "state": "live", "count": 1},
    {"id": "mei-pollen-retired", "text": "Mei allergic to pollen", "state": "retired", "count": 1},
    {"id": "mei-walnuts-live", "text": "Mei allergic to walnuts", "state": "live", "count": 1},
    {"id": "mei-city-retired", "text": "Mei lives in Porto", "state": "retired", "count": 1},
    {"id": "user-city-kept", "text": "user lives in Braga", "state": "live", "count": 1}
  ],
  "answer_gold": [
    {"id": "one-call-retires-both", "turn": "not-mine", "must_contain": "Retired 2 value(s) of user/allergic_to"},
    {"id": "history-keeps-the-reason", "turn": "mine-now", "must_contain": "retired because: they are the user's sister Mei's allergies"},
    {"id": "both-addresses-refused", "turn": "mei-fix", "must_contain": "memory_forget needs exactly one of"},
    {"id": "third-party-retired-alone", "turn": "mei-city", "must_contain": "Retired 1 value(s) of Mei/lives_in"},
    {"id": "unknown-id-retires-nothing", "turn": "unknown-id", "must_contain": "Nothing retired"},
    {"id": "one-value-retired-with-its-reason", "turn": "mei-now", "must_contain": "retired because: Mei's allergy is walnuts, not pollen"},
    {"id": "party-walnuts", "turn": "party", "must_contain": "Mei allergic to walnuts"},
    {"id": "party-shellfish", "turn": "party", "must_contain": "Mei allergic to shellfish"},
    {"id": "party-not-pollen", "turn": "party", "must_not_contain": "pollen"},
    {"id": "user-lives-in-braga", "turn": "where", "must_contain": "user lives in Braga"},
    {"id": "mei-not-in-porto", "turn": "where", "must_not_contain": "Porto"}
  ],
  "forbidden": [{"tool": "memory_end"}, {"tool": "memory_forget_matching"}],
  "requires": ["tools"],
  "negative_control": true
}
```

- [ ] **Step 3: Write `correction-erased-with-sources.json`**

```json
{
  "id": "correction-erased-with-sources",
  "description": "The user asks for their passport number to be deleted outright, including the sentence they gave it in. Unlike correction-erased, this fact has a source turn: memory_add stored the sentence, and memory_remember cited it. No tool can erase a memory, so the operator erases the claim with its source turns. From then on no read shows the number or the sentence, in the same session or the next one, an anchored search finds nothing, and memory_why cannot find the claim. A locker code the user gave in another turn, with a source turn of its own, is untouched.",
  "tier": "fast",
  "surfaces": ["stdio"],
  "env": {"user": "tester"},
  "sessions": [
    {"turns": [
      {"user": "For the visa form: my passport number is K4471902.",
       "script": [
         {"tool": "memory_add", "args": {"text": "For the visa form: my passport number is K4471902."}, "capture": {"passport_turn": "turn id\\(s\\): (ep_[0-9a-f]+)"}},
         {"tool": "memory_remember", "args": {"predicate": "passport_number", "object": "K4471902", "sources": ["{passport_turn}"]}, "capture": {"passport_id": "\\[(cl_[0-9a-f]+)\\] user passport number K4471902"}}
       ]},
      {"user": "And my gym locker code is 5521.",
       "script": [
         {"tool": "memory_add", "args": {"text": "And my gym locker code is 5521."}, "capture": {"locker_turn": "turn id\\(s\\): (ep_[0-9a-f]+)"}},
         {"tool": "memory_remember", "args": {"predicate": "locker_code", "object": "5521", "sources": ["{locker_turn}"]}, "capture": {"locker_id": "\\[(cl_[0-9a-f]+)\\] user locker code 5521"}}
       ]},
      {"id": "before", "user": "Where did my passport number come from?",
       "script": [{"tool": "memory_why", "args": {"claim_id": "{passport_id}"}}]}
    ]},
    {"turns": [
      {"user": "The visa is done. Delete my passport number completely, including what I said when I gave it to you.",
       "script": [{"op": "erase", "claim_id": "{passport_id}", "sources": true}]},
      {"id": "same-session", "user": "Is it gone?",
       "script": [{"tool": "memory_recall", "args": {"query": "passport number for the visa form", "include_episodes": true}}]},
      {"id": "same-session-anchored", "user": "Is anything about my passport left at all?",
       "script": [{"tool": "memory_search", "args": {"query": "passport number K4471902", "anchored": true}}]}
    ]},
    {"turns": [
      {"id": "next-session", "user": "What's my passport number?",
       "script": [
         {"tool": "memory_recall", "args": {"query": "what is the user's passport number for the visa form", "include_episodes": true}},
         {"tool": "memory_history", "args": {"predicate": "passport_number"}}
       ]},
      {"id": "why", "user": "Where did you get my passport number?",
       "script": [{"tool": "memory_why", "args": {"claim_id": "{passport_id}"}}]},
      {"id": "locker", "user": "What's my gym locker code?",
       "script": [
         {"tool": "memory_recall", "args": {"query": "gym locker code", "include_episodes": true}},
         {"tool": "memory_why", "args": {"claim_id": "{locker_id}"}}
       ]}
    ]}
  ],
  "store_gold": [
    {"id": "passport-erased", "text": "user passport number K4471902", "state": "absent"},
    {"id": "locker-kept", "text": "user locker code 5521", "state": "live", "count": 1}
  ],
  "answer_gold": [
    {"id": "source-shown-before", "turn": "before", "must_contain": "For the visa form: my passport number is K4471902"},
    {"id": "number-gone-in-the-same-session", "turn": "same-session", "must_not_contain": "K4471902"},
    {"id": "sentence-gone-in-the-same-session", "turn": "same-session", "must_not_contain": "visa form"},
    {"id": "anchored-search-finds-nothing", "turn": "same-session-anchored", "abstain": true},
    {"id": "number-gone-in-the-next-session", "turn": "next-session", "must_not_contain": "K4471902"},
    {"id": "sentence-gone-in-the-next-session", "turn": "next-session", "must_not_contain": "visa form"},
    {"id": "why-finds-nothing", "turn": "why", "must_contain": "is not visible here"},
    {"id": "why-shows-no-number", "turn": "why", "must_not_contain": "K4471902"},
    {"id": "locker-code-kept", "turn": "locker", "must_contain": "user locker code 5521"},
    {"id": "locker-source-kept", "turn": "locker", "must_contain": "And my gym locker code is 5521"}
  ],
  "requires": ["tools"],
  "negative_control": true
}
```

- [ ] **Step 4: Play all three**

Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions/test_adv_scenarios.py -k "correction-end-forms or correction-retire-forms or correction-erased-with-sources"`
Expected: PASS. A failure in `correction-erased-with-sources` that shows erased text is a security-class finding: remove the file from the change and report it privately.

- [ ] **Step 5: Document them.** Add three rows to the table:

```markdown
| `correction-end-forms` | Ending one value by claim id, `replaces`, `true_until`, and an end at a future instant |
| `correction-retire-forms` | Retiring every value of a fact at once, a third party's value, and one value of several |
| `correction-erased-with-sources` | Erasing a claim together with the turn it came from |
```

- [ ] **Step 6: Commit Tasks 7 and 8**

```bash
git add tests/scenarios/scripted/session-recall-every-prompt.json \
  tests/scenarios/scripted/correction-end-forms.json \
  tests/scenarios/scripted/correction-retire-forms.json \
  tests/scenarios/scripted/correction-erased-with-sources.json docs/claude/testing.md
git commit -m "Add scenarios for recalling on every prompt and for ending, retiring and erasing" -m "One scenario runs the session-start hook and the recall hook on every prompt while the agent stores facts with memory_add and memory_remember. Three more cover the forms of ending, retiring and erasing that the first correction scenarios leave out."
```

---

### Task 9: Time travel, documents and confirmation tokens

**Files:**
- Create: `tests/scenarios/scripted/time-travel-three-readings.json`, `document-lifecycle.json`, `bulk-end-tokens.json`
- Modify: `docs/claude/testing.md`

- [ ] **Step 1: Write `time-travel-three-readings.json`**

```json
{
  "id": "time-travel-three-readings",
  "description": "A client's renewal date is recorded wrongly and corrected later, and the store is read on each of its two clocks. memory_ask at the instant the wrong date was recorded gives three readings: what is now known to have been true then, what the store would have said then, and a note that the record was corrected since. The second reading is the known_at reading, which no other tool offers. memory_recall with valid_at gives a block dated to that day, memory_since lists the replacement in both of its lists, and memory_search with as_of shows what was believed then. memory_recall refuses as_of, and memory_search refuses as_of and valid_at together, and each refusal says why.",
  "tier": "fast",
  "surfaces": ["stdio"],
  "env": {"user": "tester"},
  "sessions": [
    {"turns": [
      {"user": "Our client Contoso signed in April 2025, and their contract renews on 31 March 2027.",
       "script": [
         {"tool": "memory_remember", "args": {"subject": "Contoso", "predicate": "renewal_date", "object": "31 March 2027", "true_since": "2025-04-01"}},
         {"mark": "told_then"}
       ]}
    ]},
    {"turns": [
      {"user": "I misread the contract. Contoso renews on 30 June 2027, and always did.",
       "script": [
         {"tool": "memory_search", "args": {"query": "when does the Contoso contract renew"}, "capture": {"march_id": "id=(cl_[0-9a-f]+) [^\\]]*\\] Contoso renewal date 31 March 2027"}},
         {"tool": "memory_forget", "args": {"claim_id": "{march_id}", "reason": "misread the contract: it renews on 30 June"}},
         {"tool": "memory_remember", "args": {"subject": "Contoso", "predicate": "renewal_date", "object": "30 June 2027", "true_since": "2025-04-01"}}
       ]},
      {"id": "ask-then", "user": "When I first told you, what renewal date did you have for Contoso, and what is the real one?",
       "script": [{"tool": "memory_ask", "args": {"question": "when does the Contoso contract renew", "at": "{told_then}"}}]},
      {"id": "recall-dated", "user": "As of June 2025, when was Contoso due to renew?",
       "script": [{"tool": "memory_recall", "args": {"query": "when does the Contoso contract renew", "valid_at": "2025-06-01"}}]},
      {"id": "before-signing", "user": "Did Contoso have a renewal date back in January 2025?",
       "script": [{"tool": "memory_ask", "args": {"question": "when does the Contoso contract renew", "at": "2025-01-15"}}]},
      {"id": "what-changed", "user": "What has changed in my notes since I first told you about Contoso?",
       "script": [{"tool": "memory_since", "args": {"since": "{told_then}"}}]},
      {"id": "recall-as-of", "user": "Just read me back what you believed then.",
       "script": [{"tool": "memory_recall", "args": {"query": "when does the Contoso contract renew", "as_of": "{told_then}"}, "expect_error": true}]},
      {"id": "both-clocks", "user": "What did you believe then about June 2025?",
       "script": [{"tool": "memory_search", "args": {"query": "when does the Contoso contract renew", "as_of": "{told_then}", "valid_at": "2025-06-01"}, "expect_error": true}]},
      {"id": "believed-then", "user": "So what exactly did you believe then?",
       "script": [{"tool": "memory_search", "args": {"query": "when does the Contoso contract renew", "as_of": "{told_then}"}}]}
    ]}
  ],
  "store_gold": [
    {"id": "march-retired", "text": "Contoso renewal date 31 March 2027", "state": "retired", "count": 1},
    {"id": "june-live", "text": "Contoso renewal date 30 June 2027", "state": "live", "count": 1}
  ],
  "answer_gold": [
    {"id": "ask-true-then", "turn": "ask-then", "must_contain": "Contoso renewal_date: 30 June 2027"},
    {"id": "ask-would-have-said", "turn": "ask-then", "must_contain": "this store would have said 31 March 2027"},
    {"id": "ask-names-the-correction", "turn": "ask-then", "must_contain": "it means the record was corrected after that instant"},
    {"id": "recall-says-which-day", "turn": "recall-dated", "must_contain": "as things were on 1 June 2025"},
    {"id": "recall-dated-june", "turn": "recall-dated", "must_contain": "30 June 2027"},
    {"id": "recall-dated-not-march", "turn": "recall-dated", "must_not_contain": "31 March 2027"},
    {"id": "nothing-before-signing", "turn": "before-signing", "must_contain": "nothing was true on 2025-01-15"},
    {"id": "since-counts-both", "turn": "what-changed", "must_contain": "1 arrived and 1 left"},
    {"id": "since-marks-what-left", "turn": "what-changed", "must_contain": "Believed then, not believed now"},
    {"id": "recall-refuses-as-of", "turn": "recall-as-of", "must_contain": "memory_recall takes valid_at, not as_of"},
    {"id": "search-refuses-both", "turn": "both-clocks", "must_contain": "memory_search takes as_of or valid_at, not both"},
    {"id": "believed-march", "turn": "believed-then", "must_contain": "31 March 2027"},
    {"id": "believed-not-june", "turn": "believed-then", "must_not_contain": "30 June 2027"}
  ],
  "forbidden": [{"tool": "memory_end"}],
  "requires": ["tools"],
  "negative_control": true
}
```

- [ ] **Step 2: Write `document-lifecycle.json`**

```json
{
  "id": "document-lifecycle",
  "description": "One document through its whole life: added with metadata, looked up by its custom_id, sent again with the same custom_id after it changed, listed by folder, looked up by its id, and deleted by its id. Sending it again updates the document in place, so it keeps its id, the folder lists one document, and a read finds the new text and not the old. Deleting erases its text from every read, and deleting it a second time deletes nothing.",
  "tier": "fast",
  "surfaces": ["stdio"],
  "env": {"user": "tester"},
  "workspace": {"files": {"handbook/onboarding.md": "# Onboarding\n\nNew hires get a laptop on day one. Badges are issued by the facilities desk on the ground floor.\n"}},
  "sessions": [
    {"turns": [
      {"user": "Keep our onboarding guide in memory. It's handbook/onboarding.md, and it belongs to the people team.",
       "script": [
         {"tool": "memory_add_document", "args": {"content": "{file:handbook/onboarding.md}", "custom_id": "handbook/onboarding.md", "title": "Onboarding guide", "filepath": "handbook/onboarding.md", "mime": "text/markdown", "metadata": {"team": "people"}}, "capture": {"doc_id": "document (doc_[0-9a-f]+):"}}
       ]},
      {"id": "check", "user": "Did that save properly?",
       "script": [{"tool": "memory_get_document", "args": {"id": "handbook/onboarding.md"}}]}
    ]},
    {"turns": [
      {"user": "The guide changed. Here is the new text: laptops now arrive during the first week.",
       "script": [
         {"tool": "memory_add_document", "args": {"content": "# Onboarding\n\nNew hires get a laptop during their first week. Badges are issued by the facilities desk on the ground floor.\n", "custom_id": "handbook/onboarding.md", "title": "Onboarding guide", "filepath": "handbook/onboarding.md", "mime": "text/markdown"}}
       ]},
      {"id": "listed", "user": "Which documents do you have in the handbook folder?",
       "script": [{"tool": "memory_list_documents", "args": {"filepath_prefix": "handbook/"}}]},
      {"id": "other-folder", "user": "Anything in the policies folder?",
       "script": [{"tool": "memory_list_documents", "args": {"filepath_prefix": "policies/"}}]},
      {"id": "after-update", "user": "When do new hires get their laptop?",
       "script": [{"tool": "memory_recall", "args": {"query": "when do new hires get a laptop", "include_episodes": true}}]},
      {"id": "by-id", "user": "Is it still the same document?",
       "script": [{"tool": "memory_get_document", "args": {"id": "{doc_id}"}}]},
      {"id": "delete", "user": "We're retiring that guide. Delete it.",
       "script": [{"tool": "memory_delete_document", "args": {"id": "{doc_id}"}}]},
      {"id": "delete-again", "user": "Delete it again, just to be sure.",
       "script": [{"tool": "memory_delete_document", "args": {"id": "{doc_id}"}}]},
      {"id": "after-delete", "user": "When do new hires get their laptop, according to the guide?",
       "script": [
         {"tool": "memory_get_document", "args": {"id": "handbook/onboarding.md"}},
         {"tool": "memory_recall", "args": {"query": "when do new hires get a laptop", "include_episodes": true}},
         {"tool": "memory_list_documents"}
       ]}
    ]}
  ],
  "store_gold": [],
  "answer_gold": [
    {"id": "stored-with-its-title", "turn": "check", "must_contain": "title: Onboarding guide"},
    {"id": "stored-with-its-metadata", "turn": "check", "must_contain": "metadata: team=people"},
    {"id": "update-keeps-one-document", "turn": "listed", "must_contain": "1 document(s), newest first"},
    {"id": "other-folder-is-empty", "turn": "other-folder", "abstain": true},
    {"id": "update-serves-the-new-text", "turn": "after-update", "must_contain": "New hires get a laptop during their first week"},
    {"id": "update-drops-the-old-text", "turn": "after-update", "must_not_contain": "on day one"},
    {"id": "update-keeps-the-id", "turn": "by-id", "must_contain": "custom_id: handbook/onboarding.md"},
    {"id": "delete-erases-the-text", "turn": "delete", "must_contain": "its text is erased"},
    {"id": "second-delete-deletes-nothing", "turn": "delete-again", "must_contain": "Nothing deleted"},
    {"id": "nothing-after-delete", "turn": "after-delete", "abstain": true}
  ],
  "requires": ["tools"],
  "negative_control": true
}
```

- [ ] **Step 3: Write `bulk-end-tokens.json`.** Its expired token is `memvara.confirm.Confirmer("scenario-confirm-secret-not-for-production").issue(["cl_0f0f0f0f0f0f0f0f0f0f"], "ended", now=datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc))[0]`, which expired at 2026-08-20T09:10:00Z.

```json
{
  "id": "bulk-end-tokens",
  "description": "memory_end_matching closes a group of facts in two calls, and every way its confirmation token can be wrong is refused with nothing changed. A token whose listed facts changed after the preview is refused, and the next preview lists only what is still live. A token that memory_forget_matching issued is refused by memory_end_matching. A token used a second time is refused. A token that expired last month is refused; it was minted with memvara.confirm.Confirmer under this scenario's confirm_secret for an instant in August 2026, so only its expiry is wrong. The first two servers share that secret, so a preview from one session is confirmed in the next. The third server has a new secret, so a token from before the change is refused as not issued by this server. The ended facts still answer about the time before they ended. The previews name k, because a preview has no relevance floor and fills k with loose matches (see the tool's description).",
  "tier": "fast",
  "surfaces": ["stdio"],
  "env": {"user": "tester", "confirm_secret": "scenario-confirm-secret-not-for-production"},
  "seed": [
    {"op": "remember", "subject": "Project Kestrel", "predicate": "status", "object": "in beta"},
    {"op": "remember", "subject": "Project Kestrel", "predicate": "owner", "object": "Dana Whitfield"},
    {"op": "remember", "subject": "Project Kestrel", "predicate": "beta_testers", "object": "40"},
    {"op": "remember", "subject": "Nightjar", "predicate": "status", "object": "planning"},
    {"op": "remember", "predicate": "lives_in", "object": "Lisbon"}
  ],
  "sessions": [
    {"turns": [
      {"id": "preview", "user": "Kestrel shipped last week, so its beta details are over. Please close them out.",
       "script": [
         {"mark": "while_in_beta"},
         {"tool": "memory_end_matching", "args": {"query": "Project Kestrel", "k": 3, "reason": "Kestrel shipped"}, "capture": {"stale_token": "^confirm: (\\S+)$"}}
       ]},
      {"user": "Oh, and Dana has just handed Kestrel over to someone else.",
       "script": [
         {"tool": "memory_end", "args": {"subject": "Project Kestrel", "predicate": "owner", "reason": "Dana handed Kestrel over"}}
       ]},
      {"id": "stale", "user": "Right, go ahead and close the Kestrel ones.",
       "script": [
         {"tool": "memory_end_matching", "args": {"confirm": "{stale_token}", "reason": "Kestrel shipped"}, "expect_error": true}
       ]},
      {"id": "fresh-preview", "user": "OK, show me the list again.",
       "script": [
         {"tool": "memory_end_matching", "args": {"query": "Project Kestrel", "k": 2, "reason": "Kestrel shipped"}, "capture": {"end_token": "^confirm: (\\S+)$"}}
       ]},
      {"user": "And forget everything about Nightjar. There never was such a project; I misspoke.",
       "script": [
         {"tool": "memory_forget_matching", "args": {"query": "Nightjar", "k": 1}, "capture": {"forget_token": "^confirm: (\\S+)$"}}
       ]}
    ]},
    {"turns": [
      {"id": "confirm", "user": "Yes, close the Kestrel ones now. Hold off on Nightjar for the moment.",
       "script": [
         {"tool": "memory_end_matching", "args": {"confirm": "{forget_token}", "reason": "Kestrel shipped"}, "expect_error": true},
         {"tool": "memory_end_matching", "args": {"confirm": "{end_token}", "reason": "Kestrel shipped"}}
       ]},
      {"id": "reused", "user": "Did that go through? Try it once more to be sure.",
       "script": [
         {"tool": "memory_end_matching", "args": {"confirm": "{end_token}"}, "expect_error": true}
       ]},
      {"id": "expired", "user": "I found an older confirm token in last month's notes. Try that one too.",
       "script": [
         {"tool": "memory_end_matching", "args": {"confirm": "eyJjbG9zZSI6ImVuZGVkIiwiZXhwaXJlcyI6MTc4NzIxNzAwMCwiaWRzIjpbImNsXzBmMGYwZjBmMGYwZjBmMGYwZjBmIl19.47ff12d38273c4a8871a192b62ec94369f815fff38eb88df1ef5a5f985cbf102"}, "expect_error": true}
       ]},
      {"id": "history", "user": "Why is Kestrel's status closed?",
       "script": [{"tool": "memory_history", "args": {"subject": "Project Kestrel", "predicate": "status"}}]},
      {"id": "in-beta-then", "user": "What was Kestrel's status when we started talking today?",
       "script": [{"tool": "memory_search", "args": {"query": "Project Kestrel status", "valid_at": "{while_in_beta}"}}]}
    ]},
    {"env": {"confirm_secret": "rotated-scenario-secret-not-for-production"}, "turns": [
      {"id": "rotated", "user": "OK, go ahead and forget Nightjar now.",
       "script": [
         {"tool": "memory_forget_matching", "args": {"confirm": "{forget_token}", "reason": "there never was a Nightjar project"}, "expect_error": true},
         {"tool": "memory_forget_matching", "args": {"query": "Nightjar", "k": 1}, "capture": {"new_token": "^confirm: (\\S+)$"}},
         {"tool": "memory_forget_matching", "args": {"confirm": "{new_token}", "reason": "there never was a Nightjar project"}}
       ]}
    ]}
  ],
  "store_gold": [
    {"id": "status-ended", "text": "Project Kestrel status in beta", "state": "ended", "count": 1},
    {"id": "status-not-retired", "text": "Project Kestrel status in beta", "state": "retired", "count": 0},
    {"id": "testers-ended", "text": "Project Kestrel beta testers 40", "state": "ended", "count": 1},
    {"id": "owner-ended", "text": "Project Kestrel owner Dana Whitfield", "state": "ended", "count": 1},
    {"id": "nightjar-retired", "text": "Nightjar status planning", "state": "retired", "count": 1},
    {"id": "home-untouched", "text": "user lives in Lisbon", "state": "live", "count": 1}
  ],
  "answer_gold": [
    {"id": "preview-changes-nothing", "turn": "preview", "must_contain": "Nothing has changed yet"},
    {"id": "preview-leaves-out-the-home", "turn": "preview", "must_not_contain": "Lisbon"},
    {"id": "preview-leaves-out-nightjar", "turn": "preview", "must_not_contain": "Nightjar"},
    {"id": "stale-token-refused", "turn": "stale", "must_contain": "is now ended. Nothing was changed"},
    {"id": "stale-confirm-closed-nothing", "turn": "fresh-preview", "must_contain": "Preview: 2 live match(es)"},
    {"id": "wrong-closure-refused", "turn": "confirm", "must_contain": "this token was issued to retire memories"},
    {"id": "confirmed-after-a-restart", "turn": "confirm", "must_contain": "Ended 2 value(s), with the reason 'Kestrel shipped'"},
    {"id": "reused-token-refused", "turn": "reused", "must_contain": "is now ended. Nothing was changed"},
    {"id": "expired-token-refused", "turn": "expired", "must_contain": "this confirmation token expired at"},
    {"id": "history-keeps-the-reason", "turn": "history", "must_contain": "ended because: Kestrel shipped"},
    {"id": "past-still-answers", "turn": "in-beta-then", "must_contain": "Project Kestrel status in beta"},
    {"id": "new-secret-refuses-an-old-token", "turn": "rotated", "must_contain": "was not issued by this memory server"},
    {"id": "new-preview-retires-nightjar", "turn": "rotated", "must_contain": "Retired 1 value(s), with the reason 'there never was a Nightjar project'"}
  ],
  "forbidden": [{"tool": "memory_forget"}],
  "requires": ["tools"],
  "negative_control": true
}
```

- [ ] **Step 4: Play all three**

Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions/test_adv_scenarios.py -k "time-travel-three-readings or document-lifecycle or bulk-end-tokens"`
Expected: PASS.

- [ ] **Step 5: Document them.** Add three rows:

```markdown
| `time-travel-three-readings` | `memory_ask`'s known_at reading, `valid_at` on recall, `memory_since`, and the refusals of `as_of` on recall and of both clocks on search |
| `document-lifecycle` | Getting, updating in place, listing by folder and deleting a document |
| `bulk-end-tokens` | `memory_end_matching`, and a stale, a wrong, a reused, an expired and a foreign confirmation token |
```

- [ ] **Step 6: Commit** together with Task 10.

---

### Task 10: The graph tools

**Files:**
- Create: `tests/scenarios/scripted/graph-walk.json`
- Modify: `docs/claude/testing.md`

- [ ] **Step 1: Write the scenario**

```json
{
  "id": "graph-walk",
  "description": "The graph tools over a small map of services. The walk follows only relations a vocabulary declares as graph edges, and no built-in predicate is one, so this scenario loads the engineering vocabulary, where depends_on, owner and current_host (spelled runs_on here) are. memory_neighborhood returns one-hop and two-hop chains, and min_hops leaves the one-hop chain out. memory_paths finds the route to today's host, finds the route to last year's host only at a date when it held, and otherwise says that nothing stored connects the two rather than that they are unrelated. memory_link records that one fact was inferred from another and memory_why shows the link, while a link from a fact to itself is refused and a link to an id that is not visible links nothing. The next session finds the route under other spellings of both names. Loading a vocabulary needs Python 3.11, so this scenario skips on 3.10.",
  "tier": "fast",
  "surfaces": ["stdio"],
  "env": {"user": "tester", "predicates": "engineering"},
  "sessions": [
    {"turns": [
      {"user": "Some architecture notes. checkout-web has depended on payments-api since January 2024, and the Payments team has owned payments-api since then too. payments-api ran on host-2 until June 2025 and has run on host-7 since.",
       "script": [
         {"tool": "memory_remember", "args": {"subject": "checkout-web", "predicate": "depends_on", "object": "payments-api", "true_since": "2024-01-01"}},
         {"tool": "memory_remember", "args": {"subject": "payments-api", "predicate": "owner", "object": "Payments team", "true_since": "2024-01-01"}, "capture": {"owner_id": "\\[(cl_[0-9a-f]+)\\] payments-api owner Payments team"}},
         {"tool": "memory_remember", "args": {"subject": "payments-api", "predicate": "runs_on", "object": "host-2", "true_since": "2024-01-01", "true_until": "2025-06-01"}},
         {"tool": "memory_remember", "args": {"subject": "payments-api", "predicate": "runs_on", "object": "host-7", "true_since": "2025-06-01"}}
       ]},
      {"id": "around", "user": "What does checkout-web touch?",
       "script": [{"tool": "memory_neighborhood", "args": {"entity": "checkout-web"}}]},
      {"id": "two-hops", "user": "Only the indirect ones, please.",
       "script": [{"tool": "memory_neighborhood", "args": {"entity": "checkout-web", "min_hops": 2}}]},
      {"id": "route-now", "user": "How does checkout-web reach host-7?",
       "script": [{"tool": "memory_paths", "args": {"source": "checkout-web", "target": "host-7"}}]},
      {"id": "route-then", "user": "How did checkout-web reach host-2 in January 2025?",
       "script": [{"tool": "memory_paths", "args": {"source": "checkout-web", "target": "host-2", "valid_at": "2025-01-15"}}]},
      {"id": "route-gone", "user": "And does it reach host-2 today?",
       "script": [{"tool": "memory_paths", "args": {"source": "checkout-web", "target": "host-2"}}]},
      {"id": "link", "user": "checkout-web's on-call goes to the Payments team because they own payments-api. Record that, and where it came from.",
       "script": [
         {"tool": "memory_remember", "args": {"subject": "checkout-web", "predicate": "oncall_team", "object": "Payments team"}, "capture": {"oncall_id": "\\[(cl_[0-9a-f]+)\\] checkout-web oncall team Payments team"}},
         {"tool": "memory_link", "args": {"from_id": "{oncall_id}", "to_id": "{owner_id}", "relation": "derives"}},
         {"tool": "memory_link", "args": {"from_id": "{oncall_id}", "to_id": "{oncall_id}", "relation": "extends"}, "expect_error": true},
         {"tool": "memory_link", "args": {"from_id": "{oncall_id}", "to_id": "cl_00000000000000000000", "relation": "extends"}}
       ]},
      {"id": "why", "user": "Why does the Payments team get checkout-web's pages?",
       "script": [{"tool": "memory_why", "args": {"claim_id": "{oncall_id}"}}]}
    ]},
    {"turns": [
      {"id": "next-session", "user": "Remind me: what connects Checkout-Web to the payments team?",
       "script": [{"tool": "memory_paths", "args": {"source": "Checkout-Web", "target": "payments team"}}]}
    ]}
  ],
  "store_gold": [
    {"id": "dependency-live", "text": "checkout-web depends on payments-api", "state": "live", "count": 1},
    {"id": "new-host-live", "text": "payments-api current host host-7", "state": "live", "count": 1},
    {"id": "old-host-ended", "text": "payments-api current host host-2", "state": "ended", "count": 1}
  ],
  "answer_gold": [
    {"id": "one-hop-chain", "turn": "around", "must_contain": "checkout-web -depends_on-> payments-api"},
    {"id": "two-hop-chain-to-the-owner", "turn": "around", "must_contain": "checkout-web -depends_on-> payments-api -owner-> Payments team"},
    {"id": "no-chain-to-the-old-host-today", "turn": "around", "must_not_contain": "host-2"},
    {"id": "min-hops-drops-the-one-hop-chain", "turn": "two-hops", "must_not_match": "\\[1 hop\\(s\\)"},
    {"id": "route-to-todays-host", "turn": "route-now", "must_contain": "checkout-web -depends_on-> payments-api -current_host-> host-7"},
    {"id": "route-to-last-years-host-then", "turn": "route-then", "must_contain": "checkout-web -depends_on-> payments-api -current_host-> host-2"},
    {"id": "dated-route-says-when", "turn": "route-then", "must_contain": "as true on 2025-01-15"},
    {"id": "no-route-says-nothing-connects-them", "turn": "route-gone", "must_contain": "say nothing stored connects them"},
    {"id": "link-means-inferred-from", "turn": "link", "must_contain": "meaning the first was inferred from the second"},
    {"id": "self-link-refused", "turn": "link", "must_contain": "to itself"},
    {"id": "invisible-id-links-nothing", "turn": "link", "must_contain": "one of the two ids is not visible here"},
    {"id": "why-shows-the-link", "turn": "why", "must_contain": "this derives"},
    {"id": "why-names-the-source-fact", "turn": "why", "must_contain": "payments-api owner Payments team"},
    {"id": "next-session-folds-the-names", "turn": "next-session", "must_contain": "checkout-web -depends_on-> payments-api -owner-> Payments team"}
  ],
  "requires": ["tools"],
  "negative_control": true
}
```

- [ ] **Step 2: Play it**

Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions/test_adv_scenarios.py -k graph-walk`
Expected: PASS on Python 3.11 and later; on 3.10, every test that plays it skips with the `tomllib` reason, and the format test and the negative control pass.

- [ ] **Step 3: Document it.** Add a row:

```markdown
| `graph-walk` | `memory_neighborhood`, `memory_paths` and `memory_link`, over a vocabulary whose relations the walk can follow |
```

- [ ] **Step 4: Commit Tasks 9 and 10**

```bash
git add tests/scenarios/scripted/time-travel-three-readings.json \
  tests/scenarios/scripted/document-lifecycle.json tests/scenarios/scripted/bulk-end-tokens.json \
  tests/scenarios/scripted/graph-walk.json docs/claude/testing.md
git commit -m "Add scenarios for time travel, documents, confirmation tokens and the graph tools"
```

---

### Task 11: The profile, and expiry and read-only mode together

**Files:**
- Create: `tests/scenarios/scripted/profile-session-start.json`, `expiry-off-and-read-only.json`
- Modify: `docs/claude/testing.md`

- [ ] **Step 1: Write `profile-session-start.json`**

```json
{
  "id": "profile-session-start",
  "description": "memory_profile at the start of a session: the standing preferences, what arrived recently, what is relevant to the session's topic, and buckets the caller names. The buckets here name predicates the store already uses, so the profile reads the same on Python 3.10, where the default buckets cannot be read because they come from predicate packs. A bucket that names a predicate nothing uses is reported as ignored. After the user withdraws a preference, the next profile no longer lists it, and a since instant limits the recent section to what arrived after it.",
  "tier": "fast",
  "surfaces": ["stdio"],
  "env": {"user": "tester"},
  "seed": [
    {"op": "remember", "predicate": "prefers", "object": "short answers", "memory_type": "procedural"},
    {"op": "remember", "predicate": "prefers", "object": "metric units", "memory_type": "procedural"},
    {"op": "remember", "subject": "billing-service", "predicate": "depends_on", "object": "Postgres"},
    {"op": "remember", "predicate": "decided", "object": "move billing to Postgres 16"}
  ],
  "sessions": [
    {"turns": [
      {"id": "opens", "user": "Let's pick up the billing work.",
       "script": [{"tool": "memory_profile", "args": {"query": "billing database", "buckets": {"stack": ["depends_on"], "decisions": ["decided"]}}}]},
      {"id": "unknown-bucket", "user": "Group what you know about our regions too.",
       "script": [{"tool": "memory_profile", "args": {"buckets": {"regions": ["runs_in_region"]}}}]},
      {"user": "Metric units was never my preference. Please drop it. Today I'm on the invoicing rewrite.",
       "script": [
         {"mark": "session_began"},
         {"tool": "memory_search", "args": {"query": "metric units", "memory_types": ["procedural"]}, "capture": {"metric_id": "id=(cl_[0-9a-f]+) [^\\]]*\\] user prefers metric units"}},
         {"tool": "memory_forget", "args": {"claim_id": "{metric_id}", "reason": "the user never preferred metric units"}},
         {"tool": "memory_remember", "args": {"predicate": "working_on", "object": "the invoicing rewrite"}}
       ]},
      {"id": "after", "user": "What's the picture now?",
       "script": [{"tool": "memory_profile", "args": {"since": "{session_began}", "buckets": {}}}]}
    ]}
  ],
  "store_gold": [
    {"id": "metric-retired", "text": "user prefers metric units", "state": "retired", "count": 1},
    {"id": "short-answers-kept", "text": "user prefers short answers", "state": "live", "count": 1, "memory_type": "procedural"}
  ],
  "answer_gold": [
    {"id": "two-standing-preferences", "turn": "opens", "must_contain": "Standing preferences (2)"},
    {"id": "standing-lists-short-answers", "turn": "opens", "must_contain": "user prefers short answers"},
    {"id": "relevant-section-for-the-topic", "turn": "opens", "must_contain": "Relevant to 'billing database'"},
    {"id": "stack-bucket", "turn": "opens", "must_contain": "Bucket 'stack' (1)"},
    {"id": "decisions-bucket", "turn": "opens", "must_contain": "Bucket 'decisions' (1)"},
    {"id": "unknown-predicate-is-reported", "turn": "unknown-bucket", "must_contain": "names 'runs_in_region', which nothing declares or uses, so it was ignored"},
    {"id": "one-standing-preference-left", "turn": "after", "must_contain": "Standing preferences (1)"},
    {"id": "withdrawn-preference-gone", "turn": "after", "must_not_contain": "metric units"},
    {"id": "recent-shows-todays-work", "turn": "after", "must_contain": "user working on the invoicing rewrite"},
    {"id": "recent-leaves-out-older-facts", "turn": "after", "must_not_contain": "billing-service"}
  ],
  "requires": ["tools"],
  "negative_control": true
}
```

- [ ] **Step 2: Write `expiry-off-and-read-only.json`**

```json
{
  "id": "expiry-off-and-read-only",
  "description": "What a server may erase. First a server with expiry erasure switched off stores a door code with expires_at, says that nothing will erase it, and still answers with it after the instant passes. An expires_at in the past, and an expire_reason with no expires_at, are both refused and write nothing. Then a read-only server opens the same store with expiry erasure on. Its reads and the session-start and recall hooks hide the expired code, but it erases nothing, because erasing is a write (sweep_expired in memvara/server/config.py). It refuses a write by name and reports that writes are disabled. At the end the code is still on disk.",
  "tier": "fast",
  "surfaces": ["stdio", "hooks"],
  "env": {"user": "tester"},
  "seed": [
    {"op": "remember", "predicate": "prefers", "object": "short answers", "memory_type": "procedural"}
  ],
  "sessions": [
    {"env": {"features": {"expiry_erasure": false}}, "turns": [
      {"id": "tell", "user": "The door code for the holiday flat is 7731. Only keep it until we leave.",
       "script": [
         {"mark": "leaving", "offset_seconds": 4},
         {"tool": "memory_remember", "args": {"predicate": "rental_door_code", "object": "7731", "expires_at": "{leaving}", "expire_reason": "temporary door code for the holiday flat"}}
       ]},
      {"id": "refused", "user": "Also keep the wifi password, guest-1234, but only until last week.",
       "script": [
         {"tool": "memory_remember", "args": {"predicate": "wifi_password", "object": "guest-1234", "expires_at": "2020-01-01"}, "expect_error": true},
         {"tool": "memory_remember", "args": {"predicate": "wifi_password", "object": "guest-1234", "expire_reason": "only for the stay"}, "expect_error": true}
       ]},
      {"user": "We've left the flat now.",
       "script": [{"wait_until": "leaving"}]},
      {"id": "switched-off", "user": "What was the door code?",
       "script": [{"tool": "memory_recall", "args": {"query": "rental door code"}}]}
    ]},
    {"env": {"read_only": true}, "turns": [
      {"id": "opens", "user": "Hi. Let's get started.",
       "script": [{"hook": "session_start"}, {"hook": "recall"}]},
      {"id": "asks-code", "user": "What's the rental door code?",
       "script": [{"hook": "recall"}, {"tool": "memory_recall", "args": {"query": "rental door code"}}]},
      {"id": "write", "user": "Forget that door code now.",
       "script": [{"tool": "memory_forget", "args": {"predicate": "rental_door_code"}, "expect_error": true}]},
      {"id": "stats", "user": "Can you change anything here?",
       "script": [{"tool": "memory_stats"}]}
    ]}
  ],
  "store_gold": [
    {"id": "code-still-on-disk", "text": "user rental door code 7731", "state": "live", "count": 1},
    {"id": "past-expiry-wrote-nothing", "text": "user wifi password guest-1234", "state": "absent"}
  ],
  "answer_gold": [
    {"id": "receipt-says-nothing-will-erase-it", "turn": "tell", "must_contain": "so nothing will erase this fact"},
    {"id": "past-expiry-refused", "turn": "refused", "must_contain": "is not in the future"},
    {"id": "reason-without-expiry-refused", "turn": "refused", "must_contain": "no expires_at was sent"},
    {"id": "still-answered-with-the-switch-off", "turn": "switched-off", "must_contain": "7731"},
    {"id": "read-only-hooks-still-read", "turn": "opens", "must_contain": "user prefers short answers"},
    {"id": "read-only-session-start-hides-it", "turn": "opens", "must_not_contain": "7731"},
    {"id": "read-only-reads-hide-it", "turn": "asks-code", "must_not_contain": "7731"},
    {"id": "write-refused-by-name", "turn": "write", "must_contain": "memory_forget is unavailable: this memory server is read-only"},
    {"id": "stats-say-writes-are-disabled", "turn": "stats", "must_contain": "writes: disabled"}
  ],
  "requires": ["tools", "hooks.session_start", "hooks.recall"],
  "negative_control": true
}
```

- [ ] **Step 3: Play both**

Run: `PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider tests/adversarial/sessions/test_adv_scenarios.py -k "profile-session-start or expiry-off-and-read-only"`
Expected: PASS. The second takes about five seconds, four of them waiting for the door code to expire.

- [ ] **Step 4: Document them.** Add two rows:

```markdown
| `profile-session-start` | `memory_profile`, with buckets that read the same on Python 3.10 |
| `expiry-off-and-read-only` | Expiry erasure switched off, and a read-only server and its hooks that hide an expired fact without erasing it |
```

- [ ] **Step 5: Commit**

```bash
git add tests/scenarios/scripted/profile-session-start.json \
  tests/scenarios/scripted/expiry-off-and-read-only.json docs/claude/testing.md
git commit -m "Add scenarios for the profile, expiry switched off and a read-only server"
```

---

### Task 12: Verification

- [ ] **Step 1: The fast tier's time for everything added.** Run the new files and the new scenarios together, and read the total:

```bash
NEW="session-recall-every-prompt or correction-end-forms or correction-retire-forms or correction-erased-with-sources or time-travel-three-readings or document-lifecycle or bulk-end-tokens or graph-walk or profile-session-start or expiry-off-and-read-only"
PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider --durations=10 \
  tests/adversarial/sessions/test_adv_scenarios.py -k "$NEW"
PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider --durations=10 \
  tests/adversarial/sessions/test_adv_switches.py tests/adversarial/sessions/test_adv_switches_pipe.py \
  tests/adversarial/sessions/test_adv_protocols.py tests/adversarial/sessions/test_adv_cloud_read_only.py
```

The two wall times added together are the time the new work adds.

Expected: about 25 seconds for the new work. If it is well over, move `expiry-off-and-read-only` to the nightly tier by setting its `tier` to `nightly`, and say so in the report.

- [ ] **Step 2: 20 runs in a row of each new test file,** and of the new scenarios, counting passes:

```bash
for i in $(seq 20); do PYTHONPATH=$WT TMPDIR=$TMP $PY -m pytest -q -p no:cacheprovider \
  tests/adversarial/sessions/test_adv_switches.py tests/adversarial/sessions/test_adv_switches_pipe.py \
  tests/adversarial/sessions/test_adv_protocols.py tests/adversarial/sessions/test_adv_cloud_read_only.py \
  tests/adversarial/sessions/test_adv_runner.py tests/adversarial/sessions/test_adv_scenarios.py 2>&1 | tail -1; done
```

and 20 runs of `tests/adversarial/sessions/nightly --tier nightly`. The weekly file runs once (about nine minutes).

- [ ] **Step 3: The full gate, once.** First check that no other gate is running: `pgrep -fl "coverage run -m pytest"`; wait until it prints nothing. Then:

```bash
mkdir -p $WT/local/cov
PYTHONPATH=$WT TMPDIR=$TMP COVERAGE_FILE=$WT/local/cov/.coverage.a1 $PY -m coverage run -m pytest -q -p no:cacheprovider
PYTHONPATH=$WT TMPDIR=$TMP COVERAGE_FILE=$WT/local/cov/.coverage.a1 $PY -m coverage report | tail -3
```

Expected: every test passes, and coverage of `memvara/` is 100%.

- [ ] **Step 4: Type checks**

```bash
$PY -m mypy -p memvara
$PY -m mypy tests/harness
$PY -m mypy tests/harness --ignore-missing-imports
```

Expected: "Success" from each.

- [ ] **Step 5: The report.** The branch name, each commit's short sha and subject, the files added or changed, the pass counts, the 20-run results, the gate's result line and coverage total, the mypy results, every bug found with its classification and reproduction, and every departure from the design with its reason.

---

## After the branch review

A review of the whole branch found checks that could pass while the thing they guard was broken. These changes were made after it, so the code differs from the tasks above where this list says so:

- **The read-only store check compares everything.** `switches.store_dump` reads every row of every table with `sqlite3` alone and hashes the vector file and the embedder record. It replaced `read_rows`, which compared only claims and documents and could not see a link or a turn being written. The template now also holds a fact whose expiry has passed before any server starts, so a read-only server that erases it fails the check. New tests show the dump sees a link write and a writable open's erasure.
- **The refusal and call checks have planted faults of their own.** A hidden tool that still runs, and a `filepath_prefix` ignored instead of refused, are found only by the refusal check. A handler that reads a removed argument is found only by the call check. A real server started writable while the check expects read-only mode must be reported for its list, its refusals and its store. The refusal check now sends `filepath_prefix` as well as `filters`, one per call, and the call check does the same when filters are on.
- **Gold that could pass on an empty answer.** `stale-confirm-closed-nothing` checked only the preview's count, which a preview fills with any live fact, so it was replaced by `stale-confirm-left-the-status-live` and `stale-confirm-left-the-testers-live`. The turns that had only "must not contain" items each gained a "must contain" item, and the repeated question in `session-recall-every-prompt` now also asks about work, so the recall hook has something new to bring. `correction-erased-with-sources` gained a turn in which the anchored search finds the passport number before the erase.
- **Smaller corrections.** The predicate test in `test_adv_runner.py` gained its control without the vocabulary. The cloud test checks the "writes: disabled" line first, because a timed-out credential probe is the likely cause of the failures after it. The prose now says five tools, not four, need an argument their schema leaves optional, and gives the weekly run's measured time.
