"""The scope and the read mode a server reads from its environment bind what it serves.

A client starts `memvara-mcp` with an environment block and nothing else, so these
variables are the only way an operator binds a server to a scope or changes how it
answers. `memvara/server/config.py` reads them, and `docs/DEPLOY.md` documents them:

- `MEMVARA_TENANT`, `MEMVARA_USER`, `MEMVARA_AGENT` and `MEMVARA_SESSION` bind the scope
  every write is stored under and every read looks in. A blank value means the variable
  is unset, because a settings file that writes `""` means "none".
- `MEMVARA_ANCHORED=1` makes the read tools answer nothing about an entity the store has
  never heard of, unless a call asks otherwise.
- `MEMVARA_ANCHORED`, `MEMVARA_ADVISE_REPLACEMENTS` and `MEMVARA_CLOSED_VOCABULARY` are
  flags. A value that is neither true nor false is refused at startup, naming the
  variable, rather than read as false.

Every test runs the server's own entry point, `memvara.server.cli.main`, in this process,
with the MEMVARA_ variables of the suite's child environment (`harness.env.child_env`):
the hashing embedder, and encryption and project detection off. It sends JSON-RPC lines
on standard input and reads the replies from standard output, as a client would.
"""

from __future__ import annotations

import io
import json
import pathlib
from typing import Any, Mapping, Sequence

import pytest

from harness import stores
from harness.env import child_env
from harness.skips import needs_toml
from memvara.server.cli import main

#: A call to a tool: its name and its arguments.
Call = tuple[str, Mapping[str, Any]]


def _serve(home: pathlib.Path, env: Mapping[str, str],
           calls: Sequence[Call] = ()) -> tuple[int, list[str], str]:
    """Run the server on `env` for `calls`, and return its exit status, the text of each
    reply, and what it wrote to standard error."""
    base = {name: value for name, value in child_env(home).items()
            if name.startswith("MEMVARA_")}
    lines = "".join(json.dumps({"jsonrpc": "2.0", "id": number, "method": "tools/call",
                                "params": {"name": name, "arguments": dict(arguments)}})
                    + "\n" for number, (name, arguments) in enumerate(calls, 1))
    stdout, stderr = io.StringIO(), io.StringIO()
    status = main([], env={**base, **env}, stdin=io.StringIO(lines), stdout=stdout,
                  stderr=stderr)
    texts = []
    for line in stdout.getvalue().splitlines():
        reply = json.loads(line)
        assert "error" not in reply, reply
        texts.append(reply["result"]["content"][0]["text"])
    return status, texts, stderr.getvalue()


def _scope_line(stats: str) -> str:
    return next(line for line in stats.splitlines() if line.startswith("scope: "))


@pytest.mark.covers("env:MEMVARA_TENANT", "env:MEMVARA_USER", "env:MEMVARA_AGENT",
                    "env:MEMVARA_SESSION")
def test_the_scope_variables_bind_the_scope_a_write_is_stored_under(
        tmp_path: pathlib.Path) -> None:
    """A fact written through a server started with all four scope variables is stored
    under exactly that tenant, user, agent and session. `memory_stats` reports the bound
    scope, the store holds the fact under it, and a reader in another tenant sees nothing
    of it."""
    db = tmp_path / "memory.db"
    status, (stored, stats), _ = _serve(tmp_path, {
        "MEMVARA_DB": str(db), "MEMVARA_TENANT": "acme", "MEMVARA_USER": "alice",
        "MEMVARA_AGENT": "coder", "MEMVARA_SESSION": "s1"}, [
        ("memory_remember", {"subject": "user", "predicate": "lives_in",
                             "object": "Lisbon"}),
        ("memory_stats", {})])
    assert status == 0 and "added 1" in stored
    assert _scope_line(stats).startswith("scope: acme/alice/*/coder/s1 ")

    with stores.file(db, tenant="acme", user="alice", agent="coder",
                     session="s1") as memory:
        (claim,) = memory.get_all()
    assert claim.object == "Lisbon"
    assert (claim.scope.tenant, claim.scope.user, claim.scope.agent,
            claim.scope.session) == ("acme", "alice", "coder", "s1")
    with stores.file(db, user="alice") as memory:
        assert memory.get_all() == [], "the default tenant must not see tenant acme"


@pytest.mark.covers("env:MEMVARA_TENANT", "env:MEMVARA_USER", "env:MEMVARA_AGENT",
                    "env:MEMVARA_SESSION")
def test_a_blank_scope_variable_binds_nothing(tmp_path: pathlib.Path) -> None:
    """A scope variable set to blank, or to spaces alone, means the same as one not set:
    the default tenant, and no user, agent or session. Binding the blank value instead
    would put the server's writes in a partition no other reader can name."""
    status, (stats,), _ = _serve(tmp_path, {
        "MEMVARA_DB": str(tmp_path / "memory.db"), "MEMVARA_TENANT": " ",
        "MEMVARA_USER": "  ", "MEMVARA_AGENT": "  ", "MEMVARA_SESSION": " "},
        [("memory_stats", {})])
    assert status == 0
    assert _scope_line(stats).startswith("scope: default/*/*/*/* ")


@pytest.mark.covers("env:MEMVARA_ANCHORED")
def test_an_anchored_server_answers_nothing_about_an_unknown_entity(
        tmp_path: pathlib.Path) -> None:
    """With `MEMVARA_ANCHORED=1`, a recall about somebody the store has never heard of
    says that nothing matched, even though a stored fact about somebody else is similar.
    With `MEMVARA_ANCHORED=0`, the same recall returns that similar fact."""
    calls: list[Call] = [
        ("memory_remember", {"subject": "Ivan", "predicate": "lives_in",
                             "object": "Lisbon"}),
        ("memory_recall", {"query": "where does Oscar live"})]
    for anchored, answers in (("1", False), ("0", True)):
        status, (_, recalled), _ = _serve(tmp_path, {
            "MEMVARA_DB": str(tmp_path / f"anchored-{anchored}.db"),
            "MEMVARA_USER": "alice", "MEMVARA_ANCHORED": anchored}, calls)
        assert status == 0
        assert ("Lisbon" in recalled) is answers, (anchored, recalled)
        assert ("No stored memory matched" in recalled) is not answers, (anchored, recalled)


#: A predicate file that declares three relations between entities, so that a claim
#: under each of them is an edge the graph leg can walk. Only a declared relation makes
#: an edge; an undeclared predicate's object is a plain value.
ORG_PREDICATES = "".join(
    f'[[predicate]]\nname = "{name}"\ncardinality = "one"\nvolatility = "slow"\n'
    'object_type = ["entity"]\ngraph = true\n\n'
    for name in ("reports_to", "works_at", "headquartered_in"))

#: Alice's manager is Dana, Dana works at Acme, and Acme is in Tallinn. The other facts
#: are there so that a search has something else to return.
ORG_FACTS = (("Alice", "reports_to", "Dana"), ("Dana", "works_at", "Acme"),
             ("Acme", "headquartered_in", "Tallinn"), ("Bruno", "works_at", "Acme"),
             ("Carol", "lives_in", "Lisbon"), ("Erik", "lives_in", "Oslo"),
             ("Frida", "lives_in", "Bergen"), ("Globex", "headquartered_in", "Riga"))


@needs_toml
@pytest.mark.covers("env:MEMVARA_READ_W_GRAPH")
def test_the_graph_weight_brings_a_fact_the_question_does_not_mention_into_a_search(
        tmp_path: pathlib.Path) -> None:
    """`MEMVARA_READ_W_GRAPH` gives the graph leg of retrieval its weight, and it is 0,
    the leg switched off, by default. The question asks where the employer of Alice's
    manager is based. "Dana works at Acme" is the middle step of that chain and shares
    no word with the question, so only a walk from "Alice reports to Dana" can reach it.

    With the weight at 1.0, a search for three results returns it. Without the variable,
    the same search on the same facts does not. The predicate file is TOML, so this test
    needs Python 3.11 or later."""
    predicates = tmp_path / "org.toml"
    predicates.write_text(ORG_PREDICATES, encoding="utf-8")
    calls: list[Call] = [
        *(("memory_remember", {"subject": subject, "predicate": predicate,
                               "object": obj}) for subject, predicate, obj in ORG_FACTS),
        ("memory_search", {"query": "which city is the employer of Alice's manager "
                                    "based in?", "k": 3})]
    found = {}
    for weight in ("1.0", None):
        env = {"MEMVARA_DB": str(tmp_path / f"graph-{weight}.db"), "MEMVARA_USER": "alice",
               "MEMVARA_PREDICATES": str(predicates)}
        if weight is not None:
            env["MEMVARA_READ_W_GRAPH"] = weight
        status, replies, _ = _serve(tmp_path, env, calls)
        assert status == 0 and len(replies) == len(calls)
        found[weight] = replies[-1]
    assert "Dana works at Acme" in found["1.0"], found["1.0"]
    assert "Dana works at Acme" not in found[None], found[None]


@pytest.mark.covers("env:MEMVARA_ANCHORED", "env:MEMVARA_ADVISE_REPLACEMENTS",
                    "env:MEMVARA_CLOSED_VOCABULARY")
@pytest.mark.parametrize("variable", ["MEMVARA_ANCHORED", "MEMVARA_ADVISE_REPLACEMENTS",
                                      "MEMVARA_CLOSED_VOCABULARY"])
def test_a_flag_that_is_neither_true_nor_false_stops_the_server(
        tmp_path: pathlib.Path, variable: str) -> None:
    """A flag set to something that is not a boolean is refused before the server
    serves anything. The server exits with status 2 and says which variable is wrong and
    what it accepts. Reading the typo as false would leave the operator believing the
    setting was on."""
    db = tmp_path / "memory.db"
    status, replies, stderr = _serve(tmp_path, {"MEMVARA_DB": str(db), variable: "maybe"},
                                     [("memory_stats", {})])
    assert (status, replies) == (2, [])
    assert f"{variable}='maybe' is not a boolean" in stderr
    assert not db.exists(), "a refused configuration must not create the store"
