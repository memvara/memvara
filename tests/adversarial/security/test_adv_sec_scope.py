"""Scope isolation between two server processes on one store file.

The MCP server binds its scope once, at startup, from the environment its client gave it;
no tool argument can name another scope. So the guarantee to attack is that a reader in
one scope cannot learn anything about a claim written in a sibling scope — not even
whether the id exists. `SECURITY.md` names this first, and it names the trap: an
id-addressed read (`memory_why`) that authorized differently from an enumerating read
(`memory_search`) was a real bug here. Ids are not secret — receipts and results leak
them — so "the attacker needed the id" is not a defence.

The test starts two real servers on one database file, bound to sibling scopes, and
checks that every tool taking an id answers about a foreign id exactly as it answers about
an id that was never minted, byte for byte once the id itself is normalised away.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterator

import pytest

from harness.stdio import McpProcess, claim_id_of

Mcp = Callable[..., McpProcess]

#: Each sibling pair differs from the writer in exactly one axis of the scope hierarchy.
#: `user` is the ordinary multi-tenant case; the other three are the deeper boundaries the
#: design calls out (a sibling project, agent or session, and a separate tenant).
SIBLINGS = [
    ("user", dict(user="alice"), dict(user="bob")),
    ("tenant", dict(user="u", scope={"tenant": "t1"}), dict(user="u", scope={"tenant": "t2"})),
    ("agent", dict(user="u", scope={"agent": "a1"}), dict(user="u", scope={"agent": "a2"})),
    ("session", dict(user="u", scope={"session": "s1"}),
     dict(user="u", scope={"session": "s2"})),
]

_NEVER_CLAIM = "cl_" + "0" * 20
_NEVER_DOC = "doc_" + "0" * 20


def _norm(text: str, *ids: str) -> str:
    """`text` with each id replaced by a fixed token, so only the id itself may differ."""
    for ident in ids:
        text = text.replace(ident, "<ID>")
    return text


def _seed(server: McpProcess) -> "tuple[str, str]":
    """Write one claim and one document through `server`; return their ids."""
    made = server.call("memory_remember", predicate="lives_in", object="a private value")
    claim_id = claim_id_of(made)
    doc = server.call("memory_add_document", content="a private document body",
                      custom_id="priv", title="a private title")
    doc_id = doc.text.split("document ")[1].split(":")[0]
    return claim_id, doc_id


def _seed_own_claim(server: McpProcess) -> str:
    """Write one claim through `server`, in its own scope, and return its id.

    `_id_tool_cases`' memory_link case needs a second id the reader can see: `link()`
    refuses a claim linked to itself, `refuse_self_link`, before it ever checks scope, so
    linking an id to itself would refuse identically whether or not scope was checked and
    would never exercise the boundary this file exists to test. Called only by the
    id-tool-cases test below, never from `pair`: the enumeration test that also uses
    `pair` relies on the reader's own scope holding nothing at all, and memory_search has
    no score floor by default, so even an unrelated claim there would turn its "no match"
    into a match on relevance 0.
    """
    made = server.call("memory_remember", predicate="prefers", object="tea over coffee")
    return claim_id_of(made)


@pytest.fixture
def pair(mcp: Mcp) -> Callable[[dict[str, Any], dict[str, Any]], "tuple[McpProcess, str, str]"]:
    """A writer and a reader on one store file, plus the ids the writer seeded."""
    def build(writer_scope: dict[str, Any],
              reader_scope: dict[str, Any]) -> "tuple[McpProcess, str, str]":
        writer = mcp(**writer_scope)
        writer.initialize()
        claim_id, doc_id = _seed(writer)
        reader = mcp(db=writer.db, **reader_scope)
        reader.initialize()
        return reader, claim_id, doc_id
    return build


def _id_tool_cases(
        claim_id: str, doc_id: str, reader_claim_id: str) -> "Iterator[tuple[str, dict, dict]]":
    """For each id-taking tool, the arguments naming the foreign id and the absent id.

    The absent case is the foreign case with the writer's ids swapped for ones that were
    never minted, so the two argument sets can only differ in the id under test.
    memory_link additionally names `reader_claim_id`, a claim the reader owns, on the
    other end of the link in both cases — see `_seed_own_claim` for why it cannot be
    `claim_id` linked to itself instead.
    """
    swap = {claim_id: _NEVER_CLAIM, doc_id: _NEVER_DOC}

    def absent(args: dict) -> dict:
        return {key: swap.get(value, value) for key, value in args.items()}

    foreign_cases: "list[tuple[str, dict]]" = [
        ("memory_why", {"claim_id": claim_id}),
        ("memory_forget", {"claim_id": claim_id}),
        ("memory_end", {"claim_id": claim_id}),
        ("memory_get_document", {"id": doc_id}),
        ("memory_delete_document", {"id": doc_id}),
        ("memory_link",
         {"from_id": claim_id, "to_id": reader_claim_id, "relation": "extends"}),
    ]
    for tool, foreign_args in foreign_cases:
        yield tool, foreign_args, absent(foreign_args)


@pytest.mark.covers("tool:memory_why", "tool:memory_forget", "tool:memory_get_document",
                    "tool:memory_delete_document", "tool:memory_link")
@pytest.mark.parametrize("_label, writer_scope, reader_scope", SIBLINGS)
def test_an_id_from_another_scope_answers_as_one_that_never_existed(
        pair, _label, writer_scope, reader_scope) -> None:
    """Every tool taking an id must answer about a foreign id exactly as about an id that
    was never minted, so no tool can be used to test whether an id exists elsewhere."""
    reader, claim_id, doc_id = pair(writer_scope, reader_scope)
    reader_claim_id = _seed_own_claim(reader)
    for tool, foreign_args, absent_args in _id_tool_cases(claim_id, doc_id, reader_claim_id):
        foreign = reader.call(tool, **foreign_args)
        absent = reader.call(tool, **absent_args)
        assert foreign.is_error == absent.is_error, tool
        assert (_norm(foreign.text, claim_id, doc_id)
                == _norm(absent.text, _NEVER_CLAIM, _NEVER_DOC)), tool


@pytest.mark.covers("tool:memory_list_documents")
def test_a_sibling_scope_cannot_enumerate_the_writers_claim(pair) -> None:
    """The reader's own search, recall and document listing never surface the writer's
    value: isolation is not only about id-addressed reads. A scope that resolves to
    another user's must match nothing here."""
    reader, _claim_id, doc_id = pair(dict(user="alice"), dict(user="bob"))
    # A read that finds nothing repeats the query in its reply, so the value appearing is
    # not evidence of a leak; a returned *result row* would be. `memory_search` numbers
    # its hits `1. [id=... relevance=...]`, so the absence of any such row is the check.
    search = reader.call("memory_search", query="a private value")
    assert re.search(r"[Nn]o stored memory matched", search.text)
    assert not re.search(r"^\d+\. \[id=", search.text, re.MULTILINE)
    # `memory_recall` renders hits as `- ` bullets under its header; none may appear.
    recall = reader.call("memory_recall", query="a private value")
    assert not re.search(r"^- ", recall.text, re.MULTILINE)
    # The writer's document must not be enumerable either: none of the three ways to
    # recognise it — its id, its custom_id, or its title — may appear in the reader's
    # own listing.
    listing = reader.call("memory_list_documents").text
    assert doc_id not in listing
    assert "priv" not in listing
    assert "a private title" not in listing
