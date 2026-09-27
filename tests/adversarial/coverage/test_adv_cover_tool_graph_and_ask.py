"""`memory_ask` and `memory_paths`, each called over the stdio pipe of a real server process.

The scripted sessions in tests/adversarial/sessions call both tools, but a scenario checks a
whole session and cannot say which tool it covers, so these tests call each tool directly
and check the text it returns. The promises come from each tool's description in
memvara/server/tools.py.
"""

from __future__ import annotations

from typing import Callable

import pytest

from harness.skips import needs_toml
from harness.stdio import McpProcess

Start = Callable[..., McpProcess]


def _remember(server: McpProcess, subject: str, predicate: str, obj: str,
              **extra: str) -> None:
    stored = server.call("memory_remember", subject=subject, predicate=predicate,
                         object=obj, **extra)
    assert not stored.is_error, stored.text


@pytest.mark.covers("tool:memory_ask")
def test_memory_ask_answers_in_prose_now_and_as_of_an_earlier_valid_time(
        mcp: Start) -> None:
    """`memory_ask` promises a prose answer built from what the store believes. With `at`,
    it answers about that earlier moment, and adds a note when what the store would have
    said then differs from what it says about that moment now. On an empty store it says
    that nothing matches instead of answering. The description of `memory_ask` in
    memvara/server/tools.py makes these promises."""
    server = mcp()
    server.initialize()

    empty = server.call("memory_ask", question="where does the user live")
    assert not empty.is_error, empty.text
    assert "Nothing in this scope matches" in empty.text

    _remember(server, "user", "lives_in", "Rome", true_since="2026-01-01T00:00:00Z")
    _remember(server, "user", "lives_in", "Berlin", true_since="2026-03-01T00:00:00Z")

    now = server.call("memory_ask", question="where does the user live")
    assert not now.is_error, now.text
    assert "user lives_in: Berlin." in now.text
    assert "Rome" not in now.text, "Rome stopped being true when Berlin began"
    assert "note:" not in now.text, "nothing reads differently about the present"

    then = server.call("memory_ask", question="where does the user live",
                       at="2026-02-01T00:00:00Z")
    assert not then.is_error, then.text
    assert "user lives_in: Rome." in then.text
    assert "Now: Berlin." in then.text
    # Both values were recorded today, so on 1 February the store held neither. The
    # answer has to say that what the store would have said then differs from today's
    # reading of that moment.
    assert "would have said nothing" in then.text
    assert "note: 1 of these read differently" in then.text
    assert server.close() == 0


@needs_toml
@pytest.mark.covers("tool:memory_paths")
def test_memory_paths_returns_the_route_between_two_entities_or_says_the_search_found_none(
        mcp: Start) -> None:
    """`memory_paths` promises the chain of stored facts that joins two entities, written
    hop by hop with the claim ids it used. When it finds none, it promises a reply that
    says the result is about this bounded search, not a claim that the two are unrelated.
    The description of `memory_paths` in memvara/server/tools.py makes these promises.

    Only a predicate declared as linking two entities makes an edge the walk can follow,
    so the server loads the engineering pack, whose `depends_on` and `deploys_to` are
    declared that way. A predicate pack is TOML, so this test needs Python 3.11 or
    later."""
    server = mcp(env={"MEMVARA_PREDICATES": "engineering"})
    server.initialize()
    for subject, predicate, obj in (("billing", "depends_on", "ledger"),
                                    ("ledger", "depends_on", "postgres"),
                                    ("postgres", "deploys_to", "frankfurt"),
                                    ("search", "depends_on", "opensearch")):
        _remember(server, subject, predicate, obj)

    route = server.call("memory_paths", source="billing", target="frankfurt")
    assert not route.is_error, route.text
    assert route.text.startswith("1 route(s) from billing to frankfurt"), route.text
    assert ("billing -depends_on-> ledger -depends_on-> postgres -deploys_to-> frankfurt"
            in route.text)
    assert "3 hop(s)" in route.text

    none = server.call("memory_paths", source="billing", target="opensearch")
    assert not none.is_error, none.text
    assert none.text.startswith("No route found from 'billing' to 'opensearch'"), none.text
    assert "-depends_on->" not in none.text, "a reply with no route shows no hop"
    assert "about this search rather than about the store" in none.text
    assert server.close() == 0
