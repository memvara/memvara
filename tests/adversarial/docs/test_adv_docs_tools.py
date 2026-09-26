"""The tool descriptions name only tools and arguments that exist.

A model reads these descriptions once and cannot check them. A description that names a
tool or an argument the server does not have sends the model to a call that fails, so
every name in them is checked here against the tool table, in every configuration a
server can be started in.
"""

from __future__ import annotations

import pytest

from harness import known_bugs
from memvara.server.config import FEATURE_DEFAULTS
from memvara.server.mcp import INSTRUCTIONS
from memvara.server.tools import FEATURE_ARGUMENTS, TOOLS

from .mentions import (mentions, predicates, ties, tool_names, unknown_tools,
                       unresolved_identifiers, unserved_arguments, unserved_by_any_tool,
                       wrong_ties)
from .surface import configurations, served, table, texts

#: A small tool table for the planted cases, so each expected answer can be read off by
#: hand rather than computed by the code under test.
PLANTED = {
    "memory_recall": frozenset({"query", "include_episodes", "ranked", "synthesize",
                                "query_rewrite", "valid_at", "budget"}),
    "memory_search": frozenset({"query", "k", "as_of", "valid_at"}),
    "memory_end": frozenset({"claim_id", "at", "reason"}),
    "memory_remember": frozenset({"subject", "predicate", "object", "true_since",
                                  "true_until", "memory_type"}),
}


# -- the parse, on planted text ----------------------------------------------------------

def test_a_misspelled_tool_name_is_reported() -> None:
    assert unknown_tools("Call memory_serch first, then memory_recall.", PLANTED) == [
        "memory_serch"]


def test_wildcards_paths_and_arguments_are_not_reported_as_tools() -> None:
    text = ("The memory_* tools. See memvara/server/memory_api.py and "
            "memvara.server.memory_api. Pass memory_type, then call memory_recall.")
    assert tool_names(text) == ["memory_type", "memory_recall"]
    assert unknown_tools(text, PLANTED) == []


def test_a_renamed_argument_is_reported() -> None:
    """The fault this check exists for: an argument renamed in the schema and not in the
    words, so the description sends the model to the old name."""
    text = "For a fact that will stop being true, send valid_from instead, at kk=5."
    assert unresolved_identifiers(text, PLANTED) == ["valid_from", "kk"]


def test_predicates_examples_and_labels_are_not_arguments() -> None:
    text = (
        "The relation, in snake_case: lives_in, works_at, uses_tool. "
        "A memory filed under a paraphrase ('the coverage threshold' for coverage_gate). "
        "A claim id, e.g. 'cl_1a2b3c...'. Buckets such as {\"stack\": [\"depends_on\"]}. "
        "A predicate like attended or met_with is filed as semantic. "
        "An operator sets it from memvara.calibrate_min_score. "
        "Later as_of and valid_at questions read it, with true_until=2 at k=5."
    )
    assert unresolved_identifiers(text, PLANTED) == []


def test_the_builtin_predicates_and_their_aliases_are_known() -> None:
    assert {"lives_in", "works_at", "prefers_tool", "uses_tool"} <= predicates()


def test_a_tie_to_the_wrong_tool_is_reported() -> None:
    text = "memory_search with include_episodes true returns passages from it."
    assert wrong_ties(text, PLANTED) == [("memory_search", "include_episodes")]


def test_a_tie_to_the_right_tool_is_not() -> None:
    text = ("memory_recall with include_episodes true; memory_end and claim_id; "
            "memory_remember's true_since; call memory_end again with the same claim_id; "
            "memory_end with a query; memory_recall and memory_search; "
            "memory_end and to memory_search.")
    assert ties(text)[:3] == [("memory_recall", "include_episodes"),
                              ("memory_end", "claim_id"), ("memory_remember", "true_since")]
    assert wrong_ties(text, PLANTED) == []


def test_a_one_word_name_after_a_possessive_or_and_is_english() -> None:
    """"memory_end's predicate" is the predicate of the fact memory_end closes, and "memory_end
    and query" joins two verbs. Neither names an argument, so neither is a tie, as a one-word
    name after "with a" is not."""
    assert wrong_ties("memory_end's predicate stays exactly as memory_remember first wrote "
                      "it.", PLANTED) == []
    assert wrong_ties("Call memory_end and query the store again.", PLANTED) == []
    assert ties("memory_end's predicate; memory_end and query.") == []


def test_each_way_of_naming_an_argument_is_read() -> None:
    text = ("Give 'predicate' (with 'subject', default 'user'); at k=5; ranked and "
            "synthesize each add a call; send true_since / true_until.")
    own = {"predicate", "subject", "k", "ranked", "synthesize", "true_since", "true_until"}
    found = [(m.word, m.kind) for m in mentions(text, own)]
    assert found == [("predicate", "quoted"), ("subject", "quoted"), ("k", "assigned"),
                     ("ranked", "listed"), ("synthesize", "listed"),
                     ("true_since", "snake"), ("true_until", "snake")]


def test_an_argument_a_switch_removed_is_reported() -> None:
    text = ("It rewrites the query into other phrasings (query_rewrite), and ranked and "
            "synthesize each add one more call; reach for 'budget' to save space.")
    own = PLANTED["memory_recall"]
    served_here = own - {"query_rewrite", "synthesize", "budget"}
    assert unserved_arguments(text, own, served_here) == ["query_rewrite", "synthesize",
                                                          "budget"]


def test_an_english_word_that_is_also_an_argument_is_not_a_mention() -> None:
    """With `end_reason` off, `reason` is gone from every tool, and the descriptions still
    say "a false reason" in plain English. That must not count as naming the argument."""
    text = ("Getting that backwards writes a false reason into an audit trail. The reason "
            "is that the query and the text are records.")
    own = PLANTED["memory_end"] | {"query", "text"}
    assert unserved_arguments(text, own, own - {"reason", "query", "text"}) == []


def test_text_owned_by_no_tool_naming_an_argument_a_switch_removed_is_reported() -> None:
    """The instructions a client receives on connect belong to no one tool. An argument
    they name is removed when no tool the server lists still takes it, and an argument of a
    tool the server hides is left alone, because a call to that tool is refused by name."""
    text = ("A fact written with memory_remember's true_until ends then; pass valid_at to "
            "read it.")
    served_here = {"memory_remember": PLANTED["memory_remember"] - {"true_until"},
                   "memory_search": PLANTED["memory_search"]}
    assert unserved_by_any_tool(text, PLANTED, served_here) == ["true_until"]
    assert unserved_by_any_tool(text, PLANTED, {"memory_search": PLANTED["memory_search"]}) == []


def test_every_switch_has_a_configuration() -> None:
    """Every feature switch gets a server of its own, so no switch goes unchecked."""
    labels = [configuration.label for configuration in configurations()]

    assert len(labels) == len(set(labels)), labels
    assert {"default", "read-only", "anchored"} <= set(labels)
    for feature in FEATURE_DEFAULTS:
        assert len([label for label in labels if label.split()[0] == feature]) == 1, feature
    assert "profile off" in labels, "a feature that is on by default is switched off"
    assert "agentic_extraction on" in labels, "a feature that is off by default is switched on"


def test_each_configuration_serves_what_its_switch_says() -> None:
    """Each configuration is a real, different server, so a check run over all of them
    is not the default server checked twenty-five times."""
    owned_tools = [tool for tool in TOOLS if tool.feature]
    for configuration in configurations():
        tools = {tool["name"]: tool for tool in served(configuration)}
        for owned in owned_tools:
            hidden = (owned.feature in configuration.features_off
                      or (configuration.read_only and owned.writes))
            assert (owned.name in tools) is not hidden, (configuration.label, owned.name)
        for feature, removed in FEATURE_ARGUMENTS.items():
            if feature in configuration.features_off:
                for tool in tools.values():
                    assert not set(removed) & set(tool["inputSchema"]["properties"]), (
                        configuration.label, tool["name"])
        if configuration.read_only:
            assert all(tool["annotations"]["readOnlyHint"] for tool in tools.values())
        for tool in tools.values():
            anchored = tool["inputSchema"]["properties"].get("anchored")
            if anchored is not None:
                assert anchored["default"] is configuration.anchored, (
                    configuration.label, tool["name"])

    by_label = {configuration.label: configuration for configuration in configurations()}
    assert len(served(by_label["default"])) == len(TOOLS)
    assert "memory_profile" not in {t["name"] for t in served(by_label["profile off"])}
    recall = {t["name"]: t for t in served(by_label["synthesis off"])}["memory_recall"]
    assert "synthesize" not in recall["inputSchema"]["properties"]
    assert "memory_remember" not in {t["name"] for t in served(by_label["read-only"])}


def test_the_table_keeps_every_argument_whatever_the_switches() -> None:
    """The table is what a tool can take on some server, so arguments a switch removes
    are still in it."""
    assert {"synthesize", "query_rewrite", "anchored"} <= table()["memory_recall"]
    assert "reason" in table()["memory_forget"]
    assert set(table()) == {tool.name for tool in TOOLS}


def test_texts_labels_each_description_with_where_it_came_from() -> None:
    tool = {"name": "t", "description": "d",
            "inputSchema": {"properties": {"a": {"description": "x"}, "b": {}}}}

    assert texts(tool) == [("t", "d"), ("t.a", "x"), ("t.b", "")]


# -- the real descriptions ---------------------------------------------------------------

def _every_text() -> list[tuple[str, str, str]]:
    """`(configurations, where, text)` for each distinct description any configuration
    serves, and for the instructions the server sends when a client connects.

    Most descriptions are the same on every server, so each distinct text is checked once
    and the message names the configurations that serve it.
    """
    serving: dict[tuple[str, str], list[str]] = {("INSTRUCTIONS", INSTRUCTIONS): []}
    everywhere = [configuration.label for configuration in configurations()]
    for configuration in configurations():
        for tool in served(configuration):
            for where, text in texts(tool):
                serving.setdefault((where, text), []).append(configuration.label)
    return [("every configuration" if labels in ([], everywhere) else ", ".join(labels),
             where, text) for (where, text), labels in serving.items()]


def test_every_tool_the_descriptions_name_exists() -> None:
    problems = [f"{label}: {where} names {name}, which is not a tool"
                for label, where, text in _every_text()
                for name in unknown_tools(text, table())]
    assert not problems, "\n".join(problems)
    assert any(tool_names(text) for _, _, text in _every_text()), "no tool name was read"


def test_every_identifier_in_the_descriptions_resolves() -> None:
    """A snake_case word that is not a tool, an argument or a predicate is most likely an
    argument renamed in the schema while the words kept its old name."""
    problems = [f"{label}: {where} names {word!r}, which is no tool, argument or predicate"
                for label, where, text in _every_text()
                for word in unresolved_identifiers(text, table())]
    assert not problems, "\n".join(problems)


def test_a_tool_named_with_an_argument_takes_it() -> None:
    problems = [f"{label}: {where} ties {argument!r} to {tool}, which does not take it"
                for label, where, text in _every_text()
                for tool, argument in wrong_ties(text, table())]
    assert not problems, "\n".join(problems)
    assert any(ties(text) for _, _, text in _every_text()), "no tie was read"


#: The drift #295 pins: in these configurations memory_recall's own description still
#: names an argument that the configuration's switch removed from its schema.
REMOVED_BUT_NAMED = {"query_rewrite off": {("memory_recall", "query_rewrite")},
                     "synthesis off": {("memory_recall", "synthesize")}}


def _each_configuration() -> list[object]:
    return [pytest.param(configuration, id=configuration.label,
                         marks=[known_bugs.xfail("B19")]
                         if configuration.label in REMOVED_BUT_NAMED else [])
            for configuration in configurations()]


@pytest.mark.parametrize("configuration", _each_configuration())
def test_a_tools_own_argument_named_in_its_text_is_served(configuration) -> None:
    """A switch that removes an argument removes it from the schema. A description that
    still names it tells the model to pass something the server refuses."""
    found: set[tuple[str, str]] = set()
    problems = []
    for tool in served(configuration):
        own = table()[tool["name"]]
        here = set(tool["inputSchema"]["properties"])
        for where, text in texts(tool):
            for word in unserved_arguments(text, own, here):
                found.add((tool["name"], word))
                problems.append(f"{where} names {word!r}, which this server removed")
    if found and found == REMOVED_BUT_NAMED.get(configuration.label):
        raise known_bugs.Reproduced("\n".join(problems))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("configuration", configurations(),
                         ids=[configuration.label for configuration in configurations()])
def test_the_connection_instructions_name_no_argument_a_switch_removed(configuration) -> None:
    """The instructions a client receives on connect are the same on every server, and a
    switch can remove an argument they name, so each server checks them as it checks its
    tool descriptions."""
    served_here = {tool["name"]: set(tool["inputSchema"]["properties"])
                   for tool in served(configuration)}
    removed = unserved_by_any_tool(INSTRUCTIONS, table(), served_here)
    assert not removed, (f"the instructions name {removed}, which no tool this server "
                         "lists takes")
