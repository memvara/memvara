"""A default that a description states in words equals the default in the tool's schema.

The validator fills a declared default before the handler runs, so the schema's default
is what an unqualified call gets. A description that states a different default tells the
model the opposite of what will happen, and the model cannot check.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from harness import known_bugs
from memvara.server.tools import TOOLS

from .defaults import Stated, conflicts, same, stated, stated_in_tool, undeclared
from .surface import configurations, served


# -- the parse, on planted text ----------------------------------------------------------

def test_each_way_of_stating_a_default_is_read() -> None:
    assert stated("Minimum relevance. Defaults to 0 — no floor.", "min_score") == [
        Stated("min_score", 0, True, "Defaults to 0")]
    assert stated("A ratio. Defaults to 0.5, the midpoint.", "ratio") == [
        Stated("ratio", 0.5, True, "Defaults to 0.5")]
    assert stated("How many hops out. 2 is the useful default: one hop is enough.",
                  "depth") == [Stated("depth", 2, True, "2 is the useful default")]
    assert stated("Longest route. 3 by default, because a fourth hop is damped.",
                  "depth") == [Stated("depth", 3, True, "3 by default")]
    assert stated("Nothing you write at the default 1.0 is affected.", "confidence") == [
        Stated("confidence", 1.0, True, "at the default 1.0")]
    assert stated("Default false. Set it when a wrong entity is worse.", "anchored") == [
        Stated("anchored", False, True, "Default false")]
    assert stated("Answer only from slots. Default true on this server. Leave it alone.",
                  "anchored") == [Stated("anchored", True, True, "Default true on this server")]
    assert stated("Defaults to 'api', and 'api' is a claim about provenance.",
                  "extractor") == [Stated("extractor", "api", True, "Defaults to 'api'")]


def test_a_default_computed_at_call_time_is_read_as_no_literal() -> None:
    assert stated("An ISO-8601 instant. Defaults to now, which is right only sometimes.",
                  "true_since") == [Stated("true_since", None, False, "Defaults to now")]
    assert stated("Defaults to seven days before now.", "since") == [
        Stated("since", None, False, "Defaults to seven days before now")]


def test_a_tool_description_states_the_default_of_a_quoted_argument() -> None:
    text = "Give 'predicate' (with 'subject', default 'user') to retire every value."
    assert stated_in_tool(text) == [Stated("subject", "user", True, "'subject', default 'user'")]


def test_a_default_that_ends_a_sentence_is_still_read() -> None:
    """The full stop after the value must not hide the statement, and must not cut a
    decimal short either."""
    assert stated("Relevance is weighed at the default 0.5.", "weight") == [
        Stated("weight", 0.5, True, "at the default 0.5")]
    assert stated_in_tool("Give 'predicate', with 'subject', default 'user'.") == [
        Stated("subject", "user", True, "'subject', default 'user'")]


def test_the_word_default_alone_states_nothing() -> None:
    assert stated("The read is served in the default order, and says so.", "ranked") == []
    assert stated("The default is not a harmless approximation: it asserts a start.",
                  "true_since") == []
    assert stated("The answer came back 5% of the time at the default and 41% with "
                  "min_hops=2.", "min_hops") == []
    assert stated("They become 'semantic', which is the safe default rather than a guess.",
                  "memory_type") == []
    assert stated("Because the default records it as beginning now.", "true_since") == []


def test_a_boolean_equals_only_a_boolean() -> None:
    """In Python `True == 1`, so a plain comparison would accept "Default true" against a
    declared 1."""
    assert not same(True, 1)
    assert not same(False, 0)
    assert not same(1, True)
    assert not same("1", 1)
    assert same(0, 0.0)
    assert same(True, True)
    assert same("user", "user")


PLANTED: dict[str, Any] = {
    "name": "t",
    "description": "Give 'predicate' (with 'subject', default 'user') to retire it.",
    "inputSchema": {"properties": {
        "flag": {"type": "boolean", "default": True, "description": "Default false."},
        "since": {"type": "string", "default": "2020-01-01", "description": "Defaults to now."},
        "subject": {"type": "string", "default": "someone", "description": "Who it is about."},
        "k": {"type": "integer", "default": 8, "description": "Most rows. 8 by default."},
        "role": {"type": "string", "description": "Defaults to 'api'."},
        "at": {"type": "string", "description": "Defaults to now."},
    }},
}


def test_a_stated_default_that_differs_from_the_schema_is_reported() -> None:
    assert conflicts(PLANTED) == [("flag", "Default false"), ("since", "Defaults to now"),
                                  ("subject", "'subject', default 'user'")]


def test_a_stated_default_the_schema_does_not_declare_is_reported() -> None:
    assert undeclared(PLANTED) == [("role", "Defaults to 'api'")]


# -- the real descriptions ---------------------------------------------------------------

def _every_tool() -> list[tuple[str, dict[str, Any]]]:
    """`(configurations, tool)` for each distinct tool any configuration serves. The
    anchored server rewrites the `anchored` descriptions, so it is checked on its own."""
    serving: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for configuration in configurations():
        for tool in served(configuration):
            key = json.dumps(tool, sort_keys=True)
            serving.setdefault(key, (tool, []))[1].append(configuration.label)
    everywhere = len(configurations())
    return [("every configuration" if len(labels) == everywhere else ", ".join(labels), tool)
            for tool, labels in serving.values()]


def test_a_default_stated_in_words_equals_the_declared_default() -> None:
    problems = []
    for labels, tool in _every_tool():
        properties = tool["inputSchema"]["properties"]
        problems += [f"{labels}: {tool['name']}.{argument} says {words!r}, and its schema "
                     f"declares {properties[argument]['default']!r}"
                     for argument, words in conflicts(tool)]
    assert not problems, "\n".join(problems)


def test_the_real_descriptions_state_defaults() -> None:
    """The check above passes on nothing if the parse stops reading, so each kind of
    statement must be found in the real descriptions at least once."""
    default = next(c for c in configurations() if c.label == "default")
    arguments = [statement for tool in served(default)
                 for name, schema in tool["inputSchema"]["properties"].items()
                 for statement in stated(schema.get("description", ""), name)]
    in_tools = [statement for tool in served(default)
                for statement in stated_in_tool(tool["description"])]

    assert any(statement.literal for statement in arguments), "no literal default was read"
    assert any(not statement.literal for statement in arguments), (
        "no default worked out at call time was read")
    assert in_tools, "no default stated in a tool's description was read"


#: The drift #296 pins: these tools state a default in words that their schema does not
#: declare; each handler supplies the value itself.
UNDECLARED = {"memory_recall": {"include_episodes"}, "memory_remember": {"extractor"}}


@pytest.mark.parametrize("name", [
    pytest.param(tool.name, marks=[known_bugs.xfail("B20")] if tool.name in UNDECLARED
                 else []) for tool in TOOLS])
def test_a_default_stated_in_words_is_declared_in_the_schema(name: str) -> None:
    """The validator fills only declared defaults. A default stated in words and not
    declared is implemented somewhere else, where the words and the code can drift apart
    without any check."""
    problems: dict[tuple[str, str], list[str]] = {}
    for configuration in configurations():
        for tool in served(configuration):
            if tool["name"] == name:
                for argument, words in undeclared(tool):
                    problems.setdefault((argument, words), []).append(configuration.label)
    report = "\n".join(
        f"{name}.{argument} says {words!r}, and its schema declares no default "
        f"({len(labels)} configurations)" for (argument, words), labels in problems.items())
    if problems and {argument for argument, _ in problems} == UNDECLARED.get(name):
        raise known_bugs.Reproduced(report)
    assert not problems, report
