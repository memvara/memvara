"""Defaults a description states in words, and how they are checked against the schema.

A description states a default in one of these ways, and in no other:

- a sentence that starts "Default" or "Defaults to", as in "Default false." or "Defaults to
  0 — no floor". The value is the phrase up to the next punctuation mark;
- "V is the default", or "V is the useful default" with any one word before "default";
- "V by default";
- "at the default V";
- in a tool's own description, "'argument', default V", which states that argument's
  default.

In the last four, V must be a literal: `true`, `false`, a number, or a string in single
quotes. After "Default" the phrase may instead describe a value worked out when the tool is
called, such as "now" or "seven days before now". A schema cannot hold that as a constant,
so the checks require it to declare none. A phrase that starts with a literal takes that
literal as its value, and the words after it qualify it, as in "Default true on this
server". So "Defaults to 3 days before now" would be read as 3.

The word "default" used any other way states nothing: "the default order", "The default is
not a harmless approximation", "at the default and 41%".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

_LITERAL = r"(?:true|false|-?\d+(?:\.\d+)?|'[^'\n]*')"
#: A literal ends where a word does, and may end a sentence. It may not stop at the point
#: inside a decimal, so "1.5" is never read as 1.
_END_OF_LITERAL = r"(?![\w'])(?!\.\d)"
#: The start of a sentence: the start of the text, or after a sentence's punctuation.
_START = r"(?:\A|(?<=[.;:!?]\s)|(?<=—\s)|(?<=\n))"
#: "Default V" or "Defaults to V" opening a sentence. The phrase ends at a comma, a colon,
#: a semicolon, a dash, a bracket, a line end, or a full stop followed by a space, so
#: that "0.5" is read whole.
_SENTENCE = re.compile(_START + r"(Defaults? (?:to )?([^,;:—(\n]+?))"
                       r"(?=\.(?:\s|$)|[,;:—(\n]|$)")
_PHRASES = (
    re.compile(r"(?<![\w.'])(" + _LITERAL + r") is the (?:\w+ )?default\b"),
    re.compile(r"(?<![\w.'])(" + _LITERAL + r") by default\b"),
    re.compile(r"\bat the default (" + _LITERAL + ")" + _END_OF_LITERAL),
)
_IN_TOOL = re.compile(r"'([a-z][a-z0-9_]*)',\s+default\s+(" + _LITERAL + ")"
                      + _END_OF_LITERAL)


@dataclass(frozen=True)
class Stated:
    """One default stated in words: which argument, the value, and the words themselves.

    `literal` is false for a value worked out at call time, and `value` is then `None`.
    """

    argument: str
    value: object
    literal: bool
    words: str


def literal(text: str) -> tuple[bool, object]:
    """`(True, value)` when `text` starts with a literal, and `(False, None)` otherwise."""
    match = re.match(_LITERAL + _END_OF_LITERAL, text)
    if match is None:
        return False, None
    token = match.group(0)
    if token in ("true", "false"):
        return True, token == "true"
    if token.startswith("'"):
        return True, token[1:-1]
    return True, float(token) if "." in token else int(token)


def stated(description: str, argument: str) -> list[Stated]:
    """The defaults an argument's own description states, in the order it states them."""
    found: list[tuple[int, Stated]] = []
    for match in _SENTENCE.finditer(description):
        is_literal, value = literal(match.group(2).strip())
        found.append((match.start(),
                      Stated(argument, value, is_literal, match.group(1).strip())))
    for pattern in _PHRASES:
        for match in pattern.finditer(description):
            found.append((match.start(),
                          Stated(argument, literal(match.group(1))[1], True, match.group(0))))
    return [statement for _, statement in sorted(found, key=lambda pair: pair[0])]


def stated_in_tool(description: str) -> list[Stated]:
    """The defaults a tool's own description states for its arguments, such as
    "(with 'subject', default 'user')"."""
    return [Stated(match.group(1), literal(match.group(2))[1], True, match.group(0))
            for match in _IN_TOOL.finditer(description)]


def same(stated_value: object, declared: object) -> bool:
    """Whether a value stated in words is the declared default.

    A boolean equals only a boolean, because in Python `True == 1`. A number equals a
    number of the same value, so 0 and 0.0 agree. A string equals the same string.
    """
    if isinstance(stated_value, bool) or isinstance(declared, bool):
        return (isinstance(stated_value, bool) and isinstance(declared, bool)
                and stated_value == declared)
    if isinstance(stated_value, (int, float)) and isinstance(declared, (int, float)):
        return stated_value == declared
    return type(stated_value) is type(declared) and stated_value == declared


def conflicts(tool: Mapping[str, Any]) -> list[tuple[str, str]]:
    """`(argument, words)` for each stated default that the schema contradicts.

    That is a literal that differs from the declared default, or a value worked out at
    call time on an argument that declares a constant, which the validator would fill in
    instead. A tool-level statement about an argument this schema does not have is left
    to the mention checks.
    """
    properties = tool["inputSchema"]["properties"]
    found = []
    for statement in _statements(tool):
        schema = properties.get(statement.argument)
        if schema is None or "default" not in schema:
            continue
        if not statement.literal or not same(statement.value, schema["default"]):
            found.append((statement.argument, statement.words))
    return found


def undeclared(tool: Mapping[str, Any]) -> list[tuple[str, str]]:
    """`(argument, words)` for each literal default stated in words that the schema does
    not declare.

    The validator fills only declared defaults, so such a default is implemented
    somewhere else, and the words and the code can drift apart without any check.
    """
    properties = tool["inputSchema"]["properties"]
    return [(statement.argument, statement.words) for statement in _statements(tool)
            if statement.literal and statement.argument in properties
            and "default" not in properties[statement.argument]]


def _statements(tool: Mapping[str, Any]) -> list[Stated]:
    """Every default a served tool states: each argument's own, in schema order, then the
    ones its tool description states."""
    found = []
    for name, schema in tool["inputSchema"]["properties"].items():
        found += stated(str(schema.get("description", "")), name)
    return found + stated_in_tool(str(tool["description"]))
