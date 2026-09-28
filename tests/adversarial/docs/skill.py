"""What the packaged skill names: tools, calls to them, and the arguments it ties to them.

The skill is Markdown, and it marks names as code with backticks, so names are read from
backticks and calls from Python syntax. These are the rules.

- **A tool name** is read the way `mentions.tool_names` reads one, anywhere in the text.
- **A call** is `memory_x(` followed by arguments, as a code example writes it. It is read
  with `ast`, so a call spread over several lines, or with a parenthesis inside a string,
  is read whole. A call after `#` on the same line is in a comment, not an example: the
  skill writes its counter-examples that way.
- **A tie** says which tool a backticked argument belongs to: "`tool` with `a`", "`tool`
  takes `a` and `b`", "`tool` takes those ids in `a`", "`tool` (optional `a`)", "`tool`
  and `a`", "`a` on `tool`", and "`tool` and `tool` both take ...", which ties every
  backticked argument up to the end of that clause to each of the tools.
- **What the text calls an argument**: "argument is `x`", "argument `x`" or "`x` argument".

A backticked word with no tie is not checked, because the skill also names the Python
library's parameters, such as `known_at`, which no tool takes. Windows line endings are
read the same as Unix ones.
"""

from __future__ import annotations

import ast
import pathlib
import re
import warnings
from dataclasses import dataclass
from typing import Collection, Mapping

import memvara

SKILL = pathlib.Path(memvara.__file__).resolve().parent / "skills" / "memvara"

_CALL_START = re.compile(r"(?<![\w.])(memory_[a-z0-9]+(?:_[a-z0-9]+)*)\(")
_TOOL = r"`memory_[a-z0-9]+(?:_[a-z0-9]+)*`"
#: A backticked identifier, alone or used, as in `filters: {"team": "support"}`.
_ARG = r"`[a-z][a-z0-9_]*(?:\s*[:=][^`]*)?`"
_JOIN = r"(?:\s*[,/]\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+)"
_LIST = _ARG + "(?:" + _JOIN + _ARG + ")*"
#: Each form puts the tool in group 1 and the argument or list of arguments in group 2.
_TOOL_FIRST = (
    re.compile("(" + _TOOL + r")\s+(?:with|takes|take)\s+(?:(?:[a-z]+\s+){1,3}in\s+)?("
               + _LIST + ")"),
    re.compile("(" + _TOOL + r")\s+\(optional\s+(" + _LIST + ")"),
    re.compile("(" + _TOOL + r")\s+and\s+(" + _ARG + ")"),
)
_ON = re.compile("(" + _ARG + r")\s+on\s+(" + _TOOL + ")")
_BOTH = re.compile("(" + _TOOL + r"(?:\s+and\s+" + _TOOL + r")+)\s+both\s+take\s+")
_NAME = re.compile(r"`([a-z][a-z0-9_]*)")
_CALLED = (re.compile(r"\barguments?\s+(?:is\s+|are\s+)?`([a-z][a-z0-9_]*)`"),
           re.compile(r"`([a-z][a-z0-9_]*)`\s+arguments?\b"))


@dataclass(frozen=True)
class Call:
    """One call in a code example: the tool, its keyword arguments, and its line."""

    tool: str
    keywords: tuple[str, ...]
    line: int


def skill_files() -> list[pathlib.Path]:
    """`SKILL.md`, then each reference page, in name order."""
    return [SKILL / "SKILL.md", *sorted((SKILL / "references").glob("*.md"))]


def calls(text: str) -> tuple[list[Call], list[str]]:
    """The calls in `text`, and the start of each one that Python cannot read."""
    text = _lf(text)
    found, unreadable = [], []
    for match in _CALL_START.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        if "#" in text[line_start:match.start()]:
            continue
        call = _read_call(text, match.start())
        if call is None:
            line_end = text.find("\n", match.start())
            unreadable.append(text[match.start():line_end if line_end != -1 else None]
                              .strip())
            continue
        found.append(Call(match.group(1),
                          tuple(k.arg for k in call.keywords if k.arg is not None),
                          text.count("\n", 0, match.start()) + 1))
    return found, unreadable


def ties(text: str) -> list[tuple[str, str]]:
    """Each `(tool, argument)` the text ties together, once, in order.

    A tool paired with a tool ("`memory_recall` and `memory_search`") comes back too;
    `wrong_ties` leaves it out, because only the tool table can say which names are tools.
    """
    text = _lf(text)
    found: list[tuple[int, str, str]] = []
    for pattern in _TOOL_FIRST:
        for match in pattern.finditer(text):
            tool = _names(match.group(1))[0]
            found += [(match.start(), tool, argument) for argument in _names(match.group(2))]
    for match in _ON.finditer(text):
        found.append((match.start(), _names(match.group(2))[0], _names(match.group(1))[0]))
    for match in _BOTH.finditer(text):
        tools = _names(match.group(1))
        found += [(match.start(), tool, argument)
                  for argument in _names(_clause(text, match.end())) for tool in tools]
    return list(dict.fromkeys((tool, argument) for _, tool, argument
                              in sorted(found, key=lambda item: item[0])))


def called_arguments(text: str) -> list[str]:
    """Each backticked word the text calls an argument, once, in order."""
    text = _lf(text)
    found = sorted((match.start(), match.group(1))
                   for pattern in _CALLED for match in pattern.finditer(text))
    return list(dict.fromkeys(word for _, word in found))


def wrong_calls(text: str, table: Mapping[str, Collection[str]]) -> list[tuple[str, str]]:
    """`(tool, argument)` for each keyword argument a call passes that its tool does not
    take. A call to a tool that does not exist is left to `mentions.unknown_tools`."""
    return list(dict.fromkeys((call.tool, keyword) for call in calls(text)[0]
                              if call.tool in table
                              for keyword in call.keywords
                              if keyword not in table[call.tool]))


def wrong_ties(text: str, table: Mapping[str, Collection[str]]) -> list[tuple[str, str]]:
    """`(tool, argument)` for each tie where the tool does not take the argument.

    Unlike a tie in plain text, a backticked one is unambiguous, so an argument no tool
    takes is reported too: it is most likely a name that has been changed since.
    """
    return [(tool, argument) for tool, argument in ties(text)
            if tool in table and argument not in table and argument not in table[tool]]


def unknown_arguments(text: str, table: Mapping[str, Collection[str]]) -> list[str]:
    """Each word the text calls an argument that no tool takes."""
    everything = {argument for arguments in table.values() for argument in arguments}
    return [word for word in called_arguments(text) if word not in everything]


def _read_call(text: str, start: int) -> ast.Call | None:
    """The shortest piece of `text` from `start` that Python reads as one call, or None."""
    end = text.find(")", start)
    while end != -1:
        with warnings.catch_warnings():
            # An example's string may hold a backslash Python would warn about.
            warnings.simplefilter("ignore")
            try:
                node = ast.parse(text[start:end + 1], mode="eval").body
            except SyntaxError:
                end = text.find(")", end + 1)
                continue
        return node if isinstance(node, ast.Call) else None
    return None


def _clause(text: str, start: int) -> str:
    """`text` from `start` to the end of its clause: a semicolon, a full stop followed by
    a space, or a blank line, outside backticks."""
    inside = False
    for index in range(start, len(text)):
        character = text[index]
        if character == "`":
            inside = not inside
        elif not inside and (character == ";" or text.startswith("\n\n", index)
                             or (character == "." and text[index + 1:index + 2] in (" ", "\n", ""))):
            return text[start:index]
    return text[start:]


def _names(span: str) -> list[str]:
    """The identifiers at the start of each backticked name in `span`."""
    return _NAME.findall(span)


def _lf(text: str) -> str:
    return text.replace("\r\n", "\n")
