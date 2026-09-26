"""What a piece of tool text names: tools, arguments and other identifiers.

The tool descriptions and the server's instructions are plain text written for a model.
They name tools and arguments without any markup, so this module decides from the
spelling and the words around it which words are names. These are the rules.

- **A tool name** is `memory_` followed by lower-case words joined by `_`. `memory_*` is a
  wildcard, and a name inside a file path or a dotted module path is not a tool name.
- **An identifier** is a snake_case word: two or more lower-case parts joined by `_`.
  Arguments, tools and predicates are all spelled this way, so every identifier must
  resolve to one of them or be marked as example data. `unresolved_identifiers` says how.
- **An argument is named** by an identifier, by `word=value`, by one of the tool's own
  arguments written in single quotes, or by two of the tool's own arguments joined by a
  comma, "and", "or" or "/", as in "ranked and synthesize".
- **A tie** says which tool an argument belongs to: "memory_recall with include_episodes",
  "memory_end and claim_id", "memory_remember's expires_at". A one-word name ties only
  after "with" with no article; after "and" or a possessive it is read as English.

What the parse cannot see:

- A one-word argument such as `reason`, `query` or `text`, written alone as a plain word,
  is not read as a mention, because it is usually English: "a false reason" is about a
  stored reason, not about the argument. So a description that names a removed one-word
  argument alone in plain prose goes unnoticed.
- Two of a tool's own one-word arguments joined by "and" count as a mention, even in a
  sentence that means the English words.
- A snake_case word that is not a tool, an argument or a predicate is taken as example
  data when it sits inside quotes, parentheses or braces, or in the list after "like",
  "such as" or "e.g.". So an argument renamed in the schema, whose old name the text still
  gives in one of those places, is not reported: "(query_rewrite)" would pass under a new
  name for the argument.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Collection, Iterable, Mapping, TypeVar

from memvara.schema import BUILTIN_PREDICATES

T = TypeVar("T")

#: Where a name may start and end: not inside a longer word, a file path, a dotted
#: module path, a query string or a hyphenated word.
_BEFORE = r"(?<![A-Za-z0-9_./?&-])"
_AFTER = r"(?![A-Za-z0-9_-]|\.[A-Za-z])"
_WORD = r"[a-z][a-z0-9_]*"
_TOOL = r"(memory_[a-z0-9]+(?:_[a-z0-9]+)*)"
#: What joins the words of a list: a comma, "and", "or" or "/".
_JOIN = r"(?:\s*,\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+|\s*/\s*)"

TOOL_NAME = re.compile(_BEFORE + r"memory_[a-z0-9]+(?:_[a-z0-9]+)*" + _AFTER)
SNAKE = re.compile(_BEFORE + r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+" + _AFTER)
ASSIGNED = re.compile(_BEFORE + "(" + _WORD + ")=")
QUOTED = re.compile(r"(?<![\w])'(" + _WORD + r")'(?![\w])")
#: Two words joined as a list. A lookahead, so that the second word of one pair can be
#: the first word of the next, and a list of three is read whole.
PAIR = re.compile(r"(?<![\w'.-])(?=(" + _WORD + ")" + _JOIN + "(" + _WORD + r")(?![\w'-]))")

#: The phrases that tie an argument to a tool, each with whether a one-word name counts
#: in it. After "with" it counts unless an article comes first, because "memory_search
#: with a question" is English. After "and" or a possessive it never counts:
#: "memory_end's predicate" is the predicate of a fact, and "memory_end and query" joins
#: two verbs. A snake_case word always counts.
_TIES = (
    (re.compile(_BEFORE + _TOOL + r"(?:\s+again)?\s+with\s+(the\s+same\s+|the\s+|an?\s+)?("
                + _WORD + ")"), True),
    (re.compile(_BEFORE + _TOOL + r"\s+and\s+()(" + _WORD + ")"), False),
    (re.compile(_BEFORE + _TOOL + r"'s\s+()(" + _WORD + ")"), False),
)

#: The list after "like", "such as" or "e.g.": example data, not names.
_EXAMPLE_LIST = re.compile(r"\b(?:like|such as|e\.g\.,?)\s+((?:[\w'\"]+" + _JOIN
                           + r")*[\w'\"]+)")


@dataclass(frozen=True)
class Mention:
    """One place a text names an argument: the word, how it was written, and where."""

    word: str
    #: "snake" (a snake_case word), "assigned" (`word=`), "quoted" (`'word'`) or
    #: "listed" (one of two own arguments joined as a list).
    kind: str
    start: int


def tool_names(text: str) -> list[str]:
    """Every tool name in `text`, in order and with repeats."""
    return TOOL_NAME.findall(text)


def mentions(text: str, own: Collection[str]) -> list[Mention]:
    """Every place `text` names an argument, in order.

    `own` is the arguments of the tool the text belongs to. A quoted word and a listed pair
    count only for those, because a quoted word is otherwise usually a value, and a pair of
    ordinary words is usually English.
    """
    found: dict[int, Mention] = {}

    def add(word: str, kind: str, start: int) -> None:
        # The first rule to find a word at a place names it; "min_hops=2" is a snake word.
        found.setdefault(start, Mention(word, kind, start))

    for match in SNAKE.finditer(text):
        add(match.group(0), "snake", match.start())
    for match in ASSIGNED.finditer(text):
        add(match.group(1), "assigned", match.start(1))
    for match in QUOTED.finditer(text):
        if match.group(1) in own:
            add(match.group(1), "quoted", match.start(1))
    for match in PAIR.finditer(text):
        if match.group(1) in own and match.group(2) in own:
            add(match.group(1), "listed", match.start(1))
            add(match.group(2), "listed", match.start(2))
    return [found[start] for start in sorted(found)]


def ties(text: str) -> list[tuple[str, str]]:
    """Each `(tool, word)` the text ties together, in order.

    A one-word name counts only after "with" and no article, because "memory_search with a
    question" and "memory_end's predicate" are English. A snake_case word counts in every
    form, and keeps its article: "with the same custom_id".
    """
    found = []
    for pattern, one_word in _TIES:
        for match in pattern.finditer(text):
            tool, article, word = match.group(1), match.group(2), match.group(3)
            if "_" not in word and (article or not one_word):
                continue
            found.append((match.start(1), tool, word))
    return [(tool, word) for _, tool, word in sorted(found)]


def predicates() -> frozenset[str]:
    """The built-in predicates and their aliases. The shipped packs are left out, because
    reading them needs `tomllib`, which Python 3.10 does not have."""
    names: set[str] = set()
    for spec in BUILTIN_PREDICATES:
        names.add(spec.name)
        names.update(spec.aliases)
    return frozenset(names)


def unknown_tools(text: str, table: Mapping[str, Collection[str]]) -> list[str]:
    """Tool names in `text` that are neither a tool nor an argument of one.

    `memory_type` looks like a tool name and is an argument, so arguments count as known.
    """
    known = set(table) | _arguments(table)
    return _unique(name for name in tool_names(text) if name not in known)


def unresolved_identifiers(text: str, table: Mapping[str, Collection[str]]) -> list[str]:
    """Identifiers in `text` that name nothing the code has.

    A snake_case word, or a word written as `word=`, resolves when it is a tool, an
    argument of any tool, or a built-in predicate or alias. Otherwise it resolves only as
    example data: inside quotes, parentheses or braces, or in the list after "like",
    "such as" or "e.g.". A word followed by a colon labels the list after it, as "in
    snake_case:" does, and resolves too. Anything left is most likely an argument that
    was renamed or removed while the words kept its old name.
    """
    known = set(table) | _arguments(table) | predicates()
    examples = _example_spans(text)
    unresolved = []
    for mention in mentions(text, ()):
        end = mention.start + len(mention.word)
        if (mention.word in known or text[end:end + 1] == ":"
                or any(start <= mention.start < stop for start, stop in examples)):
            continue
        unresolved.append(mention.word)
    return _unique(unresolved)


def wrong_ties(text: str, table: Mapping[str, Collection[str]]) -> list[tuple[str, str]]:
    """Each `(tool, argument)` the text ties together where the tool does not take it.

    Only a snake_case word, or a one-word argument of some tool, counts as an argument. A
    tool paired with a tool ("memory_recall and memory_search") is not a tie, and a tool
    `unknown_tools` reports is left to it.
    """
    everything = _arguments(table)
    wrong = []
    for tool, word in ties(text):
        if tool not in table or word in table:
            continue
        if "_" not in word and word not in everything:
            continue
        if word not in table[tool]:
            wrong.append((tool, word))
    return _unique(wrong)


def unserved_arguments(text: str, own: Collection[str],
                       served: Collection[str]) -> list[str]:
    """Arguments of the text's own tool that the text names and the server does not take.

    `own` is every argument the tool can take on some server, and `served` is the
    arguments it takes on this one. A switch that removes an argument removes it from the
    schema, and a description that still names it sends the model to a refused call.
    """
    return _unique(mention.word for mention in mentions(text, own)
                   if mention.word in own and mention.word not in served)


def _arguments(table: Mapping[str, Collection[str]]) -> frozenset[str]:
    return frozenset(argument for arguments in table.values() for argument in arguments)


def _example_spans(text: str) -> list[tuple[int, int]]:
    """Where `text` holds example data: quotes, parentheses, braces and example lists."""
    spans = [match.span() for match in re.finditer(r"(?<![\w])'[^'\n]+'(?![\w])", text)]
    spans += [match.span() for match in re.finditer(r'"[^"\n]*"', text)]
    for opening, closing in ("()", "{}"):
        stack: list[int] = []
        for index, character in enumerate(text):
            if character == opening:
                stack.append(index)
            elif character == closing and stack:
                spans.append((stack.pop(), index + 1))
    spans += [match.span(1) for match in _EXAMPLE_LIST.finditer(text)]
    return spans


def _unique(items: Iterable[T]) -> list[T]:
    """`items` in order, each once."""
    seen: list[T] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen
