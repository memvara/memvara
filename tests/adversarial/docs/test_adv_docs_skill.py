"""The packaged skill names only tools and arguments that exist.

The skill ships in the wheel, `memvara-mcp init` writes it into projects, and seven
plugin repositories vendor it. It routes a model between tools by name and shows calls to
copy, so a name it gets wrong sends the model to a call that fails. Its reference pages
ship with it, so they are checked too.
"""

from __future__ import annotations

import functools

from .mentions import tool_names, unknown_tools
from .planted import TABLE as PLANTED
from .skill import (Call, called_arguments, calls, skill_files, ties, unknown_arguments,
                    wrong_calls, wrong_ties)
from .surface import table


# -- the parse, on planted text ----------------------------------------------------------

def test_a_call_with_an_argument_the_tool_does_not_take_is_reported() -> None:
    text = '    memory_search(query="x", k=5, depth=2)\n'
    assert wrong_calls(text, PLANTED) == [("memory_search", "depth")]


def test_a_commented_out_call_is_not_an_example() -> None:
    text = ("    memory_standing()\n"
            "    # avoid: memory_standing(query=...) — there is no such argument\n")
    assert calls(text) == ([Call("memory_standing", (), 1)], [])
    assert wrong_calls(text, PLANTED) == []


def test_a_call_over_several_lines_is_read_whole() -> None:
    """The first `)` is inside a string, so a reader that stopped there would lose the
    second argument."""
    text = ('Intro.\n\n    memory_add(text="I moved (again) to Lisbon",\n'
            '               role="system")\n')
    assert calls(text) == ([Call("memory_add", ("text", "role"), 3)], [])


def test_a_call_that_cannot_be_read_is_reported() -> None:
    assert calls('Then:\n    memory_add(text="never closed\n') == (
        [], ['memory_add(text="never closed'])


def test_each_tie_form_is_read() -> None:
    text = ("`memory_remember` takes those ids in `sources`. `role` on `memory_add` is "
            "where it gets settled. Close the old interval with `memory_end` and `at`. "
            "`memory_end` (optional `at`) or `memory_remember` with `true_since` /\n"
            "     `true_until`. `memory_recall` and `memory_search` both take `filters`, as "
            "in `filters: {\"team\": \"support\"}`, and `filepath_prefix`.")
    assert [pair for pair in ties(text) if pair[1] not in PLANTED] == [
        ("memory_remember", "sources"), ("memory_add", "role"), ("memory_end", "at"),
        ("memory_remember", "true_since"), ("memory_remember", "true_until"),
        ("memory_recall", "filters"), ("memory_search", "filters"),
        ("memory_recall", "filepath_prefix"), ("memory_search", "filepath_prefix")]
    assert wrong_ties(text, PLANTED) == []


def test_a_negated_argument_is_not_tied() -> None:
    text = "On **MCP**: `memory_search` takes `as_of` and `valid_at`, not `known_at`."
    assert ties(text) == [("memory_search", "as_of"), ("memory_search", "valid_at")]


def test_an_argument_beside_another_tool_is_not_tied_to_it() -> None:
    text = ("If they named the date it stopped, pass that as `true_since` on the new fact "
            "(or close the old interval with `memory_end` and `at`).")
    assert ties(text) == [("memory_end", "at")]


def test_a_wrong_or_stale_tie_is_reported() -> None:
    text = ("`memory_recall` takes `as_of` and `valid_at`. Close it with `memory_end` "
            "with `true_since`. `memory_remember` takes `valid_from`.")
    assert wrong_ties(text, PLANTED) == [("memory_recall", "as_of"),
                                         ("memory_end", "true_since"),
                                         ("memory_remember", "valid_from")]


def test_what_the_text_calls_an_argument_must_be_one() -> None:
    text = ("What it recognises it writes, and the deciding argument is `role`. The "
            "`mode` argument is gone.")
    assert called_arguments(text) == ["role", "mode"]
    assert unknown_arguments(text, PLANTED) == ["mode"]


def test_windows_line_endings_read_the_same() -> None:
    """A checkout on Windows can turn every line ending into CRLF."""
    for path in skill_files():
        text = path.read_text(encoding="utf-8")
        crlf = text.replace("\n", "\r\n")
        assert calls(crlf) == calls(text), path.name
        assert ties(crlf) == ties(text), path.name
        assert called_arguments(crlf) == called_arguments(text), path.name
        assert tool_names(crlf) == tool_names(text), path.name


# -- the real skill ----------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _pages() -> tuple[tuple[str, str], ...]:
    """Each page of the packaged skill, by name, with its text."""
    return tuple((path.name, path.read_text(encoding="utf-8")) for path in skill_files())


def test_every_tool_the_skill_names_exists() -> None:
    problems = [f"{page} names {name}, which is not a tool"
                for page, text in _pages() for name in unknown_tools(text, table())]
    assert not problems, "\n".join(problems)
    assert tool_names(_pages()[0][1]), "no tool name was read in SKILL.md"


def test_every_call_in_the_skill_names_real_arguments() -> None:
    problems = []
    found = 0
    for page, text in _pages():
        read, unreadable = calls(text)
        found += len(read)
        problems += [f"{page}: {start!r} cannot be read as a call" for start in unreadable]
        problems += [f"{page}: a call passes {tool} the argument {argument!r}, which it "
                     "does not take" for tool, argument in wrong_calls(text, table())]
    assert not problems, "\n".join(problems)
    assert found, "no call was read in the skill"


def test_an_argument_the_skill_ties_to_a_tool_belongs_to_it() -> None:
    problems = [f"{page} ties {argument!r} to {tool}, which does not take it"
                for page, text in _pages() for tool, argument in wrong_ties(text, table())]
    assert not problems, "\n".join(problems)
    assert any(ties(text) for _, text in _pages()), "no tie was read in the skill"


def test_what_the_skill_calls_an_argument_is_one() -> None:
    problems = [f"{page} calls {word!r} an argument, and no tool takes it"
                for page, text in _pages() for word in unknown_arguments(text, table())]
    assert not problems, "\n".join(problems)
    assert any(called_arguments(text) for _, text in _pages()), (
        "nothing the skill calls an argument was read")
