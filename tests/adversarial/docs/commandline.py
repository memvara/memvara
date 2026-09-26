"""The command lines memvara installs, what each one accepts, and what its help says.

The truth is read from the code with `ast`, never from a list kept by hand.

- **The command lines** are the `[project.scripts]` entries in `pyproject.toml`, plus
  `python -m memvara.server`, and each subcommand their dispatch code reaches: a branch
  `args[0] == "word"` that returns a call, or a table the dispatch looks the word up in
  with `args[0] in TABLE`.
- **The help** of a command is the `*USAGE` text it prints in a branch that tests its
  arguments. The words that branch tests, such as `--help`, `-h` and `help`, print the help
  itself, so the help does not have to list them. A wrapper with no such branch, such as
  `memvara login`, is followed to the function it returns a call to.
- **The words a console script accepts** are every string its dispatch compares its
  arguments with, and the keys of a table it looks them up in.
- **The options a subcommand accepts** are the string constants that are exactly an option
  (`-x` or `--name`) in its own function, in every function it hands its argument list to,
  in whatever module that function lives, and in the module-level tuples those functions
  read. A hand-over the reader cannot follow is refused rather than skipped, because the
  code on the other side may accept options nobody can list.
- **The variables the configuration reads** are the `MEMVARA_*` names
  `memvara/server/config.py` passes to `env.get`, with a module-level name such as
  `KEY_ENV` resolved to its value, plus one variable per feature for each prefix it scans
  for with `startswith`.

`pyproject.toml` is read with a regular expression, because `tomllib` arrived in Python
3.11 and the suite runs on 3.10.
"""

from __future__ import annotations

import ast
import builtins
import functools
import importlib
import inspect
import re
import textwrap
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Collection, Iterable, Mapping

from memvara.server import config
from memvara.server.config import FEATURES

from harness.env import REPO

_OPTION = re.compile(r"^--?[a-z][a-z0-9-]*$")
_ARGV = ("args", "argv")
_VARIABLE = re.compile(r"\bMEMVARA_[A-Z0-9_]+")
#: A line that starts a variable's paragraph in a help text: two spaces, then the name.
_BLOCK_START = re.compile(r"^  (MEMVARA_[A-Z0-9_]+)")
#: An upper-case name in the feature paragraph: one with an underscore, or one written
#: as `NAME=0` or `NAME=1`. A plain acronym such as "OS" is neither.
_FEATURE = re.compile(r"(?<![\w<])([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+|[A-Z][A-Z0-9]+(?==[01]))"
                      r"(?![\w>])")
_LIST_SPLIT = re.compile(r"\s*,\s*(?:and\s+)?|\s+and\s+")
_HIDES = re.compile(r"\b([A-Z][A-Z0-9_]*)=0\s+hides\s+"
                    r"(memory_[a-z_]+(?:(?:\s*,\s*|\s+and\s+)memory_[a-z_]+)*)")
_REMOVES = re.compile(r"\b([A-Z][A-Z0-9_]*)=0\s+removes\s+the\s+"
                      r"([a-z_]+(?:(?:\s*,\s*|\s+and\s+)[a-z_]+)*)\s+arguments?\b")
_STATED = (re.compile(r"'([^'\n]*)' \(default\b"),
           re.compile(r"\bDefault '([^'\n]*)'"),
           re.compile(r"\bUnset means (-?\d+(?:\.\d+)?)\b"))
_DEFAULT_OFF = re.compile(r"Every feature is on by default(.*?)\.", re.DOTALL)


@dataclass(frozen=True)
class CommandLine:
    """One command a person can type: its name, the function that runs it, the words it
    accepts, the words that print its help, and the help itself."""

    name: str
    function: Callable[..., Any]
    accepted: frozenset[str]
    help_words: frozenset[str]
    help: str
    #: True for a console script, false for one of its subcommands.
    top: bool


def console_scripts() -> dict[str, Callable[..., Any]]:
    """Each console script `pyproject.toml` declares, and `python -m memvara.server`,
    with the function it runs."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    section = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    found: dict[str, Callable[..., Any]] = {}
    for name, target in re.findall(r'^([A-Za-z0-9_.-]+)\s*=\s*"([^"]+)"', section,
                                   re.MULTILINE):
        module, _, attribute = target.partition(":")
        found[name] = getattr(importlib.import_module(module), attribute)
    found["python -m memvara.server"] = importlib.import_module(
        "memvara.server.__main__").main
    return found


@functools.lru_cache(maxsize=None)
def command_lines() -> tuple[CommandLine, ...]:
    """Every console script and every subcommand its dispatch reaches.

    A script that runs the same function as one already listed is the same command line
    under another spelling, so it is listed once: `python -m memvara.server` runs
    `memvara-mcp`'s code.
    """
    found: list[CommandLine] = []
    functions: list[Callable[..., Any]] = []
    for name, function in console_scripts().items():
        if any(function is other for other in functions):
            continue
        functions.append(function)
        found.append(read_script(name, function))
        found += [read_subcommand(f"{name} {word}", subcommand)
                  for word, subcommand in subcommands(function).items()]
    return tuple(found)


def read_script(name: str, function: Callable[..., Any]) -> CommandLine:
    """A console script, read from its dispatch function."""
    accepted: set[str] = set()
    for node in ast.walk(_tree(function)):
        if isinstance(node, ast.Compare) and _uses_argv(node):
            accepted |= _strings(node, function.__globals__)
    branch = _help_branch(function)
    if branch is None:
        raise LookupError(f"{name}: {function.__qualname__} prints no *USAGE help")
    return CommandLine(name, function, frozenset(accepted), *branch, top=True)


def read_subcommand(name: str, function: Callable[..., Any]) -> CommandLine:
    """A subcommand, read from the first function on its path that prints a help."""
    target, branch = function, _help_branch(function)
    seen = {function}
    while branch is None:
        following = _delegate(target)
        if following is None or following in seen:
            raise LookupError(f"{name}: neither {function.__qualname__} nor anything it "
                              "hands its arguments to prints a *USAGE help")
        seen.add(following)
        target, branch = following, _help_branch(following)
    return CommandLine(name, target, frozenset(_options(target, set())), *branch, top=False)


def subcommands(function: Callable[..., Any]) -> dict[str, Callable[..., Any]]:
    """Each word a dispatch function sends to another function, with that function."""
    found: dict[str, Callable[..., Any]] = {}
    for node in ast.walk(_tree(function)):
        if not isinstance(node, ast.If):
            continue
        for test in ast.walk(node.test):
            if not (isinstance(test, ast.Compare) and isinstance(test.left, ast.Subscript)
                    and _uses_argv(test.left)):
                continue
            operator, right = test.ops[0], test.comparators[0]
            if (isinstance(operator, ast.Eq) and isinstance(right, ast.Constant)
                    and isinstance(right.value, str)):
                callee = next((_lookup(statement.value.func, function, node)
                               for statement in ast.walk(node)
                               if isinstance(statement, ast.Return)
                               and isinstance(statement.value, ast.Call)), None)
                if callable(callee):
                    found[right.value] = callee
            elif isinstance(operator, ast.In) and isinstance(right, ast.Name):
                table = function.__globals__.get(right.id)
                if isinstance(table, Mapping):
                    found.update({word: callee for word, callee in table.items()
                                  if isinstance(word, str) and callable(callee)})
    return found


def undocumented(command: CommandLine) -> list[str]:
    """The words a command accepts that are neither help words nor named in its help."""
    return sorted(word for word in command.accepted - command.help_words
                  if not re.search(r"(?<![\w-])" + re.escape(word) + r"(?![\w-])",
                                   command.help))


def reads_in(module: ModuleType, suffixes: Iterable[str]) -> frozenset[str]:
    """The `MEMVARA_*` variables `module` reads: each name it passes to `env.get`, and
    each prefix it scans for with `startswith` followed by each of `suffixes`."""
    namespace = vars(module)
    found: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.args):
            continue
        value = _value(node.args[0], namespace)
        if not (isinstance(value, str) and value.startswith("MEMVARA_")):
            continue
        if (node.func.attr == "get" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "env"):
            found.add(value)
        elif node.func.attr == "startswith":
            found |= {value + suffix.upper() for suffix in suffixes}
    return frozenset(found)


@functools.lru_cache(maxsize=None)
def config_reads() -> frozenset[str]:
    """The `MEMVARA_*` variables `memvara/server/config.py` reads."""
    return reads_in(config, FEATURES)


def variables(help: str) -> set[str]:
    """Each `MEMVARA_*` name a help text gives. `MEMVARA_FEATURE_<NAME>` comes back as the
    prefix `MEMVARA_FEATURE_`."""
    return set(_VARIABLE.findall(help))


def unread_variables(help: str, reads: Collection[str]) -> list[str]:
    """The variables a help text names that are not in `reads`. A prefix is read when a
    variable in `reads` starts with it."""
    return [name for name in sorted(variables(help))
            if not (any(read.startswith(name) for read in reads) if name.endswith("_")
                    else name in reads)]


def features(help: str) -> set[str]:
    """The feature names the `MEMVARA_FEATURE_<NAME>` paragraph of a help text gives."""
    paragraph = "\n".join(text for name, text in _paragraphs(help)
                          if name == "MEMVARA_FEATURE_")
    return {name for name in _FEATURE.findall(paragraph) if not name.startswith("MEMVARA_")}


def nonexistent_features(help: str, names: Collection[str]) -> list[str]:
    """The feature names a help text gives that are not among `names`."""
    known = {name.upper() for name in names}
    return sorted(features(help) - known)


def hides(help: str) -> list[tuple[str, tuple[str, ...]]]:
    """`(feature, tools)` for each "NAME=0 hides memory_x and memory_y" in a help text."""
    return [(match.group(1).lower(), tuple(_LIST_SPLIT.split(match.group(2))))
            for match in _HIDES.finditer(help)]


def removes(help: str) -> list[tuple[str, tuple[str, ...]]]:
    """`(feature, arguments)` for each "NAME=0 removes the a and b arguments"."""
    return [(match.group(1).lower(), tuple(_LIST_SPLIT.split(match.group(2))))
            for match in _REMOVES.finditer(help)]


def false_claims(help: str, listed: Mapping[str, Collection[str]],
                 taken: Mapping[str, Collection[str]]) -> list[tuple[str, str]]:
    """`(feature, name)` for each claim in a help text that switching a feature off hides
    a tool or removes an argument, where the servers say otherwise.

    `listed` maps a configuration's label to the tools it lists, and `taken` to the
    arguments its tools take. A claim holds when the server with that feature switched on
    has the name and the server with it switched off does not. A claim about a feature
    that has neither server does not hold.
    """
    wrong = []
    for claims, present in ((hides(help), listed), (removes(help), taken)):
        for feature, names in claims:
            on, off = _on_and_off(feature, present)
            wrong += [(feature, name) for name in names
                      if on is None or off is None or name not in on or name in off]
    return wrong


def _on_and_off(feature: str, present: Mapping[str, Collection[str]]
                ) -> tuple[Collection[str] | None, Collection[str] | None]:
    """What the server with `feature` switched on has, and what the one with it switched
    off has. For a feature that is on by default those are the default server and
    "<feature> off"; for one that is off by default, "<feature> on" and the default."""
    if f"{feature} off" in present:
        return present.get("default"), present[f"{feature} off"]
    if f"{feature} on" in present:
        return present[f"{feature} on"], present.get("default")
    return None, None


def variable_defaults(help: str) -> list[tuple[str, str]]:
    """`(variable, value)` for each default a help text states for a variable: "'V'
    (default", "Default 'V'" or "Unset means N"."""
    found = []
    for name, text in _paragraphs(help):
        if name.endswith("_"):
            continue
        stated = sorted((match.start(), match.group(1))
                        for pattern in _STATED for match in pattern.finditer(text))
        found += [(name, value) for _, value in stated]
    return found


def default_off(help: str) -> frozenset[str] | None:
    """The features a help text says are off by default, from "Every feature is on by
    default except A and B.", or None when it does not say."""
    match = _DEFAULT_OFF.search(help)
    if match is None:
        return None
    return frozenset(name.lower()
                     for name in re.findall(r"[A-Z][A-Z0-9_]*[A-Z0-9]", match.group(1)))


def _paragraphs(help: str) -> list[tuple[str, str]]:
    """`(variable, text)` for each variable's paragraph in a help text: a line that starts
    with two spaces and the name, and the more deeply indented lines after it."""
    found: list[list[str]] = []
    current: list[str] | None = None
    for line in help.replace("\r\n", "\n").split("\n"):
        start = _BLOCK_START.match(line)
        if start:
            current = [start.group(1), line[start.end():]]
            found.append(current)
        elif current is not None and line.startswith("   "):
            current[1] += "\n" + line
        else:
            current = None
    return [(name, text) for name, text in found]


def _tree(function: Callable[..., Any]) -> ast.AST:
    return ast.parse(textwrap.dedent(inspect.getsource(function)))


def _uses_argv(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Name) and child.id in _ARGV for child in ast.walk(node))


def _strings(node: ast.AST, namespace: Mapping[str, Any]) -> set[str]:
    """The string constants in `node`, and the members of each module-level table or
    tuple it names."""
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            found.add(child.value)
        elif isinstance(child, ast.Name):
            value = namespace.get(child.id)
            if isinstance(value, (Mapping, tuple, list, set, frozenset)):
                found |= {item for item in value if isinstance(item, str)}
    return found


def _help_branch(function: Callable[..., Any]) -> tuple[frozenset[str], str] | None:
    """The words that make `function` print its help, and the help, or None."""
    for node in ast.walk(_tree(function)):
        if not (isinstance(node, ast.If) and _uses_argv(node.test)):
            continue
        for statement in node.body:
            call = statement.value if isinstance(statement, ast.Expr) else None
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "print" and call.args
                    and isinstance(call.args[0], ast.Name)
                    and call.args[0].id.endswith("USAGE")):
                text = function.__globals__.get(call.args[0].id)
                if isinstance(text, str):
                    return frozenset(_strings(node.test, function.__globals__)), text
    return None


def _delegate(function: Callable[..., Any]) -> Callable[..., Any] | None:
    """The function a wrapper returns a call to, or None."""
    tree = _tree(function)
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            callee = _lookup(node.value.func, function, tree)
            if inspect.isfunction(callee):
                return callee
    return None


def _options(function: Callable[..., Any], seen: set[Any]) -> set[str]:
    """The option tokens `function` uses, and those of every function it hands its
    argument list to, in whatever module that function lives.

    A hand-over that cannot be followed raises `LookupError`, because the code on the other
    side may accept options nobody can list, and reporting none would pass on nothing. A
    built-in such as `list(argv)` parses no options and is skipped.
    """
    if function in seen:
        return set()
    seen.add(function)
    tree = _tree(function)
    found = {text for text in _strings(tree, function.__globals__) if _OPTION.match(text)}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _hands_over_argv(node)):
            continue
        callee = _lookup(node.func, function, tree)
        if _is_builtin(callee):
            continue
        if not inspect.isfunction(callee):
            raise LookupError(
                f"{function.__qualname__} hands its arguments to {ast.unparse(node.func)}, "
                "which cannot be read, so the options it accepts are unknown")
        found |= _options(callee, seen)
    return found


def _hands_over_argv(call: ast.Call) -> bool:
    """Whether a call is given the whole argument list or a slice of it, rather than one
    element of it such as `args[0]`."""
    return any(_is_argv(value)
               for value in [*call.args, *(keyword.value for keyword in call.keywords)])


def _is_argv(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _ARGV
    return (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice)
            and _is_argv(node.value))


def _is_builtin(value: object) -> bool:
    """Whether `value` is a built-in function or type, such as `len` or `list`."""
    return inspect.isbuiltin(value) or (inspect.isclass(value)
                                        and value.__module__ == "builtins")


def _lookup(node: ast.AST, function: Callable[..., Any], scope: ast.AST) -> object:
    """What a name or a dotted name used in `function` refers to, or None.

    An import inside `scope` is looked at first, then the module's globals, then the
    built-ins. A dotted name is followed only through modules.
    """
    if isinstance(node, ast.Attribute):
        owner = _lookup(node.value, function, scope)
        return getattr(owner, node.attr, None) if inspect.ismodule(owner) else None
    if not isinstance(node, ast.Name):
        return None
    for statement in ast.walk(scope):
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            for alias in statement.names:
                if (alias.asname or alias.name.partition(".")[0]) == node.id:
                    return _imported(statement, alias, function)
    if node.id in function.__globals__:
        return function.__globals__[node.id]
    return getattr(builtins, node.id, None)


def _imported(statement: ast.Import | ast.ImportFrom, alias: ast.alias,
              function: Callable[..., Any]) -> object:
    """What an import statement inside `function` binds `alias` to."""
    if isinstance(statement, ast.Import):
        # `import a.b` binds `a`, and `import a.b as c` binds `c` to `a.b`.
        return importlib.import_module(alias.name if alias.asname
                                       else alias.name.partition(".")[0])
    package = function.__module__.rpartition(".")[0]
    module = importlib.import_module("." * statement.level + (statement.module or ""),
                                     package if statement.level else None)
    return getattr(module, alias.name, None)


def _value(node: ast.AST, namespace: Mapping[str, Any]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return namespace.get(node.id)
    return None
