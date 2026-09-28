"""The MCP server tables in a Codex `config.toml`, read without a TOML library.

Codex keeps its MCP servers in `~/.codex/config.toml`, one `[mcp_servers.<name>]` table
per server, with the variables the server starts with either inline (`env = { K = "v" }`)
or in a sub-table (`[mcp_servers.<name>.env]`). The hooks look there for the store a
local install configures (`lib.ipc.server_env`).

`tomllib` reads it on Python 3.11 and later. The hooks also run on 3.10, which has no TOML
reader, so this reads the one part of the file the hooks need: table headers, and keys
whose value is a string, an array of strings (a server's `args`, on one line or several),
or an inline table of strings. Anything else -- numbers, booleans, multi-line strings,
arrays of tables -- is skipped, never guessed at, so a file this cannot read in full still
yields every server whose command and variables it could read.

    >>> text = '''
    ... model = "o3"  # a comment
    ... [mcp_servers.memvara]
    ... command = "python3"
    ... args = ["-m", "memvara.server"]
    ... env = { MEMVARA_DB = "~/m.db", "MEMVARA_USER" = 'me' }
    ... '''
    >>> _subset(text)["mcp_servers"]["memvara"]["env"]
    {'MEMVARA_DB': '~/m.db', 'MEMVARA_USER': 'me'}
    >>> _subset('[mcp_servers."memvara-dev".env]\\nMEMVARA_DB = "a # not a comment"\\n')
    {'mcp_servers': {'memvara-dev': {'env': {'MEMVARA_DB': 'a # not a comment'}}}}
"""

from __future__ import annotations

import json
import re


def read(text: str) -> "dict | None":
    """The document as a dict, or None when it is not TOML this can read."""
    try:
        import tomllib  # noqa: PLC0415 -- 3.11 and later; reached only for a .toml file
    except ImportError:
        tomllib = None
    try:
        return tomllib.loads(text) if tomllib is not None else _subset(text)
    # A string with an escape that is not valid, such as `\U` past U+10FFFF, fails the
    # whole document here as it does in `tomllib`.
    except (ValueError, RecursionError):
        return None


#: A bare key, or a basic or literal string used as a key.
_KEY = r'(?:[A-Za-z0-9_-]+|"(?:[^"\\]|\\.)*"|\'[^\']*\')'
_HEADER = re.compile(rf"\[\s*({_KEY}(?:\s*\.\s*{_KEY})*)\s*\]")
_PAIR = re.compile(rf"({_KEY})\s*=\s*")
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'')


def _key(token: str) -> str:
    return _string(token) if token[:1] in "\"'" else token


def _string(token: str) -> str:
    """A TOML basic string through JSON, whose escapes it shares apart from `\\U`; a
    literal string as it stands. ValueError for an escape that is not valid, as `tomllib`
    refuses it: `\\U` past U+10FFFF, or one JSON has and TOML lacks."""
    if token.startswith("'"):
        return token[1:-1]

    def wide(match: "re.Match[str]") -> str:
        point = int(match.group(1), 16)
        if point > 0x10FFFF:
            raise ValueError(f"\\U{match.group(1)} is past the last code point")
        return json.dumps(chr(point))[1:-1]

    return json.loads(re.sub(r"\\U([0-9A-Fa-f]{8})", wide, token))


def _dotted(path: str) -> "list[str]":
    return [_key(part) for part in re.findall(_KEY, path)]


def _inline(text: str) -> "dict[str, str] | None":
    """`{ K = "v", ... }` when every value is a string, else None."""
    body = text.strip()
    if not (body.startswith("{") and body.endswith("}")):
        return None
    found: "dict[str, str]" = {}
    rest = body[1:-1].strip()
    while rest:
        pair = _PAIR.match(rest)
        if pair is None:
            return None
        rest = rest[pair.end():]
        value = _STRING.match(rest)
        if value is None:
            return None
        found[_key(pair.group(1))] = _string(value.group(0))
        rest = rest[value.end():].lstrip()
        if rest.startswith(","):
            rest = rest[1:].lstrip()
        elif rest:
            return None
    return found


def _array(text: str) -> "list[str] | None":
    """`[ "a", 'b', ]`, possibly over several lines with comments, when every item is a
    string, else None."""
    rest = text.strip()
    if not rest.startswith("["):
        return None
    rest = rest[1:]
    items: "list[str]" = []
    while True:
        rest = rest.lstrip()
        while rest.startswith("#"):
            rest = rest.split("\n", 1)[1].lstrip() if "\n" in rest else ""
        if rest.startswith("]"):
            tail = rest[1:].strip()
            return items if not tail or tail.startswith("#") else None
        value = _STRING.match(rest)
        if value is None:
            return None
        items.append(_string(value.group(0)))
        rest = rest[value.end():].lstrip()
        while rest.startswith("#"):
            rest = rest.split("\n", 1)[1].lstrip() if "\n" in rest else ""
        if rest.startswith(","):
            rest = rest[1:]
        elif not rest.startswith("]"):
            return None


def _value(text: str) -> "str | list[str] | dict[str, str] | None":
    """A string, an array of strings or an inline table of strings, with any trailing
    comment dropped."""
    if text.lstrip().startswith("["):
        return _array(text)
    value = _STRING.match(text)
    if value is not None:
        tail = text[value.end():].strip()
        return _string(value.group(0)) if not tail or tail.startswith("#") else None
    if text.startswith("{"):
        # The table ends at the last "}" that is not inside a string.
        depth, end, index = 0, -1, 0
        while index < len(text):
            char = text[index]
            if char in "\"'":
                match = _STRING.match(text, index)
                if match is None:
                    return None
                index = match.end()
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
            index += 1
        if end < 0:
            return None
        tail = text[end + 1:].strip()
        return _inline(text[:end + 1]) if not tail or tail.startswith("#") else None
    return None


def _closed(text: str) -> bool:
    """Whether every "[" in `text` outside a string or a comment has its "]"."""
    depth, index = 0, 0
    while index < len(text):
        char = text[index]
        if char in "\"'":
            match = _STRING.match(text, index)
            if match is None:
                return True  # not a string this reads; let the value be refused
            index = match.end()
            continue
        if char == "#":
            newline = text.find("\n", index)
            if newline < 0:
                break
            index = newline
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        index += 1
    return depth <= 0


def _subset(text: str) -> dict:
    """The tables and the string, array or inline-table values in `text`; see the module
    note."""
    document: dict = {}
    table = document
    lines = iter(text.splitlines())
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[["):
            table = {}  # an array of tables; its keys are not ours to read
            continue
        header = _HEADER.match(line)
        if header is not None:
            table = document
            for part in _dotted(header.group(1)):
                nested = table.setdefault(part, {})
                table = nested if isinstance(nested, dict) else {}
            continue
        pair = _PAIR.match(line)
        if pair is None:
            continue
        rest = line[pair.end():]
        if rest.startswith("["):
            # An array may run over several lines; read on until it closes.
            while not _closed(rest):
                following = next(lines, None)
                if following is None:
                    break
                rest += "\n" + following
        value = _value(rest)
        if value is not None:
            table[_key(pair.group(1))] = value
    return document
