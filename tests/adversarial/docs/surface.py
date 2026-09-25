"""The servers the documentation checks read their tool lists from.

A switch can hide a tool, remove an argument or rewrite a description, so a description
that is right on the default server can be wrong on another. Each check therefore runs
over every configuration a server can be started in: the default server, each feature
switched away from its default, a read-only server and a server that anchors by default.
The list is built from `FEATURE_DEFAULTS`, so a new switch is checked without anyone
adding it here.

The servers run in this process over an in-memory store, and answer `tools/list` through
the same `handle_message` the stdio loop calls. Starting a real process for each of them
would cost a second apiece and tell these checks nothing more.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Mapping

from memvara.server.config import FEATURE_DEFAULTS, FEATURES_OFF_BY_DEFAULT
from memvara.server.mcp import MemvaraMCPServer
from memvara.server.tools import TOOLS

from harness import stores


@dataclass(frozen=True)
class Configuration:
    """How one server is started: a label, the features that are off, and two flags."""

    label: str
    features_off: frozenset[str]
    read_only: bool = False
    anchored: bool = False


def configurations() -> tuple[Configuration, ...]:
    """The default server, each feature switched away from its default, a read-only
    server, and a server that anchors by default."""
    found = [Configuration("default", FEATURES_OFF_BY_DEFAULT)]
    for feature, on in FEATURE_DEFAULTS.items():
        if on:
            found.append(Configuration(f"{feature} off", FEATURES_OFF_BY_DEFAULT | {feature}))
        else:
            found.append(Configuration(f"{feature} on", FEATURES_OFF_BY_DEFAULT - {feature}))
    found.append(Configuration("read-only", FEATURES_OFF_BY_DEFAULT, read_only=True))
    found.append(Configuration("anchored", FEATURES_OFF_BY_DEFAULT, anchored=True))
    return tuple(found)


@functools.lru_cache(maxsize=None)
def served(configuration: Configuration) -> tuple[dict[str, Any], ...]:
    """The tools a server started this way lists, as its `tools/list` answer gives them."""
    server = MemvaraMCPServer(stores.memory(), user="u",
                              features_off=configuration.features_off,
                              read_only=configuration.read_only,
                              anchored=configuration.anchored)
    try:
        reply = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    finally:
        server.close()
    assert reply is not None, "tools/list is a request, so it must be answered"
    return tuple(reply["result"]["tools"])


def table() -> dict[str, frozenset[str]]:
    """Every tool and all of its arguments, including the ones a switch can remove."""
    return {tool.name: frozenset(tool.properties) for tool in TOOLS}


def texts(tool: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Each description in one served tool, labelled with where it came from: the tool's
    name for its own description, and `tool.argument` for an argument's."""
    found = [(str(tool["name"]), str(tool["description"]))]
    for name, schema in tool["inputSchema"]["properties"].items():
        found.append((f"{tool['name']}.{name}", str(schema.get("description", ""))))
    return found
