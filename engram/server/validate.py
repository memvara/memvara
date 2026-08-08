r"""Argument checking for tool calls, because there is no SDK doing it for us.

A model fills these arguments in from a JSON Schema it read once, and it gets them wrong
in a small set of recoverable ways: an argument name borrowed from a sibling tool, a
number sent as a string, an enum value it invented. Every rejection here is phrased as
something the caller can act on, and every one comes back as a tool *result* rather than
a JSON-RPC error — the model sees results, whereas protocol errors are addressed to the
client, which typically renders them as a failed call and moves on.

The validated subset of JSON Schema is exactly what the tools in this package declare:
`type` (string/integer/number/array), `enum`, `minimum`, `maximum`, `default`,
`required`, and `additionalProperties: false`. Anything wider would be untested code in
a validator, which is the one place that is not acceptable.

>>> validate({"k": {"type": "integer", "default": 8}}, (), {}, tool="demo")
{'k': 8}
>>> validate({"k": {"type": "integer"}}, (), {"k": "8"}, tool="demo")
Traceback (most recent call last):
    ...
engram.server.validate.ToolError: demo.k must be an integer, got a string ('8')
"""

from __future__ import annotations

import difflib
from typing import Any, Mapping, Sequence

__all__ = ["ToolError", "validate"]


class ToolError(Exception):
    """A tool call that cannot proceed, phrased for the model that made it.

    Raised by the validator and by handlers alike. The server turns it into a normal
    tool result with `isError: true`, so the text of the message is prompt content: it
    should say what to do differently, not just what went wrong.
    """


#: Article-carrying names, so the message reads as a sentence rather than as a type dump.
_ARTICLES = {
    "string": "a string",
    "integer": "an integer",
    "number": "a number",
    "array": "an array",
}


def _describe(value: Any) -> str:
    """What the caller actually sent, named the way the schema names things."""
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, int):
        return "an integer"
    if isinstance(value, float):
        return "a number"
    if isinstance(value, list):
        return "an array"
    if value is None:
        return "null"
    return "an object"


def _suggest(key: str, vocabulary: Sequence[str]) -> str:
    """One rejected argument name, with the nearest real one when there is a plausible one.

    Mirrors what `Engram.__init__` does for keyword arguments: a bare "unknown argument"
    costs the caller a retry to discover a spelling this process already knows.
    """
    close = difflib.get_close_matches(key, vocabulary, n=2, cutoff=0.6)
    if not close:
        return repr(key)
    return f"{key!r} (did you mean {' or '.join(repr(c) for c in close)}?)"


def _checked(label: str, value: Any, spec: Mapping[str, Any]) -> Any:
    kind = spec["type"]
    if kind in ("integer", "number"):
        # `bool` is a subclass of `int` in Python, so an unguarded isinstance check would
        # accept `true` as a count and then format it as `1` somewhere downstream.
        ok = isinstance(value, int) if kind == "integer" else isinstance(value, (int, float))
        if isinstance(value, bool) or not ok:
            raise ToolError(
                f"{label} must be {_ARTICLES[kind]}, got {_describe(value)} ({value!r})")
        low, high = spec.get("minimum"), spec.get("maximum")
        if low is not None and value < low:
            raise ToolError(f"{label} must be >= {low}, got {value!r}")
        if high is not None and value > high:
            raise ToolError(f"{label} must be <= {high}, got {value!r}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ToolError(
                f"{label} must be {_ARTICLES[kind]}, got {_describe(value)} ({value!r})")
        return [_checked(f"{label}[{i}]", item, spec["items"])
                for i, item in enumerate(value)]
    elif not isinstance(value, str):
        raise ToolError(
            f"{label} must be {_ARTICLES[kind]}, got {_describe(value)} ({value!r})")

    allowed = spec.get("enum")
    if allowed is not None and value not in allowed:
        raise ToolError(
            f"{label} must be one of {', '.join(repr(a) for a in allowed)}, got {value!r}")
    return value


def validate(properties: Mapping[str, Mapping[str, Any]], required: Sequence[str],
             arguments: Any, *, tool: str) -> dict[str, Any]:
    """Check `arguments` against a tool's schema and apply its declared defaults.

    Defaults are applied here rather than in the handlers so that the schema the model
    reads and the values the handler receives cannot drift apart — a default documented
    in one place and implemented in another is a default that is eventually wrong in the
    documentation.
    """
    if not isinstance(arguments, Mapping):
        raise ToolError(
            f"{tool}: arguments must be a JSON object of named parameters, got "
            f"{_describe(arguments)}")

    unknown = [k for k in arguments if k not in properties]
    if unknown:
        known = ", ".join(sorted(properties)) or "(none)"
        raise ToolError(
            f"{tool}: unknown argument(s) "
            f"{', '.join(_suggest(k, sorted(properties)) for k in unknown)}. "
            f"Accepted: {known}.")

    missing = [k for k in required if k not in arguments]
    if missing:
        raise ToolError(
            f"{tool}: missing required argument(s) "
            f"{', '.join(repr(k) for k in missing)}.")

    out: dict[str, Any] = {}
    for name, spec in properties.items():
        if name in arguments:
            out[name] = _checked(f"{tool}.{name}", arguments[name], spec)
        elif "default" in spec:
            out[name] = spec["default"]
    return out
