r"""Argument checking for tool calls, because there is no SDK doing it for us.

A model fills these arguments in from a JSON Schema it read once, and it gets them wrong
in a small set of recoverable ways: an argument name borrowed from a sibling tool, a
number sent as a string, an enum value it invented. Every rejection here is phrased as
something the caller can act on, and every one comes back as a tool *result* rather than
a JSON-RPC error — the model sees results, whereas protocol errors are addressed to the
client, which typically renders them as a failed call and moves on.

The validated subset of JSON Schema is exactly what the tools in this package declare:
`type` (string/integer/number/boolean/array), `enum`, `minimum`, `maximum`, `maxLength`,
`default`, `required`, and `additionalProperties: false`. Anything wider would be untested
code in a validator, which is the one place that is not acceptable.

That sentence is load-bearing, and `boolean` was missing from it for as long as it was
missing from the code. `memory_recall` grew an `include_episodes` argument, declared it
`boolean` in `tools.py`, and nothing here knew the word — so the only boolean in the whole
tool surface behaved like this: a caller sending `true` or `false` got an unhandled
`KeyError` raised out of the error path itself, while a caller sending the *string*
`"false"` was accepted and read through `bool(...)` as True. The correct type crashed and
the wrong one silently inverted. Adding a type to a tool means adding it here in the same
commit; the list above is the checklist.

>>> validate({"k": {"type": "integer", "default": 8}}, (), {}, tool="demo")
{'k': 8}
>>> validate({"k": {"type": "integer"}}, (), {"k": "8"}, tool="demo")
Traceback (most recent call last):
    ...
memvara.server.validate.ToolError: demo.k must be an integer, got a string ('8')
>>> validate({"raw": {"type": "boolean"}}, (), {"raw": False}, tool="demo")
{'raw': False}
>>> validate({"raw": {"type": "boolean"}}, (), {"raw": "false"}, tool="demo")
Traceback (most recent call last):
    ...
memvara.server.validate.ToolError: demo.raw must be a boolean, got a string ('false')
"""

from __future__ import annotations

import difflib
from typing import Any, Collection, Mapping, Sequence

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
    "boolean": "a boolean",
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

    Mirrors what `Memvara.__init__` does for keyword arguments: a bare "unknown argument"
    costs the caller a retry to discover a spelling this process already knows.
    """
    close = difflib.get_close_matches(key, vocabulary, n=2, cutoff=0.6)
    if not close:
        return repr(key)
    return f"{key!r} (did you mean {' or '.join(repr(c) for c in close)}?)"


def _checked(label: str, value: Any, spec: Mapping[str, Any],
             siblings: Collection[str] = ()) -> Any:
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
    elif kind == "boolean":
        # Only a real `bool`. Not `1`, which `isinstance(value, int)` would wave through
        # the way the integer branch above has to guard against in the other direction --
        # and not the string `"false"`, which is the case that actually reached
        # production: without this branch a boolean argument fell through to the string
        # check below, so `"false"` validated, and every handler reads its flags through
        # `bool(...)`, where a non-empty string is True. The strict check is the point.
        if not isinstance(value, bool):
            raise ToolError(
                f"{label} must be {_ARTICLES[kind]}, got {_describe(value)} ({value!r})")
        # Returned here rather than falling through, as `array` does: `enum` and
        # `maxLength` have no meaning on two values, and `len()` on a bool raises.
        return value
    elif kind == "array":
        if not isinstance(value, list):
            raise ToolError(
                f"{label} must be {_ARTICLES[kind]}, got {_describe(value)} ({value!r})")
        return [_checked(f"{label}[{i}]", item, spec["items"])
                for i, item in enumerate(value)]
    elif not isinstance(value, str):
        raise ToolError(
            f"{label} must be {_ARTICLES[kind]}, got {_describe(value)} ({value!r})")
    else:
        # A lone surrogate is a `str` that cannot be encoded. Python's JSON parser
        # accepts `"\ud800"` and hands back exactly that, so it arrives here looking like
        # any other string and fails later, at whatever line first tries to write it —
        # for `memory_remember` that was the store, and what reached the model was
        # "failed: UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in
        # position 2", from the generic handler. That names a Python exception rather
        # than an argument, so a model cannot tell which argument to fix or how.
        #
        # Checked for every string argument rather than for the ones that happened to
        # crash: this is a property of the value, and any tool can be handed one.
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ToolError(
                f"{label} contains an unpaired surrogate at position {exc.start} "
                f"({value[exc.start]!r}) and cannot be encoded as UTF-8. It is half of "
                "a character; the text it came from was probably truncated or decoded "
                "twice. Send the whole character or drop it.") from None

    allowed = spec.get("enum")
    if allowed is not None and value not in allowed:
        raise ToolError(
            f"{label} must be one of {', '.join(repr(a) for a in allowed)}, got {value!r}")

    # The string counterpart of `maximum`, and it exists for the same reason a numeric
    # bound does: an argument nothing checks is one every later turn pays for. A subject
    # or predicate is a slot *name* — it is echoed by the write, rendered again by every
    # search and recall that hits it, and `recall` drops notes whole rather than trimming
    # them, so one oversized name evicts several real notes from a budgeted block. Unlike
    # a long object, there is no legitimate reading in which it is the value someone meant
    # to store.
    longest = spec.get("maxLength")
    if longest is not None and len(value) > longest:
        # Where to put the text instead, and it depends on the tool. Only
        # `memory_remember` has an `object` to move the detail into; the other five that
        # carry a `maxLength` — forget, end, history, neighborhood, paths — take names
        # that must *match* something already stored, and telling their caller to use an
        # argument the tool does not accept turns one rejected call into two.
        raise ToolError(
            f"{label} must be at most {longest} characters, got {len(value)}. "
            + ("This is a name for a fact, not the fact itself — put the detail in "
               "'object'." if "object" in siblings else
               "This is a name, not a description of one: it has to match how the thing "
               "was stored, and nothing this long was stored as a name."))
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
            out[name] = _checked(f"{tool}.{name}", arguments[name], spec, properties)
        elif "default" in spec:
            out[name] = spec["default"]
    return out
