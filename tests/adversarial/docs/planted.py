"""Planted code for the documentation checks' own tests.

Each check is shown catching a fault on something planted before it runs on the real
code. What is planted here has to live in a module of its own: an option parser that a
command imports from elsewhere, which the options reader must follow across modules.
"""

from __future__ import annotations

#: The options `parse` accepts. A command that hands its arguments to `parse` accepts them
#: too, although their spelling appears nowhere in that command's own module.
_OPTIONS = ("--city", "--quiet")


def parse(argv: list[str]) -> dict[str, str]:
    """`--name value` pairs, refusing a name that is not in `_OPTIONS`."""
    found: dict[str, str] = {}
    rest = list(argv)
    while rest:
        name = rest.pop(0)
        if name not in _OPTIONS:
            raise ValueError(name)
        found[name] = rest.pop(0) if rest else ""
    return found
