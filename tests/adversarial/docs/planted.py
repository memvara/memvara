"""Planted code and data for the documentation checks' own tests.

Each check is shown catching a fault on something planted before it runs on the real
code. What is planted here is shared by several test files, or has to live in a module of
its own: a tool table small enough to read every expected answer off by hand, and an
option parser that a command imports from elsewhere, which the options reader must follow
across modules.
"""

from __future__ import annotations

#: A small tool table for the planted cases of the mention and skill checks. Each expected
#: answer in those tests can be read off it by hand, rather than computed by the code
#: under test. It leaves arguments out on purpose: `memory_recall` takes no `as_of` here,
#: and `memory_search` no `include_episodes`.
TABLE: dict[str, frozenset[str]] = {
    "memory_recall": frozenset({"query", "include_episodes", "ranked", "synthesize",
                                "query_rewrite", "valid_at", "budget", "filters",
                                "filepath_prefix"}),
    "memory_search": frozenset({"query", "k", "as_of", "valid_at", "memory_types",
                                "filters", "filepath_prefix"}),
    "memory_end": frozenset({"claim_id", "at", "reason"}),
    "memory_remember": frozenset({"subject", "predicate", "object", "true_since",
                                  "true_until", "sources", "replaces", "memory_type"}),
    "memory_add": frozenset({"text", "role"}),
    "memory_standing": frozenset({"k"}),
}

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
