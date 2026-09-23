"""Metadata filters and the file-path filter that `search()` and `recall()` take.

A caller narrows a read with two arguments. `filters` is a mapping from a top-level key of
a memory's `meta` to the value that key must hold, and a list of values means "any of
these". `filepath_prefix` keeps only memories that came from a document whose `filepath`
starts with the given text. This module checks both arguments and turns them into one
`SearchFilter`, which is what the retriever hands to every store method that caps rows.

**The filter runs inside the store, where the limit runs.** That is design invariant 7 in
`docs/INTERNALS.md`: a store that returned its top `limit` rows and let the caller drop
the ones that do not match would return fewer than `k` matches while more exist further
down, and nothing in the result would say so. So `SearchFilter` is a store parameter, and
`SQLiteStore` evaluates it in the same SQL statement as its `LIMIT`.

**What matches.** Every key must match; the keys combine with AND. A string matches only
the same string, with case. A number matches any equal number, so `1` matches a stored
`1.0`. `True` and `False` match only a stored boolean, never `1` or `0`. A stored list or
object never matches, because the test is equality on one value.

**Each key is tested on its own**, against the row's own `meta` and against the `meta` of
every document the row came from, and the key matches when any one of those holds a
wanted value. So a claim whose own `meta` says `team: infra`, extracted from a document
whose `meta` says `year: 2025`, matches `{"team": "infra", "year": 2025}`: each key is
held by one of its sources. A document chunk is that document's own text, and a claim
came from a document when one of its sources is a chunk of it. `filepath_prefix` has only
the document route, because only a document has a `filepath`. It compares characters
exactly: `%` and `_` are ordinary characters and case matters.

**Keys are restricted to `[A-Za-z0-9_.-]{1,64}`.** Anything else is refused with
`ValueError` before a query runs, so a key can be placed inside a JSON path in any backend
without quoting rules of its own. Values are never placed in SQL text; every backend binds
them as parameters.

>>> f = search_filter({"team": "infra", "year": [2025, 2026]}, None)
>>> f.meta
(('team', ('infra',)), ('year', (2025, 2026)))
>>> f.matches({"team": "infra", "year": 2026, "other": "x"})
True
>>> f.matches({"team": "Infra", "year": 2026})
False
>>> f.matches({"team": "infra"}, {"year": 2025})
True
>>> search_filter({}, None) is None
True
>>> search_filter({"a b": 1}, None)
Traceback (most recent call last):
    ...
memvara.filters.FilterError: filters key 'a b' is not allowed. A key is 1 to 64 characters, each a letter, a digit, '_', '.' or '-'.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Mapping, Union

__all__ = ["FILTER_KEY", "FilterError", "FilterValue", "SearchFilter", "checked_filter",
           "meta_matches", "search_filter"]

#: The one pattern a filter key must match, whole.
FILTER_KEY = re.compile(r"[A-Za-z0-9_.-]{1,64}")

#: One value a filter key may equal.
Scalar = Union[str, int, float, bool]

#: What a caller may pass for one key: a value, or a list meaning "any of these".
FilterValue = Union[str, int, float, bool, list]


class FilterError(ValueError):
    """A filter argument that cannot be used, or one passed while filters are switched off.

    A `ValueError`, so a caller that already catches that keeps working. Its own class so
    the MCP tools can turn exactly these into an argument error for the model, without
    also catching every other `ValueError` a read can raise.
    """


#: Why a filtered read was refused when the `metadata_filters` switch is off. It names
#: both ways the switch is set, because the same engine serves a script and a server.
SWITCHED_OFF = (
    "filters and filepath_prefix are switched off here (metadata_filters=False, or "
    "MEMVARA_FEATURE_METADATA_FILTERS=0 on a server), so this read was refused rather than "
    "run without them. Leave both out, or switch the feature on.")


def _scalar_ok(value: Any) -> bool:
    return isinstance(value, (str, int, float))  # `bool` is an `int` in Python


def _describe(value: Any) -> str:
    return type(value).__name__


def _values(key: str, value: Any) -> tuple[Scalar, ...]:
    """The values `key` may equal, refusing anything that is not a scalar or a list of
    them."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise FilterError(
                f"filters[{key!r}] is an empty list, which matches nothing. Pass at least "
                "one value, or leave the key out.")
        bad = [v for v in value if not _scalar_ok(v)]
        if bad:
            raise FilterError(
                f"filters[{key!r}] holds a {_describe(bad[0])} ({bad[0]!r}). A list may "
                "hold only strings, numbers and booleans.")
        return tuple(value)
    if not _scalar_ok(value):
        raise FilterError(
            f"filters[{key!r}] is a {_describe(value)} ({value!r}). A filter value is a "
            "string, a number, a boolean, or a list of those meaning any one of them.")
    return (value,)


@dataclass(frozen=True)
class SearchFilter:
    """A checked filter, ready to hand to a store.

    `meta` holds one `(key, allowed values)` pair per key, sorted by key, so two calls
    that ask the same thing produce equal objects. `filepath_prefix` is `None` when the
    caller gave none. Build one with `search_filter`, which does the checking; this class
    trusts what it is given.
    """

    meta: tuple[tuple[str, tuple[Scalar, ...]], ...] = ()
    filepath_prefix: str | None = None

    @cached_property
    def spec(self) -> str:
        """The metadata half as a JSON string, for a store to bind as one parameter.

        Computed once per filter: a filtered search binds it in every capped store call.
        """
        return _spec(self.meta)

    @cached_property
    def key_specs(self) -> tuple[tuple[str, str], ...]:
        """One `(key, spec)` pair per key, each spec naming that key alone.

        What a store binds when it tests each key on its own against every source of a
        row, which is the rule this module's docstring states.
        """
        return tuple((key, _spec(((key, values),))) for key, values in self.meta)

    def wire(self) -> dict[str, Any]:
        """The metadata half as the JSON object the hosted API receives."""
        return {key: list(values) if len(values) > 1 else values[0]
                for key, values in self.meta}

    def matches(self, meta: Mapping[str, Any], *more: Mapping[str, Any]) -> bool:
        """Whether every key is held by at least one of the given `meta` mappings.

        Pass a row's own `meta` first and then the `meta` of each document it came from.
        Each key is tested on its own, so different keys may be held by different
        sources. The file path is not tested here: it belongs to a document.
        """
        sources = (meta, *more)
        return all(any(key in source and any(_equal(source[key], want) for want in wanted)
                       for source in sources)
                   for key, wanted in self.meta)


def _spec(meta: tuple[tuple[str, tuple[Scalar, ...]], ...]) -> str:
    return json.dumps({key: list(values) for key, values in meta},
                      sort_keys=True, separators=(",", ":"))


def _equal(stored: Any, wanted: Scalar) -> bool:
    """Equality with JSON's types kept apart.

    Python says `True == 1` and `"1" != 1`. The first is wrong for a filter: a caller
    asking for `archived: true` does not mean a row whose `archived` is the count 1. So a
    boolean matches only a boolean, a string only a string, and a number any equal number.
    """
    if isinstance(wanted, bool) or isinstance(stored, bool):
        return isinstance(wanted, bool) and isinstance(stored, bool) and stored == wanted
    if isinstance(wanted, str) or isinstance(stored, str):
        return isinstance(wanted, str) and isinstance(stored, str) and stored == wanted
    return isinstance(stored, (int, float)) and stored == wanted


def search_filter(filters: Mapping[str, Any] | None,
                  filepath_prefix: str | None) -> SearchFilter | None:
    """Check the two filter arguments and combine them, or return `None` for no filter.

    Raises `FilterError`, naming the argument, for a key outside `FILTER_KEY`, a value that
    is not a string, number or boolean or a non-empty list of them, and a
    `filepath_prefix` that is not a string. An empty `filters` mapping narrows nothing
    and is the same as `None`.
    """
    if filters is not None and not isinstance(filters, Mapping):
        raise FilterError(
            f"filters must be a mapping of metadata keys to values, got "
            f"{_describe(filters)}")
    if filepath_prefix is not None and not isinstance(filepath_prefix, str):
        raise FilterError(
            f"filepath_prefix must be a string, got {_describe(filepath_prefix)}")
    meta = []
    for key, value in (filters or {}).items():
        if not isinstance(key, str) or not FILTER_KEY.fullmatch(key):
            raise FilterError(
                f"filters key {key!r} is not allowed. A key is 1 to 64 characters, each a "
                "letter, a digit, '_', '.' or '-'.")
        meta.append((key, _values(key, value)))
    if not meta and filepath_prefix is None:
        return None
    return SearchFilter(meta=tuple(sorted(meta)), filepath_prefix=filepath_prefix)


def checked_filter(filters: Mapping[str, Any] | None, filepath_prefix: str | None, *,
                   enabled: bool) -> SearchFilter | None:
    """`search_filter`, and the `metadata_filters` switch, in the one place both apply.

    Every engine calls this: `Memvara` for a local read, and the hosted clients before
    they send anything. With `enabled=False` a real filter raises `FilterError` naming the
    switch. An empty `filters` mapping narrows nothing, so it is not refused.
    """
    where = search_filter(filters, filepath_prefix)
    if where is not None and not enabled:
        raise FilterError(SWITCHED_OFF)
    return where


@lru_cache(maxsize=64)
def _parsed(spec: str) -> SearchFilter:
    """A spec string back as a `SearchFilter`. Cached, because a store calls
    `meta_matches` once per candidate row with the same spec every time."""
    return SearchFilter(meta=tuple((key, tuple(values))
                                   for key, values in json.loads(spec).items()))


def meta_matches(meta: str | None, spec: str) -> bool:
    """Whether a stored `meta` column, as JSON text, satisfies `SearchFilter.spec`.

    `SQLiteStore` registers this as a SQL function and uses it when the SQLite it runs on
    has no JSON functions, which are not compiled into every build this library supports;
    with them, the store tests each key with `json_type` and `json_extract` instead, which
    is faster. Unreadable or non-object `meta` matches nothing rather than raising.

    >>> meta_matches('{"team": "infra"}', SearchFilter((("team", ("infra",)),)).spec)
    True
    >>> meta_matches('{"archived": 1}', SearchFilter((("archived", (True,)),)).spec)
    False
    """
    try:
        stored = json.loads(meta) if meta else {}
    except ValueError:
        return False
    return isinstance(stored, dict) and _parsed(spec).matches(stored)
