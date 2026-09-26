"""Compare what two surfaces returned for the same operation, and say what differs.

The parity tests run one list of operations through several surfaces, each with a store
of its own. Two stores that were told the same things still differ in three ways that say
nothing about the surfaces:

* every id is random, so the same claim has a different id in each store;
* every instant a store takes from the clock is the moment that store ran;
* a few lists come back in id order, so their order is random too.

`normalise` removes those three and turns a result into plain dicts and lists, and
`differences` lists every place where what is left disagrees, with both values.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator, Mapping

from memvara.types import (CLOSURE, ENTITY_REKEY, LAST_OBSERVED, OBJECT_ENTITY,
                           SALIENCE_BASE, SUBJECT_ENTITY, Claim, Document, Episode,
                           ForgetPreview, ForgetResult, WriteReceipt)

#: An id the store mints: a claim, a stored turn or a document, then 20 hex digits.
IDS = re.compile(r"\b(?:cl|ep|doc)_[0-9a-f]{20}\b")

#: `Claim.meta` keys where the store keeps its own bookkeeping. The hosted API carries two
#: of them as the top-level fields `salience_base` and `last_observed` and leaves the other
#: three out (`memvara/remote/hydrate.py`), so a claim is compared through its
#: `salience_base` and `last_observed` properties instead of through these keys.
BOOKKEEPING = frozenset({SALIENCE_BASE, LAST_OBSERVED, SUBJECT_ENTITY, OBJECT_ENTITY,
                         ENTITY_REKEY})

#: How close to the run an instant must be to count as one a store took from its clock.
#: It is wide enough for the default seven-day window of `memory_profile`, whose header
#: prints the start of that window. Every instant a parity test passes on purpose is
#: much further away than this, so it is compared as it is.
CLOCK_WINDOW = timedelta(days=8)

#: What an instant taken from the clock is replaced with.
WALL_CLOCK = "<wall clock>"

#: Two floats closer than this count as equal. A search score depends on how long ago a
#: claim was written, so the same search on two stores a few seconds apart differs in the
#: ninth decimal place.
TOLERANCE = 1e-6


@dataclasses.dataclass(frozen=True)
class Raised:
    """A step that raised, recorded as the exception's class name and its message."""

    kind: str
    message: str


def _every(value: Any) -> Iterator[Any]:
    """`value` and everything inside it, in a fixed order: a dataclass's fields in the
    order they are declared, a mapping's keys and values, and a sequence's items."""
    yield value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _every(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _every(key)
            yield from _every(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _every(item)


def labels(*results: Any) -> dict[str, str]:
    """A label for every id in `results`, made from what the id names.

    A claim is labelled by its subject, predicate and object, a stored turn by its text,
    and a document by its custom id or title. Two stores that were told the same things
    therefore give the same labels, whatever ids they drew. An id that none of those
    names, such as one only a string cites, is numbered in the order it first appears:
    `<cl 1>`, `<ep 1>`. That order is the same on every surface, because every surface
    runs the same program.

    Two different ids that would get the same label raise `ValueError`, because the
    comparison could no longer tell them apart. Give the two things different text.

    >>> from memvara.types import Claim
    >>> first = Claim(subject="user", predicate="lives_in", object="Berlin")
    >>> labels([first, "cl_00000000000000000000"]) == {
    ...     first.id: "<claim user lives_in Berlin>", "cl_00000000000000000000": "<cl 1>"}
    True
    """
    found: dict[str, str] = {}
    for value in _every(results):
        if isinstance(value, Claim):
            _name(found, value.id, f"<claim {value.subject} {value.predicate} {value.object}>")
        elif isinstance(value, Episode):
            _name(found, value.id, f"<turn {value.content}>")
        elif isinstance(value, Document):
            _name(found, value.id, f"<document {value.custom_id or value.title}>")
    _number_the_rest(found, (value for value in _every(results) if isinstance(value, str)))
    return found


def _name(found: dict[str, str], key: str, label: str) -> None:
    """Label `key`, refusing a label another id already has."""
    if any(other != key and given == label for other, given in found.items()):
        raise ValueError(f"two different ids are both {label}; give them different text "
                         "so the comparison can tell them apart")
    found[key] = label


def _number_the_rest(found: dict[str, str], texts: Iterable[str]) -> None:
    """Number every id in `texts` that `found` does not label yet, by its kind and in the
    order it first appears."""
    counts: dict[str, int] = {}
    for text in texts:
        for match in IDS.finditer(text):
            if match.group(0) not in found:
                kind = match.group(0).split("_", 1)[0]
                counts[kind] = counts.get(kind, 0) + 1
                found[match.group(0)] = f"<{kind} {counts[kind]}>"


def _instant(value: datetime, near: datetime) -> str:
    """`WALL_CLOCK` for an instant near `near`, and the instant itself otherwise."""
    return WALL_CLOCK if abs(value - near) <= CLOCK_WINDOW else value.isoformat()


def normalise(value: Any, names: Mapping[str, str], *, near: datetime) -> Any:
    """`value` as plain dicts, lists, strings and numbers, ready for `differences`.

    `names` labels ids (see `labels`), and an id it does not know becomes
    `<unlabelled id>`. `near` is when the run happened: an instant within
    `CLOCK_WINDOW` of it becomes `WALL_CLOCK`, and any other instant stays as it is.
    A dataclass keeps its fields under their names, plus `__type__` for its class.

    Four things are changed further, and each for a reason that holds on every surface:

    * a claim leaves out its bookkeeping keys (see `BOOKKEEPING`) and gains its `state`,
      `salience_base` and `last_observed`;
    * a write receipt leaves out `latency_ms`, which is how long the write took;
    * a preview's confirm token becomes `<token>`, because it is a signature over ids;
    * a `ForgetResult` sorts what it closed, because it closes in the token's id order.

    >>> from datetime import datetime, timezone
    >>> now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    >>> normalise({"at": now, "then": datetime(2024, 1, 1, tzinfo=timezone.utc)}, {},
    ...           near=now)
    {'at': '<wall clock>', 'then': '2024-01-01T00:00:00+00:00'}
    """
    def again(item: Any) -> Any:
        return normalise(item, names, near=near)

    if isinstance(value, str):
        return IDS.sub(lambda found: names.get(found.group(0), "<unlabelled id>"), value)
    if isinstance(value, datetime):
        return _instant(value, near)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Claim):
        return _claim(value, names, near)
    if isinstance(value, ForgetPreview):
        return {"__type__": "ForgetPreview", "close": value.close,
                "matches": again(list(value.matches.items())), "confirm": "<token>",
                "expires_at": again(value.expires_at)}
    if isinstance(value, ForgetResult):
        closed = [again(claim) for claim in value.closed]
        return {"__type__": "ForgetResult", "close": value.close, "reason": value.reason,
                "closed": sorted(closed, key=repr)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out: dict[str, Any] = {"__type__": type(value).__name__}
        for field in dataclasses.fields(value):
            if isinstance(value, WriteReceipt) and field.name == "latency_ms":
                continue
            out[field.name] = again(getattr(value, field.name))
        return out
    if isinstance(value, Mapping):
        return {again(key): again(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [again(item) for item in value]
    return value


def _claim(claim: Claim, names: Mapping[str, str], near: datetime) -> dict[str, Any]:
    out: dict[str, Any] = {"__type__": "Claim"}
    for field in dataclasses.fields(claim):
        if field.name != "meta":
            out[field.name] = normalise(getattr(claim, field.name), names, near=near)
    meta = {key: item for key, item in claim.meta.items() if key not in BOOKKEEPING}
    if CLOSURE in meta:
        # Each closure records its instant as epoch seconds.
        meta[CLOSURE] = [
            {**entry, "at": _instant(datetime.fromtimestamp(entry["at"], timezone.utc), near)}
            for entry in meta[CLOSURE]]
    out["meta"] = normalise(meta, names, near=near)
    out["state"] = claim.state
    out["salience_base"] = claim.salience_base
    out["last_observed"] = normalise(claim.last_observed, names, near=near)
    return out


def differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Every place where `actual` differs from `expected`, one line for each.

    A line names the path to the value, such as `.added[0].object_kind`, and both values.
    Two floats within `TOLERANCE` of each other are equal.

    >>> differences({"a": [1, 2.0], "b": "x"}, {"a": [1, 2.0000001], "c": "x"})
    [".b: only in the expected answer, 'x'", ".c: only in the actual answer, 'x'"]
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        out: list[str] = []
        for key in sorted(set(expected) | set(actual), key=str):
            if key not in actual:
                out.append(f"{path}.{key}: only in the expected answer, "
                           f"{_short(expected[key])}")
            elif key not in expected:
                out.append(f"{path}.{key}: only in the actual answer, "
                           f"{_short(actual[key])}")
            else:
                out += differences(expected[key], actual[key], f"{path}.{key}")
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        out = []
        if len(expected) != len(actual):
            out.append(f"{path}: {len(expected)} item(s) expected, {len(actual)} found")
        for index, (one, other) in enumerate(zip(expected, actual)):
            out += differences(one, other, f"{path}[{index}]")
        return out
    if (isinstance(expected, float) and isinstance(actual, float)
            and math.isclose(expected, actual, rel_tol=0.0, abs_tol=TOLERANCE)):
        return []
    if expected != actual or type(expected) is not type(actual):
        return [f"{path}: expected {_short(expected)}, found {_short(actual)}"]
    return []


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 200 else text[:199] + "…"


def assert_same(expected: Any, actual: Any, what: str) -> None:
    """Fail with every difference listed, or return when there is none."""
    found = differences(expected, actual)
    assert not found, (f"{what}: {len(found)} difference(s)\n  " + "\n  ".join(found))


# -- the text an MCP tool returns ------------------------------------------------------

#: A write receipt's line for a claim it added: `+ [<id>] <the claim's text>`.
_ADDED = re.compile(r"^\+ \[(cl_[0-9a-f]{20})\] (.*)$", re.MULTILINE)

#: An instant as the tools write one: to the minute, as `_stamp` does, or in ISO 8601
#: with whole seconds or the six decimal places `datetime.isoformat` writes.
_STAMPS = re.compile(r"\b\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}Z|T\d{2}:\d{2}:\d{2}(?:\.\d{6})?"
                     r"(?:Z|[+-]\d{2}:\d{2}))")

#: The token a preview of `memory_end_matching` or `memory_forget_matching` ends with.
_TOKEN = re.compile(r"^(confirm: )\S+$", re.MULTILINE)


def text_labels(texts: list[str]) -> dict[str, str]:
    """A label for every id in a session's tool replies, in the order they were written.

    A claim is labelled by its text, from the receipt line of the write that added it.
    Any other id, such as a stored turn or a document, is numbered in the order it first
    appears, which is the same on every surface because the session is the same. Two
    claims with the same text raise `ValueError`, as they do in `labels`.

    >>> text_labels(["+ [cl_0123456789abcdef0123] user likes jazz",
    ...              "turn id(s): ep_0123456789abcdef0123"])
    {'cl_0123456789abcdef0123': '<user likes jazz>', 'ep_0123456789abcdef0123': '<ep 1>'}
    """
    names: dict[str, str] = {}
    for text in texts:
        for found in _ADDED.finditer(text):
            _name(names, found.group(1), f"<{found.group(2)}>")
    _number_the_rest(names, texts)
    return names


def normalise_text(text: str, names: Mapping[str, str], *, near: datetime) -> str:
    """A tool's reply with ids labelled, the confirm token replaced, and every instant
    within `CLOCK_WINDOW` of `near` replaced with `WALL_CLOCK`.

    >>> from datetime import datetime, timezone
    >>> near = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
    >>> normalise_text("recorded 2026-09-26 11:59Z true from 2024-01-01 00:00Z", {},
    ...                near=near)
    'recorded <wall clock> true from 2024-01-01 00:00Z'
    """
    def stamp(found: re.Match[str]) -> str:
        raw = found.group(0)
        if " " in raw:
            when = datetime.strptime(raw, "%Y-%m-%d %H:%MZ").replace(tzinfo=timezone.utc)
        else:
            when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return WALL_CLOCK if abs(when - near) <= CLOCK_WINDOW else raw

    text = _TOKEN.sub(r"\1<token>", text)
    text = IDS.sub(lambda found: names.get(found.group(0), "<unlabelled id>"), text)
    return _STAMPS.sub(stamp, text)
