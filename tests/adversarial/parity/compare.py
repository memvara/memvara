"""Compare what two surfaces returned for the same operation, and say what differs.

The parity tests run one list of operations through several surfaces, each with a store
of its own. Two stores that were told the same things still differ in three ways that say
nothing about the surfaces:

* every id is random, so the same claim has a different id in each store;
* every instant a store takes from its clock is the moment that store ran;
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
from typing import Any, Callable, Iterable, Iterator, Mapping, TypeVar

from harness.clock import within
from memvara.confirm import CONFIRM_TTL
from memvara.core import PROFILE_WINDOW
from memvara.types import (CLOSURE, ENTITY_REKEY, LAST_OBSERVED, OBJECT_ENTITY,
                           SALIENCE_BASE, SUBJECT_ENTITY, Claim, Document, Episode,
                           ForgetPreview, ForgetResult, WriteReceipt, utcnow)

T = TypeVar("T")

#: An id the store mints: a claim, a stored turn or a document, then 20 hex digits.
IDS = re.compile(r"\b(?:cl|ep|doc)_[0-9a-f]{20}\b")

#: `Claim.meta` keys where the store keeps its own bookkeeping. The hosted API carries two
#: of them as the top-level fields `salience_base` and `last_observed` and leaves the other
#: three out (`memvara/remote/hydrate.py`), so a claim is compared through its
#: `salience_base` and `last_observed` properties instead of through these keys.
BOOKKEEPING = frozenset({SALIENCE_BASE, LAST_OBSERVED, SUBJECT_ENTITY, OBJECT_ENTITY,
                         ENTITY_REKEY})

#: What an instant a store took from its clock during the run is replaced with.
WALL_CLOCK = "<wall clock>"
#: What the instant a confirm token stops being accepted is replaced with: the preview's
#: moment plus `memvara.confirm.CONFIRM_TTL`.
TOKEN_EXPIRES = "<wall clock + the confirm token's lifetime>"
#: What the start of `memory_profile`'s default window is replaced with: the call's moment
#: minus `memvara.core.PROFILE_WINDOW`.
PROFILE_STARTS = "<wall clock - the profile window>"

#: The instants a store derives from its clock: the offset from the moment it read the
#: clock, the marker each one is replaced with, and how far the store may round it down.
#: `memvara.confirm` keeps a token's expiry in whole seconds. An instant at any other
#: distance from the run is compared as it is, so a surface that is off fails.
CLOCK_MARKERS: tuple[tuple[timedelta, str, timedelta], ...] = (
    (timedelta(0), WALL_CLOCK, timedelta(0)),
    (CONFIRM_TTL, TOKEN_EXPIRES, timedelta(seconds=1)),
    (-PROFILE_WINDOW, PROFILE_STARTS, timedelta(0)),
)

#: How far outside the run an instant a surface returns as a `datetime` may fall: the few
#: milliseconds by which two readings of the clock, or a round trip through epoch seconds,
#: can differ. A store reads its clock during the run, so anything further off is a
#: surface stamping the wrong time.
RESOLUTION = timedelta(milliseconds=5)

#: How far outside the run an instant written in a tool's reply may fall. The tools write
#: an instant to the minute, rounding down, so a minute is the least that covers it.
TEXT_SLACK = timedelta(minutes=1)

#: Two floats closer than this count as equal. A search score depends on how long before
#: the search each fact began or was last restated (`Claim.trace_from`), so the same
#: search on two stores a few seconds apart differs in the ninth decimal place.
TOLERANCE = 1e-6


@dataclasses.dataclass(frozen=True)
class Run:
    """When a program ran, from a moment before its first call to a moment after its
    last. Every instant a store takes from its clock during the program is inside it.

    A run that ends before it starts is refused, and so is one long enough that two
    markers' windows in `CLOCK_MARKERS` overlap, because an instant in both could then
    be replaced with either and a difference between two surfaces would go unseen. Text
    has the widest windows, so they are the ones checked; a run shorter than about eight
    minutes is accepted.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"a run that ends before it starts: {self.start} to {self.end}")
        windows = sorted(_window(self, offset, rounding, TEXT_SLACK) + (marker,)
                         for offset, marker, rounding in CLOCK_MARKERS)
        for (_low, high, one), (low, _high, other) in zip(windows, windows[1:]):
            if high >= low:
                raise ValueError(
                    f"a run that took {self.end - self.start} is too long: an instant could "
                    f"be both {one} and {other}. Compare a shorter run.")


def timed(work: Callable[[], T]) -> tuple[T, Run]:
    """Call `work`, and return what it returned and the run it took: from a moment before
    its first call to a moment after its last."""
    start = utcnow()
    result = work()
    return result, Run(start, utcnow())


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
    # One walk. The strings are kept until it ends, because an object met later in the
    # walk can name an id that a string cited earlier.
    texts: list[str] = []
    for value in _every(results):
        if isinstance(value, Claim):
            _name(found, value.id, f"<claim {value.subject} {value.predicate} {value.object}>")
        elif isinstance(value, Episode):
            _name(found, value.id, f"<turn {value.content}>")
        elif isinstance(value, Document):
            _name(found, value.id, f"<document {value.custom_id or value.title}>")
        elif isinstance(value, str):
            texts.append(value)
    _number_the_rest(found, texts)
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


def _window(run: Run, offset: timedelta, rounding: timedelta,
            slack: timedelta) -> tuple[datetime, datetime]:
    """Where an instant derived from a reading of the clock during `run` can fall."""
    return run.start + offset - rounding - slack, run.end + offset + slack


def _marker(value: datetime, run: Run, slack: timedelta) -> str | None:
    """The marker for an instant a store derived from its clock during `run`, allowing
    `slack` either side, or None."""
    for offset, marker, rounding in CLOCK_MARKERS:
        if within(value, *_window(run, offset, rounding, slack)):
            return marker
    return None


def normalise(value: Any, names: Mapping[str, str], *, run: Run) -> Any:
    """`value` as plain dicts, lists, strings and numbers, ready for `differences`.

    `names` labels ids (see `labels`), and an id it does not know becomes
    `<unlabelled id>`. An instant becomes `{"__type__": "datetime", "value": ...}`, whose
    value is one of the markers in `CLOCK_MARKERS` when a store took it from its clock
    during `run`, within `RESOLUTION`, and the instant in ISO 8601 otherwise. An enum
    becomes its class name and its value, and a dataclass keeps every field under its
    name, plus `__type__` for its class. So a value of one type never equals a value of
    another.

    Five things are changed further, and each for a reason that holds on every surface:

    * a claim leaves out its bookkeeping keys (see `BOOKKEEPING`) and gains its `state`,
      `salience_base` and `last_observed`;
    * a write receipt leaves out `latency_ms`, which is how long the write took;
    * a preview's confirm token becomes `<token>`, because it is a signature over ids;
    * a preview's matches become a list of pairs, so their ranked order is compared;
    * a `ForgetResult` sorts what it closed, because it closes in the token's id order.

    >>> from datetime import datetime, timezone
    >>> now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    >>> normalise({"at": now, "then": datetime(2024, 1, 1, tzinfo=timezone.utc)}, {},
    ...           run=Run(now, now))
    {'at': {'__type__': 'datetime', 'value': '<wall clock>'}, \
'then': {'__type__': 'datetime', 'value': '2024-01-01T00:00:00+00:00'}}
    """
    def again(item: Any) -> Any:
        return normalise(item, names, run=run)

    # Before the check for a string, because `MemoryType` and other enums here are
    # strings too, and would otherwise compare equal to their plain value.
    if isinstance(value, enum.Enum):
        return {"__type__": type(value).__name__, "value": value.value}
    if isinstance(value, str):
        return IDS.sub(lambda found: names.get(found.group(0), "<unlabelled id>"), value)
    if isinstance(value, datetime):
        return {"__type__": "datetime",
                "value": _marker(value, run, RESOLUTION) or value.isoformat()}
    if isinstance(value, Claim):
        return _claim(value, names, run)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out: dict[str, Any] = {"__type__": type(value).__name__}
        for field in dataclasses.fields(value):
            item = _field(value, field.name, getattr(value, field.name), again)
            if item is not _LEFT_OUT:
                out[field.name] = item
        return out
    if isinstance(value, Mapping):
        return {again(key): again(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [again(item) for item in value]
    return value


#: What `_field` returns for a field that is left out of the comparison.
_LEFT_OUT = object()


def _field(owner: Any, name: str, item: Any, again: Callable[[Any], Any]) -> Any:
    """How one field of a dataclass is normalised: through `again`, except for the four
    fields `normalise` names, each for its own reason."""
    if isinstance(owner, WriteReceipt) and name == "latency_ms":
        return _LEFT_OUT
    if isinstance(owner, ForgetPreview) and name == "confirm":
        return "<token>"
    if isinstance(owner, ForgetPreview) and name == "matches":
        return again(list(item.items()))
    if isinstance(owner, ForgetResult) and name == "closed":
        return sorted((again(claim) for claim in item), key=repr)
    return again(item)


def _claim(claim: Claim, names: Mapping[str, str], run: Run) -> dict[str, Any]:
    out: dict[str, Any] = {"__type__": "Claim"}
    for field in dataclasses.fields(claim):
        if field.name != "meta":
            out[field.name] = normalise(getattr(claim, field.name), names, run=run)
    meta = {key: item for key, item in claim.meta.items() if key not in BOOKKEEPING}
    if CLOSURE in meta:
        # Each closure records its instant as epoch seconds.
        meta[CLOSURE] = [{**entry, "at": datetime.fromtimestamp(entry["at"], timezone.utc)}
                         for entry in meta[CLOSURE]]
    out["meta"] = normalise(meta, names, run=run)
    out["state"] = claim.state
    out["salience_base"] = claim.salience_base
    out["last_observed"] = normalise(claim.last_observed, names, run=run)
    return out


def differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Every place where `actual` differs from `expected`, one line for each.

    A line names the path to the value, such as `.added[0].object_kind`, and both values.
    Two floats within `TOLERANCE` of each other are equal, and so are two NaNs.

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
    if isinstance(expected, float) and isinstance(actual, float) and (
            math.isnan(expected) and math.isnan(actual)
            or math.isclose(expected, actual, rel_tol=0.0, abs_tol=TOLERANCE)):
        # Two NaNs are the same answer, although NaN never equals itself. NaN against a
        # number fails `isclose` and is reported below.
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
ADDED = re.compile(r"^\+ \[(cl_[0-9a-f]{20})\] (.*)$", re.MULTILINE)

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
        for found in ADDED.finditer(text):
            _name(names, found.group(1), f"<{found.group(2)}>")
    _number_the_rest(names, texts)
    return names


def normalise_text(text: str, names: Mapping[str, str], *, run: Run) -> str:
    """A tool's reply with ids labelled, the confirm token replaced, and every instant a
    store took from its clock during `run` replaced with its marker from `CLOCK_MARKERS`.
    The tools write an instant to the minute, rounding down, so here an instant may fall
    `TEXT_SLACK` outside the run.

    >>> from datetime import datetime, timezone
    >>> now = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
    >>> normalise_text("recorded 2026-09-26 12:00Z true from 2024-01-01 00:00Z", {},
    ...                run=Run(now, now))
    'recorded <wall clock> true from 2024-01-01 00:00Z'
    """
    def stamp(found: re.Match[str]) -> str:
        raw = found.group(0)
        if " " in raw:
            when = datetime.strptime(raw, "%Y-%m-%d %H:%MZ").replace(tzinfo=timezone.utc)
        else:
            when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return _marker(when, run, TEXT_SLACK) or raw

    text = _TOKEN.sub(r"\1<token>", text)
    text = IDS.sub(lambda found: names.get(found.group(0), "<unlabelled id>"), text)
    return _STAMPS.sub(stamp, text)
