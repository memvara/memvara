"""A guard against predicate pollution: a real value filed under a slot it does not belong to.

`reject_ungrounded` refuses invention — an object sharing no vocabulary with its source
turn. It says of itself that a claim reusing real vocabulary with a misattributed meaning
passes clean. This module is that case. Measured 2026-09-03 on `phi-4-mini-instruct`
(`tests/fixtures/phi4_spike/`): given a 64-predicate vocabulary the model forces a value
it has found into whatever slot is available — `gate / lives_in / "Port 61434"`,
`gate / endpoint / "Port 61434"`, `gate / build_status / "Port 61434"`, one correct value
wearing three wrong predicates. `Port 61434` is in the turn word for word, so grounding
passes; the value is real and the slot is wrong; structure validation cannot see it.

The destructive direction is the ONE-cardinality slot. `lives_in` is `Cardinality.ONE`, so
asserting it **ends** whatever was there: a polluted claim does not add noise, it retires a
true fact and keeps answering with the false one. That is why this is a refusal at the
write path rather than a ranking concern at read time.

Three rules. Two refuse, one discounts, and the two were scored against all 255 claims in
the fixture with the spike's own scorer before either was written down here — the numbers
are in `tests/test_pollution.py`, where they are a test rather than a memory. Together
they remove 26 of 46 wrong-predicate claims and all 32 duplicates, and change the count of
keyed facts found in **no** configuration: 60 of 90 before and after.

**R1 — one value, several predicates, in one turn.** Within one source turn, a (subject,
object) pair that appears under two or more predicates is the measured failure itself: a
value the model found, forced into every slot it could think of. What survives is every
predicate the registry knows — a turn can state one value under two real slots, "born in
Lisbon, still live in Lisbon" — and, only when it knows none of them, the first emitted.
The unknown predicates beside a known one are what pollution looks like, and they go. Per
turn rather than per batch, because `add()` extracts a whole batch in one call and two
turns agreeing on a value is evidence, not a failure. 21 of the 46, and every duplicate.

**R3 — a place predicate with something that is not a place.** `lives_in`, `born_in`,
`located_now` with a digit or a URL fragment in the object, on any subject. Narrow on
purpose: broadening it to every ONE-slot builtin would refuse `job_title: "Engineer II"`
and `timezone: "UTC+5:30"`. 5 of the 46.

There is deliberately no rule about the *subject*. The first draft refused every builtin
predicate on a subject other than `user`, reasoning that `works_at` on `gate` was a slot
collision waiting for the speaker's real employer. That reasoning was wrong — `gate /
works_at` and `user / works_at` are different slots and cannot end each other — and the
rule was measured to remove nothing R3 did not, while it would have refused every fact
about a named third party: `alice / lives_in / Porto` from "my wife Alice lives in Porto",
and every claim a two-person conversation yields, the store shape `read_route_roles=False`
exists for. Numbered R3 still, so the fixture's history reads straight.

**R4 — the discount.** Not a refusal and not in the table. A claim whose predicate is novel
(the registry would acquire it, not resolve it) or whose predicate is a ONE-cardinality
builtin carrying a digit or URL in its object — `born_on` and `timezone` excepted, whose
values always do — is stored at `min(confidence, 0.4)`. The reconciler closes an incumbent
only for a candidate worth at least half of it (`write/reconcile.py`), so pollution the
three rules cannot see can be present but cannot retire anything. Of the two arm-A escapes
in the fixture, `user / live_worker_version / "9000 memories a month"` is novel and lands
at 0.4; `user / goal / refuse` is untouched, because `goal` is MANY-cardinality and a
MANY slot ends nothing whatever arrives in it — the discount would protect nothing.

What this does not do: it does not know what `product_type` means. A wrong predicate on
subject `user` with a real value and a novel or MANY predicate is stored, discounted.
Closing that needs a per-predicate object kind on `PredicateSpec` or a second model call,
and `WriteReceipt.polluted` is what will say whether either is worth building.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..schema import BUILTIN_PREDICATES, Cardinality, PredicateRegistry
from ..types import SELF_SUBJECT

#: The three place predicates R3 checks, with their aliases folded in.
PLACE_PREDICATES: frozenset[str] = frozenset(
    {p.name for p in BUILTIN_PREDICATES if p.name in ("lives_in", "born_in", "located_now")}
    | {a for p in BUILTIN_PREDICATES if p.name in ("lives_in", "born_in", "located_now")
       for a in p.aliases})

#: ONE-cardinality builtins: the slots where a polluted claim ends a true fact. Less the
#: two whose values always carry digits — a birthday and a UTC offset — which R4 would
#: otherwise discount on every write, so a real `timezone` at 0.4 could never supersede
#: the stale one it was correcting. Neither slot appeared in the fixture as pollution.
FUNCTIONAL_PREDICATES: frozenset[str] = frozenset(
    p.name for p in BUILTIN_PREDICATES
    if p.cardinality is Cardinality.ONE and p.name not in ("born_on", "timezone"))

#: "Not a place": a digit, the word `port`, a URL fragment. The fixture was first scored
#: with a bare `port`, which also matches Porto, Portland and airport — every place in
#: the fixture happened to avoid it, and the suite's `the Porto office` did not. A word
#: boundary; `tests/test_pollution.py` holds the fixture numbers under this pattern.
_NOT_A_PLACE = re.compile(r"\d|\bport\b|http|://|\.dev\b|\.com\b")

#: Where R4 lands a discounted claim. Under half of a 1.0 incumbent, so the reconciler
#: stores it beside rather than ending — and low enough that a later real value at the
#: default 0.7 supersedes it cleanly.
DISCOUNTED_CONFIDENCE = 0.4


def _snake(raw: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(raw or "").lower()).strip("_")


def _subject(item: Mapping[str, Any]) -> str:
    return (str(item.get("subject", "") or "").strip() or SELF_SUBJECT).lower()


def _object(item: Mapping[str, Any]) -> str:
    return str(item.get("object", "") or "").strip().lower()


def guard(raw: Sequence[dict[str, Any]], registry: PredicateRegistry,
          ) -> tuple[list[dict[str, Any]], int, int]:
    """Apply R1–R4 to one extraction's proposed claims.

    Returns the claims to keep, in their original order, with R4's discount applied as a
    copy where it applies; how many R1–R3 refused; and how many R4 discounted. Items
    without a predicate or an object are passed through untouched — `_claim_from_dict`
    drops those and owns that count.
    """
    normalized = [registry.normalize(_snake(item.get("predicate", ""))) for item in raw]
    refused: set[int] = set()

    # R1: within one turn, one (subject, object) under several predicates → keep every
    # known predicate; keep the first unknown only when no known one exists.
    groups: dict[tuple[str, str, Any], list[int]] = {}
    for index, item in enumerate(raw):
        if not normalized[index] or not _object(item):
            continue
        key = (_subject(item), _object(item), item.get("source_index"))
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        known = {i for i in indices if registry.known(normalized[i])}
        keep = known if known else {indices[0]}
        refused.update(i for i in indices if i not in keep)

    for index, item in enumerate(raw):
        if index in refused or not normalized[index] or not _object(item):
            continue
        # R3: a place predicate with something that is not a place.
        if normalized[index] in PLACE_PREDICATES and _NOT_A_PLACE.search(_object(item)):
            refused.add(index)

    kept: list[dict[str, Any]] = []
    discounted = 0
    for index, item in enumerate(raw):
        if index in refused:
            continue
        predicate = normalized[index]
        if predicate and _object(item):
            novel = not registry.resolve(_snake(item.get("predicate", ""))).resolved
            risky_slot = (predicate in FUNCTIONAL_PREDICATES
                          and _NOT_A_PLACE.search(_object(item)) is not None)
            if novel or risky_slot:
                try:
                    confidence = float(item.get("confidence", 0.7))
                except (TypeError, ValueError):
                    confidence = 0.7
                if confidence > DISCOUNTED_CONFIDENCE:
                    item = {**item, "confidence": DISCOUNTED_CONFIDENCE}
                    discounted += 1
        kept.append(item)
    return kept, len(refused), discounted
