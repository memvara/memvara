"""Aggregate observability: the counters that make silent degradation visible.

`WriteReceipt` and `Explanation` already answer *what did this call do?*, precisely and
in a form you can read at a REPL. Nothing here duplicates them. The question they cannot
answer is the aggregate one — *is this store getting worse?* — and that is the question
that matters over a year, because a red-team review of eleven long-horizon failure modes
classified **six of them as silent**: they degrade answer quality with no error, no
exception and nothing in any log. Before this module the package had no counters, no
gauges and no timing hooks in roughly four thousand lines, so not one of the six was
observable from outside.

The six, and the series that catches each:

===========================================  ==========================================
failure                                      signal
===========================================  ==========================================
predicate explosion                          ``consolidate.claims_per_slot``,
                                             ``predicate.learned``
reinforcement not refreshing recency         ``retrieval.observation_rank_corr`` — must
                                             stay positive
flip-flop row growth, counters reset         ``consolidate.claims_per_slot`` over time,
                                             ``consolidate.merged`` stuck at 0
salience overriding relevance                ``retrieval.quality_factor`` distribution
gate / fast-path English-centrism            ``gate.*`` and ``fast.*``, sliced by
                                             ``script``
poisoning / a retraction that retires        ``write.retraction{outcome="noop"}``
nothing
===========================================  ==========================================

Waves 1 and 2 fixed the mechanisms behind the first three. That makes these series more
valuable rather than less: they are how anyone knows the fixes are still holding in six
months. If only one of them is ever wired to a dashboard it should be
``consolidate.claims_per_slot``, which is the one that would have caught the worst bug
in the project in its first week.

**A seventh arrived with the redaction seam**, and it is the same shape as the other six
with a worse consequence. A configured `Redactor` whose rules stop matching the data —
a new phone format, a different locale, a vendor changing an id shape — raises nothing,
logs nothing, and makes the write path *faster*; the only symptom is unredacted personal
data sitting in the store, found by an auditor rather than by an alert. It is caught by
``redact.changed`` against ``redact.inspected``, sliced by field and script, and by the
disappearance of ``redact.inspected`` altogether while ``write.turns`` keeps climbing —
which is the same failure one level up, a deployment that lost its policy.

Cost, which is the whole argument of this library and therefore not negotiable. The
default recorder is ``None``, not a no-op object: an unset recorder costs one
``is not None`` test per emission point and nothing else — no call, no tuple of tags
built, no string formatted. Every metric that requires *computing* something (a script
classification, a rank correlation, a per-slot histogram) is computed **inside** that
guard, never before it. An always-on hook that showed up in a benchmark would be
self-defeating, so it must be provable rather than asserted; see
``tests/test_telemetry.py::test_an_unset_recorder_costs_nothing_measurable``.

Emission is fire-and-forget by design. A recorder that raises propagates, exactly like a
logging handler that raises: swallowing it would mean the observability layer can fail
silently, which is the failure mode this module exists to remove.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Protocol, Sequence

# The metric-name constants below are part of the public surface too and are imported
# by name; they are left out of `__all__` only so a star-import does not drop two dozen
# bare strings into a caller's namespace. `series_names()` enumerates them.
__all__ = [
    "Recorder",
    "NullRecorder",
    "MemoryRecorder",
    "script_of",
    "rank_correlation",
    "series_names",
    "CROWDED_SLOT",
]


class Recorder(Protocol):
    """Where aggregate measurements go. Implement three methods, get all of them.

    Deliberately smaller than a metrics library: counters, gauges and timings map onto
    every backend anyone would plug in here (statsd, Prometheus, OpenTelemetry, a dict)
    without this package taking a dependency on one, and a wider protocol would make
    third-party implementations harder to write for no gain in what can be measured.

    `tags` are keyword arguments so a call site reads as prose —
    ``rec.counter(GATE_DROP, reason="ack_only", script="latin")`` — and `name` and
    `value` are positional-only so a dimension may legitimately be called either.

    Implementations must not raise for an unknown metric name. The catalogue below is
    stable, but a future emission point should land in a backend as a new series rather
    than as an exception on a write path.
    """

    def counter(self, name: str, value: int = 1, /, **tags: str) -> None:
        """Add `value` to a monotonically increasing series."""
        ...

    def gauge(self, name: str, value: float, /, **tags: str) -> None:
        """Record one observation of a quantity that moves in both directions."""
        ...

    def timing(self, name: str, ms: float, /, **tags: str) -> None:
        """Record one duration, in milliseconds."""
        ...


# --- the metric catalogue ----------------------------------------------------
#
# Names live here rather than as literals at the emission points so that a rename
# cannot silently split one series in two, and so that a test asserts on the same
# constant the code emits. Dotted, lowercase, subsystem first: every backend this is
# likely to be pointed at treats that prefix as a namespace.

#: Turns handed to `WritePipeline.add`. The denominator for everything else on the
#: write path — `write.llm_calls / write.turns` is the number the design exists to
#: drive toward zero, and a ratio needs both halves.
WRITE_TURNS = "write.turns"

#: Claims written directly, without extraction — `remember()`, `supersede()`,
#: `assert_claim()` and the importer, which all converge on `WritePipeline.assert_claim`.
#: One per call, whatever the call displaced.
#:
#: **`write.turns` does not count these, and on a deployment with no extraction model
#: they are all of the write traffic there is.** That deployment's only reliable write
#: path is the one that skips extraction, so a dashboard sourced from `write.turns` alone
#: reports a store nobody is writing to while it fills up. The two series answer
#: different questions — "how much conversation was ingested" against "how many facts
#: were asserted" — and neither substitutes for the other, which is why this is a second
#: series rather than a wider definition of the first.
#:
#: Not a billing series. `write.turns` is what an allowance is spent against, and an
#: asserted fact spends none; adding this to that sum would charge for writes the
#: published contract says are free.
WRITE_CLAIMS = "write.claims"

#: Model calls actually made, aggregated. `WriteReceipt.llm_calls` is the per-call
#: answer; this is the one you alert on.
WRITE_LLM_CALLS = "write.llm_calls"

#: Tokens consumed, in and out, by every model call one write made — the extraction and
#: any predicate acquisition it triggered. Emitted only by a backend that advertises
#: `reports_usage` and actually came back with a usage block; a backend that cannot
#: measure publishes no series rather than a run of zeros, because zero here is a
#: quantity a real model call cannot consume and would read as free.
#:
#: **This is the series to bill on, not `write.llm_calls`.** Providers charge per token
#: and the ratio between the two is unbounded: a one-line turn and a 40,000-token
#: document are both exactly one call. Input and output are separate because they are
#: priced separately, usually several-fold apart, so a single total cannot be costed
#: without knowing the split.
WRITE_TOKENS_IN = "write.tokens_in"
WRITE_TOKENS_OUT = "write.tokens_out"

#: What reconciliation decided, tagged `action=add|reinforce|supersede|retract|noop`.
#: Row growth in a slot shows up here as `add` climbing while `supersede` stays flat.
WRITE_RECONCILE = "write.reconcile"

#: A negative-polarity write, tagged `outcome=retired|noop`. **A retraction that closes
#: out zero claims is an anomaly**, and it is the signature of both halves of the
#: adversarial case: a poisoned assertion the user cannot take back, and a `forget()`
#: whose predicate or object does not match what is actually on record. Silent today —
#: the API returns a perfectly ordinary receipt either way.
#:
#: `outcome="retired"` predates the `ended`/`retired` split and means "it closed
#: something out", not `Claim.state == "retired"` — a retraction ordinarily *ends* its
#: targets. The label is kept as it is because it is what existing alerts match on, and
#: the distinction this series exists to draw is hit-versus-miss, not which clock moved.
WRITE_RETRACTION = "write.retraction"

#: An embedding the store refused. The write path warns once per process and then goes
#: quiet forever, so without this a misconfigured embedder produces a store that is
#: fully populated and unsearchable by meaning, with one line in a log from last month.
WRITE_EMBEDDING_REJECTED = "write.embedding_rejected"

#: An embedding the store accepted and that carries no information: every component zero,
#: so cosine against it is zero against everything and retrieval abstains rather than
#: ranking it. The store is happy, the write succeeds, and the claim is reachable by
#: predicate and by lexical match but never by meaning. Counted separately from
#: `WRITE_EMBEDDING_REJECTED` because nothing raised — this is the failure that leaves no
#: exception anywhere, and the only two signals it can produce are this series and one
#: warning per process.
WRITE_EMBEDDING_UNUSABLE = "write.embedding_unusable"

#: End-to-end `add()` duration.
WRITE_LATENCY_MS = "write.latency_ms"

#: How long the write held the store's transaction, which on `SQLiteStore` is how long
#: it held the process-wide lock. The specific number the tier-hoist in `pipeline.py`
#: moved: extraction used to run inside this window, so it included an Anthropic round
#: trip. Watch the gap between this and `write.latency_ms` — that gap is the work the
#: rest of the process was *not* blocked by.
WRITE_LOCK_HELD_MS = "write.lock_held_ms"

#: The model round trip alone: `LLM.extract` in, out. Emitted only when a model was
#: actually consulted, so a `NullLLM` deployment reports no series rather than a series
#: of zeros — the same rule `is_noop` applies to `write.llm_calls`, and for the same
#: reason. Includes the request that raised, because a provider timeout is latency the
#: caller waited through and excluding it makes the p99 improve during an outage.
#:
#: This exists because "extraction time" was previously only recoverable as
#: `write.latency_ms` minus `write.lock_held_ms`, and a difference of two aggregates is
#: not a distribution: percentiles do not subtract, so that arithmetic has a mean and no
#: recoverable p99. The slow tail of the model call is the thing worth alerting on and it
#: was the one shape the write path could not show.
WRITE_EXTRACT_MS = "write.extract_ms"

#: Strings offered to the configured `Redactor`, and how many of them it rewrote. Both
#: tagged `field` (one of `memvara.redact.FIELDS`) and `script`, and emitted only when a
#: redactor is actually configured — no policy, no policy metrics, and no `script_of`
#: bill for the deployments that run without one.
#:
#: **The ratio is the signal, not either half.** A count of redactions on its own cannot
#: be read: zero today is "the policy has silently stopped matching" and "no personal
#: data arrived today" at the same time, and for most tenants the second is the normal
#: case. `redact.changed / redact.inspected` per field and per script is a rate that
#: holds roughly steady for a given tenant and workload, so a tenant that sat at 3% for a
#: year and now sits at 0 has had its data drift out from under its rules — silent
#: otherwise, and *cheaper* than working, since a rule that matches nothing is a rule
#: that substitutes nothing.
#:
#: `redact.changed` is emitted **even at zero**, for the reason `consolidate.merged` is:
#: a policy that ran and matched nothing has to be distinguishable from a policy that is
#: not running, and only the reported zero tells those apart. The absence of
#: `redact.inspected` while `write.turns` climbs is the second, blunter failure — the
#: `redactor=` argument was dropped from a deployment's construction — and it is worth an
#: alert on absence rather than on value.
#:
#: Sliced by `script` on the same argument as `gate.*`: pattern rules are written against
#: the formats of one locale, and a rule set matching 3% of Latin-script turns and 0% of
#: everything else reads as healthy in the aggregate while covering one population and no
#: other. Sliced by `field` because a policy is legitimately allowed to differ across
#: them — aggressive on raw turns, conservative on claim objects — so one rate over all
#: four fields would average away the field a deployment actually cares about.
#:
#: The unit is **strings inspected**, not spans removed. `Redactor.redact` returns text
#: and nothing else, so a span count would mean widening the protocol for one metric and
#: then trusting every third-party implementation to count its own work honestly.
REDACT_INSPECTED = "redact.inspected"
REDACT_CHANGED = "redact.changed"

#: Tier-1 gate outcomes, tagged `reason` (the gate's own slug) and `script`. The gate's
#: filler vocabulary and its sentence rules are English; the script slice is what turns
#: "the gate works" into "the gate works *for Latin-script users*", which is the honest
#: claim until the numbers say otherwise.
GATE_PASS = "gate.pass"
GATE_DROP = "gate.drop"

#: Whether the deterministic extractor handled a turn, tagged `script`. Its patterns are
#: English sentence forms, so a hit rate that collapses on one script means those users
#: pay for a model call on every single turn — a cost difference, not just a quality one.
FAST_HIT = "fast.hit"
FAST_MISS = "fast.miss"

#: Predicate acquisition outcomes. `predicate.learned` is *novel registrations*, the
#: rate that ran away in the simulation that produced 41 predicates for six concepts;
#: `predicate.alias` is a surface form folded onto something we already had, which is
#: the outcome you want to dominate; `predicate.capped` is the backstop firing, tagged
#: `folded=yes|no`, and any of it at all means the registry ceiling is now load-bearing.
PREDICATE_LEARNED = "predicate.learned"
PREDICATE_ALIAS = "predicate.alias"
PREDICATE_CAPPED = "predicate.capped"

#: A value written into a slot that already held live values, under a predicate the
#: registry has no spec for. One per such write, and untagged for the reason
#: `predicate.learned` is: there is one outcome here, and the dimension anyone would
#: actually want to slice by — the predicate name — is unbounded and would be a
#: cardinality bomb in every backend this is likely to be pointed at. The names are on
#: the write receipts (`WriteReceipt.accumulated`), which is where a per-slot question
#: belongs.
#:
#: **What a non-zero value means.** Somewhere in this deployment, writes are landing on
#: predicates whose cardinality nobody ever decided, in slots that were already occupied.
#: Unregistered means MANY, MANY retires nothing, so each of those slots now answers a
#: present-tense question with two or more simultaneous answers — and `predicate.capped`
#: aside, no other series moves when it happens: `write.reconcile{action="add"}` is what a
#: correct first write looks like too.
#:
#: **What to do about it.** Read the predicate names off the receipts, then decide each
#: one, which is a decision this library deliberately will not make for you: declare it
#: `Cardinality.ONE` if the slot holds one value at a time (`status`, `version`, `owner`,
#: `stage` — the vocabulary an agent recording project state reaches for, and all of them
#: single-valued in meaning), or `Cardinality.MANY` if it genuinely accumulates
#: (`tagged_with`, `attended`). Either declaration silences this for that predicate
#: permanently; the first one also makes the next write supersede instead of pile up.
#:
#: **The pair that names the cause.** This climbing while `predicate.learned` stays at
#: zero is the structural case rather than a slow vocabulary: acquisition is not running
#: at all. Two deployments do that — one with no extraction model, where there is no
#: acquisition to run, and one where the writes arrive through `remember()`, which reaches
#: the reconciler without ever passing the tier that learns a spec. Neither will ever
#: register a predicate on its own, so for both the schema has to be declared by hand and
#: this counter will not fall until it is.
PREDICATE_ACCUMULATED = "predicate.accumulated"

#: One per `search()`, tagged `script`. Pairs with the gate slices: a script with query
#: volume and no `gate.pass` is a population whose writes are being dropped.
RETRIEVAL_QUERY = "retrieval.query"

#: How many results a search actually returned. Zero is the interesting value — it is
#: `min_score` working, or the corpus not answering, and both are worth a trend line.
RETRIEVAL_RESULTS = "retrieval.results"

#: Per returned claim: the quality multiplier divided by its own nominal maximum, i.e.
#: the factor by which recency, confidence and salience moved this result away from its
#: pure retriever evidence. Expressed against the normalized score rather than against
#: `fusion_score`, because fusion no longer drives the ranking — see
#: `scoring.normalized_score`.
#:
#: **Above 1.0 is the alarm.** The design intent is that quality can only pull a result
#: *down* from its evidence, by at most `1 / span`; the one way past 1.0 is a salience
#: reinforced beyond 1.0, which `scoring.quality_boost` deliberately does not clamp
#: because it is headroom the write path earns. Emitted unclamped for exactly that
#: reason: a distribution creeping above 1.0 while precision falls is salience
#: outranking relevance, and clamping here would hide the only direct evidence of it.
RETRIEVAL_QUALITY_FACTOR = "retrieval.quality_factor"

#: Spearman correlation between a result's `observation_count` and how well it ranked,
#: per search. **Positive is correct**: a fact the user has restated many times should
#: rank above one mentioned once. It went negative when reinforcement was written onto
#: the decayed `salience` instead of the storage base, which the nightly pass then
#: erased — a failure with no error and no exception anywhere in it.
RETRIEVAL_OBSERVATION_RANK_CORR = "retrieval.observation_rank_corr"

#: End-to-end `search()` duration.
RETRIEVAL_LATENCY_MS = "retrieval.latency_ms"

#: **Live claims in the most crowded slot**, per consolidation pass. If exactly one
#: series from this module is ever put on a dashboard, make it this one: thirteen
#: simultaneously-live answers to "where does the user work?" was the worst defect in
#: the project's history, it ran for the whole simulation without producing a single
#: log line, and this number would have shown it in week one.
CONSOLIDATE_CLAIMS_PER_SLOT = "consolidate.claims_per_slot"

#: How many slots hold more than `CROWDED_SLOT` live claims. The maximum above says how
#: bad the worst case is; this says whether it is one pathological subject or the whole
#: store drifting.
CONSOLIDATE_CROWDED_SLOTS = "consolidate.crowded_slots"

#: Claims each stage changed. Emitted **even when zero**, deliberately: "merge has
#: reported 0 for three months" is only distinguishable from "nobody is running
#: consolidation" if the zero is actually reported.
CONSOLIDATE_DECAYED = "consolidate.decayed"
CONSOLIDATE_MERGED = "consolidate.merged"
CONSOLIDATE_PROMOTED = "consolidate.promoted"

#: Rows a pass wrote back, and how long the whole pass took.
CONSOLIDATE_ROWS_WRITTEN = "consolidate.rows_written"
CONSOLIDATE_LATENCY_MS = "consolidate.latency_ms"

#: Live claims in one slot beyond which the slot is called crowded. Three, because a
#: single-valued predicate should hold exactly one and a multi-valued one legitimately
#: holds a few; the pathology that motivated the metric held thirteen.
CROWDED_SLOT = 3


class NullRecorder:
    """A recorder that discards everything. Correct, and not the fast path.

    The fast path is `telemetry=None`, which skips the call entirely. This exists for
    callers who would rather hold an object than a `None` — a server wiring one recorder
    per deployment mode, a test asserting the protocol is satisfiable — and as the
    reference for what the three methods must accept.
    """

    def counter(self, name: str, value: int = 1, /, **tags: str) -> None:
        """Discard a counter increment."""

    def gauge(self, name: str, value: float, /, **tags: str) -> None:
        """Discard a gauge observation."""

    def timing(self, name: str, ms: float, /, **tags: str) -> None:
        """Discard a timing observation."""


#: A metric series: its name plus its tags, sorted so two calls that pass the same tags
#: in a different order land in the same bucket.
Series = tuple[str, tuple[tuple[str, str], ...]]


def _series(name: str, tags: Mapping[str, str]) -> Series:
    return name, tuple(sorted(tags.items()))


class MemoryRecorder:
    """Everything, in memory, keyed by series. For tests, notebooks and small servers.

    Counters accumulate; gauges and timings keep every observation, because the point of
    `retrieval.quality_factor` and `consolidate.claims_per_slot` is their *distribution*
    and a running mean throws that away. That makes this unbounded, which is correct for
    a test and wrong for a month-long process — point a real backend at anything
    long-lived.

    >>> rec = MemoryRecorder()
    >>> rec.counter(GATE_DROP, reason="ack_only", script="latin")
    >>> rec.counter(GATE_DROP, reason="question", script="latin")
    >>> rec.counter(GATE_DROP, reason="ack_only", script="han")
    >>> rec.total(GATE_DROP), rec.total(GATE_DROP, script="latin")
    (3, 2)
    >>> rec.gauge(CONSOLIDATE_CLAIMS_PER_SLOT, 13.0)
    >>> rec.values(CONSOLIDATE_CLAIMS_PER_SLOT)
    [13.0]
    """

    def __init__(self) -> None:
        self.counters: dict[Series, int] = {}
        self.gauges: dict[Series, list[float]] = {}
        self.timings: dict[Series, list[float]] = {}

    def counter(self, name: str, value: int = 1, /, **tags: str) -> None:
        key = _series(name, tags)
        self.counters[key] = self.counters.get(key, 0) + int(value)

    def gauge(self, name: str, value: float, /, **tags: str) -> None:
        self.gauges.setdefault(_series(name, tags), []).append(float(value))

    def timing(self, name: str, ms: float, /, **tags: str) -> None:
        self.timings.setdefault(_series(name, tags), []).append(float(ms))

    # -- reading it back ------------------------------------------------------

    def total(self, name: str, **tags: str) -> int:
        """Sum every counter series called `name` whose tags include `tags`.

        The tags are a *filter*, not an exact match, which is what makes a slice
        readable: `total(GATE_DROP)` is the whole rate and
        `total(GATE_DROP, script="han")` is one population's share of it, with no need
        to know which other dimensions the emission point happens to carry.
        """
        hit = sum(v for k, v in self.counters.items() if _matches(k, name, tags))
        if hit == 0 and not any(_matches(k, name, tags) for k in self.counters):
            # A gauge or a timing read through `total` answered 0, which is a number a
            # counter can legitimately have — so the caller could not tell "nothing
            # happened" from "wrong method". `consolidate.claims_per_slot` is the most
            # valuable series in this module and it is a gauge; reading it this way gave
            # a plausible zero and nothing pointed anywhere else.
            for store, how in ((self.gauges, "values"), (self.timings, "values")):
                if any(_matches(k, name, tags) for k in store):
                    raise TypeError(
                        f"{name!r} is not a counter, so `total` cannot sum it — it would "
                        f"answer 0 whatever was recorded. Use `{how}({name!r})` for the "
                        "observations, or `len(...)` for how many there were.")
        return hit

    def values(self, name: str, **tags: str) -> list[float]:
        """Every gauge or timing observation matching `name` and `tags`, in order.

        Gauges and timings share this accessor because the distinction between them is
        about how a backend should aggregate, not about what was measured, and a caller
        reading values back already knows which one it asked for.
        """
        out: list[float] = []
        for source in (self.gauges, self.timings):
            for key, observations in source.items():
                if _matches(key, name, tags):
                    out.extend(observations)
        return out

    def names(self) -> list[str]:
        """Every series name seen, sorted. The cheapest way to eyeball what was wired."""
        return sorted({name for name, _ in
                       (*self.counters, *self.gauges, *self.timings)})

    def __repr__(self) -> str:
        return (f"<MemoryRecorder {len(self.counters)} counters "
                f"{len(self.gauges)} gauges {len(self.timings)} timings>")


def _matches(key: Series, name: str, tags: Mapping[str, str]) -> bool:
    if key[0] != name:
        return False
    present = dict(key[1])
    return all(present.get(k) == v for k, v in tags.items())


# --- slicing dimensions ------------------------------------------------------

# Coarse script ranges, in the order they are tested. This is a *slicing dimension for
# metrics*, not language identification: the question it answers is "is the English-only
# gate vocabulary being applied to text it was never designed for", and for that a dozen
# buckets is plenty. Ranges are the common Unicode blocks; anything unlisted is "other",
# which is itself a useful bucket because a rising "other" means this table needs a row.
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0041, 0x024F, "latin"),      # basic Latin letters through Latin Extended-B
    (0x0370, 0x03FF, "greek"),
    (0x0400, 0x04FF, "cyrillic"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"),
    (0x0900, 0x097F, "devanagari"),
    (0x0E00, 0x0E7F, "thai"),
    (0x3040, 0x30FF, "kana"),       # hiragana + katakana
    (0x3400, 0x4DBF, "han"),        # CJK extension A
    (0x4E00, 0x9FFF, "han"),
    (0xAC00, 0xD7AF, "hangul"),
    (0xFF66, 0xFF9F, "kana"),       # halfwidth katakana
)

#: How much of a turn is examined. Script is stable across a sentence and this runs on
#: the write path, so reading the whole of a 5,000-character document to reach the same
#: answer the first line gives is cost with no information in it.
_SCRIPT_SAMPLE = 128


def script_of(text: str) -> str:
    """The dominant script in `text`, as a coarse metric dimension.

    Returns the script with the most characters in the first `_SCRIPT_SAMPLE`, ties
    broken alphabetically so the same input always produces the same bucket. Text with
    no letters at all — punctuation, digits, an empty turn — is `"none"`, which is
    distinct from `"other"`: the first says there was nothing to classify, the second
    says there was and this table does not cover it.

    Kana beats Han when both are present, because a Japanese sentence is mostly kanji by
    weight and mostly Japanese by fact; without the rule, `日本に住んでいます` and
    `我住在北京` land in the same bucket and the slice stops separating the two
    populations it exists to separate.

    >>> script_of("I moved to Berlin last spring")
    'latin'
    >>> script_of("我住在北京")
    'han'
    >>> script_of("日本に住んでいます")
    'kana'
    >>> script_of("서울에 살고 있습니다")
    'hangul'
    >>> script_of("1234 !?"), script_of("\\u05d0\\u05d1\\u05d2")
    ('none', 'hebrew')
    """
    counts: dict[str, int] = {}
    for ch in text[:_SCRIPT_SAMPLE]:
        if not ch.isalpha():
            continue
        code = ord(ch)
        name = "other"
        for low, high, script in _SCRIPT_RANGES:
            if low <= code <= high:
                name = script
                break
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "none"
    if "kana" in counts and "han" in counts:
        counts["kana"] += counts.pop("han")
    return min(counts, key=lambda s: (-counts[s], s))


def rank_correlation(values: Sequence[float]) -> float | None:
    """Spearman correlation between `values` and *rank quality*, in [-1, 1].

    `values[i]` is a quantity observed at rank `i`, best first. The result is positive
    when high values sit at good ranks, so for `observation_count` it answers the
    question the reviewer actually asked — *is re-observing a fact making it easier to
    retrieve?* — with a single number whose correct sign is known in advance.

    Spearman rather than Pearson because only the ordering is meaningful: an
    `observation_count` of 40 against 20 is not "twice as retrievable", and one heavily
    reinforced outlier must not be able to carry the statistic on its own. Ties take the
    average rank, which is what makes a store where most claims have been seen once
    report a weak correlation instead of an arbitrary one.

    `None` when there is nothing to correlate — fewer than two results, or every value
    identical. That is an absence of evidence rather than a correlation of zero, and
    emitting it as 0.0 would drag a dashboard's average toward "broken" every time a
    search returned two equally-fresh facts.

    >>> rank_correlation([9, 5, 4, 2, 1])      # best-attested ranked first
    1.0
    >>> rank_correlation([1, 2, 4, 5, 9])      # exactly backwards
    -1.0
    >>> rank_correlation([3, 3, 3]) is None    # nothing to say
    True
    >>> rank_correlation([7]) is None
    True
    """
    n = len(values)
    if n < 2:
        return None
    value_ranks = _tied_ranks(values)
    mean = (n - 1) / 2.0                      # mean of both 0..n-1 rank vectors
    spread = sum((r - mean) ** 2 for r in value_ranks)
    if spread <= 0.0:
        return None                           # every value identical: no ordering at all
    # `position` ascends with rank index (0 is best), so a positive Pearson here means
    # high values at *bad* ranks. Negated once, at the end, rather than by reversing one
    # of the two vectors - which is the same arithmetic and half as easy to check.
    covariance = sum((r - mean) * (position - mean)
                     for position, r in enumerate(value_ranks))
    positions_spread = sum((position - mean) ** 2 for position in range(n))
    return -covariance / (spread * positions_spread) ** 0.5


def _tied_ranks(values: Sequence[float]) -> list[float]:
    """Ascending ranks with ties averaged: `[10, 20, 20, 30] -> [0, 1.5, 1.5, 3]`."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        shared = (start + stop) / 2.0
        for i in order[start:stop + 1]:
            ranks[i] = shared
        start = stop + 1
    return ranks


def series_names() -> Iterable[str]:
    """Every metric name this package emits, sorted. The catalogue, programmatically.

    A dashboard, an allow-list in a metrics proxy, or a test that a new emission point
    was added to the documented set can read this instead of grepping for string
    literals.

    >>> names = list(series_names())
    >>> len(names) == len(set(names)) and all("." in n for n in names)
    True
    >>> names[:2]
    ['consolidate.claims_per_slot', 'consolidate.crowded_slots']
    """
    return sorted(
        value for name, value in globals().items()
        if name.isupper() and isinstance(value, str) and "." in value
    )
