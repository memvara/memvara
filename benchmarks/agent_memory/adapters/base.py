"""The interface a memory system implements to be benchmarked.

Five methods. `reset` clears the memory and hands over the predicate schema, `remember`
delivers one observation, `query` answers one question, `usage` reports what the run cost,
and `close` releases whatever was opened. Nothing else is required, and nothing about the
benchmark's internals is exposed to an implementation.

`usage` is easy to forget — it was missing from this list, and from the count in four other
documents, until a review noticed that `registry.build` requires it and everything
describing it said four. A system that measures nothing returns a bare `Usage()`; it still
has to return one.

## What the answer has to be

`MemoryAnswer` carries a string, a set of strings, or neither — the third case being an
explicit "I do not know", which is a real answer here and the only correct one for the
`negative` category. Scoring is deterministic (`scoring.py`): normalized comparison
against the gold value and its published aliases. No model reads the answer, so an
adapter that returns a sentence rather than a value is not penalised for style — but it
is not rewarded for hedging either, because `contains the gold string` is not how any
category is scored. Return the value.

## What an adapter may and may not do

It may use anything its system offers: structured writes, temporal queries, ranking,
graph traversal, whatever. It may read `Question.probe`, `Question.at` and
`Question.known_at`, because every adapter gets those and the dataset decides which
questions carry them.

It may not read `Question.gold`, and it is never given it: the runner strips it before
the adapter is called. That is the one rule the code enforces rather than asks for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol, Sequence, runtime_checkable

from ..dataset import MemoryEvent, PredicateDecl
from ..normalization import tokens


@dataclass(frozen=True, slots=True)
class Ask:
    """A question as an adapter sees it — the gold answer removed.

    Built by the runner from a `Question`. An adapter cannot reach the answer through
    this object, which is what makes "no system-specific hidden test knowledge" a
    property of the code rather than a promise in a README.
    """

    id: str
    category: str
    question: str
    #: `(subject, predicate)` when the dataset names the slot, else `None` and the system
    #: has to find it.
    probe: tuple[str, str] | None
    #: The world instant the question is about. Always set by the runner — `None` in the
    #: dataset means "now", and "now" is the dataset's fixed `evaluated_at`.
    at: datetime
    #: What the dataset means by "now" — a constant in `metadata.json`, not the wall
    #: clock, so a run tomorrow scores the same as a run today. `at == evaluated_at` is
    #: how an adapter tells a present-tense question from one about the past, which
    #: decides which population of memories it should be searching.
    evaluated_at: datetime
    #: The belief instant, when the question asks what the system would have said then.
    #: `None` means "as we understand it today".
    known_at: datetime | None = None
    #: The value the question is about, for the `provenance`, `change_time` and
    #: `knowledge_time` categories: "which source reported *London*", "when did *London*
    #: begin". `None` everywhere else. It is in the data rather than only in the prose so
    #: that answering does not require parsing the question — see `dataset.Question`.
    about: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryAnswer:
    """What an adapter returns.

    `value=None` with an empty `values` is an abstention, and it is scored as one.
    Returning a wrong value and returning nothing are different outcomes and the report
    keeps them apart.
    """

    #: The answer to a single-valued or date question. ISO-8601 for a date.
    value: str | None = None
    #: The answer to a set-valued question. Order is ignored.
    values: tuple[str, ...] = ()
    #: Ids — of events, claims, rows, whatever the system calls them — that justify the
    #: answer. Never scored. Printed in the failure report, where knowing which memory a
    #: wrong answer came from is most of the debugging.
    support: tuple[str, ...] = ()

    @property
    def abstained(self) -> bool:
        return self.value is None and not self.values


@dataclass
class Usage:
    """Cost counters, reported by whichever system can count them.

    Every field defaults to `None`, meaning *not measured*, which the report prints as
    `-` rather than as zero. A system that does not count its database reads should say
    so; reporting an unmeasured quantity as zero is the one way this section could lie.
    """

    llm_calls: int | None = None
    #: Model tokens spent, prompt and completion together. `None` where a system does not
    #: count them; `0` is a claim, and a true one for any system reporting `llm_calls=0`.
    #:
    #: Present although every system shipped here reports zero, because a system that
    #: *does* use a model has no field for the cost that dominates its bill, and would
    #: otherwise have to hide it in `extra` where nothing compares it. An adapter with a
    #: model in the loop must also disclose which one — see
    #: `benchmarks/agent_memory/CONTRIBUTING.md`.
    tokens: int | None = None
    #: Texts submitted for embedding, **not** requests made. Batching is an
    #: implementation detail of an embedding API — you send 64 or 100 texts per request
    #: because that is how they are billed — so counting requests would report two
    #: systems doing identical work as orders of magnitude apart.
    #:
    #: Defined because the field next to it was not, and that cost a published number
    #: once already: `rows_stored` used to be `db_writes`, undefined, and three adapters
    #: reported three different quantities through it. This one was left undefined in the
    #: same commit that fixed that, and immediately grew the same split — memvara counting
    #: texts, `vector-rag` counting calls, agreeing only because nothing batches yet.
    texts_embedded: int | None = None
    #: Rows the store **holds** once every event has been delivered — not the number of
    #: write calls, which is always the event count and so says nothing about any system.
    #:
    #: Defined here because it was not, and three adapters answered three different
    #: questions with it: memvara reported rows, the two baselines reported calls, and the
    #: published table compared them under one heading. `naive` overwrites, so it holds
    #: fewer rows than it received; memvara holds more than it currently believes, because
    #: a value that stopped being true is kept. That difference is the point of the column
    #: and the old field name hid it.
    rows_stored: int | None = None
    #: Read calls the benchmark made — one per question, for every system. Kept because a
    #: system that reads more than once per question would show it here.
    db_reads: int | None = None
    #: Anything a system wants recorded that has no field here. Serialized as-is.
    extra: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        row: dict[str, object] = {
            "llm_calls": self.llm_calls,
            "tokens": self.tokens,
            "texts_embedded": self.texts_embedded,
            "rows_stored": self.rows_stored,
            "db_reads": self.db_reads,
        }
        if self.extra:
            row["extra"] = dict(self.extra)
        return row


@runtime_checkable
class MemorySystem(Protocol):
    """What the runner needs from a memory system.

    Implementations live in this package or anywhere importable; `--system` accepts a
    dotted path, so a third-party system can be benchmarked without this repository
    knowing it exists.
    """

    #: Short identifier used on the command line and in the result file.
    name: str
    #: The version of the *system under test*, not of this adapter. Recorded in the
    #: result so a score can be traced to what produced it.
    version: str

    def reset(self, predicates: Mapping[str, PredicateDecl]) -> None:
        """Discard all memory and adopt the dataset's predicate schema.

        Called **once per run**, before the first event. Every scenario's events go into
        one memory and every question is asked against all of them — scenario is a label
        for reporting, not an isolation boundary, and entities are unique across the
        dataset so nothing collides. `runner.py` says why: a benchmark that gave each
        scenario its own three-fact store would make retrieval trivial.
        """

    def remember(self, event: MemoryEvent) -> None:
        """Take delivery of one observation.

        Events arrive in `recorded_at` order. `event.valid_from` may be earlier than
        `event.recorded_at`, and for the delayed-knowledge scenarios it is.
        """

    def query(self, ask: Ask) -> MemoryAnswer:
        """Answer one question."""

    def usage(self) -> Usage:
        """Counters accumulated since the last `reset`, or an all-`None` `Usage`."""

    def close(self) -> None:
        """Release anything held. Called once at the end of a run."""


def indexable(event: MemoryEvent) -> str:
    """The text every adapter indexes for retrieval: the triple, then the sentence.

    Here rather than in each adapter because retrieval is a scored dimension, and three
    adapters indexing three different strings would have made that dimension a comparison
    of what each one happened to feed its retriever.

    It was three different strings. `vector-rag` indexed subject, predicate and sentence;
    `naive` matched on the sentence alone; the memvara adapter passed the sentence alone
    as `Claim.text`, which is what memvara embeds and BM25-indexes. That last one cost
    real points on entities whose events are written in the first person — "I have
    relocated to Madrid" does not contain the word *Heidi*, so no query naming her could
    reach it — and the loss looked like weak ranking rather than a missing word.

    Both halves earn their place. The triple carries the subject and relation, which
    first-person sentences drop; the sentence carries the vocabulary a question is
    actually phrased in.
    """
    relation = event.predicate.replace("_", " ")
    return f"{event.subject} {relation} {event.object}. {event.text}"


#: Prepositions dropped from a predicate name before it is matched against a question.
#: Small and closed for the reason `normalization._ARTICLES` is: `works_on` would
#: otherwise score a point for the word *on*, which every second English question
#: contains, and `deploy_region` would lose a two-hop question to it on a tie.
_PREDICATE_FUNCTION_WORDS = frozenset({"on", "by", "in", "at", "of", "to", "for", "with"})

#: Suffixes stripped before a question word and a predicate word are compared. Crude, and
#: deliberately so: *deployed* has to reach `deploy_region` and *owns* has to reach
#: `owned_by`, and a real stemmer is a dependency and a second thing to explain.
#:
#: `es` is deliberately absent. With it, `lives_in` folds to *liv* while the question word
#: *live* folds to itself, and the two stop matching — the rule would then be blind to the
#: relation it most often has to recognise. Leaving it out costs a stem like *matche* for
#: *matches*, which nothing here compares against anything.
_SUFFIXES = ("ing", "ed", "s")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _predicate_words(predicate: str) -> frozenset[str]:
    return frozenset(_stem(w) for w in predicate.split("_")
                     if w not in _PREDICATE_FUNCTION_WORDS)


def pick_slot(question: str,
              candidates: Sequence[tuple[str, str]]) -> tuple[str, str] | None:
    """Choose the fact slot an unprobed question is about, from ranked candidates.

    `candidates` is `(subject, predicate)` best-first, as each system's own retrieval
    ranked them. The rule: **prefer the highest-ranked candidate whose predicate the
    question actually names**, and fall back to rank alone when none does.

    Here rather than in each adapter, and identical for all of them, because retrieval is
    a scored dimension. A rule that lived in one adapter would be measuring the harness.

    ## Why rank alone is not enough

    Taking the top hit answers the first hop of a chained question and stops. Asked
    *"Which region is the project Alice works on deployed to?"*, memvara and `vector-rag`
    both rank `alice works_on Project Atlas` first — correctly; it is the closest sentence
    to the question — and answered `Project Atlas`. The claim holding the answer,
    `Project Atlas deploy_region eu-west-1`, was second on both lists. Nothing was missing
    except a reason to prefer it.

    The predicate schema is published with the dataset and handed to every system in
    `reset()`, so matching a question against predicate names uses an input every
    adapter has, not private knowledge of the questions. It is a lexical heuristic and it
    is stated as one: it resolves *which relation* is being asked about, and nothing
    about which entity, so it cannot rescue a question whose answer is not in the
    candidate list at all.
    """
    # Deduplicated first, order preserved. A ranked list can name the same slot twice —
    # memvara returns up to `max_per_slot` claims from one, and `naive` has an entry per
    # event — and a repeat is not a second reason to prefer it.
    ranked = list(dict.fromkeys(candidates))
    if not ranked:
        return None
    words = {_stem(w) for w in tokens(question)}
    best_rank, best_score = 0, -1
    for rank, (_, predicate) in enumerate(ranked):
        score = len(_predicate_words(predicate) & words)
        if score > best_score:
            best_rank, best_score = rank, score
    return ranked[best_rank]


def wants_a_date(ask: Ask) -> bool:
    """Does this question ask *when*, rather than *what*?

    `knowledge_time` holds both kinds. "On what date did the system learn X" wants a
    date and names the value it means in `about`. "What would the system have said on
    5 March" wants a value, sets `known_at`, and carries no `about` because it is not
    about one value — it is about whichever value was believed.

    This lives here rather than in each adapter because getting it wrong is silent: the
    date branch looks for a value named in `about`, finds `None`, and abstains. Three
    adapters made that mistake independently, which is three too many for a distinction
    the benchmark can state once.
    """
    return ask.category in ("change_time", "knowledge_time") and ask.about is not None


def default_usage() -> Usage:
    """An all-unmeasured `Usage`, for a system that counts nothing."""
    return Usage()


def sort_events(events: Sequence[MemoryEvent]) -> list[MemoryEvent]:
    """Delivery order: by `recorded_at`, ties broken by id so it is total.

    Ties are the common case, not the corner — a scenario often records several facts on
    the same day — and an order that depended on the JSONL's line order would make the
    benchmark sensitive to a file edit that changed nothing.
    """
    return sorted(events, key=lambda e: (e.recorded_at, e.id))
