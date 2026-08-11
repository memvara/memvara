"""One opaque memory string, stored as a claim without pretending it is a triple.

Shared by the mem0 shim and the mem0 importer, which hit the same wall from opposite
directions: mem0's unit of memory is a sentence ("Likes pizza") and memvara's is a
(subject, predicate, object) triple. Inventing a predicate per sentence needs a model,
and a lossless import must cost zero tokens — so a *note* keeps the sentence whole and
moves the identity into the subject:

    subject    "mem0:9f2c…"   the mem0 memory id — one memory owns one slot
    predicate  "note"          single-valued, so new text retires the old through the
                               contradiction engine rather than accumulating
    object     "Likes pizza"   the sentence, verbatim
    text       "Likes pizza"   set explicitly, so the embedded and BM25-indexed string
                               is the sentence and not "mem0:9f2c… note Likes pizza"

Putting the id in the *subject* rather than in the predicate is what makes
`Memvara.history(subject, predicate)` the per-memory timeline that mem0's
`history(memory_id)` returns, and it keeps the predicate registry at one entry instead
of one per imported memory — which matters, because learned predicates are capped.

**A synthetic subject must be opaque, not readable.** The subject is folded through
`entity_key` before it keys a slot, and that fold strips punctuation — so
`langgraph:a/b#c` and `langgraph:a#b/c` both become `langgraph a b c` and share one
slot, superseding each other's data. `mem0:` and `note:` get away with putting an id
straight in only because those ids are uuids and hex survives the fold. Any adapter
minting a subject from structured parts (a namespace tuple, a path, anything with
separators) has to hash the address rather than spell it out. The LangGraph adapter
does; this note exists so the next one does too.

`write_note` goes below the `Memvara` facade on purpose. `Memvara.remember(sources=…)` and
`Memvara.supersede` now cover everything it does *except* that they also embed each source
turn — and a note's turn is a byte-identical copy of the claim's own text, so the facade
would store two identical vectors per note and charge an import twice for the second one.
`WritePipeline` and `Store` are both public, and the write is still one transaction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from ..core import Memvara
from ..schema import Cardinality, PredicateSpec, Volatility
from ..types import (
    Claim, Closure, Derivation, Episode, MemoryType, Scope, WriteReceipt, close_out,
)

#: Predicate every note lands on. Deliberately generic: it is the *subject* that
#: identifies the memory, so one predicate serves the whole corpus.
NOTE_PREDICATE = "note"

#: Prefix on the synthetic subject, so a note slot is recognisable in `get_all()` output
#: and cannot collide with a real subject a caller writes ("user", "acme_corp").
SUBJECT_PREFIX = "mem0:"


def note_subject(memory_id: str, *, prefix: str = SUBJECT_PREFIX) -> str:
    """The slot-owning subject for one mem0 memory id.

    >>> note_subject("9f2c")
    'mem0:9f2c'
    """
    return f"{prefix}{memory_id}"


def ensure_note_predicate(mem: Memvara, predicate: str, tenant: str) -> None:
    """Declare the note predicate single-valued, once, and persist that.

    ONE rather than MANY is the whole point: a mem0 UPDATE replaces a memory's text, and
    only a single-valued predicate turns "assert the new text" into "retire the old one"
    without the caller having to remember to do it.

    An existing predicate of the same name is left exactly as the caller declared it —
    silently rewriting someone's schema to suit an importer would be worse than the
    duplicate it prevents. Both callers therefore also retire the previous value
    explicitly, so correctness never depends on this having run.
    """
    if mem.registry.known(predicate):
        return
    spec = PredicateSpec(
        name=mem.registry.normalize(predicate),
        cardinality=Cardinality.ONE,
        volatility=Volatility.SLOW,
        memory_type=MemoryType.SEMANTIC,
        # Declared, not learned: this is a schema decision made by code that was written
        # down, and learned predicates are capped (`DEFAULT_LEARNED_CAP`). An import of
        # 200 memories must not exhaust a budget meant for a model's guesses.
        learned=False,
    )
    mem.registry.register(spec)
    # Durable, so the next process opens the store knowing the note slot is single-valued
    # rather than treating it as multi-valued and silently accumulating.
    mem.store.put_spec(spec, tenant)


def build_note(
    *,
    memory_id: str,
    text: str,
    scope: Scope,
    ts: datetime,
    predicate: str = NOTE_PREDICATE,
    subject_prefix: str = SUBJECT_PREFIX,
    meta: Mapping[str, Any] | None = None,
    role: str = "user",
    memory_type: MemoryType = MemoryType.SEMANTIC,
    extractor: str = "mem0",
) -> tuple[Claim, Episode]:
    """A note claim and the episode holding the text it came from. Nothing is written.

    Construction is separate from writing because a supersession has to know the new
    claim's id *before* the old one is retired — `invalidated_by` is what makes
    `why()` report what replaced what.

    Both times are set to `ts`: valid time because the memory was true from then, and
    transaction time because mem0 believed it from then. Backdating transaction time is
    the point of the import — it is what `search(as_of=…)` reads.
    """
    episode = Episode(content=text, scope=scope, role=role, ts=ts, meta=dict(meta or {}))
    claim = Claim(
        subject=note_subject(memory_id, prefix=subject_prefix),
        predicate=predicate,
        object=text,
        # Explicit, so retrieval sees the sentence rather than the slot address. A Claim
        # renders its own text only when it is empty, and the reconciler only re-renders
        # text that matches what it would have generated — so this survives.
        text=text,
        scope=scope,
        memory_type=memory_type,
        valid_from=ts,
        recorded_at=ts,
        sources=[episode.id],
        derivation=Derivation.USER,
        extractor=extractor,
        meta=dict(meta or {}),
    )
    return claim, episode


def write_note(mem: Memvara, claim: Claim, episode: Episode, *,
               retire: Claim | None = None, at: datetime | None = None,
               close: Closure = "ended") -> WriteReceipt:
    """Persist the source turn, close out the value it replaces, assert the claim — atomic.

    Separately committed, a crash between the turn and the claim leaves a claim citing an
    episode that does not exist — a dangling `why()` in the one library whose pitch is
    that provenance always resolves — and a crash between the retirement and the
    assertion leaves the slot empty, which is worse: an import that drops a memory rather
    than updating it.

    `retire` is the value this note replaces and `at` the instant it was replaced, read
    on whichever clock `close` names — `"ended"` by default, because a mem0 UPDATE row
    says the memory's text changed, not that the previous text had been a mistake. The
    closure records `claim.id` as what replaced it, which is the pointer `Memvara.delete`
    cannot write and the whole reason an import reconstructs history rather than just
    replaying its final state. It goes **before** the assertion, so the reconciler cannot
    get there first and stamp the wall clock over `at` — which would silently turn a
    backdated import into a pile of things that all changed today.

    This is `Memvara._write_claim` minus one step, and the missing step is deliberate:
    the facade also *embeds* each source turn, and a note's episode is a byte-identical
    copy of the claim's own text. Going through the facade would put two identical
    vectors in the index for every note, return the same sentence twice under
    `include_episodes=True`, and double the encode bill of an import for no recall
    anyone can name. The turn is still stored and still BM25-indexed by `add_episode`.
    """
    with mem.store.batch():
        mem.store.add_episode(episode)
        if retire is not None:
            # `retire` and `at` arrive as independent optionals, so `retire=old` with no
            # `at` is a legal call — and passing that `None` straight through is not a
            # no-op, it is corruption: it would leave a NULL instant on whichever axis
            # `close` names, which either reads as *not retired* beside an
            # `invalidated_by` saying otherwise, or **reopens** an interval that was
            # already closed. The row would then assert "superseded by X" and "still
            # live" at once. Today the reconciler happens to rescue the note path a
            # moment later, because the note predicate is single-valued and
            # `assert_claim` supersedes through the same slot — correct by accident, from
            # two statements away.
            when = at if at is not None else claim.recorded_at
            close_out(retire, when, claim.id, close)
            mem.store.put_claim(retire)
        return mem.writer.assert_claim(claim, close=close)
