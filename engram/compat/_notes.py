"""One opaque memory string, stored as a claim without pretending it is a triple.

Shared by the mem0 shim and the mem0 importer, which hit the same wall from opposite
directions: mem0's unit of memory is a sentence ("Likes pizza") and engram's is a
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
`Engram.history(subject, predicate)` the per-memory timeline that mem0's
`history(memory_id)` returns, and it keeps the predicate registry at one entry instead
of one per imported memory — which matters, because learned predicates are capped.

Nothing here is reachable from the `Engram` facade: writing a claim with its own
`sources`, `meta` and backdated timestamps needs `Engram.writer.assert_claim`, and
storing the source turn without running extraction over it needs `Engram.store`. Both
are public objects (`WritePipeline` and `Store` are exported from `engram`), but the
facade has no equivalent — see the workstream report.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from ..core import Engram
from ..schema import Cardinality, PredicateSpec, Volatility
from ..types import Claim, Derivation, Episode, MemoryType, Scope, WriteReceipt

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


def ensure_note_predicate(mem: Engram, predicate: str, tenant: str) -> None:
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


def write_note(mem: Engram, claim: Claim, episode: Episode) -> WriteReceipt:
    """Persist the source turn and the claim derived from it, in one transaction.

    Separately committed, a crash between the two leaves a claim citing an episode that
    does not exist — a dangling `why()` in the one library whose pitch is that provenance
    always resolves.
    """
    with mem.store.batch():
        mem.store.add_episode(episode)
        return mem.writer.assert_claim(claim)


def supersede(mem: Engram, old: Claim, at: datetime, by: str | None) -> None:
    """Retire `old` as of `at`, recording what replaced it.

    `Engram.delete` does the same thing and is scope-checked, but it cannot record
    `invalidated_by` — and without that pointer `why()` on the new claim reports nothing
    superseded, which is exactly the history an import exists to reconstruct. Callers
    here own the claim they are retiring (they wrote it moments ago), so the scope check
    it forgoes has nothing to protect.
    """
    with mem.store.batch():
        mem.store.invalidate(old.id, at, by)
        # Both axes together: committed separately, an `as_of` query between the two
        # sees a claim that is retracted and still in force.
        mem.store.set_valid_to(old.id, at)
