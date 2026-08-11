"""Importing a mem0 store, including the history mem0 keeps and cannot query.

The migration problem is not "can memvara hold these strings" — it is that nobody accepts
"re-ingest your history by hand". So this importer is built around the one asset a mem0
deployment already owns and gets nothing from: `~/.mem0/history.db`, the mutation log
behind every `add()`, `update()` and `delete()` mem0 ever performed.

    history(id, memory_id, old_memory, new_memory, event, created_at, updated_at,
            is_deleted, actor_id, role)

Every ADD row carries the memory's full text, so the log is by itself a complete export —
no vector-store dump required, and `memories=` is there only to recover the entity ids
(`user_id`/`agent_id`/`run_id`) that mem0 keeps in the vector payload instead.

**Phase 1 is lossless and costs zero tokens.** Each memory becomes a note (see
`_notes`) written at its original timestamp on *both* axes, and the log is then replayed
in `created_at` order: an UPDATE retires the old value through the same slot and asserts
the new one with an `invalidated_by` pointer, and a DELETE closes both axes at the
instant mem0 stopped believing it.

That replay is the demo. mem0's own log is a transaction-time history it cannot answer
questions about — `history(memory_id)` returns rows, not a queryable past — and running
it through a bitemporal store turns it into `search(as_of=…)`, `history()` and `why()`
for free. Nothing was extracted, nothing was inferred, and no model was called.

*A DELETE row is replayed as a retirement, not an erasure* — the opposite of what the
`Memory.delete()` shim does, and deliberately. A live `delete()` is a caller stating an
intention now; a DELETE row is a historical event, and "we believed this from March to
July" is precisely the record being imported. Erasing it would destroy the history the
import exists to reconstruct. Use `Memvara.purge()` if a scope must genuinely go.

**Phase 2 is opt-in and costs tokens.** `extract=True` runs the configured model over the
imported notes and asserts real triples, with `sources` still pointing at the phase-1
episodes so `why()` on a structured claim resolves back to the mem0 row it came from.
Every text version is extracted, not just the current one, so a memory that mem0 updated
twice arrives as a proper superseded chain rather than a single present-tense fact.

**The receipt is the pitch.** `ImportReceipt` is a value, not a log line, and
`contested` on it names every slot left holding more than one live value — with the
undeclared predicates first, because those are the questions the store now has two live
answers to and no rule for choosing between them. mem0 cannot produce that list: its
conflicts are settled per-write by a model looking at a top-k, and nothing ever looks
again. Here it is an indexed lookup, and it arrives with the import.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from ..core import Memvara
from ..llm.base import LLM, Usage
from ..types import Claim, Derivation, Episode, Scope
from ._notes import NOTE_PREDICATE, build_note, ensure_note_predicate, note_subject
from ._notes import SUBJECT_PREFIX, write_note

#: Payload keys mem0 has used for the memory text across versions.
_TEXT_KEYS = ("memory", "data", "text")

#: Events the replay understands. Anything else (mem0's NONE, or a schema that grows a
#: new one) is counted as ignored rather than guessed at.
_EVENTS = ("ADD", "UPDATE", "DELETE")

#: Episodes per extraction call in phase 2. Turns share context and the per-request
#: overhead dominates at this size; unbounded batching would blow the context window on
#: a large store.
_EXTRACT_BATCH = 16


# --- reading mem0's log -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HistoryRow:
    """One row of `~/.mem0/history.db`, with its timestamp already made absolute."""

    id: str
    memory_id: str
    event: str
    created_at: datetime
    old_memory: str | None = None
    new_memory: str | None = None
    updated_at: datetime | None = None
    is_deleted: int = 0
    actor_id: str | None = None
    role: str | None = None


def _parse_ts(value: Any, *, where: str) -> datetime:
    """mem0 timestamps, as something two time axes can be compared on.

    Naive strings are read as UTC rather than as local time. Both readings are guesses,
    but only one of them changes answers when the importing machine's timezone differs
    from the machine that wrote the log — and an import that silently shifts a user's
    history by eight hours is not recoverable after the fact.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"{where}: cannot read timestamp {value!r}") from None
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        raise ValueError(f"{where}: cannot read timestamp {value!r}")
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _column(row: sqlite3.Row, name: str) -> Any:
    """A column that older mem0 schemas may not have. Missing is not empty, but here
    both mean "nothing to carry over", so they collapse safely."""
    return row[name] if name in row.keys() else None


def read_history_db(path: str | os.PathLike[str]) -> list[HistoryRow]:
    """Read mem0's mutation log, oldest first.

    Opened read-only: an import must not be able to damage the store it is migrating
    off, and a half-migrated deployment still needs mem0 to work.

    `~` is expanded. mem0's default location *is* `~/.mem0/history.db`, that is the path
    the README documents on the first line of the migration story, and `sqlite3.connect`
    does not expand it — so the documented call failed with "unable to open database
    file", which reads like a permissions problem rather than a path problem. The MCP
    server already expands `~` for `MEMVARA_DB` "because that is what people type in a
    JSON file"; the same is true of a migration one-liner.
    """
    con = sqlite3.connect(
        f"file:{os.path.expanduser(os.fspath(path))}?mode=ro", uri=True)
    try:
        con.row_factory = sqlite3.Row
        raw = con.execute("SELECT * FROM history").fetchall()
    finally:
        con.close()

    rows = [
        HistoryRow(
            id=str(r["id"]),
            memory_id=str(r["memory_id"]),
            event=str(r["event"] or "").strip().upper(),
            created_at=_parse_ts(r["created_at"], where=f"history row {r['id']}"),
            old_memory=_column(r, "old_memory"),
            new_memory=_column(r, "new_memory"),
            updated_at=_column(r, "updated_at"),
            is_deleted=int(_column(r, "is_deleted") or 0),
            actor_id=_column(r, "actor_id"),
            role=_column(r, "role"),
        )
        for r in raw
    ]
    # Replay order is causal, so it follows mem0's clock rather than its rowids: a log
    # merged from two processes can have ids that do not agree with time. The id breaks
    # ties so two runs of the same import produce the same store.
    rows.sort(key=lambda r: (r.created_at, r.id))
    return rows


# --- the receipt --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContestedSlot:
    """A (subject, predicate) slot left holding more than one live value.

    Read `declared` first. A declared multi-valued predicate holding three values is
    working as intended — a person may like three things. An **undeclared** one is the
    finding: nothing has said how many values it takes, the conservative default is
    "many", and so two answers to one question sit there live, both retrievable, with
    nothing to make them contradict. That is the state a mem0 store arrives in for every
    predicate its extractor invented, and it is fixed by one `PredicateSpec`.

    mem0 cannot produce this list at all: its conflicts are decided by asking a model
    about whatever the vector search returned, so one that missed the top-k or sat under
    the similarity threshold was never seen, and nothing afterwards ever looks again.
    Here the slot is an index and the question is a lookup.
    """

    subject: str
    predicate: str
    values: tuple[str, ...]
    #: Whether the predicate registry knows this predicate. False means its cardinality
    #: was never decided, and these values are competing rather than co-existing.
    declared: bool = False

    def __str__(self) -> str:
        origin = "declared" if self.declared else "UNDECLARED"
        return (f"<ContestedSlot {self.subject}.{self.predicate} ({origin}) "
                f"= {list(self.values)}>")

    __repr__ = __str__


@dataclass(slots=True)
class ImportReceipt:
    """What the import did, and what it found. Returned by `import_mem0`.

    `contested` is the headline and is a value, not a log line: it is the list a
    migration write-up is built from, and it is the first thing a mem0 deployment learns
    about its own store that mem0 could not tell it.
    """

    memories: int = 0        # distinct mem0 memories imported
    events: int = 0          # log rows replayed
    claims: int = 0          # note claims written
    updated: int = 0         # UPDATE rows that retired a previous value
    deleted: int = 0         # DELETE rows replayed as retirements
    duplicates: int = 0      # text already present in the slot; reinforced instead
    skipped: int = 0         # memories already in the store (see `skip_existing`)
    ignored: int = 0         # rows carrying no usable text, and events we do not model
    extracted: int = 0       # phase-2 structured claims
    llm_calls: int = 0       # phase-2 model calls; phase 1 is always 0
    #: Phase-2 tokens, when the configured backend reports them (`LLM.reports_usage`);
    #: 0 when it does not, which is indistinguishable from an import that made no calls.
    #: An import is the largest single spend most callers ever make here — it pays once
    #: for the whole history — and `llm_calls` cannot be costed, so this is the number a
    #: migration write-up needs.
    tokens_in: int = 0
    tokens_out: int = 0
    contested: list[ContestedSlot] = field(default_factory=list)

    def __str__(self) -> str:
        # The contested count is last and spelled out rather than abbreviated: it is the
        # one number here that says something about the store being migrated off rather
        # than about the migration.
        return (
            f"<ImportReceipt {self.memories} memories from {self.events} events: "
            f"+{self.claims} ~{self.updated} -{self.deleted} dup={self.duplicates} "
            f"skip={self.skipped} ignored={self.ignored} "
            f"extracted={self.extracted} llm={self.llm_calls}; "
            f"{len(self.contested)} slots hold more than one live value>"
        )

    __repr__ = __str__


# --- the import ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Event:
    """A log row or a synthesized one, resolved against the scope it belongs in."""

    memory_id: str
    event: str
    text: str
    ts: datetime
    scope: Scope
    meta: dict[str, Any]
    order: int


def _text_of(payload: Mapping[str, Any]) -> str:
    for key in _TEXT_KEYS:
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def _clean(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if v is not None}


def _payload_scope(payload: Mapping[str, Any], fallback: Scope) -> Scope:
    """mem0's flat entity triple as an memvara scope, defaulting to the import's."""
    return Scope(
        fallback.tenant,
        payload.get("user_id", fallback.user),
        payload.get("agent_id", fallback.agent),
        payload.get("run_id", fallback.session),
    )


def _scope_kw(scope: Scope) -> dict[str, Any]:
    return {"tenant": scope.tenant, "user": scope.user, "agent": scope.agent,
            "session": scope.session}


def _events(memories: Iterable[Mapping[str, Any]] | None,
            history: Sequence[HistoryRow], fallback: Scope) -> list[_Event]:
    """One ordered stream out of the two things mem0 keeps.

    The log is authoritative for *what happened*; the payloads are authoritative for
    *whose it is*, because entity ids live in the vector store and never reach the log.
    A payload whose creation the log does not record still becomes an ADD — a store
    whose history was pruned must not import as empty.
    """
    scopes: dict[str, Scope] = {}
    extra: dict[str, dict[str, Any]] = {}
    seeds: list[_Event] = []
    order = 0

    with_add = {r.memory_id for r in history if r.event == "ADD"}
    for payload in memories or ():
        memory_id = str(payload.get("id", ""))
        if not memory_id:
            continue
        scope = _payload_scope(payload, fallback)
        scopes[memory_id] = scope
        extra[memory_id] = dict(payload.get("metadata") or {})
        if memory_id in with_add:
            continue
        seeds.append(_Event(
            memory_id=memory_id, event="ADD", text=_text_of(payload),
            ts=_parse_ts(payload.get("created_at"), where=f"memory {memory_id}"),
            scope=scope,
            meta=_clean({"source": "mem0", "mem0_id": memory_id, "mem0_event": "ADD",
                         **extra[memory_id]}),
            order=order,
        ))
        order += 1

    out = list(seeds)
    for row in history:
        scope = scopes.get(row.memory_id, fallback)
        out.append(_Event(
            memory_id=row.memory_id, event=row.event,
            # A DELETE carries no new text; `old_memory` is what is going away.
            text=str(row.new_memory or ""), ts=row.created_at, scope=scope,
            meta=_clean({"source": "mem0", "mem0_id": row.memory_id,
                         "mem0_event": row.event, "mem0_history_id": row.id,
                         "actor_id": row.actor_id, "role": row.role,
                         **extra.get(row.memory_id, {})}),
            order=order,
        ))
        order += 1

    # Seeds sort before log rows sharing their instant, so a payload never lands after
    # the update that superseded it.
    out.sort(key=lambda e: (e.ts, e.order))
    return out


def import_mem0(
    mem: Memvara,
    *,
    history_db: str | os.PathLike[str] | None = None,
    memories: Iterable[Mapping[str, Any]] | None = None,
    tenant: str | None = None,
    user: str | None = None,
    agent: str | None = None,
    session: str | None = None,
    predicate: str = NOTE_PREDICATE,
    subject_prefix: str = SUBJECT_PREFIX,
    skip_existing: bool = True,
    extract: bool = False,
    llm: LLM | None = None,
    batch_size: int = _EXTRACT_BATCH,
) -> ImportReceipt:
    """Import a mem0 store into `mem`, replaying its history. Returns the receipt.

    `history_db` is the path to mem0's `~/.mem0/history.db`; `memories` is an optional
    iterable of vector-store payloads (`{"id", "memory", "created_at", "user_id", …}`),
    needed only to recover entity ids and metadata the log does not carry. At least one
    of the two is required.

    Phase 1 costs no model calls at all. `extract=True` adds phase 2, which runs `llm`
    (defaulting to the one `mem` was built with) over every imported text version and
    asserts structured triples that keep pointing at the phase-1 episodes.

    `skip_existing` leaves alone any memory whose slot already holds history, so an
    import that died halfway can simply be run again. Turn it off to import the same log
    into the same store twice on purpose — identical text reinforces rather than
    duplicating, but a memory with updates will replay its whole chain a second time.
    """
    if history_db is None and memories is None:
        raise ValueError(
            "nothing to import: pass history_db=<~/.mem0/history.db>, memories=<vector "
            "store payloads>, or both. The log alone is enough — every ADD row carries "
            "the memory's full text."
        )
    d = mem.default_scope
    fallback = Scope(tenant or d.tenant, user or d.user, agent or d.agent,
                     session or d.session)
    history = read_history_db(history_db) if history_db is not None else []
    ensure_note_predicate(mem, predicate, fallback.tenant)

    receipt = ImportReceipt()
    live: dict[str, Claim] = {}
    started: set[str] = set()
    skipping: set[str] = set()
    # Memories that actually put a claim in the store. Distinct from `started`, which
    # includes ids whose every row turned out to be unusable — counting those as
    # imported is how a receipt reports a successful import of nothing.
    imported: set[str] = set()
    sources: list[Episode] = []
    # Every scope the import wrote into, so the contested-slot sweep at the end looks
    # where the claims actually landed rather than only where it was told to.
    touched: dict[str, Scope] = {fallback.key(): fallback}

    for event in _events(memories, history, fallback):
        receipt.events += 1
        touched.setdefault(event.scope.key(), event.scope)
        if event.memory_id not in started:
            started.add(event.memory_id)
            if skip_existing and mem.history(
                    note_subject(event.memory_id, prefix=subject_prefix), predicate,
                    **_scope_kw(event.scope)):
                skipping.add(event.memory_id)
                receipt.skipped += 1
        if event.memory_id in skipping:
            continue

        if event.event == "DELETE":
            current = live.pop(event.memory_id, None)
            if current is None:
                receipt.ignored += 1
                continue
            # Transaction time closes here: mem0 stopped believing it at this instant,
            # and that is the fact being imported. `delete` closes both axes in one
            # transaction and is scope-checked, which costs nothing — the claim was
            # written by this import, into this event's scope.
            mem.delete(current.id, at=event.ts, **_scope_kw(event.scope))
            receipt.deleted += 1
            continue

        if event.event not in _EVENTS or not event.text.strip():
            # mem0's NONE rows, and any row whose text did not survive its export.
            receipt.ignored += 1
            continue

        claim, episode = build_note(
            memory_id=event.memory_id, text=event.text, scope=event.scope, ts=event.ts,
            predicate=predicate, subject_prefix=subject_prefix, meta=event.meta,
            role=str(event.meta.get("role") or "user"), extractor="mem0-import",
        )
        previous = live.get(event.memory_id)
        if previous is not None:
            receipt.updated += 1
        # Retirement and assertion in one transaction, the retirement first and carrying
        # the new id — so `why()` reports what replaced what, the reconciler does not
        # stamp today's clock over mem0's, and a crash mid-import cannot leave the slot
        # empty rather than updated.
        written = write_note(mem, claim, episode, retire=previous, at=event.ts)
        imported.add(event.memory_id)
        if written.added:
            receipt.claims += 1
            live[event.memory_id] = written.added[0]
        else:
            # The slot already holds this exact text: a re-import, or a mem0 UPDATE that
            # restored a previous value. Evidence, not a new fact.
            receipt.duplicates += 1
            live[event.memory_id] = written.reinforced[0]
        sources.append(episode)

    receipt.memories = len(imported)
    if extract:
        _extract(mem, sources, llm or mem.llm, batch_size, receipt)
    receipt.contested = _contested(mem, list(touched.values()),
                                   mem.registry.normalize(predicate))
    return receipt


# --- phase 2 ------------------------------------------------------------------


def _confidence(raw: Any) -> float:
    try:
        return min(1.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        # Model output is a trust boundary; an unreadable score is not a reason to drop
        # an otherwise well-formed claim, and 0.7 is what the write path uses.
        return 0.7


def _triple(mem: Memvara, item: Mapping[str, Any], chunk: Sequence[Episode],
            extractor: str) -> Claim | None:
    """One extracted dict as a claim, or None if it cannot be given provenance."""
    index = item.get("source_index")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(chunk):
        # No source means no `why()`, and an unprovenanced claim is the one thing this
        # library exists not to store.
        return None
    episode = chunk[index]
    predicate = mem.registry.normalize(str(item.get("predicate") or ""))
    obj = str(item.get("object") or "").strip()
    if not predicate or not obj:
        return None
    return Claim(
        subject=str(item.get("subject") or "").strip() or "user",
        predicate=predicate,
        object=obj,
        scope=episode.scope,
        polarity=-1 if item.get("polarity") == -1 else 1,
        memory_type=mem.registry.spec(predicate).memory_type,
        # Both axes inherit the mem0 row's timestamp, so the structured claims supersede
        # each other in mem0's order rather than all arriving "now" and racing.
        valid_from=episode.ts,
        recorded_at=episode.ts,
        confidence=_confidence(item.get("confidence", 0.7)),
        sources=[episode.id],
        derivation=Derivation.LLM_EXTRACT,
        extractor=extractor,
        meta=_clean({"source": "mem0", "mem0_id": episode.meta.get("mem0_id")}),
    )


def _extract(mem: Memvara, sources: Sequence[Episode], llm: LLM, batch_size: int,
             receipt: ImportReceipt) -> None:
    """Phase 2: pay tokens once to turn the imported notes into structured facts.

    Runs over every text version, not just the current one, so a memory mem0 updated
    twice arrives as a superseded chain — which is the whole reason to do this rather
    than leave the notes as they are.
    """
    vocabulary = mem.registry.prompt_vocabulary()
    extractor = getattr(llm, "name", "llm")
    # One accumulator for the whole import, on the same terms as the write path: only for
    # a backend that advertised it will fill one, and never sent to a backend that did not.
    usage = Usage() if getattr(llm, "reports_usage", False) else None
    for start in range(0, len(sources), batch_size):
        chunk = list(sources[start:start + batch_size])
        raw = llm.extract(chunk, vocabulary) if usage is None else llm.extract(
            chunk, vocabulary, usage=usage)
        receipt.llm_calls += 1
        for item in raw:
            claim = _triple(mem, item, chunk, extractor)
            if claim is None:
                continue
            written = mem.writer.assert_claim(claim)
            receipt.extracted += len(written.added)
    if usage is not None and usage.reported:
        receipt.tokens_in, receipt.tokens_out = usage.input_tokens, usage.output_tokens


# --- the pitch ----------------------------------------------------------------


def _contested(mem: Memvara, scopes: Sequence[Scope], note_predicate: str) -> list[ContestedSlot]:
    """Slots holding more than one live value, across every scope the import touched.

    Note slots are excluded: they hold one value each by construction, so counting them
    would report the import's own bookkeeping as a finding.
    """
    claims: dict[str, Claim] = {}
    for scope in scopes:
        for claim in mem.get_all(**_scope_kw(scope)):
            claims[claim.id] = claim

    slots: dict[str, list[Claim]] = {}
    for claim in claims.values():
        if claim.predicate != note_predicate:
            slots.setdefault(claim.fact_key, []).append(claim)

    found = []
    for group in slots.values():
        values = sorted({c.object for c in group})
        if len(values) > 1:
            found.append(ContestedSlot(
                group[0].subject, group[0].predicate, tuple(values),
                declared=mem.registry.known(group[0].predicate)))
    # Undeclared slots first: they are the ones that need a decision, and a long tail of
    # legitimately multi-valued `likes` should not bury them.
    found.sort(key=lambda s: (s.declared, s.subject, s.predicate))
    return found
