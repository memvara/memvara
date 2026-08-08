"""CrewAI: the unified-memory `StorageBackend`, implemented on engram.

Written against **crewai 1.x**, whose memory system was rewritten: the old
`crewai.memory.storage.interface.Storage` (`save`/`search`/`reset`) no longer exists.
The current contract is `crewai.memory.storage.backend.StorageBackend`, a
`runtime_checkable` `Protocol` — which is a gift, because it means nothing here has to
subclass anything and the class body needs no CrewAI import at all. Only the two
*return* types (`MemoryRecord`, `ScopeInfo`) do, and those are resolved on first use.

    from crewai.memory import Memory
    from engram.integrations.crewai import EngramStorage

    storage = EngramStorage(mem, user="alice")
    memory = Memory(storage=storage, embedder=storage.embedder)

**`embedder=storage.embedder` is not optional, and this is the whole story of the
adapter.** `StorageBackend.search` is handed a `query_embedding` and never the query
text — CrewAI embeds the query with its own model before the backend is reached. Engram
does not retrieve from a vector: it fuses BM25 with vector search and rescores by a
per-predicate half-life, all of it keyed on the query *string*, and a vector from
`text-embedding-3-large` is not even a point in engram's space. Two dishonest ways out
were available and both were rejected — store CrewAI's vectors and degrade to cosine
top-k (engram becomes a numpy matmul with extra steps), or re-embed text this interface
never provides (impossible). So the backend supplies the embedder, remembers what it
embedded, and recovers the query text from the vector it produced. Given a vector it did
not produce, it **raises** and names the wiring, rather than searching in a space where
the answer is noise.

**What this adapter cannot give you, stated up front: deterministic contradiction
resolution.** CrewAI's unit of memory is an opaque sentence with an id of its own, so
"Alice lives in Berlin" and "Alice moved to Lisbon" arrive as two records, land on two
slots, and both stay live — measured against the real package, not argued. Engram's
keyed lookup fires on `(subject, predicate)`, and there is no subject or predicate in a
`MemoryRecord`. The LangChain and LlamaIndex adapters do not have this problem because
they hand engram *raw turns* and extraction runs; this one is handed CrewAI's already-
extracted output. Getting it back means extracting triples from the sentences, which
costs a model call per record — the same phase-2 trade `engram.compat.mem0_import`
makes, and left to the caller as one extra line (`mem.add(record.content)`) rather than
charged silently on every save.

Three more things differ and are handled rather than hidden:

* **Scope trees point in opposite directions.** CrewAI's `/company/team/user` is an
  aggregation tree: a search at `/company` should see everything beneath it. Engram's
  `tenant > user > agent > session` is a *visibility* tree: a session sees the user's
  durable memory, never the other way round. They are not the same structure, so the
  adapter does not pretend — the engram scope is fixed at construction (and is what
  isolates two `EngramStorage`s from each other), while CrewAI's path is stored per
  record and filtered as data. `oversample` exists because that filtering happens after
  ranking.
* **`delete()` retires by default.** CrewAI's own dedup path calls `delete(record_ids=…)`
  when a record is superseded, and retirement is the *right* answer there: the value
  stops being returned and `history()` still has it. It is the wrong answer to "delete
  my data", so the first one warns and names `on_delete="erase"`. `reset()` is the one
  deletion that maps exactly — CrewAI means "wipe", engram's `purge()` is a wipe.
* **`update()` maps better than CrewAI's own contract.** "Replace the record with the
  same ID" becomes a supersession on a single-valued slot: the new text is asserted, the
  old value is retired with `invalidated_by` pointing at its replacement, and
  `history(subject, "note")` walks every version. CrewAI overwrites; nothing is kept.
"""

from __future__ import annotations

import asyncio
import warnings
from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from ..compat import NOTE_PREDICATE, note_subject
from ..compat._notes import ensure_note_predicate
from ..types import Claim, as_utc
from ._common import IntegrationError, bind, require, scope_kw

_PKG = "crewai.memory.types"
_NEEDS = "crewai>=1.0"

#: Prefix on the synthetic subject that owns one CrewAI record's slot. Distinct from the
#: mem0 importer's `mem0:`, so the two can share one store and one `note` predicate
#: without a record and an imported memory ever landing on the same slot.
SUBJECT_PREFIX = "crewai:"

#: The single `Claim.meta` key this adapter owns. One nested blob rather than six flat
#: keys, because `MemoryRecord.metadata` is the caller's own dict and any flat scheme is
#: one unlucky key name away from an adapter field being overwritten by user data.
CREWAI_META = "crewai"

_ON_DELETE = ("warn", "retire", "erase")

_NO_QUERY_TEXT = (
    "CrewAI's StorageBackend.search() is handed an embedding and never the query text, "
    "and engram does not retrieve from a vector — it fuses BM25 with vector search and "
    "rescores by a per-predicate half-life, all of it keyed on the query string. A "
    "vector from another model is not even a point in engram's space.\n\n"
    "So this backend embeds through engram's own embedder and remembers what it "
    "embedded, which is what makes the query text recoverable here. Wire it up:\n"
    "    storage = EngramStorage(mem, user='alice')\n"
    "    Memory(storage=storage, embedder=storage.embedder)\n\n"
    "The vector just received has {got} dimensions and engram's embedder produces "
    "{want}. If those differ, CrewAI is still using its own embedder. If they match, "
    "the query cache holds the last {cache} texts and this one was evicted — raise "
    "EngramStorage(query_cache=...)."
)

_NO_METADATA_FILTER = (
    "engram has no metadata index, so metadata_filter= would have to be applied after "
    "ranking — which silently returns fewer results than asked for, or none, with no "
    "way for the caller to tell that from 'nothing matched'. Filter by scope_prefix= or "
    "categories= (both are indexed here as record fields), or read the claim's "
    "meta['crewai']['metadata'] yourself through Engram.search()."
)

_RETIRED_NOT_ERASED = (
    "CrewAI's delete() removes a record; this call retired it instead. The record stops "
    "answering search(), get_record() and list_records(), and its text, its source turn "
    "and its embedding remain on disk — still reachable through Engram.history() and "
    "Engram.search(as_of=...).\n\n"
    "That is the right default for CrewAI's own consolidation path, which deletes a "
    "record because something superseded it and is better off keeping the trail. It is "
    "the wrong answer to a data-deletion request: pass EngramStorage(on_delete='erase') "
    "to erase the record and its source turn outright, or 'retire' to keep this "
    "behaviour and silence this warning once you have decided."
)


class CrewAICompatError(IntegrationError):
    """A CrewAI storage call with no honest translation onto engram."""


class CrewAIDeletionWarning(UserWarning):
    """`delete()` retired a record instead of erasing it.

    Its own category so a deployment that has read the message and decided can silence
    exactly this without silencing everything else the library says.
    """


def _under(path: str, prefix: str | None) -> bool:
    """Whether a CrewAI scope path lies at or beneath `prefix`.

    Segment-wise, not string-wise: `/acme` contains `/acme/eng` and does **not** contain
    `/acmecorp`, which a bare `startswith` would happily match into the wrong tenant's
    memories.
    """
    if not prefix or not prefix.strip("/"):
        return True
    here = "/" + path.strip("/")
    there = "/" + prefix.strip("/")
    return here == there or here.startswith(there + "/")


def _normalize(path: str) -> str:
    """A CrewAI scope path in one spelling, so `/a`, `a` and `/a/` are one scope."""
    stripped = (path or "").strip("/")
    return f"/{stripped}" if stripped else "/"


class EngramStorage:
    """Engram behind CrewAI's `StorageBackend` protocol.

    Structural, not nominal: `StorageBackend` is a `Protocol`, so this class satisfies
    it by having the methods and never imports it. Construct one per engram scope — that
    binding is what keeps two crews' memories apart, and it cannot be widened by anything
    CrewAI passes in.

    `types=` injects the namespace providing `MemoryRecord` and `ScopeInfo`, the only two
    CrewAI classes this adapter needs; left unset they are imported from
    `crewai.memory.types` the first time a record has to be *returned*. It is the same
    seam as `OpenAILLM(client=…)` and it is what lets the suite exercise the whole
    surface with CrewAI not installed.
    """

    def __init__(self, memory: Any, *, tenant: str | None = None, user: str | None = None,
                 agent: str | None = None, session: str | None = None,
                 on_delete: str = "warn", oversample: int = 4,
                 query_cache: int = 1024, types: Any = None) -> None:
        if on_delete not in _ON_DELETE:
            raise ValueError(
                f"on_delete={on_delete!r} is not one of {_ON_DELETE}; see "
                "engram.integrations.crewai.CrewAIDeletionWarning for what each one does"
            )
        self.memory, self.scope = bind(memory, tenant=tenant, user=user, agent=agent,
                                       session=session)
        self.on_delete = on_delete
        #: How many ranked results to pull per requested one when `scope_prefix` or
        #: `categories` will thin them afterwards. Post-ranking filters can under-fill a
        #: `limit`; this is the knob, and the fact that it exists is the honest statement
        #: that the filtering is not part of the index.
        self.oversample = max(1, oversample)
        self.query_cache = max(1, query_cache)
        # See `embedder`. Insertion-ordered so eviction is oldest-first.
        self._queries: OrderedDict[tuple[float, ...], str] = OrderedDict()
        self._types = types
        # Once per instance, not per record: a deletion sweep would otherwise emit one
        # warning per record and get filtered wholesale, taking the message with it.
        self._warned_delete = False
        # Declares the note slot single-valued and persists that, which is what turns
        # `update()` into a supersession instead of a second live value.
        ensure_note_predicate(self.memory, NOTE_PREDICATE, self.scope.tenant)

    # -- the embedder seam ---------------------------------------------------

    @property
    def embedder(self) -> Any:
        """The callable CrewAI must be constructed with: `Memory(embedder=…)`.

        Two jobs, and the second one is why this exists at all. It encodes with
        *engram's* embedder, so every vector CrewAI holds is a point in the space
        engram's index actually uses — and it remembers the text behind each vector, so
        `search()` can recover the query CrewAI declined to pass. See `_NO_QUERY_TEXT`.
        """
        return self._embed

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode with engram's embedder, and remember what each vector was.

        Returns a **list of lists**, not the `ndarray` the `Embedder` protocol produces.
        CrewAI's `embed_text` guards its result with `if not result:`, and that is a
        `ValueError: the truth value of an array with more than one element is
        ambiguous` on an ndarray — found by running against the real package, and
        invisible to any test that fakes the embedder's consumer.
        """
        out: list[list[float]] = []
        for text, vector in zip(texts, self.memory.embedder.encode(list(texts))):
            values = vector.tolist()
            out.append(values)
            key = tuple(values)
            self._queries[key] = text
            self._queries.move_to_end(key)
        while len(self._queries) > self.query_cache:
            self._queries.popitem(last=False)
        return out

    def _query_for(self, embedding: Sequence[float]) -> str:
        text = self._queries.get(tuple(float(x) for x in embedding))
        if text is None:
            raise CrewAICompatError(_NO_QUERY_TEXT.format(
                got=len(embedding), want=self.memory.embedder.dim,
                cache=self.query_cache))
        return text

    # -- CrewAI's types, resolved on first use -------------------------------

    def _crewai(self) -> Any:
        if self._types is None:
            record, scope_info = require(_PKG, "MemoryRecord", "ScopeInfo",
                                         extra="crewai", needs=_NEEDS)
            self._types = SimpleNamespace(MemoryRecord=record, ScopeInfo=scope_info)
        return self._types

    # -- record <-> claim ----------------------------------------------------

    @property
    def _kw(self) -> dict[str, Any]:
        return scope_kw(self.scope)

    @staticmethod
    def _is_record(claim: Claim) -> bool:
        return (claim.predicate == NOTE_PREDICATE
                and claim.subject.startswith(SUBJECT_PREFIX))

    @staticmethod
    def _blob(record: Any) -> dict[str, Any]:
        """The CrewAI fields engram's data model has no column for.

        `created_at` is stored exactly as given, naive or aware. CrewAI builds it with
        `datetime.utcnow()` and then does naive arithmetic on it in
        `compute_composite_score`; handing back an aware datetime we had helpfully
        normalized would raise `TypeError` inside CrewAI's scorer, on their line, for
        our reason.
        """
        return {
            "record_id": record.id,
            "scope": _normalize(record.scope),
            "categories": list(record.categories),
            "importance": float(record.importance),
            "source": record.source,
            "private": bool(record.private),
            "created_at": record.created_at.isoformat(),
            "metadata": dict(record.metadata),
        }

    def _to_record(self, claim: Claim) -> Any:
        blob = claim.meta.get(CREWAI_META) or {}
        stamp = blob.get("created_at")
        return self._crewai().MemoryRecord(
            id=blob.get("record_id") or claim.subject[len(SUBJECT_PREFIX):],
            content=claim.object,
            scope=blob.get("scope", "/"),
            categories=list(blob.get("categories", ())),
            metadata=dict(blob.get("metadata", {})),
            importance=float(blob.get("importance", 0.5)),
            created_at=(datetime.fromisoformat(stamp) if isinstance(stamp, str)
                        else claim.recorded_at),
            # Engram does not track reads, so the honest answer is the last write. A
            # fabricated "just now" would make CrewAI's recency scoring a function of
            # how often something was looked at, which is not something we measured.
            last_accessed=claim.recorded_at,
            source=blob.get("source"),
            private=bool(blob.get("private", False)),
        )

    def _claim_for(self, record_id: str) -> Claim | None:
        """The live claim on one record's slot, or `None`.

        Through `history()` rather than a scan: the slot is an indexed lookup on
        `(subject, predicate)`, which is the same mechanism that makes contradiction
        resolution free.
        """
        subject = note_subject(record_id, prefix=SUBJECT_PREFIX)
        for claim in self.memory.history(subject, NOTE_PREDICATE, **self._kw):
            if claim.invalidated_at is None:
                return claim
        return None

    def _live_claims(self) -> list[Claim]:
        return [c for c in self.memory.get_all(**self._kw) if self._is_record(c)]

    def _write(self, record: Any, *, at: datetime | None) -> None:
        self.memory.remember(
            note_subject(record.id, prefix=SUBJECT_PREFIX),
            NOTE_PREDICATE,
            record.content,
            # Explicit, so what gets embedded and BM25-indexed is the sentence rather
            # than "crewai:9f2c… note <sentence>".
            text=record.content,
            valid_from=at, recorded_at=at,
            extractor="crewai-storage",
            **{CREWAI_META: self._blob(record)},
            **self._kw,
        )

    # -- StorageBackend: writing ---------------------------------------------

    def save(self, records: Sequence[Any]) -> None:
        """Persist records. A repeated record reinforces rather than duplicating.

        Backdated to `record.created_at` on both engram axes, so `search(as_of=…)` over
        a CrewAI store answers what the crew knew at an instant rather than reporting
        that everything was learned at import time.
        """
        for record in records:
            self._write(record, at=as_utc(record.created_at))

    def update(self, record: Any) -> None:
        """Replace a record's text — as a supersession, keeping the version it replaced.

        CrewAI overwrites the row. Here the new text is asserted onto the same
        single-valued slot, so the reconciler retires the old value and stamps
        `invalidated_by` with the new claim's id; `Engram.history(subject, "note")`
        then walks every version the record ever had, and `search(as_of=…)` still
        returns the old one.

        Recorded at *now* rather than at `record.created_at`: CrewAI carries the
        original creation instant on an updated record, and using it as transaction time
        would date our belief in the new text to before the old text was retired. The
        original instant is kept as data instead, so the record CrewAI reads back is
        unchanged.
        """
        self._write(record, at=None)

    # -- StorageBackend: reading ---------------------------------------------

    def search(self, query_embedding: Sequence[float], scope_prefix: str | None = None,
               categories: Sequence[str] | None = None,
               metadata_filter: Mapping[str, Any] | None = None, limit: int = 10,
               min_score: float = 0.0) -> list[tuple[Any, float]]:
        """Hybrid retrieval, from the query text behind `query_embedding`.

        Not a cosine top-k: the vector is used only to recover what was asked, and the
        answer comes from BM25 fused with vector search and rescored by recency,
        confidence and salience. `min_score` is engram's, on the same normalized [0, 1]
        scale CrewAI expects.
        """
        if metadata_filter:
            raise CrewAICompatError(_NO_METADATA_FILTER)
        wanted = set(categories or ())
        results = self.memory.search(
            self._query_for(query_embedding), k=max(limit, 1) * self.oversample,
            min_score=min_score, **self._kw)
        out: list[tuple[Any, float]] = []
        for result in results:
            if len(out) >= limit:
                break
            if not self._is_record(result.claim):
                continue
            record = self._to_record(result.claim)
            if not _under(record.scope, scope_prefix):
                continue
            if wanted and not wanted.intersection(record.categories):
                continue
            out.append((record, result.score))
        return out

    def get_record(self, record_id: str) -> Any | None:
        """One live record by id, or `None`. A retired one is gone from here on purpose.

        `Engram.history()` still reaches every version, which is the recovery path a
        CrewAI application does not know it has.
        """
        claim = self._claim_for(record_id)
        return None if claim is None else self._to_record(claim)

    def list_records(self, scope_prefix: str | None = None, limit: int = 200,
                     offset: int = 0) -> list[Any]:
        """Live records in scope, newest first."""
        records = [r for r in self._records() if _under(r.scope, scope_prefix)]
        return records[offset:offset + limit]

    def _records(self) -> list[Any]:
        # `get_all` is already newest-first with a deterministic tie-break, which is the
        # order CrewAI documents for this call.
        return [self._to_record(c) for c in self._live_claims()]

    def count(self, scope_prefix: str | None = None) -> int:
        return sum(1 for r in self._records() if _under(r.scope, scope_prefix))

    def list_scopes(self, parent: str = "/") -> list[str]:
        """Immediate child scope paths under `parent`, derived from what is stored."""
        base = _normalize(parent)
        head = "" if base == "/" else base
        children = set()
        for record in self._records():
            path = _normalize(record.scope)
            if path == base or not _under(path, base):
                continue
            rest = path[len(head) + 1:]
            children.add(f"{head}/{rest.split('/')[0]}")
        return sorted(children)

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._records():
            if not _under(record.scope, scope_prefix):
                continue
            for category in record.categories:
                counts[category] = counts.get(category, 0) + 1
        return counts

    def get_scope_info(self, scope: str) -> Any:
        """Counts, categories and date range for one scope, subscopes included."""
        path = _normalize(scope)
        records = [r for r in self._records() if _under(r.scope, path)]
        stamps = sorted(r.created_at for r in records)
        return self._crewai().ScopeInfo(
            path=path,
            record_count=len(records),
            categories=sorted({c for r in records for c in r.categories}),
            oldest_record=stamps[0] if stamps else None,
            newest_record=stamps[-1] if stamps else None,
            child_scopes=self.list_scopes(path),
        )

    # -- StorageBackend: deleting --------------------------------------------

    def delete(self, scope_prefix: str | None = None,
               categories: Sequence[str] | None = None,
               record_ids: Sequence[str] | None = None,
               older_than: datetime | None = None,
               metadata_filter: Mapping[str, Any] | None = None) -> int:
        """Remove matching records. Retires by default — see `_RETIRED_NOT_ERASED`.

        Every given criterion must match (CrewAI's own callers pass exactly one). No
        criteria at all deletes everything in scope, which is what CrewAI's signature
        means and is why `reset()` exists as the separate, unambiguous call.
        """
        if metadata_filter:
            raise CrewAICompatError(_NO_METADATA_FILTER)
        victims = list(self._matching(scope_prefix, categories, record_ids, older_than))
        for claim in victims:
            self._remove(claim)
        if victims and self.on_delete == "warn" and not self._warned_delete:
            self._warned_delete = True
            warnings.warn(_RETIRED_NOT_ERASED, CrewAIDeletionWarning, stacklevel=2)
        return len(victims)

    def _matching(self, scope_prefix: str | None, categories: Sequence[str] | None,
                  record_ids: Sequence[str] | None,
                  older_than: datetime | None) -> Iterable[Claim]:
        wanted = set(categories or ())
        ids = set(record_ids or ())
        for claim in self._live_claims():
            record = self._to_record(claim)
            if ids and record.id not in ids:
                continue
            if not _under(record.scope, scope_prefix):
                continue
            if wanted and not wanted.intersection(record.categories):
                continue
            if older_than is not None and not record.created_at < older_than:
                continue
            yield claim

    def _remove(self, claim: Claim) -> None:
        if self.on_delete == "erase":
            # `sources=True` is right here and would be wrong for an extracted fact: a
            # record *is* its source turn, holding the same text and nothing else, so
            # leaving the episode behind would erase the memory and keep the sentence.
            self.memory.erase(claim.id, sources=True, **self._kw)
        else:
            self.memory.delete(claim.id, **self._kw)

    def reset(self, scope_prefix: str | None = None) -> None:
        """Wipe. The one deletion that maps exactly, so it does not warn.

        With no `scope_prefix` this is `Engram.purge()` on the bound scope — claims,
        source turns, embeddings and the text index, gone, and no `history()` left
        behind. CrewAI's `reset` means the same thing, so there is nothing to disclose.

        A `scope_prefix` narrows to CrewAI's own scope tree, which engram cannot purge
        by (see the module docstring), so those records are erased one at a time —
        the same erasure, reached the long way.
        """
        if not scope_prefix or not scope_prefix.strip("/"):
            self.memory.purge(**self._kw)
            return
        for claim in list(self._matching(scope_prefix, None, None, None)):
            self.memory.erase(claim.id, sources=True, **self._kw)

    # -- StorageBackend: the async half --------------------------------------
    #
    # `asyncio.to_thread`, not an async rewrite. Engram is synchronous and SQLite has no
    # async driver worth the name (see `engram.aio`); what these buy is an event loop
    # that keeps serving while an encode and a write happen, which is the whole ask.

    async def asave(self, records: Sequence[Any]) -> None:
        await asyncio.to_thread(self.save, records)

    async def asearch(self, query_embedding: Sequence[float],
                      scope_prefix: str | None = None,
                      categories: Sequence[str] | None = None,
                      metadata_filter: Mapping[str, Any] | None = None, limit: int = 10,
                      min_score: float = 0.0) -> list[tuple[Any, float]]:
        return await asyncio.to_thread(
            self.search, query_embedding, scope_prefix, categories, metadata_filter,
            limit, min_score)

    async def adelete(self, scope_prefix: str | None = None,
                      categories: Sequence[str] | None = None,
                      record_ids: Sequence[str] | None = None,
                      older_than: datetime | None = None,
                      metadata_filter: Mapping[str, Any] | None = None) -> int:
        return await asyncio.to_thread(
            self.delete, scope_prefix, categories, record_ids, older_than,
            metadata_filter)

    def __repr__(self) -> str:
        return (f"<EngramStorage {self.scope.key()} on_delete={self.on_delete} "
                f"of {self.memory!r}>")


__all__ = ["EngramStorage", "CrewAICompatError", "CrewAIDeletionWarning",
           "SUBJECT_PREFIX", "CREWAI_META"]
