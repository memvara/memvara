"""mem0's method surface, mapped onto Engram.

A drop-in for the calls a mem0 application actually makes, so an existing integration
runs against engram without being rewritten. Written against mem0 2.x, where entity ids
moved into a `filters=` dict (a top-level `user_id=` is rejected), `limit` became
`top_k`, and `search` grew `threshold`, `rerank` and `explain`.

Three calls do **not** have an honest translation, and this module refuses each of them
loudly rather than guessing. That refusal is the useful part: a shim that silently means
something else is worse than no shim, because the difference shows up as data loss
months later.

``delete()``
    mem0's erases. Engram's `delete()` *retires*: the claim stops answering `search()`,
    `get()` and `get_all()`, and its text, its source episode and its embedding stay on
    disk, still returned by `history()` and by `search(as_of=…)`. For a GDPR deletion
    that is the worst possible outcome — the caller believes the data is gone. Engram has
    no per-claim erasure at all (only `purge()`, which erases a whole scope), so this
    shim retires and says so once, and `on_delete="erase"` turns the gap into an
    exception at the call site instead of a quiet under-delete.

``update()``
    A claim is immutable; that immutability is what `history()` and `as_of` are built
    out of. The analogue is asserting the new value, which retires the old one through
    the same slot — a different id, on purpose.

``Memory.from_config()``
    mem0 configs name Qdrant, Chroma and OpenAI providers. Engram's equivalents are
    constructor arguments on `Engram`, not entries in a provider registry, so there is
    nothing to translate them into.

Two behavioural differences are worth knowing before the first surprise:

* **A memory id is a version id.** Engram never mutates a claim, so a supersession mints
  a new id and retires the old one. mem0 code that caches an id and re-`get()`s it after
  an update gets `None`; `history(old_id)` still walks the whole slot, which is the
  recovery path.
* **`add()` reports a supersession as two rows** — an ADD for the new value and a DELETE
  for the one it retired — because that is what happened. mem0 emits a single UPDATE.
"""

from __future__ import annotations

import uuid
import warnings
from typing import Any, Iterable, Mapping, Sequence

from ..core import Engram
from ..types import Claim, Episode, MemoryType, Result, Scope
from ._notes import NOTE_PREDICATE, build_note, ensure_note_predicate, write_note


class Mem0CompatError(NotImplementedError):
    """A mem0 call with no honest translation onto engram.

    `NotImplementedError` rather than a bespoke base so `except NotImplementedError`
    around a migration shim catches it, and its own type so a caller can tell "engram
    will never do this" from "this Store has not implemented that yet".
    """


class Mem0DeletionWarning(UserWarning):
    """`delete()` retired a memory instead of erasing it.

    Its own category so a deployment that has read the message and decided can silence
    exactly this (`warnings.filterwarnings`, or `on_delete="retire"`) without silencing
    everything else the library says.
    """


#: mem0's entity ids, and the engram scope field each one is. Engram's hierarchy is
#: tenant > user > agent > session and visibility widens upward; mem0's triple is flat,
#: so this mapping is a narrowing, not a rename.
ENTITY_FILTERS = {"user_id": "user", "agent_id": "agent", "run_id": "session"}

#: Arguments mem0 2.x renamed. Worth naming in the error because the old spelling is
#: still all over the tutorials the caller learned from.
RENAMED_ARGS = {"limit": "top_k"}

_ON_DELETE = ("warn", "retire", "erase")

_NO_ERASURE = (
    "mem0's delete() erases a memory; engram's retires it. The claim stops answering "
    "search(), get() and get_all(), and its text, its source episode and its embedding "
    "remain on disk — still returned by history() and by search(as_of=...).\n\n"
    "If this call is a GDPR/CCPA erasure, retirement does not satisfy it. Engram "
    "exposes no per-claim erasure; the calls that really erase are scope-wide:\n"
    "    Memory.delete_all(filters={'user_id': 'alice'})   # -> Engram.purge()\n"
    "    Memory.reset()                                     # -> the whole tenant\n\n"
    "Memory(on_delete=...) picks what this method does: 'warn' (default) retires and "
    "says so once, 'retire' retires silently once you have decided, 'erase' raises here "
    "rather than letting an erasure quietly under-delete."
)

_NO_UPDATE = (
    "engram claims are immutable, and that is what history() and as_of are built out "
    "of: rewriting a claim's text in place would rewrite the past it is evidence for. "
    "The analogue of an update is asserting the new value — add() it, or call "
    "Engram.remember(subject, predicate, new_value) — which retires the old value "
    "through the same slot and gives the new one its own id."
)

_NO_CONFIG = (
    "mem0 configs name providers (Qdrant, Chroma, OpenAI) that engram has no registry "
    "for; its equivalents are constructor arguments. Build the Engram yourself and wrap "
    "it:\n"
    "    from engram import Engram\n"
    "    from engram.llm.anthropic import AnthropicLLM\n"
    "    Memory(Engram('memory.db', llm=AnthropicLLM()))"
)


def _reject_entity_kwargs(kwargs: Mapping[str, Any], method: str) -> None:
    """Reject `user_id=`/`agent_id=`/`run_id=` the way mem0 2.x does.

    Accepting them quietly would be the friendlier-looking choice and the wrong one: an
    application that still passes them is running against a 1.x contract, and every
    *other* 1.x assumption it makes (`limit=`, a default `threshold`) differs too. Fail
    on the first one, naming the fix.
    """
    if not kwargs:
        return
    parts = [f"{method}() got unexpected keyword argument(s): "
             + ", ".join(sorted(kwargs))]
    entity = sorted(k for k in kwargs if k in ENTITY_FILTERS)
    if entity:
        shown = ", ".join(f"{k!r}: {kwargs[k]!r}" for k in entity)
        parts.append(
            f"mem0 2.x moved entity ids into filters=; pass filters={{{shown}}} instead."
        )
    renamed = sorted(k for k in kwargs if k in RENAMED_ARGS)
    if renamed:
        parts.append("mem0 2.x renamed "
                     + ", ".join(f"{k}= to {RENAMED_ARGS[k]}=" for k in renamed) + ".")
    raise TypeError(" ".join(parts))


def _memory_type(raw: str | MemoryType) -> MemoryType:
    """Map mem0's memory-type strings onto engram's enum.

    mem0 spells its one non-default kind `procedural_memory`; engram's enum values are
    the bare words. Anything else is rejected rather than defaulted, because a wrong
    memory type sets the wrong decay half-life and the mistake is invisible for weeks.
    """
    text = raw.value if isinstance(raw, MemoryType) else str(raw).strip().lower()
    text = text[: -len("_memory")] if text.endswith("_memory") else text
    try:
        return MemoryType(text)
    except ValueError:
        raise ValueError(
            f"unknown memory_type {raw!r}; engram has "
            + ", ".join(m.value for m in MemoryType)
        ) from None


class Memory:
    """mem0's `Memory`, backed by engram.

    >>> from engram import Engram, NullLLM
    >>> m = Memory(Engram(llm=NullLLM(), user="alice"))
    >>> [r["memory"] for r in m.add("I live in Berlin")["results"]]
    ['user lives in Berlin']
    >>> [r["memory"] for r in m.search("where do they live?")["results"]]
    ['user lives in Berlin']

    Wrap an `Engram` you built (the usual case — that is where the store path, the
    embedder and the extraction model are chosen), or pass `Engram` keywords straight
    through for a throwaway one.
    """

    def __init__(self, memory: Engram | None = None, *, on_delete: str = "warn",
                 **engram_kwargs: Any) -> None:
        if memory is not None and engram_kwargs:
            raise TypeError(
                "pass either a ready Engram or the keywords to build one, not both: "
                f"{', '.join(sorted(engram_kwargs))} would be silently ignored"
            )
        if on_delete not in _ON_DELETE:
            raise ValueError(
                f"on_delete={on_delete!r} is not one of {_ON_DELETE}; see "
                "engram.compat.mem0.Mem0DeletionWarning for what each one does"
            )
        self.engram = memory if memory is not None else Engram(**engram_kwargs)
        self.on_delete = on_delete
        # Once per instance, not per call: a deletion sweep would otherwise emit one
        # warning per memory and get filtered wholesale, taking the message with it.
        self._warned_delete = False

    # -- scope ----------------------------------------------------------------

    def _scope_kw(self, filters: Mapping[str, Any] | None) -> dict[str, Any]:
        """mem0 `filters` -> engram scope keywords. Absent keys inherit the Engram's."""
        out: dict[str, Any] = {}
        for key, value in (filters or {}).items():
            field = ENTITY_FILTERS.get(key)
            if field is None:
                raise ValueError(
                    f"unsupported filter {key!r}: engram filters by scope "
                    f"(tenant > user > agent > session), not by arbitrary metadata. "
                    f"Supported keys are {', '.join(sorted(ENTITY_FILTERS))}."
                )
            out[field] = value
        return out

    def _scope(self, filters: Mapping[str, Any] | None) -> Scope:
        d = self.engram.default_scope
        kw = self._scope_kw(filters)
        return Scope(d.tenant, kw.get("user", d.user), kw.get("agent", d.agent),
                     kw.get("session", d.session))

    # -- writing ---------------------------------------------------------------

    def add(self, messages: str | Mapping[str, Any] | Iterable[Any], *,
            filters: Mapping[str, Any] | None = None,
            metadata: Mapping[str, Any] | None = None,
            infer: bool = True,
            memory_type: str | MemoryType | None = None,
            prompt: str | None = None,
            **legacy: Any) -> dict[str, list[dict[str, Any]]]:
        """Ingest messages. Returns mem0's `{"results": [{"id", "memory", "event"}]}`.

        `infer=True` runs engram's write path, which reaches a model only for turns the
        deterministic tiers could not handle — so the call count in the receipt is
        usually zero rather than mem0's one-per-turn. `infer=False` stores each message
        verbatim as a note, the same shape the importer uses.

        A supersession appears as an ADD and a DELETE rather than mem0's single UPDATE,
        because engram wrote a new claim and retired the old one; the retired id is still
        readable through `history()`.
        """
        _reject_entity_kwargs(legacy, "add")
        if prompt is not None:
            raise Mem0CompatError(
                "prompt= overrides mem0's extraction prompt. Engram's prompt belongs to "
                "the LLM backend: subclass or replace it (Engram(llm=...)) rather than "
                "threading a prompt through every write."
            )
        scope = self._scope(filters)
        if not infer:
            return {"results": self._add_verbatim(messages, scope, metadata, memory_type)}
        if memory_type is not None:
            raise Mem0CompatError(
                "memory_type= with infer=True would override the predicate registry, "
                "which owns memory type and therefore decay half-life. Two writes of "
                "one predicate would then decay differently depending on the call site. "
                "Declare it once instead — PredicateSpec(..., memory_type=...) — or use "
                "infer=False, where the note is yours to type."
            )
        receipt = self.engram.add(self._to_episodes(messages, scope, metadata))
        rows = [{"id": c.id, "memory": c.text, "event": "ADD"} for c in receipt.added]
        rows += [{"id": c.id, "memory": c.text, "event": "DELETE"}
                 for c in receipt.invalidated]
        # mem0 emits NONE for a turn that changed nothing. Engram's equivalent is a
        # reinforcement: the fact was already believed and is now believed harder.
        rows += [{"id": c.id, "memory": c.text, "event": "NONE"}
                 for c in receipt.reinforced]
        return {"results": rows}

    @staticmethod
    def _to_episodes(messages: Any, scope: Scope,
                     metadata: Mapping[str, Any] | None) -> list[Episode]:
        """mem0's message shapes as episodes, with `metadata` attached to each.

        Built here rather than handed to `Engram.add` as raw dicts because `metadata` is
        a mem0 argument with no engram keyword: an `Episode` is the only place to put it.
        """
        items: Sequence[Any]
        items = [messages] if isinstance(messages, (str, Mapping)) else list(messages)
        out = []
        for item in items:
            if isinstance(item, Mapping):
                content, role = str(item.get("content", "")), str(item.get("role", "user"))
            else:
                content, role = str(item), "user"
            out.append(Episode(content=content, scope=scope, role=role,
                               meta=dict(metadata or {})))
        return out

    def _add_verbatim(self, messages: Any, scope: Scope,
                      metadata: Mapping[str, Any] | None,
                      memory_type: str | MemoryType | None) -> list[dict[str, Any]]:
        kind = MemoryType.SEMANTIC if memory_type is None else _memory_type(memory_type)
        ensure_note_predicate(self.engram, NOTE_PREDICATE, scope.tenant)
        rows = []
        # `_to_episodes` is reused here only to normalize mem0's three message shapes;
        # the episode actually stored is the one `build_note` mints, so the claim and its
        # source are written together and `why()` cannot dangle.
        for turn in self._to_episodes(messages, scope, metadata):
            claim, source = build_note(
                # No mem0 id to inherit, so the slot gets one of its own — which is what
                # makes `history()` on a verbatim note the timeline of that note alone.
                memory_id=uuid.uuid4().hex,
                text=turn.content, scope=scope, ts=turn.ts,
                subject_prefix="note:", meta=turn.meta, role=turn.role,
                memory_type=kind, extractor="mem0-compat",
            )
            write_note(self.engram, claim, source)
            rows.append({"id": claim.id, "memory": claim.text, "event": "ADD"})
        return rows

    def update(self, memory_id: str, data: str | None = None, *,
               text: str | None = None) -> dict[str, str]:
        """Always raises. See `_NO_UPDATE` — a claim is immutable by construction."""
        raise Mem0CompatError(_NO_UPDATE)

    def delete(self, memory_id: str) -> dict[str, str]:
        """Retire one memory. **Does not erase it** — see the module docstring.

        Raises `KeyError` for an id this scope cannot see, rather than reporting a
        success that deleted nothing.
        """
        if self.on_delete == "erase":
            raise Mem0CompatError(_NO_ERASURE)
        if self.engram.get(memory_id) is None:
            raise KeyError(
                f"no memory {memory_id!r} in scope {self.engram.default_scope.key()}"
            )
        self.engram.delete(memory_id)
        if self.on_delete == "warn" and not self._warned_delete:
            self._warned_delete = True
            warnings.warn(_NO_ERASURE, Mem0DeletionWarning, stacklevel=2)
        return {"message": "Memory retired (not erased); see Mem0DeletionWarning"}

    def delete_all(self, *, filters: Mapping[str, Any] | None = None,
                   **legacy: Any) -> dict[str, Any]:
        """Erase a scope for real: claims, episodes, embeddings and text index.

        This one *is* erasure — it is `Engram.purge()` — which makes it the call a
        deletion request should route to, and the reason `delete()` can afford to be
        honest about retiring instead.
        """
        _reject_entity_kwargs(legacy, "delete_all")
        kw = self._scope_kw(filters)
        if not kw:
            raise ValueError(
                "delete_all() with no filters would erase the entire tenant. Name the "
                "scope (filters={'user_id': ...}), or call reset() if the whole tenant "
                "is genuinely what you mean."
            )
        counts = self.engram.purge(**kw)
        return {"message": "Memories erased", "counts": counts}

    def reset(self) -> dict[str, Any]:
        """Erase everything this `Engram`'s own scope covers. Irreversible.

        Narrower than mem0's, which wipes the store: engram scope arguments only ever
        narrow, so a `Memory` wrapping `Engram(user="alice")` cannot widen back out to
        the tenant and resets alice. Wrap an unbound `Engram` if `reset()` has to mean
        everything. The learned predicate schema survives either way — it is a
        vocabulary, not user data.
        """
        return {"message": "Memory store reset", "counts": self.engram.reset()}

    # -- reading ---------------------------------------------------------------

    def search(self, query: str, *, filters: Mapping[str, Any] | None = None,
               top_k: int = 10, threshold: float | None = None, rerank: bool = False,
               explain: bool = False, **legacy: Any) -> dict[str, list[dict[str, Any]]]:
        """Hybrid retrieval, in mem0's response shape.

        `threshold` defaults to **no floor**, not to mem0's 0.1. That is deliberate:
        `Result.score` is normalized into [0, 1], but the value that separates a correct
        answer from the best wrong one drifts with corpus size — measured, the usable
        windows at 5 claims and at 1,000 do not intersect, so no constant is right at
        both ends and the failure is silent in the worse direction. Measure your own with
        `engram.calibrate_min_score` and re-measure as the store grows.

        `explain=True` attaches engram's retrieval explanation, which is per-leg (vector
        rank, BM25 rank, recency, salience) rather than mem0's prose.
        """
        _reject_entity_kwargs(legacy, "search")
        if rerank:
            raise Mem0CompatError(
                "rerank=True asks for a cross-encoder pass; engram has no reranker "
                "model. Its ranking already fuses BM25 with vector search and rescores "
                "by recency, confidence and salience — pass explain=True to see each "
                "leg's contribution."
            )
        results = self.engram.search(
            query, k=top_k, min_score=0.0 if threshold is None else threshold,
            **self._scope_kw(filters))
        return {"results": [self._row(r.claim, result=r, explain=explain)
                            for r in results]}

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """One memory by id, or None — including for one that has been retired.

        Retired claims are excluded on purpose: mem0's `get()` returns nothing after a
        `delete()`, and a shim that kept answering would make the retirement look like it
        had not happened. `history()` still reaches them.
        """
        claim = self.engram.get(memory_id)
        if claim is None or not claim.is_live():
            return None
        return self._row(claim)

    def get_all(self, *, filters: Mapping[str, Any] | None = None, top_k: int = 100,
                **legacy: Any) -> dict[str, list[dict[str, Any]]]:
        """Every live memory in scope, newest first."""
        _reject_entity_kwargs(legacy, "get_all")
        claims = self.engram.get_all(**self._scope_kw(filters))[:top_k]
        return {"results": [self._row(c) for c in claims]}

    def history(self, memory_id: str) -> list[dict[str, Any]]:
        """mem0's mutation log for one memory, synthesized from the slot's timeline.

        The two are different axes and ours is the larger one: mem0 records what happened
        to a *memory row*, engram records every value a *slot* ever held, including the
        ones a supersession retired. Walking the timeline pairwise recovers mem0's rows —
        the first value is an ADD, each later one an UPDATE off its predecessor, and a
        final retirement with nothing after it is a DELETE.

        Unknown ids give `[]` rather than raising, matching mem0 and refusing to act as
        an existence oracle for another scope's ids.
        """
        anchor = self.engram.get(memory_id)
        if anchor is None:
            return []
        timeline = self.engram.history(anchor.subject, anchor.predicate)
        rows: list[dict[str, Any]] = []
        previous: Claim | None = None
        created = timeline[0].recorded_at if timeline else anchor.recorded_at
        for claim in timeline:
            rows.append({
                "id": claim.id,
                # Stable across the timeline, as mem0's memory_id is — engram's per-value
                # ids are the `id` column above.
                "memory_id": memory_id,
                "old_memory": None if previous is None else previous.text,
                "new_memory": claim.text,
                "event": "ADD" if previous is None else "UPDATE",
                "created_at": created.isoformat(),
                "updated_at": None if previous is None else claim.recorded_at.isoformat(),
                "is_deleted": 0,
                "actor_id": claim.meta.get("actor_id"),
                "role": claim.meta.get("role"),
            })
            previous = claim
        if previous is not None and previous.invalidated_at is not None:
            # Retired with nothing taking its place: that is a deletion, not an update,
            # and mem0 records it as its own row.
            rows.append({
                "id": previous.id, "memory_id": memory_id,
                "old_memory": previous.text, "new_memory": None, "event": "DELETE",
                "created_at": created.isoformat(),
                "updated_at": previous.invalidated_at.isoformat(),
                "is_deleted": 1,
                "actor_id": previous.meta.get("actor_id"),
                "role": previous.meta.get("role"),
            })
        return rows

    # -- rendering --------------------------------------------------------------

    @staticmethod
    def _row(claim: Claim, *, result: Result | None = None,
             explain: bool = False) -> dict[str, Any]:
        """One claim in mem0's memory shape, with engram's extra structure alongside.

        The extra `engram` key is additive: a mem0 consumer reads `memory` and ignores
        the rest, and anything porting *off* the shim can see what the triple actually
        was without a second call.
        """
        row: dict[str, Any] = {
            "id": claim.id,
            "memory": claim.text,
            "hash": claim.value_key,
            "metadata": dict(claim.meta),
            "created_at": claim.recorded_at.isoformat(),
            # Engram never edits a claim, so a live one has never been updated; a retired
            # one was last touched when it was retired.
            "updated_at": (None if claim.invalidated_at is None
                           else claim.invalidated_at.isoformat()),
            "user_id": claim.scope.user,
            "agent_id": claim.scope.agent,
            "run_id": claim.scope.session,
            "engram": {
                "subject": claim.subject,
                "predicate": claim.predicate,
                "object": claim.object,
                "memory_type": claim.memory_type.value,
                "confidence": claim.confidence,
                "salience": claim.salience,
                "valid_from": claim.valid_from.isoformat(),
                "valid_to": None if claim.valid_to is None else claim.valid_to.isoformat(),
            },
        }
        if result is not None:
            row["score"] = result.score
            if explain:
                row["explanation"] = result.explain.summary()
        return row

    # -- construction -----------------------------------------------------------

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Memory":
        """Always raises. mem0 configs name providers engram has no registry for."""
        raise Mem0CompatError(_NO_CONFIG)

    def __repr__(self) -> str:
        return f"<mem0.Memory on_delete={self.on_delete} of {self.engram!r}>"
