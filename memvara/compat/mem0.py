"""mem0's method surface, mapped onto Memvara.

A drop-in for the calls a mem0 application actually makes, so an existing integration
runs against memvara without being rewritten. Written against mem0 2.x, compared with the
real package at 2.0.0 and 2.2.1 by the nightly framework tests. There `add()` and
`delete_all()` take the entity ids `user_id`, `agent_id` and `run_id` as keywords, while
`search()` and `get_all()` take them only inside a `filters=` dict and refuse a top-level
one with `ValueError`; `limit` became `top_k`; and `search` grew `threshold`, `rerank` and
`explain`. The shim takes every method and argument mem0's `Memory` takes, with mem0's
defaults, and refuses with `Mem0CompatError` each argument it cannot honour.

Two calls do **not** have an honest translation, and this module refuses each of them
loudly rather than guessing. That refusal is the useful part: a shim that silently means
something else is worse than no shim, because the difference shows up as data loss
months later.

``update()``
    A claim is immutable; that immutability is what `history()` and `as_of` are built
    out of. The analogue is asserting the new value, which retires the old one through
    the same slot — a different id, on purpose.

``Memory.from_config()``
    mem0 configs name Qdrant, Chroma and OpenAI providers. Memvara's equivalents are
    constructor arguments on `Memvara`, not entries in a provider registry, so there is
    nothing to translate them into.

Four arguments are refused too, when they are given a value; `UNSUPPORTED` gives each
one's reason. mem0's `expiration_date=` hides a memory until `show_expired=True`, where
memvara's closest thing, `expires_at`, erases it; so nothing here is ever hidden as
expired, and `show_expired=True` would ask for memories that cannot exist. `timestamp=`
is documented by mem0 itself as unsupported outside its hosted platform, and
`reference_date=` has no memvara reading that means the same thing.

Two behavioural differences are worth knowing before the first surprise:

* **A memory id is a version id.** Memvara never mutates a claim, so a supersession mints
  a new id and retires the old one. mem0 code that caches an id and re-`get()`s it after
  an update gets `None`; `history(old_id)` still walks the whole slot, which is the
  recovery path.
* **`delete()` retires by default, and mem0's erases.** A retired claim stops answering
  `search()`, `get()` and `get_all()`, but its text, its source turn and its embedding
  stay on disk and `history()` and `search(as_of=…)` still return it — so for a GDPR
  request the default is the worst outcome, a caller who believes the data is gone. It
  is still the default because it is memvara's semantics and the divergence should be
  noticed, not absorbed; the warning fires once and names the fix. `on_delete="erase"`
  matches mem0: it erases the memory outright, and every earlier version of it, each
  with its source turn. An earlier version is a claim of its own, because a changed
  value is ended rather than overwritten, so erasing only the current claim would leave
  the old text readable. `"retire"` keeps retirement and silences the warning once you
  have decided.
* **`add()` reports a supersession as two rows** — an ADD for the new value and a DELETE
  for the one it retired — because that is what happened. mem0 emits a single UPDATE.
"""

from __future__ import annotations

import uuid
import warnings
from typing import Any, Iterable, Mapping, Sequence

from ..core import Memvara
from ..types import (
    RESERVED_META,
    ENTITY_REKEY,
    LAST_OBSERVED,
    OBJECT_ENTITY,
    SALIENCE_BASE,
    SUBJECT_ENTITY,
    Claim,
    Episode,
    MemoryType,
    Result,
    Scope,
)
from ._notes import NOTE_PREDICATE, build_note, ensure_note_predicate, write_note


class Mem0CompatError(NotImplementedError):
    """A mem0 call with no honest translation onto memvara.

    `NotImplementedError` rather than a bespoke base so `except NotImplementedError`
    around a migration shim catches it, and its own type so a caller can tell "memvara
    will never do this" from "this Store has not implemented that yet".
    """


class Mem0DeletionWarning(UserWarning):
    """`delete()` retired a memory instead of erasing it.

    Its own category so a deployment that has read the message and decided can silence
    exactly this (`warnings.filterwarnings`, or `on_delete="retire"`) without silencing
    everything else the library says.
    """


#: mem0's entity ids, and the memvara scope field each one is. Memvara's hierarchy is
#: tenant > user > agent > session and visibility widens upward; mem0's triple is flat,
#: so this mapping is a narrowing, not a rename.
ENTITY_FILTERS = {"user_id": "user", "agent_id": "agent", "run_id": "session"}

#: Arguments mem0 2.x renamed. Worth naming in the error because the old spelling is
#: still all over the tutorials the caller learned from.
RENAMED_ARGS = {"limit": "top_k"}

_ON_DELETE = ("warn", "retire", "erase")

_RETIRED_NOT_ERASED = (
    "mem0's delete() erases a memory; this call retired it instead. The claim stops "
    "answering search(), get() and get_all(), and its text, its source episode and its "
    "embedding remain on disk — still returned by history() and by search(as_of=...).\n\n"
    "If this call is a GDPR/CCPA erasure, retirement does not satisfy it. Pass "
    "Memory(on_delete='erase') to erase the memory, every earlier version of it and "
    "their source turns outright, or 'retire' to keep this behaviour and silence this "
    "warning once you have decided."
)

_NO_UPDATE = (
    "memvara claims are immutable, and that is what history() and as_of are built out "
    "of: rewriting a claim's text in place would rewrite the past it is evidence for. "
    "The analogue of an update is asserting the new value — add() it, or call "
    "Memvara.remember(subject, predicate, new_value) — which retires the old value "
    "through the same slot and gives the new one its own id."
)

_NO_CONFIG = (
    "mem0 configs name providers (Qdrant, Chroma, OpenAI) that memvara has no registry "
    "for; its equivalents are constructor arguments. Build the Memvara yourself and wrap "
    "it:\n"
    "    from memvara import Memvara\n"
    "    from memvara.llm.anthropic import AnthropicLLM\n"
    "    Memory(Memvara('memory.db', llm=AnthropicLLM()))"
)


def _reject_legacy_kwargs(kwargs: Mapping[str, Any], method: str) -> None:
    """Refuse the keyword arguments a mem0 2.x method does not take, as mem0 does.

    `search()` and `get_all()` take the entity ids only inside `filters=`, and mem0 2.x
    refuses a top-level `user_id=` there with `ValueError`, so the shim raises the same
    type and a caller's `except` clause catches both (#359). Any other unknown argument
    is a `TypeError`, naming the renamed ones: an application that still passes `limit=`
    is running against a 1.x contract, and every other 1.x assumption it makes differs
    too.
    """
    if not kwargs:
        return
    entity = sorted(k for k in kwargs if k in ENTITY_FILTERS)
    if entity:
        shown = ", ".join(f"{k!r}: {kwargs[k]!r}" for k in entity)
        raise ValueError(
            f"Top-level entity parameters {', '.join(entity)} are not supported in "
            f"{method}(). Use filters={{{shown}}} instead."
        )
    parts = [f"{method}() got unexpected keyword argument(s): "
             + ", ".join(sorted(kwargs))]
    renamed = sorted(k for k in kwargs if k in RENAMED_ARGS)
    if renamed:
        parts.append("mem0 2.x renamed "
                     + ", ".join(f"{k}= to {RENAMED_ARGS[k]}=" for k in renamed) + ".")
    raise TypeError(" ".join(parts))


def _with_entities(filters: Mapping[str, Any] | None,
                   ids: Mapping[str, str | None]) -> Mapping[str, Any] | None:
    """`filters` with the entity ids `add()` and `delete_all()` take as keywords added.

    mem0 2.x's `add()` requires one of them and its `delete_all()` takes them, so a mem0
    call site passes them there (#359). `ids` maps each of `ENTITY_FILTERS`' names to
    what the caller passed. The shim's own `filters=` still works; an id given both ways
    must agree.
    """
    assert set(ids) == set(ENTITY_FILTERS), ids
    given = {key: value for key, value in ids.items() if value is not None}
    if not given:
        return filters
    merged = dict(filters or {})
    for key, value in given.items():
        if key in merged and merged[key] != value:
            raise ValueError(f"{key}={value!r} and filters={{{key!r}: {merged[key]!r}}} "
                             "name two different scopes; pass one")
        merged[key] = value
    return merged


#: The mem0 arguments this shim takes and refuses when given a value, each with why.
UNSUPPORTED = {
    "expiration_date": "mem0's expiration_date= hides a memory until show_expired=True, "
                       "where memvara's closest thing erases it outright "
                       "(Memvara.remember(expires_at=...)).",
    "show_expired": "nothing here is ever hidden as expired, because memvara erases an "
                    "expiring fact rather than hiding it, so there is nothing for "
                    "show_expired=True to show.",
    "timestamp": "mem0 documents timestamp= as unsupported outside its hosted platform.",
    "reference_date": "reference_date= has no memvara reading that means the same "
                      "thing; Memvara.search(valid_at=...) reads the world as of a date.",
}


def _refuse_unsupported(method: str, **given: Any) -> None:
    """Refuse, with `Mem0CompatError`, each mem0 argument this shim cannot honour that
    was given a value, with the reason `UNSUPPORTED` gives for it (#360)."""
    named = sorted(name for name, value in given.items()
                   if value is not None and value is not False)
    if named:
        raise Mem0CompatError(
            f"{method}() does not support {', '.join(f'{n}=' for n in named)}. "
            + " ".join(UNSUPPORTED[name] for name in named))


def _memory_type(raw: str | MemoryType) -> MemoryType:
    """Map mem0's memory-type strings onto memvara's enum.

    mem0 spells its one non-default kind `procedural_memory`; memvara's enum values are
    the bare words. Anything else is rejected rather than defaulted, because a wrong
    memory type sets the wrong decay half-life and the mistake is invisible for weeks.
    """
    text = raw.value if isinstance(raw, MemoryType) else str(raw).strip().lower()
    text = text[: -len("_memory")] if text.endswith("_memory") else text
    try:
        return MemoryType(text)
    except ValueError:
        raise ValueError(
            f"unknown memory_type {raw!r}; memvara has "
            + ", ".join(m.value for m in MemoryType)
        ) from None


class Memory:
    """mem0's `Memory`, backed by memvara.

    >>> from memvara import Memvara, NullLLM
    >>> m = Memory(Memvara(llm=NullLLM(), user="alice"))
    >>> [r["memory"] for r in m.add("I live in Berlin")["results"]]
    ['user lives in Berlin']
    >>> [r["memory"] for r in m.search("where do they live?")["results"]]
    ['user lives in Berlin']

    Wrap an `Memvara` you built (the usual case — that is where the store path, the
    embedder and the extraction model are chosen), or pass `Memvara` keywords straight
    through for a throwaway one.

    Those keywords include `api_key=`, so `Memory(api_key=...)` wraps a `RemoteMemvara`
    against a hosted deployment. `self.memvara` is annotated `Memvara` because that is
    what it holds in every other case, and the two classes are deliberately unrelated by
    inheritance — so read the annotation as naming the surface, not the class. Most of
    this shim runs unchanged over either, and two calls do not: `reset()` and
    `search(rerank=True)` reach for `Memvara.reset` and `Memvara.reader`, neither of which
    a hosted deployment exposes, and both raise `AttributeError` naming what is missing.
    That is `RemoteMemvara`'s own rule — what is absent is absent rather than a method
    that returns a plausible nothing — and it is stated here because the shim is where a
    caller meets it.
    """

    def __init__(self, memory: Memvara | None = None, *, on_delete: str = "warn",
                 **memvara_kwargs: Any) -> None:
        if memory is not None and memvara_kwargs:
            raise TypeError(
                "pass either a ready Memvara or the keywords to build one, not both: "
                f"{', '.join(sorted(memvara_kwargs))} would be silently ignored"
            )
        if on_delete not in _ON_DELETE:
            raise ValueError(
                f"on_delete={on_delete!r} is not one of {_ON_DELETE}; see "
                "memvara.compat.mem0.Mem0DeletionWarning for what each one does"
            )
        self.memvara = memory if memory is not None else Memvara(**memvara_kwargs)
        self.on_delete = on_delete
        # Once per instance, not per call: a deletion sweep would otherwise emit one
        # warning per memory and get filtered wholesale, taking the message with it.
        self._warned_delete = False

    # -- scope ----------------------------------------------------------------

    def _scope_kw(self, filters: Mapping[str, Any] | None) -> dict[str, Any]:
        """mem0 `filters` -> memvara scope keywords. Absent keys inherit the Memvara's."""
        out: dict[str, Any] = {}
        for key, value in (filters or {}).items():
            field = ENTITY_FILTERS.get(key)
            if field is None:
                raise ValueError(
                    f"unsupported filter {key!r}: memvara filters by scope "
                    f"(tenant > user > agent > session), not by arbitrary metadata. "
                    f"Supported keys are {', '.join(sorted(ENTITY_FILTERS))}."
                )
            out[field] = value
        return out

    def _scope(self, filters: Mapping[str, Any] | None) -> Scope:
        d = self.memvara.default_scope
        kw = self._scope_kw(filters)
        return Scope(d.tenant, kw.get("user", d.user), kw.get("agent", d.agent),
                     kw.get("session", d.session))

    # -- writing ---------------------------------------------------------------

    def add(self, messages: str | Mapping[str, Any] | Iterable[Any], *,
            user_id: str | None = None,
            agent_id: str | None = None,
            run_id: str | None = None,
            metadata: Mapping[str, Any] | None = None,
            timestamp: Any = None,
            expiration_date: Any = None,
            infer: bool = True,
            memory_type: str | MemoryType | None = None,
            prompt: str | None = None,
            filters: Mapping[str, Any] | None = None,
            **legacy: Any) -> dict[str, list[dict[str, Any]]]:
        """Ingest messages. Returns mem0's `{"results": [{"id", "memory", "event"}]}`.

        `infer=True` runs memvara's write path, which reaches a model only for turns the
        deterministic tiers could not handle — so the call count in the receipt is
        usually zero rather than mem0's one-per-turn. `infer=False` stores each message
        verbatim as a note, the same shape the importer uses.

        A supersession appears as an ADD and a DELETE rather than mem0's single UPDATE,
        because memvara wrote a new claim and retired the old one; the retired id is still
        readable through `history()`.

        `user_id`, `agent_id` and `run_id` scope the write as mem0's do; with none of them
        it inherits the wrapped `Memvara`'s scope. `filters=` is this shim's own spelling
        of the same thing, kept for code written against it.
        """
        _reject_legacy_kwargs(legacy, "add")
        _refuse_unsupported("add", timestamp=timestamp, expiration_date=expiration_date)
        filters = _with_entities(filters, {"user_id": user_id, "agent_id": agent_id,
                                           "run_id": run_id})
        if prompt is not None:
            raise Mem0CompatError(
                "prompt= overrides mem0's extraction prompt. Memvara's prompt belongs to "
                "the LLM backend: subclass or replace it (Memvara(llm=...)) rather than "
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
        receipt = self.memvara.add(self._to_episodes(messages, scope, metadata))
        rows = [{"id": c.id, "memory": c.text, "event": "ADD"} for c in receipt.added]
        # `closed`, not `ended`: mem0's vocabulary has one word for both readings, so a
        # claim the world moved past and a claim we got wrong both land as DELETE here.
        rows += [{"id": c.id, "memory": c.text, "event": "DELETE"}
                 for c in receipt.closed]
        # mem0 emits NONE for a turn that changed nothing. Memvara's equivalent is a
        # reinforcement: the fact was already believed and is now believed harder.
        rows += [{"id": c.id, "memory": c.text, "event": "NONE"}
                 for c in receipt.reinforced]
        return {"results": rows}

    @staticmethod
    def _to_episodes(messages: Any, scope: Scope,
                     metadata: Mapping[str, Any] | None) -> list[Episode]:
        """mem0's message shapes as episodes, with `metadata` attached to each.

        Built here rather than handed to `Memvara.add` as raw dicts because `metadata` is
        a mem0 argument with no memvara keyword: an `Episode` is the only place to put it.
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
        ensure_note_predicate(self.memvara, NOTE_PREDICATE, scope.tenant)
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
            write_note(self.memvara, claim, source)
            rows.append({"id": claim.id, "memory": claim.text, "event": "ADD"})
        return rows

    def update(self, memory_id: str, data: str | None = None, *,
               text: str | None = None, metadata: Mapping[str, Any] | None = None,
               expiration_date: Any = None) -> dict[str, str]:
        """Always raises. See `_NO_UPDATE` — a claim is immutable by construction.

        It takes every argument mem0's `update()` takes, so a call site that passes them
        meets this refusal rather than a `TypeError` (#360)."""
        raise Mem0CompatError(_NO_UPDATE)

    def delete(self, memory_id: str) -> dict[str, Any]:
        """Retire one memory, or erase it under `on_delete="erase"`.

        The default retires and warns once, because retirement is *not* what mem0's
        `delete()` does and silently doing the weaker thing is how a GDPR request gets
        quietly under-served. `on_delete="erase"` matches mem0: it erases every version of
        the memory `memory_id` names, as `_versions` defines them, each with its source
        turns. The reply lists the ids of the versions it erased under `erased`, oldest
        first, so its length is how many claims went. `memory_id` may name an earlier
        version, and the whole memory is erased then too, up to its current value.

        Raises `KeyError` for an id this scope cannot see, rather than reporting a
        success that deleted nothing.
        """
        named = self.memvara.get(memory_id)
        if named is None:
            raise KeyError(
                f"no memory {memory_id!r} in scope {self.memvara.default_scope.key()}"
            )
        if self.on_delete == "erase":
            # `sources=True` erases each version's source turn once no remaining claim
            # cites it. A note *is* its source turn, holding the same text and nothing
            # else, so leaving the turn behind would erase the memory and keep the
            # sentence. A turn that a claim outside this memory still cites is kept.
            erased = [version.id for version in self._versions(named)
                      if self.memvara.erase(version.id, sources=True)]
            return {"message": "Memory erased", "erased": erased}
        self.memvara.delete(memory_id)
        if self.on_delete == "warn" and not self._warned_delete:
            self._warned_delete = True
            warnings.warn(_RETIRED_NOT_ERASED, Mem0DeletionWarning, stacklevel=2)
        return {"message": "Memory retired (not erased); see Mem0DeletionWarning"}

    def _versions(self, named: Claim) -> list[Claim]:
        """Every version of the memory that `named` is one version of, oldest first.

        A memory here is what mem0 calls one memory: a value, and the values its updates
        replaced. When a value changes, memvara keeps the old value as a claim of its own
        and points that claim's `invalidated_by` at the claim that replaced it, and a
        retraction does the same with the claim that records it. So the versions are
        `named` itself, the claims that replaced it, forward to the current value, and the
        claims it replaced, directly or through an earlier version, back to the first
        one. They are looked for in `named`'s own slot and nowhere else.

        That is deliberately not every claim in the slot. A predicate that holds many
        values keeps each of them live in one slot as a memory with its own id: `likes`
        can hold "pizza" and "sushi" at once, and erasing one must not erase the other.
        For the same reason a claim linked to `named` only through a shared successor is
        not a version of it, such as another value that one retraction ended at the same
        time. A claim stored with no link to this chain is a memory of its own, which
        `add()` reported with its own ADD row: a restatement dated before the value on
        record, for example, is stored beside it rather than linked to it, and stays.

        A version can sit in another scope of the same user, because a write can end a
        value held in a broader scope it reads or in a narrower scope beneath it. A value
        written for the user ends the value a session holds for the same fact, which links
        the session's claim into the user-level chain. The chain is followed through such
        a claim, but `erase()` refuses a claim in a scope this `Memory` cannot read, so
        `delete()` leaves it as it is and does not list it. A version in a broader scope
        that this `Memory` reads, such as the user level for a `Memory` bound to a
        session, is erased.

        A local `Memvara` holds the store, and the slot is read from it whole. A hosted
        deployment gives this shim no store, and its `history()` reads the slot only at
        this `Memory`'s scope and the scopes beneath it. So there the walk also asks
        `get()` for a claim that replaced a version and `why()` for the claims a version
        replaced, because both of those read the broader scopes this `Memory` reads as
        well. One chain is still out of reach there: a version in a broader scope that the
        chain reaches only through a claim this `Memory` cannot read, because `why()`
        answers nothing about such a claim.
        """
        store = getattr(self.memvara, "store", None)
        if store is not None:
            slot = store.slot_history(named.scope.tenant, named.fact_key)
        else:
            timeline = self.memvara.history(named.subject, named.predicate)
            slot = [claim for claim in timeline if claim.fact_key == named.fact_key]
        stored = {claim.id: claim for claim in slot}
        stored[named.id] = named
        replaced: dict[str, list[Claim]] = {}
        for claim in stored.values():
            if claim.invalidated_by is not None:
                replaced.setdefault(claim.invalidated_by, []).append(claim)
        found = {named.id: named}
        # Forward, through the claims that replaced it, to the current value.
        successor = named.invalidated_by
        while successor is not None and successor not in found:
            later = stored.get(successor)
            if later is None and store is None:
                later = self.memvara.get(successor)
            if later is None or later.fact_key != named.fact_key:
                break
            found[successor] = later
            successor = later.invalidated_by
        # Back, through the claims it replaced, to the first value.
        pending = [named.id]
        while pending:
            current = pending.pop()
            earlier = list(replaced.get(current, ()))
            if store is None:
                provenance = self.memvara.why(current)
                if provenance is not None:
                    earlier += provenance.superseded
            for claim in earlier:
                if claim.id not in found:
                    found[claim.id] = claim
                    pending.append(claim.id)
        return sorted(found.values(), key=lambda c: (c.recorded_at, c.id))

    def delete_all(self, user_id: str | None = None, agent_id: str | None = None,
                   run_id: str | None = None, *, filters: Mapping[str, Any] | None = None,
                   **legacy: Any) -> dict[str, Any]:
        """Erase a scope for real: claims, episodes, embeddings and text index.

        This one *is* erasure — it is `Memvara.purge()` — which makes it the call a
        deletion request should route to, and the reason `delete()` can afford to be
        honest about retiring instead.

        `user_id`, `agent_id` and `run_id` name the scope as mem0's do (#359), and
        `filters=` is this shim's own spelling of the same thing.
        """
        _reject_legacy_kwargs(legacy, "delete_all")
        kw = self._scope_kw(_with_entities(filters, {"user_id": user_id,
                                                     "agent_id": agent_id,
                                                     "run_id": run_id}))
        if not kw:
            raise ValueError(
                "delete_all() with no filters would erase the entire tenant. Name the "
                "scope (filters={'user_id': ...}), or call reset() if the whole tenant "
                "is genuinely what you mean."
            )
        counts = self.memvara.purge(**kw)
        return {"message": "Memories erased", "counts": counts}

    def reset(self) -> dict[str, Any]:
        """Erase everything this `Memvara`'s own scope covers. Irreversible.

        Narrower than mem0's, which wipes the store: memvara scope arguments only ever
        narrow, so a `Memory` wrapping `Memvara(user="alice")` cannot widen back out to
        the tenant and resets alice. Wrap an unbound `Memvara` if `reset()` has to mean
        everything. The learned predicate schema survives either way — it is a
        vocabulary, not user data.
        """
        return {"message": "Memory store reset", "counts": self.memvara.reset()}

    # -- reading ---------------------------------------------------------------

    def search(self, query: str, *, top_k: int = 20,
               filters: Mapping[str, Any] | None = None,
               threshold: float | None = None, rerank: bool = False,
               explain: bool = False, reference_date: Any = None,
               show_expired: bool = False, rewrite: bool = False,
               **legacy: Any) -> dict[str, list[dict[str, Any]]]:
        """Hybrid retrieval, in mem0's response shape.

        `threshold` defaults to **no floor**, not to mem0's 0.1. That is deliberate:
        `Result.score` is normalized into [0, 1], but the value that separates a correct
        answer from the best wrong one drifts with corpus size — measured, the usable
        windows at 5 claims and at 1,000 do not intersect, so no constant is right at
        both ends and the failure is silent in the worse direction. Measure your own with
        `memvara.calibrate_min_score` and re-measure as the store grows.

        `explain=True` attaches memvara's retrieval explanation, which is per-leg (vector
        rank, BM25 rank, recency, salience) rather than mem0's prose.

        `rerank=True` is a **requirement**, not a switch: it asserts that a cross-encoder
        pass must have run, and it is satisfiable only when the wrapped `Memvara` was
        built with one (`Memvara(read_reranker=...)`, see `memvara.rerank`). Reranking is
        opt-in at construction because it is a model, and a keyword argument cannot
        conjure one — so the honest answers are "it ran" and a refusal that says how to
        make it run. `rerank=False`, mem0's default and what an unmodified call site
        passes, expresses no opinion and leaves whatever the instance is configured with
        exactly as it is.

        `rewrite=True` turns on memvara's query rewrite for this search: one call to the
        wrapped `Memvara`'s chat backend for other phrasings of the query and the dates it
        names (see `Memvara.search`). It is off by default here, unlike on
        `Memvara.search`, because mem0's `search()` makes no model call and a caller
        moving from mem0 is promised the same deterministic hybrid retrieval. It has no
        mem0 counterpart.

        `top_k` defaults to 20, as mem0's does (#361). `reference_date=` and
        `show_expired=True` are refused with `Mem0CompatError`; see the module docstring.
        """
        _reject_legacy_kwargs(legacy, "search")
        _refuse_unsupported("search", reference_date=reference_date,
                            show_expired=show_expired)
        if rerank and getattr(self.memvara.reader, "reranker", None) is None:
            raise Mem0CompatError(
                "rerank=True asks for a cross-encoder pass, and this Memvara has no "
                "reranker configured. Reranking is a model, so it is opt-in at "
                "construction: Memvara(read_reranker=CrossEncoderReranker()) after "
                "pip install 'memvara[rerank]', or CoverageReranker() for a lexical one "
                "that needs no download — see memvara.rerank. Without it the ranking "
                "still fuses BM25 with vector search and rescores by recency, confidence "
                "and salience — pass explain=True to see each leg's contribution."
            )
        results = self.memvara.search(
            query, k=top_k, min_score=0.0 if threshold is None else threshold,
            query_rewrite=rewrite, **self._scope_kw(filters))
        return {"results": [self._row(r.claim, result=r, explain=explain)
                            for r in results]}

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """One memory by id, or None — including for one that has been retired.

        Retired claims are excluded on purpose: mem0's `get()` returns nothing after a
        `delete()`, and a shim that kept answering would make the retirement look like it
        had not happened. `history()` still reaches them.
        """
        claim = self.memvara.get(memory_id)
        if claim is None or not claim.is_live():
            return None
        row = self._row(claim)
        # mem0's get() row carries `score`, empty because nothing was searched (#360).
        row["score"] = None
        return row

    def get_all(self, *, filters: Mapping[str, Any] | None = None, top_k: int = 20,
                show_expired: bool = False,
                **legacy: Any) -> dict[str, list[dict[str, Any]]]:
        """Up to `top_k` live memories in scope, newest first. `top_k` defaults to 20, as
        mem0's does (#361)."""
        _reject_legacy_kwargs(legacy, "get_all")
        _refuse_unsupported("get_all", show_expired=show_expired)
        claims = self.memvara.get_all(**self._scope_kw(filters))[:top_k]
        return {"results": [self._row(c) for c in claims]}

    def history(self, memory_id: str) -> list[dict[str, Any]]:
        """mem0's mutation log for one memory, synthesized from the slot's timeline.

        The two are different axes and ours is the larger one: mem0 records what happened
        to a *memory row*, memvara records every value a *slot* ever held, including the
        ones a supersession retired. Walking the timeline pairwise recovers mem0's rows —
        the first value is an ADD, each later one an UPDATE off its predecessor, and a
        final retirement with nothing after it is a DELETE.

        Unknown ids give `[]` rather than raising, matching mem0 and refusing to act as
        an existence oracle for another scope's ids.
        """
        anchor = self.memvara.get(memory_id)
        if anchor is None:
            return []
        timeline = self.memvara.history(anchor.subject, anchor.predicate)
        rows: list[dict[str, Any]] = []
        previous: Claim | None = None
        created = timeline[0].recorded_at if timeline else anchor.recorded_at
        for claim in timeline:
            rows.append({
                "id": claim.id,
                # Stable across the timeline, as mem0's memory_id is — memvara's per-value
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
    def _user_metadata(claim: Claim) -> dict[str, Any]:
        """Only what the caller put there.

        Memvara keeps its own bookkeeping in `Claim.meta` — storage strength, the last
        observation instant, resolved entity identities. That is internal state, not
        something the caller attached, and mem0 documents `metadata` as the caller's own
        dict. Echoing ours back would leak implementation detail through a compatibility
        surface and invite someone to depend on it.
        """
        return {k: v for k, v in claim.meta.items() if k not in RESERVED_META}

    @staticmethod
    def _row(claim: Claim, *, result: Result | None = None,
             explain: bool = False) -> dict[str, Any]:
        """One claim in mem0's memory shape, with memvara's extra structure alongside.

        The extra `memvara` key is additive: a mem0 consumer reads `memory` and ignores
        the rest, and anything porting *off* the shim can see what the triple actually
        was without a second call.
        """
        row: dict[str, Any] = {
            "id": claim.id,
            "memory": claim.text,
            "hash": claim.value_key,
            "metadata": Memory._user_metadata(claim),
            "created_at": claim.recorded_at.isoformat(),
            # Memvara never edits a claim, so a live one has never been updated; a retired
            # one was last touched when it was retired.
            "updated_at": (None if claim.invalidated_at is None
                           else claim.invalidated_at.isoformat()),
            "user_id": claim.scope.user,
            "agent_id": claim.scope.agent,
            "run_id": claim.scope.session,
            "memvara": {
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
    def from_config(cls, config_dict: Mapping[str, Any] | None = None,
                    **config: Any) -> "Memory":
        """Always raises. mem0 configs name providers memvara has no registry for.

        The argument is named `config_dict`, as mem0's is (#360), and any other spelling
        is taken too, so every call meets this refusal rather than a `TypeError`."""
        raise Mem0CompatError(_NO_CONFIG)

    # -- lifetime ----------------------------------------------------------------

    def close(self) -> None:
        """Close the wrapped `Memvara`, as mem0's `close()` releases its connections.

        The `Memvara` is closed whether this shim built it or was handed it, because the
        shim is backed by it: a caller that closes the shim is done with the memory."""
        self.memvara.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<mem0.Memory on_delete={self.on_delete} of {self.memvara!r}>"
