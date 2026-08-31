"""LangGraph: `BaseStore`, implemented on memvara.

Written against **langgraph-checkpoint 4.2.0**. That is the distribution, not a typo:
`pip install langgraph` gives you a wheel with no `langgraph/store/` in it at all, and
pulls `langgraph-checkpoint>=4.1,<5` in as a dependency — which is where
`langgraph.store.base` actually lives, alongside `langgraph.store.memory.InMemoryStore`.
So `memvara[langgraph]` names `langgraph-checkpoint`, and an application that already
depends on `langgraph` already satisfies it.

**Why this interface and not another.** `BaseStore.search` is

    search(namespace_prefix, *, query: str | None, filter, limit, offset)

and `query` is *the text*. Every other framework surface memvara has been asked to stand
behind either hands over a pre-computed embedding (CrewAI's `StorageBackend.search`,
LlamaIndex's `BasePydanticVectorStore.query`) or a message list (LangChain's chat
history), and memvara retrieves from a string: BM25 fused with vectors, rescored by a
per-predicate half-life. This is the one interface where nothing is lost on the way in.

**What a LangGraph item is, in memvara's terms.** This is the decision the adapter turns
on, so it is worth stating exactly. `put(namespace, key, value)` supplies three things:

    namespace + key   who/what this is about      ->  subject
    a key of `value`  which question is answered  ->  predicate
    its value         the answer                  ->  object

That is a triple, and LangGraph is the only adapter here whose caller supplies all three
parts of one. So an item is stored as **one claim per field**, not one claim per item:
`{"city": "Berlin", "food": "pizza"}` is two claims on two slots, and a later
`put(..., {"city": "Lisbon", "food": "pizza"})` **ends** `Berlin` — stamped with
`invalidated_by` pointing at Lisbon — while `pizza` is recognised as a re-observation and
is not rewritten at all. `Memvara.history(subject, "note")` then walks that one field's
versions and `search(as_of=...)` answers what the item held on a date. CrewAI cannot do
any of this, and the reason is precisely that its unit of memory is a sentence with no
subject and no predicate in it.

**What does not survive, in the order it will matter:**

* **The field name is the caller's vocabulary, not memvara's.** It is part of the slot
  identity — that is what makes per-field supersession real — but it never goes through
  `PredicateRegistry`, so it is not resolved, aliased or folded. A LangGraph field named
  `home_city` therefore does **not** contradict a fact extracted from conversation as
  `lives_in`; the two are different slots and both stay live. Within an item
  contradiction resolution is exact; across the store-vs-extraction boundary it does not
  fire at all. Going through the registry was tried and rejected: `normalize()` resolves
  morphologically, so two distinct dict keys can fold onto one predicate and silently
  cost the item a field — data loss, to buy contradiction detection nobody asked for.
* **Provenance has nothing to point at.** `put()` supplies no source turn, so `why()`
  resolves the claim, its derivation and what it superseded, and lists no episodes.
  Nothing is missing that was ever offered; it is just thinner than a fact extracted from
  a conversation, where the turn is on disk.
* **TTL.** `supports_ttl` stays `False`, so `put(ttl=...)` raises out of the base class.
  This is not a gap to be filled later: memvara retires and erases, and neither is
  expiry-on-read. A `PutOp` carrying a ttl straight into `batch()` — the one path that
  skips the base class's check — is refused rather than silently ignored, because a
  store that accepts a retention policy and does not implement it is the worst of the
  three options.

**The filter mini-language, which is the other decision.** `$eq $ne $gt $gte $lt $lte`
over nested paths cannot be answered by memvara's index, and the CrewAI adapter's
objection to post-ranking filters stands: thinning a ranked list after the fact
under-fills `limit`, and the caller cannot tell that from "nothing matched". Refusing is
not available here — `InMemoryStore` supports filters and most LangGraph apps use them —
so the shape of the answer is different: **the filter is not applied to the ranking, it
is applied to an enumeration.** Every live item in scope is reconstructed, the filter is
evaluated against it exactly, and only then is `query` used to *order* what survived. So

* a filter-only `search()` (no `query`) is **exact and complete**, with no budget in it;
* a filtered `search(query=...)` is exact in *membership* and bounded in *ordering*: the
  ranker is asked for `(offset + limit) * oversample` claims, doubling up to `max_scan`
  until the page is provably right — and it is provably right whenever the page comes
  back full of scored items, because anything the ranker did not return scores no higher
  than the last thing it did;
* when that cannot be proven, the page says so. `search()` returns a `SearchPage`, which
  is a `list[SearchItem]` carrying `complete=False`, and the store warns once naming
  `oversample=`/`max_scan=`. "No more results" and "the budget ran out" are different
  values of one attribute, which is the property the CrewAI adapter could only get by
  refusing the call.

**Two more things differ and are handled rather than hidden:**

* **Namespaces are not memvara scopes.** LangGraph's `("memories", user_id)` is an
  addressing path a search can walk from the root; memvara's `tenant > user > agent >
  session` is a *visibility* tree that only widens upward. They are not the same
  structure, so the adapter does not pretend one is the other: the memvara scope is
  fixed at construction — that binding is what isolates two stores from each other and
  nothing LangGraph passes in can widen it — and the namespace is stored per item and
  matched as data.
* **`delete()` retires by default**, exactly as in the CrewAI adapter and for the same
  reason: the item stops answering `get()`, `search()` and `list_namespaces()`, and
  `history()` and `as_of` still reach it. That is right for a graph node replacing state
  and wrong for "delete my data", so the first one warns and names
  `on_delete="erase"`.

    from langgraph.graph import StateGraph
    # pip install 'memvara[langgraph]'
    from memvara.integrations.langgraph import MemvaraStore

    store = MemvaraStore(mem, user="alice")
    graph = builder.compile(store=store)
"""

from __future__ import annotations

import asyncio
import json
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timedelta
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

from ..compat import NOTE_PREDICATE, ensure_note_predicate
from ..types import Claim, Result, content_hash, utcnow
from ._common import IntegrationError, bind, require, scope_kw

#: Import path and distribution this adapter is written against. The two names differ
#: and that is the whole point of spelling `needs` out — someone who reads "install
#: langgraph" and does exactly that still has no `langgraph.store`.
_PKG = "langgraph.store.base"
_NEEDS = "langgraph-checkpoint>=4.1 (which `pip install langgraph` already pulls in)"

#: Everything this adapter needs from `langgraph.store.base`, resolved in one call. The
#: two path helpers are in here deliberately: `index=["context[*].content"]` is a path
#: language with an owner, and reimplementing it is how an adapter and its framework
#: come to disagree about what a path means.
_NAMES = ("BaseStore", "Item", "SearchItem", "GetOp", "SearchOp", "PutOp",
          "ListNamespacesOp", "get_text_at_path", "tokenize_path")

#: Prefix on the synthetic subject owning one field's slot. Distinct from the mem0
#: importer's `mem0:` and the CrewAI adapter's `crewai:`, so one memvara store can hold
#: all three and no two of them can ever land on the same slot.
SUBJECT_PREFIX = "langgraph:"

#: The single `Claim.meta` key this adapter owns. One nested blob rather than five flat
#: keys, for the reason the CrewAI adapter gives: the item's value is the caller's own
#: dict, and any flat scheme is one unlucky field name away from user data overwriting
#: the namespace an item is filed under.
LANGGRAPH_META = "langgraph"

#: Recorded on every claim, so a later audit can tell an item a graph wrote from a fact
#: memvara extracted from a conversation.
EXTRACTOR = "langgraph-store"

_ON_DELETE = ("warn", "retire", "erase")

#: The six operators `SearchOp.filter` defines. Ordering ones go through `float()`
#: because that is what the reference implementation does, and a store that ordered
#: `"10"` before `"9"` where `InMemoryStore` did not would be a silent behaviour change
#: on the swap this adapter exists to make painless.
_ORDERING = {
    "$gt": lambda a, b: a > b,
    "$gte": lambda a, b: a >= b,
    "$lt": lambda a, b: a < b,
    "$lte": lambda a, b: a <= b,
}

_NO_TTL = (
    "LangGraph's PutOp carried ttl={ttl}, and this store does not implement expiry.\n\n"
    "Memvara has two deletions and neither of them is a TTL: retirement (the value "
    "stops answering queries and history() keeps it) and erasure (the row, its text and "
    "its vector are gone). 'Expire this item N minutes after it was last read' is a "
    "third thing, and a store that accepted the argument and dropped it would be "
    "reporting a retention policy it is not enforcing — which is worse than not "
    "supporting it, because nothing ever fails.\n\n"
    "BaseStore.put()/aput() already raise on this via supports_ttl=False; you are seeing "
    "this message because the op was handed to batch() directly. Drop the ttl, or expire "
    "items yourself with store.delete(namespace, key)."
)

_UNSUPPORTED_OPERATOR = (
    "{op!r} is not one of LangGraph's filter operators: $eq, $ne, $gt, $gte, $lt, $lte. "
    "A filter key starting with '$' is read as an operator, so a *field* genuinely named "
    "{op!r} cannot be filtered on here — the same restriction InMemoryStore has."
)

_NOT_SERIALIZABLE = (
    "the value of field {field!r} in item {address} is not JSON-serializable "
    "({error}).\n\n"
    "LangGraph requires item values to be JSON-serializable and memvara stores each "
    "field as text, so this is a hard stop rather than a lossy write: a repr() would "
    "round-trip as a string and the item would come back subtly wrong on the read that "
    "matters. Convert the value at the call site."
)

_RETIRED_NOT_ERASED = (
    "LangGraph's delete() removes an item; this call retired it instead. The item stops "
    "answering get(), search() and list_namespaces(), and every field's text remains on "
    "disk — still reachable through Memvara.history() and Memvara.search(as_of=...).\n\n"
    "That is the right default for a graph replacing its own state, which deletes a key "
    "because something superseded it and is better off keeping the trail. It is the "
    "wrong answer to a data-deletion request: pass MemvaraStore(on_delete='erase') to "
    "erase the fields outright, or 'retire' to keep this behaviour and silence this "
    "warning once you have decided."
)

_RANKING_TRUNCATED = (
    "search(query=..., filter=...) could not prove this page is the top {limit} by "
    "relevance, so some of it is ordered arbitrarily.\n\n"
    "The filter is exact — every item in it really does match, and no matching item is "
    "missing from the store's view. What ran out is the *ranking* budget: memvara's "
    "index cannot answer a filter, so the adapter ranks up to max_scan={max_scan} claims "
    "and orders the survivors. Here the filter was selective enough that the ranked "
    "claims did not cover a full page, and the remainder is filled with matching items "
    "carrying score=None, newest first.\n\n"
    "The returned list is a SearchPage: `page.complete` is False exactly when this "
    "happened, so 'nothing matched' and 'the budget ran out' are distinguishable without "
    "reading warnings. Raise MemvaraStore(oversample=...) or max_scan=... to buy "
    "certainty, or narrow namespace_prefix= so fewer claims are in play."
)


class LangGraphCompatError(IntegrationError):
    """A LangGraph store call with no honest translation onto memvara."""


class UnsupportedFilterOperator(LangGraphCompatError, ValueError):
    """A `$`-prefixed filter key that is not one of the six operators.

    Both bases on purpose. `InMemoryStore` raises `ValueError` here, so an application
    that already handles a bad filter keeps working when it swaps this store in; and it
    is an `IntegrationError`, so an application wiring up two frameworks catches it with
    the same clause as everything else these adapters refuse.
    """


class LangGraphDeletionWarning(UserWarning):
    """`delete()` retired an item instead of erasing it.

    Its own category so a deployment that has read the message and decided can silence
    exactly this without silencing everything else the library says.
    """


class LangGraphRankingWarning(UserWarning):
    """A filtered ranked search could not be proven to be the true top-N.

    Separate from the deletion warning because they are answers to different questions
    and a deployment will want to silence them independently — this one is a sizing
    knob, that one is a policy decision.
    """


class SearchPage(list):
    """`list[SearchItem]`, plus whether the ranking behind it was complete.

    A `list` subclass rather than a wrapper because `BaseStore.search` is typed to return
    `list[SearchItem]` and every LangGraph caller iterates it, indexes it and takes its
    length. Those all keep working, and the one extra thing is the attribute this adapter
    exists to be honest about:

        page = store.search(("memories",), query="food", filter={"kind": "pref"})
        if not page.complete:
            ...            # this page is a best effort, not the top-N

    `complete` is `True` for every unfiltered or unranked search — those are exact — and
    `False` only when the ranking budget was exhausted before a full page of scored items
    came back. Slicing a `SearchPage` gives a plain `list`, which is correct: a slice is
    not the answer to the query and should not carry the query's provenance.
    """

    __slots__ = ("complete", "scanned")

    def __init__(self, items: Iterable[Any] = (), *, complete: bool = True,
                 scanned: int = 0) -> None:
        super().__init__(items)
        #: Whether this page is provably the top-`limit` matching items.
        self.complete = complete
        #: How many ranked claims were examined to build it. Zero when no ranking was
        #: needed, which is the common case and the fast one.
        self.scanned = scanned


# --- the value codec ----------------------------------------------------------------


def encode_value(value: Any) -> tuple[str, bool]:
    """One field's value as the string a claim's `object` is, and whether it is JSON.

    A string is stored verbatim and everything else as JSON. The flag is what makes the
    round trip unambiguous, and it is not decoration: without it the string `"123"` and
    the integer `123` are the same three bytes on disk, and a `filter={"n": {"$gt": 100}}`
    would start matching a field whose value is the *word* "123". Storing strings raw
    rather than JSON-quoted keeps the text index readable and keeps
    `Memvara.search()` over the same store from returning `"pizza"` with quotes around it.

    `sort_keys` so a nested dict has one spelling on disk, which is what lets memvara
    recognise an unchanged field as a re-observation rather than a new value.
    """
    if isinstance(value, str):
        return value, False
    return json.dumps(value, sort_keys=True, separators=(",", ":")), True


def decode_value(stored: str, is_json: bool) -> Any:
    """The inverse of `encode_value`."""
    return json.loads(stored) if is_json else stored


# --- the filter mini-language -------------------------------------------------------


def _apply_operator(stored: Any, op: str, wanted: Any) -> bool:
    """One `$`-operator against one stored value.

    `$eq`/`$ne` compare as-is; the four ordering operators coerce through `float`, which
    is what `InMemoryStore._apply_operator` does. What is *not* copied is what that
    function does when the coercion fails: it lets the `TypeError` out, so a single item
    missing the field — `float(None)` — makes `filter={"score": {"$gt": 4}}` raise
    instead of return, and one heterogeneous item poisons the namespace. Here an
    unorderable value is simply not a match, which is the answer every SQL engine gives
    and the only one that lets a mixed namespace be searched at all.
    """
    if op == "$eq":
        return bool(stored == wanted)
    if op == "$ne":
        return bool(stored != wanted)
    compare = _ORDERING.get(op)
    if compare is None:
        raise UnsupportedFilterOperator(_UNSUPPORTED_OPERATOR.format(op=op))
    try:
        return compare(float(stored), float(wanted))
    except (TypeError, ValueError):
        return False


def matches_filter(stored: Any, wanted: Any) -> bool:
    """Whether one stored value satisfies one filter value, JSONB-style.

    Mirrors `InMemoryStore._compare_values`: a dict of `$`-keys is a set of operators, any
    other dict recurses key by key (so `{"meta": {"tag": "x"}}` reaches a nested field), a
    list matches element-wise and only at the same length, and anything else is equality.
    """
    if isinstance(wanted, dict):
        if any(str(key).startswith("$") for key in wanted):
            return all(_apply_operator(stored, key, value)
                       for key, value in wanted.items())
        if not isinstance(stored, dict):
            return False
        return all(matches_filter(stored.get(key), value)
                   for key, value in wanted.items())
    if isinstance(wanted, (list, tuple)):
        return (isinstance(stored, (list, tuple))
                and len(stored) == len(wanted)
                and all(matches_filter(s, w) for s, w in zip(stored, wanted)))
    return bool(stored == wanted)


# --- namespace matching -------------------------------------------------------------


def under_prefix(namespace: Sequence[str], prefix: Sequence[str]) -> bool:
    """Whether `namespace` lies at or beneath `prefix`. Segment-wise, never string-wise.

    `SearchOp.namespace_prefix` carries no wildcards — that is `ListNamespacesOp`'s
    job — so this is the plain tuple-prefix test, and comparing tuples rather than joined
    strings is what stops `("acme",)` reaching into `("acmecorp",)`.
    """
    head = tuple(prefix)
    return tuple(namespace)[:len(head)] == head


def matches_path(namespace: Sequence[str], path: Sequence[str], *, suffix: bool) -> bool:
    """One `MatchCondition` against one namespace, with `*` matching any one segment.

    A path longer than the namespace never matches, which is `InMemoryStore._does_match`'s
    rule and the reason `suffix=("a", "b")` does not match `("b",)`.
    """
    here, there = tuple(namespace), tuple(path)
    if len(here) < len(there):
        return False
    pairs = zip(reversed(here), reversed(there)) if suffix else zip(here, there)
    return all(want == "*" or have == want for have, want in pairs)


def field_subject(namespace: Sequence[str], key: str, field: str) -> str:
    """The slot-owning subject for one field of one item.

    A digest, and it has to be. Memvara folds a subject to its *entity* identity before
    keying a slot (`entity_key`), and that fold drops punctuation — so the readable
    spelling `langgraph:a/b#c` collides with `langgraph:a#b/c`, which is namespace
    `("a", "b")` key `"c"` colliding with namespace `("a",)` key `"b c"`. Two unrelated
    items would share one slot and supersede each other. Hex survives the fold intact,
    so the address is hashed and the readable form is carried in `Claim.meta` for
    anything that wants to display it.
    """
    address = json.dumps([list(namespace), key, field], separators=(",", ":"))
    return f"{SUBJECT_PREFIX}{content_hash(address)}"


@dataclass(slots=True)
class _Item:
    """One LangGraph item, reassembled from the claims holding its fields."""

    namespace: tuple[str, ...]
    key: str
    #: field -> the live claim on that field's slot.
    claims: dict[str, Claim] = dataclass_field(default_factory=dict)
    #: field -> this adapter's meta blob from that claim.
    blobs: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)

    @property
    def address(self) -> tuple[tuple[str, ...], str]:
        return (self.namespace, self.key)

    @property
    def value(self) -> dict[str, Any]:
        """The caller's dict, rebuilt.

        Keys come back **sorted**, not in the order they were written. Dict equality
        ignores order so a round trip still compares equal, and the alternative was
        storing a position per field — which memvara's exact-duplicate detection would
        then refuse to update on a re-ordering write, leaving the item claiming an order
        it no longer has. A promise that is always true beats one that is usually true.
        """
        return {f: decode_value(self.claims[f].object, bool(self.blobs[f].get("json")))
                for f in sorted(self.claims)}

    @property
    def created_at(self) -> datetime:
        """When this item was first written — carried forward across every update.

        `InMemoryStore` cannot do this and does not: it overwrites the whole `Item` on
        every `put`, so `created_at` there is really "when it was last written" and equals
        `updated_at` always. Memvara keeps the timeline, so the honest answer is available
        and this returns it. Stated because it is a visible divergence from the reference
        implementation, in the direction of the field's documented meaning.
        """
        return min(_parse(b["created_at"]) for b in self.blobs.values())

    @property
    def updated_at(self) -> datetime:
        """When any field of this item last changed.

        Transaction time, not valid time: it is when memvara came to believe the current
        contents. An idempotent re-`put` does **not** move it, because the write path
        recognises unchanged values as re-observations and never rewrites the claim —
        so this says "last changed" rather than "last written to", which is the more
        useful of the two and the only one the store can honestly report.
        """
        return max(c.recorded_at for c in self.claims.values())


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


@lru_cache(maxsize=None)
def _langgraph() -> Any:
    """The LangGraph names this adapter uses, imported once, on first use."""
    return SimpleNamespace(**dict(zip(
        _NAMES, require(_PKG, *_NAMES, extra="langgraph", needs=_NEEDS))))


class _Store:
    """Everything `MemvaraStore` does, with no LangGraph in sight.

    A mixin rather than a subclass, for the reason `_common` gives: `BaseStore` cannot be
    imported at module scope without making langgraph-checkpoint a hard dependency of
    `import memvara`. The composed class is built on demand (`_store_class`), so all the
    behaviour is here — testable with nothing installed — and composition is one line.

    Unlike the LangChain and LlamaIndex mixins this one *does* construct: `BaseStore` is a
    plain ABC with no `__init__` of its own, so there is no pydantic model to hand fields
    to and the scope binding happens here.
    """

    def __init__(self, memory: Any, *, tenant: str | None = None, user: str | None = None,
                 agent: str | None = None, session: str | None = None,
                 on_delete: str = "warn", oversample: int = 4, max_scan: int = 1000,
                 min_score: float = 0.0,
                 clock: Callable[[], datetime] | None = None) -> None:
        if on_delete not in _ON_DELETE:
            raise ValueError(
                f"on_delete={on_delete!r} is not one of {_ON_DELETE}; see "
                "memvara.integrations.langgraph.LangGraphDeletionWarning for what each "
                "one does"
            )
        self.memory, self.scope = bind(memory, tenant=tenant, user=user, agent=agent,
                                       session=session)
        self.on_delete = on_delete
        #: How many ranked claims to pull per item a filtered page needs. Doubling from
        #: here up to `max_scan`; see `_rank`. The knob exists because the filter is not
        #: part of the index, and saying so out loud is the point of it.
        self.oversample = max(1, oversample)
        #: The hard ceiling on that doubling. Past it the page is served with what is in
        #: hand and marked `complete=False` rather than growing without bound.
        self.max_scan = max(1, max_scan)
        #: Floor on memvara's normalized relevance, for the ranked leg. LangGraph has no
        #: way to express one; it rides on the object, exactly as `as_of` does elsewhere.
        self.min_score = min_score
        #: Read once per write. Injectable so a test can state its instants instead of
        #: patching the clock, which is this repo's rule and also the only way to test
        #: `as_of` over a store whose interface carries no timestamps at all.
        self.clock = clock if clock is not None else utcnow
        self._last_instant: datetime | None = None
        # Once per instance rather than per item: deleting a hundred keys would otherwise
        # emit a hundred warnings, get filtered wholesale, and take the message with it.
        self._warned_delete = False
        self._warned_ranking = False
        # Declares the note slot single-valued and persists that, which is what turns a
        # second `put` of the same field into a supersession instead of a second live
        # value. Both callers of the write path also retire explicitly, so correctness
        # never depends on this having run — but the `invalidated_by` pointer does.
        ensure_note_predicate(self.memory, NOTE_PREDICATE, self.scope.tenant)

    # -- plumbing ------------------------------------------------------------

    @property
    def _kw(self) -> dict[str, Any]:
        return scope_kw(self.scope)

    def _instant(self) -> datetime:
        """One transaction instant per batch, strictly increasing.

        The monotonic step is load-bearing rather than tidy. Two writes to one field at
        the same instant produce a supersession whose predecessor was ended at exactly
        the moment its replacement began, i.e. a zero-length interval that `as_of` cannot
        resolve to either value. `utcnow()` usually separates two calls on its own and on
        a coarse clock does not; an injected test clock never does. A microsecond is
        cheaper than a bitemporal ambiguity.
        """
        now = self.clock()
        last = self._last_instant
        if last is not None and now <= last:
            now = last + timedelta(microseconds=1)
        self._last_instant = now
        return now

    @staticmethod
    def _blob_of(claim: Claim) -> dict[str, Any] | None:
        """This adapter's bookkeeping for a claim, or `None` if it is not one of ours.

        Three conditions, not one. A memvara store legitimately holds LangGraph items,
        CrewAI records, imported mem0 notes and ordinary extracted facts side by side —
        they share the `note` predicate — and handing a `lives_in` triple back as an
        `Item` would mean inventing a namespace and a key for it.
        """
        if claim.predicate != NOTE_PREDICATE:
            return None
        if not claim.subject.startswith(SUBJECT_PREFIX):
            return None
        blob = claim.meta.get(LANGGRAPH_META)
        return blob if isinstance(blob, dict) else None

    def _snapshot(self) -> list[_Item]:
        """Every live item in scope, newest-updated first.

        One scan per `batch()`, shared by every op in it. This is O(claims visible at
        this scope) and it is the honest cost of an interface whose filter memvara cannot
        index: the alternative is post-filtering a ranked list, which is the thing this
        adapter exists not to do. A `Store.claims_by_subject_prefix` would retire it; see
        the workstream report.
        """
        items: dict[tuple[tuple[str, ...], str], _Item] = {}
        for claim in self.memory.get_all(**self._kw):
            blob = self._blob_of(claim)
            if blob is None:
                continue
            address = (tuple(blob["namespace"]), blob["key"])
            item = items.get(address)
            if item is None:
                item = items[address] = _Item(address[0], address[1])
            field = blob["field"]
            item.claims[field] = claim
            item.blobs[field] = blob
        ordered = sorted(items.values(), key=lambda i: (i.namespace, i.key))
        # Stable, so the sort above is the tie-break: two items updated in the same
        # `batch()` share an instant, and paging over an order nothing chose is how
        # `offset=` starts skipping and repeating rows.
        ordered.sort(key=lambda i: i.updated_at, reverse=True)
        return ordered

    # -- writing -------------------------------------------------------------

    def _indexed_texts(self, value: Mapping[str, Any],
                       index: Any) -> dict[str, list[str]]:
        """field -> the strings LangGraph says to index for it. Absent means "not indexed".

        `index=None` means the store's own default, and `IndexConfig.fields` defaults to
        `["$"]` — the whole object — so every field is indexed. `False` means none.
        A list is a list of *paths*, parsed by LangGraph's own `tokenize_path` and
        resolved by its own `get_text_at_path`, because `"context[*].content"` is their
        language and an adapter that reimplemented it would eventually disagree with them
        about what it meant. A path is attributed to the top-level field it starts at,
        which is the finest granularity a claim-per-field model has.
        """
        if index is False:
            return {}
        lg = _langgraph()
        if index is None:
            return self._whole(lg, value)
        out: dict[str, list[str]] = {}
        for path in index:
            tokens = lg.tokenize_path(path)
            if not tokens:
                continue
            if tokens[0] == "$":
                # `get_text_at_path(value, ["$"])` is empty — the sentinel is only
                # honoured as the bare string — so "index everything" is expanded here
                # rather than silently indexing nothing.
                for field, texts in self._whole(lg, value).items():
                    out.setdefault(field, []).extend(texts)
                continue
            texts = lg.get_text_at_path(value, tokens)
            if texts:
                out.setdefault(tokens[0], []).extend(texts)
        return out

    @staticmethod
    def _whole(lg: Any, value: Mapping[str, Any]) -> dict[str, list[str]]:
        """Every field indexed on its own value, which is what `fields=["$"]` amounts to."""
        return {field: lg.get_text_at_path(value, [field]) for field in value}

    @staticmethod
    def _text_for(field: str, texts: list[str] | None) -> str:
        """What gets embedded and BM25-indexed for one field.

        `"<field>: <value>"` is memvara's own `Claim.render()` shape — predicate then
        object — with the field standing in for the predicate, which is exactly what it
        is here. The subject is a digest and deliberately stays out of it.

        A field LangGraph said not to index contributes **only its name**. That is
        stronger than ranking it last: the value never enters the text index or the
        vector index at all, so `index=False` on a blob of raw tool output keeps it out
        of everything memvara can retrieve while `get()` still returns it verbatim.
        """
        if not texts:
            return field
        return f"{field}: {' '.join(texts)}"

    def _write_field(self, namespace: Sequence[str], key: str, field: str, value: Any,
                     texts: list[str] | None, created: datetime, at: datetime) -> None:
        try:
            stored, is_json = encode_value(value)
        except (TypeError, ValueError) as exc:
            raise LangGraphCompatError(_NOT_SERIALIZABLE.format(
                field=field, address=(tuple(namespace), key), error=exc)) from exc
        # Typed `dict[str, Any]` rather than inlined for the reason the CrewAI adapter
        # gives: `remember` takes `**meta`, and a narrower value type is checked against
        # every named keyword the blob could theoretically land on.
        blob: dict[str, Any] = {LANGGRAPH_META: {
            "namespace": list(namespace),
            "key": key,
            "field": field,
            "json": is_json,
            "indexed": texts is not None,
            "created_at": created.isoformat(),
        }}
        self.memory.remember(
            field_subject(namespace, key, field), NOTE_PREDICATE, stored,
            text=self._text_for(field, texts),
            # Both axes at the same instant: LangGraph carries no notion of when a value
            # became true in the world, so the only honest valid time is the one we
            # learned it. Backdating one axis and not the other would invent a history.
            valid_from=at, recorded_at=at,
            extractor=EXTRACTOR, **blob, **self._kw,
        )

    def _apply_put(self, existing: _Item | None, op: Any, at: datetime) -> None:
        """One `PutOp`: whole-item replace, as LangGraph defines it.

        Fields present in the new value are asserted onto their slots — unchanged ones
        are recognised as re-observations and cost nothing — and fields the item used to
        have and no longer does are **retired**, not erased, whatever `on_delete` says.
        A `put` is an update rather than a deletion request, and the bitemporal reading
        of "this field is gone" is that belief in it ended here.
        """
        if op.ttl is not None:
            raise LangGraphCompatError(_NO_TTL.format(ttl=op.ttl))
        namespace, key = tuple(op.namespace), op.key
        if op.value is None:
            self._remove(existing, at)
            return
        created = existing.created_at if existing is not None else at
        texts = self._indexed_texts(op.value, op.index)
        for field, value in op.value.items():
            self._write_field(namespace, key, field, value, texts.get(field), created, at)
        if existing is not None:
            for field, claim in existing.claims.items():
                if field not in op.value:
                    # No `invalidated_by`: nothing replaced this field, it stopped
                    # existing, and pointing it at an unrelated sibling would make
                    # `why()` report a supersession that never happened.
                    self.memory.delete(claim.id, at=at, **self._kw)

    def _remove(self, existing: _Item | None, at: datetime) -> None:
        """`delete(namespace, key)`, or a `None`-valued `PutOp`. Retires by default."""
        if existing is None:
            return
        for claim in existing.claims.values():
            if self.on_delete == "erase":
                # `sources=False`, and this is where the LangGraph adapter differs from
                # the CrewAI one: a CrewAI record *is* its source turn, so erasing the
                # record has to take the turn with it. An item written by `put()` has no
                # source turn — nothing was said, a graph asserted it — so there is
                # nothing else to reach for.
                self.memory.erase(claim.id, **self._kw)
            else:
                self.memory.delete(claim.id, at=at, **self._kw)
        if self.on_delete == "warn" and not self._warned_delete:
            self._warned_delete = True
            warnings.warn(_RETIRED_NOT_ERASED, LangGraphDeletionWarning, stacklevel=2)

    # -- reading -------------------------------------------------------------

    def _to_item(self, item: _Item) -> Any:
        lg = _langgraph()
        return lg.Item(value=item.value, key=item.key, namespace=item.namespace,
                       created_at=item.created_at, updated_at=item.updated_at)

    def _to_search_item(self, item: _Item, score: float | None) -> Any:
        lg = _langgraph()
        return lg.SearchItem(item.namespace, item.key, item.value, item.created_at,
                             item.updated_at, score)

    def _rank(self, query: str, wanted: set[tuple[tuple[str, ...], str]],
              need: int) -> tuple[dict[tuple[tuple[str, ...], str], float], bool, int]:
        """Relevance for as many of `wanted` as the budget reaches.

        Returns `(scores, proven, scanned)`.

        The completeness argument, because it is the whole answer to "silently under-filling
        a limit is indistinguishable from nothing matched":

        `Memvara.search(k=budget)` returns the top `budget` claims across the scope, in
        descending score. Anything it did not return scores no higher than the last thing
        it did. So the page is *provably* the true top-`need` in either of two cases — the
        ranker came back with fewer results than we asked for, meaning it had nothing left
        to give, or we already have `need` distinct matching items with real scores, in
        which case nothing unseen can outrank the worst of them. Only when neither holds at
        `max_scan` is the answer a best effort, and that is exactly when `complete` is
        `False`.

        Doubling rather than one big `k`: the common case is a filter that barely thins
        anything, where the first budget already proves the page and a full-corpus rank
        would have been paid for nothing.
        """
        scores: dict[tuple[tuple[str, ...], str], float] = {}
        if not wanted or need <= 0:
            return scores, True, 0
        budget = min(self.max_scan, need * self.oversample)
        while True:
            results = cast("list[Result]", self.memory.search(
                query, k=budget, min_score=self.min_score, **self._kw))
            scores = {}
            for result in results:
                blob = self._blob_of(result.claim)
                if blob is None:
                    continue
                address = (tuple(blob["namespace"]), blob["key"])
                # Max pooling over an item's fields, which is what `InMemoryStore` does
                # over an item's several vectors: an item is as relevant as its best part.
                if address in wanted and result.score > scores.get(address, -1.0):
                    scores[address] = result.score
            if len(results) < budget or len(scores) >= need:
                return scores, True, len(results)
            if budget >= self.max_scan:
                return scores, False, len(results)
            budget = min(self.max_scan, budget * 2)

    def _search(self, snapshot: Sequence[_Item], op: Any) -> SearchPage:
        """One `SearchOp`. The filter is exact; only the ordering is budgeted."""
        matching = [item for item in snapshot
                    if under_prefix(item.namespace, op.namespace_prefix)
                    and self._passes(item, op.filter)]
        if op.query is None:
            # No ranking to be done, so no budget and nothing to be uncertain about:
            # this page is exactly the newest-first slice of everything that matched.
            page = matching[op.offset:op.offset + op.limit]
            return SearchPage([self._to_search_item(i, None) for i in page])
        need = max(0, op.offset) + max(0, op.limit)
        scores, proven, scanned = self._rank(
            op.query, {item.address for item in matching}, need)
        ranked = sorted((i for i in matching if i.address in scores),
                        key=lambda i: (-scores[i.address], i.namespace, i.key))
        # Unranked matches keep the snapshot's newest-first order and carry `score=None`,
        # which is `InMemoryStore`'s own answer for an item it holds no vector for: they
        # fill the page rather than being dropped, because they *do* match the filter.
        ordered = ranked + [i for i in matching if i.address not in scores]
        page = ordered[op.offset:op.offset + op.limit]
        complete = proven or all(i.address in scores for i in page)
        if not complete and not self._warned_ranking:
            self._warned_ranking = True
            warnings.warn(_RANKING_TRUNCATED.format(limit=op.limit,
                                                    max_scan=self.max_scan),
                          LangGraphRankingWarning, stacklevel=4)
        return SearchPage([self._to_search_item(i, scores.get(i.address)) for i in page],
                          complete=complete, scanned=scanned)

    @staticmethod
    def _passes(item: _Item, criteria: Mapping[str, Any] | None) -> bool:
        if not criteria:
            return True
        value = item.value
        return all(matches_filter(value.get(field), wanted)
                   for field, wanted in criteria.items())

    def _list_namespaces(self, snapshot: Sequence[_Item], op: Any) -> list[tuple[str, ...]]:
        """One `ListNamespacesOp`, over the namespaces that currently hold something.

        A namespace whose every item has been deleted is **not** listed.
        `InMemoryStore` lists it, because deleting an item pops the key and leaves the
        namespace's empty dict behind in a `defaultdict` — an artefact of its storage
        rather than a decision, and one that makes `list_namespaces()` report places
        nothing lives.
        """
        namespaces = {item.namespace for item in snapshot}
        for condition in (op.match_conditions or ()):
            namespaces = {ns for ns in namespaces
                          if matches_path(ns, condition.path,
                                          suffix=condition.match_type == "suffix")}
        if op.max_depth is not None:
            namespaces = {ns[:op.max_depth] for ns in namespaces}
        return sorted(namespaces)[op.offset:op.offset + op.limit]

    # -- the two abstract methods --------------------------------------------

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        """Execute `ops`, one result each, in order.

        Reads observe the state as it was *before* the batch and writes land after all of
        them, which is `InMemoryStore`'s ordering and therefore the one LangGraph's own
        batching layer is written against. Two puts to one address in a single batch
        collapse to the last, for the same reason.

        The whole write half is one memvara transaction. A `put` of a five-field item is
        five asserts and however many retirements, and separately committed a crash in
        the middle leaves an item that is half its old value and half its new one — with
        no way to tell, because both halves look live. The cost is that the store's write
        lock is held across those encodes; that is the right trade here and the wrong one
        in `Memvara.add`, which holds it across a model call instead.
        """
        lg = _langgraph()
        snapshot = self._snapshot()
        by_address = {item.address: item for item in snapshot}
        results: list[Any] = []
        puts: dict[tuple[tuple[str, ...], str], Any] = {}
        for op in ops:
            if isinstance(op, lg.GetOp):
                found = by_address.get((tuple(op.namespace), op.key))
                results.append(None if found is None else self._to_item(found))
            elif isinstance(op, lg.SearchOp):
                results.append(self._search(snapshot, op))
            elif isinstance(op, lg.ListNamespacesOp):
                results.append(self._list_namespaces(snapshot, op))
            elif isinstance(op, lg.PutOp):
                puts[(tuple(op.namespace), op.key)] = op
                results.append(None)
            else:
                # `InMemoryStore`'s own wording, so an application that already matches on
                # this message keeps matching.
                raise ValueError(f"Unknown operation type: {type(op)}")
        if puts:
            at = self._instant()
            batch = getattr(self.memory.store, "batch", None)
            with (batch() if batch is not None else nullcontext()):
                for address, op in puts.items():
                    self._apply_put(by_address.get(address), op, at)
        return results

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        """`batch` off the event-loop thread.

        `asyncio.to_thread`, not an async rewrite: memvara is synchronous and SQLite has
        no async driver worth the name (see `memvara.aio`). What this buys is a loop that
        keeps serving while an encode and a write happen, which is the whole ask — and it
        matters more here than in the other adapters, because LangGraph awaits the store
        on the hot path of every node that touches memory.
        """
        return await asyncio.to_thread(self.batch, list(ops))

    # -- escape hatches ------------------------------------------------------

    def search_memory(self, query: str, **kw: Any) -> list[Any]:
        """`Memvara.search` at this store's scope — scores, provenance, `as_of=`.

        A `SearchItem` has nowhere to put the triple, the two time axes, the ranking
        explanation or the source turn ids; `value` is the caller's own dict and writing
        into it would corrupt the round trip. So this is where they live, and it is the
        same escape hatch `MemvaraChatMessageHistory.search` and
        `MemvaraMemoryBlock.search` are.
        """
        return self.memory.search(query, **self._kw, **kw)

    def history(self, namespace: Sequence[str], key: str, field: str) -> list[Any]:
        """Every value one field of one item has ever held, oldest first.

        The thing no other key-value store can answer, and the reason to put memvara
        behind this interface: `get()` is the current value, this is the timeline —
        each entry carrying when it was believed, when belief ended and what replaced it.
        """
        return self.memory.history(field_subject(namespace, key, field), NOTE_PREDICATE,
                                   **self._kw)

    def __repr__(self) -> str:
        return (f"<MemvaraStore {self.scope.key()} on_delete={self.on_delete} "
                f"of {self.memory!r}>")


@lru_cache(maxsize=None)
def _store_class(base: type) -> type:
    """`_Store` composed with `BaseStore`.

    Cached so the class is minted once and `isinstance` is stable — two calls returning
    two structurally identical classes is the kind of thing that passes every test and
    then fails one `isinstance` check in somebody's dispatch table.
    """

    class MemvaraStore(_Store, base):  # type: ignore[misc, valid-type]
        """Memvara behind LangGraph's `BaseStore`, bound to one scope.

            store = MemvaraStore(mem, user="alice")
            graph = builder.compile(store=store)

        An item is stored as one claim per field, so a `put` that changes one field
        supersedes exactly that field and `history(namespace, key, field)` walks its
        versions. `search(query=...)` runs memvara's hybrid retrieval on the query text.
        `filter=` is evaluated exactly, against a full enumeration rather than a ranked
        list, and the returned `SearchPage` says whether its *ordering* was complete.

        `supports_ttl` stays `False`: memvara retires and erases, and neither is expiry.
        """

        # Inherited from `BaseStore` and restated because it is a decision rather than a
        # default we never got round to changing. See the module docstring.
        supports_ttl = False

    return MemvaraStore


def __getattr__(name: str) -> Any:
    # PEP 562, same as `memvara.llm`: naming the class here must not import
    # langgraph-checkpoint, or the numpy-only install stops being able to
    # `import memvara` — which is a CI job, not a slogan.
    if name == "MemvaraStore":
        return _store_class(_langgraph().BaseStore)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MemvaraStore", "SearchPage", "LangGraphCompatError", "UnsupportedFilterOperator",
    "LangGraphDeletionWarning", "LangGraphRankingWarning",
    "encode_value", "decode_value", "matches_filter", "under_prefix", "matches_path",
    "field_subject", "SUBJECT_PREFIX", "LANGGRAPH_META", "EXTRACTOR",
]
