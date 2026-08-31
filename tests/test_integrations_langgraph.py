"""LangGraph's `BaseStore`, implemented on memvara.

**langgraph-checkpoint is not installed here, and is not needed.** That is the property
under test, not a convenience: `import memvara` must keep working with numpy alone, so
every framework import in `memvara/integrations/**` happens inside a function, and the
way to prove it is a suite that drives the whole surface with nothing installed. The
fakes below go into `sys.modules` under the real import path, so the adapter's own
`require()` and `__getattr__` run for real against them.

The fakes mirror **langgraph-checkpoint 4.2.0** — `langgraph.store.base.BaseStore` and
`langgraph.store.memory.InMemoryStore` — read out of the installed wheel rather than
remembered. Where `FakeBaseStore` reproduces behaviour rather than just shape
(`_validate_namespace`, the `supports_ttl` gate, the exact positional order every
concrete method packs its op in) it is because the adapter *depends* on that behaviour,
and a fake that agrees with the adapter is worth nothing.

Four things were measured against the real package and are pinned here as tests:

* `langgraph` the distribution ships **no** `langgraph/store/` — the module lives in
  `langgraph-checkpoint`, which is why the extra names that one.
* `InMemoryStore.search(filter={"score": {"$gt": 1}})` raises `TypeError` when a single
  item in the namespace lacks `score`, because its `_apply_operator` calls `float(None)`.
  This adapter declines to reproduce that.
* `InMemoryStore` resets `created_at` on every `put`, so it equals `updated_at` always.
  Memvara keeps the timeline and reports the real one.
* `BaseStore.put(ttl=...)` raises out of the base class given `supports_ttl = False`,
  so the adapter gets that refusal for free — but `batch()` skips it, and that path is
  its own test.

What most of these tests assert is not "it works" but **what is exact, what is bounded,
and what is lost**. A key-value store is a much better fit for memvara than a message
list is, and the interesting part is precisely where it stops fitting.
"""

from __future__ import annotations

import asyncio
import json
import sys
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, NamedTuple

import numpy as np
import pytest

from memvara import HashingEmbedder, Memvara, NullLLM
from memvara.compat import NOTE_PREDICATE
from memvara.integrations import IntegrationError
from memvara.integrations import langgraph as lg

TZ = timezone.utc
T0 = datetime(2024, 3, 1, 12, 0, tzinfo=TZ)
NS = ("memories", "alice")


def run(coro):
    return asyncio.run(coro)


# =====================================================================================
# The fake framework: langgraph.store.base, as langgraph-checkpoint 4.2.0 defines it.
# =====================================================================================


class FakeItem:
    """`langgraph.store.base.Item`. Keyword-only, exactly as the real one is."""

    def __init__(self, *, value, key, namespace, created_at, updated_at):
        self.value = value
        self.key = key
        self.namespace = tuple(namespace)
        self.created_at = created_at
        self.updated_at = updated_at


class FakeSearchItem(FakeItem):
    """`SearchItem`. Positional, and *not* keyword-only — the signatures genuinely differ,
    and an adapter that called one the way it calls the other raises `TypeError`."""

    def __init__(self, namespace, key, value, created_at, updated_at, score=None):
        super().__init__(value=value, key=key, namespace=namespace,
                         created_at=created_at, updated_at=updated_at)
        self.score = score


class FakeGetOp(NamedTuple):
    namespace: tuple[str, ...]
    key: str
    refresh_ttl: bool = True


class FakeSearchOp(NamedTuple):
    namespace_prefix: tuple[str, ...]
    filter: dict[str, Any] | None = None
    limit: int = 10
    offset: int = 0
    query: str | None = None
    refresh_ttl: bool = True


class FakeMatchCondition(NamedTuple):
    match_type: Literal["prefix", "suffix"]
    path: tuple[str, ...]


class FakeListNamespacesOp(NamedTuple):
    match_conditions: tuple[FakeMatchCondition, ...] | None = None
    max_depth: int | None = None
    limit: int = 100
    offset: int = 0


class FakePutOp(NamedTuple):
    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any] | None
    index: Any = None
    ttl: float | None = None


class InvalidNamespaceError(ValueError):
    pass


def _validate_namespace(namespace):
    """The real one, because `put()` runs it before the adapter is ever reached."""
    if not namespace:
        raise InvalidNamespaceError("Namespace cannot be empty.")
    for label in namespace:
        if not isinstance(label, str) or "." in label or not label:
            raise InvalidNamespaceError(f"Invalid namespace label {label!r}")
    if namespace[0] == "langgraph":
        raise InvalidNamespaceError('Root label for namespace cannot be "langgraph".')


class FakeBaseStore:
    """`langgraph.store.base.BaseStore`.

    Every concrete method is the real one: it packs exactly one op, in exactly the
    positional order the real base uses, and returns `batch(...)[0]`. That ordering is
    contract — `SearchOp(prefix, filter, limit, offset, query, refresh_ttl)` puts `query`
    fifth, and a base that spelled it differently would let a broken adapter pass.
    """

    supports_ttl: bool = False
    ttl_config = None

    def batch(self, ops):  # pragma: no cover - the adapter overrides it
        raise NotImplementedError

    async def abatch(self, ops):  # pragma: no cover - the adapter overrides it
        raise NotImplementedError

    def get(self, namespace, key, *, refresh_ttl=None):
        return self.batch([FakeGetOp(namespace, str(key), True)])[0]

    def search(self, namespace_prefix, /, *, query=None, filter=None, limit=10,
               offset=0, refresh_ttl=None):
        return self.batch([FakeSearchOp(namespace_prefix, filter, limit, offset, query,
                                        True)])[0]

    def put(self, namespace, key, value, index=None, *, ttl=None):
        _validate_namespace(namespace)
        if ttl is not None and not self.supports_ttl:
            raise NotImplementedError(
                f"TTL is not supported by {self.__class__.__name__}. "
                "Use a store implementation that supports TTL or set ttl=None."
            )
        self.batch([FakePutOp(namespace, str(key), value, index=index, ttl=ttl)])

    def delete(self, namespace, key):
        self.batch([FakePutOp(namespace, str(key), None, ttl=None)])

    def list_namespaces(self, *, prefix=None, suffix=None, max_depth=None, limit=100,
                        offset=0):
        conditions = []
        if prefix:
            conditions.append(FakeMatchCondition("prefix", prefix))
        if suffix:
            conditions.append(FakeMatchCondition("suffix", suffix))
        return self.batch([FakeListNamespacesOp(tuple(conditions), max_depth, limit,
                                                offset)])[0]

    async def aget(self, namespace, key, *, refresh_ttl=None):
        return (await self.abatch([FakeGetOp(namespace, str(key), True)]))[0]

    async def asearch(self, namespace_prefix, /, *, query=None, filter=None, limit=10,
                      offset=0, refresh_ttl=None):
        return (await self.abatch(
            [FakeSearchOp(namespace_prefix, filter, limit, offset, query, True)]))[0]

    async def aput(self, namespace, key, value, index=None, *, ttl=None):
        _validate_namespace(namespace)
        await self.abatch([FakePutOp(namespace, str(key), value, index=index, ttl=ttl)])

    async def adelete(self, namespace, key):
        await self.abatch([FakePutOp(namespace, str(key), None)])

    async def alist_namespaces(self, *, prefix=None, suffix=None, max_depth=None,
                               limit=100, offset=0):
        conditions = []
        if prefix:
            conditions.append(FakeMatchCondition("prefix", prefix))
        if suffix:
            conditions.append(FakeMatchCondition("suffix", suffix))
        return (await self.abatch(
            [FakeListNamespacesOp(tuple(conditions), max_depth, limit, offset)]))[0]


def fake_tokenize_path(path):
    """`langgraph.store.base.tokenize_path`: "a.b[*].c" -> ["a", "b", "[*]", "c"]."""
    if not path:
        return []
    tokens = []
    for chunk in path.split("."):
        head, _, rest = chunk.partition("[")
        if head:
            tokens.append(head)
        while rest:
            index, _, rest = rest.partition("]")
            tokens.append(f"[{index}]")
            rest = rest.lstrip("[")
    return tokens


def fake_get_text_at_path(obj, path):
    """`langgraph.store.base.get_text_at_path`, including the two behaviours that matter.

    A missing field yields `[]` rather than a blank, and a non-string leaf is rendered —
    scalars with `str`, containers as sorted-key JSON. Note that `["$"]` is *not* the
    whole-object sentinel: only the bare string `"$"` is, so a tokenized `"$"` resolves
    to a missing field. The adapter expands that case itself, which is why it is tested.
    """
    if path == "$":
        return [json.dumps(obj, sort_keys=True)]
    nodes = [obj]
    for token in path:
        step = []
        for node in nodes:
            if token == "[*]":
                step.extend(node if isinstance(node, list) else [])
            elif token.startswith("[") and token.endswith("]"):
                index = int(token[1:-1])
                if isinstance(node, list) and -len(node) <= index < len(node):
                    step.append(node[index])
            elif isinstance(node, dict) and token in node:
                step.append(node[token])
        nodes = step
    out = []
    for node in nodes:
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, (dict, list)):
            out.append(json.dumps(node, sort_keys=True))
        elif node is not None:
            out.append(str(node))
    return out


FAKE_MODULE = dict(
    BaseStore=FakeBaseStore, Item=FakeItem, SearchItem=FakeSearchItem, GetOp=FakeGetOp,
    SearchOp=FakeSearchOp, PutOp=FakePutOp, ListNamespacesOp=FakeListNamespacesOp,
    MatchCondition=FakeMatchCondition, get_text_at_path=fake_get_text_at_path,
    tokenize_path=fake_tokenize_path, InvalidNamespaceError=InvalidNamespaceError,
)


def install(monkeypatch):
    """Put the fakes on the real import path, so `require()` finds them."""
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "langgraph.store.base",
                        SimpleNamespace(**FAKE_MODULE))


# =====================================================================================
# Fixtures
# =====================================================================================


@pytest.fixture(autouse=True)
def _fresh_module_state():
    """Clear the adapter's per-process caches between tests.

    Both the composed class and the resolved module are memoized, which is right in a
    process and wrong in a suite: one test's fake would otherwise decide what a later
    test — including the one asserting a *missing* package — gets to see.
    """
    for cache in (lg._langgraph, lg._store_class):
        cache.cache_clear()
    yield
    for cache in (lg._langgraph, lg._store_class):
        cache.cache_clear()


@pytest.fixture()
def mem():
    # NullLLM by name, not by default: the default warns about degraded extraction, and
    # a suite that trips that warning teaches everyone to filter the category.
    m = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice")
    yield m
    m.close()


@pytest.fixture()
def clock():
    """A stated sequence of instants, one per write.

    Explicit datetimes rather than a patched clock, which is the house rule — and here it
    is also the only way to test `as_of` at all, since `BaseStore` carries no timestamps
    in either direction.
    """
    ticks = iter(T0 + timedelta(minutes=i) for i in range(200))
    return lambda: next(ticks)


@pytest.fixture()
def store(mem, clock, monkeypatch):
    install(monkeypatch)
    return lg.MemvaraStore(mem, user="alice", clock=clock)


def addresses(page):
    return [(i.namespace, i.key) for i in page]


def keys(page):
    return [i.key for i in page]


def lexical_hits(mem, query):
    """The claim texts BM25 matched — i.e. what is genuinely in the *text* index.

    "`mem.search()` returned nothing" is not a statement about the index: the vector leg
    returns its top-k however irrelevant the query, so a hashing embedder will always
    hand something back. `Explanation.lexical_rank` is the leg that only fires on a real
    term match, which is what the `index=` tests below actually need to assert.
    """
    return [r.text for r in mem.search(query, k=20)
            if r.explain.lexical_rank is not None]


# =====================================================================================
# How a dict becomes claims — the decision the adapter turns on
# =====================================================================================


def test_the_store_is_a_real_basestore_so_langgraph_will_accept_it(store):
    """`BaseStore` is an ABC with two abstract methods, not a Protocol: a class that
    merely had the methods would fail at `builder.compile(store=...)`, not here."""
    assert isinstance(store, FakeBaseStore)
    assert store.supports_ttl is False


def test_the_composed_class_is_minted_once_so_isinstance_stays_stable(monkeypatch):
    install(monkeypatch)
    assert lg.MemvaraStore is lg.MemvaraStore


def test_one_item_becomes_one_claim_per_field_not_one_claim_per_item(store, mem):
    """The whole mapping in one assertion. LangGraph supplies a subject (namespace+key),
    a predicate (the dict key) and an object (its value), which is a triple — and
    splitting the item into per-field slots is what makes the supersession below real."""
    store.put(NS, "m1", {"city": "Berlin", "food": "pizza"})
    fields = [c for c in mem.get_all() if lg.LANGGRAPH_META in c.meta]
    assert sorted(c.meta[lg.LANGGRAPH_META]["field"] for c in fields) == ["city", "food"]
    assert sorted(c.object for c in fields) == ["Berlin", "pizza"]
    assert {c.predicate for c in fields} == {NOTE_PREDICATE}


def test_a_changed_field_supersedes_only_itself_and_leaves_the_others_alone(store, mem):
    """What CrewAI cannot do, and the reason this adapter exists. Its unit of memory is
    an opaque sentence, so "moved to Lisbon" lands on a second slot and both stay live.
    Here the city is a slot of its own, so the move ends exactly the city."""
    store.put(NS, "m1", {"city": "Berlin", "food": "pizza"})
    store.put(NS, "m1", {"city": "Lisbon", "food": "pizza"})
    assert store.get(NS, "m1").value == {"city": "Lisbon", "food": "pizza"}
    timeline = store.history(NS, "m1", "city")
    assert [c.object for c in timeline] == ["Berlin", "Lisbon"]
    assert timeline[0].invalidated_by == timeline[1].id
    # `ended`, not `retired`, and the difference is the whole point of the distinction:
    # the world moved, the record was not wrong. Asserted rather than described because
    # the description was the part that went astray — five documents, this docstring
    # among them, said a changed field is "retired". Only the companion test below,
    # where a field is *dropped*, is a retirement.
    assert timeline[0].state == "ended"
    assert timeline[0].invalidated_at is None
    assert len(store.history(NS, "m1", "food")) == 1     # untouched, not rewritten


def test_time_travel_works_over_a_store_whose_interface_has_no_timestamps(store, mem):
    """`as_of` is a property of the query, so it survives an interface that never
    imagined it — and a key-value store that can answer "what did this hold in March"
    is not something `BaseStore` can express at all."""
    store.put(NS, "m1", {"city": "Berlin"})
    store.put(NS, "m1", {"city": "Lisbon"})
    then = [r.claim.object for r in mem.search("city", as_of=T0 + timedelta(seconds=30))]
    assert then == ["Berlin"]
    assert store.get(NS, "m1").value == {"city": "Lisbon"}


def test_a_dropped_field_is_retired_rather_than_left_live(store):
    """`put` is a whole-item replace, so a field the new value omits has to stop being
    returned. Retirement is the bitemporal reading of that, and the field's history
    survives it — but `get()` must not still be showing it."""
    store.put(NS, "m1", {"city": "Berlin", "food": "pizza"})
    store.put(NS, "m1", {"city": "Berlin"})
    assert store.get(NS, "m1").value == {"city": "Berlin"}
    retired = store.history(NS, "m1", "food")[-1]
    assert retired.invalidated_at is not None
    # Nothing replaced it, so nothing is claimed to have: a pointer at an unrelated
    # sibling field would make `why()` report a supersession that never happened.
    assert retired.invalidated_by is None


def test_an_unchanged_field_is_a_re_observation_so_updated_at_does_not_move(store):
    """Memvara recognises an identical value rather than rewriting it, which means
    `updated_at` says "last changed" instead of "last written to". Stated as a test
    because the reference implementation cannot tell the two apart and this one can."""
    store.put(NS, "m1", {"city": "Berlin"})
    first = store.get(NS, "m1").updated_at
    store.put(NS, "m1", {"city": "Berlin"})
    assert store.get(NS, "m1").updated_at == first


def test_created_at_survives_an_update_where_the_reference_store_resets_it(store):
    """`InMemoryStore` builds a fresh `Item` on every put, so its `created_at` is really
    "last written" and equals `updated_at` always. Memvara keeps the timeline, so the
    documented meaning of the field is answerable and this returns it."""
    store.put(NS, "m1", {"city": "Berlin"})
    born = store.get(NS, "m1").created_at
    store.put(NS, "m1", {"city": "Lisbon"})
    back = store.get(NS, "m1")
    assert back.created_at == born and back.updated_at > born


def test_a_deleted_and_recreated_key_starts_its_life_again(store):
    """The other half of the rule above: `created_at` is carried forward across updates
    to a *live* item, and an item that ceased to exist did not have its creation
    back-dated to a previous tenant of the same key."""
    store.put(NS, "m1", {"city": "Berlin"})
    born = store.get(NS, "m1").created_at
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", lg.LangGraphDeletionWarning)
        store.delete(NS, "m1")
    store.put(NS, "m1", {"city": "Lisbon"})
    assert store.get(NS, "m1").created_at > born


def test_two_writes_can_never_share_a_transaction_instant(mem, monkeypatch):
    """A supersession whose predecessor was ended at exactly the moment its replacement
    began is a zero-length interval that `as_of` cannot resolve to either value. A coarse
    clock — or a test one — hands out the same instant twice, so the store steps."""
    install(monkeypatch)
    frozen = lg.MemvaraStore(mem, user="alice", clock=lambda: T0)
    frozen.put(NS, "m1", {"city": "Berlin"})
    frozen.put(NS, "m1", {"city": "Lisbon"})
    stamps = [c.recorded_at for c in frozen.history(NS, "m1", "city")]
    assert stamps == [T0, T0 + timedelta(microseconds=1)]
    assert frozen.get(NS, "m1").value == {"city": "Lisbon"}


def test_the_field_name_is_not_a_memvara_predicate_so_it_cannot_fold_onto_one(store, mem):
    """The honest limit of the mapping, pinned so nobody later "fixes" it into data loss.

    Field names stay out of `PredicateRegistry`: it resolves morphologically, so two
    distinct dict keys can fold onto one predicate and cost the item a field. The price
    is that a stored `home_city` does not contradict an extracted `lives_in` — they are
    different slots and both stay live — and that is the trade, stated."""
    mem.remember("user", "lives_in", "Berlin")
    store.put(NS, "m1", {"home_city": "Lisbon"})
    live = {(c.predicate, c.object) for c in mem.get_all()}
    assert ("lives_in", "Berlin") in live and (NOTE_PREDICATE, "Lisbon") in live
    assert mem.registry.known("home_city") is False


def test_a_claim_that_is_not_a_langgraph_item_is_never_returned_as_one(store, mem):
    """One memvara store legitimately holds LangGraph items, CrewAI records, imported
    mem0 notes and ordinary extracted facts — the last three share the `note` predicate.
    Handing a `lives_in` triple back as an `Item` would mean inventing a namespace."""
    mem.remember("user", "lives_in", "Berlin")
    mem.remember("mem0:abc", NOTE_PREDICATE, "Likes pizza", text="Likes pizza")
    store.put(NS, "m1", {"city": "Lisbon"})
    assert keys(store.search(())) == ["m1"]
    assert store.list_namespaces() == [NS]


@pytest.mark.parametrize("claim_kw", [
    {"subject": "langgraph:deadbeef", "predicate": "lives_in"},   # wrong predicate
    {"subject": "someone_else", "predicate": NOTE_PREDICATE},     # wrong subject prefix
])
def test_only_a_claim_with_all_three_of_our_markings_is_read_back(store, mem, claim_kw):
    """Three conditions, not one, and each has to be load-bearing on its own."""
    mem.remember(obj="x", **claim_kw)
    assert store.search(()) == []


def test_a_note_claim_under_our_prefix_with_no_blob_is_still_not_ours(store, mem):
    """The third condition. Nothing writes this shape today, but a store shared with a
    future adapter might — and reading `blob["namespace"]` off it is a `KeyError` at
    read time, in a scan the caller cannot avoid."""
    mem.remember(f"{lg.SUBJECT_PREFIX}deadbeef", NOTE_PREDICATE, "x")
    assert store.search(()) == []


def test_two_stores_on_one_memvara_cannot_see_each_other(mem, monkeypatch, clock):
    """The scope binding is what isolates them, and nothing LangGraph passes in can widen
    it — a namespace is an address inside a store, not a way out of one."""
    install(monkeypatch)
    alice = lg.MemvaraStore(mem, user="alice", clock=clock)
    bob = lg.MemvaraStore(mem, user="bob", clock=clock)
    alice.put(NS, "m1", {"city": "Berlin"})
    bob.put(NS, "m1", {"city": "Lisbon"})
    assert alice.get(NS, "m1").value == {"city": "Berlin"}
    assert bob.get(NS, "m1").value == {"city": "Lisbon"}


# =====================================================================================
# Subject encoding — why the address is hashed
# =====================================================================================


def test_two_different_addresses_that_punctuation_alone_separates_get_different_slots():
    """The bug a readable subject would have shipped. Memvara folds a subject to its
    entity identity before keying a slot, and that fold *drops* punctuation — so
    `langgraph:a/b#c` and `langgraph:a#b/c` are one slot. Namespace ("a","b") key "c"
    and namespace ("a",) key "b c" would supersede each other's data."""
    from memvara.entities import entity_key

    one = lg.field_subject(("a", "b"), "c", "f")
    two = lg.field_subject(("a",), "b c", "f")
    assert one != two
    assert entity_key(one) != entity_key(two)      # the fold keeps them apart too


def test_the_field_is_part_of_the_slot_identity(store):
    """Per-field supersession is exactly this: two fields of one item are two slots."""
    assert lg.field_subject(NS, "m1", "city") != lg.field_subject(NS, "m1", "food")
    assert lg.field_subject(NS, "m1", "city") == lg.field_subject(NS, "m1", "city")


def test_the_readable_address_is_kept_as_data_since_the_subject_is_a_digest(store, mem):
    """A digest is unreadable, so the namespace and key have to survive somewhere or an
    operator reading `get_all()` cannot tell what they are looking at."""
    store.put(NS, "m1", {"city": "Berlin"})
    blob = next(c.meta[lg.LANGGRAPH_META] for c in mem.get_all())
    assert blob["namespace"] == list(NS) and blob["key"] == "m1"
    assert blob["field"] == "city"


def test_user_data_cannot_collide_with_the_adapters_own_bookkeeping(store, mem):
    """One nested blob rather than five flat keys: the item's value is the caller's own
    dict, and any flat scheme is one unlucky field name away from user data deciding
    which namespace an item is filed under."""
    store.put(NS, "m1", {"namespace": "/fake", "key": "hijack", "field": "nope"})
    assert store.get(NS, "m1").value == {"namespace": "/fake", "key": "hijack",
                                         "field": "nope"}
    assert addresses(store.search(())) == [(NS, "m1")]


# =====================================================================================
# The value codec
# =====================================================================================


@pytest.mark.parametrize("value", [
    "pizza", 3, 4.5, True, None, ["x", "y"], {"a": 1, "b": {"c": [2, 3]}}, [],
])
def test_every_json_value_round_trips_as_itself(store, value):
    """`Claim.object` is text, so a type that came back as a string would break the very
    filters this store has to support — `{"n": {"$gt": 100}}` must not match the word."""
    store.put(NS, "m1", {"f": value})
    assert store.get(NS, "m1").value == {"f": value}


def test_a_string_and_the_json_for_it_are_told_apart(store):
    """The reason the codec carries a flag rather than guessing on the way back. `"123"`
    and `123` are the same three bytes on disk; without the marker one becomes the
    other and a numeric filter starts matching text."""
    store.put(NS, "m1", {"a": "123", "b": 123})
    back = store.get(NS, "m1").value
    assert back == {"a": "123", "b": 123}
    assert isinstance(back["a"], str) and isinstance(back["b"], int)


def test_a_string_is_stored_raw_so_the_text_index_stays_readable(store, mem):
    """Not JSON-quoted. `Memvara.search()` over the same store returns claim text to a
    human and to a prompt, and `"pizza"` with quotes around it is noise in both."""
    store.put(NS, "m1", {"food": "pizza"})
    claim = next(c for c in mem.get_all() if lg.LANGGRAPH_META in c.meta)
    assert claim.object == "pizza" and claim.text == "food: pizza"


def test_a_value_that_cannot_be_serialized_stops_the_write_naming_the_field(store):
    """A `repr()` fallback would round-trip as a string and the item would come back
    subtly wrong on exactly the read that matters. The message has to name the field,
    because the caller sees one `put` and the store saw six."""
    with pytest.raises(lg.LangGraphCompatError, match=r"field 'bad'"):
        store.put(NS, "m1", {"ok": "fine", "bad": {1, 2}})


def test_the_codec_is_reachable_on_its_own_because_it_defines_the_on_disk_shape():
    assert lg.encode_value("x") == ("x", False)
    assert lg.encode_value({"b": 1, "a": 2}) == ('{"a":2,"b":1}', True)
    assert lg.decode_value('{"a":2}', True) == {"a": 2}
    assert lg.decode_value("plain", False) == "plain"


# =====================================================================================
# The filter mini-language
# =====================================================================================


@pytest.mark.parametrize("stored, wanted, expected", [
    ("x", "x", True),
    ("x", "y", False),
    (5, {"$eq": 5}, True),
    (5, {"$ne": 5}, False),
    (5, {"$gt": 4}, True),
    (5, {"$gt": 5}, False),
    (5, {"$gte": 5}, True),
    (5, {"$lt": 6}, True),
    (5, {"$lte": 4}, False),
    (5, {"$gte": 4, "$lt": 6}, True),
    (5, {"$gte": 4, "$lt": 5}, False),
    ({"tag": "x"}, {"tag": "x"}, True),
    ({"tag": "x"}, {"tag": "y"}, False),
    ("flat", {"tag": "x"}, False),
    (["a", "b"], ["a", "b"], True),
    (["a", "b"], ["a"], False),
    ("a", ["a"], False),
])
def test_the_filter_language_means_what_the_reference_implementation_means(
        stored, wanted, expected):
    """Mirrors `InMemoryStore._compare_values` case for case. A store that disagreed
    about `{"a": ["x"]}` would break the swap this adapter exists to make painless."""
    assert lg.matches_filter(stored, wanted) is expected


def test_a_missing_field_is_not_greater_than_anything_where_the_reference_raises(store):
    """Measured against langgraph-checkpoint 4.2.0: `_apply_operator` calls `float(None)`
    for an item that lacks the field, so `InMemoryStore.search(filter={"score":
    {"$gt": 1}})` raises `TypeError` and one heterogeneous item makes the whole namespace
    unsearchable. Not matching is the answer every SQL engine gives."""
    store.put(NS, "m1", {"score": 5})
    store.put(NS, "m2", {"text": "no score here"})
    assert keys(store.search(NS, filter={"score": {"$gt": 1}})) == ["m1"]


def test_an_unorderable_value_is_not_a_match_rather_than_an_exception(store):
    """Same rule, reached the other way: a field that exists and is not a number."""
    store.put(NS, "m1", {"score": "not a number"})
    assert store.search(NS, filter={"score": {"$gte": 0}}) == []


def test_an_unknown_operator_is_refused_as_both_a_valueerror_and_an_integrationerror():
    """`InMemoryStore` raises `ValueError`, so an app that already handles a bad filter
    keeps working; and it is an `IntegrationError`, so an app wiring up two frameworks
    catches it with one clause. Being both is what makes the swap non-breaking."""
    with pytest.raises(lg.UnsupportedFilterOperator, match=r"\$eq, \$ne") as excinfo:
        lg.matches_filter(1, {"$regex": ".*"})
    assert isinstance(excinfo.value, ValueError)
    assert isinstance(excinfo.value, IntegrationError)


def test_a_filter_reaches_into_a_nested_value(store):
    """The nested path form. Worth its own test because the value arrives as JSON on
    disk and is decoded before the filter ever sees it — a filter applied to the stored
    text would match nothing here and look like an empty namespace."""
    store.put(NS, "m1", {"meta": {"tag": "x", "n": 2}})
    store.put(NS, "m2", {"meta": {"tag": "y", "n": 2}})
    assert keys(store.search(NS, filter={"meta": {"tag": "x"}})) == ["m1"]
    # Newest-updated first, which is this store's enumeration order for an unranked
    # search: `InMemoryStore` returns dict-insertion order, which does not survive a
    # restart, and `offset=` over an order nothing chose is how paging skips rows.
    assert keys(store.search(NS, filter={"meta": {"n": 2}})) == ["m2", "m1"]


# =====================================================================================
# Search: the filter is exact, only the ordering is budgeted
# =====================================================================================


def test_an_unranked_filtered_search_is_exact_and_says_so(store):
    """The common `InMemoryStore` usage, and the case the CrewAI adapter had to refuse.
    With no query there is no ranking, so there is no budget and nothing to be uncertain
    about — every matching item is here."""
    for i in range(30):
        store.put(NS, f"m{i:02d}", {"kind": "pref" if i % 3 == 0 else "fact", "i": i})
    page = store.search(NS, filter={"kind": "pref"}, limit=100)
    assert len(page) == 10 and page.complete is True
    assert page.scanned == 0                    # nothing was ranked, so nothing was paid


def test_a_ranked_search_answers_from_the_query_text_which_is_the_point(store):
    """The reason this interface was worth building against: `query` is a *string*, so
    memvara's hybrid retrieval runs whole. CrewAI hands over a vector and LlamaIndex's
    vector store hands over a vector; here nothing is lost on the way in."""
    store.put(NS, "m1", {"text": "I live in Berlin"})
    store.put(NS, "m2", {"text": "I prefer pytest over unittest"})
    assert keys(store.search(NS, query="pytest"))[0] == "m2"
    assert keys(store.search(NS, query="Berlin"))[0] == "m1"


def test_every_returned_score_is_on_the_normalized_scale_langgraph_expects(store):
    store.put(NS, "m1", {"text": "I live in Berlin"})
    page = store.search(NS, query="Berlin")
    assert all(0.0 <= i.score <= 1.0 for i in page)


def test_an_item_is_as_relevant_as_its_best_field(store):
    """Max pooling, which is what `InMemoryStore` does across an item's several vectors.
    Summing would make a wide item beat a precise one for having more fields."""
    store.put(NS, "wide", {"a": "unrelated", "b": "also unrelated", "c": "Berlin"})
    store.put(NS, "thin", {"a": "Berlin"})
    page = store.search(NS, query="Berlin")
    assert set(keys(page)) == {"wide", "thin"}
    assert abs(page[0].score - page[1].score) < 0.5    # not inflated by field count


def test_a_selective_filter_that_outruns_the_budget_says_so_instead_of_lying(mem,
                                                                            monkeypatch,
                                                                            clock):
    """The core of it. Post-ranking filters under-fill a limit and the caller cannot tell
    that from "nothing matched" — which is why the CrewAI adapter refuses
    `metadata_filter=` outright. Refusing is not available here, so the page carries the
    answer: `complete` is False exactly when the ranking budget ran out."""
    install(monkeypatch)
    tight = lg.MemvaraStore(mem, user="alice", clock=clock, oversample=1, max_scan=2)
    for i in range(20):
        tight.put(NS, f"noise{i}", {"text": f"alice noise {i}", "kind": "noise"})
    tight.put(NS, "signal", {"text": "alice signal", "kind": "wanted"})
    with pytest.warns(lg.LangGraphRankingWarning, match="oversample"):
        page = tight.search(NS, query="alice", filter={"kind": "wanted"}, limit=1)
    # Exact in membership — the item really is here, and it is the only one that matches.
    assert keys(page) == ["signal"]
    # Bounded in ordering — and that is the bit the caller is told about.
    assert page.complete is False and page[0].score is None


def test_a_bigger_budget_turns_the_same_query_into_a_proven_page(mem, monkeypatch, clock):
    """The other side of the knob, so the flag is shown to track something real rather
    than being permanently pessimistic."""
    install(monkeypatch)
    wide = lg.MemvaraStore(mem, user="alice", clock=clock, oversample=200)
    for i in range(20):
        wide.put(NS, f"noise{i}", {"text": f"alice noise {i}", "kind": "noise"})
    wide.put(NS, "signal", {"text": "alice signal", "kind": "wanted"})
    page = wide.search(NS, query="alice", filter={"kind": "wanted"}, limit=1)
    assert keys(page) == ["signal"] and page.complete is True
    assert page[0].score is not None and page.scanned > 0


def test_the_ranking_warning_fires_once_per_store_not_once_per_search(mem, monkeypatch,
                                                                     clock):
    """A paging loop would otherwise emit one warning per page, get filtered wholesale,
    and take the message with it."""
    install(monkeypatch)
    tight = lg.MemvaraStore(mem, user="alice", clock=clock, oversample=1, max_scan=2)
    for i in range(20):
        tight.put(NS, f"noise{i}", {"text": f"alice noise {i}", "kind": "noise"})
    tight.put(NS, "signal", {"text": "alice signal", "kind": "wanted"})
    with pytest.warns(lg.LangGraphRankingWarning):
        tight.search(NS, query="alice", filter={"kind": "wanted"}, limit=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert tight.search(NS, query="alice", filter={"kind": "wanted"},
                            limit=1).complete is False


def test_the_budget_doubles_rather_than_ranking_the_whole_corpus_up_front(mem,
                                                                         monkeypatch,
                                                                         clock):
    """A filter that barely thins anything is the common case, and it must not pay for a
    full-corpus rank. The escalation is what lets the default budget be small."""
    install(monkeypatch)
    small = lg.MemvaraStore(mem, user="alice", clock=clock, oversample=1, max_scan=500)
    for i in range(12):
        small.put(NS, f"m{i:02d}", {"text": f"alice fact {i}",
                                    "kind": "wanted" if i == 11 else "noise"})
    page = small.search(NS, query="alice", filter={"kind": "wanted"}, limit=1)
    assert keys(page) == ["m11"] and page.complete is True
    # It escalated past the opening budget of 1 rather than giving up at it.
    assert page.scanned > 1


def test_a_full_page_of_scored_items_is_proven_even_though_the_ranker_had_more(store):
    """The completeness argument, on its own. `search(k=budget)` returns claims in
    descending score, so anything it did not return scores no higher than the last thing
    it did — which makes a page filled with real scores provably the true top-N even
    when the scan stopped early."""
    for i in range(30):
        store.put(NS, f"m{i:02d}", {"text": f"alice fact number {i}"})
    page = store.search(NS, query="alice fact", limit=2)
    assert len(page) == 2 and page.complete is True
    assert all(i.score is not None for i in page)


def test_unrelated_claims_in_the_same_scope_consume_the_ranking_budget(store, mem):
    """Honest about a cost that is easy to miss: memvara ranks over everything visible at
    the scope, so a store shared with extracted conversation facts has less of its budget
    left for items. The dilution is real, which is why the budget is a knob."""
    for i in range(40):
        mem.remember(f"person{i}", "lives_in", f"city {i}")
    store.put(NS, "m1", {"text": "alice lives in Berlin"})
    page = store.search(NS, query="lives in", limit=1)
    assert keys(page) == ["m1"]


def test_offset_and_limit_page_through_a_ranked_search(store):
    store.put(NS, "m1", {"text": "alpha beta"})
    store.put(NS, "m2", {"text": "alpha gamma"})
    store.put(NS, "m3", {"text": "alpha delta"})
    everything = keys(store.search(NS, query="alpha", limit=10))
    assert len(everything) == 3
    assert keys(store.search(NS, query="alpha", limit=1, offset=1)) == [everything[1]]


def test_a_zero_limit_search_returns_nothing_and_ranks_nothing(store):
    """`need` is zero, so there is no page to prove and no reason to pay a ranker."""
    store.put(NS, "m1", {"text": "alpha"})
    page = store.search(NS, query="alpha", limit=0)
    assert page == [] and page.complete is True and page.scanned == 0


def test_a_query_that_matches_no_item_in_the_namespace_ranks_nothing(store):
    """`wanted` is empty, so the ranker is never called: the filter already answered."""
    store.put(("other",), "m1", {"text": "alpha"})
    page = store.search(NS, query="alpha")
    assert page == [] and page.complete is True and page.scanned == 0


def test_a_namespace_prefix_matches_whole_segments_not_string_prefixes(store):
    """`("acme",)` must not reach into `("acmecorp",)`. Comparing joined strings would
    hand one tenant's items to another and look perfectly fine doing it."""
    store.put(("acme", "eng"), "ours", {"text": "ours"})
    store.put(("acmecorp",), "theirs", {"text": "theirs"})
    assert keys(store.search(("acme",))) == ["ours"]
    assert len(store.search(())) == 2


def test_unranked_matches_fill_the_page_rather_than_being_dropped(mem, monkeypatch,
                                                                 clock):
    """They match the filter, so withholding them would be the under-fill this design
    exists to avoid. `score=None` is `InMemoryStore`'s own answer for an item it holds no
    vector for, and it is the signal that the ordering, not the membership, is a guess."""
    install(monkeypatch)
    tight = lg.MemvaraStore(mem, user="alice", clock=clock, oversample=1, max_scan=1)
    for i in range(6):
        tight.put(NS, f"m{i}", {"text": f"alice fact {i}"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", lg.LangGraphRankingWarning)
        page = tight.search(NS, query="alice", limit=6)
    assert len(page) == 6 and page.complete is False
    assert page[0].score is not None and page[-1].score is None


def test_a_search_page_is_a_list_so_every_langgraph_caller_keeps_working(store):
    """The flag rides on the return value, which means the return value has to still be
    the `list[SearchItem]` `BaseStore` promises — iterated, indexed and len()'d by code
    that has never heard of this adapter."""
    store.put(NS, "m1", {"text": "alpha"})
    page = store.search(NS)
    assert isinstance(page, list) and len(page) == 1
    assert [i.key for i in page] == ["m1"] and page[0].key == "m1"
    assert isinstance(page[:1], list) and not isinstance(page[:1], lg.SearchPage)


# =====================================================================================
# index=: what reaches the retrieval index
# =====================================================================================


def test_index_false_keeps_the_value_out_of_the_index_entirely(store, mem):
    """Stronger than ranking it last. A blob of raw tool output marked `index=False`
    never enters the text index or the vector index, so nothing memvara can retrieve —
    including `Memvara.search()` on the same store — will surface it. `get()` still
    returns it verbatim, which is the whole contract of the flag."""
    store.put(NS, "m1", {"secret": "hunter2 correct horse"}, index=False)
    assert store.get(NS, "m1").value == {"secret": "hunter2 correct horse"}
    claim = next(c for c in mem.get_all() if lg.LANGGRAPH_META in c.meta)
    assert claim.text == "secret" and claim.object == "hunter2 correct horse"
    assert lexical_hits(mem, "hunter2 correct horse") == []
    assert lexical_hits(mem, "secret") == ["secret"]      # the name still is indexed


def test_index_none_indexes_every_field_because_that_is_the_store_default(store, mem):
    """`IndexConfig.fields` defaults to `["$"]` — the whole object — so the default for
    an item is that all of it is searchable."""
    store.put(NS, "m1", {"a": "Berlin", "b": "pizza"})
    assert {c.text for c in mem.get_all()} == {"a: Berlin", "b: pizza"}


def test_a_list_of_paths_indexes_only_those_and_uses_langgraphs_own_parser(store, mem):
    """`"context[*].content"` is LangGraph's path language, and it has an owner. Parsing
    it here with `tokenize_path`/`get_text_at_path` is what stops the adapter and the
    framework drifting apart about what a path means."""
    store.put(NS, "m1",
              {"memory": "Will likes ai", "meta": {"title": "T", "n": 3},
               "context": [{"content": "first"}, {"content": "second"}]},
              index=["meta.title", "context[*].content"])
    texts = {c.meta[lg.LANGGRAPH_META]["field"]: c.text for c in mem.get_all()}
    assert texts["meta"] == "meta: T"                       # the nested leaf, not the blob
    assert texts["context"] == "context: first second"      # every element
    assert texts["memory"] == "memory"                      # not named, so not indexed
    assert lexical_hits(mem, "Will likes ai") == []


def test_the_dollar_path_is_expanded_rather_than_silently_indexing_nothing(store, mem):
    """Measured: `get_text_at_path(value, ["$"])` is empty, because only the *bare
    string* `"$"` is the whole-object sentinel and `tokenize_path` turns it into a list.
    An adapter that passed the tokens straight through would index nothing and look like
    it had worked."""
    store.put(NS, "m1", {"a": "Berlin", "b": "pizza"}, index=["$"])
    assert {c.text for c in mem.get_all()} == {"a: Berlin", "b: pizza"}


def test_a_path_that_names_nothing_is_skipped_rather_than_indexing_a_blank(store, mem):
    store.put(NS, "m1", {"a": "Berlin"}, index=["", "nope", "a"])
    assert {c.text for c in mem.get_all()} == {"a: Berlin"}


def test_an_empty_index_list_is_not_the_same_as_no_index_argument(store, mem):
    """`[]` is "index these zero paths", `None` is "use the store default". `index=[]`
    would be `False` if the two were conflated."""
    store.put(NS, "m1", {"a": "Berlin"}, index=[])
    assert {c.text for c in mem.get_all()} == {"a"}


def test_whether_a_field_was_indexed_is_recorded_so_a_later_read_can_tell(store, mem):
    store.put(NS, "m1", {"a": "x", "b": "y"}, index=["a"])
    flags = {c.meta[lg.LANGGRAPH_META]["field"]: c.meta[lg.LANGGRAPH_META]["indexed"]
             for c in mem.get_all()}
    assert flags == {"a": True, "b": False}


# =====================================================================================
# list_namespaces
# =====================================================================================


@pytest.fixture()
def populated(store):
    for namespace in [("a", "b", "c"), ("a", "b", "d", "e"), ("a", "b", "d", "i"),
                      ("a", "b", "f"), ("a", "c", "f")]:
        store.put(namespace, "k", {"text": "x"})
    return store


def test_max_depth_truncates_and_deduplicates_exactly_as_the_docstring_promises(
        populated):
    """The example out of `BaseStore.list_namespaces`' own docstring, run."""
    assert populated.list_namespaces(prefix=("a", "b"), max_depth=3) == [
        ("a", "b", "c"), ("a", "b", "d"), ("a", "b", "f")]


def test_prefix_and_suffix_conditions_both_take_wildcards(populated):
    assert populated.list_namespaces(prefix=("a", "*", "f")) == [("a", "b", "f"),
                                                                 ("a", "c", "f")]
    assert populated.list_namespaces(suffix=("*", "f")) == [("a", "b", "f"),
                                                            ("a", "c", "f")]


def test_prefix_and_suffix_together_must_both_hold(populated):
    assert populated.list_namespaces(prefix=("a", "c"), suffix=("f",)) == [("a", "c", "f")]


def test_a_condition_longer_than_the_namespace_never_matches(populated):
    """`InMemoryStore._does_match`'s first rule, and the reason `suffix=("a","b")` does
    not match `("b",)` — a suffix is not a substring."""
    assert lg.matches_path(("b",), ("a", "b"), suffix=True) is False
    assert populated.list_namespaces(prefix=("a", "b", "c", "d", "e")) == []


def test_namespaces_page(populated):
    everything = populated.list_namespaces()
    assert len(everything) == 5
    assert populated.list_namespaces(limit=2, offset=1) == everything[1:3]


def test_an_emptied_namespace_stops_being_listed(store):
    """`InMemoryStore` keeps listing it: deleting an item pops the key and leaves the
    namespace's empty dict behind in a `defaultdict`. That is an artefact of its storage
    rather than a decision, and it makes `list_namespaces()` report places nothing lives."""
    store.put(("gone",), "k", {"text": "x"})
    store.put(("stays",), "k", {"text": "x"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", lg.LangGraphDeletionWarning)
        store.delete(("gone",), "k")
    assert store.list_namespaces() == [("stays",)]


# =====================================================================================
# get, delete, and the retirement policy
# =====================================================================================


def test_an_unknown_key_is_none_rather_than_an_error(store):
    store.put(NS, "m1", {"a": "x"})
    assert store.get(NS, "nope") is None
    assert store.get(("elsewhere",), "m1") is None


def test_deleting_retires_by_default_and_says_so_once(store, mem):
    """Retirement is right for a graph replacing its own state and wrong for "delete my
    data", so the default must not quietly under-serve the second reading."""
    store.put(NS, "m1", {"city": "Berlin"})
    store.put(NS, "m2", {"city": "Lisbon"})
    with pytest.warns(lg.LangGraphDeletionWarning, match="on_delete='erase'"):
        store.delete(NS, "m1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # once per store, not per item
        store.delete(NS, "m2")
    assert store.get(NS, "m1") is None and store.search(()) == []
    assert len(store.history(NS, "m1", "city")) == 1     # the trail survives


def test_deleting_nothing_does_not_warn_about_a_retirement_that_did_not_happen(store):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store.delete(NS, "never-existed")


def test_erase_mode_removes_the_text_rather_than_retiring_it(mem, monkeypatch, clock):
    """The other reading of delete, for the caller who has decided. Nothing is left for
    `history()` or `as_of` to find."""
    install(monkeypatch)
    hard = lg.MemvaraStore(mem, user="alice", clock=clock, on_delete="erase")
    hard.put(NS, "m1", {"city": "Berlin"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # an informed choice does not lecture
        hard.delete(NS, "m1")
    assert hard.get(NS, "m1") is None
    assert hard.history(NS, "m1", "city") == []
    assert mem.count() == 0


def test_retire_mode_is_the_informed_choice_and_stays_silent(mem, monkeypatch, clock):
    install(monkeypatch)
    quiet = lg.MemvaraStore(mem, user="alice", clock=clock, on_delete="retire")
    quiet.put(NS, "m1", {"city": "Berlin"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet.delete(NS, "m1")
    assert quiet.get(NS, "m1") is None and len(quiet.history(NS, "m1", "city")) == 1


def test_a_put_that_drops_a_field_retires_it_even_in_erase_mode(mem, monkeypatch, clock):
    """`on_delete` answers "what does delete() mean", not "what does an update mean". A
    put is an update, and erasing the field's history because the caller once asked for
    hard deletes would destroy the audit trail as a side effect of a write."""
    install(monkeypatch)
    hard = lg.MemvaraStore(mem, user="alice", clock=clock, on_delete="erase")
    hard.put(NS, "m1", {"city": "Berlin", "food": "pizza"})
    hard.put(NS, "m1", {"city": "Berlin"})
    assert hard.get(NS, "m1").value == {"city": "Berlin"}
    assert len(hard.history(NS, "m1", "food")) == 1


def test_an_unknown_on_delete_is_rejected_at_construction(mem, monkeypatch):
    install(monkeypatch)
    with pytest.raises(ValueError, match="on_delete='wipe'"):
        lg.MemvaraStore(mem, user="alice", on_delete="wipe")


def test_the_store_repr_names_the_scope_and_the_deletion_policy(store):
    assert "default/alice/*/*" in repr(store) and "on_delete=warn" in repr(store)


# =====================================================================================
# TTL
# =====================================================================================


def test_put_with_a_ttl_is_refused_by_the_base_class_for_free(store):
    """`supports_ttl = False` is `BaseStore`'s default and this store keeps it, so the
    refusal costs no code — but it has to actually happen, and a subclass that set the
    flag to be helpful would start silently dropping retention policies."""
    with pytest.raises(NotImplementedError, match="TTL is not supported"):
        store.put(NS, "m1", {"a": "x"}, ttl=5.0)


def test_a_ttl_smuggled_straight_into_batch_is_refused_rather_than_ignored(store):
    """`batch()` skips the base class's check, and it is a public method LangGraph's own
    batching layer calls. A store that accepted a retention policy and did not enforce it
    is worse than one that has none, because nothing ever fails."""
    with pytest.raises(lg.LangGraphCompatError, match="does not implement expiry"):
        store.batch([FakePutOp(NS, "m1", {"a": "x"}, ttl=5.0)])
    assert store.get(NS, "m1") is None       # and it did not half-write the item


def test_refresh_ttl_is_accepted_and_ignored_because_there_is_no_ttl_to_refresh(store):
    """The framework's own docstring says the argument is ignored when no TTL was set,
    and none ever is here — so this is compliance rather than a silent drop."""
    store.put(NS, "m1", {"a": "x"})
    assert store.get(NS, "m1", refresh_ttl=True).value == {"a": "x"}
    assert keys(store.search(NS, refresh_ttl=False)) == ["m1"]


# =====================================================================================
# batch()
# =====================================================================================


def test_batch_returns_one_result_per_op_in_order(store):
    store.put(NS, "m1", {"a": "x"})
    results = store.batch([
        FakeGetOp(NS, "m1"),
        FakeSearchOp(NS),
        FakeListNamespacesOp(),
        FakePutOp(NS, "m2", {"a": "y"}),
    ])
    assert results[0].value == {"a": "x"}
    assert keys(results[1]) == ["m1"]
    assert results[2] == [NS]
    assert results[3] is None


def test_reads_in_a_batch_see_the_state_before_it_and_writes_land_after(store):
    """`InMemoryStore` collects puts and applies them last, so a get beside a put in one
    batch returns the old value. LangGraph's own batching layer is written against that
    ordering, so an adapter that ordered it the intuitive way would change behaviour
    under a graph that never called `batch()` directly."""
    store.put(NS, "m1", {"a": "old"})
    results = store.batch([FakePutOp(NS, "m1", {"a": "new"}), FakeGetOp(NS, "m1")])
    assert results[1].value == {"a": "old"}
    assert store.get(NS, "m1").value == {"a": "new"}


def test_two_puts_to_one_address_in_one_batch_collapse_to_the_last(store):
    """The reference dedupes by `(namespace, key)`. Applying both would put a version in
    the history that the store never held for any observable instant."""
    store.batch([FakePutOp(NS, "m1", {"a": "first"}), FakePutOp(NS, "m1", {"a": "last"})])
    assert store.get(NS, "m1").value == {"a": "last"}
    assert len(store.history(NS, "m1", "a")) == 1


def test_a_whole_put_is_one_memvara_transaction(store, mem):
    """A five-field item is five asserts plus its retirements, and separately committed a
    crash in the middle leaves an item that is half its old value and half its new one —
    with no way to tell, because both halves look live."""
    depth = []
    real = mem.store.batch

    def watched():
        depth.append(mem.store._batch_depth)
        return real()

    store.put(NS, "m1", {"a": "1", "b": "2", "c": "3"})
    mem.store.batch = watched
    store.put(NS, "m1", {"a": "9", "b": "8"})
    # The outermost `batch()` the adapter opens is at depth 0; every write inside nests.
    assert depth[0] == 0 and max(depth) > 0


def test_a_store_that_predates_batch_still_works(mem, monkeypatch, clock):
    """`getattr(store, "batch", None)` is the codebase's idiom for an optional `Store`
    capability, and a third-party store without it must not lose the write."""
    class WithoutBatch:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "batch":
                raise AttributeError(name)
            return getattr(self._inner, name)

    install(monkeypatch)
    mem.store = WithoutBatch(mem.store)
    plain = lg.MemvaraStore(mem, user="alice", clock=clock)
    plain.put(NS, "m1", {"a": "x"})
    assert plain.get(NS, "m1").value == {"a": "x"}


def test_a_read_only_batch_costs_no_clock_read_and_no_transaction(store):
    """No puts means no instant, which is what keeps `get()` from advancing an injected
    clock and making every test's timestamps depend on how many reads ran before them."""
    store.put(NS, "m1", {"a": "x"})
    before = store._last_instant
    store.batch([FakeGetOp(NS, "m1"), FakeSearchOp(NS)])
    assert store._last_instant == before


def test_an_unknown_op_is_rejected_in_the_reference_implementations_own_words(store):
    """An application matching on that message keeps matching."""
    with pytest.raises(ValueError, match="Unknown operation type"):
        store.batch([object()])


# =====================================================================================
# async
# =====================================================================================


def test_the_async_half_is_the_sync_half_off_the_loop_thread(store):
    """LangGraph awaits the store on the hot path of every node that touches memory, so
    running a synchronous encode and a SQLite write straight from `abatch` would block
    the loop for exactly as long as the write takes."""
    run(store.aput(NS, "m1", {"text": "async works"}))
    assert run(store.aget(NS, "m1")).value == {"text": "async works"}
    assert keys(run(store.asearch(NS, query="async"))) == ["m1"]
    assert run(store.alist_namespaces()) == [NS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", lg.LangGraphDeletionWarning)
        run(store.adelete(NS, "m1"))
    assert run(store.aget(NS, "m1")) is None


# =====================================================================================
# The escape hatches, and the guarantees that do not fit through a SearchItem
# =====================================================================================


def test_search_memory_is_where_the_provenance_lives(store):
    """A `SearchItem` has nowhere to put the triple, the two time axes, the ranking
    explanation or the source turn ids — `value` is the caller's dict and writing into it
    would corrupt the round trip. So they live here, as in every other adapter."""
    store.put(NS, "m1", {"text": "I live in Berlin"})
    result = store.search_memory("Berlin")[0]
    assert result.claim.object == "I live in Berlin"
    assert result.explain.summary() and result.claim.recorded_at == T0


def test_why_resolves_the_claim_and_has_no_source_turn_to_show(store, mem):
    """Stated rather than left to be discovered. Nothing was *said* — a graph asserted
    the value — so there is no episode, and `sources=` would have had to invent one."""
    store.put(NS, "m1", {"text": "I live in Berlin"})
    claim = next(c for c in mem.get_all() if lg.LANGGRAPH_META in c.meta)
    provenance = mem.why(claim.id)
    assert provenance.episodes == [] and provenance.extractor == lg.EXTRACTOR


def test_history_is_the_thing_no_other_key_value_store_can_answer(store):
    """`get()` is the current value; this is the timeline, each entry carrying when it
    held, when it stopped holding and what replaced it.

    Every superseded field reads `ended`: a `put` says the value changed, which is a
    world event, and LangGraph has no way to express "the previous value was a mistake".
    `_remove` is the call that stops belief, and it is a different call."""
    for city in ("Berlin", "Lisbon", "Porto"):
        store.put(NS, "m1", {"city": city})
    timeline = store.history(NS, "m1", "city")
    assert [c.object for c in timeline] == ["Berlin", "Lisbon", "Porto"]
    assert [c.state for c in timeline] == ["ended", "ended", "live"]


# =====================================================================================
# Lazy import — the property the whole suite rests on
# =====================================================================================


def test_naming_the_class_without_the_sdk_names_the_distribution_to_install():
    """And it has to name `langgraph-checkpoint`, not `langgraph`. Measured: the
    `langgraph` wheel contains no `langgraph/store/` at all — it depends on
    `langgraph-checkpoint>=4.1,<5`, which is what actually ships `langgraph.store.base`.
    Someone who reads "install langgraph", does exactly that, and still has no
    `langgraph.store` has lost an afternoon to a correct-sounding error message."""
    with pytest.raises(ImportError, match=r"memvara\[langgraph\]") as excinfo:
        lg.MemvaraStore
    assert "langgraph-checkpoint" in str(excinfo.value)


def test_a_present_package_missing_a_name_is_reported_as_version_skew(monkeypatch):
    """Not as "not installed". `require()` tells the two apart and this adapter asks for
    nine names, so a skew is the more likely of the two failures."""
    from types import SimpleNamespace
    partial = {k: v for k, v in FAKE_MODULE.items() if k != "SearchItem"}
    monkeypatch.setitem(sys.modules, "langgraph.store.base",
                        SimpleNamespace(**partial))
    with pytest.raises(ImportError, match="version skew"):
        lg.MemvaraStore


def test_the_module_names_nothing_else(monkeypatch):
    install(monkeypatch)
    with pytest.raises(AttributeError, match="NotAThing"):
        lg.NotAThing


def test_the_adapter_imports_with_numpy_alone():
    """The CI assertion, run here too so it fails in a second rather than in a workflow.
    Importing this module must not import langgraph-checkpoint, however convenient."""
    import importlib

    monkeypatched = sys.modules.pop("langgraph.store.base", None)
    assert monkeypatched is None
    importlib.reload(lg)
    assert "langgraph" not in sys.modules
    assert np.__name__ == "numpy"
