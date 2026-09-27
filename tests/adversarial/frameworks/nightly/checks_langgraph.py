"""LangGraph: `MemvaraStore`, against the real langgraph-checkpoint and langgraph.

Each `check_*` function runs inside a virtual environment that holds langgraph-checkpoint
at the floor memvara declares or at the newest release, with the newest `langgraph` pip
finds compatible beside it (see `probe.py`). The promises they check are the ones the
adapter's docstrings and `docs/integrations/frameworks.md` make. Several compare the
store with langgraph's own `InMemoryStore`, the reference implementation of the same
interface. LangGraph is imported inside each check, never at module level, because the
suite imports this module to list the checks and has no LangGraph installed.
"""

from __future__ import annotations

import asyncio
import importlib
import warnings
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Iterator, TypedDict

from memvara.integrations import langgraph as adapter

if TYPE_CHECKING:
    from ..probe import Context

NS = ("memories", "alice")
T0 = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)

#: Items that give every filter operator and namespace condition below something to
#: tell apart: a field an item lacks, a number written as a string, a list, a nested
#: object, and a namespace whose name another one starts with.
ITEMS = [
    (("docs", "alice"), "a", {"kind": "pref", "score": 5, "tags": ["x", "y"],
                              "meta": {"lang": "en"}}),
    (("docs", "alice"), "b", {"kind": "fact", "score": 2, "tags": ["x"],
                              "meta": {"lang": "de"}}),
    (("docs", "bob"), "c", {"kind": "pref", "score": "10", "tags": [],
                            "meta": {"lang": "en"}}),
    (("docs", "bob", "notes"), "d", {"kind": "note", "score": 7.5, "flag": True}),
    (("other",), "e", {"kind": "pref", "score": 1}),
    (("docsx",), "f", {"kind": "pref", "score": 3}),
]


class _State(TypedDict):
    """The state of the two-node graph in the compile check."""

    text: str
    answer: str


def _lg(module: str) -> Any:
    return importlib.import_module(module)


def _ticks() -> Iterator[datetime]:
    moment = T0
    while True:
        yield moment
        moment += timedelta(minutes=1)


def _store(ctx: Context, **options: Any) -> Any:
    """A store whose clock moves a minute per write, so no two writes share an instant."""
    ticks = _ticks()
    options.setdefault("clock", lambda: next(ticks))
    return adapter.MemvaraStore(ctx.memvara(), user="alice", **options)


def _load(store: Any) -> None:
    for namespace, key, value in ITEMS:
        store.put(namespace, key, value)


def check_the_store_is_a_real_basestore_without_ttl(ctx: Context) -> None:
    """The composed class subclasses langgraph's BaseStore, is made once, and declares no
    TTL support, because memvara retires and erases and neither of those is expiry."""
    store = _store(ctx)
    assert isinstance(store, _lg("langgraph.store.base").BaseStore)
    assert adapter.MemvaraStore is adapter.MemvaraStore
    assert type(store).supports_ttl is False


def check_every_json_value_round_trips_through_put_and_get(ctx: Context) -> None:
    """Each JSON type comes back as itself, and the string "123" stays apart from the
    number 123."""
    store = _store(ctx)
    value = {"text": "Berlin", "number": 123, "digits": "123", "real": 2.5, "yes": True,
             "nothing": None, "list": [1, "two", [3]], "nested": {"a": {"b": [True]}}}
    store.put(NS, "k", value)
    assert store.get(NS, "k").value == value


def check_a_replaced_field_ends_and_a_dropped_field_is_retired(ctx: Context) -> None:
    """A put that changes one field ends that field's old value: its world clock closes,
    its belief clock stays open, and the row stays. A field the new value leaves out is
    retired: its belief clock closes and its world clock stays open. Each end moves
    exactly one clock. The ended value points at the value that replaced it, and the
    retired one points at nothing, because nothing replaced it."""
    store = _store(ctx)
    store.put(NS, "profile", {"city": "Berlin", "food": "pizza", "pet": "cat"})
    store.put(NS, "profile", {"city": "Lisbon", "food": "pizza"})
    city = store.history(NS, "profile", "city")
    assert [(c.object, c.state) for c in city] == [("Berlin", "ended"), ("Lisbon", "live")]
    assert city[0].valid_to is not None and city[0].invalidated_at is None
    assert city[0].invalidated_by == city[1].id, city
    pet = store.history(NS, "profile", "pet")
    assert [(c.object, c.state) for c in pet] == [("cat", "retired")]
    assert pet[0].invalidated_at is not None and pet[0].valid_to is None
    assert pet[0].invalidated_by is None, pet
    assert store.get(NS, "profile").value == {"city": "Lisbon", "food": "pizza"}


def check_an_unchanged_field_is_not_rewritten(ctx: Context) -> None:
    """Putting the same value again is a re-observation: the field keeps one claim and
    updated_at does not move, while created_at survives a real change."""
    store = _store(ctx)
    store.put(NS, "profile", {"food": "pizza"})
    first = store.get(NS, "profile")
    store.put(NS, "profile", {"food": "pizza"})
    again = store.get(NS, "profile")
    assert len(store.history(NS, "profile", "food")) == 1
    assert again.updated_at == first.updated_at
    store.put(NS, "profile", {"food": "ramen"})
    changed = store.get(NS, "profile")
    assert changed.created_at == first.created_at < changed.updated_at


def check_delete_retires_by_default_and_warns_once(ctx: Context) -> None:
    """delete() retires the item's fields, which history() still reaches, and warns once
    per store that it did not erase them."""
    store = _store(ctx)
    store.put(NS, "a", {"v": "one"})
    store.put(NS, "b", {"v": "two"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store.delete(NS, "a")
        store.delete(NS, "b")
    ours = [w for w in caught if w.category is adapter.LangGraphDeletionWarning]
    assert len(ours) == 1, [str(w.message) for w in caught]
    assert store.get(NS, "a") is None
    assert [c.state for c in store.history(NS, "a", "v")] == ["retired"]


def check_erase_mode_removes_the_items_text(ctx: Context) -> None:
    """With on_delete="erase", deleting an item erases its fields' claims, and the store
    can prove that no row, index entry or vector of them is left."""
    store = _store(ctx, on_delete="erase")
    store.put(NS, "a", {"v": "one", "w": "two"})
    ids = [c.id for field in ("v", "w") for c in store.history(NS, "a", field)]
    store.delete(NS, "a")
    assert store.get(NS, "a") is None
    for claim_id in ids:
        proof = store.memory.prove_erased(claim_id)
        assert proof.proven, proof


def check_search_ranks_by_the_query_text_on_the_normalised_scale(ctx: Context) -> None:
    """search(query=...) answers from the query text, puts the relevant item first, gives
    scores in [0, 1], and returns a SearchPage that says its ranking was complete."""
    store = _store(ctx)
    store.put(NS, "home", {"city": "Lisbon"})
    store.put(NS, "food", {"dish": "ramen"})
    page = store.search(("memories",), query="Lisbon")
    assert isinstance(page, adapter.SearchPage) and page.complete is True
    assert page[0].key == "home", [(i.key, i.score) for i in page]
    assert all(0.0 <= item.score <= 1.0 for item in page)


def check_filters_select_what_inmemorystore_selects(ctx: Context) -> None:
    """Every filter operator, over nested values and lists, selects exactly the items
    langgraph's own InMemoryStore selects. The results are compared as sets, because the
    adapter documents newest-first order for a search with no query and InMemoryStore
    keeps the order items were written in."""
    reference = _lg("langgraph.store.memory").InMemoryStore()
    store = _store(ctx)
    _load(reference)
    _load(store)
    filters: list[dict[str, Any]] = [
        {"kind": "pref"}, {"kind": {"$eq": "pref"}}, {"kind": {"$ne": "pref"}},
        {"score": {"$gt": 4}}, {"score": {"$gte": 5}}, {"score": {"$lt": 3}},
        {"score": {"$lte": 2}}, {"tags": ["x", "y"]}, {"tags": ["x"]},
        {"meta": {"lang": "en"}}, {"flag": True}, {"kind": "pref", "score": {"$gt": 2}},
    ]
    differences = []
    for criteria in filters:
        for prefix in [(), ("docs",), ("docs", "alice")]:
            ours = sorted((i.namespace, i.key)
                          for i in store.search(prefix, filter=criteria, limit=50))
            theirs = sorted((i.namespace, i.key)
                            for i in reference.search(prefix, filter=criteria, limit=50))
            if ours != theirs:
                differences.append((prefix, criteria, ours, theirs))
    assert differences == [], differences


def check_list_namespaces_answers_what_inmemorystore_answers(ctx: Context) -> None:
    """Prefix and suffix conditions with wildcards, max_depth, limit and offset list the
    same namespaces as langgraph's own InMemoryStore."""
    reference = _lg("langgraph.store.memory").InMemoryStore()
    store = _store(ctx)
    _load(reference)
    _load(store)
    questions: list[dict[str, Any]] = [
        {}, {"prefix": ("docs",)}, {"suffix": ("notes",)}, {"prefix": ("docs", "*")},
        {"suffix": ("*",)}, {"max_depth": 1}, {"max_depth": 2},
        {"prefix": ("docs",), "max_depth": 2}, {"limit": 2}, {"limit": 2, "offset": 2}]
    differences = [(q, store.list_namespaces(**q), reference.list_namespaces(**q))
                   for q in questions
                   if store.list_namespaces(**q) != reference.list_namespaces(**q)]
    assert differences == [], differences


def check_the_documented_differences_from_inmemorystore_hold(ctx: Context) -> None:
    """The three places the adapter says it departs from InMemoryStore still hold against
    the real one. InMemoryStore resets created_at on every put and memvara keeps it. A
    filter on a field an item lacks makes InMemoryStore raise TypeError, where memvara
    treats the item as not matching. A namespace whose last item was deleted is still
    listed by InMemoryStore and not by memvara."""
    reference = _lg("langgraph.store.memory").InMemoryStore()
    store = _store(ctx)
    for target in (reference, store):
        target.put(NS, "k", {"v": 1})
    first_theirs, first_ours = reference.get(NS, "k"), store.get(NS, "k")
    for target in (reference, store):
        target.put(NS, "k", {"v": 2})
    theirs, ours = reference.get(NS, "k"), store.get(NS, "k")
    assert theirs.created_at > first_theirs.created_at, "InMemoryStore now keeps created_at"
    assert ours.created_at == first_ours.created_at < ours.updated_at
    for target in (reference, store):
        target.put(NS, "without", {"other": True})
    try:
        reference.search(NS, filter={"v": {"$gt": 1}})
    except TypeError:
        pass
    else:
        raise AssertionError("InMemoryStore no longer raises on a field an item lacks")
    assert [i.key for i in store.search(NS, filter={"v": {"$gt": 1}})] == ["k"]
    lonely = ("lonely",)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for target in (reference, store):
            target.put(lonely, "only", {"v": 1})
            target.delete(lonely, "only")
    assert lonely in reference.list_namespaces()
    assert lonely not in store.list_namespaces()


def check_ttl_is_refused_by_put_and_by_batch(ctx: Context) -> None:
    """put(ttl=...) is refused by langgraph's own base class, and a PutOp carrying a ttl
    handed straight to batch() is refused by the adapter rather than ignored."""
    base = _lg("langgraph.store.base")
    store = _store(ctx)
    try:
        store.put(NS, "k", {"v": 1}, ttl=5)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("put(ttl=5) was accepted")
    try:
        store.batch([base.PutOp(NS, "k", {"v": 1}, ttl=5)])
    except adapter.LangGraphCompatError as exc:
        assert "ttl=5" in str(exc), str(exc)
    else:
        raise AssertionError("a PutOp carrying a ttl was accepted by batch()")
    assert store.get(NS, "k") is None


def check_index_paths_are_parsed_by_langgraph_itself(ctx: Context) -> None:
    """index= takes LangGraph's own path language: only the texts the paths name become
    searchable, index=False keeps a value out of the index, and get() returns the whole
    value either way."""
    store = _store(ctx)
    value = {"context": [{"content": "kayaking trip"}, {"content": "ramen night"}],
             "secret": "zebra crossing"}
    store.put(NS, "doc", value, index=["context[*].content"])
    store.put(NS, "raw", {"blob": "zebra crossing"}, index=False)
    assert store.get(NS, "doc").value == value
    assert [i.key for i in store.search(NS, query="kayaking")][:1] == ["doc"]
    indexed = [result.claim.text for result in store.search_memory("zebra crossing", k=10)]
    assert all("zebra" not in text for text in indexed), indexed


def check_the_async_methods_answer_like_the_sync_ones(ctx: Context) -> None:
    """aput, aget, asearch, alist_namespaces and adelete go through abatch and answer as
    their synchronous twins do."""
    store = _store(ctx, on_delete="retire")

    async def scenario() -> None:
        await store.aput(NS, "k", {"v": "one"})
        assert (await store.aget(NS, "k")).value == {"v": "one"}
        assert [i.key for i in await store.asearch(NS, query="one")] == ["k"]
        assert await store.alist_namespaces() == [NS]
        await store.adelete(NS, "k")
        assert await store.aget(NS, "k") is None

    asyncio.run(scenario())


def check_a_compiled_graph_reads_and_writes_through_the_store(ctx: Context) -> None:
    """The docstring's `builder.compile(store=store)`: a node that writes through the
    store and a node that searches it both work, under invoke and ainvoke. The nodes reach
    the store through `langgraph.config.get_store()`, LangGraph's own accessor."""
    graph_module = _lg("langgraph.graph")
    get_store = _lg("langgraph.config").get_store
    store = _store(ctx)

    def remember(state: _State) -> dict[str, Any]:
        get_store().put(NS, "fact", {"text": state["text"]})
        return {}

    def answer(state: _State) -> dict[str, Any]:
        found = get_store().search(NS, query=state["text"])
        return {"answer": found[0].value["text"] if found else ""}

    builder = graph_module.StateGraph(_State)
    builder.add_node("remember", remember)
    builder.add_node("answer", answer)
    builder.add_edge(graph_module.START, "remember")
    builder.add_edge("remember", "answer")
    builder.add_edge("answer", graph_module.END)
    graph = builder.compile(store=store)
    first = graph.invoke({"text": "the user likes tea", "answer": ""})
    assert first["answer"] == "the user likes tea"
    later = asyncio.run(graph.ainvoke({"text": "the user likes coffee", "answer": ""}))
    assert later["answer"] == "the user likes coffee"


def check_history_and_search_memory_reach_the_structure(ctx: Context) -> None:
    """history() walks one field's values oldest first, and search_memory() returns
    memvara's own results, with the claim behind each item."""
    store = _store(ctx)
    store.put(NS, "profile", {"city": "Berlin"})
    store.put(NS, "profile", {"city": "Lisbon"})
    assert [c.object for c in store.history(NS, "profile", "city")] == ["Berlin", "Lisbon"]
    results = store.search_memory("Lisbon")
    assert results and results[0].claim.object == "Lisbon"
