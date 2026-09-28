"""CrewAI: `MemvaraStorage`, against the real crewai.

Each `check_*` function runs inside a virtual environment that holds crewai at the floor
memvara declares or at the newest release (see `probe.py`). The promises they check are
the ones the adapter's docstrings and `docs/integrations/frameworks.md` make. crewai is
imported inside each check, never at module level, because the suite imports this module
to list the checks and has no crewai installed.

CrewAI's `Memory` builds its analysis model the first time it saves anything, even when
every field of the memory is given and no analysis is needed, and its default model needs
an OpenAI key. The probe runs with no key, so each check that uses `Memory` hands it a
model that records every call and then raises. CrewAI catches that error and carries on
without the model, so the raise alone would not fail the check. Instead, each check that
uses `Memory` fails afterwards if the model was called at all. That also proves the
adapter's own writes cost no model call.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import json
import re
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterator

from memvara.compat import NOTE_PREDICATE
from memvara.integrations import crewai as adapter

if TYPE_CHECKING:
    from ..probe import Context

#: The fields every memory in these checks has unless a check gives its own, whether it is
#: saved through `Memory.remember` or built as a `MemoryRecord` by `_record`. They are the
#: fields CrewAI's analysis would otherwise ask a model for, and giving all three keeps
#: `Memory.remember` on its path that makes no model call. The importance is 0.5, the
#: value CrewAI gives a memory that names none (`MemoryRecord.importance` and
#: `MemoryConfig.default_importance`, at both pins), so a record here is the record
#: CrewAI itself would write.
FIELDS: dict[str, Any] = {"scope": "/crew/alice", "categories": ["profile"],
                          "importance": 0.5}


def _ca(module: str) -> Any:
    return importlib.import_module(f"crewai{module}")


def _no_model(calls: list[str]) -> Any:
    """A CrewAI model that appends what it was asked to `calls`, then raises."""

    class NoModel(_ca(".llms.base_llm").BaseLLM):  # type: ignore[misc]
        def call(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append(str(messages)[-300:])
            raise AssertionError("CrewAI called its model")

        async def acall(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
            return self.call(messages)

    return NoModel(model="no-model")


def _consolidator(asked: list[str]) -> Any:
    """A CrewAI model that answers the consolidation question by deleting every record it
    is shown and inserting the new one. It appends each question it gets to `asked`."""

    class Consolidator(_ca(".llms.base_llm").BaseLLM):  # type: ignore[misc]
        def supports_function_calling(self) -> bool:
            return False

        def call(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
            asked.append(messages[-1]["content"])
            ids = re.findall(r"id=(\S+)", messages[-1]["content"])
            return json.dumps({"actions": [{"action": "delete", "record_id": i}
                                           for i in ids], "insert_new": True})

    return Consolidator(model="scripted")


@contextlib.contextmanager
def _memory(storage: Any, llm: Any = None) -> Iterator[Any]:
    """CrewAI's `Memory` with the documented wiring,
    `Memory(storage=storage, embedder=storage.embedder)`, closed when the block ends.

    Unless the check hands it another model, it gets one that must not be called, and the
    block fails when it ends if anything called that model. When the block raised as well,
    the failure names both, so the call to the model is not hidden behind the error."""
    calls: list[str] = []
    memory = _ca(".memory").Memory(storage=storage, embedder=storage.embedder,
                                   llm=llm if llm is not None else _no_model(calls))

    def called() -> str:
        return (f"CrewAI called its model, which no check here allows. Calls: "
                f"{len(calls)}. The last one ended with: {calls[-1]}")

    try:
        yield memory
    except Exception as exc:
        if calls:
            raise AssertionError(f"{called()}\nThe check also failed with "
                                 f"{type(exc).__name__}: {exc}") from exc
        raise
    finally:
        close = getattr(memory, "close", None)
        if close is not None:
            close()
    assert calls == [], called()


def _record(content: str, **fields: Any) -> Any:
    """A CrewAI `MemoryRecord` with `FIELDS`, and `fields` in place of any of them."""
    return _ca(".memory.types").MemoryRecord(content=content, **{**FIELDS, **fields})


def check_the_storage_satisfies_the_storagebackend_protocol(ctx: Context) -> None:
    """MemvaraStorage passes CrewAI's runtime-checkable StorageBackend check, and every
    method the protocol declares takes the same parameters, by name, in the same order."""
    protocol = _ca(".memory.storage.backend").StorageBackend
    storage = adapter.MemvaraStorage(ctx.memvara(), user="alice")
    assert isinstance(storage, protocol)
    differences = []
    for name, declared in vars(protocol).items():
        if name.startswith("_") or not callable(declared):
            continue
        theirs = list(inspect.signature(declared).parameters)
        ours = list(inspect.signature(getattr(adapter.MemvaraStorage, name)).parameters)
        if ours != theirs:
            differences.append((name, theirs, ours))
    assert differences == [], differences


def check_crewais_memory_remembers_and_recalls_through_the_storage(ctx: Context) -> None:
    """The documented wiring saves a memory and recalls it, with no model call."""
    storage = adapter.MemvaraStorage(ctx.memvara(), user="alice")
    with _memory(storage) as memory:
        saved = memory.remember("Alice lives in Berlin", **FIELDS)
        memory.remember("Alice drinks green tea", scope="/crew/alice",
                        categories=["taste"], importance=0.4)
        matches = memory.recall("where does Alice live", depth="shallow")
    assert saved is not None and saved.content == "Alice lives in Berlin"
    assert matches and matches[0].record.content == "Alice lives in Berlin", matches


def check_a_record_round_trips_as_crewais_own_memoryrecord(ctx: Context) -> None:
    """A MemoryRecord saved and read back is CrewAI's own type, with every field it
    carried: id, content, scope, categories, metadata, importance, source, privacy and
    its creation time."""
    storage = adapter.MemvaraStorage(ctx.memvara(), user="alice")
    sent = _record("Alice lives in Berlin", metadata={"from": "onboarding"},
                   source="agent-7", private=True)
    storage.save([sent])
    got = storage.get_record(sent.id)
    assert type(got) is type(sent)
    for name in ("id", "content", "scope", "categories", "metadata", "importance",
                 "source", "private", "created_at"):
        assert getattr(got, name) == getattr(sent, name), name


def check_crewais_scorer_accepts_the_records_it_gets_back(ctx: Context) -> None:
    """CrewAI's own compute_composite_score does naive datetime arithmetic on created_at,
    and a record read back from memvara goes through it without a TypeError."""
    types = _ca(".memory.types")
    storage = adapter.MemvaraStorage(ctx.memvara(), user="alice")
    storage.save([_record("Alice lives in Berlin")])
    found = storage.search(storage.embedder(["where does Alice live"])[0])
    assert found, "nothing was found"
    score, reasons = types.compute_composite_score(found[0][0], found[0][1],
                                                   types.MemoryConfig())
    assert score >= 0.0, (score, reasons)


def check_the_embedder_satisfies_crewais_embed_helpers(ctx: Context) -> None:
    """CrewAI's embed_text and embed_texts accept the storage's embedder and get plain
    lists of floats back, not an ndarray whose truth value is ambiguous."""
    types = _ca(".memory.types")
    storage = adapter.MemvaraStorage(ctx.memvara(), user="alice")
    one = types.embed_text(storage.embedder, "Alice lives in Berlin")
    many = types.embed_texts(storage.embedder, ["one", "two"])
    assert isinstance(one, list) and one and all(isinstance(x, float) for x in one)
    assert len(many) == 2 and all(isinstance(vector, list) for vector in many)


def check_a_vector_from_another_model_is_refused(ctx: Context) -> None:
    """search() given a vector the storage's embedder did not make refuses and names the
    wiring, rather than searching in a space where the answer is noise."""
    storage = adapter.MemvaraStorage(ctx.memvara(), user="alice")
    try:
        storage.search([0.5] * 1536)
    except adapter.CrewAICompatError as exc:
        assert "embedder=storage.embedder" in str(exc), str(exc)
    else:
        raise AssertionError("a vector from another model was searched")


def check_update_ends_the_old_text_and_delete_retires_it(ctx: Context) -> None:
    """CrewAI's Memory.update() supersedes the record: the old text is ended, with its
    world clock closed and its belief clock open, and its row stays. Memory.forget()
    retires the record: its belief clock closes and its world clock stays open. Each end
    moves exactly one clock, and the ended text points at the text that replaced it."""
    mem = ctx.memvara()
    storage = adapter.MemvaraStorage(mem, user="alice", on_delete="retire")
    record = _record("Alice lives in Berlin")
    storage.save([record])
    subject = f"{adapter.SUBJECT_PREFIX}{record.id}"
    with _memory(storage) as memory:
        memory.update(record.id, content="Alice lives in Lisbon")
        current = storage.get_record(record.id)
        assert current is not None and current.content == "Alice lives in Lisbon"
        versions = mem.history(subject, NOTE_PREDICATE)
        assert [(c.object, c.state) for c in versions] == [
            ("Alice lives in Berlin", "ended"), ("Alice lives in Lisbon", "live")]
        assert versions[0].valid_to is not None and versions[0].invalidated_at is None
        # The adapter's docstring: the old value is ended "with `invalidated_by`
        # pointing at its replacement".
        assert versions[0].invalidated_by == versions[1].id, versions
        assert memory.forget(record_ids=[record.id]) == 1
    gone = mem.history(subject, NOTE_PREDICATE)[-1]
    assert gone.state == "retired", gone
    assert gone.invalidated_at is not None and gone.valid_to is None
    assert storage.get_record(record.id) is None


def check_forget_warns_once_that_it_retired(ctx: Context) -> None:
    """With the default on_delete="warn", forgetting retires, and warns once per storage,
    naming on_delete='erase'."""
    storage = adapter.MemvaraStorage(ctx.memvara(), user="alice")
    records = [_record("Alice lives in Berlin"), _record("Alice drinks tea")]
    storage.save(records)
    with _memory(storage) as memory, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for record in records:
            memory.forget(record_ids=[record.id])
    ours = [w for w in caught if w.category is adapter.CrewAIDeletionWarning]
    assert len(ours) == 1 and "on_delete='erase'" in str(ours[0].message), ours


def check_reset_leaves_nothing_behind(ctx: Context) -> None:
    """Memory.reset() is a wipe: every claim the storage wrote is erased, and the store
    can prove that no row, index entry or vector of them is left."""
    mem = ctx.memvara()
    storage = adapter.MemvaraStorage(mem, user="alice")
    storage.save([_record("Alice lives in Berlin"), _record("Alice drinks tea")])
    ids = [claim.id for claim in mem.get_all()]
    with _memory(storage) as memory:
        memory.reset()
    assert storage.count() == 0 and mem.get_all() == []
    for claim_id in ids:
        proof = mem.prove_erased(claim_id)
        assert proof.proven, proof


def check_listing_and_scope_info_come_back_as_crewais_types(ctx: Context) -> None:
    """list_records is newest first, scopes and categories are counted from what is
    stored, and get_scope_info returns CrewAI's own ScopeInfo, which Memory.info reads."""
    types = _ca(".memory.types")
    storage = adapter.MemvaraStorage(ctx.memvara(), user="alice")
    older = _record("Alice lives in Berlin", scope="/crew/alice",
                    created_at=datetime(2024, 3, 1))
    newer = _record("Bob likes tea", scope="/crew/bob", categories=["taste"],
                    created_at=datetime(2024, 6, 1))
    storage.save([older, newer])
    assert [r.content for r in storage.list_records()] == ["Bob likes tea",
                                                          "Alice lives in Berlin"]
    assert storage.list_scopes("/crew") == ["/crew/alice", "/crew/bob"]
    assert storage.list_categories() == {"profile": 1, "taste": 1}
    info = storage.get_scope_info("/crew")
    assert isinstance(info, types.ScopeInfo) and info.record_count == 2
    with _memory(storage) as memory:
        assert memory.info("/crew").record_count == 2


def check_a_metadata_filter_is_refused(ctx: Context) -> None:
    """metadata_filter= is refused by search and by delete, because filtering after the
    ranking would under-fill a page with no way to tell."""
    storage = adapter.MemvaraStorage(ctx.memvara(), user="alice")
    vector = storage.embedder(["anything"])[0]
    for call in (lambda: storage.search(vector, metadata_filter={"k": "v"}),
                 lambda: storage.delete(metadata_filter={"k": "v"})):
        try:
            call()
        except adapter.CrewAICompatError:
            pass
        else:
            raise AssertionError("a metadata filter was accepted")


def check_the_async_methods_answer_like_the_sync_ones(ctx: Context) -> None:
    """asave, asearch and adelete run the synchronous methods off the event loop."""
    storage = adapter.MemvaraStorage(ctx.memvara(), user="alice", on_delete="retire")
    record = _record("Alice lives in Berlin")

    async def scenario() -> None:
        await storage.asave([record])
        found = await storage.asearch(storage.embedder(["where does Alice live"])[0])
        assert [r.content for r, _ in found] == ["Alice lives in Berlin"]
        assert await storage.adelete(record_ids=[record.id]) == 1

    asyncio.run(scenario())
    assert storage.get_record(record.id) is None


def check_two_storages_on_one_memvara_cannot_see_each_other(ctx: Context) -> None:
    """Two backends bound to two users of one store never see each other's records."""
    mem = ctx.memvara()
    alice = adapter.MemvaraStorage(mem, user="alice")
    bob = adapter.MemvaraStorage(mem, user="bob")
    mine = _record("Alice lives in Berlin")
    alice.save([mine])
    bob.save([_record("Bob lives in Rome")])
    assert [r.content for r in bob.list_records()] == ["Bob lives in Rome"]
    assert bob.get_record(mine.id) is None
    found = bob.search(bob.embedder(["Alice lives in Berlin"])[0])
    assert [r.content for r, _ in found] == ["Bob lives in Rome"]


def check_a_repeated_memory_reaches_crewais_consolidation(ctx: Context) -> None:
    """CrewAI asks its model to consolidate a new memory with any stored record whose
    search score is at least 0.85, and the adapter says CrewAI's own dedup path deletes a
    superseded record through it. So remembering one sentence twice must reach that
    question once, and with a model that answers "delete the old one", leave one live
    record."""
    storage = adapter.MemvaraStorage(ctx.memvara(), user="alice", on_delete="retire")
    asked: list[str] = []
    with _memory(storage, llm=_consolidator(asked)) as memory:
        memory.remember("Alice lives in Berlin", **FIELDS)
        memory.remember("Alice lives in Berlin", **FIELDS)
        live = [r.content for r in memory.list_records()]
    top = storage.search(storage.embedder(["Alice lives in Berlin"])[0])[0][1]
    assert len(asked) == 1 and live == ["Alice lives in Berlin"], (
        f"CrewAI asked its model to consolidate {len(asked)} times: the stored copy "
        f"scored {top:.2f} against CrewAI's threshold of 0.85, and {len(live)} live "
        "copies remain")
