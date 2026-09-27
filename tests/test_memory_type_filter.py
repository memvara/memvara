"""`search()` and `recall()` refuse a memory type that does not exist (#289).

`memory_types` keeps only the kinds of memory it names, so a misspelled name used to keep
no claim at all: the call returned nothing and raised nothing, and a caller who made a
typo could not tell that from a store with nothing relevant in it. `remember()` already
refuses an unknown name (#288), and these calls now refuse it with the same words, before
anything is read.
"""

from __future__ import annotations

import asyncio

import pytest

from memvara import AsyncMemvara, Memvara, MemoryType
from memvara.embed import HashingEmbedder
from memvara.llm import NullLLM

REFUSAL = "memory_type must be one of episodic, semantic, procedural, not 'procedurel'"


@pytest.fixture
def mem() -> Memvara:
    mem = Memvara(":memory:", embedder=HashingEmbedder(), llm=NullLLM())
    mem.remember("user", "prefers", "tabs")
    return mem


def test_search_refuses_a_memory_type_that_does_not_exist(mem: Memvara) -> None:
    with pytest.raises(ValueError) as caught:
        mem.search("tabs", memory_types=["procedurel"])
    assert str(caught.value) == REFUSAL


def test_recall_refuses_a_memory_type_that_does_not_exist(mem: Memvara) -> None:
    with pytest.raises(ValueError) as caught:
        mem.recall("tabs", memory_types=["semantic", "procedurel"])
    assert str(caught.value) == REFUSAL


def test_the_async_class_refuses_it_too(mem: Memvara) -> None:
    async def main() -> None:
        amem = AsyncMemvara(mem)
        with pytest.raises(ValueError, match="memory_type must be one of"):
            await amem.search("tabs", memory_types=["procedurel"])
        with pytest.raises(ValueError, match="memory_type must be one of"):
            await amem.recall("tabs", memory_types=["procedurel"])

    asyncio.run(main())


@pytest.mark.parametrize("types", [["procedural"], [MemoryType.PROCEDURAL],
                                   ["semantic", MemoryType.PROCEDURAL]])
def test_a_memory_type_named_as_a_string_or_a_member_still_filters(
        mem: Memvara, types: list) -> None:
    """`prefers` is a procedural predicate, so the stored preference is procedural."""
    assert [r.claim.object for r in mem.search("tabs", memory_types=types)] == ["tabs"]
    assert "tabs" in mem.recall("tabs", memory_types=types)


def test_a_filter_for_another_type_still_finds_nothing(mem: Memvara) -> None:
    """A real type that no stored claim has is an ordinary empty answer, not a refusal."""
    assert list(mem.search("tabs", memory_types=["episodic"])) == []
