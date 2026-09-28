"""The invariants of retrieval, from `docs/claude/retrieval.md`, that no older test
checked in full.

Each test names the part of its invariant that the older tests left open. The parts they
already check are covered by marks on those tests.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from memvara import Memvara
from memvara.aio import AsyncMemvara
from memvara.core import ScopedMemvara
from memvara.retrieve import EpisodeResult
from memvara.retrieve.spread import seed_keys
from memvara.types import Claim

from harness import stores

#: The reads the retrieval page names as taking the time keywords.
EIGHT_READS = {"search", "get_all", "count", "history", "why", "produced", "neighborhood",
               "paths_between"}
TIME_KEYWORDS = {"as_of", "valid_at", "known_at"}
FACADES = (Memvara, ScopedMemvara, AsyncMemvara)


def parameters(function: Any) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(function).parameters)


# -- RT1 ---------------------------------------------------------------------------------


@pytest.mark.covers("inv:RT1")
def test_recall_takes_valid_at_and_refuses_every_keyword_that_could_reach_a_retired_claim(
        ) -> None:
    """`docs/claude/retrieval.md` says `recall()` takes `valid_at` and no other time
    keyword: `as_of` and `states` stay on `search()`, and the signature is spelled out
    rather than taking `**kw`, because either keyword could put a retired claim into a
    live prompt.

    On each of the three local facades, `recall` must take `valid_at`, none of the other
    keywords, and no `**kw`. A call that passes `states` must fail rather than be passed
    on to `search()`.
    """
    for facade in FACADES:
        taken = parameters(facade.recall)
        assert "valid_at" in taken, facade.__name__
        assert not {"as_of", "known_at", "states", "include_invalidated"} & set(taken), (
            facade.__name__)
        assert not [p for p in taken.values() if p.kind is p.VAR_KEYWORD], (
            f"{facade.__name__}.recall takes **kw, which would reach search()")

    with stores.memory(user="u1") as mem:
        mem.remember("user", "lives_in", "Berlin")
        mem.forget("user", "lives_in")
        with pytest.raises(TypeError):
            mem.recall("where", states=["retired"])  # type: ignore[call-arg]


# -- RT2 ---------------------------------------------------------------------------------


@pytest.mark.covers("inv:RT2")
def test_exactly_the_eight_reads_take_the_time_keywords_and_ask_takes_at() -> None:
    """`docs/claude/retrieval.md` says eight reads take the time keywords: `search`,
    `get_all`, `count`, `history`, `why`, `produced`, `neighborhood` and
    `paths_between`, and that `ask()` spells its own as `at=`.

    On each local facade, every one of the eight must take `as_of`, `valid_at` and
    `known_at`. No other public method may take `as_of` or `known_at`, and `recall` may
    take only `valid_at`. `ask` must take `at` and none of the three.
    """
    for facade in FACADES:
        taking: dict[str, set[str]] = {}
        for name in dir(facade):
            method = getattr(facade, name)
            if name.startswith("_") or not callable(method):
                continue
            keywords = TIME_KEYWORDS & set(parameters(method))
            if keywords:
                taking[name] = keywords
        assert taking == {**{read: TIME_KEYWORDS for read in EIGHT_READS},
                          "recall": {"valid_at"}}, facade.__name__
        ask = parameters(facade.ask)
        assert "at" in ask and not TIME_KEYWORDS & set(ask), facade.__name__


# -- RT4 ---------------------------------------------------------------------------------


ANCHOR = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)


def temporal_votes(*days_before: float) -> dict[str, int | None]:
    """The temporal leg's rank for each turn found, by text, when a store holding one
    turn per entry of `days_before`, each that many days before `ANCHOR`, is searched at
    `ANCHOR` with the temporal leg on."""
    with stores.memory(user="u1", read_w_temporal=1.0) as mem:
        for n, days in enumerate(days_before):
            mem.add(f"Deploy notes number {n} for the payments service.", role="system",
                    ts=ANCHOR - timedelta(days=days))
        found = mem.search("deploy notes payments service", include_episodes=True, k=10,
                           valid_at=ANCHOR, query_rewrite=False)
    return {r.episode.content: r.explain.temporal_rank
            for r in found if isinstance(r, EpisodeResult)}


@pytest.mark.covers("inv:RT4")
def test_the_temporal_leg_abstains_when_no_turn_is_within_a_half_life_of_the_anchor(
        ) -> None:
    """`docs/claude/retrieval.md` says a leg abstains rather than contributing noise:
    the temporal leg returns nothing when no turn is within a half-life of the anchor,
    because fusion reads positions and would otherwise take its top ranks from a leg
    whose every score was near zero.

    Two turns two hundred days and a year before the anchor are found by the other legs,
    and the temporal leg must rank neither of them. Once a turn one day before the anchor
    is added, the leg has an opinion and ranks the turns.
    """
    far = temporal_votes(200, 365)
    assert len(far) == 2
    assert set(far.values()) == {None}, f"the temporal leg voted with nothing near: {far}"

    near = temporal_votes(200, 1)
    assert len(near) == 2
    assert None not in near.values(), f"the temporal leg did not vote: {near}"


# -- RT5 ---------------------------------------------------------------------------------


@pytest.mark.covers("inv:RT5")
def test_the_graph_leg_seeds_the_same_entities_whatever_ids_the_claims_were_given(
        ) -> None:
    """`docs/claude/retrieval.md` says the graph leg seeds on content, never on ids:
    `spread.seed_keys()` breaks ties on `value_key`, because a claim id is a `uuid4` and
    seeding off it would make the walk depend on which ingest ran.

    Two claims tie on score. The same two claims, given their two ids the other way
    round and handed over in either order, must seed the same entities in the same
    order. With a limit of two, only the first claim's entities are seeded, so a tie
    broken on the id would seed a different pair when the ids swap.
    """
    def pair(first_id: str, second_id: str) -> list[tuple[Claim, float]]:
        return [(Claim(id=first_id, subject="Ada", predicate="works_at", object="Acme"),
                 0.5),
                (Claim(id=second_id, subject="Bo", predicate="works_at", object="Zeta"),
                 0.5)]

    seeds = set()
    for ids in (("cl_0001", "cl_9999"), ("cl_9999", "cl_0001")):
        for ranked in (pair(*ids), list(reversed(pair(*ids)))):
            seeds.add(seed_keys(ranked, 2))
    assert len(seeds) == 1, f"the seeds depended on the claim ids: {seeds}"


# -- RT6 ---------------------------------------------------------------------------------


@pytest.mark.covers("inv:RT6")
def test_a_documents_passages_are_found_only_when_episodes_are_asked_for() -> None:
    """`docs/claude/retrieval.md` says a document's passages are episodes: each chunk is
    stored as a `role="system"` turn with `meta["document_id"]`, and the episode legs
    find the passages only when `include_episodes=True` is asked for.

    A search for a phrase from the document must return no passage by default, and must
    return the passage, as a system turn naming the document, when episodes are asked
    for.
    """
    with stores.memory(user="u1") as mem:
        doc = mem.add_document("The refund window for annual plans is fourteen days "
                               "from the renewal date.", title="Refunds", extract=False)
        query = "refund window annual plans"
        plain = mem.search(query, k=5, query_rewrite=False)
        assert not [r for r in plain if isinstance(r, EpisodeResult)]

        found = mem.search(query, k=5, include_episodes=True, query_rewrite=False)
        passages = [r.episode for r in found if isinstance(r, EpisodeResult)]
        assert len(passages) == 1
        assert passages[0].role == "system"
        assert passages[0].meta["document_id"] == doc.id
