"""The invariants of consolidation and of the graph gate, from
`docs/claude/consolidation-and-graph.md`, that no older test checked.

Consolidation is the scheduled pass of `memvara/consolidate/`: decay, merge and promote.
The graph gate is `HybridRetriever._store_has_joins` in `memvara/retrieve/hybrid.py`,
which closes the graph leg on a store where no claim leads to another.
"""

from __future__ import annotations

import inspect
import warnings
from collections import Counter
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Iterator

import pytest

from memvara import Memvara
from memvara.consolidate import Consolidator, Sweep
from memvara.consolidate import decay as decay_module
from memvara.consolidate import merge as merge_module
from memvara.embed import HashingEmbedder
from memvara.retrieve import HybridRetriever
from memvara.schema import BUILTIN_PREDICATES, PredicateRegistry, PredicateSpec
from memvara.store import SQLiteStore
from memvara.types import Claim, MemoryType, Scope, utcnow

from harness import stores

from ..model_faults.scripted import ScriptedModel

SCOPE = Scope(tenant="acme", user="u1")

#: An instant two years after the claims below were written. `works_at` has a two-year
#: half-life, so a claim written at `WRITTEN` has salience 0.5 at `LATER`.
WRITTEN = utcnow()
LATER = WRITTEN + timedelta(days=730)


def put(store: SQLiteStore, predicate: str, obj: str, **fields: Any) -> Claim:
    """Store a live claim that became true at `WRITTEN`."""
    claim = Claim(subject="user", predicate=predicate, object=obj, scope=SCOPE,
                  valid_from=WRITTEN, recorded_at=WRITTEN, **fields)
    store.put_claim(claim)
    return claim


def seed(store: SQLiteStore) -> dict[str, Claim]:
    """Claims that give each stage of a pass something to do at `LATER`.

    Every claim decays. The two `likes` claims differ only in case, so merge folds one
    into the other. The episodic `goal` has been observed four times, so promote makes it
    semantic.
    """
    return {
        "acme": put(store, "works_at", "acme"),
        "coffee": put(store, "likes", "coffee"),
        "Coffee": put(store, "likes", "Coffee", observation_count=2),
        "goal": put(store, "goal", "ship v2", memory_type=MemoryType.EPISODIC,
                    observation_count=4),
        "born": put(store, "born_in", "lisbon"),
    }


def state(store: SQLiteStore) -> dict[str, tuple[Any, ...]]:
    """Every claim of the tenant, in every state, as the fields a pass may change."""
    return {c.id: (c.salience, dict(c.meta), c.observation_count, c.memory_type,
                   c.invalidated_at, c.invalidated_by, tuple(c.sources))
            for c in store.iter_claims("acme", include_invalidated=True)}


# -- CG1 ---------------------------------------------------------------------------------


@pytest.mark.covers("inv:CG1")
def test_consolidation_asks_the_model_nothing_and_a_second_pass_changes_nothing() -> None:
    """`docs/claude/consolidation-and-graph.md` says consolidation is deterministic and
    calls no model, that no stage takes an `llm` parameter, and that every stage is
    idempotent, so a scheduler that fires twice leaves the same store as one that fires
    once.

    The store here has a scripted model with an empty script, so any call to it is
    recorded as unscripted. `Memvara.consolidate()` and two passes at one instant must
    leave the model uncalled. The first pass at that instant changes something in every
    stage, and the second changes nothing and leaves every row as the first left it.
    """
    for stage in (Consolidator.__init__, Consolidator.run, Consolidator.decay,
                  Consolidator.merge_duplicates, Consolidator.promote, Consolidator._one,
                  Sweep.__init__, decay_module.decay_pass, merge_module.merge_pass,
                  merge_module.promote_pass):
        assert "llm" not in inspect.signature(stage).parameters, stage.__qualname__

    model = ScriptedModel()
    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(dim=512), llm=model, user="u1",
                  tenant="acme")
    seed(store)

    first = mem.consolidator.run("acme", now=LATER)
    after_first = state(store)
    second = mem.consolidator.run("acme", now=LATER)
    assert state(store) == after_first
    mem.consolidate()

    assert model.calls == [] and model.unscripted == [], "consolidation called the model"
    assert first == {"decayed": 5, "merged": 1, "promoted": 1}
    assert second == {"decayed": 0, "merged": 0, "promoted": 0}
    mem.close()


# -- CG2 ---------------------------------------------------------------------------------


class CountingStore:
    """A store that counts the snapshot reads, the row writes and the transactions a pass
    makes, and passes everything else to the store it wraps."""

    def __init__(self, inner: SQLiteStore) -> None:
        self._inner = inner
        self.snapshots = 0
        self.writes: Counter[str] = Counter()
        self.transactions: list[int] = []

    def iter_claims(self, *args: Any, **kwargs: Any) -> Any:
        self.snapshots += 1
        return self._inner.iter_claims(*args, **kwargs)

    def put_claim(self, claim: Claim) -> None:
        self.writes[claim.id] += 1
        if self.transactions:
            self.transactions[-1] += 1
        self._inner.put_claim(claim)

    @contextmanager
    def batch(self) -> Iterator[None]:
        self.transactions.append(0)
        with self._inner.batch():
            yield

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@pytest.mark.covers("inv:CG2")
def test_a_pass_reads_its_snapshot_once_and_writes_each_claim_once_in_bounded_batches(
        ) -> None:
    """`docs/claude/consolidation-and-graph.md` says `Sweep` reads its snapshot once and
    writes back in bounded transactions, so a pass neither scans the table once per stage
    nor holds the write lock for its own length.

    Decay, merge and promote all run here, and some claims are changed by two of them.
    The pass must read the claims once and write each changed claim once. A transaction
    closes once it holds `window` rows, but the two rows of one merge are one unit that is
    never split across two transactions, so the transaction that reaches the merge takes
    both rows and holds one more than `window`. Every transaction therefore holds at most
    `window` rows plus the largest unit's rows less one.
    """
    store = SQLiteStore(":memory:")
    claims = seed(store)
    counting = CountingStore(store)
    window = 2
    consolidator = Consolidator(counting, HashingEmbedder(dim=512), PredicateRegistry(),
                                window=window)

    # Promote needs its claim at the threshold, so the pass at `LATER` has all three
    # stages at work: every claim decays, one `likes` merges, `goal` is promoted.
    counts = consolidator.run("acme", now=LATER)

    assert counts["decayed"] == len(claims) and counts["merged"] == 1
    assert counts["promoted"] == 1
    assert counting.snapshots == 1, "the pass read the table more than once"
    assert set(counting.writes) == {c.id for c in claims.values()}
    assert set(counting.writes.values()) == {1}, f"a claim was written twice: {counting.writes}"
    assert counting.transactions == [3, 2]
    largest_unit = 2  # the merge's survivor and the duplicate it retires
    assert max(counting.transactions) <= window + largest_unit - 1
    assert sum(counting.transactions) == len(claims)
    store.close()


# -- CG3 ---------------------------------------------------------------------------------


@pytest.mark.covers("inv:CG3")
def test_every_stage_of_a_pass_uses_the_one_instant_the_caller_passed() -> None:
    """`docs/claude/consolidation-and-graph.md` says `now` is read once for the whole
    pass, and that passing it explicitly evaluates two passes at the same instant.

    The pass here runs two years after the claims were written, far from the wall clock.
    Decay must compute salience at that instant, merge must retire the duplicate at that
    instant, and a second pass at the same instant must find nothing to do. A stage that
    read the wall clock instead would give a claim almost its full salience and stamp
    the merge with today.
    """
    store = SQLiteStore(":memory:")
    claims = seed(store)
    consolidator = Consolidator(store, HashingEmbedder(dim=512), PredicateRegistry())

    assert consolidator.run("acme", now=LATER)["merged"] == 1

    acme = store.get_claim(claims["acme"].id)
    assert acme is not None and acme.salience == pytest.approx(0.5, abs=1e-6)
    retired = [c for c in store.iter_claims("acme", include_invalidated=True)
               if c.invalidated_at is not None]
    assert len(retired) == 1
    assert retired[0].invalidated_at == LATER
    assert consolidator.run("acme", now=LATER) == {"decayed": 0, "merged": 0,
                                                     "promoted": 0}
    store.close()


# -- CG4 ---------------------------------------------------------------------------------


class CountingTraverser:
    """The store's traverser, counting how often the graph leg walks."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.walks = 0

    def spread(self, *args: Any, **kwargs: Any) -> Any:
        self.walks += 1
        return self._inner.spread(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


#: A question the query classifier reads as a chain, so intent weighting opens the leg.
RELATIONAL = "who is the manager of the person who uses pytest"


def walks(chains: bool) -> tuple[int, dict[str, int]]:
    """How often the graph leg walked for `RELATIONAL`, and the store's join counts.

    `uses`, `configured_in` and `near` are declared entity-valued, so their claims carry
    edges. In the store that chains, `pytest` is the object of one claim and the subject
    of another. In the star, both claims hang off `user` and nothing leads further.
    """
    registry = PredicateRegistry(BUILTIN_PREDICATES + tuple(
        PredicateSpec(name, object_type=("entity",), graph=True)
        for name in ("uses", "configured_in", "near")))
    mem = stores.memory(registry=registry)
    mem.remember("user", "uses", "pytest")
    if chains:
        mem.remember("pytest", "configured_in", "pyproject.toml")
    else:
        mem.remember("user", "near", "Delhi")
    traverser = CountingTraverser(mem.traverser)
    retriever = HybridRetriever(mem.store, mem.embedder, mem.registry, w_graph=1.0,
                                traverser=traverser)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        retriever.search(RELATIONAL, mem.default_scope, k=5)
    counts = mem.connectivity()
    mem.close()
    return traverser.walks, counts


@pytest.mark.covers("inv:CG4")
def test_the_graph_leg_does_not_walk_a_store_with_no_joins_and_walks_one_with_a_single_join(
        ) -> None:
    """`docs/claude/consolidation-and-graph.md` says the graph leg is closed when the
    store provably has no joins, that the condition is `joinable_claims == 0` rather than
    a percentage, and that the gate sits after intent weighting so it can undo what the
    query classifier opened.

    The question here is one the classifier opens the leg for. On a store with no joins
    the leg must not walk. On a store where one claim of two joins, a join rate of 50%,
    it must walk, which a percentage threshold above that would stop.
    """
    star_walks, star = walks(chains=False)
    chain_walks, chain = walks(chains=True)

    assert star == {"live_claims": 2, "joinable_claims": 0}
    assert chain == {"live_claims": 2, "joinable_claims": 1}
    assert star_walks == 0, "the graph leg walked a store with no joins"
    assert chain_walks >= 1, "the graph leg did not walk a store that has a join"
