"""The invariants of telemetry and the benchmark harnesses, from
`docs/claude/telemetry-and-benchmarks.md`, that no older test checked in full.

A benchmark invariant is a promise about the code in `bench/` and `demo/`, so these tests
build stores the way the harnesses build them and check what the harness produced.
"""

from __future__ import annotations

import inspect
import pathlib
import sys
import threading
from typing import Any

import pytest

from memvara.telemetry import (
    RETRIEVAL_MODEL_QUERY, RETRIEVAL_REWRITE_MS, RETRIEVAL_SELECT_MS,
    RETRIEVAL_SYNTHESIS_MS, MemoryRecorder,
)
import memvara.telemetry

from harness import stores
from harness.skips import needs_toml

from ..model_faults.handles import with_model
from ..model_faults.scripted import Forever, ScriptedModel, Text

# `bench/` is a directory of scripts that import each other by bare name, as
# `tests/test_bench_eval.py` explains, so the directory goes on the path.
BENCH = pathlib.Path(__file__).resolve().parents[3] / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import evalkit as ek  # noqa: E402
import locomo  # noqa: E402
import longmemeval as lme  # noqa: E402
import multihop  # noqa: E402
import temporal  # noqa: E402
import twowiki  # noqa: E402

from demo import baselines as bl  # noqa: E402

#: The embedder `tests/test_bench_eval.py` pins for the harness builders.
HASHING = ek.build_embedder("hashing")

#: A small 2WikiMultihopQA question whose evidence chains: `Person B` is the object of one
#: relation and the subject of the next.
TWOWIKI = {"_id": "q1", "question": "Who is the father of the director of Film A?",
           "answer": "Person C", "type": "compositional",
           "evidences": [["Film A", "director", "Person B"],
                         ["Person B", "father", "Person C"]]}


# -- TB4 ---------------------------------------------------------------------------------


@pytest.mark.covers("inv:TB4")
@needs_toml
def test_the_retrieval_harnesses_build_their_stores_with_no_extraction_model() -> None:
    """`docs/claude/telemetry-and-benchmarks.md` says the benchmark harnesses do not
    exercise ingestion: the retrieval corpora run against structured data with no
    extraction model, so a regression that broke extraction would not show up there.

    The LOCOMO and LongMemEval builders must default to a model that is a no-op. The
    graph and temporal corpora must be written as structured facts through a no-op
    model, which leaves the store holding claims and no turn at all.
    """
    conversational = [locomo.build_memory(locomo.fixture()[0], ek.RetrievalBudget(),
                                          embedder=HASHING),
                      lme.build_memory("u1", ek.RetrievalBudget(), embedder=HASHING)]
    structured = [multihop.load(multihop.Corpus(staff=4, companies=2), padding=2),
                  twowiki.ingest([twowiki.Sample(TWOWIKI)]),
                  temporal.build(1)[0]]
    for mem in conversational + structured:
        assert mem.llm.is_noop, f"a harness store was built with {mem.llm!r}"
    for mem in structured:
        stats = mem.stats()
        assert stats["claims"] > 0 and stats["episodes"] == 0, stats
    for mem in conversational + structured:
        mem.close()


# -- TB5 ---------------------------------------------------------------------------------


@pytest.mark.covers("inv:TB5")
def test_recording_telemetry_starts_no_thread() -> None:
    """`docs/claude/telemetry-and-benchmarks.md` says telemetry adds no background
    thread: `MemoryRecorder` keeps its series in memory for a deployment to read.

    A store that records every write, read and consolidation pass into a
    `MemoryRecorder` must start no thread, and the telemetry module must not import the
    threading module at all.
    """
    before = set(threading.enumerate())
    recorder = MemoryRecorder()
    with stores.memory(user="u1", telemetry=recorder) as mem:
        mem.remember("user", "likes", "green tea")
        mem.add("I live in Berlin.")
        mem.search("green tea", query_rewrite=False)
        mem.consolidate()
    assert recorder.names(), "the store recorded nothing, so the test measured nothing"
    assert set(threading.enumerate()) - before == set()
    assert "threading" not in inspect.getsource(memvara.telemetry)


# -- TB6 ---------------------------------------------------------------------------------

#: One reply that the query rewrite, the ranked stage and the synthesis can each read.
CHAT = Text('{"queries": [], "date_range": null, "synthesis": "The notes mention Lisbon.", '
            '"kept": []}')


@pytest.mark.covers("inv:TB6")
def test_a_read_that_calls_the_model_three_times_is_metered_three_times() -> None:
    """`docs/claude/telemetry-and-benchmarks.md` says every read-path model call is
    counted, whichever stage made it: the ranked stage, the query rewrite and the
    synthesis all emit `retrieval.model_query`, so a quota that sums that series meters
    every answered call. Each stage also has its own timer.

    One recall here runs all three stages. The model must answer three calls,
    `retrieval.model_query` must total three, and each stage's timer must have one
    value.
    """
    model = ScriptedModel(chat=[Forever(CHAT)])
    recorder = MemoryRecorder()
    mem = with_model(model, ranked=True, telemetry=recorder)
    mem.remember("user", "likes", "Lisbon trams")
    for day in range(2):
        mem.add(f"We talked about the Lisbon trip again on day {day}.", role="system")

    mem.recall("Lisbon trip", ranked=True, include_episodes=True, synthesize=True)

    assert model.count("chat") == 3 and model.unscripted == []
    assert recorder.total(RETRIEVAL_MODEL_QUERY) == 3
    assert recorder.total(RETRIEVAL_MODEL_QUERY, stage="rewrite") == 1
    assert recorder.total(RETRIEVAL_MODEL_QUERY, stage="synthesis") == 1
    for timer in (RETRIEVAL_SELECT_MS, RETRIEVAL_REWRITE_MS, RETRIEVAL_SYNTHESIS_MS):
        assert len(recorder.values(timer)) == 1, timer
    mem.close()


# -- TB7 ---------------------------------------------------------------------------------


@pytest.mark.covers("inv:TB7")
def test_the_harness_builders_switch_the_query_rewrite_off_for_a_model_that_can_chat(
        ) -> None:
    """`docs/claude/telemetry-and-benchmarks.md` says benchmark reads are plain, and that
    the harnesses' `Memvara` builders pass `**PLAIN_READ`, so an extraction model that
    can chat does not also rewrite the questions a benchmark asks.

    Each builder that takes a model is handed one that can chat and has nothing
    scripted. A read that does not mention the rewrite must then make no model call.
    """
    built: list[tuple[str, ScriptedModel, Any]] = []
    for name, build in (
            ("locomo", lambda m: locomo.build_memory(locomo.fixture()[0],
                                                     ek.RetrievalBudget(), llm=m,
                                                     embedder=HASHING)),
            ("longmemeval", lambda m: lme.build_memory("u1", ek.RetrievalBudget(), llm=m,
                                                       embedder=HASHING)),
            ("demo", lambda m: bl.build_memory([], llm=m)[0])):
        model = ScriptedModel()
        built.append((name, model, build(model)))
    for name, model, mem in built:
        mem.remember("user", "likes", "green tea")
        mem.search("what does the user like to drink")
        mem.recall("what does the user like to drink")
        assert model.calls == [] and model.unscripted == [], (
            f"the {name} builder's store rewrote a benchmark question: {model.unscripted}")
        mem.close()


# -- TB8 ---------------------------------------------------------------------------------


@pytest.mark.covers("inv:TB8")
@needs_toml
def test_the_graph_harnesses_write_stores_whose_relations_carry_edges() -> None:
    """`docs/claude/telemetry-and-benchmarks.md` says a graph harness declares its
    corpus's relations, or it measures a store with no edges: `bench/multihop.py` builds
    its registry in `vocabulary()` and `bench/twowiki.py` loads its pack.

    The older tests check the two vocabularies. This one checks that each harness
    writes its store with its vocabulary: the stores `multihop.load` and
    `twowiki.ingest` write must hold claims that join, and a store written without a
    declared vocabulary holds none.
    """
    org = multihop.load(multihop.Corpus(staff=6, companies=2), padding=3)
    wiki = twowiki.ingest([twowiki.Sample(TWOWIKI)])
    undeclared = stores.memory()
    for subject, predicate, obj in multihop.Corpus(staff=6, companies=2).facts():
        undeclared.remember(subject, predicate, obj)

    assert org.connectivity()["joinable_claims"] > 0
    assert wiki.connectivity() == {"live_claims": 2, "joinable_claims": 1}
    assert undeclared.connectivity()["joinable_claims"] == 0, (
        "without a vocabulary the corpus should carry no edge, or this test shows nothing")
    for mem in (org, wiki, undeclared):
        mem.close()
