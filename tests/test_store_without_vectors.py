"""`Memvara(path, embeddings=False)`: a store that keeps no vectors.

A caller that reads its store by subject and predicate, and never searches by meaning,
pays for vectors it does not use. memvara-code measured them at about half of every store
it writes. These tests hold the promises the option makes: nothing is embedded on any
write or read, the store finds things by text alone, a store that already holds vectors
is refused until they are dropped, and a store written without vectors is refused by an
embedder until it embeds everything.

`NoEmbedder.encode` raises, so a path these tests exercise that tried to embed would
fail, or, where a call is wrapped to keep a write going, warn; the tests that cover those
paths turn warnings into errors.
"""

from __future__ import annotations

import json
import os
import pathlib
import warnings

import pytest

from memvara import EmbedderMismatchError, Memvara, NullLLM
from memvara.embed import CachedEmbedder, HashingEmbedder, NoEmbedder, embeds
from memvara.store.sqlite import SCHEMA_VERSION
from memvara.types import Episode, Scope

from test_open_before_upgrade import stamp, version_of

#: Every test here holds `docs/claude/retrieval.md`'s bullet "A store can keep no vectors at
#: all."
pytestmark = pytest.mark.covers("inv:RT11")


def _record(db: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(f"{db}.embedder.json").read_text())


def _vectored(db: pathlib.Path) -> None:
    with Memvara(str(db), embedder=HashingEmbedder(dim=16), llm=NullLLM()) as mem:
        mem.remember("user", "lives_in", "Lisbon")


def test_a_store_without_vectors_writes_none_and_finds_by_text(tmp_path):
    db = tmp_path / "s.db"
    with Memvara(str(db), embeddings=False, llm=NullLLM()) as mem:
        mem.remember("user", "lives_in", "Lisbon")
        mem.add("I work at Acme Corporation in Lisbon.")
        stats = mem.stats()
        hits = mem.search("Lisbon")
        consolidated = mem.consolidate()

    assert stats["claims"] >= 1 and stats["embeddings"] == 0
    assert any(hit.claim.object == "Lisbon" for hit in hits)
    assert consolidated["merged"] == 0
    assert not (tmp_path / "s.db.vecs").exists()
    assert _record(db) == {"embedder": "none", "dim": 0}
    assert "embed=none" in repr(Memvara(str(db), embeddings=False, llm=NullLLM()))


def test_a_turn_cited_by_a_remembered_claim_gets_no_vector(tmp_path):
    with Memvara(str(tmp_path / "s.db"), embeddings=False, llm=NullLLM()) as mem:
        turn = Episode(role="user", content="We moved the office to Porto.",
                       scope=Scope("default"))
        mem.remember("office", "located_in", "Porto", sources=[turn])
        assert mem.stats()["embeddings"] == 0


def test_an_embedder_cannot_be_passed_with_embeddings_off():
    with pytest.raises(TypeError, match="embeddings=False and embedder="):
        Memvara(embeddings=False, embedder=HashingEmbedder(dim=8))


def test_replacement_advice_needs_vectors():
    with pytest.raises(TypeError, match="nearest neighbours by vector"):
        Memvara(embeddings=False, advise_replacements=True)


def test_a_hosted_client_refuses_embeddings_off():
    with pytest.raises(TypeError, match="^embeddings cannot be combined with api_key="):
        Memvara(api_key="mv_test", embeddings=False)


def test_a_store_that_holds_vectors_is_refused_until_they_are_dropped(tmp_path):
    db = tmp_path / "s.db"
    _vectored(db)

    with pytest.raises(EmbedderMismatchError, match="holds vectors of width 16"):
        Memvara(str(db), embeddings=False, llm=NullLLM())

    with Memvara(str(db), embeddings=False, llm=NullLLM(), reembed=True) as mem:
        assert mem.stats()["claims"] == 1
        assert mem.stats()["embeddings"] == 0
        assert [c.object for c in mem.history("user", "lives_in")] == ["Lisbon"]
    assert _record(db) == {"embedder": "none", "dim": 0}
    with Memvara(str(db), embeddings=False, llm=NullLLM()) as mem:
        assert mem.stats()["embeddings"] == 0


def test_an_older_store_that_holds_vectors_is_refused_before_it_is_upgraded(tmp_path):
    db = tmp_path / "old.db"
    _vectored(db)
    stamp(db, SCHEMA_VERSION - 1)

    with pytest.raises(EmbedderMismatchError, match="embeddings=False keeps none"):
        Memvara(str(db), embeddings=False, llm=NullLLM())
    assert version_of(db) == SCHEMA_VERSION - 1


def test_reembed_drops_the_vectors_of_a_store_that_keeps_none(tmp_path):
    db = tmp_path / "s.db"
    _vectored(db)
    with Memvara(str(db), embedder=HashingEmbedder(dim=16), llm=NullLLM()) as mem:
        assert mem.reembed(NoEmbedder()) == 0
        assert mem.stats()["embeddings"] == 0
        assert not embeds(mem.writer.embedder) and not embeds(mem.reader.embedder)
        mem.remember("user", "works_at", "Acme")
        assert mem.stats()["embeddings"] == 0


def test_an_embedder_is_refused_on_a_store_written_without_vectors(tmp_path):
    db = tmp_path / "s.db"
    with Memvara(str(db), embeddings=False, llm=NullLLM()) as mem:
        mem.remember("user", "lives_in", "Lisbon")
        mem.remember("user", "works_at", "Acme")

    with pytest.raises(EmbedderMismatchError, match="written with embeddings=False"):
        Memvara(str(db), embedder=HashingEmbedder(dim=16), llm=NullLLM())

    with Memvara(str(db), embedder=HashingEmbedder(dim=16), llm=NullLLM(),
                 reembed=True) as mem:
        assert mem.stats()["embeddings"] == 2
        assert any(h.claim.object == "Acme" for h in mem.search("where does the user work"))
    assert _record(db) == {"embedder": "hashing:16:3-5", "dim": 16}


@pytest.mark.parametrize("leftover", ["claims", "turns"])
def test_an_embedder_is_refused_on_a_store_holding_only_turns_too(tmp_path, leftover):
    db = tmp_path / "s.db"
    with Memvara(str(db), embeddings=False, llm=NullLLM()) as mem:
        if leftover == "claims":
            mem.remember("user", "lives_in", "Lisbon")
        else:
            mem.store.add_episode(Episode(role="user", content="hello",
                                          scope=Scope("default")))
    with pytest.raises(EmbedderMismatchError):
        Memvara(str(db), embedder=HashingEmbedder(dim=16), llm=NullLLM())


def test_a_record_left_beside_a_new_empty_store_does_not_refuse_an_embedder(tmp_path):
    """A deleted store's record must not decide anything about the new file."""
    db = tmp_path / "s.db"
    with Memvara(str(db), embeddings=False, llm=NullLLM()):
        pass
    for leftover in tmp_path.iterdir():
        if leftover.name != "s.db.embedder.json":
            os.remove(leftover)

    with Memvara(str(db), embedder=HashingEmbedder(dim=16), llm=NullLLM()) as mem:
        mem.remember("user", "lives_in", "Lisbon")
        assert mem.stats()["embeddings"] == 1


def test_no_embedder_refuses_to_encode_and_is_seen_through_a_wrapper():
    with pytest.raises(RuntimeError, match="keeps no vectors"):
        NoEmbedder().encode(["anything"])
    assert not embeds(CachedEmbedder(NoEmbedder()))
    assert embeds(CachedEmbedder(HashingEmbedder(dim=8)))
    assert repr(NoEmbedder()) == "<NoEmbedder>"


def test_a_slot_holding_several_claims_is_not_merged_and_nothing_is_embedded(tmp_path):
    """The merge pass compares the vectors of every slot that holds more than one live
    claim, so a store without vectors skips it rather than embedding them."""
    with Memvara(str(tmp_path / "s.db"), embeddings=False, llm=NullLLM()) as mem:
        mem.remember("user", "likes", "green tea")
        mem.remember("user", "likes", "black coffee")
        assert mem.consolidate()["merged"] == 0
        assert len(mem.history("user", "likes")) == 2


def test_turns_and_rewritten_phrasings_are_searched_by_text_alone(tmp_path):
    from test_read_stages import FakeChat, rewrite_reply
    from memvara.select import QueryRewriter

    with Memvara(str(tmp_path / "s.db"), embeddings=False, llm=NullLLM(),
                 read_rewriter=QueryRewriter(FakeChat(rewrite_reply("bicycle")))) as mem:
        mem.remember("user", "owns", "bicycle")
        mem.add("My bicycle is green.")
        hits = mem.search("what does the user ride", k=3)
        turns = mem.recall("bicycle", include_episodes=True)
    assert any(hit.claim.object == "bicycle" for hit in hits)
    assert "bicycle" in turns


def test_an_invented_fact_is_still_rejected_without_an_embedder_to_vouch_for_it():
    """`reject_ungrounded="auto"` keeps a claim that shares no words with its turn only
    when the embedder says it is a paraphrase. With no embedder, nothing vouches for it,
    so the default guard against invented facts still rejects it. A first version kept
    every such claim, which switched that guard off for good."""
    from test_pipeline import CountingLLM

    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "likes", "object": "tea", "polarity": 1,
         "memory_type": "semantic", "confidence": 0.9, "source_index": 0}])
    mem = Memvara(embeddings=False, llm=llm)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        receipt = mem.add("The quarterly review is next Tuesday.")
    assert receipt.added == [] and receipt.ungrounded == 1


def test_replacement_advice_stops_when_the_vectors_are_dropped_later(tmp_path):
    """`reembed(NoEmbedder())` drops the vectors from an instance built to advise. Its
    next write asks no judge and warns of no failure."""
    from test_advisory import Judge

    judge = Judge(replaces=lambda new, old: True)
    with Memvara(str(tmp_path / "s.db"), embedder=HashingEmbedder(dim=64), llm=judge,
                 advise_replacements=True) as mem:
        mem.remember("user", "lives_in", "Lisbon")
        mem.reembed(NoEmbedder())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            receipt = mem.remember("user", "works_at", "Acme")
    # It closed nothing, so it is a write the advisor is asked about.
    assert receipt.added and not receipt.closed
    assert receipt.may_replace == [] and judge.pairs == []


def test_reembed_on_a_store_with_no_vectors_does_not_need_it_to_itself(tmp_path):
    """There is nothing to drop, so a second handle opening with `reembed=True` out of
    habit is not refused while the first is open."""
    db = str(tmp_path / "s.db")
    with Memvara(db, embeddings=False, llm=NullLLM()) as first:
        first.remember("user", "lives_in", "Lisbon")
        with Memvara(db, embeddings=False, llm=NullLLM(), reembed=True) as second:
            assert second.stats()["claims"] == 1


def test_embeddings_takes_only_true_or_false():
    with pytest.raises(TypeError, match="takes True or False"):
        Memvara(embeddings=None, llm=NullLLM())  # type: ignore[arg-type]


def test_the_merge_step_is_still_counted_at_zero():
    from memvara.telemetry import CONSOLIDATE_MERGED, MemoryRecorder

    recorder = MemoryRecorder()
    mem = Memvara(embeddings=False, llm=NullLLM(), telemetry=recorder)
    mem.remember("user", "likes", "green tea")
    mem.remember("user", "likes", "black coffee")
    mem.consolidate()
    assert any(name == CONSOLIDATE_MERGED for name, *_ in recorder.counters)
    assert recorder.total(CONSOLIDATE_MERGED) == 0


def test_the_crewai_backend_refuses_a_store_without_vectors():
    from memvara.integrations.crewai import MemvaraStorage

    with pytest.raises(TypeError, match="CrewAI searches by vector"):
        MemvaraStorage(Memvara(embeddings=False, llm=NullLLM()))


def test_the_server_says_how_to_open_a_store_written_without_vectors(tmp_path):
    import io

    from memvara.server.cli import main

    path = str(tmp_path / "memory.db")
    with Memvara(path, embeddings=False, llm=NullLLM()) as mem:
        mem.remember("user", "lives_in", "Lisbon")
    err = io.StringIO()
    assert main([], env={"MEMVARA_DB": path, "MEMVARA_EMBEDDER": "hashing"},
                stdout=io.StringIO(), stderr=err) == 2
    assert "reembed=True" in err.getvalue()
    assert "set MEMVARA_EMBEDDER" not in err.getvalue()


def test_the_agentic_search_tool_answers_by_text_alone():
    from memvara.store import SQLiteStore
    from memvara.write.agentic import AgenticExtractor, _Session

    store = SQLiteStore(":memory:")
    scope = Scope("default")
    with Memvara(store=store, embeddings=False, llm=NullLLM()) as mem:
        mem.remember("office", "located_in", "Porto")
        session = _Session(
            extractor=AgenticExtractor(NullLLM(), store, NoEmbedder()),
            episodes=[Episode(role="user", content="where is the office", scope=scope)],
            now=mem.history("office", "located_in")[0].valid_from)
        assert "Porto" in session.search_memories({"query": "office Porto"})
