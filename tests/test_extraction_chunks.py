"""Extracting a long turn in pieces: the `extraction_chunks` switch.

With the switch on, a turn longer than `EXTRACTION_CHUNK_CHARS` (6,000 characters) is sent
to the model one piece at a time. Every claim still cites the whole episode, so `why()`
keeps pointing at the turn the user wrote, and a claim that two pieces both produced is
merged into one before reconciliation. The switch is off by default because the release
bar in `docs/ROADMAP.md` ("Reversed" list) has not been met; the last group of tests
below holds the evidence the fixture allows.

Every test counts model calls, as `tests/test_pipeline.py` does, because a call per piece
is the cost this feature adds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import pytest

from memvara import Memvara
from memvara.embed import HashingEmbedder
from memvara.llm.base import Usage
from memvara.schema import PredicateRegistry
from memvara.server.config import (
    FEATURES,
    FEATURES_OFF_BY_DEFAULT,
    ConfigError,
    ServerConfig,
    build_memvara,
)
from memvara.store import SQLiteStore
from memvara.types import Episode, Scope
from memvara.write import WritePipeline
from memvara.write import split
from memvara.write.split import EXTRACTION_CHUNK_CHARS, split_for_extraction

SCOPE = Scope("acme", "alice")
FIXTURE = Path(__file__).parent / "fixtures" / "phi4_spike"


def item(subject: str, predicate: str, obj: str, *, index: int = 0,
         confidence: float = 0.9) -> dict[str, Any]:
    return {"subject": subject, "predicate": predicate, "object": obj, "polarity": 1,
            "memory_type": "semantic", "confidence": confidence, "source_index": index}


class PieceLLM:
    """Answers each call from a function of the turns it was handed, and records them."""

    name = "fake/pieces"

    def __init__(self, respond) -> None:
        self._respond = respond
        self.calls: list[list[Episode]] = []
        self.classified = 0

    def extract(self, episodes, known_predicates):
        self.calls.append(list(episodes))
        return self._respond(episodes)

    def classify_predicate(self, predicate, example):
        self.classified += 1
        return {"cardinality": "many", "volatility": "slow", "memory_type": "semantic"}


def build(llm, **kw):
    store = SQLiteStore(":memory:")
    pipe = WritePipeline(store, HashingEmbedder(), PredicateRegistry(), llm, **kw)
    return pipe, store


def ep(content: str) -> Episode:
    return Episode(content=content, scope=SCOPE)


def long_turn(*facts: str, filler_sentences: int = 50) -> str:
    """A turn over 6,000 characters with `facts` spread across it, one per stretch."""
    filler = "The rest of this note is ordinary working detail about the week. "
    parts: list[str] = []
    for fact in facts:
        parts.append(filler * filler_sentences)
        parts.append(fact + " ")
    parts.append(filler * filler_sentences)
    text = "".join(parts).strip()
    assert len(text) > EXTRACTION_CHUNK_CHARS
    return text


def live(store) -> list[tuple[str, str]]:
    return sorted((c.predicate, c.object) for c in store.iter_claims("acme") if c.is_live())


# -- the splitter ------------------------------------------------------------------------


def test_a_turn_at_the_limit_is_one_piece_and_is_not_touched():
    text = "  x" * (EXTRACTION_CHUNK_CHARS // 3)
    assert len(text) == EXTRACTION_CHUNK_CHARS
    assert split_for_extraction(text) == [text], "not even stripped"


def test_pieces_cut_after_a_sentence_and_lose_no_word():
    text = long_turn("The offsite is booked in Lisbon for May.", "The gate runs on port 61434.")
    pieces = split_for_extraction(text)
    assert len(pieces) == 2
    assert all(len(p) <= EXTRACTION_CHUNK_CHARS for p in pieces)
    assert all(p.endswith(".") for p in pieces), "every piece ends on a sentence"
    assert " ".join(pieces).split() == text.split(), "no word lost or added"


def test_a_line_break_is_a_boundary_too():
    text = "first line without a stop\nsecond line without a stop\nthird"
    assert split_for_extraction(text, limit=30) == [
        "first line without a stop", "second line without a stop", "third"]


def test_a_word_longer_than_the_limit_is_cut_at_the_limit():
    assert split_for_extraction("abcdefghij", limit=4) == ["abcd", "efgh", "ij"]


def test_whitespace_between_pieces_does_not_become_a_piece():
    assert split_for_extraction("One.    \n\n   Two.", limit=5) == ["One.", "Two."]


def test_a_limit_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        split_for_extraction("anything", limit=0)


def test_the_limit_is_read_when_called_so_it_can_be_changed_in_one_place(monkeypatch):
    monkeypatch.setattr(split, "EXTRACTION_CHUNK_CHARS", 10)
    assert split_for_extraction("Four words go here. And two more.") == [
        "Four words", "go here.", "And two", "more."]


# -- the switch is off unless asked for ---------------------------------------------------


def test_off_by_default_a_long_turn_is_one_call_with_the_whole_text():
    llm = PieceLLM(lambda eps: [])
    pipe, store = build(llm)
    assert pipe.extraction_chunks is False
    turn = ep(long_turn("The offsite is booked in Lisbon for May."))
    receipt = pipe.add([turn])
    assert [[e.content for e in call] for call in llm.calls] == [[turn.content]]
    assert receipt.llm_calls == 1
    store.close()


def test_on_a_turn_under_the_limit_is_still_one_call_for_the_whole_batch():
    llm = PieceLLM(lambda eps: [item("user", "lives_in", "Lisbon")])
    pipe, store = build(llm, extraction_chunks=True)
    receipt = pipe.add([ep("The offsite is booked in Lisbon for May."), ep("We adopted a cat, Miso.")])
    assert [len(call) for call in llm.calls] == [2]
    assert receipt.llm_calls == 1
    store.close()


# -- a long turn, in pieces ---------------------------------------------------------------


def _answer_by_content(eps: Sequence[Episode]) -> list[dict[str, Any]]:
    """What a model would find in each turn it is shown: the planted facts, by text."""
    out = []
    for i, e in enumerate(eps):
        if "Lisbon" in e.content:
            out.append(item("user", "lives_in", "Lisbon", index=i))
        if "61434" in e.content:
            out.append(item("gate", "port", "61434", index=i))
        if "Miso" in e.content:
            out.append(item("user", "owns_pet", "Miso", index=i))
    return out


def test_on_each_piece_is_its_own_call_and_every_claim_cites_the_whole_turn():
    llm = PieceLLM(_answer_by_content)
    pipe, store = build(llm, extraction_chunks=True)
    turn = ep(long_turn("The offsite is booked in Lisbon for May.", "The gate runs on port 61434."))
    receipt = pipe.add([turn])

    assert [len(call) for call in llm.calls] == [1, 1], "one call per piece"
    pieces = [call[0] for call in llm.calls]
    assert [p.content for p in pieces] == split_for_extraction(turn.content)
    assert {p.id for p in pieces} == {turn.id}, "each piece carries the turn's id"
    assert receipt.llm_calls == 2 + llm.classified, "two extractions, plus acquisition"
    assert live(store) == [("lives_in", "Lisbon"), ("port", "61434")]
    for claim in receipt.added:
        assert claim.sources == [turn.id]
    assert store.get_episode(turn.id).content == turn.content, "the turn is stored whole"
    store.close()


def test_short_turns_share_one_call_and_long_ones_are_cut_and_indexes_map_back():
    llm = PieceLLM(_answer_by_content)
    pipe, store = build(llm, extraction_chunks=True)
    short_a = ep("We adopted a cat, Miso, last spring.")
    big = ep(long_turn("The offsite is booked in Lisbon for May.", "The gate runs on port 61434."))
    short_b = ep("Nothing much else happened, but the gate still binds 61434.")
    receipt = pipe.add([short_a, big, short_b])

    assert [len(call) for call in llm.calls] == [2, 1, 1]
    assert [e.id for e in llm.calls[0]] == [short_a.id, short_b.id]
    cited = sorted((c.predicate, c.object, c.sources[0]) for c in receipt.added)
    assert ("owns_pet", "Miso", short_a.id) in cited
    assert ("lives_in", "Lisbon", big.id) in cited
    assert ("port", "61434", big.id) in cited
    # `short_b` states the same port; it is a different turn, so its claim is not merged
    # with the big turn's and reconciles as a reinforcement of it.
    assert ("port", "61434") in [(c.predicate, c.object) for c in receipt.reinforced]
    assert receipt.llm_calls == 3 + llm.classified
    store.close()


def test_a_source_index_outside_the_piece_is_dropped_not_moved():
    """A piece is a batch of one, so only index 0 is valid. Index 1 names no turn in
    that call; mapping it to the batch's second turn would misattribute the claim."""
    def respond(eps):
        return [item("user", "lives_in", "Lisbon", index=1)] if len(eps) == 1 else []

    llm = PieceLLM(respond)
    pipe, store = build(llm, extraction_chunks=True)
    receipt = pipe.add([ep(long_turn("The offsite is booked in Lisbon for May.")),
                        ep("The offsite is booked in Lisbon for May, as I said.")])
    assert receipt.added == []
    store.close()


# -- merging across pieces ----------------------------------------------------------------


def test_the_same_claim_from_two_pieces_is_merged_before_reconciliation():
    """The measured failure in the ROADMAP entry: each piece restates what the model
    already said, and the reconciler absorbs the repeats as reinforcements, which raises
    the claim's salience for no new evidence. Merged, it is one claim at the higher of
    the two confidences."""
    def respond(eps):
        conf = 0.6 if "Lisbon" in eps[0].content else 0.8
        return [item("user", "lives_in", "Lisbon", confidence=conf)]

    llm = PieceLLM(respond)
    pipe, store = build(llm, extraction_chunks=True)
    receipt = pipe.add([ep(long_turn("The offsite is booked in Lisbon for May.",
                                     "The gate runs on port 61434."))])
    assert len(llm.calls) == 2
    assert [(c.predicate, c.object) for c in receipt.added] == [("lives_in", "Lisbon")]
    assert receipt.reinforced == [], "a merged repeat is not a second observation"
    assert receipt.added[0].confidence == pytest.approx(0.8)
    store.close()


def test_a_merge_needs_the_same_value_and_the_same_slot():
    """Two pieces giving two different values for one slot are two claims, and the
    reconciler decides between them as it would for any two claims."""
    def respond(eps):
        value = "Lisbon" if "Lisbon" in eps[0].content else "61434"
        return [item("user", "mentions", value), item("gate", "mentions", value)]

    llm = PieceLLM(respond)
    pipe, store = build(llm, extraction_chunks=True)
    pipe.add([ep(long_turn("The offsite is booked in Lisbon for May.", "The gate runs on port 61434."))])
    assert sorted((c.subject, c.object) for c in store.iter_claims("acme")) == [
        ("gate", "61434"), ("gate", "Lisbon"), ("user", "61434"), ("user", "Lisbon")]
    store.close()


def test_a_repeat_inside_one_piece_is_left_to_the_reconciler_as_before():
    """The merge is for repeats across pieces. Inside one call nothing changes, so a
    turn that is not cut behaves exactly as it did before the switch existed."""
    def respond(eps):
        if "Lisbon" not in eps[0].content:
            return []
        return [item("gate", "port", "61434"), item("gate", "port", "61434")]

    llm = PieceLLM(respond)
    pipe, store = build(llm, extraction_chunks=True, reject_polluted=False)
    receipt = pipe.add([ep(long_turn("The offsite is booked in Lisbon for May.",
                                     "The gate runs on port 61434."))])
    assert len(receipt.added) == 1 and len(receipt.reinforced) == 1
    store.close()


# -- failure and cost ---------------------------------------------------------------------


def test_a_failed_piece_defers_the_whole_batch_so_a_retry_reads_the_turn_again():
    """No claim from the pieces that did answer is kept. `reextract()` skips a turn that
    already has claims, so keeping half a turn's claims would stop the retry from ever
    reading the other half."""
    class FailsSecondPiece(PieceLLM):
        def extract(self, episodes, known_predicates):
            if len(self.calls) == 1:
                self.calls.append(list(episodes))
                raise RuntimeError("429 rate limited")
            return super().extract(episodes, known_predicates)

    llm = FailsSecondPiece(_answer_by_content)
    mem = Memvara(":memory:", llm=llm, embedder=HashingEmbedder(dim=32),
                  write_extraction_chunks=True, tenant="acme", user="alice")
    receipt = mem.add(long_turn("The offsite is booked in Lisbon for May.", "The gate runs on port 61434."))
    assert receipt.deferred and receipt.unextracted == 1
    assert receipt.added == [] and receipt.llm_calls == 2, "no acquisition was reached"
    assert len(mem.pending_extraction()) == 1

    retry = mem.reextract()
    assert sorted((c.predicate, c.object) for c in retry.added) == [
        ("lives_in", "Lisbon"), ("port", "61434")]
    assert retry.llm_calls == 2 + llm.classified
    mem.close()


def test_tokens_from_every_piece_land_on_one_receipt():
    class Metered(PieceLLM):
        reports_usage = True

        def extract(self, episodes, known_predicates, *, usage=None):
            usage.add(input_tokens=1000, output_tokens=50)
            return super().extract(episodes, known_predicates)

    llm = Metered(lambda eps: [])
    pipe, store = build(llm, extraction_chunks=True)
    receipt = pipe.add([ep(long_turn("The offsite is booked in Lisbon for May.",
                                     "The gate runs on port 61434."))])
    assert (receipt.tokens_in, receipt.tokens_out) == (2000, 100)
    store.close()


def test_a_backend_that_consults_no_model_is_not_called_for_any_piece():
    from memvara.llm import NullLLM
    pipe, store = build(NullLLM(), extraction_chunks=True)
    receipt = pipe.add([ep(long_turn("The offsite is booked in Lisbon for May."))])
    assert (receipt.llm_calls, receipt.unextracted) == (0, 1)
    store.close()


# -- how a deployment turns it on ---------------------------------------------------------


def test_the_option_reaches_the_pipeline_through_memvara_s_write_prefix():
    mem = Memvara(":memory:", llm=PieceLLM(lambda eps: []), embedder=HashingEmbedder(dim=32),
                  write_extraction_chunks=True)
    assert mem.writer.extraction_chunks is True
    mem.close()


def test_the_feature_is_listed_and_off_unless_the_environment_turns_it_on():
    assert "extraction_chunks" in FEATURES
    assert FEATURES_OFF_BY_DEFAULT == frozenset({"extraction_chunks"})
    assert "extraction_chunks" in ServerConfig().features_off
    assert "extraction_chunks" in ServerConfig.from_env({"MEMVARA_DB": ":memory:"}).features_off
    on = ServerConfig.from_env({"MEMVARA_DB": ":memory:", "MEMVARA_FEATURE_EXTRACTION_CHUNKS": "1"})
    assert "extraction_chunks" not in on.features_off
    blank = ServerConfig.from_env({"MEMVARA_DB": ":memory:", "MEMVARA_FEATURE_EXTRACTION_CHUNKS": ""})
    assert "extraction_chunks" in blank.features_off, "blank means the default"
    # A feature that is on by default is still switched off with 0, and back on with 1.
    assert "profile" in ServerConfig.from_env({"MEMVARA_DB": ":memory:", "MEMVARA_FEATURE_PROFILE": "0"}).features_off
    assert "profile" not in ServerConfig.from_env(
        {"MEMVARA_DB": ":memory:", "MEMVARA_FEATURE_PROFILE": "1"}).features_off
    with pytest.raises(ConfigError, match="MEMVARA_FEATURE_EXTRACTION_CHUNKS"):
        ServerConfig.from_env({"MEMVARA_DB": ":memory:", "MEMVARA_FEATURE_EXTRACTION_CHUNKS": "maybe"})


@pytest.mark.parametrize("env, expected", [({}, False),
                                           ({"MEMVARA_FEATURE_EXTRACTION_CHUNKS": "1"}, True)])
def test_the_server_builds_its_memory_with_the_switch_it_was_given(env, expected):
    config = ServerConfig.from_env({"MEMVARA_DB": ":memory:", **env})
    mem = build_memvara(config)
    assert mem.writer.extraction_chunks is expected
    mem.close()


# -- the release bar, and what the recorded fixture can say about it ----------------------
#
# The bar, from the ROADMAP entry this feature reverses: on the longest episode in
# `tests/fixtures/phi4_spike/`, chunked extraction must recover 5 of 5 key facts with 0
# duplicates. The only chunked run ever measured was on that episode padded to 13,687
# characters, with `phi-4-mini`, and it found 4 of 5 with 3 duplicates. Its per-piece
# outputs were not kept, so they cannot be replayed here. What the fixture does hold is
# the model's verbatim output for each of the three turns, and the tests below use it for
# the two things it can answer.

KEYS = json.loads((FIXTURE / "keys.json").read_text())
TURNS = {e["name"]: e["content"] for e in json.loads((FIXTURE / "episodes.json").read_text())}
CONFIGS = ("12", "24", "32", "none", "A", "B")


def _recorded(config: str, name: str) -> list[dict[str, Any]]:
    return json.loads((FIXTURE / "claims" / f"{config}-{name}.json").read_text())


def _score(claims: Sequence[dict[str, Any]], names: Sequence[str]) -> tuple[int, int]:
    """Key facts found and duplicates, the way the spike that set the bar scored them:
    a fact is found when a claim's object matches its pattern, and a duplicate is a
    second claim with the same predicate and object."""
    blob = [str(c["object"]) for c in claims]
    facts = [f for name in names for f in KEYS["facts"][name]]
    found = sum(1 for f in facts if any(re.search(f["object_re"], o, re.I) for o in blob))
    pairs = [(c["predicate"], str(c["object"]).lower()) for c in claims]
    return found, len(pairs) - len(set(pairs))


def _replay(config: str):
    """A backend that answers a turn with what `phi-4-mini` answered for that exact text."""
    by_text = {TURNS[name]: _recorded(config, name) for name in TURNS}

    def respond(eps):
        out = []
        for i, e in enumerate(eps):
            out.extend({**c, "source_index": i} for c in by_text[e.content])
        return out
    return PieceLLM(respond)


def _stored(store) -> list[dict[str, Any]]:
    return [{"predicate": c.predicate, "object": c.object}
            for c in store.iter_claims("acme")]


def test_the_fixture_s_longest_turn_is_under_the_threshold_so_it_cannot_test_the_bar():
    """The longest turn in the fixture is 902 characters. The switch leaves anything
    under 6,000 whole, so the fixture as recorded cannot exercise the bar: whatever it
    scores with the switch on, it scores with the switch off."""
    longest = max(TURNS.values(), key=len)
    assert len(longest) == 902 < EXTRACTION_CHUNK_CHARS
    assert split_for_extraction(longest) == [longest]
    for config in CONFIGS:
        results = []
        for chunks in (False, True):
            llm = _replay(config)
            pipe, store = build(llm, extraction_chunks=chunks)
            pipe.add([ep(longest)])
            results.append((len(llm.calls), _score(_stored(store), ["gate"])))
            store.close()
        assert results[0] == results[1], config


def test_recorded_output_replayed_through_the_pieces_loses_nothing_and_adds_no_duplicate(
        monkeypatch):
    """The three recorded turns, joined into one turn and cut back into exactly those
    three pieces, so every piece is a text the model was really shown and its answer is
    recorded. The limit is lowered to 1,000 characters for this, because the joined turn
    is 2,481 characters, and 1,000 is the one value at which the pieces fall on the turn
    boundaries.

    What this measures is the pipeline's side of chunking with real model output: the
    index mapping, the merge and the guards. What it cannot measure is the model's side,
    which is where the ROADMAP run lost its fact: how a model answers a piece it was cut
    from a longer text. Replaying a whole turn's answer as a piece's answer is exact here
    only because each piece is a whole recorded turn."""
    monkeypatch.setattr(split, "EXTRACTION_CHUNK_CHARS", 1000)
    names = ("gate", "billing", "retrieval")
    joined = "\n\n".join(TURNS[n] for n in names)
    assert len(joined) == 2481
    assert split_for_extraction(joined) == [TURNS[n] for n in names]

    measured = {}
    for config in CONFIGS:
        llm = _replay(config)
        pipe, store = build(llm, extraction_chunks=True)
        receipt = pipe.add([ep(joined)])
        assert len(llm.calls) == 3 and receipt.llm_calls >= 3
        chunked = _score(_stored(store), names)
        store.close()

        pipe, store = build(_replay(config))
        for n in names:
            pipe.add([ep(TURNS[n])])
        separate = _score(_stored(store), names)
        store.close()
        measured[config] = (chunked, separate)

    # (key facts found of 15, duplicates): in pieces, then as three separate turns.
    assert measured == {
        "12": ((13, 0), (13, 0)),
        "24": ((13, 0), (13, 0)),
        "32": ((13, 0), (13, 0)),
        "none": ((13, 0), (13, 0)),
        "A": ((13, 0), (13, 0)),
        "B": ((8, 0), (8, 0)),
    }


def test_the_switch_ships_off_because_the_bar_is_not_met():
    """The one measurement of chunked extraction on the bar's episode found 4 of 5 key
    facts. Merging removes duplicates of the same claim, so it could bring the 3
    duplicates to 0, but it cannot bring back a fact the model never stated. Until a
    recorded run on a long turn meets 5 of 5 with 0 duplicates, the switch stays off by
    default, and `memory_stats` on a server says so."""
    assert "extraction_chunks" in FEATURES_OFF_BY_DEFAULT
    pipe, store = build(PieceLLM(lambda eps: []))
    assert pipe.extraction_chunks is False
    store.close()
