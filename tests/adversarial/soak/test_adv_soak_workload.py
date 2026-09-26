"""The soak's workload: seeded, inside its own vocabulary, and planting what each detector needs.

A detector can only stay quiet on a healthy store if the workload gives it something to
measure, so these tests check the plants themselves: the near-duplicate preferences the
merge pass must fold, the personal data the redactor must remove, the retractions that
must each close a live value, and the probes whose right answer the workload knows.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime, timezone

import pytest

import soak
from memvara import Memvara, NullLLM
from memvara.embed import HashingEmbedder
from memvara.embed.calibration import calibration_of
from memvara.redact import EPISODE, PatternRedactor
from memvara.schema import PredicateRegistry
from memvara.telemetry import script_of
from memvara.types import Episode, Scope
from memvara.write.fast import FastExtractor

LIKE = re.compile(r"I like (\w+)\.")
UNLIKE = re.compile(r"I no longer like (\w+)\.")


@pytest.fixture(scope="module")
def turns() -> list[soak.Turn]:
    return list(soak.Workload(soak.SoakConfig(2_000)))


def test_the_same_seed_gives_the_same_turns_and_another_seed_does_not() -> None:
    first = list(soak.Workload(soak.SoakConfig(300, seed=1)))
    again = list(soak.Workload(soak.SoakConfig(300, seed=1)))
    other = list(soak.Workload(soak.SoakConfig(300, seed=2)))
    assert len(first) == 300 and first == again and first != other


def test_a_soak_shorter_than_its_introductions_is_refused() -> None:
    with pytest.raises(ValueError, match="100"):
        soak.SoakConfig(99)


@pytest.mark.parametrize("turns, per_day", [(100, 5), (200, 10), (10_000, 476),
                                            (100_000, 4_762)])
def test_consolidation_runs_once_per_simulated_day(turns: int, per_day: int) -> None:
    assert soak.SoakConfig(turns).per_day == per_day


def test_every_predicate_the_workload_writes_is_in_its_vocabulary(
        turns: list[soak.Turn]) -> None:
    registry = PredicateRegistry()
    fast = FastExtractor(registry)
    used: set[str] = set()
    for turn in turns:
        if turn.kind == "remember":
            used.add(registry.normalize(turn.predicate))
        elif turn.kind == "say" and turn.script == "latin" and turn.fact:
            episode = Episode(content=turn.text, role="user", scope=Scope(),
                              ts=datetime.now(timezone.utc))
            claims = fast.extract(episode)
            # Every Latin fact the workload says is one the fast path recognises,
            # except the planted personal data, which carries no fact.
            assert bool(claims) != turn.planted, turn.text
            used.update(claim.predicate for claim in claims)
    assert used == set(soak.VOCABULARY)


def test_the_workload_spells_predicates_with_their_aliases(turns: list[soak.Turn]) -> None:
    spellings = {turn.predicate for turn in turns if turn.kind == "remember"}
    assert spellings - set(soak.VOCABULARY), "a vocabulary with no aliases cannot explode"


def test_each_preference_and_its_near_duplicates_reach_the_merge_threshold() -> None:
    embedder = HashingEmbedder(dim=512)
    threshold = calibration_of(embedder).merge
    for sentence in soak.PREFERENCES:
        for which in (0, 1):
            near = soak.variant(sentence, which)
            assert near != sentence
            vectors = embedder.encode([f"user prefers {sentence}", f"user prefers {near}"])
            assert float(vectors[0] @ vectors[1]) >= threshold, near


def test_a_preference_and_its_near_duplicate_stay_two_claims_until_a_merge() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mem = Memvara(embedder=HashingEmbedder(dim=512), llm=NullLLM(), user="soak")
    sentence = soak.PREFERENCES[0]
    mem.remember("user", "prefers", sentence)
    mem.remember("user", "prefers", soak.variant(sentence, 0))
    assert len(mem.get_all()) == 2
    assert mem.consolidate()["merged"] == 1
    assert len(mem.get_all()) == 1
    mem.close()


def test_personal_data_is_planted_exactly_where_a_turn_says(
        turns: list[soak.Turn]) -> None:
    redactor = PatternRedactor()
    planted = 0
    for turn in turns:
        if turn.kind != "say":
            continue
        changed = redactor.redact(turn.text, field=EPISODE, scope=Scope()) != turn.text
        assert changed == turn.planted, turn.text
        planted += turn.planted
    assert planted > 0


def test_every_retraction_takes_back_a_like_that_is_still_held(
        turns: list[soak.Turn]) -> None:
    held: set[str] = set()
    taken_back: set[str] = set()
    for turn in turns:
        if turn.kind != "say":
            continue
        if (like := LIKE.fullmatch(turn.text)) is not None:
            assert like[1] not in taken_back, "a like taken back is never said again"
            held.add(like[1])
        elif (unlike := UNLIKE.fullmatch(turn.text)) is not None:
            assert unlike[1] in held, unlike[1]
            held.remove(unlike[1])
            taken_back.add(unlike[1])
    assert taken_back


def test_every_probe_asks_for_the_city_the_workload_last_gave(
        turns: list[soak.Turn]) -> None:
    registry = PredicateRegistry()
    cities: dict[str, str] = {}
    probes = 0
    for turn in turns:
        if turn.kind == "remember" and registry.normalize(turn.predicate) == "lives_in":
            cities[turn.subject] = turn.obj
        elif turn.kind == "probe":
            assert turn.gold is not None
            person = turn.gold[0]
            assert turn.gold == (person, "lives_in", cities[person])
            assert person in turn.text
            probes += 1
    assert probes > 0


def test_each_non_latin_turn_is_written_in_the_script_it_names(
        turns: list[soak.Turn]) -> None:
    seen = set()
    for turn in turns:
        if turn.kind == "say" and turn.script != "latin":
            assert turn.fact and script_of(turn.text) == turn.script, turn.text
            seen.add(turn.script)
    assert seen == {"arabic", "cyrillic", "devanagari", "greek", "han", "hangul", "kana"}
