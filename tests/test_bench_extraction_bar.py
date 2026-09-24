"""bench/extraction_bar.py: the release-bar measurement for agentic extraction.

The script runs the same turns through two stores that differ only in the
`agentic_extraction` switch, with the same model, and compares what each wrote. Its
second mode does the same over the 199-question LongMemEval sample and compares judged
accuracy against the recorded noise floor. Both modes need a model key to produce a
number, so everything here runs against scripted models with no network: the counting,
the fallback reporting, the verdicts, the resume, and the refusal to start without a key.

`bench/` is outside the coverage gate, so these tests are the only guard on the script.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import extraction_bar as bar  # noqa: E402
import longmemeval as lme  # noqa: E402

from memvara.types import Claim, Scope  # noqa: E402
from memvara.write.agentic import AGENTIC_SYNC_TIMEOUT, AGENTIC_TIMEOUT  # noqa: E402

T0 = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)

#: Turns the salience gate passes and the fast path does not recognise, so every one of
#: them reaches the model tier.
TURNS = [
    "The team relocated the whole office to Porto over the summer.",
    "Our deploy target is the Frankfurt cluster, and the build uses 8 threads.",
    "The quarterly review with the auditors moved to the second week of April.",
]


def batches(texts: Sequence[str] = TURNS) -> list[list[dict[str, Any]]]:
    return [[{"role": "user", "content": text, "ts": T0 + timedelta(days=i)}]
            for i, text in enumerate(texts)]


class PlainModel:
    """A backend with `extract` and nothing else, so agentic extraction cannot run on it."""

    name = "fake/plain"
    is_noop = False
    reports_usage = False
    accepts_guidance = True

    def extract(self, episodes, known_predicates, guidance=None):
        return [{"subject": "user", "predicate": "mentioned",
                 "object": " ".join(ep.content.split()[:3]).lower(), "source_index": i,
                 "confidence": 0.9} for i, ep in enumerate(episodes)]

    def classify_predicate(self, predicate, example):
        return {"cardinality": "many", "volatility": "slow", "memory_type": "semantic"}


def claim(subject: str, predicate: str, obj: str) -> Claim:
    return Claim(subject=subject, predicate=predicate, object=obj,
                 scope=Scope("default", "customer"))


# -- the sample -------------------------------------------------------------------------


def test_the_sample_is_the_published_199_questions():
    ids = bar.load_sample()
    assert len(ids) == 199 and len(set(ids)) == 199
    assert ids == sorted(ids)
    assert sum(1 for qid in ids if qid.endswith("_abs")) == 11


def test_a_dataset_missing_a_sample_question_is_refused_by_name():
    items = lme.fixture()
    with pytest.raises(SystemExit, match="fx_missing"):
        bar.select_sample(items, ["fx_single_user", "fx_missing"])


def test_the_sample_keeps_its_own_order_and_nothing_else():
    items = lme.fixture()
    chosen = bar.select_sample(items, ["fx_temporal_abs", "fx_single_user"])
    assert [i.qid for i in chosen] == ["fx_temporal_abs", "fx_single_user"]


# -- counting duplicates ----------------------------------------------------------------


def test_a_value_repeated_in_one_slot_counts_once_per_extra_copy():
    live = [claim("user", "lives_in", "Porto"), claim("user", "lives_in", " porto. "),
            claim("user", "lives_in", "Porto"), claim("user", "lives_in", "Lisbon")]
    assert bar.duplicates(live) == (2, 0)


def test_a_value_filed_under_a_second_predicate_is_counted_apart():
    live = [claim("user", "lives_in", "Porto"), claim("user", "home_city", "Porto"),
            claim("acme", "based_in", "Porto")]
    assert bar.duplicates(live) == (0, 1)


# -- the claims comparison --------------------------------------------------------------


def test_both_arms_read_the_same_turns_with_the_same_model():
    model = bar.RehearsalChat()
    tallies = bar.compare_claims(batches(), model)
    single, agentic = tallies["single_call"], tallies["agentic"]
    assert single.turns == agentic.turns == 3
    assert single.writes == agentic.writes == 3
    assert single.agentic_runs == 0 and single.fallbacks == {}
    assert agentic.agentic_runs == 3 and agentic.fallbacks == {}
    assert single.claims_written == agentic.claims_written == 3
    assert single.live_claims == agentic.live_claims == 3
    assert model.extract_calls == 3 and model.tool_runs == 3


def test_a_backend_without_tools_shows_up_as_fallbacks_and_the_bar_is_not_measured():
    tallies = bar.compare_claims(batches(), PlainModel())
    agentic = tallies["agentic"]
    assert agentic.agentic_runs == 0
    assert agentic.fallbacks == {"unsupported": 3}
    verdict, why = bar.claims_verdict(tallies)
    assert verdict == "not measured"
    assert "unsupported" in why


def test_proposals_the_write_refused_are_counted_by_reason():
    model = bar.RehearsalChat(end_unread=True)
    agentic = bar.compare_claims(batches(), model)["agentic"]
    assert agentic.refused == {"not_read": 3}


def test_the_worker_path_reads_stored_turns_with_the_background_budget():
    """The hosted worker stores turns first and reads them later through `reextract()`,
    which gives the loop 180 seconds. `add()` gives it 25, because a caller is waiting."""
    via_add = bar.RehearsalChat()
    bar.compare_claims(batches(), via_add, path="add")
    via_worker = bar.RehearsalChat()
    tallies = bar.compare_claims(batches(), via_worker, path="worker", batch=1)
    assert set(via_add.timeouts) == {AGENTIC_SYNC_TIMEOUT}
    assert set(via_worker.timeouts) == {AGENTIC_TIMEOUT}
    assert tallies["agentic"].writes == 3 and tallies["agentic"].claims_written == 3


def test_the_worker_path_batches_as_asked():
    model = bar.RehearsalChat()
    tallies = bar.compare_claims(batches(), model, path="worker", batch=2)
    assert tallies["agentic"].writes == 2
    assert tallies["agentic"].claims_written == 3


def test_the_claims_bar_is_met_only_with_no_fewer_claims_and_no_more_duplicates():
    def pair(single: dict[str, int], agentic: dict[str, int]) -> dict[str, bar.Tally]:
        base = {"writes": 3, "turns": 3, "agentic_runs": 0}
        return {"single_call": bar.Tally(**{**base, **single}),
                "agentic": bar.Tally(**{**base, "agentic_runs": 3, **agentic})}

    same = pair({"claims_written": 5}, {"claims_written": 5})
    assert bar.claims_verdict(same)[0] == "met"
    fewer = pair({"claims_written": 5}, {"claims_written": 4})
    assert bar.claims_verdict(fewer) == ("not met", "agentic wrote 4 claims, single-call 5")
    dupes = pair({"claims_written": 5, "duplicate_same_slot": 1},
                 {"claims_written": 6, "duplicate_same_slot": 1,
                  "duplicate_other_predicate": 1})
    assert bar.claims_verdict(dupes) == (
        "not met", "agentic left 2 duplicates, single-call 1")


def test_a_run_that_fell_back_part_of_the_time_says_how_often():
    tallies = {"single_call": bar.Tally(writes=4, turns=4, claims_written=4),
               "agentic": bar.Tally(writes=4, turns=4, claims_written=4, agentic_runs=3,
                                    fallbacks={"timeout": 1})}
    verdict, why = bar.claims_verdict(tallies)
    assert verdict == "met"
    assert "1 of 4 agentic batches fell back" in why


# -- the accuracy comparison ------------------------------------------------------------


def test_the_accuracy_bar_uses_the_eight_question_floor():
    assert bar.NOISE_FLOOR_QUESTIONS == 8
    ids = [f"q{i}" for i in range(20)]
    single = {q: i < 15 for i, q in enumerate(ids)}
    within = {q: i < 8 for i, q in enumerate(ids)}       # 7 fewer
    outside = {q: i < 7 for i, q in enumerate(ids)}      # 8 fewer
    better = {q: True for q in ids}
    assert bar.accuracy_verdict(single, within, expected=ids)[0] == "met"
    assert bar.accuracy_verdict(single, outside, expected=ids) == (
        "not met", "agentic answered 7 of 20 correctly, single-call 15: 8 fewer, and "
                   "the noise floor is 8")
    assert bar.accuracy_verdict(single, better, expected=ids)[0] == "met"


def test_the_accuracy_bar_is_not_decided_on_part_of_the_sample():
    ids = ["a", "b", "c"]
    verdict, why = bar.accuracy_verdict({"a": True, "b": True, "c": True},
                                        {"a": True, "b": True}, expected=ids)
    assert verdict == "not measured"
    assert "2 of 3" in why


def test_the_accuracy_bar_is_not_decided_without_a_judge():
    ids = ["a"]
    assert bar.accuracy_verdict({"a": None}, {"a": None}, expected=ids)[0] == "not measured"


# -- the command line -------------------------------------------------------------------


def test_a_real_run_without_a_key_is_refused_before_anything_is_spent(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        bar.main(["claims", "--llm", "openai", "--llm-model", "gpt-5.4"])
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        bar.main(["longmemeval", "--llm", "anthropic"])


def test_a_real_run_needs_an_extraction_model_named():
    with pytest.raises(SystemExit, match="--llm"):
        bar.main(["claims"])


def test_the_claims_dry_run_prints_both_arms_and_refuses_to_give_a_verdict():
    lines: list[str] = []
    assert bar.main(["claims", "--dry-run"], out=lines.append) == 0
    text = "\n".join(lines)
    assert "single_call" in text and "agentic" in text
    assert "claims written" in text and "duplicates" in text
    assert "REHEARSAL" in text
    assert "bar: not measured" in text


def test_the_longmemeval_dry_run_scores_both_arms_and_resumes_from_its_output(tmp_path):
    out = tmp_path / "bar.jsonl"
    lines: list[str] = []
    assert bar.main(["longmemeval", "--dry-run", "--out", str(out)],
                    out=lines.append) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert sorted((r["arm"], r["qid"]) for r in rows) == sorted(
        (arm, qid) for arm in bar.ARMS
        for qid in ("fx_single_user", "fx_knowledge_update", "fx_temporal_abs"))
    assert all("extraction" in r and "judged" in r for r in rows)
    text = "\n".join(lines)
    assert "noise floor" in text and "bar: not measured" in text

    again: list[str] = []
    assert bar.main(["longmemeval", "--dry-run", "--out", str(out)],
                    out=again.append) == 0
    assert len(out.read_text().splitlines()) == len(rows)
    assert "6 already scored" in "\n".join(again)
