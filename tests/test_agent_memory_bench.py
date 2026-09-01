"""The Agent Memory Benchmark, exercised end to end with no key and no network.

A benchmark is a measuring instrument, and an uncalibrated instrument is worse than
none: it produces numbers that look like measurements. So the checks here are less about
the code running and more about the two ways this instrument could be quietly wrong.

**The golds could be wrong.** `datasets/build_v1.py` authors every gold answer by hand,
and `timeline.Truth` derives the same answers from the events under published rules.
`test_every_gold_agrees_with_the_timeline_model` asserts the two agree on all 100
questions. Two independent derivations that must match is the only defence against a
scoring bug that every system fails identically, which nobody questions because the
numbers look plausible.

**The scorer could be too kind.** Lenient matching accepts a value inside a short
sentence, and that is exactly the rule that could be gamed. The tests pin both edges: an
answer naming two candidate values is wrong however confident it sounds, and an answer
longer than the token ceiling is wrong even when the gold is in it.

The rest is the ordinary surface — loading, validation, serialization, the CLI — plus an
integration test that drives the real memvara adapter over the four scenarios the whole
dataset was built around.
"""

from __future__ import annotations

import doctest
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                       # `benchmarks` is not an installed package
    sys.path.insert(0, str(ROOT))

from benchmarks.agent_memory import BENCHMARK_VERSION, DEFAULT_DATASET  # noqa: E402
from benchmarks.agent_memory import cli, dataset as ds, normalization as nz  # noqa: E402
from benchmarks.agent_memory import registry, report, results, runner, scoring, timeline  # noqa: E402
from benchmarks.agent_memory.adapters import base  # noqa: E402
from benchmarks.agent_memory.adapters.base import MemoryAnswer, Usage  # noqa: E402

UTC = timezone.utc


def at(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=UTC)


def _child_env() -> dict[str, str]:
    """The parent environment with `PYTHONPATH` pinned to the repository root.

    Derived from `os.environ` rather than replaced wholesale, which is the convention in
    `tests/test_examples.py` and `tests/test_docs.py`. A bare `env={"PYTHONPATH": ...}`
    strips PATH, TEMP and — the one that matters — SYSTEMROOT, without which the child
    interpreter commonly fails to start on Windows. The CI matrix runs windows-latest, so
    that failure would have appeared on one job and nowhere else.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    return env


#: Every dataset in the tree, and the generator that produces each. A superseded version
#: stays runnable — that is the promise in `README.md`'s *Versioning* section — so its
#: integrity is checked here on every run rather than trusted because it once passed.
SHIPPED: dict[str, str] = {"v1": "build_v1.py", "v2": "build_v2.py"}


@pytest.fixture(scope="module")
def data() -> ds.Dataset:
    """The dataset a bare run uses. Most tests want this one."""
    return ds.load()


@pytest.fixture(scope="module")
def truth(data: ds.Dataset) -> timeline.Truth:
    return timeline.Truth(data)


@pytest.fixture(params=sorted(SHIPPED), scope="module")
def shipped(request) -> ds.Dataset:
    """Each shipped dataset in turn, for the checks that must hold for all of them."""
    return ds.load(request.param)


# --- the dataset itself -----------------------------------------------------

def test_the_shipped_dataset_loads_and_is_not_trivially_small(shipped):
    """A benchmark small enough to answer by luck measures luck."""
    assert shipped.version in SHIPPED
    assert len(shipped.events) >= 200
    assert len(shipped.questions) >= 80
    assert len(shipped.scenarios) >= 10
    assert set(q.category for q in shipped.questions) == set(ds.CATEGORIES)


def test_every_category_carries_enough_questions_to_mean_something(shipped):
    """A category with two questions reports 0%, 50% or 100% and nothing in between."""
    counts = {c: sum(1 for q in shipped.questions if q.category == c)
              for c in ds.CATEGORIES}
    thin = {c: n for c, n in counts.items() if n < 5}
    assert not thin, f"too few questions to report a rate: {thin}"


@pytest.mark.parametrize("version,script", sorted(SHIPPED.items()))
def test_the_dataset_is_regenerated_byte_for_byte(tmp_path, version, script):
    """The committed files are the generator's output and nothing else.

    Run in a subprocess because the generator accumulates into module-level lists, so
    importing and calling it twice in one process would double the corpus. That is also
    the command a contributor runs, so this tests the documented path.
    """
    proc = subprocess.run(
        [sys.executable, f"benchmarks/agent_memory/datasets/{script}",
         "--out", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        env=_child_env(), timeout=300)
    assert proc.returncode == 0, proc.stderr
    committed = ROOT / "benchmarks" / "agent_memory" / "datasets" / version
    for name in ("metadata.json", "events.jsonl", "questions.jsonl"):
        fresh, kept = (tmp_path / name).read_bytes(), (committed / name).read_bytes()
        if fresh != kept and fresh.replace(b"\r\n", b"\n") == kept.replace(b"\r\n", b"\n"):
            # Said outright rather than left to the byte diff, because the generic
            # message below sends the reader to look for a content change that is not
            # there. This failed on the windows-latest job alone before `.gitattributes`
            # declared these files `eol=lf`, and a stale checkout can still produce it.
            crlf = "the generator" if b"\r\n" in fresh else "the committed copy"
            raise AssertionError(
                f"{name} differs from the committed copy by line endings and nothing "
                f"else — the content is identical, and {crlf} has CRLF. Both sides must "
                "be LF: `.gitattributes` pins this directory to `eol=lf` for the "
                f"checkout, and `{script}::_write` passes `newline=\"\\n\"` for the "
                "generator. A checkout made before that `.gitattributes` line existed "
                "needs `git rm --cached -r . && git reset --hard` to pick it up.")
        assert fresh == kept, (
            f"regenerating produced a different {name}; the committed dataset is no "
            "longer the generator's output, so the published scores cannot be traced "
            "to anything")


def test_every_gold_agrees_with_the_timeline_model(shipped):
    """The double-entry check. Authored golds against derived ones, on every question.

    A failure here means one of two independent derivations is wrong, and the run cannot
    say which — that is the point. Read the disagreement, decide which side is right, and
    fix that one.
    """
    truth = timeline.Truth(shipped)
    disagreements = []
    for question in shipped.questions:
        slot = truth.slot(*question.probe) if question.probe else None
        when = question.at or shipped.evaluated_at
        gold = question.gold
        if question.category == "negative":
            if slot is not None and slot.values_at(when, question.known_at):
                disagreements.append((question.id, "slot answers a negative question"))
            continue
        if slot is None:
            continue                              # unprobed: no slot to derive from
        if question.category == "provenance":
            carrying = [v for v in slot.versions if v.object == question.about]
            first = min(carrying, key=lambda v: v.recorded_at) if carrying else None
            got = first.source if first else None
        elif gold.kind == "date":
            kept = [v for v in slot.versions if v.object == question.about and not v.retired]
            field = "valid_from" if question.category == "change_time" else "recorded_at"
            got = {getattr(v, field).date().isoformat() for v in kept}
        elif gold.kind == "set":
            got = (slot.distinct_values if question.category == "change_detection"
                   else slot.values_at(when, question.known_at))
        else:
            got = slot.values_at(when, question.known_at)

        if gold.kind == "date":
            ok = gold.value in got
        elif gold.kind == "set":
            ok = {nz.normalize(v) for v in got} == {nz.normalize(v) for v in gold.values}
        elif question.category == "provenance":
            ok = got is not None and nz.normalize(got) == nz.normalize(gold.value)
        else:
            ok = {nz.normalize(v) for v in got} == {nz.normalize(gold.value)}
        if not ok:
            disagreements.append((question.id, f"authored {gold.to_json()}, derived {got}"))
    assert not disagreements, disagreements


#: Every unprobed chained question in the tree, written as the walk it describes: a
#: starting entity and the relations to follow. `~p` is a reverse hop — the entity whose
#: `p` is the value in hand — which is what "the service owned by team-payments" asks for.
#:
#: This is the double-entry check for the questions that have no probe, and until v2
#: there was none: `test_every_gold_agrees_with_the_timeline_model` derives from
#: `question.probe` and skips every question without one, which is all eighteen chained
#: ones. An authored gold nobody could check is a benchmark scoring against a guess.
CHAINS: dict[str, tuple[str, tuple[str, ...]]] = {
    "q-chain-atlas": ("alice", ("works_on", "deploy_region")),
    "q-chain-globex": ("bob", ("works_at", "hq_city")),
    "q-chain-lead-now": ("checkout-service", ("owned_by", "team_lead")),
    "q-chain-lead-then": ("checkout-service", ("owned_by", "team_lead")),
    "q-chain-datastore": ("team-payments", ("~owned_by", "datastore")),
    "q-chain-languages": ("Project Atlas", ("~works_on", "speaks")),
    "q2-chain-payments-lead-city": ("team-payments", ("team_lead", "lives_in")),
    "q2-chain-checkout-lead-city": ("checkout-service",
                                    ("owned_by", "team_lead", "lives_in")),
    "q2-chain-pricing-lead": ("pricing-service", ("owned_by", "team_lead")),
    "q2-chain-search-lead-languages": ("team-search", ("team_lead", "speaks")),
    "q2-chain-infra-lead-employer-city": ("team-infra",
                                          ("team_lead", "works_at", "hq_city")),
    "q2-chain-kestrel-region": ("nadia", ("works_on", "deploy_region")),
    "q2-chain-kestrel-region-then": ("nadia", ("works_on", "deploy_region")),
    "q2-chain-vantage-owner": ("omar", ("works_on", "owned_by")),
    "q2-chain-vantage-owner-lead": ("Project Vantage", ("owned_by", "team_lead")),
    "q2-chain-kestrel-lead-editor": ("Project Kestrel",
                                     ("owned_by", "team_lead", "favourite_editor")),
    "q2-chain-trust-lead-employer": ("team-trust", ("team_lead", "works_at")),
    "q2-chain-growth-lead-city": ("team-growth", ("team_lead", "lives_in")),
}


def _walk(truth: timeline.Truth, start: str, relations, when) -> set[str]:
    """Follow `relations` out of `start` at `when`, returning the values reached."""
    here = {start}
    for relation in relations:
        reached: set[str] = set()
        if relation.startswith("~"):
            predicate = relation[1:]
            for (subject, name), slot in truth.slots.items():
                if name == predicate and set(slot.values_at(when, None)) & here:
                    reached.add(subject)
        else:
            for subject in here:
                slot = truth.slot(subject, relation)
                if slot is not None:
                    reached.update(slot.values_at(when, None))
        assert reached, f"{start} runs out of graph at {relation}"
        here = reached
    return here


def test_every_chained_gold_agrees_with_the_walk_it_describes(shipped):
    """The double-entry check for questions with no probe.

    The chain is written out here, independently of the generator, and walked over the
    timeline model. A disagreement means one of the two is wrong and the run cannot say
    which — which is the point of deriving it twice.
    """
    asked = {q.id: q for q in shipped.questions}
    truth = timeline.Truth(shipped)
    checked = 0
    for qid, (start, relations) in CHAINS.items():
        question = asked.get(qid)
        if question is None:
            continue                                # a chain v1 does not ask
        reached = _walk(truth, start, relations, question.at or shipped.evaluated_at)
        got = {nz.normalize(v) for v in reached}
        want = ({nz.normalize(v) for v in question.gold.values}
                if question.gold.kind == "set" else {nz.normalize(question.gold.value)})
        assert got == want, f"{qid}: walk reaches {sorted(got)}, gold says {sorted(want)}"
        checked += 1
    assert checked == sum(1 for q in shipped.questions if q.category == "multi_hop"), (
        "every multi_hop question must have its chain written out above; one that does "
        "not is a gold nothing checks")


#: Every negative question that carries no probe, and the slot it is really about. The
#: dataset withholds the slot from the systems on purpose — finding it is the difficulty —
#: which also means `test_every_gold_agrees_with_the_timeline_model` skips these, and
#: nothing checked that the store really holds nothing there. A gold of *nothing* that was
#: wrong would be invisible: every system would be marked right for abstaining and marked
#: wrong for the correct answer.
OPEN_NEGATIVES: dict[str, tuple[str, str]] = {
    "q-absent-open-1": ("Oscar", "lives_in"),
    "q-absent-open-2": ("reporting-service", "auth_strategy"),
    "q-absent-open-3": ("Project Chronos", "deploy_region"),
    "q2-none-globex-plan": ("Globex", "plan"),
    "q2-none-frank-title": ("frank", "job_title"),
    "q2-none-meridian-region": ("Project Meridian", "deploy_region"),
    "q2-none-orbit-lead": ("team-orbit", "team_lead"),
}


def test_every_open_negative_really_is_about_nothing(shipped):
    """The other half of the double entry, for the questions that name no slot."""
    truth = timeline.Truth(shipped)
    asked = {q.id: q for q in shipped.questions}
    unprobed = [q for q in shipped.questions
                if q.category == "negative" and q.probe is None]
    for question in unprobed:
        assert question.id in OPEN_NEGATIVES, (
            f"{question.id} is an open negative with no slot written out above, so "
            "nothing checks that its gold of *nothing* is true")
        slot = truth.slot(*OPEN_NEGATIVES[question.id])
        held = slot.values_at(question.at or shipped.evaluated_at, question.known_at) \
            if slot is not None else ()
        assert not held, f"{question.id} says nothing is held, and the store holds {held}"
    for qid in OPEN_NEGATIVES:
        if qid in asked:
            assert asked[qid].probe is None, f"{qid} now carries a probe"


def test_v2_is_v1_plus_and_changes_nothing_it_inherited(shipped):
    """A superseded version stays reproducible, and the newer one has to contain it.

    `README.md` says v1 stays where it is when v2 appears. That is a promise about the
    files; this is the promise about the content — every inherited row is byte-identical,
    so a per-question v1 result and a per-question v2 result compare directly even though
    the totals cannot.
    """
    if shipped.version == "v1":
        return
    inherited = ds.load("v1")
    for older, newer in ((inherited.events, shipped.events),
                         (inherited.questions, shipped.questions)):
        index = {row.id: row for row in newer}
        for row in older:
            assert row.id in index, f"v2 drops {row.id}, which v1 published"
            assert index[row.id].to_json() == row.to_json(), (
                f"v2 changes {row.id}, which v1 published — a v1 score for that question "
                "would no longer mean what it meant")


def test_no_question_carries_a_real_persons_data(shipped):
    """Everything published has to be synthetic. The check is crude on purpose: it
    catches the paste, not the invention."""
    blob = " ".join(e.text for e in shipped.events).lower()
    for leak in ("@gmail", "@anthropic", "api_key", "sk-", "password", "bearer "):
        assert leak not in blob, f"{leak!r} appears in the dataset text"


# --- validation -------------------------------------------------------------

def _minimal(**overrides):
    row = {"id": "e1", "recorded_at": "2026-01-01", "valid_from": "2026-01-01",
           "subject": "a", "predicate": "lives_in", "object": "Berlin",
           "text": "a lives in Berlin", "source": "s"}
    row.update(overrides)
    return ds.MemoryEvent.from_json(row)


def _dataset(events, questions, predicates=None):
    return ds.Dataset(
        version="test", evaluated_at=at("2026-08-01"),
        predicates=predicates or {"lives_in": ds.PredicateDecl("lives_in")},
        events=tuple(events), questions=tuple(questions))


def _question(**overrides):
    row = {"id": "q1", "category": "current_state", "question": "where?",
           "probe": ["a", "lives_in"], "gold": {"kind": "value", "value": "Berlin"}}
    row.update(overrides)
    return ds.Question.from_json(row)


@pytest.mark.parametrize("kwargs, message", [
    ({"predicate": "unknown_pred"}, "does not declare"),
    ({"valid_to": "2025-01-01"}, "holds at no"),
])
def test_a_bad_event_is_a_load_error_not_a_low_score(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ds.validate(_dataset([_minimal(**kwargs)], []))


def test_a_duplicate_event_id_is_refused():
    with pytest.raises(ValueError, match="duplicate event id"):
        ds.validate(_dataset([_minimal(), _minimal()], []))


def test_a_duplicate_question_id_is_refused():
    with pytest.raises(ValueError, match="duplicate question id"):
        ds.validate(_dataset([_minimal()], [_question(), _question()]))


def test_a_question_probing_a_slot_nothing_writes_is_refused():
    """It could only ever score wrong, and a benchmark that ships one is measuring itself."""
    with pytest.raises(ValueError, match="no event"):
        ds.validate(_dataset([_minimal()], [_question(probe=["nobody", "lives_in"])]))


def test_a_negative_question_may_probe_an_empty_slot():
    ds.validate(_dataset([_minimal()], [_question(
        id="q2", category="negative", probe=["nobody", "lives_in"], gold={"kind": "none"})]))


def test_a_set_gold_may_not_publish_aliases():
    with pytest.raises(ValueError, match="which member"):
        ds.validate(_dataset([_minimal()], [_question(
            gold={"kind": "set", "values": ["Berlin"], "aliases": ["BER"]})]))


def test_a_date_question_must_name_the_value_it_is_about():
    with pytest.raises(ValueError, match="Set `about`"):
        ds.validate(_dataset([_minimal()], [_question(
            category="change_time", gold={"kind": "date", "value": "2026-01-01"})]))


def test_a_date_question_about_a_value_held_twice_is_refused():
    """Reversion makes `about` ambiguous, and an ambiguous question has more than one
    defensible answer. This guard is here because that mistake shipped once."""
    events = [_minimal(id="e1", object="Berlin", valid_from="2025-01-01", recorded_at="2025-01-01"),
              _minimal(id="e2", object="London", valid_from="2025-06-01", recorded_at="2025-06-01"),
              _minimal(id="e3", object="Berlin", valid_from="2026-01-01", recorded_at="2026-01-01")]
    with pytest.raises(ValueError, match="separate intervals"):
        ds.validate(_dataset(events, [_question(
            category="change_time", about="Berlin",
            gold={"kind": "date", "value": "2026-01-01"})]))


def test_an_unknown_category_is_refused():
    with pytest.raises(ValueError, match="unknown category"):
        _question(category="vibes")


def test_an_unknown_answer_kind_is_refused():
    with pytest.raises(ValueError, match="unknown answer kind"):
        _question(gold={"kind": "vibes", "value": "x"})


def test_filtering_keeps_every_event(data):
    """A partial run asks fewer questions of the same memory. Dropping events would make
    a `--quick` number incomparable with a full one and would say so nowhere."""
    narrowed = data.filter(categories=["current_state"], limit=3)
    assert len(narrowed.events) == len(data.events)
    assert len(narrowed.questions) == 3
    assert {q.category for q in narrowed.questions} == {"current_state"}


def test_filtering_on_an_unknown_category_is_an_error(data):
    with pytest.raises(ValueError, match="unknown categor"):
        data.filter(categories=["nonsense"])


def test_a_limit_spreads_across_categories(data):
    """`--quick` is a smoke test only if it touches every dimension."""
    assert len({q.category for q in data.filter(limit=20).questions}) >= 8


# --- the temporal model -----------------------------------------------------

def _slot(rows, single=True):
    events = [_minimal(id=f"e{i}", object=o, valid_from=vf, recorded_at=ra)
              for i, (o, vf, ra) in enumerate(rows)]
    decl = ds.PredicateDecl("lives_in", cardinality="one" if single else "many")
    data = _dataset(events, [], {"lives_in": decl})
    return timeline.Truth(data).slot("a", "lives_in")


def test_rule_one_later_valid_time_wins():
    slot = _slot([("Berlin", "2026-01-10", "2026-01-10"),
                  ("London", "2026-03-15", "2026-03-15")])
    assert slot.values_at(at("2026-02-01")) == ("Berlin",)
    assert slot.values_at(at("2026-04-01")) == ("London",)


def test_rule_two_at_equal_valid_time_the_later_record_retires_the_earlier():
    slot = _slot([("London", "2026-04-01", "2026-04-02"),
                  ("Paris", "2026-04-01", "2026-04-05")])
    assert slot.values_at(at("2026-04-03")) == ("Paris",)
    assert slot.distinct_values == ("Paris",), "a retracted value is not one it ever held"


def test_rule_three_an_ending_is_only_visible_once_the_successor_was_recorded():
    """The sentence a one-clock store cannot say. Rewound to 5 March, the store had not
    heard about the move, so the old value is still in force with no end in sight."""
    slot = _slot([("us-east-1", "2025-09-01", "2025-09-01"),
                  ("eu-west-1", "2026-03-01", "2026-03-10")])
    assert slot.values_at(at("2026-03-05")) == ("eu-west-1",)
    assert slot.values_at(at("2026-03-05"), known_at=at("2026-03-05")) == ("us-east-1",)


def test_a_retirement_is_itself_dated_on_the_belief_clock():
    slot = _slot([("London", "2026-04-01", "2026-04-02"),
                  ("Paris", "2026-04-01", "2026-04-05")])
    assert slot.values_at(at("2026-04-03"), known_at=at("2026-04-03")) == ("London",)


def test_rule_four_repeating_a_value_is_not_a_change():
    slot = _slot([("MySQL", "2025-01-10", "2025-01-10"),
                  ("MySQL", "2025-04-02", "2025-04-02"),
                  ("MySQL", "2025-08-19", "2025-08-19"),
                  ("PostgreSQL", "2026-05-05", "2026-05-05")])
    assert slot.distinct_values == ("MySQL", "PostgreSQL")
    assert slot.values_at(at("2025-06-01")) == ("MySQL",)


def test_a_multi_valued_slot_accumulates_instead_of_resolving():
    slot = _slot([("English", "2024-01-01", "2024-01-01"),
                  ("German", "2025-03-01", "2025-03-01")], single=False)
    assert set(slot.values_at(at("2026-01-01"))) == {"English", "German"}
    assert slot.values_at(at("2024-06-01")) == ("English",)


def test_competitors_without_a_probe_are_the_whole_scenario(truth, data):
    """Broader, and therefore stricter: an answer naming two of them is ambiguous."""
    assert len(truth.competitors(None, "london_crowd")) > 1
    assert set(truth.competitors(("alice", "lives_in"), "alice_relocation")) == {
        "Berlin", "London", "New York"}


# --- answer matching --------------------------------------------------------

@pytest.mark.parametrize("answer", ["London", "london", " LONDON. ", "the London"])
def test_formatting_never_costs_a_point(answer):
    assert nz.matches_value(answer, "London")


def test_a_published_alias_counts():
    assert nz.matches_value("NYC", "New York", aliases=["NYC"])


def test_a_value_inside_a_short_sentence_counts():
    assert nz.matches_value("She lived in London.", "London", competitors=["Berlin"])


def test_an_answer_naming_two_candidates_is_ambiguous_and_wrong():
    """The rule that stops lenient matching from being a way to win by saying everything."""
    assert not nz.matches_value("Berlin, then London", "London", competitors=["Berlin"])


def test_an_answer_longer_than_the_ceiling_is_not_searched():
    """Finding a value inside a paragraph is not evidence the system knows it."""
    limit = nz.CONTAINMENT_TOKEN_LIMIT
    at_ceiling = " ".join(["word"] * (limit - 1) + ["London"])
    over_ceiling = " ".join(["word"] * limit + ["London"])
    assert nz.matches_value(at_ceiling, "London", competitors=["Berlin"])
    assert not nz.matches_value(over_ceiling, "London", competitors=["Berlin"])


def test_strict_matching_requires_equality():
    assert not nz.matches_value("She lived in London.", "London", lenient=False)
    assert nz.matches_value("London", "London", lenient=False)


def test_a_substring_is_not_a_match():
    assert not nz.matches_value("Yorkshire", "York")


def test_a_set_answer_must_be_exactly_the_gold_set():
    assert nz.matches_set(["b", "a"], ["a", "b"])
    assert not nz.matches_set(["a", "b", "c"], ["a", "b"]), "naming everything must not win"
    assert not nz.matches_set(["a"], ["a", "b"])


def test_dates_are_iso_only_and_prose_is_not_guessed_at():
    assert nz.matches_date("2026-03-15T09:00:00Z", "2026-03-15")
    assert not nz.matches_date("the fifteenth of March", "2026-03-15")
    assert nz.parse_date("last spring") is None


# --- scoring ----------------------------------------------------------------

def _judged(question, answer, data, truth):
    return scoring.judge(question, answer, truth, data.evaluated_at)


def test_an_abstention_is_correct_only_where_nothing_was_known(data, truth):
    negative = next(q for q in data.questions if q.category == "negative")
    assert _judged(negative, MemoryAnswer(), data, truth).correct
    positive = next(q for q in data.questions if q.id == "q-alice-current")
    assert not _judged(positive, MemoryAnswer(), data, truth).correct


def test_answering_todays_value_to_a_question_about_march_is_named_as_such(data, truth):
    question = next(q for q in data.questions if q.id == "q-alice-hist-mar")
    judgement = _judged(question, MemoryAnswer(value="New York"), data, truth)
    assert not judgement.correct
    assert judgement.reason == "answered_current_state"


def test_a_value_from_another_entity_is_named_as_such(data, truth):
    question = next(q for q in data.questions if q.id == "q-crowd-heidi-hist")
    judgement = _judged(question, MemoryAnswer(value="Lisbon"), data, truth)
    assert judgement.reason == "answered_other_entity"


def test_a_value_that_appears_nowhere_is_named_as_such(data, truth):
    question = next(q for q in data.questions if q.id == "q-alice-current")
    assert _judged(question, MemoryAnswer(value="Reykjavik"), data, truth).reason == "unknown_value"


def test_crossing_the_two_clocks_is_named_as_such(data, truth):
    """The atlas slot moved on 1 March and was recorded on the 10th. Answering the
    knowledge-time question with the world-time date is the specific confusion."""
    question = next(q for q in data.questions if q.id == "q-atlas-ktime")
    judgement = _judged(question, MemoryAnswer(value="2026-03-01"), data, truth)
    assert judgement.reason == "answered_before_it_knew"


def test_a_date_that_is_not_iso_is_named_as_a_format_failure(data, truth):
    question = next(q for q in data.questions if q.id == "q-atlas-ktime")
    assert _judged(question, MemoryAnswer(value="March"), data, truth).reason == "unparsable_date"


@pytest.mark.parametrize("given, reason", [
    (("Berlin", "London", "New York", "Paris"), "over_answered"),
    (("Berlin",), "under_answered"),
    (("Berlin", "London", "Paris"), "wrong_set"),
])
def test_set_failures_say_which_way_they_went(data, truth, given, reason):
    question = next(q for q in data.questions if q.id == "q-alice-changes")
    assert _judged(question, MemoryAnswer(values=given), data, truth).reason == reason


def test_a_set_where_one_value_was_asked_is_not_called_an_unknown_value(data, truth):
    """The reason is derived from `answer.value`, which such an answer leaves `None`."""
    question = next(q for q in data.questions if q.id == "q-alice-current")
    judgement = _judged(question, MemoryAnswer(values=("Berlin", "New York")), data, truth)
    assert not judgement.correct
    assert judgement.reason == "over_answered"


def test_every_named_reason_is_one_the_report_can_explain():
    assert set(scoring.REASONS) == set(report.REASON_HELP)


def test_an_empty_category_reports_a_dash_rather_than_zero(data):
    narrowed = data.filter(categories=["current_state"])
    card = scoring.score([], narrowed)
    assert card.by_category["historical_state"].accuracy is None
    assert report.pct(card.by_category["historical_state"]).strip() == "-"


def test_dimension_totals_add_up_to_the_overall_total(data):
    """A reader should be able to check the arithmetic, so the dimensions must partition
    the categories rather than overlap."""
    assigned = [c for members in data.dimensions.values() for c in members]
    assert sorted(assigned) == sorted(ds.CATEGORIES)


# --- the adapter interface --------------------------------------------------

@pytest.mark.parametrize("name", ["naive", "vector-rag", "memvara", "memvara-graph"])
def test_every_shipped_adapter_satisfies_the_protocol(name):
    system = registry.build(name)
    try:
        assert isinstance(system, base.MemorySystem)
        assert isinstance(system.name, str) and isinstance(system.version, str)
    finally:
        system.close()


#: `four` was wrong in five documents at once, propagated out of `adapters/base.py`'s
#: docstring, so there was no correct copy for anything to disagree with.
_NUMBER_WORDS = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def test_the_documented_adapter_method_count_is_the_enforced_one():
    """Every document stating how many methods an adapter needs must say the real number.

    `registry.build` is the authority: it refuses a system missing any of them. The count
    was documented as four in five places while five were required, and `usage` — the one
    omitted from the prose — is exactly the one a contributor would then be told about by
    a TypeError.
    """
    import re

    required = ("reset", "remember", "query", "usage", "close")
    correct = _NUMBER_WORDS[len(required)]
    wrong = {w for n, w in _NUMBER_WORDS.items() if n != len(required)}

    sources = [ROOT / "benchmarks" / "agent_memory" / "adapters" / "base.py",
               ROOT / "docs" / "ROADMAP.md",
               *(ROOT / "benchmarks").rglob("*.md"),
               *(ROOT / "docs" / "benchmarks").rglob("*.md")]
    pattern = re.compile(r"\b(\w+)[ -]methods?\b", re.IGNORECASE)
    bad = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            word = match.group(1).lower()
            if word in wrong:
                line = text[:match.start()].count("\n") + 1
                bad.append(f"{path.relative_to(ROOT)}:{line}: {match.group(0)!r}")
    assert not bad, (
        f"an adapter needs {len(required)} methods ({', '.join(required)}), so these "
        f"should say {correct!r}:\n" + "\n".join(bad))


def test_the_registry_checks_exactly_the_protocol_methods():
    """The list `registry.build` enforces and the protocol's own members must agree, or
    one of them is documentation that nothing keeps true.

    Derived from the class namespace rather than `__protocol_attrs__`, which `typing`
    only grows in 3.12 — this test's first version used it, passed on the 3.13 it was
    written on, and failed the 3.10 and 3.11 jobs. `requires-python` is `>=3.10`, so the
    floor is what a guard has to run on.
    """
    import inspect

    from benchmarks.agent_memory.adapters.base import MemorySystem

    declared = {name for name, value in vars(MemorySystem).items()
                if inspect.isfunction(value) and not name.startswith("_")}
    assert declared == {"reset", "remember", "query", "usage", "close"}


def test_an_adapter_can_be_named_by_import_path():
    system = registry.build("benchmarks.agent_memory.adapters.naive:build")
    try:
        assert system.name == "naive"
    finally:
        system.close()


def test_an_incomplete_adapter_fails_at_construction_not_mid_run(monkeypatch):
    class Half:
        name, version = "half", "0"

        def reset(self, predicates): ...

    monkeypatch.setattr(registry, "_import", lambda target: (lambda **kw: Half()))
    with pytest.raises(TypeError, match="remember"):
        registry.build("half")


def test_an_adapter_without_a_version_is_refused(monkeypatch):
    class Anonymous:
        name = "anon"

        def reset(self, predicates): ...
        def remember(self, event): ...
        def query(self, ask): ...
        def usage(self): ...
        def close(self): ...

    monkeypatch.setattr(registry, "_import", lambda target: (lambda **kw: Anonymous()))
    with pytest.raises(TypeError, match="`version`"):
        registry.build("anon")


def test_an_unresolvable_system_name_says_what_the_options_are():
    with pytest.raises(ValueError, match="neither a known system"):
        registry.build("nonsense")


def test_the_adapter_is_never_handed_the_gold_answer(data):
    """Enforced rather than asked for: `Ask` has no field that could carry it."""
    question = data.questions[0]
    ask = runner.ask_of(question, data.evaluated_at)
    assert not hasattr(ask, "gold")
    assert set(vars(type(ask))["__slots__"]) == {
        "id", "category", "question", "probe", "at", "evaluated_at", "known_at", "about"}


def test_now_is_resolved_from_the_dataset_not_the_wall_clock(data):
    """An adapter must never see `at=None`, or its answers would depend on the day."""
    for question in data.questions:
        assert runner.ask_of(question, data.evaluated_at).at is not None


def test_every_adapter_indexes_the_same_text(data):
    """Retrieval is a scored dimension, so the three adapters must not feed their
    retrievers three different strings. They did: the memvara adapter passed the bare
    sentence as `Claim.text`, which is what memvara indexes, and "I have relocated to
    Madrid" does not contain the word Heidi."""
    event = next(e for e in data.events if e.subject == "heidi")
    text = base.indexable(event)
    assert "heidi" in text and event.object in text and event.text in text


def test_the_slot_rule_prefers_the_relation_the_question_names():
    """Rank alone answers the first hop of a chained question and stops.

    All three systems rank `alice works_on Project Atlas` first for this question, and
    correctly — it is the closest sentence to the wording. The claim holding the answer
    is in the same list, and nothing was missing except a reason to prefer it.
    """
    candidates = [("alice", "works_on"), ("Project Atlas", "deploy_region"),
                  ("alice", "speaks")]
    chosen = base.pick_slot("Which region is the project Alice works on deployed to?",
                            candidates)
    assert chosen == ("Project Atlas", "deploy_region")


def test_the_slot_rule_falls_back_to_rank_when_no_relation_is_named():
    """A question that names no predicate must not be re-ordered by this rule."""
    candidates = [("frank", "lives_in"), ("alice", "lives_in")]
    assert base.pick_slot("Tell me about Frank.", candidates) == ("frank", "lives_in")
    assert base.pick_slot("anything", []) is None


def test_the_slot_rule_does_not_count_a_repeated_candidate_twice():
    """memvara returns up to `max_per_slot` claims from one slot and `naive` has an entry
    per event, so a ranked list names slots more than once. A repeat is not a second
    reason to prefer one."""
    candidates = [("bob", "works_at"), ("bob", "works_at"), ("Globex", "hq_city")]
    assert base.pick_slot("In which city is Bob's employer headquartered?",
                          candidates) == ("Globex", "hq_city")


def test_a_preposition_in_a_predicate_name_earns_no_point():
    """`works_on` would otherwise score for the word *on*, which half of English
    questions contain, and beat `deploy_region` on a tie."""
    assert base.pick_slot("Which region is it deployed to?",
                          [("x", "works_on"), ("y", "deploy_region")]) == ("y", "deploy_region")


@pytest.mark.parametrize("name", ["naive", "vector-rag", "memvara"])
def test_every_adapter_resolves_a_chain_with_the_same_rule(data, name):
    """The rule is shared, so a chained question must reach the second hop in all three.

    Retrieval is a scored dimension. A slot-selection rule that lived in one adapter
    would be measuring the harness, so the check is that no adapter has its own.
    """
    system = registry.build(name)
    try:
        runner.ingest(system, data)
        question = next(q for q in data.questions if q.id == "q-chain-atlas")
        ask = runner.ask_of(question, data.evaluated_at)
        # memvara's resolver also takes the ask, because which claim states it searches
        # depends on whether the question rewinds a clock.
        chosen = (system._resolve(ask.question, ask) if name == "memvara"
                  else system._resolve(ask.question))
        assert chosen == ("Project Atlas", "deploy_region"), chosen
    finally:
        system.close()


def test_events_are_delivered_in_recorded_order(data):
    ordered = base.sort_events(data.events)
    stamps = [e.recorded_at for e in ordered]
    assert stamps == sorted(stamps)
    assert base.sort_events(list(reversed(data.events))) == ordered, "order must be total"


# --- results ----------------------------------------------------------------

@pytest.fixture(scope="module")
def naive_run(data):
    system = registry.build("naive")
    try:
        return runner.run(system, data, timestamp="2026-08-01T00:00:00+00:00")
    finally:
        system.close()


def test_a_result_carries_everything_needed_to_reproduce_it(naive_run):
    payload = naive_run.to_json()
    assert payload["benchmark"] == "agent-memory"
    assert payload["benchmark_version"] == BENCHMARK_VERSION
    assert payload["dataset_version"] == DEFAULT_DATASET
    assert payload["system_version"]
    for field in ("python", "implementation", "platform", "machine", "cpu_count"):
        assert payload["environment"][field]
    assert set(payload["metrics"]) >= {"overall", "by_category", "by_dimension"}


#: What a published result file must never contain, as word-boundary patterns.
#:
#: Substrings once, which is how `token` came to match the legitimate `tokens` cost
#: field; deleting `token` then dropped `refresh_token`, `api_token` and a bare `"token"`
#: along with the false positive, and nothing noticed until a review compared the two
#: lists. Both mistakes are the same one — a guard tuned by what it happens to reject
#: rather than by what it is for.
CREDENTIAL_PATTERNS = (
    r"\btoken\b", r"\brefresh_token\b", r"\bapi_token\b", r"\baccess_token\b",
    r"\bauth_token\b", r"\bapi_key\b", r"\bapikey\b", r"\bpassword\b",
    r"\bpasswd\b", r"\bsecret\b", r"bearer ", r"\bauthorization\b",
    r"/users/", r"/home/",
)


@pytest.mark.parametrize("leak", [
    '"token": "sk-live-abc"', '"refresh_token": "x"', '"api_token": "y"',
    '"access_token": "z"', '"api_key": "k"', '"password": "p"',
    "authorization: bearer abc", "/users/somebody/x", "/home/somebody/x",
])
def test_the_credential_patterns_catch_what_they_are_for(leak):
    import re

    assert any(re.search(p, leak) for p in CREDENTIAL_PATTERNS), leak


@pytest.mark.parametrize("innocent", [
    '"tokens": 0', '"llm_calls": 0, "tokens": 520', '"rows_stored": 241',
    '"texts_embedded": 520', '"db_reads": 100',
])
def test_the_credential_patterns_do_not_fire_on_cost_fields(innocent):
    """The false positive that started this: narrowing the list to silence it cost real
    coverage, and word boundaries give both."""
    import re

    assert not any(re.search(p, innocent) for p in CREDENTIAL_PATTERNS), innocent


def test_a_result_carries_no_secrets(naive_run, tmp_path):
    """Result files are written to be published. Assembled by name, never swept."""
    path = tmp_path / "r.json"
    naive_run.write(path)
    import re

    blob = path.read_text().lower()
    # Word boundaries rather than substrings. `token` was in this list as a substring and
    # matched the legitimate `tokens` cost field the moment one existed — but deleting it
    # also dropped `refresh_token`, `api_token` and a bare `"token"`, none of which any
    # other entry covers. `\btoken\b` matches `"token": "sk-live-…"` and does not match
    # `"tokens": 0`, so the false positive goes and the coverage stays. A guard that fails
    # on correct code teaches people to weaken it; one weakened to stop it doing so is the
    # same mistake finished.
    for leak in CREDENTIAL_PATTERNS:
        assert not re.search(leak, blob), f"{leak!r} reached a publishable result file"


def test_every_adapter_counts_rows_the_same_way(data):
    """`rows_stored` means rows held, for all three, or the column compares nothing.

    It used to mean rows for memvara and *write calls* for the baselines, and the
    published table headed the mixture "rows written" — reporting that the dictionary
    stored more rows than the bitemporal store, which is the reverse of the truth.
    """
    held = {}
    for name in ("memvara", "naive", "vector-rag"):
        system = registry.build(name)
        try:
            runner.ingest(system, data)
            held[name] = system.usage().rows_stored
        finally:
            system.close()
    assert held["naive"] < held["memvara"] < held["vector-rag"], held
    assert held["naive"] < len(data.events), (
        "naive overwrites single-valued slots, so it must hold fewer rows than it was "
        f"handed: {held['naive']} against {len(data.events)} events")
    assert held["vector-rag"] == len(data.events), "this store appends every observation"


def test_two_values_starting_on_one_day_are_not_a_correction(data):
    """A multi-valued relation gains values; it does not contradict itself.

    Every filler person is given two languages with the same `valid_from`, so without a
    cardinality guard the second write claims the first record was wrong — 30 times.
    """
    from benchmarks.agent_memory.adapters.memvara_adapter import MemvaraMemory

    system = registry.build("memvara")
    try:
        assert isinstance(system, MemvaraMemory)
        system.reset(data.predicates)
        misread = 0
        for event in base.sort_events(data.events):
            if not data.predicates[event.predicate].single_valued and system._is_correction(event):
                misread += 1
            system.remember(event)
        assert misread == 0, f"{misread} multi-valued writes were called corrections"
        spoken = [c.object for c in system.mem.history("alice", "speaks") if c.state != "retired"]
        assert set(spoken) == {"English", "German", "Portuguese"}
    finally:
        system.close()


def test_the_leaderboard_prints_dimension_names_in_full(naive_run):
    """The header was sliced to 13 characters, so the command the README puts first
    printed `knowledge_tim` while every table in the docs said `knowledge_time`."""
    text = report.leaderboard([naive_run])
    for dimension in naive_run.scorecard.by_dimension:
        assert dimension in text, f"{dimension!r} was truncated out of the header"


def test_the_documented_result_schema_names_exactly_the_usage_fields():
    """The published schema and the dataclass must agree, in both directions.

    This is the guard the last two reviews wanted and neither wrote. `db_writes` was
    renamed to `rows_stored` and `embedding_calls` to `texts_embedded`, each because the
    field meant something other than its name; both times the schema table had to be
    updated by hand, and nothing would have failed if it had not been. A field added
    without a doc entry and a doc entry left behind after a rename are the same defect
    seen from two sides, so this checks both.
    """
    import dataclasses
    import re

    report_page = ROOT / "docs" / "benchmarks" / "agent-memory-benchmark.md"
    row = next((line for line in report_page.read_text(encoding="utf-8").splitlines()
                if line.startswith("| `usage` |")), None)
    assert row is not None, "the result-schema table no longer has a `usage` row"

    #: Backticked words in that row that are prose rather than field names.
    not_fields = {"usage", "null"}
    # `[a-z0-9_]+`, not `[a-z_]+`: a name with a digit in it — `p95_ms`, `tokens_v2` —
    # would otherwise be captured as the fragments either side of the digit, and this
    # guard would fail while the documentation was right. The sibling `latency` row
    # already contains `query_p50_ms` and `query_p95_ms`, so the letter-only pattern
    # reads that row as four names rather than six.
    documented = {word for word in re.findall(r"`([a-z0-9_]+)`", row)} - not_fields
    declared = {f.name for f in dataclasses.fields(Usage)}

    assert documented == declared, (
        f"the schema table and `Usage` disagree.\n"
        f"  documented but not a field: {sorted(documented - declared)}\n"
        f"  a field but undocumented:   {sorted(declared - documented)}")


def test_the_documented_result_schema_names_exactly_the_latency_fields():
    """The same guard, one row down.

    Written after `repeats` and `p50_spread_ms` were added to `Latency` and the schema
    table was updated by hand — which is how the two `Usage` renames went wrong twice.
    A guard that covers one row of a table and not the row beside it is the instance
    fixed and the class left alone.
    """
    import dataclasses
    import re

    report_page = ROOT / "docs" / "benchmarks" / "agent-memory-benchmark.md"
    row = next((line for line in report_page.read_text(encoding="utf-8").splitlines()
                if line.startswith("| `latency` |")), None)
    assert row is not None, "the result-schema table no longer has a `latency` row"

    #: Backticked words in that row that are prose rather than field names.
    not_fields = {"latency"}
    documented = {w for w in re.findall(r"`([a-z0-9_]+)`", row)} - not_fields
    declared = {f.name for f in dataclasses.fields(results.Latency)}
    assert documented == declared, (
        f"the schema table and `Latency` disagree.\n"
        f"  documented but not a field: {sorted(documented - declared)}\n"
        f"  a field but undocumented:   {sorted(declared - documented)}")


def test_the_documented_environment_block_is_the_one_that_is_written():
    """The environment block is a fixed list assembled by name, and the schema table
    names it. `cpu_count` was added to one and had to be added to the other by hand."""
    import re

    report_page = ROOT / "docs" / "benchmarks" / "agent-memory-benchmark.md"
    row = next((line for line in report_page.read_text(encoding="utf-8").splitlines()
                if line.startswith("| `environment` |")), None)
    assert row is not None, "the result-schema table no longer has an `environment` row"
    documented = {w for w in re.findall(r"`([a-z0-9_]+)`", row)} - {"environment"}
    assert documented == set(results.environment()), (
        f"documented but not written: {sorted(documented - set(results.environment()))}; "
        f"written but undocumented: {sorted(set(results.environment()) - documented)}")


def test_every_usage_field_is_rendered_by_the_report():
    """A field nothing prints is a field nobody reads, however well documented.

    `tokens` was added to `Usage` and, with the cost block hardcoding one line per field,
    was simply absent from the report until this noticed.
    """
    import dataclasses

    declared = {f.name for f in dataclasses.fields(Usage)} - {"extra"}
    assert set(report.COST_LABELS) == declared, (
        "report.COST_LABELS and `Usage` disagree; a field with no label is never printed")

    populated = Usage(**{f.name: (7 if f.name != "extra" else {"thing": 9})
                         for f in dataclasses.fields(Usage)})
    text = "\n".join(report._cost_block(_stub_result(populated)))
    for label in report.COST_LABELS.values():
        assert label in text
    assert "thing" in text, "`extra` entries are printed too"


def test_an_unmeasured_cost_stays_none_rather_than_becoming_zero():
    """`0` is a claim. `null` is the absence of one, and the report prints `-`."""
    assert Usage().to_json()["llm_calls"] is None
    assert Usage().to_json()["rows_stored"] is None
    assert "-" in "\n".join(report._cost_block(_stub_result()))


def _stub_result(usage: Usage | None = None):
    return results.RunResult(
        system="s", system_version="0", dataset_version="v1", timestamp="t",
        scorecard=scoring.score([], _dataset([], [])),
        latency=results.latency_of(0.0, 0, []), usage=usage or Usage(), judgements=())


def test_percentiles_are_nearest_rank_and_never_invent_a_measurement():
    latency = results.latency_of(1.0, 10, [0.001, 0.002, 0.003, 0.004])
    assert latency.query_max_ms == pytest.approx(4.0)
    assert latency.query_p50_ms in (2.0, 3.0)


def test_every_latency_field_is_rendered_by_the_report():
    """A field added to `Latency` and forgotten in the report is silently absent.

    The same guard as `test_every_usage_field_is_rendered_by_the_report`, and for the
    same reason: `tokens` was added to `Usage` and went unprinted for exactly this
    reason, because the block was a list of hardcoded f-strings.
    """
    named = [field for _, fields, _ in report.LATENCY_ROWS for field in fields]
    assert sorted(named) == sorted(results.Latency.__dataclass_fields__), (
        "LATENCY_ROWS and the Latency dataclass disagree; a field in one and not the "
        "other is a number that is measured and never shown, or shown and never measured")
    assert len(named) == len(set(named)), f"a field prints twice: {named}"


def test_repeating_the_timings_does_not_change_the_cost_counters(data):
    """`usage()` is read between the scored pass and the repeats, and the order matters.

    Read after them, `db_reads` would multiply by `--latency-repeats` and a system would
    look more expensive because the benchmark got more careful about its clock.
    """
    small = data.filter(limit=12)
    counts = {}
    for repeats in (1, 3):
        system = registry.build("naive")
        try:
            result = runner.run(system, small, timestamp="2026-08-01T00:00:00+00:00",
                                latency_repeats=repeats)
        finally:
            system.close()
        counts[repeats] = result.usage.db_reads
        assert result.latency.repeats == max(1, repeats - 1)
    assert counts[1] == counts[3] == len(small.questions), counts


def test_a_single_pass_reports_a_spread_of_zero_and_says_so(data):
    """Zero spread across one pass means *not measured*, and the report must not let it
    read as *perfectly stable*."""
    system = registry.build("naive")
    try:
        result = runner.run(system, data.filter(limit=8),
                            timestamp="2026-08-01T00:00:00+00:00")
    finally:
        system.close()
    assert result.latency.repeats == 1
    assert result.latency.p50_spread_ms == 0.0
    assert "one pass" in report.scorecard(result, data)


def test_a_repeat_count_below_one_is_refused(capsys):
    assert cli.main(["--system", "naive", "--latency-repeats", "0"]) == 2
    assert "at least 1" in capsys.readouterr().err


def test_the_reproducibility_check_ignores_timings_and_internal_ids(naive_run):
    """Otherwise a busy laptop reports nondeterminism, and everyone learns to ignore it."""
    payload = naive_run.to_json()
    trimmed = results.comparable(payload)
    assert "latency" not in trimmed
    assert all("support" not in row for row in trimmed["questions"])


# --- the memvara adapter, end to end ----------------------------------------

@pytest.fixture(scope="module")
def memvara_run(data):
    system = registry.build("memvara")
    try:
        return runner.run(system, data, timestamp="2026-08-01T00:00:00+00:00")
    finally:
        system.close()


def _verdict(run, qid):
    return next(j for j in run.judgements if j.question_id == qid)


@pytest.mark.parametrize("qid", [
    "q-atlas-hist-gap",   # today's belief about an instant the store had not heard about
    "q-atlas-asof",       # both clocks rewound: what it would have said that day
    "q-auth-asof",
    "q-dana-asof",        # a correction that had not arrived yet
    "q-dana-hist",        # the same instant, with today's understanding
    "q-quotes-asof",
])
def test_the_memvara_adapter_answers_the_two_clock_questions(memvara_run, qid):
    """These six are why the dataset exists. A regression here is the regression."""
    judgement = _verdict(memvara_run, qid)
    assert judgement.correct, f"{qid}: expected {judgement.expected}, got {judgement.given}"


def test_the_memvara_adapter_writes_a_correction_as_a_retirement(memvara_run):
    """`ended` says the world changed; `retired` says the record was wrong. Reporting one
    as the other records a false reason for a change that nothing downstream can detect."""
    assert _verdict(memvara_run, "q-dana-changes").correct
    assert _verdict(memvara_run, "q-quotes-changes").correct


def test_the_memvara_write_path_makes_no_model_calls(memvara_run):
    """True by construction here — the adapter uses `remember()`, not `add()` — and
    recorded so that a change which introduced one would be visible."""
    assert memvara_run.usage.llm_calls == 0


def test_memvara_counts_what_it_embeds(memvara_run, data):
    """It embeds the claim and the turn the claim came from, and reported neither.

    The column read `-` — *not measured*, which was honest and left the system doing the
    most embedding as the one with no figure.
    """
    counted = memvara_run.usage.texts_embedded
    assert counted is not None, "texts_embedded is measured now, not `-`"
    expected = (memvara_run.usage.rows_stored + memvara_run.usage.extra["episodes"]
                + sum(1 for q in data.questions if q.probe is None))
    assert counted == expected, (
        "claims + source episodes on the way in, then one per unprobed question")


def test_the_counting_embedder_keeps_the_embedders_identity():
    """`memvara.embed.fingerprint` derives a store's recorded identity from `name` and
    `dim`. A wrapper that shadowed either would make a file-backed store refuse to reopen
    with the very embedder that wrote it."""
    from memvara.embed import Embedder, HashingEmbedder, fingerprint_of

    from benchmarks.agent_memory.adapters.memvara_adapter import _CountingEmbedder

    inner = HashingEmbedder(dim=512)
    wrapped = _CountingEmbedder(inner)
    assert isinstance(wrapped, Embedder)
    assert fingerprint_of(wrapped) == fingerprint_of(inner)
    wrapped.encode(["a", "b", "c"])
    assert wrapped.texts_embedded == 3, "texts, not calls: a batch and a loop are equal work"


def test_the_graph_entry_is_the_same_adapter_with_one_thing_changed():
    """`memvara-graph` must differ from `memvara` by configuration and nothing else.

    A second entry that was quietly a second implementation would publish the graph
    leg's contribution as a difference it did not cause.
    """
    from benchmarks.agent_memory.adapters import memvara_adapter as ma

    plain, graph = registry.build("memvara"), registry.build("memvara-graph")
    try:
        assert type(plain) is type(graph)
        assert plain.name == "memvara" and graph.name == "memvara-graph"
        assert plain._tuning == {}
        assert graph._tuning == ma.GRAPH_TUNING
        assert set(ma.GRAPH_TUNING) == {"read_w_graph", "read_intent_weighting"}
    finally:
        plain.close()
        graph.close()


def test_the_dataset_holds_enough_graph_for_a_walk_to_pay(shipped):
    """The reason v2 exists, measured with memvara's own instrument.

    `Memvara.connectivity()` reports the share of live claims whose object is another
    claim's subject. On v1 it is 3 of 193 — 1.6% — and memvara's `docs/BENCHMARKS.md`
    says a graph walk cannot pay for itself at that rate. Six chained questions over
    three edges measure the wording of the six questions and nothing else.
    """
    system = registry.build("memvara")
    try:
        runner.ingest(system, shipped)
        counts = system.mem.connectivity()
    finally:
        system.close()
    rate = counts["joinable_claims"] / counts["live_claims"]
    if shipped.version == "v1":
        assert rate < 0.05, counts       # the finding v2 answers; pinned so it stays true
    else:
        assert rate > 0.15, (
            f"v2 exists to give chained questions something to walk and this is "
            f"{rate:.1%} joinable: {counts}")


def test_the_memvara_adapter_uses_the_provenance_api(memvara_run):
    """`why()` doing provenance work, rather than a shortcut through the claim's meta."""
    assert _verdict(memvara_run, "q-billing-prov").correct, "the earliest of four sources"


def test_two_identical_runs_give_identical_answers(data):
    """Accuracy has to be deterministic or no published number can be reproduced.
    Timings and claim ids are excluded; see results.comparable."""
    narrowed = data.filter(limit=30)
    payloads = []
    for _ in range(2):
        system = registry.build("memvara")
        try:
            payloads.append(results.comparable(
                runner.run(system, narrowed, timestamp="t").to_json()))
        finally:
            system.close()
    assert payloads[0] == payloads[1]


def test_an_adapter_that_raises_is_reported_as_an_adapter_defect(data):
    class Exploding:
        name, version = "boom", "0"

        def reset(self, predicates): ...
        def remember(self, event): ...
        def query(self, ask): raise RuntimeError("nope")
        def usage(self): return Usage()
        def close(self): ...

    with pytest.raises(RuntimeError, match="should return an empty MemoryAnswer"):
        runner.run(Exploding(), data.filter(limit=1))


def test_an_adapter_returning_the_wrong_type_is_caught(data):
    class Wrong:
        name, version = "wrong", "0"

        def reset(self, predicates): ...
        def remember(self, event): ...
        def query(self, ask): return "London"
        def usage(self): return Usage()
        def close(self): ...

    with pytest.raises(TypeError, match="must return"):
        runner.run(Wrong(), data.filter(limit=1))


# --- the report and the CLI -------------------------------------------------

def test_the_failure_report_shows_the_timeline_beside_the_wrong_answer(data, truth):
    system = registry.build("naive")
    try:
        result = runner.run(system, data, timestamp="t")
    finally:
        system.close()
    wrong = runner.failures(result, data, limit=5)
    text = report.failure_report(wrong, data, truth)
    assert "FAIL" in text and "Expected:" in text and "Reason:" in text
    assert "->" in text or "asked about" in text, "no timeline was rendered"


def test_the_scorecard_prints_only_measured_numbers(naive_run, data):
    text = report.scorecard(naive_run, data)
    assert "OVERALL" in text
    assert "'-' means not measured, not zero." in text


def test_the_leaderboard_orders_by_overall_accuracy(naive_run, memvara_run):
    text = report.leaderboard([naive_run, memvara_run])
    assert text.index("memvara") < text.index("naive")


def test_the_cli_runs_a_smoke_benchmark(capsys, tmp_path):
    code = cli.main(["--system", "naive", "--quick", "--output", str(tmp_path / "r.json")])
    assert code == 0
    assert (tmp_path / "r.json").is_file()
    payload = json.loads((tmp_path / "r.json").read_text())
    assert payload["config"]["limit"] == cli.QUICK
    assert "OVERALL" in capsys.readouterr().out


def test_the_cli_writes_one_file_per_system(capsys, tmp_path):
    code = cli.main(["--system", "naive", "--system", "vector-rag", "--quick", "--compare",
                     "--quiet", "--output", str(tmp_path / "r.json")])
    assert code == 0
    assert (tmp_path / "r-naive.json").is_file()
    assert (tmp_path / "r-vector-rag.json").is_file()
    assert "System" in capsys.readouterr().out


@pytest.mark.parametrize("module", ["benchmarks.agent_memory",
                                   "benchmarks.agent_memory.run"])
def test_both_spellings_of_the_command_run(module, tmp_path):
    """`...agent_memory` is canonical and `...agent_memory.run` is the spelling people
    reach for first. Both are asserted because a documented command that has never been
    executed is a documented command that does not work."""
    out = tmp_path / "r.json"
    proc = subprocess.run(
        [sys.executable, "-m", module, "--system", "naive", "--quick", "--quiet",
         "--output", str(out)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        env=_child_env(), timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text())["system"] == "naive"


def test_importing_the_alias_does_not_run_the_benchmark():
    """`run` has an importable dotted name, unlike `__main__`. Without the `__name__`
    guard, `from benchmarks.agent_memory import run` would run every question and then
    kill the interpreter with SystemExit."""
    from benchmarks.agent_memory import run as run_module

    assert run_module.main is cli.main


@pytest.mark.parametrize("system, expected", [
    ("naive", "r-naive.json"),
    ("mypackage.adapters:build", "r-mypackage.adapters-build.json"),
    ("weird//name", "r-weird-name.json"),
])
def test_a_per_system_filename_is_safe_on_every_platform(system, expected):
    """`--system` takes a dotted import path, and CONTRIBUTING.md documents the colon
    form. Interpolated raw it produced `r-mypackage.adapters:build.json`, which raises
    OSError on Windows — after every benchmark in the run had already finished."""
    assert cli._output_path("r.json", system, many=True).name == expected


def test_a_single_system_run_uses_the_output_path_verbatim():
    assert cli._output_path("out/r.json", "naive", many=False) == Path("out/r.json")


def test_the_cli_reports_an_empty_selection_rather_than_dividing_by_zero(capsys):
    assert cli.main(["--system", "naive", "--limit", "0"]) == 2
    assert "No questions selected" in capsys.readouterr().err


def test_the_cli_repeat_check_passes_for_a_deterministic_system(capsys):
    assert cli.main(["--system", "naive", "--quick", "--quiet", "--repeat-check"]) == 0
    assert "identical answers twice" in capsys.readouterr().out


def test_the_cli_catches_a_system_that_is_not_deterministic(capsys, monkeypatch):
    import itertools

    counter = itertools.count()

    class Drifting:
        name, version = "drift", "0"

        def reset(self, predicates): ...
        def remember(self, event): ...
        def query(self, ask): return MemoryAnswer(value=str(next(counter)))
        def usage(self): return Usage()
        def close(self): ...

    monkeypatch.setitem(registry.BUILTIN, "drift", "x:y")
    monkeypatch.setattr(registry, "_import", lambda target: (lambda **kw: Drifting()))
    assert cli.main(["--system", "drift", "--quick", "--quiet", "--repeat-check"]) == 1
    assert "NONDETERMINISM" in capsys.readouterr().err


def test_strict_matching_is_selectable_from_the_command_line(tmp_path):
    assert cli.main(["--system", "naive", "--quick", "--quiet", "--match", "strict",
                     "--output", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["config"]["match"] == "strict"


# --- the docs in this package execute ---------------------------------------

@pytest.mark.parametrize("module", [nz, ds])
def test_the_examples_in_the_benchmark_docstrings_still_run(module):
    """`pyproject.toml` points `--doctest-modules` at `tests` and `memvara`, so nothing
    would otherwise execute these. Documentation that is not run drifts silently."""
    failures, _ = doctest.testmod(module, verbose=False)
    assert failures == 0
