"""bench/hosted.py — scoring a probe suite against a store.

`bench/` is outside the coverage gate, so this file is the only guard on the
harness. Scoring functions are pure and tested on plain data; the runner is
tested against an in-memory store with planted claims. Nothing here touches
the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import hosted  # noqa: E402


def test_a_hit_probe_scores_by_gold_membership_and_rank():
    probe = {"id": "p1", "class": "hit", "query": "q", "gold": ["cl_b"]}
    results = [("cl_a", 0.9), ("cl_b", 0.7), ("cl_c", 0.2)]
    row = hosted.score_probe(probe, results, injected_ids=["cl_a", "cl_b"])
    assert row["hit"] is True
    assert row["gold_rank"] == 2
    assert row["false_injection"] is None
    assert row["top_score"] == 0.9


def test_a_hit_probe_misses_when_no_gold_id_returns():
    probe = {"id": "p1", "class": "hit", "query": "q", "gold": ["cl_z"]}
    row = hosted.score_probe(probe, [("cl_a", 0.9)], injected_ids=["cl_a"])
    assert row["hit"] is False
    assert row["gold_rank"] is None


def test_an_abstain_probe_fails_when_anything_is_injected():
    probe = {"id": "p2", "class": "abstain", "query": "haiku", "gold": []}
    row = hosted.score_probe(probe, [("cl_a", 0.31)], injected_ids=["cl_a"])
    assert row["false_injection"] is True
    assert row["top_score"] == 0.31
    assert row["hit"] is None


def test_an_abstain_probe_passes_only_on_an_empty_injection():
    probe = {"id": "p2", "class": "abstain", "query": "haiku", "gold": []}
    row = hosted.score_probe(probe, [("cl_a", 0.31)], injected_ids=[])
    assert row["false_injection"] is False


def test_a_verbatim_probe_requires_rank_one_exactly():
    probe = {"id": "p3", "class": "verbatim", "query": "text", "gold": ["cl_b"]}
    second = hosted.score_probe(probe, [("cl_a", 0.9), ("cl_b", 0.8)],
                                injected_ids=["cl_a"])
    first = hosted.score_probe(probe, [("cl_b", 0.9), ("cl_a", 0.8)],
                               injected_ids=["cl_b"])
    assert second["hit"] is False and second["gold_rank"] == 2
    assert first["hit"] is True and first["gold_rank"] == 1


def test_aggregate_reports_each_class_separately():
    rows = [
        {"probe_id": "a", "cls": "hit", "hit": True, "gold_rank": 1,
         "false_injection": None, "top_score": 0.8},
        {"probe_id": "b", "cls": "hit", "hit": False, "gold_rank": None,
         "false_injection": None, "top_score": 0.5},
        {"probe_id": "c", "cls": "ambiguous", "hit": True, "gold_rank": 3,
         "false_injection": None, "top_score": 0.6},
        {"probe_id": "d", "cls": "abstain", "hit": None, "gold_rank": None,
         "false_injection": True, "top_score": 0.45},
        {"probe_id": "e", "cls": "abstain", "hit": None, "gold_rank": None,
         "false_injection": False, "top_score": None},
    ]
    agg = hosted.aggregate(rows)
    assert agg["hit"]["n"] == 2 and agg["hit"]["hit_at_k"] == 0.5
    assert agg["hit"]["mean_gold_rank"] == 1.0
    # ambiguous is its own row, never folded into hit — the spec's judgment/fact split
    assert agg["ambiguous"]["n"] == 1 and agg["ambiguous"]["hit_at_k"] == 1.0
    assert agg["abstain"]["n"] == 2
    assert agg["abstain"]["false_injection_rate"] == 0.5
    assert agg["abstain"]["headroom"] == [0.45]


def _write_probes(tmp_path, lines):
    p = tmp_path / "probes.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    return p


def test_load_probes_accepts_a_valid_file(tmp_path):
    path = _write_probes(tmp_path, [
        {"id": "p1", "class": "hit", "query": "q", "gold": ["cl_a"]},
        {"id": "p2", "class": "abstain", "query": "haiku", "gold": []},
    ])
    probes = hosted.load_probes(path)
    assert [p["id"] for p in probes] == ["p1", "p2"]


@pytest.mark.parametrize("row, complaint", [
    ({"id": "p1", "class": "sonnet", "query": "q", "gold": []}, "class"),
    ({"id": "p1", "class": "hit", "query": "q", "gold": "cl_a"}, "gold"),
    ({"id": "p1", "class": "abstain", "query": "q", "gold": ["cl_a"]}, "abstain"),
    ({"class": "hit", "query": "q", "gold": ["cl_a"]}, "'id'"),
    ({"id": "p1", "class": "hit", "gold": ["cl_a"]}, "query"),
])
def test_load_probes_refuses_malformed_rows_naming_the_line(tmp_path, row, complaint):
    path = _write_probes(tmp_path, [row])
    with pytest.raises(SystemExit) as exc:
        hosted.load_probes(path)
    assert complaint in str(exc.value).lower()
    assert "line 1" in str(exc.value)


def test_load_probes_refuses_a_duplicate_id(tmp_path):
    path = _write_probes(tmp_path, [
        {"id": "p1", "class": "hit", "query": "a", "gold": ["cl_a"]},
        {"id": "p1", "class": "hit", "query": "b", "gold": ["cl_b"]},
    ])
    with pytest.raises(SystemExit, match="duplicate") as exc:
        hosted.load_probes(path)
    assert "line 2" in str(exc.value)


def test_load_probes_refuses_an_unedited_draft_row(tmp_path):
    # A drafted probe quotes the claim's own text, which measures lexical echo.
    # The runner refusing the mark is what makes --draft safe to ship (spec).
    path = _write_probes(tmp_path, [
        {"id": "p1", "class": "hit", "query": "q", "gold": ["cl_a"], "draft": True},
    ])
    with pytest.raises(SystemExit, match="draft") as exc:
        hosted.load_probes(path)
    assert "line 1" in str(exc.value)


from memvara import HashingEmbedder, Memvara, NullLLM


@pytest.fixture()
def planted():
    """A tiny store whose contents the tests know exactly."""
    with Memvara(":memory:", user="probe",
                 embedder=HashingEmbedder(dim=64), llm=NullLLM()) as mem:
        mem.remember("larkspur", "test_flag", "the Larkspur suite needs -j1")
        mem.remember("kestrel", "deploy_branch", "Kestrel deploys from release")
        ids = {c.subject: c.id for c in mem.get_all()}
        yield mem, ids


def test_run_probes_scores_a_hit_against_the_planted_store(planted):
    mem, ids = planted
    probes = [{"id": "p1", "class": "hit",
               "query": "which suite needs -j1?", "gold": [ids["larkspur"]]}]
    rows = hosted.run_probes(mem, probes, k=4)
    assert rows[0]["hit"] is True
    assert rows[0]["results"], "the row must carry the raw results for --out"
    assert rows[0]["latency_ms"] >= 0.0


def test_run_probes_records_injection_for_an_abstain_probe(planted):
    mem, _ = planted
    probes = [{"id": "p2", "class": "abstain",
               "query": "write a haiku about rain", "gold": []}]
    rows = hosted.run_probes(mem, probes, k=4)
    # Baseline truth, stated by the spec: with no floor anywhere, the store
    # injects on anything. If this ever starts passing None/False at k=4 with
    # min_score=0, the recall surface changed and the spec's baseline claim
    # needs re-verifying — that is a real signal, not a flaky test.
    assert rows[0]["false_injection"] is True
    assert rows[0]["top_score"] is not None


def test_run_probes_verbatim_uses_rank_one(planted):
    mem, ids = planted
    probes = [{"id": "p3", "class": "verbatim",
               "query": "the Larkspur suite needs -j1",
               "gold": [ids["larkspur"]]}]
    rows = hosted.run_probes(mem, probes, k=4)
    assert rows[0]["gold_rank"] is not None


def test_store_fingerprint_names_what_a_drift_warning_needs(planted):
    mem, _ = planted
    fp = hosted.store_fingerprint(mem)
    assert fp["claims"] == 2
    assert fp["surface"] == "local"
    assert fp["embedder"] and "hashing" in fp["embedder"]
    assert fp["when"]


def test_render_table_states_every_metric_present():
    agg = {
        "hit": {"n": 2, "hit_at_k": 0.5, "mean_gold_rank": 1.0},
        "verbatim": {"n": 1, "hit_at_k": 1.0, "mean_gold_rank": 2.0},
        "abstain": {"n": 2, "false_injection_rate": 1.0, "headroom": [0.45, 0.31]},
    }
    fp = {"claims": 10, "surface": "local", "when": "2026-09-01T00:00:00",
          "embedder": "hashing:64:3-5"}
    table = hosted.render_table(agg, fp)
    # Positive statements, so a deleted line fails as loudly as a wrong one.
    assert "hit@k" in table and "50.0%" in table
    assert "mean gold-rank 1.0" in table, "gold-rank must be visible — agg sets it, the render must show it"
    assert "verbatim" in table
    assert "self-retrieval@1" in table, (
        "verbatim only hits at rank 1 exactly, never at rank k — the generic "
        "hit@k label misdescribes it; the design doc names this self-retrieval@1")
    assert "mean gold-rank 2.0" in table
    assert "false-injection" in table and "100.0%" in table
    assert "0.45" in table, "headroom must be visible — it is the abstain fix's brief"
    assert "10 claims" in table


def test_main_writes_fingerprint_then_rows_to_out(tmp_path, planted):
    mem, ids = planted
    probes_path = _write_probes(tmp_path, [
        {"id": "p1", "class": "hit", "query": "which suite needs -j1?",
         "gold": [ids["larkspur"]]},
    ])
    out = tmp_path / "run.jsonl"
    rc = hosted.main(["--probes", str(probes_path), "--out", str(out)],
                     mem=mem)
    assert rc == 0
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert "claims" in lines[0], "first line is the fingerprint"
    assert lines[1]["probe_id"] == "p1"


def test_compare_runs_warns_when_the_store_moved(tmp_path):
    def run_file(path, claims, hit_rate):
        rows = [json.dumps({"claims": claims, "surface": "local",
                            "embedder": "hashing:64:3-5", "when": "t"})]
        rows.append(json.dumps({"probe_id": "p1", "cls": "hit",
                                "hit": hit_rate > 0, "gold_rank": 1,
                                "false_injection": None, "top_score": 0.5}))
        path.write_text("\n".join(rows) + "\n")
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_file(a, 10, 1)
    run_file(b, 25, 0)
    text = hosted.compare_runs(a, b)
    assert "10" in text and "25" in text and "moved" in text.lower()
    assert "hit" in text


def test_draft_probes_mark_every_row_as_draft(planted):
    mem, _ = planted
    rows = hosted.draft_probes(mem, 2)
    assert rows and all(r["draft"] is True for r in rows)
    classes = {r["class"] for r in rows}
    assert classes == {"hit", "verbatim"}
    assert all(r["gold"] for r in rows)


def test_drafted_rows_are_refused_by_the_runner_end_to_end(tmp_path, planted):
    # The full circle: what --draft emits, load_probes must refuse verbatim.
    mem, _ = planted
    path = _write_probes(tmp_path, hosted.draft_probes(mem, 1))
    with pytest.raises(SystemExit, match="draft"):
        hosted.load_probes(path)


def test_main_draft_prints_jsonl_and_runs_nothing(tmp_path, planted, capsys):
    mem, _ = planted
    rc = hosted.main(["--draft", "2", "--probes", str(tmp_path / "absent.jsonl")],
                     mem=mem)
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert all(json.loads(l)["draft"] for l in out)


def _write_recalled(tmp_path, events):
    d = tmp_path / "recalled"
    d.mkdir()
    for i, e in enumerate(events):
        (d / f"{i}.json").write_text(json.dumps(e))
    return d


def test_seed_dump_samples_filters_and_shuffles_deterministically(tmp_path):
    events = [
        {"seen": [], "query": "how do I run the larkspur suite?"},
        {"seen": [], "query": "Extract durable facts from the exchange below: ..."},
        {"seen": [], "query": "what branch does kestrel deploy from"},
    ]
    d = _write_recalled(tmp_path, events)
    dump = tmp_path / "pairs.jsonl"
    n = hosted.seed_dump(d, dump, sample=10, seed=7)
    rows = [json.loads(l) for l in dump.read_text().splitlines()]
    # The extraction prompt is internal traffic, not a user question; it must
    # not reach a judge (it would waste the judging budget the seeding spends).
    assert n == 2 and len(rows) == 2
    assert all("Extract durable facts" not in r["query"] for r in rows)
    # Blinding is the mechanism this helper rests on: a dump row must carry
    # only the query, never a results/seen field a judge could be anchored by.
    assert all(set(r.keys()) == {"id", "query"} for r in rows)
    dump2 = tmp_path / "pairs2.jsonl"
    hosted.seed_dump(d, dump2, sample=10, seed=7)
    assert dump.read_text() == dump2.read_text(), "same seed, same dump"


def test_seed_answers_turns_judgments_into_probes(tmp_path):
    dump = tmp_path / "pairs.jsonl"
    dump.write_text(json.dumps({"id": "d1", "query": "larkspur suite?"}) + "\n"
                    + json.dumps({"id": "d2", "query": "haiku please"}) + "\n"
                    + json.dumps({"id": "d3", "query": "noise"}) + "\n")
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"id": "d1", "gold": ["cl_a"]}) + "\n"
                       + json.dumps({"id": "d2", "gold": []}) + "\n"
                       + json.dumps({"id": "d3", "skip": True}) + "\n")
    probes = hosted.seed_answers(dump, answers, judged="2026-09-01")
    by_class = {p["class"]: p for p in probes}
    assert by_class["ambiguous"]["gold"] == ["cl_a"]
    assert by_class["ambiguous"]["judged"] == "2026-09-01"
    assert by_class["abstain"]["gold"] == []
    assert "judged" not in by_class["abstain"], (
        "a judgment date belongs only to ambiguous probes; a stale judged "
        "leaking onto an abstain row must not pass silently")
    assert len(probes) == 2, "a skipped row produces no probe"


def test_main_seed_answers_without_dump_refuses(tmp_path):
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"id": "d1", "gold": []}) + "\n")
    with pytest.raises(SystemExit, match="--dump") as exc:
        hosted.main(["--seed-from-recalled", str(tmp_path), "--answers",
                     str(answers), "--judged", "2026-09-01"])
    assert "--dump" in str(exc.value)
