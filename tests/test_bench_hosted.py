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
    # min_score=0 deliberately: this test is about rank, and the gold claim
    # scores 0.294 here — a hair above the shipped floor, which would make a
    # ranking test turn on a third decimal place of the hashing embedder.
    rows = hosted.run_probes(mem, probes, k=4, min_score=0.0)
    assert rows[0]["hit"] is True
    assert rows[0]["results"], "the row must carry the raw results for --out"
    assert rows[0]["latency_ms"] >= 0.0


def test_run_probes_passes_the_floor_to_both_read_surfaces(planted):
    """The floor is the measurement, so it must reach search() and recall().

    An earlier draft of this harness left `min_score` at the library default of
    0.0 on both calls and the surrounding documents called the resulting 100%
    false-injection rate a baseline. It was an artefact: `plugin/hooks/recall.py`
    has shipped `MIN_SCORE = 0.29` at both of its call sites since before this
    branch was cut, so an unfloored run measured a configuration no surface uses.

    "haiku about rain" scores 0.0 against this planted store, so the two floors
    give opposite answers — which is what makes this a guard rather than a
    restatement. Drop `min_score` from either call in `run_probes` and the
    floored half goes red.
    """
    mem, _ = planted
    probes = [{"id": "p2", "class": "abstain",
               "query": "write a haiku about rain", "gold": []}]
    unfloored = hosted.run_probes(mem, probes, k=4, min_score=0.0)
    assert unfloored[0]["false_injection"] is True
    assert unfloored[0]["top_score"] is not None

    floored = hosted.run_probes(mem, probes, k=4,
                                min_score=hosted.DEFAULT_MIN_SCORE)
    assert floored[0]["false_injection"] is False
    assert floored[0]["injected"] == []
    assert floored[0]["results"] == [], (
        "the floor must reach search() too, not only recall() — otherwise "
        "gold-rank and headroom describe a different read than the verdict does")


def test_bench_defaults_equal_the_hook_constants():
    """The mirrored constants, checked against the hook rather than themselves.

    `bench/hosted.py` copies `K` and `MIN_SCORE` instead of importing
    `plugin/hooks/recall.py`, because that module inserts its own directory at
    `sys.path[0]` on import and would put `core`/`lib` on the path of anything
    that merely runs the bench. A copy is only safe with a guard that reads the
    referent, which is what this is: change either value in either file and this
    goes red.
    """
    hooks = Path(__file__).resolve().parent.parent / "plugin" / "hooks"
    if str(hooks) not in sys.path:
        sys.path.insert(0, str(hooks))
    import recall as recall_hook  # noqa: PLC0415 - the referent, read at test time

    assert hosted.DEFAULT_MIN_SCORE == recall_hook.MIN_SCORE, (
        "bench/hosted.py mirrors the recall hook's MIN_SCORE; they have drifted")
    assert hosted.DEFAULT_K == recall_hook.K, (
        "bench/hosted.py mirrors the recall hook's K; they have drifted")


def test_run_probes_verbatim_uses_rank_one(planted):
    mem, ids = planted
    probes = [{"id": "p3", "class": "verbatim",
               "query": "the Larkspur suite needs -j1",
               "gold": [ids["larkspur"]]}]
    rows = hosted.run_probes(mem, probes, k=4, min_score=0.0)
    # rank 1 exactly, and the hit that follows from it. "is not None" would
    # stay green if the rule relaxed to "present at any rank", which is the
    # whole behaviour self-retrieval@1 exists to pin.
    assert rows[0]["gold_rank"] == 1
    assert rows[0]["hit"] is True


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
        # 1.0, not 2.0: a verbatim hit is rank 1 by definition, so `aggregate`
        # can never produce any other value here and a fixture saying 2.0 was
        # blessing an unreachable state.
        "verbatim": {"n": 1, "hit_at_k": 1.0, "mean_gold_rank": 1.0},
        "abstain": {"n": 2, "false_injection_rate": 1.0, "headroom": [0.45, 0.31]},
    }
    fp = {"claims": 10, "surface": "local", "when": "2026-09-01T00:00:00",
          "embedder": "hashing:64:3-5"}
    table = hosted.render_table(agg, fp)
    # Positive statements, so a deleted line fails as loudly as a wrong one.
    assert "hit@k" in table and "50.0%" in table
    assert "verbatim" in table
    assert "self-retrieval@1" in table, (
        "verbatim only hits at rank 1 exactly, never at rank k — the generic "
        "hit@k label misdescribes it; the design doc names this self-retrieval@1")
    hit_line = next(l for l in table.splitlines() if l.strip().startswith("hit"))
    verbatim_line = next(l for l in table.splitlines()
                         if l.strip().startswith("verbatim"))
    assert "mean gold-rank 1.0" in hit_line, (
        "gold-rank must be visible on the hit row — agg sets it, the render must show it")
    assert "mean gold-rank" not in verbatim_line, (
        "a verbatim hit is rank 1 by definition, so a rank column there is a "
        "constant dressed as a measurement; the spec assigns gold-rank to "
        "hit/ambiguous only")
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
    rc = hosted.main(["--probes", str(probes_path), "--out", str(out),
                      "--min-score", "0"], mem=mem)
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


def test_compare_runs_is_silent_when_the_store_did_not_move(tmp_path):
    """A warning that fires on every comparison is the same as no warning.

    The drift banner is only readable as a signal if the no-drift case has none,
    so the absence is guarded beside the presence — and the deltas are asserted
    positively, so a compare_runs that stopped saying anything at all fails here
    rather than passing this test by silence.
    """
    def run_file(path, hit):
        rows = [json.dumps({"claims": 10, "surface": "local",
                            "embedder": "hashing:64:3-5", "when": "t"}),
                json.dumps({"probe_id": "p1", "cls": "hit", "hit": hit,
                            "gold_rank": 1, "false_injection": None,
                            "top_score": 0.5})]
        path.write_text("\n".join(rows) + "\n")
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_file(a, True)
    run_file(b, False)
    text = hosted.compare_runs(a, b)
    assert "hit@k 100.0% -> 0.0%" in text, "the deltas must still be reported"
    assert "WARNING" not in text and "moved" not in text.lower(), (
        "same fingerprint, no drift warning — a banner on every run is noise, "
        "not a signal")


def test_draft_probes_mark_every_row_as_draft(planted):
    mem, _ = planted
    rows = hosted.draft_probes(mem, 2)
    assert rows and all(r["draft"] is True for r in rows)
    classes = {r["class"] for r in rows}
    assert classes == {"hit", "verbatim"}
    assert all(r["gold"] for r in rows)


def test_draft_probes_sample_rather_than_take_the_id_sorted_prefix():
    """A sample, as the docstring and the spec both promise.

    Claim ids are content digests, so `sorted(...)[:n]` is an arbitrary slice of
    the store that never moves: ten drafts in a row would propose the same
    corner of it forever. Seeded, so a draft is still reproducible.
    """
    with Memvara(":memory:", user="probe",
                 embedder=HashingEmbedder(dim=64), llm=NullLLM()) as mem:
        for i in range(8):
            mem.remember(f"subject{i}", "note", f"the {i}th planted value")
        every = sorted(c.id for c in mem.get_all())
        prefix = set(every[:3])

        assert hosted.draft_probes(mem, 3, seed=7) == \
            hosted.draft_probes(mem, 3, seed=7), \
            "same store and same seed must give the same draft"

        picks = [{r["gold"][0] for r in hosted.draft_probes(mem, 3, seed=s)}
                 for s in range(20)]
        assert all(len(p) == 3 and p <= set(every) for p in picks)
        assert any(p != prefix for p in picks), (
            "every seed returned the id-sorted prefix — draft_probes is "
            "slicing, not sampling")

        # n larger than the store is the whole store, not an error.
        assert len({r["gold"][0] for r in hosted.draft_probes(mem, 100)}) == 8


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


def test_main_seed_answers_without_judged_refuses(tmp_path):
    """The other half of the seeding refusal, which had no guard at all.

    An `ambiguous` probe's gold is a human judgment and a judgment ages as the
    store changes, so a seeded probe with no date is a claim with no shelf life
    printed on it. `--dump` was guarded; this one was not, and an unguarded
    refusal is a refusal that can be deleted silently.
    """
    dump = tmp_path / "pairs.jsonl"
    dump.write_text(json.dumps({"id": "d1", "query": "larkspur suite?"}) + "\n")
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"id": "d1", "gold": ["cl_a"]}) + "\n")
    with pytest.raises(SystemExit, match="--judged") as exc:
        hosted.main(["--seed-from-recalled", str(tmp_path),
                     "--answers", str(answers), "--dump", str(dump)])
    assert "--judged" in str(exc.value)


# -- the hosted surface ------------------------------------------------------
#
# `RemoteMemvara.recall` is not `Memvara.recall`, and the harness's first draft
# assumed it was: it called `recall(..., with_ids=True)`, which raises TypeError
# against a hosted store, and read the ids back through
# `getattr(recalled, "claim_ids", []) or []`. Dropping the kwarg to "fix" the
# TypeError would then have made a `str` yield an empty list, scoring every
# abstain probe as a pass and reporting a flawless 0% false-injection rate on
# the one path this tool is named for. The stubs below carry the remote's real
# signature, checked against the real class, so the mismatch cannot be assumed
# away a second time.

import inspect  # noqa: E402

from memvara.remote.api import RemoteMemvara  # noqa: E402


class RemoteShaped:
    """A store with `RemoteMemvara`'s read surface: recall() returns `str`."""

    def __init__(self, rows):
        self.rows = rows
        self.recall_calls = []
        self.search_calls = []

    def search(self, query, *, k=10, min_score=0.0, memory_types=None,
               include_episodes=False):
        self.search_calls.append({"k": k, "min_score": min_score})
        return [_Row(cid, score) for cid, score in self.rows
                if score >= min_score][:k]

    def recall(self, query, *, k=8, min_score=0.0, memory_types=None,
               include_episodes=False, budget=None):
        self.recall_calls.append({"k": k, "min_score": min_score})
        return "MEMORY\n- a rendered memory"


class _Row:
    def __init__(self, cid, score):
        self.claim = type("C", (), {"id": cid})()
        self.score = score


class NamesNothing:
    """Declares `with_ids` and returns a bare `str` anyway.

    The exact shape the deleted `getattr(..., [])` fallback read as "nothing was
    injected". Not a hosted store — a local-shaped one whose recall surface has
    stopped naming what it rendered.
    """

    def search(self, query, *, k=10, min_score=0.0):
        return [_Row("cl_a", 0.9)]

    def recall(self, query, *, k=8, min_score=0.0, with_ids=False):
        return "MEMORY\n- a rendered memory"


def test_the_remote_shaped_stub_carries_the_real_remote_signature():
    """Compared against the referent, not against a convenient stand-in.

    If `RemoteMemvara.recall` ever grows `with_ids`, this goes red and the
    harness's hosted branch should be revisited — that is the point of reading
    the parameter list off the real class here rather than trusting the stub.
    """
    real = list(inspect.signature(RemoteMemvara.recall).parameters)
    assert "with_ids" not in real, (
        "RemoteMemvara.recall grew with_ids — bench/hosted.py can now read the "
        "hosted injection surface directly instead of inferring it from search()")
    assert list(inspect.signature(RemoteShaped.recall).parameters) == real, (
        "the stub has drifted from the class it stands in for; a stub that does "
        "not match its referent guards nothing")


def test_run_probes_against_a_hosted_store_derives_injection_from_search():
    store = RemoteShaped([("cl_a", 0.55), ("cl_b", 0.10)])
    probes = [{"id": "p1", "class": "abstain", "query": "haiku", "gold": []}]
    rows = hosted.run_probes(store, probes, k=4, min_score=0.29)
    # The defect this replaces reported False here, from an empty list a `str`
    # never had — a clean 0% that no store earned.
    assert rows[0]["false_injection"] is True
    assert rows[0]["injected"] == ["cl_a"]
    # recall() is still called on this route, so a hosted recall failure fails
    # the run rather than passing unmeasured; and the floor reaches both calls.
    assert store.recall_calls == [{"k": 4, "min_score": 0.29}]
    assert store.search_calls == [{"k": 4, "min_score": 0.29}]


def test_a_hosted_abstain_probe_can_still_pass():
    """The verdict has to be able to go both ways on this surface too."""
    store = RemoteShaped([("cl_a", 0.10)])
    probes = [{"id": "p1", "class": "abstain", "query": "haiku", "gold": []}]
    rows = hosted.run_probes(store, probes, k=4, min_score=0.29)
    assert rows[0]["false_injection"] is False
    assert rows[0]["injected"] == []


def test_run_probes_refuses_a_recall_that_cannot_name_what_it_rendered():
    probes = [{"id": "p1", "class": "abstain", "query": "haiku", "gold": []}]
    with pytest.raises(SystemExit, match="claim_ids") as exc:
        hosted.run_probes(NamesNothing(), probes, k=4, min_score=0.29)
    assert "abstain" in str(exc.value), (
        "the refusal must say what the silent fallback would have cost, or the "
        "next reader restores it")


# -- store lifetime ----------------------------------------------------------


class Counting:
    """A store that counts its own closes and proxies everything else."""

    def __init__(self, inner):
        self._inner = inner
        self.closes = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def close(self):
        self.closes += 1


def _probe_file(tmp_path, ids):
    return _write_probes(tmp_path, [
        {"id": "p1", "class": "hit", "query": "which suite needs -j1?",
         "gold": [ids["larkspur"]]},
    ])


def test_main_closes_a_store_it_opened_exactly_once(tmp_path, planted, monkeypatch):
    mem, ids = planted
    counting = Counting(mem)
    monkeypatch.setattr(hosted, "_open_store", lambda args: counting)
    rc = hosted.main(["--probes", str(_probe_file(tmp_path, ids)),
                      "--min-score", "0"])
    assert rc == 0
    assert counting.closes == 1, "a self-opened store is closed once, not twice"


def test_main_closes_a_store_it_opened_even_when_the_run_refuses(tmp_path, planted,
                                                                 monkeypatch):
    """The refusal paths are where a leak hides: they are the untaken ones."""
    mem, _ = planted
    counting = Counting(mem)
    monkeypatch.setattr(hosted, "_open_store", lambda args: counting)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(SystemExit, match="no probes"):
        hosted.main(["--probes", str(empty)])
    assert counting.closes == 1


def test_main_never_closes_a_store_it_was_handed(tmp_path, planted, monkeypatch):
    """`mem=` is the caller's handle; closing it would shut a store this call
    did not open — the tests' own fixture among them."""
    mem, ids = planted
    counting = Counting(mem)

    def refuse(args):  # pragma: no cover - reaching this is the failure
        raise AssertionError("main opened a store although one was handed to it")

    monkeypatch.setattr(hosted, "_open_store", refuse)
    rc = hosted.main(["--probes", str(_probe_file(tmp_path, ids)),
                      "--min-score", "0"], mem=counting)
    assert rc == 0
    assert counting.closes == 0


def test_main_draft_passes_the_seed_through(tmp_path, capsys):
    """`--seed` must reach `--draft`, not only `--seed-from-recalled`.

    `draft_probes` grew a `seed` parameter and `main` kept calling it
    positionally, so the flag silently steered one subcommand and did nothing
    for the other. A flag that quietly does nothing is worse than an absent
    one, and only a test driven through `main` can see it — `draft_probes`
    itself was already guarded and stayed green.
    """
    with Memvara(":memory:", user="probe",
                 embedder=HashingEmbedder(dim=64), llm=NullLLM()) as mem:
        for i in range(8):
            mem.remember(f"subject{i}", "note", f"the {i}th planted value")

        def draft(seed):
            rc = hosted.main(["--draft", "3", "--seed", str(seed),
                              "--probes", str(tmp_path / "absent.jsonl")],
                             mem=mem)
            assert rc == 0
            return capsys.readouterr().out

        base = draft(11)
        assert draft(11) == base, "same seed through the CLI must draft the same rows"
        assert any(draft(s) != base for s in range(20) if s != 11), (
            "no seed changed the draft — main is not passing --seed to "
            "draft_probes, so the flag steers only --seed-from-recalled")
