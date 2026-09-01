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


@pytest.mark.parametrize("raw, description", [
    (None, "unset — the constant stands"),
    ("0.45", "a recalibrated floor"),
    ("0", "zero is a setting, not an absence"),
    ("1.7", "clamped to 1.0"),
    ("-3", "clamped to 0.0"),
    ("banana", "unparseable — the constant stands"),
])
def test_bench_default_floor_resolves_as_the_hook_resolves_it(monkeypatch, raw,
                                                              description):
    """The *effective* floor, not just the constant, compared to the referent.

    The hook's shipped value is `_min_score()`, which honours
    `MEMVARA_RECALL_MIN_SCORE` — and the hook's own comment tells a store owner
    to recalibrate and set it. A bench that mirrored only `MIN_SCORE` would
    measure 0.29 for anyone who followed that advice while the documentation
    claimed it measured what the hook injects. Every branch of the resolution
    is compared against the hook's own function, so the two cannot diverge:
    change either clamp, either fallback, or the variable name, and this goes
    red for the case that changed.
    """
    hooks = Path(__file__).resolve().parent.parent / "plugin" / "hooks"
    if str(hooks) not in sys.path:
        sys.path.insert(0, str(hooks))
    import recall as recall_hook  # noqa: PLC0415 - the referent, read at test time

    if raw is None:
        monkeypatch.delenv(hosted.ENV_MIN_SCORE, raising=False)
    else:
        monkeypatch.setenv(hosted.ENV_MIN_SCORE, raw)
    assert hosted.default_min_score() == recall_hook._min_score(), description


def test_main_takes_its_default_floor_from_the_environment(tmp_path, planted,
                                                           monkeypatch):
    """The resolution has to reach the run, not only the helper.

    A `default=` computed once at import, or a helper nothing calls, would
    leave `--min-score` on the constant while `default_min_score` tested clean.
    1.0 filters everything on this store, 0.0 filters nothing, so the two
    settings give opposite verdicts on the same probe.
    """
    mem, ids = planted
    probes = _write_probes(tmp_path, [
        {"id": "p1", "class": "hit", "query": "which suite needs -j1?",
         "gold": [ids["larkspur"]]}])

    def run(value, out):
        monkeypatch.setenv(hosted.ENV_MIN_SCORE, value)
        assert hosted.main(["--probes", str(probes), "--out", str(out)],
                           mem=mem) == 0
        return [json.loads(l) for l in out.read_text().splitlines()][1]

    assert run("0", tmp_path / "low.jsonl")["hit"] is True
    assert run("1.0", tmp_path / "high.jsonl")["hit"] is False, (
        "MEMVARA_RECALL_MIN_SCORE did not reach the run's --min-score default")


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


def test_store_fingerprint_names_a_wrapped_embedder(planted):
    """A wrapper must not fingerprint as `embedder: None`.

    `getattr(embedder, "name", None)` returns None for anything that delegates,
    and a fingerprint that cannot state the embedder cannot warn that it
    changed — which is the whole job `compare_runs`' drift banner reads it for.
    The library's own `embedder_name` unwraps `.inner` and falls back to the
    class name; this pins that the bench uses it.
    """
    from memvara.embed.fingerprint import embedder_name

    mem, _ = planted

    class Wrapping:
        """Declares no `name`; delegates to what it wraps, as `_name_of` expects."""

        def __init__(self, inner):
            self.inner = inner

    class Nameless:
        pass

    class _Store:
        def __init__(self, embedder):
            self.embedder = embedder

        def count(self):
            return 1

    wrapped = Wrapping(mem.embedder)
    fp = hosted.store_fingerprint(_Store(wrapped))
    assert fp["embedder"] == embedder_name(wrapped) == embedder_name(mem.embedder)
    assert hosted.store_fingerprint(_Store(Nameless()))["embedder"] == "Nameless", (
        "a nameless embedder falls back to its class name, never to None")


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


def test_compare_runs_labels_the_verbatim_delta_self_retrieval(tmp_path):
    """The two renderers must name the metric the same way.

    `render_table` branches to self-retrieval@1 for verbatim with a comment
    saying hit@k would misdescribe it; `compare_runs` hard-coded hit@k for
    every non-abstain class, so the same number carried two names depending on
    which command printed it. Neither existing compare test had a verbatim row,
    which is why the two could disagree undisturbed.
    """
    def run_file(path, hit):
        rows = [json.dumps({"claims": 10, "surface": "local",
                            "embedder": "hashing:64:3-5", "when": "t"}),
                json.dumps({"probe_id": "p1", "cls": "verbatim", "hit": hit,
                            "gold_rank": 1 if hit else 2,
                            "false_injection": None, "top_score": 0.5}),
                json.dumps({"probe_id": "p2", "cls": "hit", "hit": True,
                            "gold_rank": 1, "false_injection": None,
                            "top_score": 0.5})]
        path.write_text("\n".join(rows) + "\n")
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_file(a, True)
    run_file(b, False)
    text = hosted.compare_runs(a, b)
    verbatim_line = next(l for l in text.splitlines() if "verbatim" in l)
    assert "self-retrieval@1 100.0% -> 0.0%" in verbatim_line, (
        "verbatim only counts a hit at rank 1 exactly; the compare view must "
        "use the same label render_table does")
    hit_line = next(l for l in text.splitlines() if l.strip().startswith("hit:"))
    assert "hit@k" in hit_line, "the hit class keeps its own label"


def test_draft_probes_mark_every_row_as_draft(planted):
    mem, _ = planted
    rows = hosted.draft_probes(mem, 2)
    assert rows and all(r["draft"] is True for r in rows)
    classes = {r["class"] for r in rows}
    assert classes == {"hit", "verbatim"}
    assert all(r["gold"] for r in rows)


def test_draft_probes_query_the_embedded_text_not_the_raw_object_slot(planted):
    """`Claim.text` is what retrieval embedded; `Claim.object` is a slot value.

    self-retrieval@1 asks whether a claim's own text returns that claim first,
    so a verbatim probe drafted from the raw object ("Lisbon" where the indexed
    string is "user lives in Lisbon") measures something weaker than the metric
    is named for. The planted store's two fields genuinely differ — `remember`
    renders `text` as "<subject> <predicate> <object>" — so a regression to
    `.object` fails here rather than passing on a fixture where they coincide.
    """
    mem, _ = planted
    by_id = {c.id: c for c in mem.get_all()}
    rows = hosted.draft_probes(mem, 2)
    assert rows
    for row in rows:
        claim = by_id[row["gold"][0]]
        assert claim.text != claim.object, (
            "this fixture cannot tell the two fields apart — it guards nothing")
        assert row["query"] == claim.text, (
            "a drafted query must be the embedded text, not the object slot")


def test_draft_probes_fall_back_to_the_object_when_a_claim_has_no_text():
    """`Claim.text` defaults to `""`, and an empty query is refused by
    `load_probes` — so a textless claim must still draft something askable."""
    class _Textless:
        id = "cl_textless"
        text = ""
        object = "the Larkspur suite needs -j1"

    class _Store:
        def get_all(self):
            return [_Textless()]

    rows = hosted.draft_probes(_Store(), 1)
    assert {r["query"] for r in rows} == {"the Larkspur suite needs -j1"}


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


#: The recall hook's own truncation length, mirrored so the fixture below can
#: write a query at the boundary the real writer produces. Read off the hook in
#: `test_the_recalled_fixture_matches_what_the_hook_writes` rather than trusted.
MAX_CARRY_CHARS = 300


def test_main_draft_zero_is_a_no_op_not_a_full_run(tmp_path, planted, capsys):
    """`--draft 0` must draft nothing, not fall through to a scoring pass.

    The flag defaulted to `0`, so `if args.draft:` could not tell an explicit
    zero from an absent flag. A wrapper computing `--draft {remaining}` then
    got a full probe-suite run — here, against a probes path that does not
    exist. `--probes` names a missing file deliberately: a fall-through raises
    rather than passing quietly, so this test cannot go green by accident.
    """
    mem, _ = planted
    rc = hosted.main(["--draft", "0", "--probes", str(tmp_path / "absent.jsonl")],
                     mem=mem)
    assert rc == 0
    assert capsys.readouterr().out == "", "--draft 0 drafts nothing"


def _write_recalled(tmp_path, queries):
    """The directory `plugin/hooks/recall.py` actually writes.

    One file per **session**, named for the session id, rewritten in place on
    every turn — so each file holds exactly one query: that session's most
    recent substantive prompt, truncated to `MAX_CARRY_CHARS`. Beside it the
    real writer stores `seen` (prompt-line digests) and the `standing` pair.

    An earlier fixture here wrote `0.json`, `1.json`, `2.json` and called them
    events. That model of the directory cannot exist, and modelling it is how
    the harness came to describe itself as sampling recall traffic when it
    samples session tails. A fixture that models the writer honestly is the
    guard; one that invents a shape guards the invention.
    """
    d = tmp_path / "recalled"
    d.mkdir()
    for i, q in enumerate(queries):
        session = f"0a1b2c3d-0000-4000-8000-00000000000{i}"
        (d / f"{session}.json").write_text(json.dumps({
            "seen": ["a" * 16, "b" * 16],
            "query": q[:MAX_CARRY_CHARS],
            "standing": "c" * 16,
            "standing_at": 1756684800.0,
        }))
    return d


def test_the_recalled_fixture_matches_what_the_hook_writes(tmp_path, monkeypatch):
    """The fixture's shape, checked against the writer instead of itself.

    `_write_recalled` hard-codes a filename convention, a truncation length and
    a key set. All three are the hook's, so all three are taken from the hook
    here — by running `_write_state` into a temporary directory and reading
    what it produced. A fixture checked against a copy of its own assumptions
    is what let "one file per recall event" stand; this compares against the
    referent, and `SEEN_DIR` is redirected so the real `~/.memvara` is never
    touched, read or written.
    """
    hooks = Path(__file__).resolve().parent.parent / "plugin" / "hooks"
    if str(hooks) not in sys.path:
        sys.path.insert(0, str(hooks))
    import recall as recall_hook  # noqa: PLC0415 - the referent, read at test time

    assert MAX_CARRY_CHARS == recall_hook.MAX_CARRY_CHARS, (
        "the fixture truncates at the hook's MAX_CARRY_CHARS; they have drifted")

    real = tmp_path / "hook-seen"
    monkeypatch.setattr(recall_hook, "SEEN_DIR", str(real))
    session = "0a1b2c3d-0000-4000-8000-000000000000"
    recall_hook._write_state(session, ["a" * 16], "q " * 400, ("c" * 16, 1.0))

    produced = sorted(p.name for p in real.glob("*.json"))
    assert produced == [f"{session}.json"], (
        "one file per session, named for the session — not one per recall event")
    written = json.loads((real / f"{session}.json").read_text())
    assert len(written["query"]) == MAX_CARRY_CHARS, (
        "the writer truncates the carried query; the fixture must too")

    mine = json.loads(next(_write_recalled(tmp_path, ["q"]).glob("*.json")).read_text())
    assert set(mine) == set(written), (
        "the fixture writes the hook's key set; _write_state's keys have moved")


def test_seed_dump_samples_filters_and_shuffles_deterministically(tmp_path):
    long_one = "how do I run the larkspur suite " + "and everything after it " * 20
    d = _write_recalled(tmp_path, [
        long_one,
        "Extract durable facts from the exchange below: ...",
        "what branch does kestrel deploy from",
    ])
    dump = tmp_path / "pairs.jsonl"
    n, skipped = hosted.seed_dump(d, dump, sample=10, seed=7)
    rows = [json.loads(l) for l in dump.read_text().splitlines()]
    # The extraction prompt is internal traffic, not a user question; it must
    # not reach a judge (it would waste the judging budget the seeding spends).
    assert (n, skipped) == (2, 0) and len(rows) == 2
    assert all("Extract durable facts" not in r["query"] for r in rows)
    # Blinding is the mechanism this helper rests on: a dump row must carry
    # only the query, never a results/seen field a judge could be anchored by.
    assert all(set(r.keys()) == {"id", "query"} for r in rows)
    # The writer truncates, so a judge sees a clipped query and the dump must
    # carry it as-is rather than pretending the tail was ever recorded.
    clipped = next(r for r in rows if r["query"].startswith("how do I run"))
    assert len(clipped["query"]) <= MAX_CARRY_CHARS
    assert clipped["query"] == " ".join(long_one[:MAX_CARRY_CHARS].split())
    dump2 = tmp_path / "pairs2.jsonl"
    hosted.seed_dump(d, dump2, sample=10, seed=7)
    assert dump.read_text() == dump2.read_text(), "same seed, same dump"


def test_seed_dump_counts_the_files_it_could_not_read(tmp_path):
    """A skip and a never-ran must not print the same line.

    Valid-but-not-an-object JSON parses and then raises on `.get`, which took
    the whole run down over one bad file; the caught case dropped files with no
    count at all, so a directory where every file failed printed exactly what
    an empty directory prints.
    """
    d = _write_recalled(tmp_path, ["what branch does kestrel deploy from"])
    (d / "11111111-0000-4000-8000-000000000000.json").write_text("null")
    (d / "22222222-0000-4000-8000-000000000000.json").write_text("[1, 2]")
    (d / "33333333-0000-4000-8000-000000000000.json").write_text("{not json")
    dump = tmp_path / "pairs.jsonl"
    n, skipped = hosted.seed_dump(d, dump, sample=10, seed=7)
    assert n == 1, "the readable file still seeds a query"
    assert skipped == 3, "every unreadable file is counted, not dropped in silence"


def test_main_seed_dump_reports_the_skipped_count(tmp_path, capsys):
    d = _write_recalled(tmp_path, ["what branch does kestrel deploy from"])
    (d / "44444444-0000-4000-8000-000000000000.json").write_text("null")
    rc = hosted.main(["--seed-from-recalled", str(d),
                      "--dump", str(tmp_path / "pairs.jsonl")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "skipped 1" in out, (
        "the count must reach the person at the terminal — a skip reported "
        "nowhere is a skip that looks like an empty directory")
    assert "wrote 1 blinded queries" in out


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


def test_seed_answers_refuses_a_twice_judged_id(tmp_path):
    """Last-write-wins is the worst of the three options, per `FileReader`.

    An answers file assembled by concatenating two judging passes, or written
    by an agent that retried an item, would otherwise seed whichever judgment
    happened to be last in the file — and which one was meant is not
    recoverable from it.
    """
    dump = tmp_path / "pairs.jsonl"
    dump.write_text(json.dumps({"id": "d1", "query": "larkspur suite?"}) + "\n")
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"id": "d1", "gold": ["cl_a"]}) + "\n"
                       + json.dumps({"id": "d1", "gold": ["cl_b"]}) + "\n")
    with pytest.raises(SystemExit, match="already") as exc:
        hosted.seed_answers(dump, answers, judged="2026-09-01")
    assert "line 2" in str(exc.value) and "line 1" in str(exc.value), (
        "the refusal must name both lines, or the file has to be searched by hand")


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


# --- `_open_store`, against real stores on disk ------------------------------
#
# The one part of this tool that only ever runs in production. Every test above
# hands `main` a `mem=`, so `_open_store` shipped with no guard at all and
# `--db` — its own documented local invocation — could not open a store built by
# anything other than whatever `default_embedder()` returns in this process. The
# stores here are built under `tmp_path` with `HashingEmbedder`, which is
# deterministic, offline and costs milliseconds; nothing in this section reads
# `~/.memvara/` or the network.

import argparse  # noqa: E402


def _args(db, **kw):
    """The namespace `main` would have built, so a new flag cannot silently
    default to whatever `getattr` happened to return in a test."""
    return argparse.Namespace(db=str(db), tenant=kw.pop("tenant", "default"),
                              user=kw.pop("user", ""), **kw)


def _store_on_disk(path, embedder, **scope):
    """A one-claim store at `path`, written by `embedder`, closed. Returns its ids."""
    with Memvara(str(path), embedder=embedder, llm=NullLLM(), **scope) as mem:
        mem.remember("larkspur", "test_flag", "the Larkspur suite needs -j1")
        return {c.subject: c.id for c in mem.get_all()}


class _NamedLikeAModel(HashingEmbedder):
    """Real vectors under a name no constructor can be recovered from.

    Stands in for a store built by a neural model: `LocalEmbedder` fingerprints
    itself as `local:<model id>`, which names the model but is not something this
    bench may turn back into an object — doing so would mean deciding, for
    somebody, to import torch and possibly fetch weights. A `HashingEmbedder`
    underneath keeps the test offline and deterministic; only the name matters.
    """

    def __init__(self, name: str, dim: int) -> None:
        super().__init__(dim=dim)
        self._name = name

    @property
    def name(self) -> str:
        return self._name


def test_open_store_opens_a_store_built_with_a_non_default_embedder(tmp_path):
    """The shipped defect, at the only layer that could have caught it.

    `Memvara(path)` takes `default_embedder()`, which is a 384-dimensional
    sentence-transformers model wherever `memvara[local-embed]` is installed and
    a 512-dimensional `HashingEmbedder` where it is not. This store is neither,
    so against the old one-line body — `return Memvara(args.db)` — this raises
    `EmbedderMismatchError` on both kinds of machine, which is exactly what
    running the tool against a real store did.
    """
    db = tmp_path / "probe.db"
    ids = _store_on_disk(db, HashingEmbedder(dim=64))

    mem = hosted._open_store(_args(db))
    try:
        assert hosted.store_fingerprint(mem)["embedder"] == "hashing:64:3-5", (
            "the store must be opened with the embedder that wrote it, read back "
            "off the fingerprint the run records")
        found = [r.claim.id for r in
                 mem.search("which suite needs -j1?", k=4, min_score=0.0)]
        assert ids["larkspur"] in found, (
            "opened, but not readable — a store opened with an embedder of the "
            "right width but the wrong space would look exactly like this")
    finally:
        mem.close()


def test_open_store_refuses_a_store_whose_embedder_it_cannot_reconstruct(
        tmp_path, monkeypatch):
    """A name that is not a hashing embedder and is not what this process has.

    The refusal has to name the store's recorded embedder, because "cannot open
    it" without saying what wrote it leaves the reader with nothing to install
    or pass.

    `default_embedder` is stubbed so the verdict is the same on a machine with
    sentence-transformers and one without — and so no test here is the thing that
    imports torch.
    """
    db = tmp_path / "unknown.db"
    _store_on_disk(db, _NamedLikeAModel("local:pretend/tiny-model", 48))
    monkeypatch.setattr("memvara.embed.default_embedder",
                        lambda *a, **k: HashingEmbedder(dim=512))

    with pytest.raises(SystemExit) as exc:
        hosted._open_store(_args(db))
    message = str(exc.value)
    assert "local:pretend/tiny-model" in message, (
        "the refusal must name what wrote the store")
    assert str(db) in message, "and which store it is talking about"


def test_open_store_still_uses_the_default_when_that_is_what_wrote_the_store(
        tmp_path, monkeypatch):
    """The branch that keeps the sentence-transformers case working.

    A store built by a neural model opens fine today, because `default_embedder()`
    returns that same model wherever it is installed. Reconstructing hashing
    embedders and refusing everything else would have taken that away, and the
    only people who would have noticed are the ones with the extra installed.
    """
    db = tmp_path / "modelled.db"
    ids = _store_on_disk(db, _NamedLikeAModel("local:pretend/tiny-model", 48))
    monkeypatch.setattr(
        "memvara.embed.default_embedder",
        lambda *a, **k: _NamedLikeAModel("local:pretend/tiny-model", 48))

    mem = hosted._open_store(_args(db))
    try:
        found = [r.claim.id for r in
                 mem.search("which suite needs -j1?", k=4, min_score=0.0)]
        assert ids["larkspur"] in found
    finally:
        mem.close()


@pytest.mark.parametrize("recorded_name, recorded_dim", [
    # Reconstructable: the pre-guard code built HashingEmbedder(dim=64) off this
    # name and opened a brand-new store at the stale width, silently.
    ("hashing:64:3-5", 64),
    # Not reconstructable: the pre-guard code refused outright, claiming a store
    # that does not exist held vectors it could not read.
    ("local:pretend/tiny-model", 48),
])
def test_open_store_ignores_a_fingerprint_with_no_vectors_behind_it(
        tmp_path, monkeypatch, recorded_name, recorded_dim):
    """A sidecar outliving its store must not decide anything.

    `Memvara._check_embedder` reads `stored_dim(self.store)` next to the recorded
    fingerprint and skips the whole compatibility check when it is `None`: a store
    with no vectors has nothing to be incompatible with, so the embedder in hand
    becomes its owner. Honouring the sidecar without asking that question regressed
    two cases the shipped one-liner got right for free, by never reading the sidecar
    at all — this file's entire subject is measurement honesty, so a fix that
    invents a constraint the library does not have is worse than the defect.

    Both sidecars here are orphans: the `.db` they name was never created. The
    assertion is positive — the *default* embedder must be the one that ends up
    owning the store — because "not the stale one" would also pass on a refusal.

    `default_embedder` is pinned rather than reached: `tests/conftest.py` makes
    reaching it a hard failure so the suite's vector space is not a property of what
    happens to be installed, and it also redirects `HOME`, so the real one cannot
    see the model cache and goes to the network. Pinning it to
    `HashingEmbedder(dim=512)` — what `default_embedder()` returns with
    sentence-transformers absent — is that rule honoured, not evaded, and it is the
    only way to exercise a branch whose whole content is *fall through to the
    library default*.

    Both names are pinned, because they are two different bindings and this test
    can traverse either: `Memvara.__init__` looks up `memvara.core.default_embedder`
    and `_store_embedder` imports `memvara.embed.default_embedder`. Pinning only the
    first made the sabotage run for this test hang on a model download instead of
    going red, which is a guard that cannot be proven to fail.
    """
    from memvara.embed.fingerprint import SIDECAR_SUFFIX

    db = tmp_path / "orphaned.db"
    (tmp_path / ("orphaned.db" + SIDECAR_SUFFIX)).write_text(
        json.dumps({"embedder": recorded_name, "dim": recorded_dim}))
    assert not db.exists(), "the point of the case is that no store is there yet"
    for binding in ("memvara.core.default_embedder", "memvara.embed.default_embedder"):
        monkeypatch.setattr(binding, lambda dim=512: HashingEmbedder(dim=512))

    mem = hosted._open_store(_args(db))
    try:
        assert hosted.store_fingerprint(mem)["embedder"] == "hashing:512:3-5", (
            "an empty store must be owned by the embedder this process would "
            "otherwise use, not by whatever a leftover sidecar names")
        mem.remember("larkspur", "test_flag", "the Larkspur suite needs -j1")
        assert mem.search("which suite needs -j1?", k=4, min_score=0.0), (
            "and it must be readable afterwards")
    finally:
        mem.close()


# --- scope: the store is a file, but a run reads one scope out of it ---------
#
# The second shipped defect, and the same shape as the first: `_open_store` built
# a `Memvara` with no scope, so `--db PATH` opened at `tenant="default"` against
# a real store whose claims all live under a named tenant. The tool's own
# documented invocation exited 0 and printed nothing; `store_fingerprint` said
# `claims: 0`. A wrongly-scoped store and a genuinely empty one were
# byte-identical in the output.
#
# Every store here is built under `tmp_path` with `HashingEmbedder` — offline,
# deterministic, milliseconds. Nothing reads `~/.memvara/` or the network.


def test_store_fingerprint_names_the_scope_it_read_at(planted):
    """Positively, both keys, so a fingerprint that stopped recording the scope
    fails here rather than passing by absence."""
    fp = hosted.store_fingerprint(planted[0])
    assert fp["tenant"] == "default"
    assert fp["user"] == "probe", (
        "the planted fixture reads at user=probe; a fingerprint that cannot "
        "state the scope cannot warn that two runs read different populations")


def test_compare_runs_warns_when_only_the_scope_moved(tmp_path):
    """Same file, same embedder, same claim count — different populations.

    Two runs at different scopes are no more a before/after than two different
    stores are, which is why the drift banner covers the scope keys the same
    way it covers `claims` and `embedder`.
    """
    def run_file(path, tenant, user):
        rows = [json.dumps({"claims": 10, "surface": "local",
                            "embedder": "hashing:64:3-5", "tenant": tenant,
                            "user": user, "when": "t"}),
                json.dumps({"probe_id": "p1", "cls": "hit", "hit": True,
                            "gold_rank": 1, "false_injection": None,
                            "top_score": 0.5})]
        path.write_text("\n".join(rows) + "\n")
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_file(a, "workstation", None)
    run_file(b, "default", "probe")
    text = hosted.compare_runs(a, b)
    assert "WARNING" in text
    assert "tenant workstation -> default" in text
    assert "user None -> probe" in text


def test_open_store_reads_the_tenant_the_claims_are_under(tmp_path):
    """The shipped defect, at the layer that could have caught it.

    Unscoped, this store reads as empty — that is the whole bug, so it is
    asserted rather than merely implied. `--tenant` then makes the same file
    answer, which is the fix.
    """
    db = tmp_path / "scoped.db"
    ids = _store_on_disk(db, HashingEmbedder(dim=64), tenant="workstation")

    blind = hosted._open_store(_args(db))
    try:
        assert blind.count() == 0, (
            "the premise: a store whose claims are all under another tenant "
            "reads as empty, and that is indistinguishable from empty")
    finally:
        blind.close()

    mem = hosted._open_store(_args(db, tenant="workstation"))
    try:
        assert mem.count() == 1
        assert hosted.store_fingerprint(mem)["tenant"] == "workstation"
        found = [r.claim.id for r in
                 mem.search("which suite needs -j1?", k=4, min_score=0.0)]
        assert ids["larkspur"] in found, (
            "opened at the right tenant and still not readable is a different "
            "defect from opening at the wrong one")
    finally:
        mem.close()


def test_open_store_reads_the_user_scope_the_claims_are_under(tmp_path):
    """`--user` is the second half, and is a real narrowing on both routes."""
    db = tmp_path / "user.db"
    _store_on_disk(db, HashingEmbedder(dim=64), tenant="workstation",
                   user="inder")
    mem = hosted._open_store(_args(db, tenant="workstation", user="inder"))
    try:
        assert mem.count() == 1
        assert hosted.store_fingerprint(mem)["user"] == "inder"
    finally:
        mem.close()


def test_the_hosted_route_refuses_a_tenant_flag_rather_than_ignoring_it(tmp_path):
    """A flag that changes only the fingerprint is worse than no flag.

    `RemoteMemvara` accepts `tenant=` and never sends it — the facade resolves
    the tenant from the bearer token — so forwarding `--tenant` there would
    record a scope the run did not read at. The refusal happens before any
    client is constructed, so this test performs no network call and needs no
    credential.
    """
    with pytest.raises(SystemExit) as exc:
        hosted._open_store(_args("", tenant="workstation"))
    message = str(exc.value)
    assert "--db" in message, "the refusal must say what would make it work"
    assert "workstation" in message, "and name the value it is refusing"


def test_main_warns_on_stderr_when_the_scope_sees_nothing_the_file_holds(
        tmp_path, capsys):
    """The shipped invocation, end to end: silence becomes an account of itself.

    On stdout there must still be nothing — `--draft` writes JSONL there and a
    person redirects it into a probe file, so a warning printed to stdout would
    corrupt the very file it is trying to help them build. Both numbers and the
    scope are asserted positively: "some warning fired" would pass on a line
    that told the reader nothing they could act on.
    """
    db = tmp_path / "scoped.db"
    _store_on_disk(db, HashingEmbedder(dim=64), tenant="workstation")

    assert hosted.main(["--db", str(db), "--draft", "2"]) == 0, (
        "this tool has no pass/fail exit code and must not grow one here")
    out = capsys.readouterr()
    assert out.out == "", "stdout carries drafted probes and nothing else"
    assert "WARNING" in out.err
    assert "0 of this store's 1 claims" in out.err, (
        "both numbers, so the reader can tell a mis-scoped store from an empty "
        "one without opening the file themselves")
    assert "default/*/*/*" in out.err, "and the scope that could not see them"
    assert "--tenant" in out.err, "and what to do about it"


def test_the_scope_warning_is_silent_once_the_scope_is_right(tmp_path, capsys):
    """A warning that fires on every run is the same as no warning."""
    db = tmp_path / "scoped.db"
    _store_on_disk(db, HashingEmbedder(dim=64), tenant="workstation")

    assert hosted.main(["--db", str(db), "--tenant", "workstation",
                        "--draft", "1"]) == 0
    out = capsys.readouterr()
    assert out.out.strip(), "the drafted rows are the positive half of this"
    assert "WARNING" not in out.err


def test_the_scope_warning_is_silent_on_a_file_that_is_genuinely_empty(
        tmp_path, capsys, monkeypatch):
    """Nothing visible and nothing stored is not a scope problem.

    Warning here would be the mirror of the defect: telling someone their scope
    is wrong when their store is simply empty sends them looking for a tenant
    that does not exist.

    A store with no vectors binds no recorded embedder, so `_open_store` hands
    `Memvara` `embedder=None` and the library default owns it. Both bindings are
    pinned for the reason
    `test_open_store_ignores_a_fingerprint_with_no_vectors_behind_it` states:
    `conftest.py` redirects `HOME` away from the model cache, so an unpinned
    `default_embedder()` reaches the network and a sabotage run of this test
    would hang rather than go red.
    """
    for binding in ("memvara.core.default_embedder",
                    "memvara.embed.default_embedder"):
        monkeypatch.setattr(binding, lambda dim=512: HashingEmbedder(dim=512))
    db = tmp_path / "empty.db"
    with Memvara(str(db), embedder=HashingEmbedder(dim=64), llm=NullLLM()):
        pass

    assert hosted.main(["--db", str(db), "--draft", "1"]) == 0
    assert "WARNING" not in capsys.readouterr().err


def test_the_scope_warning_fires_on_a_scoring_run_too(tmp_path, capsys):
    """Not only on `--draft`. A table reading `0 claims` is the same ambiguity."""
    db = tmp_path / "scoped.db"
    _store_on_disk(db, HashingEmbedder(dim=64), tenant="workstation")
    probes = tmp_path / "probes.jsonl"
    probes.write_text(json.dumps(
        {"id": "p1", "class": "hit", "query": "which suite needs -j1?",
         "gold": ["cl_whatever"]}) + "\n")

    assert hosted.main(["--db", str(db), "--probes", str(probes)]) == 0
    out = capsys.readouterr()
    assert "0 claims" in out.out, "the ambiguous table is still printed"
    assert "0 of this store's 1 claims" in out.err, "and now accounted for"


def test_the_hosted_route_is_never_asked_a_whole_store_question(capsys):
    """`stats(None)` discloses every tenant's cardinality; hosted is shared.

    `SQLiteStore.stats`' own docstring is the referent: "unfiltered counts
    disclose how much data other tenants hold", and only the unfiltered call
    reports the whole matrix. So the check is gated on `--db`. This asserts the
    gate by making the unfiltered call fail loudly if it is ever reached on a
    store this tool did not open from a local file.
    """
    class Leaky:
        class store:
            @staticmethod
            def stats(tenant=None):
                raise AssertionError(
                    "bench/hosted.py asked a hosted store for whole-store "
                    "counts — that is another tenant's cardinality")

        default_scope = None

        def count(self):
            return 0

        def get_all(self):
            return []

    assert hosted.main(["--draft", "1"], mem=Leaky()) == 0
    assert "WARNING" not in capsys.readouterr().err
