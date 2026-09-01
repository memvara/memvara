# bench/hosted.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A probe-suite harness that measures core retrieval (`search()`/`recall()`) against a real store — hit@k, gold-rank, false-injection rate, self-retrieval@1 — with private probe files and two authoring helpers.

**Architecture:** One script, `bench/hosted.py`, following the `bench/` conventions: a flat module of importable functions, argparse `main`, imported by tests via `sys.path.insert` of the `bench/` directory (the `tests/test_bench_eval.py` pattern). Scoring is pure functions over plain dicts so tests never touch the network; the store handle is built once in `main` and passed down.

**Tech Stack:** Python stdlib + the memvara library already in the repo. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-01-hosted-retrieval-bench-design.md` — read it first; every decision below argues from it.

> **Superseded in two places, 2026-09-01.** This plan is kept as the record of what was
> executed; the whole-branch review then found two premises in it that were false against
> the tree, and the shipped code follows the corrected spec, not this document.
>
> 1. **"`recall()` has no floor."** It does. `plugin/hooks/recall.py` defines
>    `MIN_SCORE = 0.29` and passes it at both call sites, and that shipped before this
>    branch was cut. Every "false-injection is 100% by construction" line below — including
>    the test comment quoted around the Task 3 verification step — is stale. The harness
>    takes `--min-score`, defaulting to the hook's own constant.
> 2. **"`RemoteMemvara` takes the same two calls."** It does not take `with_ids`, and
>    `recall(..., with_ids=True)` raises `TypeError` against a hosted store. The harness
>    now asks the signature and derives the injected set from `search()` on that route.
>
> Both were claims checked against a log and against an assumption rather than against
> their referent, which is the failure mode this repository's `CLAUDE.md` names first.

## Global Constraints

- Measurement only: no retrieval behaviour may change anywhere in `memvara/`.
- No LLM anywhere; scoring is claim-id matching.
- No new dependencies; stdlib + memvara only.
- `k` defaults to 4 (the hook's `K`), `--k` overrides.
- Probe files are private: nothing under `bench/` or `tests/` may contain real store content; fixtures plant their own claims.
- `bench/` is outside the coverage gate, so tests in `tests/test_bench_hosted.py` are the only guard — every scoring guard is sabotaged (break the watched thing, watch it fail) before it is believed. The sabotage steps are in the tasks; do not skip them.
- No AI attribution in any commit message (user's global CLAUDE.md; overrides harness defaults).
- Work happens on branch `bench/hosted-store-measurement` in the worktree `/Applications/workstation/memvara/.claude/worktrees/bench-hosted`.

**Interfaces this plan consumes from the library (verified against the tree):**

- `Memvara.search(query, *, k=10, min_score=0.0, ...) -> list[Result]`; `Result.score` is normalized [0, 1]; `Result.claim.id` is the claim id (`cl`-prefixed).
- `Memvara.recall(query, *, k=8, ..., with_ids=False)` — with `with_ids=True` returns a `RecallResult` whose `.claim_ids` is the rendered claim ids in order and whose text is `.text` (check the dataclass in `memvara/types.py:980` when binding; if the text attribute is named differently, use what the dataclass declares).
- `Memvara.count() -> int`, `Memvara.get_all() -> list[Claim]`.
- `RemoteMemvara(api_key=None, base_url=None, ...)` resolves credentials itself via `memvara/remote/creds.resolve` when both are `None`; it has the same `search`/`recall`/`count` surface.
- In-memory fixture: `Memvara(":memory:", user="probe", embedder=HashingEmbedder(dim=64), llm=NullLLM())`, claims planted with `mem.remember(subject, predicate, object)`, ids read back from `mem.get_all()`.

---

### Task 1: Scoring core

**Files:**
- Create: `bench/hosted.py`
- Create: `tests/test_bench_hosted.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `score_probe(probe: dict, results: list[tuple[str, float]], injected_ids: list[str]) -> dict` and `aggregate(rows: list[dict]) -> dict`. `results` is `(claim_id, score)` in rank order from `search()`; `injected_ids` is what `recall(with_ids=True)` rendered. The returned row dict has keys `probe_id`, `cls`, `hit` (bool | None), `gold_rank` (int | None, 1-based), `false_injection` (bool | None), `top_score` (float | None). `aggregate` returns per-class dicts: `{"hit": {"n": int, "hit_at_k": float, "mean_gold_rank": float | None}, "abstain": {"n": int, "false_injection_rate": float, "headroom": list[float]}, ...}` with `verbatim` and `ambiguous` shaped like `hit`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bench_hosted.py`:

```python
"""bench/hosted.py — scoring a probe suite against a store.

`bench/` is outside the coverage gate, so this file is the only guard on the
harness. Scoring functions are pure and tested on plain data; the runner is
tested against an in-memory store with planted claims. Nothing here touches
the network.
"""

from __future__ import annotations

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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Applications/workstation/memvara/.claude/worktrees/bench-hosted && python3 -m pytest tests/test_bench_hosted.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'hosted'`.

- [ ] **Step 3: Implement the scoring core**

Create `bench/hosted.py`:

```python
"""Probe-suite measurement of retrieval against a real store.

    PYTHONPATH=. python3 bench/hosted.py --probes ~/.memvara/probes.jsonl
    PYTHONPATH=. python3 bench/hosted.py --draft 10
    PYTHONPATH=. python3 bench/hosted.py --seed-from-recalled ~/.memvara/.hooks/recalled --dump pairs.jsonl

Design: docs/superpowers/specs/2026-09-01-hosted-retrieval-bench-design.md.
Measurement only — this tool changes no retrieval behaviour. Probe files are
private to a store and never live in this repository; see the spec for why.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

#: The recall hook's own K, mirrored so the default measurement is of what the
#: hook actually injects. See plugin/hooks/recall.py.
DEFAULT_K = 4

CLASSES = ("hit", "abstain", "verbatim", "ambiguous")


def score_probe(probe: dict, results: "Sequence[tuple[str, float]]",
                injected_ids: "Sequence[str]") -> dict:
    """One probe against one retrieval, as a flat row.

    `results` is (claim_id, score) in rank order from search(); `injected_ids`
    is what recall(with_ids=True) actually rendered — the injection surface.
    The two differ on purpose: rank and headroom come from the scored surface,
    the abstain verdict from the surface the hook injects.
    """
    cls = probe["class"]
    gold = set(probe.get("gold", ()))
    top_score = results[0][1] if results else None
    row = {"probe_id": probe["id"], "cls": cls, "hit": None, "gold_rank": None,
           "false_injection": None, "top_score": top_score}
    if cls == "abstain":
        row["false_injection"] = bool(injected_ids)
        return row
    rank = next((i for i, (cid, _) in enumerate(results, start=1)
                 if cid in gold), None)
    row["gold_rank"] = rank
    if cls == "verbatim":
        row["hit"] = rank == 1
    else:
        row["hit"] = rank is not None
    return row


def aggregate(rows: "Sequence[dict]") -> dict:
    """Per-class summary. `ambiguous` is never folded into `hit`: its gold is
    a judgment, not a fact, and the two must stay tellable apart (spec)."""
    out: dict[str, dict] = {}
    for cls in CLASSES:
        mine = [r for r in rows if r["cls"] == cls]
        if not mine:
            continue
        if cls == "abstain":
            failures = [r for r in mine if r["false_injection"]]
            out[cls] = {
                "n": len(mine),
                "false_injection_rate": len(failures) / len(mine),
                "headroom": [r["top_score"] for r in failures
                             if r["top_score"] is not None],
            }
        else:
            hits = [r for r in mine if r["hit"]]
            ranks = [r["gold_rank"] for r in hits if r["gold_rank"]]
            out[cls] = {
                "n": len(mine),
                "hit_at_k": len(hits) / len(mine),
                "mean_gold_rank": (sum(ranks) / len(ranks)) if ranks else None,
            }
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bench_hosted.py -q`
Expected: 6 passed.

- [ ] **Step 5: Sabotage check, then commit**

Temporarily change `rank == 1` to `rank is not None` in the `verbatim` branch; run the suite; `test_a_verbatim_probe_requires_rank_one_exactly` must FAIL. Revert. Then:

```bash
git add bench/hosted.py tests/test_bench_hosted.py
git commit -m "bench/hosted: scoring core for the probe suite"
```

---

### Task 2: Probe loading, validation, and the draft refusal

**Files:**
- Modify: `bench/hosted.py`
- Modify: `tests/test_bench_hosted.py`

**Interfaces:**
- Consumes: `CLASSES` from Task 1.
- Produces: `load_probes(path: Path) -> list[dict]` — raises `SystemExit` with a message naming the line number and the problem on: unknown class, non-list `gold`, `abstain` with non-empty `gold`, missing `id`/`query`, duplicate `id`, or a row still carrying `"draft": true`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bench_hosted.py`:

```python
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
    ({"class": "hit", "query": "q", "gold": ["cl_a"]}, "id"),
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
    with pytest.raises(SystemExit, match="duplicate"):
        hosted.load_probes(path)


def test_load_probes_refuses_an_unedited_draft_row(tmp_path):
    # A drafted probe quotes the claim's own text, which measures lexical echo.
    # The runner refusing the mark is what makes --draft safe to ship (spec).
    path = _write_probes(tmp_path, [
        {"id": "p1", "class": "hit", "query": "q", "gold": ["cl_a"], "draft": True},
    ])
    with pytest.raises(SystemExit, match="draft"):
        hosted.load_probes(path)
```

Add `import json` to the test file's imports if not present.

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bench_hosted.py -q`
Expected: new tests FAIL with `AttributeError: module 'hosted' has no attribute 'load_probes'`.

- [ ] **Step 3: Implement**

Add to `bench/hosted.py`:

```python
def load_probes(path: Path) -> list[dict]:
    """Read and validate a probe file, refusing rather than skipping.

    Every refusal names the line, because the file is hand-edited and 'invalid
    probe file' with no address costs a search. A row still marked draft:true
    is refused outright: a drafted query is the claim's own text, and scoring
    it would measure lexical echo — the bias this tool exists to escape.
    """
    probes: list[dict] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except ValueError as exc:
            raise SystemExit(f"{path}: line {lineno}: not JSON: {exc}")
        for field in ("id", "query", "class"):
            if not isinstance(row.get(field), str) or not row.get(field):
                raise SystemExit(f"{path}: line {lineno}: missing or empty {field!r}")
        if row["class"] not in CLASSES:
            raise SystemExit(
                f"{path}: line {lineno}: unknown class {row['class']!r}; "
                f"expected one of {', '.join(CLASSES)}")
        if not isinstance(row.get("gold"), list):
            raise SystemExit(f"{path}: line {lineno}: gold must be a list of claim ids")
        if row["class"] == "abstain" and row["gold"]:
            raise SystemExit(
                f"{path}: line {lineno}: an abstain probe's gold must be empty — "
                "a non-empty gold is a hit probe")
        if row["class"] != "abstain" and not row["gold"]:
            raise SystemExit(
                f"{path}: line {lineno}: a {row['class']} probe needs at least "
                "one gold claim id")
        if row.get("draft"):
            raise SystemExit(
                f"{path}: line {lineno}: probe {row['id']!r} is still marked "
                "draft: true. Edit the query into how you would actually ask, "
                "then remove the mark — a drafted query measures lexical echo.")
        if row["id"] in seen:
            raise SystemExit(f"{path}: line {lineno}: duplicate probe id {row['id']!r}")
        seen.add(row["id"])
        probes.append(row)
    if not probes:
        raise SystemExit(f"{path}: no probes. See the schema in docs/BENCHMARKS.md.")
    return probes
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bench_hosted.py -q`
Expected: all pass.

- [ ] **Step 5: Sabotage check, then commit**

Temporarily delete the `if row.get("draft"):` block; the draft test must FAIL. Revert. Then:

```bash
git add bench/hosted.py tests/test_bench_hosted.py
git commit -m "bench/hosted: probe file loading that refuses rather than skips"
```

---

### Task 3: The runner, against a planted store

**Files:**
- Modify: `bench/hosted.py`
- Modify: `tests/test_bench_hosted.py`

**Interfaces:**
- Consumes: `score_probe` (Task 1), probe dicts (Task 2).
- Produces: `run_probes(mem, probes: list[dict], *, k: int) -> list[dict]` — each row is `score_probe`'s row plus `"results": [[claim_id, score], ...]`, `"injected": [claim_id, ...]`, and `"latency_ms": float`. `mem` is anything with the `search`/`recall` surface (`Memvara` or `RemoteMemvara`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bench_hosted.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bench_hosted.py -q`
Expected: new tests FAIL (`no attribute 'run_probes'`).

- [ ] **Step 3: Implement**

Add to `bench/hosted.py`:

```python
import time


def run_probes(mem: Any, probes: "Sequence[dict]", *, k: int) -> list[dict]:
    """Every probe through both read surfaces, scored.

    search() supplies ranks and scores; recall(with_ids=True) supplies what
    would actually be injected. Both run per probe because they answer
    different halves of the question (spec: a core ranking defect and a
    surface gating defect must show up as different numbers).
    """
    rows: list[dict] = []
    for probe in probes:
        t0 = time.perf_counter()
        results = [(r.claim.id, r.score)
                   for r in mem.search(probe["query"], k=k)]
        recalled = mem.recall(probe["query"], k=k, with_ids=True)
        elapsed = (time.perf_counter() - t0) * 1000.0
        injected = list(getattr(recalled, "claim_ids", []) or [])
        row = score_probe(probe, results, injected)
        row["results"] = [[cid, round(score, 4)] for cid, score in results]
        row["injected"] = injected
        row["latency_ms"] = round(elapsed, 2)
        rows.append(row)
    return rows
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bench_hosted.py -q`
Expected: all pass. If `test_run_probes_records_injection_for_an_abstain_probe` fails because nothing was injected, STOP and check `recall()`'s default `min_score` on this tree before touching the test — the spec's baseline claim rests on it.

- [ ] **Step 5: Commit**

```bash
git add bench/hosted.py tests/test_bench_hosted.py
git commit -m "bench/hosted: run probes through both read surfaces"
```

---

### Task 4: Fingerprint, output, table, CLI

**Files:**
- Modify: `bench/hosted.py`
- Modify: `tests/test_bench_hosted.py`

**Interfaces:**
- Consumes: `load_probes`, `run_probes`, `aggregate`.
- Produces: `store_fingerprint(mem) -> dict` (`{"claims": int, "surface": "local"|"hosted", "when": iso-str, "embedder": str | None}`); `render_table(agg: dict, fingerprint: dict) -> str`; `compare_runs(a: Path, b: Path) -> str`; `main(argv) -> int`. CLI flags: `--probes PATH` (default `~/.memvara/probes.jsonl`), `--k INT` (default `DEFAULT_K`), `--db PATH` (local store; otherwise hosted via credentials), `--out PATH` (per-probe JSONL, one row per line, first line the fingerprint), `--compare A B` (two `--out` files: per-class deltas, prefixed by a drift warning when the fingerprints differ — the spec's "comparing two result files with different fingerprints prints a warning naming what moved").

`main`'s body order, which Tasks 5 and 6 slot branches into — keep it exactly:
1. parse args; 2. the `--seed-from-recalled` branch (needs no store); 3. open the store (or take the injected `mem`); 4. the `--draft` branch; 5. `--compare` branch (needs no store either — place before 3 with the seeding branch); 6. `load_probes`; 7. run, aggregate, write `--out`, print table; 8. `finally:` close the store iff this call opened it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bench_hosted.py`:

```python
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
        "abstain": {"n": 2, "false_injection_rate": 1.0, "headroom": [0.45, 0.31]},
    }
    fp = {"claims": 10, "surface": "local", "when": "2026-09-01T00:00:00",
          "embedder": "hashing:64:3-5"}
    table = hosted.render_table(agg, fp)
    # Positive statements, so a deleted line fails as loudly as a wrong one.
    assert "hit@k" in table and "50.0%" in table
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bench_hosted.py -q`
Expected: new tests FAIL.

- [ ] **Step 3: Implement**

Add to `bench/hosted.py` (note `main(argv, mem=None)` — the injectable `mem` is what keeps the tests off the network; `_open_store` is only reached from a real command line):

```python
from datetime import datetime, timezone


def store_fingerprint(mem: Any) -> dict:
    """What a drift warning needs: enough to say the store moved, no content."""
    embedder = getattr(mem, "embedder", None)
    name = getattr(embedder, "name", None) if embedder is not None else None
    return {
        "claims": mem.count(),
        "surface": "local" if embedder is not None else "hosted",
        "embedder": name,
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def render_table(agg: dict, fingerprint: dict) -> str:
    lines = [
        f"  store: {fingerprint['claims']} claims"
        f"  surface={fingerprint['surface']}"
        + (f"  embedder={fingerprint['embedder']}" if fingerprint["embedder"] else ""),
        "",
        "  class       n   metric",
        "  ---------  --   ------",
    ]
    for cls in ("hit", "ambiguous", "verbatim"):
        if cls in agg:
            a = agg[cls]
            rank = (f"  mean gold-rank {a['mean_gold_rank']:.1f}"
                    if a["mean_gold_rank"] is not None else "")
            lines.append(f"  {cls:<9} {a['n']:>3}   hit@k {a['hit_at_k']:.1%}{rank}")
    if "abstain" in agg:
        a = agg["abstain"]
        lines.append(f"  {'abstain':<9} {a['n']:>3}   false-injection "
                     f"{a['false_injection_rate']:.1%}")
        if a["headroom"]:
            tops = ", ".join(f"{s:.2f}" for s in sorted(a["headroom"]))
            lines.append(f"                  scores on failures: {tops}  "
                         "(a floor above a value silences that failure)")
    return "\n".join(lines)


def compare_runs(a: Path, b: Path) -> str:
    """Two --out files side by side, led by a drift warning when due.

    The warning comes first because it changes how the deltas read: a hit@k
    that moved on a store that also moved is not a before/after, and saying so
    below the numbers is saying it too late."""
    def load(path: Path) -> tuple[dict, dict]:
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        return lines[0], aggregate(lines[1:])
    fp_a, agg_a = load(a)
    fp_b, agg_b = load(b)
    out: list[str] = []
    moved = [f"{key} {fp_a.get(key)} -> {fp_b.get(key)}"
             for key in ("claims", "embedder", "surface")
             if fp_a.get(key) != fp_b.get(key)]
    if moved:
        out.append("  WARNING: the store moved between these runs — "
                   + "; ".join(moved))
        out.append("  deltas below compare two different stores, not one store twice")
    for cls in CLASSES:
        if cls in agg_a and cls in agg_b:
            if cls == "abstain":
                out.append(f"  {cls}: false-injection "
                           f"{agg_a[cls]['false_injection_rate']:.1%} -> "
                           f"{agg_b[cls]['false_injection_rate']:.1%}")
            else:
                out.append(f"  {cls}: hit@k {agg_a[cls]['hit_at_k']:.1%} -> "
                           f"{agg_b[cls]['hit_at_k']:.1%}")
    return "\n".join(out)


def _open_store(args: argparse.Namespace) -> Any:
    if args.db:
        from memvara import Memvara
        return Memvara(args.db)
    from memvara.remote.api import RemoteMemvara
    return RemoteMemvara()


def main(argv: "Sequence[str] | None" = None, *, mem: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probes", default=str(Path.home() / ".memvara" / "probes.jsonl"))
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="results per probe; 4 is the recall hook's own K")
    parser.add_argument("--db", default="", help="local store path; omit for hosted")
    parser.add_argument("--out", default="", help="write per-probe JSONL here")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None)
    args = parser.parse_args(argv)

    if args.compare:
        print(compare_runs(Path(args.compare[0]), Path(args.compare[1])))
        return 0

    probes = load_probes(Path(args.probes))
    close = False
    if mem is None:
        mem = _open_store(args)
        close = True
    try:
        fingerprint = store_fingerprint(mem)
        rows = run_probes(mem, probes, k=args.k)
    finally:
        if close:
            mem.close()
    agg = aggregate(rows)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(json.dumps(fingerprint) + "\n")
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    print(render_table(agg, fingerprint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bench_hosted.py -q`
Expected: all pass.

- [ ] **Step 5: Sabotage check, then commit**

Temporarily delete the headroom line from `render_table`; `test_render_table_states_every_metric_present` must FAIL on the `0.45` assertion. Then temporarily make `compare_runs` skip the `moved` block; `test_compare_runs_warns_when_the_store_moved` must FAIL. Revert both. Then:

```bash
git add bench/hosted.py tests/test_bench_hosted.py
git commit -m "bench/hosted: fingerprint, table, JSONL output, CLI"
```

---

### Task 5: --draft

**Files:**
- Modify: `bench/hosted.py`
- Modify: `tests/test_bench_hosted.py`

**Interfaces:**
- Consumes: `_open_store`, `main`'s parser.
- Produces: `draft_probes(mem, n: int) -> list[dict]` — samples up to `n` live claims (`mem.get_all()`, first `n` after sorting by id for determinism) and emits, per claim, one `hit` and one `verbatim` skeleton, every row carrying `"draft": True`. `--draft N` on the CLI prints them as JSONL to stdout and exits 0 without running probes.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bench_hosted.py -q` — new tests FAIL.

- [ ] **Step 3: Implement**

Add to `bench/hosted.py`; in `main`, add `parser.add_argument("--draft", type=int, default=0, metavar="N")` and, immediately after `mem` is resolved (before `load_probes` — a draft run must not require a probe file to exist, so move the `load_probes` call below this branch):

```python
def draft_probes(mem: Any, n: int) -> list[dict]:
    """Skeleton probes from live claims, every one refusing to run as-is.

    The query IS the claim's text, which is exactly what a probe must not be —
    so each row carries draft: true and load_probes refuses it until a person
    rewrites the query into how they would actually ask.
    """
    claims = sorted(mem.get_all(), key=lambda c: c.id)[:n]
    rows: list[dict] = []
    for i, claim in enumerate(claims, start=1):
        text = claim.object
        rows.append({"id": f"draft-hit-{i}", "class": "hit",
                     "query": text, "gold": [claim.id], "draft": True})
        rows.append({"id": f"draft-verbatim-{i}", "class": "verbatim",
                     "query": text, "gold": [claim.id], "draft": True})
    return rows
```

And in `main`, the branch:

```python
    if args.draft:
        for row in draft_probes(mem, args.draft):
            print(json.dumps(row))
        if close:
            mem.close()
        return 0
```

(The body order in Task 4's Interfaces block is the contract: store opening at step 3, the draft branch at step 4, `load_probes` at step 6. Every path through `main` closes `mem` exactly once, and only when this call opened it.)

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bench_hosted.py -q` — all pass.

- [ ] **Step 5: Commit**

```bash
git add bench/hosted.py tests/test_bench_hosted.py
git commit -m "bench/hosted: --draft emits skeletons the runner refuses"
```

---

### Task 6: --seed-from-recalled (dump and answers phases)

**Files:**
- Modify: `bench/hosted.py`
- Modify: `tests/test_bench_hosted.py`

**Interfaces:**
- Consumes: `main`'s parser; the recalled-event format `{"seen": [...], "query": str, ...}` (one JSON file per event).
- Deliberate deviation from the spec's letter: the spec names "the `bench/evalkit.FileReader` dump/answers shape", whose rows carry `system_prompt`/`prompt`. Here there is no prompt — only a query — so the dump rows are `{"id", "query"}`. The *pattern* (two-phase blinded round trip, deterministic shuffle, key held back) is what the spec means; do not import or mimic `FileReader` itself, and do not add empty prompt fields to match its schema.
- Produces: `seed_dump(recalled_dir: Path, dump_path: Path, *, sample: int, seed: int) -> int` (events written); `seed_answers(dump_path: Path, answers_path: Path, judged: str) -> list[dict]` (ambiguous probes). Dump rows are `{"id": <digest>, "query": <text>}`, shuffled with `random.Random(seed)`; answers rows are `{"id": ..., "gold": [claim ids]}` (empty list = judged irrelevant → becomes an `abstain` probe instead; `"skip"` key true = dropped). CLI: `--seed-from-recalled DIR --dump PATH [--sample N] [--seed INT]`, then later `--seed-from-recalled DIR --answers PATH --judged YYYY-MM-DD --dump PATH`, which prints probe JSONL to stdout.

- [ ] **Step 1: Write the failing tests**

```python
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
    assert len(probes) == 2, "a skipped row produces no probe"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bench_hosted.py -q` — new tests FAIL.

- [ ] **Step 3: Implement**

Add to `bench/hosted.py`:

```python
import hashlib
import random

#: Internal traffic that reaches the recall hook but was never a user
#: question. Judged by prefix, because these are this repository's own
#: prompts and their openings are stable.
_INTERNAL_PREFIXES = ("Extract durable facts",)


def seed_dump(recalled_dir: Path, dump_path: Path, *, sample: int,
              seed: int = 20260901) -> int:
    """Phase one of closing the judgment loop: real queries, blinded, dumped.

    Blinding here means order (shuffled by seed) and absence of results — the
    judge sees only the query text and answers from the store, not from what
    the hook happened to return that day.
    """
    queries: dict[str, str] = {}
    for f in sorted(recalled_dir.glob("*.json")):
        try:
            query = json.loads(f.read_text()).get("query", "")
        except ValueError:
            continue
        query = " ".join(query.split())
        if not query or any(query.startswith(p) for p in _INTERNAL_PREFIXES):
            continue
        digest = hashlib.blake2b(query.encode(), digest_size=8).hexdigest()
        queries[digest] = query
    rows = [{"id": d, "query": q} for d, q in sorted(queries.items())]
    random.Random(seed).shuffle(rows)
    rows = rows[:sample]
    with open(dump_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return len(rows)


def seed_answers(dump_path: Path, answers_path: Path, judged: str) -> list[dict]:
    """Phase two: judgments back, probes out.

    gold=[] is a judgment too — 'the store has nothing for this' — and becomes
    an abstain probe rather than being dropped; only skip:true drops a row.
    """
    dumped = {json.loads(l)["id"]: json.loads(l)["query"]
              for l in dump_path.read_text().splitlines() if l.strip()}
    probes: list[dict] = []
    for raw in answers_path.read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get("skip") or row["id"] not in dumped:
            continue
        gold = row.get("gold", [])
        cls = "abstain" if not gold else "ambiguous"
        probe = {"id": f"seeded-{row['id']}", "class": cls,
                 "query": dumped[row["id"]], "gold": gold}
        if cls == "ambiguous":
            probe["judged"] = judged
        probes.append(probe)
    return probes
```

CLI wiring in `main` (before store opening — seeding needs no store):

```python
    parser.add_argument("--seed-from-recalled", default="", metavar="DIR")
    parser.add_argument("--dump", default="", metavar="PATH")
    parser.add_argument("--answers", default="", metavar="PATH")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--judged", default="", metavar="YYYY-MM-DD")
    ...
    if args.seed_from_recalled:
        if args.answers:
            if not args.judged:
                raise SystemExit("--answers needs --judged YYYY-MM-DD: a judgment "
                                 "ages as the store changes, and the date is how "
                                 "a future reader knows how stale it is")
            for probe in seed_answers(Path(args.dump), Path(args.answers),
                                      judged=args.judged):
                print(json.dumps(probe))
            return 0
        if not args.dump:
            raise SystemExit("--seed-from-recalled needs --dump PATH on the "
                             "first pass, then --answers PATH on the second")
        n = seed_dump(Path(args.seed_from_recalled), Path(args.dump),
                      sample=args.sample, seed=args.seed)
        print(f"wrote {n} blinded queries to {args.dump}; judge them into "
              f"{{\"id\", \"gold\": [claim ids]}} rows (gold [] = nothing "
              f"relevant; \"skip\": true = drop), then re-run with "
              f"--answers PATH --judged YYYY-MM-DD")
        return 0
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bench_hosted.py -q` — all pass.

- [ ] **Step 5: Sabotage check, then commit**

Temporarily empty `_INTERNAL_PREFIXES`; the filter test must FAIL. Revert. Then:

```bash
git add bench/hosted.py tests/test_bench_hosted.py
git commit -m "bench/hosted: seed ambiguous probes from real recall traffic"
```

---

### Task 7: Documentation, changelog, full gate

**Files:**
- Modify: `docs/BENCHMARKS.md` (new section at the end, before any closing nav line — check the file's tail for the `Previous:`/`Next:` navigation convention and keep it last)
- Modify: `CHANGELOG.md` (Unreleased → Added)
- Test: the whole suite

**Interfaces:** none new — this task makes the shipped state legible.

- [ ] **Step 1: Write the BENCHMARKS.md section**

Append (adjusting only to sit above the nav footer):

```markdown
### Measuring against your own store: `bench/hosted.py`

Every corpus above was built for its benchmark. None has the shape a real
store develops — the roadmap's census of one production store found ~95% of
claims on predicates outside the declared vocabulary and a join rate of 0.5%.
`bench/hosted.py` measures the read path against the store you actually have:

    PYTHONPATH=. python3 bench/hosted.py --probes ~/.memvara/probes.jsonl

You author the probes once — `hit` (a question whose answer you know is
stored, gold = its claim id), `abstain` (a question the store cannot answer,
gold = nothing), `verbatim` (a claim's own text, which must return that claim
first), `ambiguous` (real prompts from your logs, judged) — and the run
reports hit@k, mean gold-rank, false-injection rate with per-failure score
headroom, and self-retrieval@1. `--draft` and `--seed-from-recalled` help
author; both refuse to produce a probe no person has reviewed.

Probe files are private to a store and never belong in this repository. The
numbers are per-store and are not memvara scores: nothing measured here is
comparable between two stores, let alone publishable against another system.
False-injection starts at 100% by construction — `recall()` has no floor —
and that number existing is the point: it is the baseline an abstention
design would be judged against.
```

- [ ] **Step 2: Write the CHANGELOG entry**

Under `## [Unreleased]` / `### Added` (create the subsection if absent, above existing `### Fixed`):

```markdown
- **`bench/hosted.py` — measure retrieval against your own store.** Every
  published benchmark runs on a corpus built for it; this runs on the store
  you have. Four numbers from an owner-authored probe file: hit@k and mean
  gold-rank, false-injection rate on questions the store cannot answer (100%
  at baseline, by construction — `recall()` has no floor, which is the fact
  the metric exists to track), and self-retrieval@1, which pins the recorded
  defect where a claim's own text returns a different, higher-confidence
  claim. Probes stay private to the store; `--draft` and
  `--seed-from-recalled` help author them and refuse to emit anything a
  person has not reviewed.
```

- [ ] **Step 3: Run the docs guards and the full suite**

Run: `python3 -m pytest tests/test_docs.py tests/test_doc_links.py tests/test_bench_hosted.py -q` first (fast feedback on the docs edits — note `docs/BENCHMARKS.md` prose is scanned by `test_no_other_count_is_stated_anywhere`, so avoid writing any "N tools" phrasing), then the full suite: `python3 -m pytest -q`.
Expected: all pass. **Read the verdict line in full** — no `tail | head` truncation; a skip is not a pass.

- [ ] **Step 4: Commit**

```bash
git add docs/BENCHMARKS.md CHANGELOG.md
git commit -m "bench/hosted: document the probe suite and its baseline claims"
```

- [ ] **Step 5: Smoke the real thing once**

The deliverable is a tool someone runs, so run it (verify the deliverable, not the repository):

```bash
PYTHONPATH=. python3 bench/hosted.py --draft 3 --db /tmp/hosted-smoke.db 2>&1 | head -8
```

Expected: JSONL skeleton probes on stdout (the store is empty, so possibly zero rows — then plant one first with a tiny `python3 -c` using `Memvara("/tmp/hosted-smoke.db").remember(...)`). Then confirm the runner refuses a draft row end-to-end exactly as the tests say. Remove `/tmp/hosted-smoke.db*` afterwards.

---

## After the tasks

Open the PR from `bench/hosted-store-measurement`, then run `/code-review high <PR#>` in the open-review-fix window, fix findings on the branch, and note the review model in a PR comment if it is not the latest Sonnet. Do not merge without the review. The first real measurement run (authoring your actual probes, seeding from your 1,052 recall events) is deliberately after merge — it is use, not development.
