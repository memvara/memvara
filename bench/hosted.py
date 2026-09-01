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
import hashlib
import inspect
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

#: The recall hook's own `K` and `MIN_SCORE`, mirrored so the default measurement
#: is of what the hook actually injects — not of an unfloored read path no shipped
#: surface uses. `plugin/hooks/recall.py` passes `min_score=_min_score()` at both
#: of its call sites, and `MIN_SCORE` is that function's default.
#:
#: Mirrored rather than imported because importing the hook runs
#: `sys.path.insert(0, plugin/hooks)` at its own module scope, which would put
#: generic package names (`core`, `lib`) on the path of every process that merely
#: runs this bench. The copy is therefore checked against its referent instead of
#: trusted: `tests/test_bench_hosted.py::test_bench_defaults_equal_the_hook_constants`
#: imports the hook module and asserts both values match. Change one, that test
#: goes red — it never compares this file against itself.
DEFAULT_K = 4
DEFAULT_MIN_SCORE = 0.29

#: The environment variable the recall hook's own `_min_score()` reads, mirrored
#: for the same reason the constants above are.
ENV_MIN_SCORE = "MEMVARA_RECALL_MIN_SCORE"

CLASSES = ("hit", "abstain", "verbatim", "ambiguous")


def default_min_score() -> float:
    """The floor the recall hook would actually apply on this machine.

    `MIN_SCORE` is the hook's *constant*; `_min_score()` is its effective value,
    and the hook's own comment above `MIN_SCORE` tells a store owner to
    recalibrate and set `MEMVARA_RECALL_MIN_SCORE`. Mirroring only the constant
    would mean that anyone who followed that advice measured 0.29 rather than
    their shipped configuration — the bench would be describing a deployment
    they do not have while claiming to measure what the hook injects.

    So the resolution is mirrored too, clamp included: `0` is a legitimate
    setting (it restores the unfiltered read path), a value above 1.0 filters
    everything, and a value that is not a number leaves the constant standing.
    `tests/test_bench_hosted.py::test_bench_default_floor_resolves_as_the_hook_resolves_it`
    compares this against the hook's own `_min_score()` on every branch —
    unset, set, both clamps, unparseable — so the two resolutions cannot drift
    apart in silence.
    """
    raw = os.environ.get(ENV_MIN_SCORE)
    if raw is None:
        return DEFAULT_MIN_SCORE
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return DEFAULT_MIN_SCORE


def _class_label(cls: str) -> str:
    """The metric name for a class, in the one place both renderers read it.

    verbatim only counts a hit at rank 1 exactly, never at rank k — "hit@k"
    would misdescribe it, and self-retrieval@1 is the design doc's own name for
    it. `render_table` and `compare_runs` both print this label and had it
    spelled out separately; one of the two said "hit@k" for every class,
    including the one the other explicitly says it must not.
    """
    return "self-retrieval@1" if cls == "verbatim" else "hit@k"


def score_probe(probe: dict, results: "Sequence[tuple[str, float]]",
                injected_ids: "Sequence[str]") -> dict:
    """One probe against one retrieval, as a flat row.

    `results` is (claim_id, score) in rank order from search(); `injected_ids`
    is the injection surface — what would actually go into the prompt. Where
    that comes from depends on the engine, and `_injected_ids` is the whole of
    the difference: the local one reads it off `recall(with_ids=True)`, the
    hosted one infers it from `search()` because `RemoteMemvara.recall` returns
    prose that names nothing. This function is told the answer, not how it was
    obtained, and scores the same either way.

    `results` and `injected_ids` differ on purpose: rank and headroom come from
    the scored surface, the abstain verdict from the surface the hook injects.
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
                # None, not `evalkit.mean`'s 0.0: "no hits to average" must
                # render as an absent column, where 0.0 reads as rank zero.
                "mean_gold_rank": (sum(ranks) / len(ranks)) if ranks else None,
            }
    return out


def _names_what_it_rendered(mem: Any) -> bool:
    """Whether this store's `recall()` can name the claims it injected.

    `Memvara.recall` takes `with_ids` and returns a `RecallResult`;
    `RemoteMemvara.recall` takes no such argument and returns a plain `str`
    (`memvara/remote/api.py`). Asked of the signature rather than by catching
    `TypeError`, because a `TypeError` raised *inside* `recall` would answer the
    same way and quietly send the hosted run down the degraded path.
    """
    try:
        return "with_ids" in inspect.signature(mem.recall).parameters
    except (TypeError, ValueError):  # pragma: no cover - not a callable surface
        return False


def _injected_ids(recalled: Any, results: "Sequence[tuple[str, float]]",
                  *, named: bool) -> list[str]:
    """What the recall surface put in the prompt, as claim ids.

    Two routes, and the difference is the reason this is a function.

    `named` — the local engine — reads `claim_ids` off the `RecallResult`, and
    **refuses** an object that does not carry them. There is no empty-list
    fallback on purpose: `claim_ids` missing would score every abstain probe as
    a pass and report a flawless 0% false-injection rate, which is precisely a
    guard a deletion satisfies.

    Otherwise — the hosted engine — `recall()` returns prose and names nothing,
    so the injected set is taken from `search()` at the same `k` and
    `min_score`. That is what `recall()` renders: `Memvara.recall` calls
    `search(query, k=k, min_score=min_score, ...)` and renders those claim rows
    in order. The cost is stated rather than hidden — on the hosted route this
    measures the set `POST /v1/search` returns, not the block `POST /v1/recall`
    actually rendered, so a server-side divergence between the two (rendering,
    a budget this bench never asks for, episode handling) would not show up
    here. `recall()` is still called on that route, so a hosted recall failure
    still fails the run rather than passing unmeasured.
    """
    if not named:
        return [cid for cid, _ in results]
    claim_ids = getattr(recalled, "claim_ids", None)
    if claim_ids is None:
        raise SystemExit(
            f"recall(with_ids=True) returned {type(recalled).__name__} with no "
            "claim_ids: the injection surface cannot be read. Refusing rather "
            "than treating it as an empty injection, which would score every "
            "abstain probe as a pass and report 0% false injection.")
    return list(claim_ids)


def run_probes(mem: Any, probes: "Sequence[dict]", *, k: int,
               min_score: float) -> list[dict]:
    """Every probe through both read surfaces, scored.

    search() supplies ranks and scores; recall() supplies what would actually
    be injected. Both run per probe because they answer different halves of the
    question (spec: a core ranking defect and a surface gating defect must show
    up as different numbers).

    `min_score` goes to both, and defaults at the CLI to whatever the recall
    hook's own `_min_score()` would resolve to here (`default_min_score`).
    Measuring an unfloored read path would measure a configuration no shipped
    surface uses.
    """
    named = _names_what_it_rendered(mem)
    rows: list[dict] = []
    for probe in probes:
        t0 = time.perf_counter()
        results = [(r.claim.id, r.score)
                   for r in mem.search(probe["query"], k=k, min_score=min_score)]
        extra = {"with_ids": True} if named else {}
        recalled = mem.recall(probe["query"], k=k, min_score=min_score, **extra)
        elapsed = (time.perf_counter() - t0) * 1000.0
        injected = _injected_ids(recalled, results, named=named)
        row = score_probe(probe, results, injected)
        row["results"] = [[cid, round(score, 4)] for cid, score in results]
        row["injected"] = injected
        row["latency_ms"] = round(elapsed, 2)
        rows.append(row)
    return rows


def store_fingerprint(mem: Any) -> dict:
    """What a drift warning needs: enough to say the store moved, no content.

    The embedder identity comes from the library's own `embedder_name`, not
    from `getattr(embedder, "name", None)`: that helper unwraps a wrapper's
    `.inner` and falls back to the class name, where the hand-rolled lookup
    reported `embedder: None` for both — a fingerprint that cannot report the
    embedder cannot warn that it changed, which is the drift this exists for.

    `tenant` and `user` are here for the same reason `claims` is. Two runs at
    different scopes read two different populations out of one file, so their
    deltas are not a before/after any more than two different stores' are, and
    `compare_runs` warns on all five keys alike. A result file written before
    these keys existed records neither, so comparing one against a new file
    warns `tenant None -> ...`: that is a true statement — the earlier run did
    not record the scope it read — and the conservative direction for a warning
    whose whole job is to say "these two may not be comparable".

    On the hosted route the scope is the client's own view of itself. The
    facade resolves the tenant from the bearer token and `RemoteMemvara` never
    sends the parameter (`memvara/remote/api.py`), which is why `--tenant` is
    refused there rather than recorded.
    """
    from memvara.embed.fingerprint import embedder_name

    embedder = getattr(mem, "embedder", None)
    name = embedder_name(embedder) if embedder is not None else None
    scope = getattr(mem, "default_scope", None)
    return {
        "claims": mem.count(),
        "surface": "local" if embedder is not None else "hosted",
        "embedder": name,
        "tenant": getattr(scope, "tenant", None),
        "user": getattr(scope, "user", None),
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def scope_blindspot(mem: Any) -> "str | None":
    """The warning for a store whose claims are all outside the scope being read.

    The shipped defect this exists for: `--db PATH` opened at `tenant="default"`
    against a real store whose claims all live under a named tenant. `--draft`
    printed nothing, the run exited 0, and the fingerprint said `claims: 0` — a
    wrongly-scoped store and a genuinely empty one were the same output.
    "Skipped" and "never ran" must not look alike.

    Two numbers make them tellable apart, and the second is the only one that is
    not already on the page. `mem.count()` is what the configured scope can see.
    `SQLiteStore.stats(None)` — the *unfiltered* call — reports the whole file;
    its docstring says so, and says why only the unfiltered call may:
    "unfiltered counts disclose how much data other tenants hold". So this is
    asked of a local store and never of a hosted one. `RemoteMemvara` talks to a
    shared multi-tenant server, and asking it for whole-store counts is exactly
    the disclosure that docstring guards against — hence the `--db`-only gate at
    the call site in `main`. Where the question cannot be asked cheaply and
    safely, it is not asked: silence there is correct, and a warning built on a
    number this tool has no business holding would be worse than none.

    **`live_claims`, not `claims`, and the difference is a false positive.** The
    two numbers have to carry the same state semantics or they are not
    comparable. `count()` resolves its states through `resolve_states(None,
    None)`, which is `("live",)`; `stats`' `claims` is `SELECT COUNT(*) FROM
    claims` with no state filter at all, so it counts retired and ended rows
    too. Against `claims`, a store whose facts are all at the *right* tenant but
    retired reads as `count() == 0` beside a non-zero total, and the tool tells
    someone their scope is wrong and to re-run with `--tenant` — advice that
    cannot help, because nothing is misscoped. `live_claims` compiles from the
    same `base.state_predicate` at the same one state, so the two line up by
    construction rather than by coincidence. Measured on all four non-live
    cases — retired, superseded, ended, and a claim whose valid interval has not
    opened yet — `count()` and `live_claims` were both 0 while `claims` was not,
    and on the store that produced this bug `count()` at the right tenant and
    the file's `live_claims` were the same number.

    A store holding *only* claims none of the three live-time states names — a
    fact scheduled to start next month — stays silent, which is right: no scope
    would show it either, so `--tenant` is not the answer there.

    Returns None when there is nothing to say — claims are visible, the file
    holds no live claims, or the backend cannot answer the unfiltered question.
    A backend whose `stats` predates `live_claims` says nothing rather than
    falling back to `claims`: falling back would reinstate exactly the false
    positive this key exists to avoid. It never raises and never changes the
    exit code: this tool has no pass/fail exit status, and a measurement harness
    that starts failing runs is a different tool.
    """
    if mem.count():
        return None
    stats = getattr(getattr(mem, "store", None), "stats", None)
    if stats is None:
        return None
    try:
        live = stats(None).get("live_claims")
    except TypeError:  # a Store predating the tenant argument
        return None
    if not live:
        return None
    scope = getattr(mem, "default_scope", None)
    key = scope.key() if scope is not None else "unknown"
    return (f"WARNING: this store holds {live} live claims and none of them are "
            f"visible at scope {key} — every probe will miss and --draft will "
            f"print nothing. Re-run with --tenant/--user naming the scope the "
            f"claims are under.")


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
            # No rank on the verbatim row. A verbatim hit is rank 1 by
            # definition, so the number could only ever print 1.0 or vanish —
            # a column that cannot vary is not a measurement, and the spec
            # assigns gold-rank to hit/ambiguous only.
            rank = ("" if cls == "verbatim" else
                    f"  mean gold-rank {a['mean_gold_rank']:.1f}"
                    if a["mean_gold_rank"] is not None else "")
            label = _class_label(cls)
            lines.append(f"  {cls:<9} {a['n']:>3}   {label} {a['hit_at_k']:.1%}{rank}")
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
             for key in ("claims", "embedder", "surface", "tenant", "user")
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
                out.append(f"  {cls}: {_class_label(cls)} "
                           f"{agg_a[cls]['hit_at_k']:.1%} -> "
                           f"{agg_b[cls]['hit_at_k']:.1%}")
    return "\n".join(out)


def draft_probes(mem: Any, n: int, *, seed: int = 20260901) -> list[dict]:
    """Skeleton probes from live claims, every one refusing to run as-is.

    The query IS the claim's text, which is exactly what a probe must not be —
    so each row carries draft: true and load_probes refuses it until a person
    rewrites the query into how they would actually ask.

    `Claim.text` and not `Claim.object`: `text` is the "natural-language
    rendering, used for embedding" (`memvara/types.py`), so it is the string
    retrieval actually indexed, and self-retrieval@1 asks whether a claim's own
    text returns that claim first. Drafting from the raw object slot ("Lisbon"
    where the embedded text is "user lives in Lisbon") measures something
    weaker than the metric is named for. `text` defaults to `""` on the
    dataclass, so `object` remains the fallback for a claim that has none.

    A **sample**, drawn with a seeded `random.Random` off the id-sorted
    population: same store and same seed give the same rows, but the rows are
    not the lexicographically-first ids. Claim ids are content digests, so a
    lexicographic prefix is an arbitrary slice of the store that never moves —
    ten drafts in a row would keep proposing the same corner of it.

    Note the population itself is bounded on a hosted store: `RemoteMemvara.get_all`
    pages at `limit=100` and this asks once, so against a hosted deployment the
    sample is drawn from the first hundred claims, not from all of them.
    """
    population = sorted(mem.get_all(), key=lambda c: c.id)
    claims = random.Random(seed).sample(population, min(n, len(population)))
    rows: list[dict] = []
    for i, claim in enumerate(claims, start=1):
        text = claim.text or claim.object
        rows.append({"id": f"draft-hit-{i}", "class": "hit",
                     "query": text, "gold": [claim.id], "draft": True})
        rows.append({"id": f"draft-verbatim-{i}", "class": "verbatim",
                     "query": text, "gold": [claim.id], "draft": True})
    return rows


#: Internal traffic that reaches the recall hook but was never a user
#: question. Judged by prefix, because these are this repository's own
#: prompts and their openings are stable.
_INTERNAL_PREFIXES = ("Extract durable facts",)


def seed_dump(recalled_dir: Path, dump_path: Path, *, sample: int,
              seed: int = 20260901) -> "tuple[int, int]":
    """Phase one of closing the judgment loop: real queries, blinded, dumped.

    Blinding here means order (shuffled by seed) and absence of results — the
    judge sees only the query text and answers from the store, not from what
    the hook happened to return that day.

    **What this actually samples, which is narrower than "real recall
    traffic".** The directory is written by `plugin/hooks/recall.py`, which
    keys one file *per session* (`_seen_path` → `f"{session}.json"`) and
    rewrites it with `open(path, "w")` on every turn. So each file holds one
    query: that session's most recent substantive prompt, truncated to
    `MAX_CARRY_CHARS` (300) characters. `_prune_seen` deletes files older than
    `SEEN_TTL_SECONDS` (14 days). There is no append-only log of recall events
    anywhere on disk, and this does not reconstruct one.

    The consequence for anyone judging these rows: the sample is one query per
    recent session, skewed toward whatever each session happened to ask last,
    and blind to everything asked earlier in that session. It is still a
    thousand-odd distinct real queries on a working machine — worth judging —
    but it is a sample of session tails, not of traffic.

    Returns `(written, skipped)`. Files that could not be read as a JSON object
    are counted rather than dropped in silence: a directory where every file
    failed to parse would otherwise print exactly what an empty directory
    prints.
    """
    queries: dict[str, str] = {}
    skipped = 0
    for f in sorted(recalled_dir.glob("*.json")):
        try:
            query = json.loads(f.read_text()).get("query", "")
        except (ValueError, OSError, AttributeError, TypeError):
            # Valid-but-not-an-object JSON (`null`, `[]`, a bare number) parses
            # and then fails on `.get`, which crashed the whole run over one
            # bad file. `_state_json` in the hook tolerates the same shapes.
            skipped += 1
            continue
        query = " ".join(query.split()) if isinstance(query, str) else ""
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
    return len(rows), skipped


def seed_answers(dump_path: Path, answers_path: Path, judged: str) -> list[dict]:
    """Phase two: judgments back, probes out.

    gold=[] is a judgment too — 'the store has nothing for this' — and becomes
    an abstain probe rather than being dropped; only skip:true drops a row.

    A repeated id refuses, following `bench/evalkit.FileReader._load_answers`:
    an answers file assembled by concatenating two judging passes would
    otherwise be scored against whichever copy happened to be last, and which
    judgment was meant is not recoverable from the file.
    """
    dumped: dict[str, str] = {}
    for line in dump_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            dumped[row["id"]] = row["query"]
    probes: list[dict] = []
    seen: dict[str, int] = {}
    for lineno, raw in enumerate(answers_path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row["id"] in seen:
            raise SystemExit(
                f"{answers_path}: line {lineno}: id {row['id']!r} was already "
                f"judged on line {seen[row['id']]}. Which judgment was meant is "
                "not recoverable, and taking the last one silently seeds a probe "
                "nobody can reproduce. Remove the duplicate and re-run.")
        seen[row["id"]] = lineno
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


#: A `HashingEmbedder`'s recorded name. Every parameter that defines its vector
#: space is in the string — `hashing:<dim>:<lo>-<hi>`, built by
#: `HashingEmbedder.name` in `memvara/embed/base.py`, which says the dimension and
#: the n-gram range both belong in the fingerprint — so the name is enough to
#: construct the embedder back, and the reconstruction is checked against
#: `fingerprint_of` rather than trusted.
_HASHING_NAME = re.compile(r"^hashing:(\d+):(\d+)-(\d+)$")


def _store_embedder(store: Any) -> Any:
    """The embedder this store's vectors were written by, or None to let the library pick.

    `Memvara(path)` with no embedder takes `default_embedder()`, which returns a
    384-dimensional sentence-transformers model as soon as that package is
    importable and a 512-dimensional `HashingEmbedder` otherwise. So on a machine
    with the extra installed, `--db` against a store built by the fallback opened
    with `EmbedderMismatchError: ... holds 512-dimensional vectors ... but the
    configured embedder is local:... (dim 384)`. The tool could not open the store
    it exists to measure, and nothing caught it because every test injects `mem=`.

    The store records who wrote it — that is how the library detects the mismatch —
    so this reads that record back and honours it, on exactly the condition
    `Memvara._check_embedder` honours it on. Three answers, in order:

    * **Nothing binding**: None, which is today's behaviour unchanged, and the
      library's own dimension check still stands behind it. Either there is no
      record (no sidecar, an older store, a copy taken without it) *or* there are
      no vectors for a record to describe.
    * **A hashing embedder**: reconstructed from the name. `fingerprint_of` is then
      asked whether the reconstruction *is* what the store records, so a sidecar
      whose name and dim disagree falls through rather than opening a store with an
      embedder that merely looks right.
    * **Anything else**: `default_embedder()` is used only when it is itself what
      wrote the store — the sentence-transformers case, which works today and must
      keep working — and otherwise this refuses.

    **The record only binds while the store has vectors.** `_check_embedder` reads
    `stored_dim(self.store)` alongside the fingerprint and, when it is `None`,
    skips the compatibility check outright: a store with nothing in it has nothing
    to be incompatible with, and the embedder in hand simply becomes its owner.
    Reading the sidecar without asking the same question regressed two cases the
    pre-fix one-liner handled correctly, because it never read the sidecar at all —
    an orphaned sidecar with no store behind it refused an empty store that would
    have opened fine, and a reconstructable stale name silently opened a brand-new
    store at the *old* dimension. `stored_dim` is read off the vectors themselves
    and so cannot be stale; the sidecar can.

    What it deliberately does not do is construct a `local:MODEL` name by hand.
    The name does carry the model id, but building it means this bench deciding, on
    someone's behalf, to import torch and possibly download a model — and if the
    package is absent the answer is not a guess this tool gets to make. Where
    `default_embedder()` already returns that model, the branch above covers it.
    """
    from memvara.embed import HashingEmbedder, default_embedder
    from memvara.embed.fingerprint import fingerprint_of, read_fingerprint, stored_dim

    recorded = read_fingerprint(store)
    if recorded is None or stored_dim(store) is None:
        return None

    match = _HASHING_NAME.match(recorded.name)
    if match:
        dim, lo, hi = (int(g) for g in match.groups())
        candidate = HashingEmbedder(dim=dim, ngram=(lo, hi))
        if fingerprint_of(candidate) == recorded:
            return candidate

    fallback = default_embedder()
    if fingerprint_of(fallback) == recorded:
        return fallback

    # `store.path`, unguarded: a fingerprint was read, and `read_fingerprint` only
    # locates a sidecar for a store that has a non-empty string path.
    db = store.path
    raise SystemExit(
        f"{db}: these vectors were written by {recorded}, and this bench cannot "
        f"construct that embedder. The embedder it would otherwise use is "
        f"{fingerprint_of(fallback)}; opening with it raises EmbedderMismatchError "
        "at a different width, and at the same width raises nothing while every "
        "score compares two unrelated vector spaces.\n"
        "If that name is a model you have not installed, install it — "
        "pip install 'memvara[local-embed]' for a 'local:' name — and re-run. "
        "Otherwise open the store yourself and hand it in:\n"
        "    import bench.hosted as hosted\n"
        "    from memvara import Memvara\n"
        f"    hosted.main([...], mem=Memvara({db!r}, embedder=<{recorded.name}>))")


def _open_store(args: argparse.Namespace) -> Any:
    """The store to measure: a local file under `--db`, else the hosted deployment.

    `--db` opens with the embedder the store was built with (`_store_embedder`),
    not with whatever `default_embedder()` returns in this process.

    **And at the scope `--tenant`/`--user` name, not at `tenant="default"`.**
    A store file is not one population — scopes inherit upward, and a file whose
    claims all live under a named tenant answers nothing at `default`. Opened
    unscoped, a real store read as empty and the tool printed nothing at all.

    `--tenant` is refused on the hosted route rather than forwarded. The
    parameter exists on `RemoteMemvara` but is "held for `default_scope`'s sake
    and never sent" — the facade resolves the tenant from the bearer token — so
    forwarding it would move the recorded fingerprint without moving a single
    byte of what was read, which is a silent no-op wearing the appearance of a
    setting. `--user` *is* a real narrowing on both routes and is forwarded to
    both.

    The `SQLiteStore` is built here rather than left to `Memvara(path)` because
    deciding the embedder means reading the store — both its recorded fingerprint
    and, decisively, whether it holds any vectors for that record to describe. One
    store object answers that question and then goes on to be the one measured, so
    there is no second open and no chance of the two disagreeing. `Memvara.close()`
    closes it, which is what the `finally` in `main` already relies on; a refusal
    happens before any `Memvara` exists, so it closes the store itself.
    """
    if args.db:
        from memvara import Memvara
        from memvara.store import SQLiteStore
        store = SQLiteStore(args.db)
        try:
            embedder = _store_embedder(store)
        except BaseException:
            store.close()
            raise
        return Memvara(store=store, embedder=embedder,
                       tenant=args.tenant, user=args.user or None)
    if args.tenant != "default":
        raise SystemExit(
            f"--tenant {args.tenant!r} needs --db PATH. On the hosted route the "
            "facade resolves the tenant from your credential and the client "
            "never sends the parameter, so honouring the flag here would change "
            "what this run records about itself without changing what it read.")
    from memvara.remote.api import RemoteMemvara
    return RemoteMemvara(user=args.user or None)


def main(argv: "Sequence[str] | None" = None, *, mem: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probes", default=str(Path.home() / ".memvara" / "probes.jsonl"))
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="results per probe; 4 is the recall hook's own K")
    parser.add_argument("--min-score", type=float, default=default_min_score(),
                        help="relevance floor on both read surfaces; the default "
                             "resolves exactly as the recall hook's own "
                             "_min_score() does (MEMVARA_RECALL_MIN_SCORE if set, "
                             "else MIN_SCORE), so the run measures what the hook "
                             "actually injects on this machine. Pass 0 to measure "
                             "the unfloored read path.")
    parser.add_argument("--db", default="", help="local store path; omit for hosted")
    parser.add_argument("--tenant", default="default",
                        help="tenant to read at; --db only, because the hosted "
                             "facade resolves the tenant from your credential")
    parser.add_argument("--user", default="",
                        help="user scope to read at; omit to read the whole tenant")
    parser.add_argument("--out", default="", help="write per-probe JSONL here")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None)
    # Default None, not 0: `--draft 0` computed by a wrapper ("draft up to the
    # remaining budget") must be a no-op, not a fall-through into a full
    # scoring run against a probe file that may not exist.
    parser.add_argument("--draft", type=int, default=None, metavar="N")
    parser.add_argument("--seed-from-recalled", default="", metavar="DIR")
    parser.add_argument("--dump", default="", metavar="PATH")
    parser.add_argument("--answers", default="", metavar="PATH")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--judged", default="", metavar="YYYY-MM-DD")
    args = parser.parse_args(argv)

    # --compare needs no store — two result files, read and diffed.
    if args.compare:
        print(compare_runs(Path(args.compare[0]), Path(args.compare[1])))
        return 0

    # Seeding needs no store either — it reads/writes the recall hook's
    # per-session state files and the dump/answers files directly.
    if args.seed_from_recalled:
        if args.answers:
            if not args.dump:
                raise SystemExit("--answers needs --dump PATH: the dump phase one "
                                 "wrote, so judgments can be matched back to "
                                 "their queries")
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
        n, skipped = seed_dump(Path(args.seed_from_recalled), Path(args.dump),
                               sample=args.sample, seed=args.seed)
        if skipped:
            print(f"skipped {skipped} unreadable file(s) in "
                  f"{args.seed_from_recalled}")
        print(f"wrote {n} blinded queries to {args.dump}; judge them into "
              f"{{\"id\", \"gold\": [claim ids]}} rows (gold [] = nothing "
              f"relevant; \"skip\": true = drop), then re-run with "
              f"--answers PATH --judged YYYY-MM-DD")
        return 0

    close = False
    if mem is None:
        mem = _open_store(args)
        close = True
    try:
        # Before every branch that reads the store, so the one warning covers a
        # --draft run that prints nothing and a scoring run whose table reads
        # `0 claims` alike. stderr, not stdout: --draft writes JSONL to stdout
        # and a person redirects it into a probe file, so a warning printed
        # there would corrupt the file it is trying to help them build.
        #
        # `args.db`, not `close`: the gate is *local store*, not *this function
        # opened it*, so a caller who hands in `mem=` for a file it names is
        # covered too — and the hosted route is never asked a whole-store
        # question it has no business answering.
        if args.db:
            blindspot = scope_blindspot(mem)
            if blindspot:
                print(blindspot, file=sys.stderr)
        # Opened before load_probes so a --draft branch (Task 5) can use the
        # store handle to generate probes for someone with no probe file yet.
        if args.draft is not None:
            for row in draft_probes(mem, args.draft, seed=args.seed):
                print(json.dumps(row))
            return 0
        probes = load_probes(Path(args.probes))
        fingerprint = store_fingerprint(mem)
        rows = run_probes(mem, probes, k=args.k, min_score=args.min_score)
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
