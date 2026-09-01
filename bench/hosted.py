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
import time
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
                "mean_gold_rank": (sum(ranks) / len(ranks)) if ranks else None,
            }
    return out


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
