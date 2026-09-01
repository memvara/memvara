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
import json
import random
import sys
import time
from datetime import datetime, timezone
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
            # verbatim only counts a hit at rank 1 exactly, never at rank k —
            # "hit@k" would misdescribe it. self-retrieval@1 is the design
            # doc's own name for this metric.
            label = "self-retrieval@1" if cls == "verbatim" else "hit@k"
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
    parser.add_argument("--draft", type=int, default=0, metavar="N")
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

    # Seeding needs no store either — it reads/writes recalled-event and
    # dump/answers files directly.
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

    close = False
    if mem is None:
        mem = _open_store(args)
        close = True
    try:
        # Opened before load_probes so a --draft branch (Task 5) can use the
        # store handle to generate probes for someone with no probe file yet.
        if args.draft:
            for row in draft_probes(mem, args.draft):
                print(json.dumps(row))
            return 0
        probes = load_probes(Path(args.probes))
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
