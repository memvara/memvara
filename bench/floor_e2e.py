"""End-to-end measurement of the candidate floor (#155) against a real store.

    PYTHONPATH=. python3 bench/floor_e2e.py before            # hosted, as deployed today
    PYTHONPATH=. python3 bench/floor_e2e.py replay            # local copy, floor 0 vs 50
    PYTHONPATH=. python3 bench/floor_e2e.py after             # hosted, once the box has the branch
    PYTHONPATH=. python3 bench/floor_e2e.py compare before after

Every step is `bench/hosted.py` under a fixed set of arguments, so the numbers here are the
numbers the issue was measured with: the hook's `k=4`, the hook's own relevance floor, and
one result file per run under `local/floor-e2e/`. `SESSION-2026-09-02-candidate-floor-e2e.md`
says which order to run them in and what decides.

The floor lives in the retriever, which runs inside memvara-cloud, so `before` and `after`
differ only in what the box is running. `replay` is the same measurement against a copy of
the store rebuilt here — every claim exported over the API, re-embedded with the model the
deployment uses — and it is evidence rather than the decision: SQLite's lexical leg is not
Postgres's, and a copy taken now is not the store as it was when the probes were judged.
Floor 0 on the copy should agree with `before`; how closely says how far to trust floor 50
as a prediction of `after`.

Credentials are the ones `bench/hosted.py` already reads: `MEMVARA_API_KEY` and
`MEMVARA_SERVER_URL`, or the file `memvara-mcp login` writes. Nothing here stores them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import bench.hosted as hosted

OUT_DIR = Path("local") / "floor-e2e"
PROBES = Path.home() / ".memvara" / "probes.jsonl"
K = hosted.DEFAULT_K

#: The floor the branch ships, and the value that reproduces `main`. Both are spelled
#: here so a reader of the result files knows what each was run at.
FLOOR_SHIPPED = 50
FLOOR_OFF = 0

#: `GET /v1/memories` caps a page at 500.
PAGE = 500


def _hosted_run(name: str, args: argparse.Namespace) -> int:
    out = OUT_DIR / f"{name}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    argv = ["--probes", str(args.probes), "--k", str(args.k), "--out", str(out)]
    if args.user:
        argv += ["--user", args.user]
    print(f"== {name}: hosted store, k={args.k}, writing {out}")
    return hosted.main(argv)


def export_claims(source: Any) -> list[Any]:
    """Every claim the credential can see, in all three states, one page at a time.

    All three states rather than `live` alone: a search runs its state filter in the
    store, so a copy holding only live rows would rank against a smaller population
    than the deployment does, and the whole point of the copy is the population.
    """
    states = ["live", "ended", "retired"]
    claims: list[Any] = []
    offset = 0
    while True:
        page = source.get_all(states=states, limit=PAGE, offset=offset)
        claims.extend(page)
        if len(page) < PAGE:
            return claims
        offset += PAGE


def build_copy(claims: Sequence[Any], db: Path, embedder: Any) -> Any:
    """A local store holding `claims` under their own ids and scopes, embedded by
    `embedder`, with the fingerprint sidecar `bench/hosted.py --db` reads.

    `reembed` is the library's own migration path and is used for exactly what it
    says: it encodes `Claim.text` — the same field the write path embeds — and writes
    the fingerprint, so `hosted._store_embedder` reconstructs the embedder from the
    sidecar and refuses if it cannot.
    """
    from memvara import Memvara, NullLLM
    from memvara.embed.fingerprint import fingerprint_of
    from memvara.store import SQLiteStore

    if db.exists():
        db.unlink()
    store = SQLiteStore(str(db))
    for claim in claims:
        store.put_claim(claim)
    mem = Memvara(store=store, embedder=embedder, llm=NullLLM(),
                  tenant=claims[0].scope.tenant)
    embedded = mem.reembed()
    print(f"   {len(claims)} claims copied, {embedded} embedded by "
          f"{fingerprint_of(embedder)}")
    return mem


def probe_scope(claims: Sequence[Any]) -> Any:
    """The scope the probes are read at: the one most of the claims live under.

    Scopes inherit upward and never sideways, so a copy queried at the tenant sees
    only tenant-level rows, and a store the plugin wrote is almost entirely one user
    under one tenant. The hosted run reads at the credential's own scope; this picks
    the scope that holds the most claims and says how many that is, so a store spread
    across users is visible as such rather than silently measured at one of them.
    """
    from collections import Counter

    counts = Counter(c.scope for c in claims)
    scope, n = counts.most_common(1)[0]
    print(f"   reading at scope {scope}, which holds {n} of {len(claims)} claims"
          + ("" if len(counts) == 1 else f" ({len(counts)} scopes in the export)"))
    return scope


def replay(args: argparse.Namespace) -> int:
    from memvara import Memvara, NullLLM
    from memvara.embed import default_embedder
    from memvara.embed.fingerprint import fingerprint_of
    from memvara.remote.api import RemoteMemvara

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = OUT_DIR / "copy.db"

    print("== replay: exporting the hosted store")
    remote = RemoteMemvara(user=args.user or None)
    try:
        claims = export_claims(remote)
    finally:
        remote.close()
    if not claims:
        print("   the credential sees no claims; nothing to replay", file=sys.stderr)
        return 2

    embedder = default_embedder()
    name = fingerprint_of(embedder).name
    if not name.startswith("local:"):
        # The deployment embeds with sentence-transformers. A hashing embedder here
        # would rank a different vector space and the replay would measure nothing
        # about the store.
        print(f"   default_embedder() is {name}; install memvara[local-embed] so the "
              f"copy is embedded the way the deployment embeds it", file=sys.stderr)
        return 2

    base = build_copy(claims, db, embedder)
    scope = probe_scope(claims)
    try:
        for floor in (FLOOR_OFF, FLOOR_SHIPPED):
            out = OUT_DIR / f"replay-floor-{floor}.jsonl"
            mem = Memvara(store=base.store, embedder=embedder, llm=NullLLM(),
                          tenant=scope.tenant, user=scope.user, agent=scope.agent,
                          session=scope.session, read_candidate_floor=floor)
            print(f"== replay: floor {floor}, k={args.k}, writing {out}")
            argv = ["--probes", str(args.probes), "--db", str(db),
                    "--tenant", scope.tenant, "--k", str(args.k), "--out", str(out)]
            if scope.user:
                argv += ["--user", scope.user]
            hosted.main(argv, mem=mem)
    finally:
        base.close()
    print("== replay: floor 0 against floor 50")
    print(hosted.compare_runs(OUT_DIR / f"replay-floor-{FLOOR_OFF}.jsonl",
                              OUT_DIR / f"replay-floor-{FLOOR_SHIPPED}.jsonl"))
    return 0


def compare(args: argparse.Namespace) -> int:
    a, b = (OUT_DIR / f"{n}.jsonl" for n in args.runs)
    for path in (a, b):
        if not path.exists():
            print(f"{path} does not exist; run that step first", file=sys.stderr)
            return 2
    print(hosted.compare_runs(a, b))
    return 0


def main(argv: "Sequence[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--probes", type=Path, default=PROBES)
    parser.add_argument("--k", type=int, default=K, help=f"the hook's own K, {K}")
    parser.add_argument("--user", default="", help="narrow to one user, on every step")
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("before", help="hosted store as deployed today")
    sub.add_parser("replay", help="local copy of the store at floor 0 and floor 50")
    sub.add_parser("after", help="hosted store once the box carries the branch")
    cmp = sub.add_parser("compare", help="diff two result files by name")
    cmp.add_argument("runs", nargs=2, metavar="RUN",
                     help="before, after, replay-floor-0 or replay-floor-50")
    args = parser.parse_args(argv)

    if args.step in ("before", "after"):
        return _hosted_run(args.step, args)
    if args.step == "replay":
        return replay(args)
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
