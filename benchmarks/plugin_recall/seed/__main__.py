"""Seed a store, or regenerate the hit corpus.

    python -m benchmarks.plugin_recall.seed --target supermemory --url http://localhost:6767
    python -m benchmarks.plugin_recall.seed --target memvara --db /tmp/bench.db
    python -m benchmarks.plugin_recall.seed --emit-cases benchmarks/plugin_recall/cases/v1_hits.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from . import emit_cases, facts

#: Both stores are namespaced so a benchmark run cannot be confused with real memory, and
#: so the seeded rows are findable afterwards by whoever has to clear them out.
CONTAINER = "plugin-recall-bench"


def seed_supermemory(url: str, container: str) -> int:
    written = 0
    for fact in facts():
        body = json.dumps({
            "content": fact["prose"],
            "containerTag": container,
            "metadata": {"sm_source": "plugin-recall-bench", "fact_id": fact["id"]},
        }).encode()
        request = urllib.request.Request(
            f"{url.rstrip('/')}/v3/documents", data=body,
            # An explicit User-Agent, always. The stdlib default is rejected outright by
            # some edges, and the 403 that comes back says nothing about the client's name
            # being the cause -- a trap this repository has already paid for once.
            headers={"Content-Type": "application/json", "User-Agent": "plugin-recall-bench/1"},
            method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status >= 300:
                raise RuntimeError(f"{fact['id']}: HTTP {response.status}")
        written += 1
    return written


def seed_memvara(db: Path) -> int:
    # Built through the *server's* config, not `Memvara(path)`, so the store is created
    # with the same embedder the hook will later open it with. Seeding with the library
    # default instead produced a 384-dimensional store that the hook -- configured for
    # `hashing:512:3-5` -- refused to open, and `open_store()` answers that refusal with
    # `None`, which sends the hook to the hosted store instead. The benchmark then scored
    # a 0% hit rate against a store it had never read: every fact present, every write
    # successful, and the reader looking somewhere else entirely.
    import os

    from memvara.server.config import ServerConfig, build_memvara

    env = {**os.environ, "MEMVARA_DB": str(db), "MEMVARA_MODE": "local"}
    memory = build_memvara(ServerConfig.from_env(env))
    try:
        for fact in facts():
            subject, predicate, obj = fact["triple"]
            # `remember`, not `add`. Extraction from prose needs a model, and with none
            # configured the deterministic fast path matches only a fixed set of sentence
            # forms -- so half these facts would be accepted and silently not stored, and
            # the benchmark would report a retrieval failure for a write that never
            # happened.
            memory.remember(subject, predicate, obj, text=fact["prose"])
        return len(facts())
    finally:
        memory.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", choices=("supermemory", "memvara"))
    parser.add_argument("--url", default="http://localhost:6767",
                        help="supermemory base URL. Point this at a local server; seeding "
                             "a hosted account leaves rows you may not be able to delete.")
    parser.add_argument("--container", default=CONTAINER)
    parser.add_argument("--db", type=Path, help="memvara store path. Use a throwaway file.")
    parser.add_argument("--emit-cases", type=Path)
    args = parser.parse_args(argv)

    if args.emit_cases:
        print(f"wrote {emit_cases(args.emit_cases)} hit cases to {args.emit_cases}")
    if args.target == "supermemory":
        print(f"seeded {seed_supermemory(args.url, args.container)} facts into "
              f"{args.url} (containerTag={args.container})")
    elif args.target == "memvara":
        if not args.db:
            print("error: --db is required for memvara", file=sys.stderr)
            return 2
        print(f"seeded {seed_memvara(args.db)} facts into {args.db}")
    if not args.target and not args.emit_cases:
        parser.error("nothing to do: pass --target or --emit-cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
