"""What a metadata filter costs a search, with SQLite's JSON functions and without them.

Run:  PYTHONPATH=. python3 bench/filters.py [N ...]

For each store size N (default 15,000 and 40,000, the sizes design invariant 7 in
`docs/INTERNALS.md` is measured at), the store holds N competing claims tagged
`team=infra` that match the query well and 20 tagged `team=web` that match it weakly. It
times a search with `k=10`: unfiltered, then filtered to `team=web` using SQLite's JSON
functions (`json_type` and `json_extract`), then filtered with the Python callback
`mv_meta_match` that a SQLite built without those functions uses. Each figure is the
median of 15 searches after 3 warm-up searches, with `query_rewrite` off so no model is
involved. It also checks that every filtered search returned 10 `web` rows.
"""

from __future__ import annotations

import statistics
import sys
import time

from memvara import HashingEmbedder, Memvara, NullLLM

QUERY = "orbital cluster"
MATCHING = 20


def build(n: int) -> Memvara:
    mem = Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=256), user="alice")
    with mem.store.batch():
        for i in range(n):
            mem.remember(f"node{i}", "uses", f"{QUERY} {i}", team="infra")
        for i in range(MATCHING):
            mem.remember(f"site{i}", "notes",
                         f"the {QUERY} appears once in this longer note number {i}",
                         team="web")
    return mem


def median_ms(mem: Memvara, **kw: object) -> float:
    for _ in range(3):
        mem.search(QUERY, k=10, query_rewrite=False, **kw)
    times = []
    for _ in range(15):
        t0 = time.perf_counter()
        hits = mem.search(QUERY, k=10, query_rewrite=False, **kw)
        times.append((time.perf_counter() - t0) * 1000)
        if kw:
            assert len(hits) == 10 and all(h.claim.meta.get("team") == "web" for h in hits)
    return statistics.median(times)


def main(sizes: list[int]) -> None:
    print(f"{'claims':>8} {'unfiltered':>12} {'json functions':>16} {'python callback':>17}")
    for n in sizes:
        mem = build(n)
        plain = median_ms(mem)
        fast = median_ms(mem, filters={"team": "web"})
        mem.store._json_functions = False
        slow = median_ms(mem, filters={"team": "web"})
        print(f"{n:>8} {plain:>10.1f}ms {fast:>14.1f}ms {slow:>15.1f}ms")
        mem.close()


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [15_000, 40_000])
