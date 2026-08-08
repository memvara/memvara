"""Throughput and scaling profile.

Run:  PYTHONPATH=. python3 bench/perf.py [--profile]

Measures the operations that actually run in a loop — writes, searches, and the
consolidation pass — at increasing store sizes, and reports per-operation cost so
super-linear behavior is visible rather than inferred.
"""

from __future__ import annotations

import cProfile
import pstats
import sys
import time
from io import StringIO

from engram import Engram, HashingEmbedder
from engram.types import Claim, Scope

CITIES = ["Berlin", "Lisbon", "Osaka", "Nairobi", "Lima", "Oslo", "Cairo", "Perth"]


def build(n: int, *, path: str = ":memory:") -> Engram:
    """A store with `n` live claims spread over many slots and users."""
    mem = Engram(path, embedder=HashingEmbedder(dim=256), user="alice")
    for i in range(n):
        mem.remember("user", f"pred_{i % 400}", f"{CITIES[i % len(CITIES)]}_{i}",
                     user=f"u{i % 50}")
    return mem


def timed(label: str, fn, reps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    dt = (time.perf_counter() - t0) / reps
    print(f"    {label:<28}{dt * 1000:>9.3f} ms/op   {1 / dt:>10.0f} ops/s")
    return dt


def scaling() -> None:
    print("\n=== scaling ===")
    prev: dict[str, float] = {}
    for n in (500, 2_000, 8_000):
        print(f"\n  n = {n:,} claims")
        t0 = time.perf_counter()
        mem = build(n)
        build_s = time.perf_counter() - t0
        print(f"    {'write (bulk)':<28}{build_s / n * 1000:>9.3f} ms/op   "
              f"{n / build_s:>10.0f} ops/s")

        w = timed("write (single)", lambda: mem.remember("user", "extra", "x"), 50)
        s = timed("search k=10", lambda: mem.search("Berlin lives", k=10), 50)
        g = timed("get_all", lambda: mem.get_all(user="u1"), 20)
        c = timed("consolidate", lambda: mem.consolidate(), 3)

        cur = {"write": w, "search": s, "get_all": g, "consolidate": c}
        if prev:
            growth = n / prev["_n"]
            print(f"    -- scaling vs previous ({growth:.0f}x more claims):")
            for k in ("write", "search", "get_all", "consolidate"):
                ratio = cur[k] / prev[k]
                verdict = "flat" if ratio < growth * 0.35 else (
                    "sub-linear" if ratio < growth * 0.8 else "LINEAR+")
                print(f"       {k:<24}{ratio:>6.1f}x slower   {verdict}")
        cur["_n"] = n
        prev = cur
        mem.close()


def profile_hotspots() -> None:
    print("\n=== hotspots (8k claims) ===")
    mem = build(8_000)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(30):
        mem.search("Berlin lives", k=10)
    for _ in range(30):
        mem.remember("user", "extra", "x")
    mem.consolidate()
    pr.disable()

    buf = StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(18)
    for line in buf.getvalue().splitlines():
        if line.strip():
            print("   ", line)
    mem.close()


if __name__ == "__main__":
    scaling()
    if "--profile" in sys.argv:
        profile_hotspots()
