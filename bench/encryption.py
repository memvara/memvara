"""What encryption at rest costs: write latency, search latency, open time and memory.

Run:  PYTHONPATH=. python3 bench/encryption.py [--n 8000 20000] [--reps 300]

Builds the store `bench/perf.py` builds (n claims over 400 predicates and 50 users,
`HashingEmbedder(dim=256)`), once unencrypted and once encrypted, each in a file in a
temporary directory. Every measurement runs in a fresh child process per mode and size,
so the resident-memory figure of one run is not inflated by the other.

Needs the `encrypt` extra. The key is passed to `SQLiteStore` directly, so this script
never reads the OS keychain, `MEMVARA_DB_KEY` or `~/.memvara/db.key`.

What each column means:

* **build**: writing the n claims with `remember()`, per claim.
* **write p50/p95**: one more `remember()` into a populated store, `--reps` times.
* **search p50/p95**: `search(k=10)` against a populated scope, `--reps` times, after
  one warm-up search.
* **open**: constructing the store object on an existing file.
* **first search**: the first `search()` after opening, which loads the vector index.
  For an unencrypted store that maps the vector file; for an encrypted one it decrypts
  every row of it into memory.
* **peak RSS**: the child process's peak resident memory after the first search and
  the timed searches (`ru_maxrss`). A memory-mapped matrix counts toward RSS once its
  pages are touched, so the difference between the two modes is smaller than the size
  of the matrix suggests; what differs is that the encrypted matrix is private to the
  process, where the mapped one is shared with every other process that maps the file.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import secrets
import statistics
import subprocess
import sys
import tempfile
import time

CITIES = ["Berlin", "Lisbon", "Osaka", "Nairobi", "Lima", "Oslo", "Cairo", "Perth"]


def _memvara(path: str, key: bytes | None):
    from memvara import HashingEmbedder, Memvara, NullLLM
    from memvara.store import SQLiteStore

    store = SQLiteStore(path, key=key)
    return Memvara(store=store, embedder=HashingEmbedder(dim=256), llm=NullLLM(),
                   user="alice")


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Bytes on macOS, kilobytes on Linux.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def child(mode: str, n: int, reps: int, directory: str, key_hex: str) -> dict:
    key = bytes.fromhex(key_hex) if mode == "encrypted" else None
    path = os.path.join(directory, f"{mode}-{n}.db")
    if not os.path.exists(path):
        mem = _memvara(path, key)
        t0 = time.perf_counter()
        for i in range(n):
            mem.remember("user", f"pred_{i % 400}", f"{CITIES[i % len(CITIES)]}_{i}",
                         user=f"u{i % 50}")
        build = (time.perf_counter() - t0) / n
        mem.close()
        return {"build_ms": build * 1000}

    from memvara.select import PLAIN_READ

    t0 = time.perf_counter()
    mem = _memvara(path, key)
    opened = time.perf_counter() - t0
    t0 = time.perf_counter()
    assert mem.search("Berlin lives", k=10, user="u1", **PLAIN_READ), "search must hit a populated scope"
    first = time.perf_counter() - t0

    searches = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mem.search("Berlin lives", k=10, user="u1", **PLAIN_READ)
        searches.append(time.perf_counter() - t0)
    writes = []
    for i in range(reps):
        t0 = time.perf_counter()
        mem.remember("user", f"extra_{i}", f"x{i}", user="u1")
        writes.append(time.perf_counter() - t0)
    rss = _peak_rss_mb()
    mem.close()
    size = os.path.getsize(path) + (os.path.getsize(path + ".vecs")
                                    if os.path.exists(path + ".vecs") else 0)
    return {
        "open_ms": opened * 1000, "first_search_ms": first * 1000,
        "search_p50_ms": _pct(searches, 0.5) * 1000,
        "search_p95_ms": _pct(searches, 0.95) * 1000,
        "search_mean_ms": statistics.fmean(searches) * 1000,
        "write_p50_ms": _pct(writes, 0.5) * 1000,
        "write_p95_ms": _pct(writes, 0.95) * 1000,
        "write_mean_ms": statistics.fmean(writes) * 1000,
        "peak_rss_mb": rss, "files_mb": size / (1024 * 1024),
    }


def _run_child(mode: str, n: int, reps: int, directory: str, key_hex: str) -> dict:
    done = subprocess.run(
        [sys.executable, __file__, "--child", mode, "--n", str(n), "--reps", str(reps),
         "--dir", directory, "--key", key_hex],
        check=True, capture_output=True, text=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, nargs="+", default=[8_000, 20_000])
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--child")
    parser.add_argument("--dir")
    parser.add_argument("--key")
    args = parser.parse_args()
    if args.child:
        print(json.dumps(child(args.child, args.n[0], args.reps, args.dir, args.key)))
        return

    key_hex = secrets.token_hex(32)
    with tempfile.TemporaryDirectory() as directory:
        for n in args.n:
            rows = {}
            for mode in ("plain", "encrypted"):
                built = _run_child(mode, n, args.reps, directory, key_hex)
                rows[mode] = {**built, **_run_child(mode, n, args.reps, directory,
                                                    key_hex)}
            print(f"\n=== n = {n:,} claims, dim 256, {args.reps} timed ops each ===")
            fields = [("build_ms", "build ms/claim"), ("write_p50_ms", "write p50 ms"),
                      ("write_p95_ms", "write p95 ms"), ("search_p50_ms", "search p50 ms"),
                      ("search_p95_ms", "search p95 ms"), ("open_ms", "open ms"),
                      ("first_search_ms", "first search ms"),
                      ("peak_rss_mb", "peak RSS MB"), ("files_mb", "files on disk MB")]
            print(f"  {'':<18}{'plain':>12}{'encrypted':>12}{'change':>10}")
            for name, label in fields:
                a, b = rows["plain"][name], rows["encrypted"][name]
                print(f"  {label:<18}{a:>12.3f}{b:>12.3f}{(b / a - 1) * 100:>+9.0f}%")


if __name__ == "__main__":
    main()
