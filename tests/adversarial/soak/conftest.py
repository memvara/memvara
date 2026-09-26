"""Put `bench/` on the import path, so these tests can import `soak` and `perf_budget`.

`bench/` is a folder of scripts rather than a package, and its scripts import each other
by bare name (`import evalkit`), as they do when run as `python bench/soak.py`.
`tests/test_bench_eval.py` handles the same folder the same way.
"""

from __future__ import annotations

import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[3] / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
