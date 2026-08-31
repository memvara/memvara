"""`python -m benchmarks.agent_memory.run` — the same command, spelled the long way.

`python -m benchmarks.agent_memory` is the canonical form and is what the documentation
uses. This module exists because `...agent_memory.run` is the spelling people reach for
first, and a benchmark that answers a near-miss with `No module named` has spent its one
chance at being run.

Both routes call `cli.main`; there is no second implementation to drift.

Unlike `__main__.py`, this module has an importable dotted name, so somebody can write
`from benchmarks.agent_memory import run` — and without the guard below that import would
run the whole benchmark and then kill the interpreter with `SystemExit`. The guard is not
decoration: `-m` sets `__name__` to `"__main__"`, so the command still works, and the
import becomes the no-op a reader expects.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
