"""Agent Memory Benchmark — does a memory system get *changing* facts right?

Retrieval benchmarks ask whether the right sentence comes back. This one asks whether
the right sentence is still true, and if not, when it stopped being true, when we found
out, and who told us. Those are different questions, and a store that scores well on the
first can score zero on the rest.

The benchmark is system-neutral by construction. Everything specific to a memory system
lives behind `adapters.base.MemorySystem`; the dataset, the questions and the scoring
never mention one. `benchmarks/agent_memory/README.md` is the front door, and
`docs/benchmarks/agent-memory-benchmark.md` is the public report.

    python -m benchmarks.agent_memory --system memvara
    python -m benchmarks.agent_memory --system naive --output results.json

Nothing here reaches the network, and no run needs an API key.
"""

#: Bumped when the questions, the scoring or the dataset change in a way that makes a
#: new number incomparable with an old one. `docs/benchmarks/agent-memory-benchmark.md`
#: explains what counts as material; the short version is that anything which could move
#: a published score is material.
#:
#: 2.0 ships dataset v2 and a shared slot-selection rule for every adapter. Both move
#: published scores, and v1 is still in the tree and still runs: `--dataset v1`.
BENCHMARK_VERSION = "2.0"

#: The dataset a bare run uses. Older versions stay in `datasets/` and stay loadable, so
#: a published v1 number can be reproduced after v2 exists — which is the whole reason
#: the versioning rule says an old version stays where it is.
DEFAULT_DATASET = "v2"

__all__ = ["BENCHMARK_VERSION", "DEFAULT_DATASET"]
