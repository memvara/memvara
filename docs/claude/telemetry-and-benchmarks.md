# Telemetry and benchmarks

These two subsystems answer the same question from opposite ends: is this store, or this
change, actually any good? Telemetry answers it about a running deployment over months.
Benchmarks answer it about a code change today, with numbers somebody else can reproduce.

Both exist because most of the ways this library can get worse do not raise. A red-team
review of eleven long-horizon failure modes classified six of them as silent: they degrade
answer quality with no error, no exception and nothing in any log.

## Where the code is

- Telemetry: `memvara/telemetry.py` — the `Recorder` protocol, `NullRecorder` (discards
  everything, and is not the default), `MemoryRecorder`, `series_names()`,
  `rank_correlation()`, `script_of()`. The module docstring lists the six silent failures
  from the red-team review, the seventh that arrived with the redaction seam, and the series
  that catches each.
- Retrieval and ranking benchmarks: `bench/locomo.py`, `bench/longmemeval.py`,
  `bench/twowiki.py`, `bench/multihop.py`, `bench/temporal.py`.
- Comparison and cost: `bench/compare.py`, `bench/mem0_real.py`, `bench/baseline.py`,
  `bench/extract_cost.py`, `bench/perf.py`, `bench/evalkit.py`.
- Against a hosted store: `bench/hosted.py`.
- End-to-end answer quality: `demo/harness.py`, `demo/scenario.py`, `demo/baselines.py`,
  with recorded runs under `demo/runs/`.
- The public benchmark others can enter: `benchmarks/agent_memory/`, with its own
  `benchmarks/agent_memory/README.md` and adapters under
  `benchmarks/agent_memory/adapters/`.
- Tests: `tests/test_telemetry.py`, `tests/test_bench_eval.py`, `tests/test_bench_hosted.py`,
  `tests/test_agent_memory_bench.py`, `tests/test_demo.py`, `tests/test_demo_scenario.py`,
  `tests/test_plugin_recall_bench.py`.
- Documentation: [BENCHMARKS.md](../BENCHMARKS.md) is the results document — every number,
  what it measures, and what it does not.
  [The Agent Memory Benchmark page](../benchmarks/agent-memory-benchmark.md) is the reader's
  entry point.

## How the pieces fit

Telemetry is opt-in and off by default. A store's `telemetry` is `None` unless a deployment
passes a recorder, and `None` skips the recording call entirely; that is the fast path.
`NullRecorder` is a recorder that discards everything, correct but not free, and its own
docstring says so. Nothing is written anywhere the operator did not ask for. `WriteReceipt` and `Explanation` already answer "what did this call
do?" precisely; telemetry answers the aggregate question "is this store getting worse?",
which is the one that matters over a year and the one nothing else can see.

If only one series is ever wired to a dashboard it should be
`consolidate.claims_per_slot`, which is the one that would have caught the worst bug in the
project in its first week.

The benchmark scripts under `bench/` are run by hand and write their numbers into
[BENCHMARKS.md](../BENCHMARKS.md) alongside the caveats that make each number less than it
looks. `demo/harness.py` is the end-to-end run: an authored corpus, an agent reading the
memory block, and a scored answer file. `benchmarks/agent_memory/` is different in kind — it
is written to be entered by somebody adding their own memory system, so it has adapters, a
contributor guide and a CLI rather than a script per corpus.

## Invariants and assumptions

- **"Verify" means comparing an output, never that a command exited 0.** A count, a series, a
  diff. This is the local form of the fourth Karpathy guideline and it is in `CLAUDE.md` for
  that reason.
- **A number goes into the results document with its caveat attached.** An invented or
  unqualified figure is worse than none; where nothing was measured, the text says so.
- **The offline run repeats exactly.** `demo/harness.py`'s scored run is one command and
  deterministic, so a change in the score is a change in the system rather than in the
  weather.
- **The benchmark harnesses do not exercise ingestion.** The retrieval corpora run against
  structured data with no extraction model, so a regression that broke extraction would not
  show up there. `demo/harness.py` is the run that would catch "zero claims extracted".
- **Telemetry adds no required dependency and no background thread.** `MemoryRecorder` keeps
  series in memory for a deployment to scrape; the protocol is what a real backend
  implements.

## Read next

The module docstring in `memvara/telemetry.py` is the argument for the module and the table
mapping each silent failure to the series that catches it.
[BENCHMARKS.md](../BENCHMARKS.md) is long on purpose: read the section for the claim you are
about to make rather than the summary.

Next: [releases and the plugin repositories](release-and-plugins.md).
