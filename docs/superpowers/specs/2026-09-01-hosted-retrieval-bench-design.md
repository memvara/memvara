# bench/hosted.py — measuring retrieval against a real store

**Date:** 2026-09-01
**Status:** approved in discussion; this document is the written form
**Scope:** measurement only. This tool fixes nothing.

## Why this exists

Every retrieval number this repository publishes is measured against a corpus built for
the purpose — the Agent Memory Benchmark's structured facts, 2Wiki's entity chains,
LOCOMO's conversations. No number is measured against the store shape the hosted product
actually develops: per `docs/ROADMAP.md`'s own census, ~95% of a real store's claims use
predicates outside the declared vocabulary, the join rate is 0.5%, and the content is an
annotated log rather than a fact graph. Tuning against the benchmarks and hoping it
transfers is exactly the mistake the benchmarks were built to prevent.

Meanwhile the recall hook — the surface a user meets on every prompt — was shipped with
its measurement loop deliberately open. `plugin/hooks/recall.py` logs prompt and
injected memories to `recall-sample.log` so that, in its own words, a human can be the
judge. Nothing closes that loop. On this machine's log the open loop is visible as a
concrete failure: a prompt with no connection to the store ("write a haiku about rain")
still receives four injected memories, every time, because `memory_recall` returns no
scores and the hook has no floor — there is no abstention mechanism on that path at all.

`bench/hosted.py` closes the loop: a probe suite a store's owner authors once and runs
whenever they want a number.

## What is being measured

The **core read path** — `search()` and `recall()` — which every surface shares: the
recall hook, the MCP tools, the REST facade, and library callers. The hook contributes
one surface-specific behaviour (inject-or-not is binary, scoreless) that is measured
*through*, so a core ranking defect and a surface gating defect show up as different
numbers rather than one blended complaint.

Four metrics per run:

`k` defaults to 4 — the hook's own `K` — and `--k` overrides it. `ambiguous` probes
score exactly as `hit` probes do and are reported as their own row, never folded in:
their gold is a judgment, not a fact, and the two must stay tellable apart.

| metric | probe class | question it answers |
|---|---|---|
| hit@k | `hit`, `ambiguous` | does the gold claim appear in the top k the hook would see? |
| gold-rank | `hit`, `ambiguous` | where? (moves before hit@k does; the leading indicator) |
| false-injection rate | `abstain` | on a query the store cannot answer, would anything be injected? |
| self-retrieval@1 | `verbatim` | does a claim's own text return that claim first? |

Two clarifications that keep the numbers honest:

- **False-injection rate reads 100% at baseline, by construction.** The hook always
  injects K=4 and `search()`'s `min_score` defaults to 0.0. That is the point: the
  metric exists so an abstention mechanism, when one is designed, has a number to move.
  Each abstain failure also records the top score, so every run reports the headroom —
  "a floor at 0.35 would have silenced 4 of 6" — the way the same analysis was done on
  the Agent Memory Benchmark corpus in memvara/memvara#129.
- **self-retrieval@1 exists because of a recorded production defect**: the published
  relevance score blends confidence with similarity, and a verbatim-text query for a
  low-confidence claim was measured returning a different, higher-confidence claim
  first, in both directions of an A/B swap. This metric turns that note into a
  regression number.

Latency (p50/p95 per probe class) is recorded as a by-product, not a target.

## Probes are private; the harness is not

`hit` and `verbatim` probes quote the store's own content, so probes cannot live in the
repository. The repo ships the tool, the schema, a fixture suite for tests, and a
documentation section; the probe file lives with the store owner.

- Default path `~/.memvara/probes.jsonl`, overridden by `--probes PATH`.
- One JSON object per line:

```jsonl
{"id": "p001", "class": "hit",      "query": "what suite must run with -j1?",  "gold": ["<claim-id>"]}
{"id": "p002", "class": "abstain",  "query": "write a haiku about rain",       "gold": []}
{"id": "p003", "class": "verbatim", "query": "<the claim's own object text>",  "gold": ["<claim-id>"]}
{"id": "p004", "class": "ambiguous","query": "<a real prompt from the log>",   "gold": ["<claim-id>"], "judged": "2026-09-01"}
```

`gold` is a list of claim ids: `hit`/`verbatim`/`ambiguous` score a hit if any listed id
appears; `abstain` requires the empty list and scores a failure if anything at all comes
back above the injection surface. `ambiguous` rows carry `judged` (the date the human
judgment was made) because a judgment ages as the store changes.

### Authoring helpers

Two subcommand-style flags, both write-nothing-without-review:

- `--draft N` samples N live claims from the configured store and emits skeleton `hit`
  and `verbatim` probes on stdout — the owner edits the queries into how they would
  actually ask, then appends to their probe file. Drafted queries are deliberately not
  auto-usable: a query generated from the claim's own text measures lexical echo, which
  is the bias this tool exists to escape. The draft marks each row `"draft": true`, and
  the runner refuses rows still carrying the mark.
- `--seed-from-recalled DIR` closes the loop the hook left open. It samples real recall
  events (`~/.memvara/.hooks/recalled/`, 1,052 on this machine), dumps blinded
  query/result pairs in the `bench/evalkit.FileReader` dump/answers shape, and on the
  second pass (`--answers PATH`) converts the judgments into `ambiguous` probes. The
  same blinding discipline applies and the same caveat is printed: judged rows are a
  sanity anchor, not a reproducible measurement.

## How it queries

The configured store, through the public read path:

- **Local** (`MEMVARA_DB`): a `Memvara` over the file, `search()` for scored results and
  `recall()` for the rendered-text behaviour the hook sees.
- **Hosted** (`MEMVARA_MODE=cloud` / credentials file): `RemoteMemvara`, same two calls.
  One probe at a time is exactly the hook's own traffic shape, so the no-bulk-API
  limitation that ruled the HTTP client out for mass ingestion does not apply here.

Every result file records a **store fingerprint**: claim count, embedder identity (from
the store's own fingerprint), server version where hosted, and the run timestamp. The
store drifts between runs; comparing two result files with different fingerprints prints
a warning naming what moved. There is no attempt to pin `known_at` in v1 — a probe suite
is re-judged when the store changes character, and pretending otherwise would dress a
sanity check as a time series.

## Output

The table format the other `bench/` tools use, plus `--out PATH` writing per-probe
JSONL: probe id, class, results (ids + scores), gold hit/miss, gold rank, top score,
latency. No pass/fail exit code in v1 — this is a measurement, and a gate needs a
baseline to be set against first.

## Testing

The scoring core is pure functions — `score_hit`, `score_abstain`, aggregation — taking
probe rows and result rows, no I/O. Tests run them against an in-memory `Memvara`
fixture with planted claims (the repo-shipped fixture suite), no network anywhere.

Noted constraint from this repository's own records: **`bench/` sits outside the
coverage gate**, so the tests in `tests/` are the only guard. Every scoring guard is
sabotaged before it is believed — break the thing it watches, watch it go red — per the
CLAUDE.md rule, and the drafted-probe refusal above gets the same treatment.

## Documentation

Ships in the same commit: a section in `docs/BENCHMARKS.md` stating what this measures,
what it deliberately does not (see below), and that probe files are private to a store.

## Non-goals, stated so they stay decisions

- **No fixes.** No abstention mechanism, no ranking change, no `min_score` default
  change — those are separate work, each judged against the baseline this produces.
- **No LLM anywhere.** Authoring is human; scoring is id matching.
- **No published numbers.** Results are per-store and private by nature; nothing here
  lands in README or BENCHMARKS.md as a memvara score.
- **No CI wiring in v1.** A per-store measurement has no place in the repo's CI; if a
  fixture-store variant proves useful as a regression gate later, that is its own
  change.

## Open questions carried, not blocking

- Whether `recall()`'s rendered-text surface should learn to expose ids/scores so the
  hook could gate — that is a core API question the baseline will inform, and it belongs
  to the abstention work, not here.
- Whether `ambiguous` probes should expire (`judged` date + store drift) automatically
  or by warning only. v1 warns.
