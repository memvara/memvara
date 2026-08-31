# The Agent Memory Benchmark

**Can an agent remember information that changes?**

Most evaluations of AI memory ask whether the right sentence comes back. That question is
largely answered. The one underneath it is not, and it is the one that produces the
failures users actually report: an agent that confidently tells you something that stopped
being true, and cites its own memory as the reason.

This benchmark measures that directly. It is offline, deterministic, and reproducible from
a clone in about a second per system. Every question, every gold answer and every scoring
rule is in a file you can read.

```bash
git clone https://github.com/memvara/memvara && cd memvara && pip install -e .
python -m benchmarks.agent_memory --system memvara --system naive --system vector-rag --compare
```

---

## The problem

A memory system is told three things over five months:

```
2026-01-10   Alice lives in Berlin.
2026-03-15   Alice relocated to the London office.
2026-06-02   Alice has moved to New York.
```

Ask *where does Alice live?* and almost any memory layer answers **New York**. Ask any of
these and most of them cannot:

- Where did Alice live on 20 March?
- Which cities has she lived in?
- On what date did she move to London?
- When did *we* find out?
- Which report caused us to believe London?
- What would we have told you in April?

None of those is exotic. They are the questions a support agent, a coding agent and an
audit all need answered, and a store that keeps a current value with an `updated_at`
column cannot answer any of them. Worse, it does not fail visibly: it returns the current
value, and the answer looks like an answer.

The failure gets sharper when news arrives late. Project Atlas moved region on 1 March;
the deployment log was imported on the 10th. There is a nine-day window in which *what was
true* and *what was known* are different answers, and a system with one clock has to give
the same answer to both. Which one it gives decides whether an incident review is correct.

## What is tested

Ten question categories, grouped into seven dimensions. There is no weighted composite
score: a single number mixing temporal reasoning with retrieval hides which half a system
is bad at, which is the only thing worth knowing. The dimensions partition the categories,
so their totals add up and a reader can check the arithmetic.

| Dimension | Categories | The question behind it |
|---|---|---|
| current_state | `current_state` | What is true now? |
| temporal | `historical_state`, `change_time`, `change_detection` | What was true then, when did it change, what has it been? |
| knowledge_time | `knowledge_time` | When did we find out — and what would we have said that day? |
| contradiction | `contradiction` | Two reports about one instant: which stands, and what happens to the other? |
| provenance | `provenance` | Which source caused this belief? |
| retrieval | `multi_hop`, `distractor` | Can the right memory be found among hundreds? |
| irrelevance | `negative` | Does it say nothing when it knows nothing? |

Cost and latency are measured alongside, and reported only where a system actually counts
them. An unmeasured quantity prints as `-`, never as `0`.

## The dataset

`benchmarks/agent_memory/datasets/v1/` — **262 events, 100 questions, 16 scenarios, 66
entities**. Entirely synthetic: no real person, conversation, credential or system appears
in it, and it is safe to publish in full.

Roughly half the scenarios are personal facts and half are engineering facts, because
coding agents are a memory system's hardest customer and their facts change fastest: a
service migrates its datastore, an authentication strategy is replaced, a deployment moves
region, a team changes lead.

Scenarios are chosen for the shapes that break stores, not for variety:

- **Progression** and **reversion** — Charlie leaves Berlin for London and comes back;
  Acme Corp upgrades to enterprise and downgrades again. The middle period is the question.
- **Delayed knowledge** — the fact becomes true before anyone records it.
- **Same-instant contradiction** — a colleague says London, and three days later Dana says
  Paris *about the same date*. That is a correction, not a move, and the difference decides
  whether London is a city Dana has lived in.
- **Repeated observation** — one fact restated four times, then a real change. Five writes,
  two values.
- **Multi-valued relations** — a later value that must join the earlier ones rather than
  replace them.
- **Distractors and filler** — 210 unrelated facts, and four of five people in one scenario
  living in the same city.
- **Hard negatives** — an entity the system has heard of, asked about a fact it was never
  told.

### Every gold answer follows four published rules

The rules are stated so that a system can implement them, and so a reader can check a gold
answer without trusting anyone:

1. **Later valid time wins.** The earlier value is *ended*: no longer in force, still
   answering about the period it held.
2. **At equal valid time, later record wins.** The earlier value is *retired*: we stopped
   believing it, so it answers nothing about the world.
3. **An ending is a belief, and it is dated.** A value stops being in force only from the
   moment its successor was *recorded*.
4. **Repeating a value is not a change.**

Rule 3 is the one a single-clock store cannot express, and rules 2 and 3 together are most
of the difference the results show.

Nothing is scored on source confidence, though it is carried in the data. Resolving a
conflict by reliability is a *policy*, and scoring one would measure whether a system had
implemented this benchmark's policy rather than whether it had a memory.

## Methodology

**Accuracy is `correct / total`**, per category and overall. No partial credit: an agent
that half-remembers a customer's plan tells the customer something false.

**Scoring is deterministic. There is no LLM judge, in any mode.** Four answer kinds, one
rule each:

- **Value** — normalized equality with the gold or a published alias, so formatting never
  costs a point. Plus *unambiguous containment*: the gold may appear inside a short answer
  provided no competing value for the same slot also does. `"She lived in London."` counts;
  `"Berlin, then London"` does not. `--match strict` requires equality.
- **Set** — exact set equality, order ignored. Not overlap, so "name everything" cannot win.
- **Date** — same calendar day, ISO-8601 only.
- **None** — the system must return nothing.

**No system is handed a gold answer.** The runner passes adapters an object that has no
field capable of carrying one; this is enforced by the code, not promised in a document.

**Every system gets identical input in identical order**, including the predicate schema —
whether a relation holds one value or many is published with the dataset rather than left
for each system to guess.

**A published rule available to one system is available to all.** The correction rule is
implemented in the memvara adapter and in the strongest baseline. A rule one system had
and the others were denied would be the benchmark rigging itself.

## Systems

| Name | What it is |
|---|---|
| `naive` | A dictionary of current values, overwritten on each write, with the source kept beside it. What most agent memory actually is. |
| `vector-rag` | Retrieval over the whole write log, with **one clock**: it keeps every observation and answers a question about a past instant with the most recent write it had received by then. The strongest baseline that is not bitemporal. |
| `memvara` | This repository's library through its public API — `remember()`, `history()`, `search()`, `why()`. |

Neither baseline is built to lose. `vector-rag` is completely correct on current state,
provenance, change time and knowledge time.

## Results

Measured at the commit this document landed in. Python 3.13.14, macOS arm64, `numpy`
2.5.1, memvara 0.9.0, no model and no network. Reproduce with the command at the top.

| System | Overall | current_state | temporal | knowledge_time | contradiction | provenance | retrieval | irrelevance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| memvara 0.9.0 | **92.0%** | 100.0% | **100.0%** | 100.0% | 100.0% | 100.0% | 64.3% | 50.0% |
| vector-rag 1.0 | 89.0% | 100.0% | 91.5% | 100.0% | 100.0% | 100.0% | **71.4%** | 50.0% |
| naive 1.0 | 50.0% | 100.0% | 34.0% | 42.9% | 60.0% | 54.5% | 64.3% | 50.0% |

By category:

| Category | memvara | vector-rag | naive | n |
|---|---:|---:|---:|---:|
| current_state | 100.0% | 100.0% | 100.0% | 10 |
| historical_state | **100.0%** | 85.2% | 33.3% | 27 |
| change_detection | 100.0% | 100.0% | 27.3% | 11 |
| change_time | 100.0% | 100.0% | 44.4% | 9 |
| knowledge_time | 100.0% | 100.0% | 42.9% | 7 |
| provenance | 100.0% | 100.0% | 54.5% | 11 |
| contradiction | 100.0% | 100.0% | 60.0% | 5 |
| multi_hop | 16.7% | **33.3%** | **33.3%** | 6 |
| distractor | 100.0% | 100.0% | 87.5% | 8 |
| negative | 50.0% | 50.0% | 50.0% | 6 |

Cost and latency from the same runs:

| System | LLM calls | texts embedded | rows stored | write, per event | query, mean | query p95 |
|---|---:|---:|---:|---:|---:|---:|
| memvara | 0 | 520 | 241 | 0.8 – 2.9 ms | 0.9 – 2.6 ms | 2 – 15 ms |
| vector-rag | 0 | 279 | 262 | 0.02 – 0.25 ms | 0.09 – 0.51 ms | 0.3 – 1.5 ms |
| naive | 0 | 0 | **193** | 0.007 – 0.016 ms | 0.02 – 0.03 ms | 0.1 – 0.2 ms |

**memvara embeds 520 texts against `vector-rag`'s 279** — texts submitted, not requests made, which is the same quantity for both only because neither batches. The split is exact: 241
claims plus 262 source episodes on the way in, then one per unprobed question. It embeds
the claim *and* the turn the claim came from, which is what makes `why()` able to answer
later and is the second cost the row above is the first half of. Until this was measured
the column read `-` for memvara, so the system doing the most embedding was the one with
no figure.

**Ranges, not points, and wide ones.** These span every run taken on one laptop that was
also doing other work, and the same figure moved by about 3× between runs depending on
that load. Quoting a single number from this table would be false precision.

**What is stable is the ordering**, which holds in every run: `naive` is fastest on both
axes, then `vector-rag`, then memvara — which is doing SQLite, embedding and
reconciliation on every write, and hybrid retrieval with a graph leg on every unprobed
read. memvara is roughly two orders of magnitude slower per write than a dictionary. That
is the trade the rest of this table is the other half of, and it is a real cost rather
than a rounding error. Reproduce it on your own machine before quoting anything.

**The `rows stored` column is the price of answering about the past, and it reads in the
right direction only if all three numbers mean the same thing.** They did not until this
was corrected: memvara reported rows while the baselines reported write *calls*, so the
table said the dictionary stored more rows than the bitemporal store. It stores fewer.

- **naive holds 193 rows** for 262 events, because it overwrites. Those 193 are exactly
  the current values and nothing else.
- **memvara holds 241** — the same 193 live, plus 46 *ended* (they stopped being true) and
  2 *retired* (the record was wrong: the hearsay about Dana, and the standup note about
  the quotes service). So **48 extra rows, a 25% overhead, is what the entire `temporal`
  and `contradiction` result is bought with.** The two-against-forty-six split is itself
  the distinction those categories score.
- **vector-rag holds 262**, one per observation: it keeps everything, including the four
  restatements of the billing service's datastore that say nothing new.

memvara's write receipts report 241 added and 21 reinforced — a restatement of something
already believed bumps the existing claim rather than adding a row, which is why it holds
fewer rows than `vector-rag` while remembering strictly more.

### What the cost columns do not measure

Named because an absent metric reads as a zero one, and because two of these were asked
for and are missing rather than declined:

- **Storage operations** — SQL statements, index writes, page reads. No system reports
  them and none can be made to through a system-neutral interface: they live below every
  adapter's public API, and instrumenting one system's internals would produce a column
  only that system could fill. `db_reads` is a different thing and says so — read calls
  the *benchmark* made, one per question, identical for every system.
- **Wall-clock cost of the model** — no shipped system uses one, so `llm_calls` and
  `tokens` are `0` for all three. Those are measurements rather than blanks: zero calls is
  zero tokens. The fields exist for adapters that do use a model, which
  [the contributor guide](https://github.com/memvara/memvara/blob/main/benchmarks/agent_memory/CONTRIBUTING.md)
  requires to disclose the model, version and temperature.
- **Memory footprint** — bytes on disk or in RSS. Not measured, and not comparable
  between an in-process dictionary and a SQLite file without saying which is being
  counted.

## Interpretation

**Three points separate memvara from a baseline written in numpy in an afternoon.** That
is a narrower margin than the case for bitemporal memory would lead you to expect, and it
is the most useful number on the page. Read the rows rather than the total.

**The bitemporal advantage is real and it is narrow.** memvara scores 100.0% on the
temporal dimension against `vector-rag`'s 91.5%, and the four questions that separate them
are the four delayed-knowledge and correction scenarios. Everywhere else in the dataset the
two clocks coincide, and a single-clock store is exactly right. **The claim this supports is
not "bitemporal memory is better at remembering." It is "when news arrives after the fact,
a single clock has to give one answer to two different questions, and roughly nine per cent
of realistic temporal questions are that case."** Whether nine per cent matters depends
entirely on how often your facts arrive late — which, for logs, imports, backfills and
anything a human reports second-hand, is often.

**The gap against a current-value store is not narrow at all.** `naive` scores 33.3% on
historical state and 27.3% on change detection, against 100% for both other systems. Its
single largest failure mode, twenty questions, is `answered_current_state`: it gave today's
value to a question about the past, without hesitating. That is the failure this benchmark
was built to make visible, and it is what most agent memory does today.

**memvara loses on retrieval, and that is reported as measured.** 64.3% against
`vector-rag`'s 71.4%, and the whole of it is `multi_hop`, where memvara answers 1 of 6 and
is the **worst of the three systems**. The failure signature is unusually clear: asked
*"Which region is the project Alice works on deployed to?"* it answers `Project Atlas`,
and asked who leads the team that owns the checkout service it answers `team-payments`. It
finds the first hop and stops, because taking the top-ranked slot is all this adapter
does. memvara ships `neighborhood()` and `paths_between()` and this adapter uses neither.
A graph-aware adapter would very likely do better — nobody has measured that, so nothing
here claims it.

**One dimension currently discriminates nothing.** All three systems score 50% on
`irrelevance`, failing the same three questions the same way: asked about a fact they were
never told, each answers from the nearest match instead of abstaining. That row describes
a shared gap in the field rather than a difference between these systems, and until either
a system improves or the questions get harder it carries no information.

**These numbers replace an earlier set, and the reason is worth reading.** An earlier run
had memvara at 90.0% overall and 50.0% on `retrieval`. Investigating that loss found two
defects in the harness rather than in any memory system: the memvara adapter searched
ended and retired claims for present-tense questions, and the three adapters fed their
retrievers three different strings — so `retrieval` was comparing adapters rather than
retrievers. Both are fixed, both fixes helped **every** system (`retrieval` rose 7.2
points for `naive`, 7.1 for `vector-rag` and 14.3 for memvara), and memvara still loses
the category. No question, gold answer or scoring rule was changed. The benchmark's own
README records the diagnosis in full.

**memvara's zero LLM calls on the write path is true by construction here, not a finding.**
The benchmark hands every system a structured fact, so the adapter uses `remember()` and
there is nothing to extract. It is recorded so that a change introducing a model call would
be visible.

## Reproduce it yourself

```bash
git clone https://github.com/memvara/memvara && cd memvara
pip install -e .

# the table above
python -m benchmarks.agent_memory --system memvara --system naive --system vector-rag --compare

# every wrong answer, with the fact's real history beside it
python -m benchmarks.agent_memory --system memvara --show-failures

# prove the run is deterministic: two runs, identical verdicts
python -m benchmarks.agent_memory --system memvara --repeat-check

# machine-readable
python -m benchmarks.agent_memory --system memvara --output results.json
```

Accuracy is deterministic: the same commit and dataset give the same figures on any
machine, and `--repeat-check` asserts it. Latency is not, and is excluded from that check
because it measures the machine rather than the system.

Every result file records what produced it — benchmark version, dataset version, system
version, Python version, platform, dependency versions, git commit, and the run's
configuration. No environment variables, paths or credentials are collected; the
environment block is assembled from a fixed list of fields.

## Add your own system

One file, five methods, and no changes to the benchmark:

```bash
python -m benchmarks.agent_memory --system mypackage.adapters:build
```

[The contributor guide](https://github.com/memvara/memvara/blob/main/benchmarks/agent_memory/CONTRIBUTING.md)
has the interface and the rules. Adapters that beat memvara in a category are welcome and
their rows go into the table as measured — there is already one in it.

## Limitations

The reasons to discount the numbers above, in the order we think they matter.

1. **Extraction is out of scope, and this is the largest limitation.** Every event carries
   a structured triple alongside its sentence, and every system gets both. That holds
   extraction quality constant so the benchmark measures what a system does with a fact once
   it has it — but it says nothing about whether a system could have got the fact out of the
   sentence, which for most deployments is where the errors are. A system with excellent
   temporal reasoning and a useless extractor scores well here and would fail in production.
2. **The corpus is authored by the maintainers of one of the systems under test.** The
   scenarios are what agent memory gets wrong, and they are also what memvara was built for.
   The mitigation is partial: golds are derived from published rules, the baselines
   implement every rule memvara's adapter does, and every question is readable. A benchmark
   authored by an interested party is worth less than one that is not, whatever its
   methodology. This is the reason to run it against your own system rather than to read
   the table.
3. **One dimension does not discriminate.** All three systems tie at 50% on
   `irrelevance`.
4. **The vector baseline uses hashed TF-IDF, not a neural embedder.** That buys no API key
   and byte-identical runs, and costs paraphrase robustness. A sentence-transformer baseline
   would likely score higher on `distractor` and `multi_hop`. Unmeasured.
5. **262 events is small.** Large enough that retrieval is not trivial, small enough to run
   in a second, and silent about behaviour at a million memories.
6. **Answers are values, not prose.** A real agent reads memory and writes a sentence, and
   nothing here measures that step.
7. **Latency is one machine, single-process, no concurrency.** An order of magnitude, not a
   number.
8. **Date answers must be ISO-8601.** A system answering "the 15th of March" is marked
   wrong for format.
9. **No question asks about a value a slot held twice**, because reversion makes such a
   question ambiguous — the loader refuses one rather than shipping two defensible answers.
   Reversion itself is tested, through questions about the middle period.

## Versioning

The dataset and the methodology are treated as an API. This is **Agent Memory Benchmark
v1.0**, dataset **v1**. Any change that could move a published score — a question added or
reworded, a gold answer changed, a matching rule changed, a supersession rule changed — is
material and produces a new version, with the old one left in place. Adding a system is not
material. Changes are recorded in `CHANGELOG.md`.

## The result schema

Documented so a benchmark page, a leaderboard or a regression check can read a run without
reading the code. One JSON object per run:

| Field | Meaning |
|---|---|
| `benchmark` | always `"agent-memory"` |
| `benchmark_version` | the methodology version, e.g. `"1.0"` |
| `dataset_version` | e.g. `"v1"` |
| `system`, `system_version` | the adapter's name and the version of the system it drives |
| `timestamp` | UTC ISO-8601 |
| `counts` | `events`, `questions`, `scenarios` actually run |
| `config` | `match`, `categories`, `limit`, `questions_asked` |
| `environment` | `python`, `implementation`, `platform`, `machine`, `git_commit`, `numpy`, `memvara` |
| `metrics.overall` | `{correct, total, accuracy}` |
| `metrics.by_category` | the same, keyed by category |
| `metrics.by_dimension` | the same, keyed by dimension — the leaderboard columns |
| `metrics.by_scenario` | the same, keyed by scenario |
| `metrics.failure_reasons` | reason to count, most common first |
| `latency` | `write_total_ms`, `write_mean_ms`, `query_mean_ms`, `query_p50_ms`, `query_p95_ms`, `query_max_ms` |
| `usage` | `llm_calls`, `tokens` (prompt and completion together), `texts_embedded` (texts submitted for embedding, not requests made), `rows_stored` (rows the store holds after ingestion, not write calls), `db_reads`, `extra`. **`null` means not measured** and must not be rendered as zero |
| `questions[]` | per question: `id`, `category`, `scenario`, `correct`, `given`, `expected`, `reason`, `latency_s`, `support` |

`accuracy` is `null` for an empty group, which a renderer must show as `-` rather than 0%.

Adding a field is safe. Renaming or repurposing one breaks any consumer reading it, so
it needs a deprecation window once results are being consumed — but it is **not** what
`benchmark_version` tracks. That number is about whether two *scores* are comparable, and
it moves when the questions, the scoring or the dataset move. A field rename changes no
score.

One rename has happened, and it is recorded here rather than left for a reader to
discover: `embedding_calls` became `texts_embedded`, because memvara reported texts
through it and `vector-rag` reported requests, and the two agreed only because neither
batches yet. It was renamed within hours of the schema first being published and before
anything consumed it. A later one would not get that treatment.

The full methodology, the layout and the fairness rules are in
[the benchmark's own README](https://github.com/memvara/memvara/blob/main/benchmarks/agent_memory/README.md).

---

Previous: [Benchmarks](../BENCHMARKS.md) · Next: [Limitations](../LIMITATIONS.md) · [Documentation index](../README.md)
