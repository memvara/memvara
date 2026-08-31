# Agent Memory Benchmark

A reproducible test of whether a memory system gets **changing** facts right.

```bash
python -m benchmarks.agent_memory --system memvara --system naive --system vector-rag --compare
```

Offline, no API key, no download, about a second per system. The dataset is committed
beside the code and every gold answer is derived from published rules.

---

## What this measures, and why it is not another retrieval benchmark

Most memory evaluations ask whether the right sentence comes back. That is a real
problem and it is largely solved. The problem underneath it is not:

> Alice lived in Berlin. In March she moved to London. In June she moved to New York.
> **Where did Alice live on 20 March?**

A store that keeps the current value answers *New York*, confidently, and there is no
error to see. The same store cannot say when she moved, when it found out, which report
caused the belief, or what it would have told you in April. An agent built on it will
tell a customer something false and cite itself.

So the questions here are about **state over time**:

| Category | Question it asks | Example |
|---|---|---|
| `current_state` | What is true now? | Where does Alice live now? |
| `historical_state` | What was true then, as we understand it today? | Where did Alice live on 2026-03-20? |
| `change_detection` | Which values has this slot held? | Which cities has Alice lived in? |
| `change_time` | When did the world change? | On what date did Alice move to London? |
| `knowledge_time` | When did *we* find out — and what would we have said then? | On what date did the system learn Atlas had moved? |
| `provenance` | Which source caused this belief? | Which source reported that Alice lives in London? |
| `contradiction` | Two reports about one instant: which stands? | Where does Dana live now? |
| `multi_hop` | An answer that needs two facts joined | Which region is the project Alice works on deployed to? |
| `distractor` | The same value held by many entities | Where does Ivan live? (four of five neighbours are in London) |
| `negative` | A fact the system was never told | Where does Oscar live? |

Ten categories roll into seven **dimensions**, and the dimensions partition the
categories — their totals add up to the overall total, so a reader can check the
arithmetic. There is no weighted composite index. A single number that mixes temporal
reasoning with retrieval hides the thing you want to know: which half a system is bad at.

---

## The dataset

`datasets/v1/` — **262 events, 100 questions, 16 scenarios, 66 entities**. Everything in
it is invented. No real person, conversation, credential or system appears anywhere, and
the whole corpus is safe to publish.

Three files, and one generator that produces them:

```
datasets/v1/events.jsonl      what each system is told, in the order it is told
datasets/v1/questions.jsonl   what it is then asked, with the gold answer
datasets/v1/metadata.json     the predicate schema, the dimension map, the counts
datasets/build_v1.py          regenerates all three; the test suite asserts it reproduces them
```

`events.jsonl` is not called `facts.jsonl` on purpose. It holds *observations* — things a
system is handed, one of which is wrong and several of which are superseded later. The
facts are in the gold answers, and conflating the two is how a memory benchmark ends up
scoring its own input.

### The scenarios

Personal and engineering facts in roughly equal measure, because coding agents are a
memory system's hardest customer and their facts change fastest.

- **Progression** — Alice: Berlin, London, New York. Bob: four job titles in eighteen months.
- **Reversion** — Charlie leaves Berlin for London and comes back. Acme Corp upgrades to
  enterprise and downgrades again. The middle period is the question.
- **Delayed knowledge** — Project Atlas moved region on 1 March; the log was imported on
  the 10th. The auth service cut over on 10 February; the RFC landed on the 24th. There is
  a window where what was true and what was known are different answers.
- **Same-instant contradiction** — a colleague says Dana moved to London; three days later
  Dana says Paris, about the same date. That is a correction, not a change, and the
  difference decides whether London is a city Dana has lived in.
- **Repeated observation** — the billing service is reported to use MySQL four times, then
  actually migrates. Five writes, two values.
- **Multi-valued** — Alice speaks three languages. A later value joins the earlier ones
  rather than replacing them.
- **Distractors** — four of five people in one scenario live in London.
- **Multi-hop** — Alice works on Project Atlas; Project Atlas is deployed to a region that
  changed in March.
- **Hard negatives** — Mallory is in memory with a birthplace and an editor, and no
  residence. Project Chronos does not exist.
- **Filler** — 210 unrelated facts about thirty people and fifteen services, never asked
  about, so that retrieval has to do some work.

### The supersession rules, published

Every gold answer follows from four sentences, stated in `timeline.py` and repeated here
because a scoring rule a system cannot read is one it cannot be expected to implement.

1. **Later valid time wins.** The earlier value is *ended*: no longer in force, still
   answering about the period it held.
2. **At equal valid time, later record wins.** That is a correction. The earlier value is
   *retired*: we stopped believing it, so it answers nothing about the world.
3. **An ending is a belief, and it is dated.** A value stops being in force only from the
   moment its successor was *recorded*. Rewind the belief clock before that and the value
   is still open-ended.
4. **Repeating a value is not a change.**

Nothing is scored on source confidence. It is in the data because the conflicting-source
scenarios are more honest with it than without, and it is deliberately not part of any
gold — resolving a conflict by reliability is a *policy*, and picking one here would score
systems on whether they had implemented this benchmark's policy.

---

## Evaluation

`correct / total`, per category and overall. No partial credit anywhere: an agent that
half-remembers a customer's plan tells the customer something false.

Four gold kinds, each with one rule:

- **`value`** — normalized equality with the gold or a published alias. Case, punctuation
  and articles are removed first, so `"london."` and `"London"` are the same answer.
  Additionally, *unambiguous containment*: the gold may appear as a whole phrase inside a
  short answer, provided **no competing value for the same slot also appears**. That is
  what lets `"She lived in London."` count and stops `"Berlin, then London"` from
  counting. `--match strict` turns containment off and requires equality.
- **`set`** — exact set equality on normalized values, order ignored. Not overlap:
  returning every value ever seen must not win `change_detection`.
- **`date`** — same calendar day, ISO-8601 only. Prose dates are not parsed, because
  parsing them needs a locale and a calendar and every one of those is a place for the
  benchmark to be quietly wrong.
- **`none`** — the system must return nothing. The only category where silence is correct.

**No LLM judge, in any mode.** Every rule above is deterministic, so two runs of the same
configuration produce identical scores. `--repeat-check` runs each system twice and
asserts it.

### When a system gets it wrong, the benchmark says why

`--show-failures` prints each wrong answer with the slot's real history and a named
reason from a closed list:

```
FAIL  q-alice-hist-mar   [historical_state]
  Question:  Where did Alice live on 2026-03-20?
  Expected:  London
  Answered:  New York
  Reason:    answered_current_state — gave today's value to a question about a past instant
  Timeline for alice / lives_in:
    2026-01-10  Berlin  [alice]
    2026-03-15  London  [hr_directory]
    ---------- asked about 2026-03-20 ----------
    2026-06-02  New York  [alice]
```

The timeline is rendered from the dataset, not from the system: it is what the system
should have been able to reconstruct. Reading it beside the answer is usually the whole
diagnosis.

---

## Systems

Three adapters ship. `--system` also accepts a dotted import path, so a memory system in
another repository is benchmarked without forking this one.

| Name | What it is |
|---|---|
| `naive` | A dictionary of current values, overwritten on each write, with the source kept beside it. What most agent memory actually is. |
| `vector-rag` | Retrieval over the whole write log, with **one clock**. Keeps every observation, indexes each sentence, and answers a question about a past instant with the most recent write it had received by then. The strongest baseline that is not bitemporal. |
| `memvara` | This repository's library, through its public API — `remember()`, `history()`, `search()`, `why()`. |

Neither baseline is a strawman. `vector-rag` gets current state, provenance, change time
and knowledge time completely right, and reconstructs history correctly wherever a fact was
recorded on the day it became true — which is most of the dataset.

---

## Results

Measured on this repository at the commit these files landed in, Python 3.13.14 on macOS
arm64, `numpy` 2.5.1, memvara 0.9.0. Reproduce with the command at the top of this file.

| System | Overall | current_state | temporal | knowledge_time | contradiction | provenance | retrieval | irrelevance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| memvara 0.9.0 | **92.0%** | 100.0% | **100.0%** | 100.0% | 100.0% | 100.0% | 64.3% | 50.0% |
| vector-rag 1.0 | 89.0% | 100.0% | 91.5% | 100.0% | 100.0% | 100.0% | **71.4%** | 50.0% |
| naive 1.0 | 50.0% | 100.0% | 34.0% | 42.9% | 60.0% | 54.5% | 64.3% | 50.0% |

By category, with the number of questions in each:

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

**Three points separate memvara from a baseline built out of numpy in an afternoon**, and
that is narrower than the pitch for bitemporal memory would suggest. Read the rows rather
than the total:

- **`temporal` is the whole of memvara's lead.** 100.0% against `vector-rag`'s 91.5%. The
  four questions `vector-rag` misses are the four delayed-knowledge and correction
  scenarios — where the news arrived after the fact, so "what was true then" and "what had
  we heard by then" are different answers and one clock has to give the same one to both.
  Everywhere else in the dataset the two clocks coincide and a single-clock store is
  exactly right.
- **memvara loses on retrieval**, 64.3% against `vector-rag`'s 71.4%, and it is now
  entirely `multi_hop`: memvara answers 1 of 6, which is the **worst of the three
  systems**. The failure signature is unusually clear — asked *"Which region is the
  project Alice works on deployed to?"* it answers `Project Atlas`, and asked who leads
  the team that owns the checkout service it answers `team-payments`. It finds the first
  hop and stops, because that is all this adapter does: take the top-ranked slot. memvara
  ships `neighborhood()` and `paths_between()` and this adapter uses neither. A
  graph-aware adapter would very likely do better; nobody has measured that, so nothing
  here claims it.
- **`irrelevance` is 50% for all three, so it currently discriminates nothing.** Every
  system answers the three open negative questions from the nearest match rather than
  abstaining. That is a real and shared failure — memvara's own `search()` documentation
  warns about exactly it — and a category where three systems tie is not yet earning its
  place. See *Limitations*.
- **`naive` scores 100% on current state.** It is not bad at memory. It is bad at *time*,
  and the benchmark's job is to say which.

### The harness had two defects, and they were found by investigating a loss

Worth stating plainly, because it is the kind of thing a benchmark's author is tempted to
fix quietly. An earlier run had memvara at 90.0% overall and 50.0% on `retrieval`. Asked
*"Where does Frank live?"*, memvara returned Alice — and the diagnosis was not weak
ranking. It was two mistakes in this harness:

1. **The memvara adapter searched the wrong population.** It passed
   `states=["live", "ended", "retired"]` for every question, so a present-tense question
   competed against values nobody holds any more. The correct claim was BM25 rank 0 and
   lost the fused ranking to `alice lives_in Berlin`, which Alice stopped holding in
   March. Present-tense questions now search live claims.
2. **The three adapters indexed three different strings.** `vector-rag` indexed subject,
   predicate and sentence; `naive` matched on the sentence alone; the memvara adapter
   passed the bare sentence as `Claim.text`, which is what memvara embeds and
   BM25-indexes. Events written in the first person then lost their subject entirely — *"I
   have relocated to Madrid"* does not contain the word *Heidi*, so no query naming her
   could reach it. All three now index the same string, built by one shared function
   (`adapters/base.py::indexable`), because retrieval is a scored dimension and three
   different inputs made it a comparison of adapters rather than of retrievers.

**Fixing them helped every system**, which is the part that matters for trusting the
result: `retrieval` went 57.1% → 64.3% for `naive`, 64.3% → 71.4% for `vector-rag`, and
50.0% → 64.3% for memvara. Overall went 49.0% → 50.0%, 88.0% → 89.0% and 90.0% → 92.0%.
**memvara still loses the category**, and it moved from tying on `multi_hop` to being last
on it. No question, gold answer or scoring rule was touched.

Cost and latency, measured in the same runs:

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

memvara's **zero LLM calls on the write path is true by construction here, not a finding**:
the benchmark hands every system a structured fact, so the adapter uses `remember()` and
there is nothing to extract. It is recorded so that a change introducing a model call
would be visible, not as evidence about extraction. An unmeasured counter prints `-`,
never `0`.

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

---

## Running it

```bash
# everything, all three systems, side by side
python -m benchmarks.agent_memory --system memvara --system naive --system vector-rag --compare

# fast enough to run while you work: 40 questions, spread across all ten categories
python -m benchmarks.agent_memory --system memvara --quick

# what went wrong, with the timeline beside each answer
python -m benchmarks.agent_memory --system memvara --show-failures

# one category
python -m benchmarks.agent_memory --system memvara --category historical_state --category knowledge_time

# machine-readable, for a leaderboard or a regression check
python -m benchmarks.agent_memory --system memvara --output results.json

# prove the run is deterministic
python -m benchmarks.agent_memory --system memvara --repeat-check
```

`python -m benchmarks.agent_memory.run` is the same command under a longer name, for
anyone who reaches for that spelling first. Both call the same entry point.

From a clone, with nothing installed but `numpy` (the two baselines) plus this repository
(the memvara adapter):

```bash
git clone https://github.com/memvara/memvara && cd memvara
pip install -e .
python -m benchmarks.agent_memory --system memvara --system vector-rag --compare
```

## Reproducing a published number

Everything that can move a score is recorded in the result JSON: benchmark version,
dataset version, system version, Python version, platform, `numpy` and memvara versions,
git commit, and the configuration the run used. Accuracy is deterministic — the same
commit and the same dataset give the same figures on any machine. Latency is not, and is
excluded from the reproducibility check, because it measures the machine.

## Benchmark fairness

Enforced by the code where the code can:

- **Same dataset, same questions, same order, for every system.** The runner has no
  per-system branch.
- **No system sees a gold answer.** The runner hands adapters an `Ask`, which has no field
  that could carry one.
- **No hardcoded answers.** Every adapter goes through its system's own API.
- **The predicate schema is published input, not a system's private advantage.** Whether a
  relation holds one value or many is declared in `metadata.json` and handed to every
  adapter's `reset()`.
- **A published rule is available to everyone.** The correction rule (rule 2) is
  implemented in the memvara adapter *and* in `vector-rag`; a rule one system had and the
  others were denied would be the benchmark rigging itself.
- **Which questions are easier is fixed by the dataset, not chosen by the system.** A
  question either names its fact slot for everyone or for nobody.
- **Shared discriminators and shared index text live in `adapters/base.py`**, so three
  adapters cannot read the same question three subtly different ways, and cannot feed
  their retrievers three different strings. Both had happened; see *The harness had two
  defects*.
- **Everything runs offline and deterministically.** No model, no judge, no network.
- **Changing the dataset or the methodology increments the version.** See *Versioning*.

Disclosed, because the code cannot enforce it:

- The memvara adapter is written by this repository's authors and is the best-informed of
  the three. A better `vector-rag` is possible and pull requests improving a baseline are
  as welcome as ones adding a system.
- `vector-rag`'s embedder is hashed TF-IDF, not a neural model. See *Limitations*.

## Adding your memory system

See [CONTRIBUTING.md](CONTRIBUTING.md). It is one file with five methods, and one line in
`registry.py` — or no line at all, if you point `--system` at your own import path.

## Versioning

The dataset and the scoring are treated as an API. `BENCHMARK_VERSION` is `1.0` and the
dataset is `v1`. A change that could move a published score is material and gets a new
version: adding or rewording questions, changing a gold answer, changing a matching rule,
changing the supersession rules. Adding a *system* is not material. Fixing a typo in a
`note` is not material. When `v1` changes materially it becomes `v2`, `v1` stays where it
is, and the change is documented in `CHANGELOG.md`.

## Limitations

The honest list. Several of these are reasons to discount a number above.

1. **Extraction is out of scope, and that is the largest limitation.** Every event carries
   a `(subject, predicate, object)` triple alongside its sentence, and every system gets
   both. This holds extraction quality constant so that what is measured is what a system
   does with a fact once it has it — but it means the benchmark says nothing about whether
   a system could have got the fact out of the sentence, which for most deployments is
   where the errors are. A system with excellent temporal reasoning and a useless extractor
   scores well here and would fail in production.
2. **The corpus is authored by the maintainers of one of the systems under test.** The
   scenarios were chosen because they are what agent memory gets wrong, and they are also
   what memvara was built for. That is a real bias and the mitigation is partial: the golds
   are derived from published rules, the baselines implement every rule memvara's adapter
   does, and every question is in a file you can read. A benchmark authored by an
   interested party is worth less than one that is not, whatever its methodology.
3. **The `irrelevance` dimension does not discriminate.** All three systems score 50%,
   failing the same three questions. Until either a system abstains or the questions get
   harder, that row carries no information.
4. **No adapter tested does multi-hop.** All three take the top-ranked slot and stop, so
   `multi_hop` scores between 16.7% and 33.3% across the board. That is a limitation of
   the adapters as written rather than a property of the systems — memvara ships
   `neighborhood()` and `paths_between()` and this adapter uses neither — and it is the
   most obvious thing a contributor could improve.
5. **`vector-rag`'s embedder is hashed TF-IDF, not a neural one.** It buys no API key and
   byte-identical runs; it costs paraphrase robustness. A sentence-transformer baseline
   would likely score higher on `multi_hop`. The questions are not written to defeat
   lexical matching, so the gap is probably small, but it is unmeasured.
6. **262 events is small.** It is large enough that retrieval is not trivial and small
   enough to run in a second. It says nothing about behaviour at a million memories.
7. **Answers are values, not prose.** A real agent reads memory and writes a sentence, and
   nothing here measures that step. `demo/` in this repository is the harness for that
   question and it is a different one.
8. **Latency is measured on one machine, single-process, with no concurrency.** Treat the
   numbers as an order of magnitude.
9. **Date answers must be ISO-8601.** A system that answers `"the 15th of March"` is marked
   wrong for format. This is stated up front rather than papered over with a parser.
10. **No question asks about a value a slot held twice.** Reversion makes "when did X move
    to London" ambiguous when London occurs twice, and `dataset.validate` refuses such a
    question rather than shipping one with two defensible answers. Reversion itself *is*
    tested — through the questions about the middle period.

## Layout

```
benchmarks/agent_memory/
├── README.md            this file
├── CONTRIBUTING.md      adding a memory system
├── dataset.py           the data model, loading, and validation
├── timeline.py          the supersession rules, and the golds derived from them
├── normalization.py     how an answer is compared
├── scoring.py           correct/total, and the named failure reasons
├── results.py           the result schema and the environment block
├── report.py            the tables and the failure report
├── runner.py            the run loop
├── registry.py          --system NAME, and dotted import paths
├── cli.py               the command line
├── run.py               `python -m benchmarks.agent_memory.run`, an alias for the above
├── adapters/
│   ├── base.py          the interface, and the rules every adapter shares
│   ├── naive.py         a dictionary of current values
│   ├── vector_rag.py    full write log, vector retrieval, one clock
│   └── memvara_adapter.py
└── datasets/
    ├── build_v1.py      the generator
    └── v1/              the committed dataset
```

Tests are in `tests/test_agent_memory_bench.py`, with the rest of this repository's suite.
