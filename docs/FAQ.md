# FAQ

## What is Memvara?

A memory layer for AI agents that stores facts as **claims** — `(subject, predicate,
object)` triples — each carrying two independent dates: when it was true in the world, and
when this store came to believe it. That is what "bitemporal" means, and it is what makes
*where did Alice live in March* a lookup rather than a guess.

It is a Python library (`pip install memvara`), an MCP server (`memvara-mcp`), and a
hosted service. The library is the product; the hosted service runs this same code.

## Why not just use a vector database?

You can, and Memvara uses vector search internally. What a vector database does not have
is a notion of one fact **replacing** another.

Store *"invoices go to Coldharbour Road"* in January and *"invoices go to Bramble
Cottage"* in March, and both chunks are in the index, both are relevant to *where do
invoices go*, and nothing says the second replaced the first. Retrieval returns whichever
embeds closest to the phrasing of the question. That is not a ranking failure you can tune
away — the information that would fix it was never stored.

Memvara's March write closes the January claim's interval, so the old value is out of the
live set by construction and still there when you ask about February.

## Is Memvara a replacement for RAG?

No. They answer different questions and the useful arrangement is both — RAG answers from
your corpus, memory supplies the state the corpus does not know. A policy document should
stay a document; a customer's current plan should be a slot with one value and a history.
[RAG and memory](concepts/rag-vs-memory.md) has the composition, including where to put
the seam.

## What makes memory *bitemporal*?

Two clocks instead of one.

- **Valid time** — the interval the fact held in the world.
- **Transaction time** — when this store believed it.

A correction that arrives in August about June has an August transaction time and a June
valid time. With one column you record either when you learned it or when it was true, and
either choice makes a class of question unanswerable. All three reads are available:

```python
mem.get_all(valid_at=T)   # what we believe today about how the world was at T
mem.get_all(known_at=T)   # what we believed at T, about the world as it is now
mem.get_all(as_of=T)      # both — what we believed at T, about T
```

## How are contradictions handled?

Deterministically, with no model. Normalise the predicate (`lives_in`, `resides_in`,
`based_in` are one slot), fold the entity (`Acme Corp.` and `ACME, Inc.` are one
employer), look up `(subject, predicate)` in an index, and if the predicate is declared
single-valued, close the interval of what is there.

Same two facts, same result, every run. The alternative — embed, retrieve neighbours, ask
a model whether they conflict — can miss (the conflicting value need not be in top-k) and
is not repeatable.

One guard: a low-confidence extraction that would displace something somebody stated
outright is kept **beside** it and reported on `WriteReceipt.disputed`, rather than
silently winning. See [contradiction resolution](concepts/contradiction-resolution.md).

## Does every memory write require an LLM?

No, and this is the design claim rather than an optimisation.

- **`remember()` never calls one.** You hand over the triple; there is nothing to parse.
  Contradiction resolution, retrieval, the two clocks, provenance and consolidation never
  needed a model either.
- **`add()` on prose runs three model-free tiers first** — hash dedupe, near-duplicate
  detection, a salience gate, and a rule-based extractor — and batches whatever survives
  into a single extraction call. A predicate the registry has not seen before costs a
  second one, for acquisition.
- **With no `llm=` configured there is no model tier at all.** Turns the rules do not
  recognise are dropped, counted on `WriteReceipt.unextracted`, and the constructor warns
  once. That is the honest limit on the offline claim: the library runs with no API key,
  extraction from arbitrary prose does not.

`WriteReceipt.llm_calls` reports the cost on every write, so the claim is checkable.

## Can I inspect memory history?

Yes, and nothing is deleted to make it possible.

```python
[(c.object, c.state) for c in mem.history("Alice", "lives_in")]
# [('Berlin', 'ended'), ('Oslo', 'retired'), ('Lisbon', 'live')]

mem.why(claim_id)          # the source turns, the extractor, and what it replaced
mem.produced(episode_id)   # the same link, backwards
mem.since(when)            # what changed while you were away
mem.ask(question, at=T)    # narrated: what is true now, what was true then,
                           # and what this store would have SAID then
```

**`ended` and `retired` are different events.** Berlin was true and the world changed;
Oslo was never true and the record was wrong. A store with one "deleted" flag reports the
same thing for both, and *served a value that expired* is a different incident from
*served a value that was never true*.

For actual deletion — where the text itself must cease to exist, as a GDPR Article 17
request requires — `erase()` and `purge()` are separate calls, and `prove_erased()`
re-queries every table the content could survive in.

## Can I use Memvara with coding agents?

Yes, and it is one of the cases the design is aimed at. Install the MCP plugin
(`/plugin marketplace add memvara/claude-memvara`) and the agent gets fourteen tools plus
a packaged skill telling it when to write and how to correct.

The one piece of setup that is not optional is vocabulary: the built-in predicates are a
personal-assistant set, and an engineering store matches none of them. Load the shipped
packs — `MEMVARA_PREDICATES=engineering,decisions` — or nothing supersedes. See the
[coding-agents guide](guides/coding-agents.md) and
[`examples/coding_agent.py`](../examples/coding_agent.py).

## Is Memvara open source?

**Apache-2.0, for everything in this repository** — the engine, the store, the retrieval
stack, the MCP server, the framework adapters, the benchmarks and the tests. There is no
open-core-with-holes arrangement inside it: nothing here is crippled to sell an upgrade.

The REST API and the hosted control plane are the commercial half and are not in this
repository. [Open core](OPEN-CORE.md) says exactly where the line is and why it does not
move — including why `memvara/store/remote.py` raises `NotImplementedError` in the places
it does rather than quietly writing through a facade that would reinterpret every field.

## What are the honest limitations?

The list is longer than most projects publish and it is on its own page,
[Limitations](LIMITATIONS.md). The four that catch people first:

1. **The default embedder is lexical, not semantic**, and Latin-only — text in Han, Kana,
   Hangul, Arabic or Hebrew produces an all-zero vector and is never returned by meaning.
   `memvara[local-embed]` fixes it.
2. **Extraction from arbitrary prose needs a model.** See above.
3. **Entity resolution folds surface forms, not the world.** `Acme Corp` and `acme, inc.`
   collapse; `Big Blue` and `IBM` do not, unless you say so.
4. **The published LOCOMO and LongMemEval numbers are retrieval, not answer accuracy**,
   and must never be quoted as if they were. A judged answer-accuracy number
   now exists alongside them — 88.9% on a LongMemEval-S sample, through the shipped
   0.11.0 read path — and it is a separate measurement, on a different sample, made with
   a reader. See [Answer accuracy, judged, in the MemoryBench harness](BENCHMARKS.md#answer-accuracy-judged-in-the-memorybench-harness).

## How do I get started?

```bash
pip install memvara
```

Then pick one:

- **Five minutes with the API** → [Quickstart](getting-started/quickstart.md)
- **See it run** → [`examples/temporal_memory.py`](../examples/temporal_memory.py)
- **Memory in your editor** → [MCP](integrations/mcp.md)
- **Understand it first** → [Why Memvara?](concepts/why-memvara.md)

---

Previous: [Documentation index](README.md) · Next: [Quickstart](getting-started/quickstart.md)
