<h1 align="center">Memvara</h1>

<p align="center">
  <b>Bitemporal memory for AI agents.</b><br/>
  Know what was true. Know when it was true. Know why you believe it.
</p>

<p align="center">
  <a href="https://pypi.org/project/memvara/"><img alt="PyPI" src="https://img.shields.io/pypi/v/memvara?color=2b6cb0"></a>
  <a href="https://pypi.org/project/memvara/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/memvara"></a>
  <a href="https://github.com/memvara/memvara/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <a href="https://github.com/memvara/memvara/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/memvara/memvara/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://memvara.dev"><img alt="Site" src="https://img.shields.io/badge/site-memvara.dev-1a202c"></a>
</p>

<p align="center">
  <a href="https://github.com/memvara/memvara#the-90-second-demo"><b>90-second demo</b></a> ·
  <a href="https://github.com/memvara/memvara#quickstart">Quickstart</a> ·
  <a href="https://github.com/memvara/memvara/blob/main/docs/README.md">Documentation</a> ·
  <a href="https://pypi.org/project/memvara/">PyPI</a> ·
  <a href="https://memvara.dev">memvara.dev</a> ·
  <a href="https://github.com/memvara/memvara/issues">Issues</a>
</p>

```python
mem.remember("Alice", "lives_in", "Berlin",   valid_from=jan)
mem.remember("Alice", "lives_in", "London",   valid_from=mar)
mem.remember("Alice", "lives_in", "New York", valid_from=jun)

[c.object for c in mem.get_all()]              # ['New York']   where now?
[c.object for c in mem.get_all(as_of=mar_20)]  # ['London']     and on 20 March?
[c.object for c in mem.get_all(as_of=jan_20)]  # ['Berlin']     and on 20 January?
```

Three questions, three different correct answers, no model call. A store that keeps one
value per fact answers the first and gets the other two wrong — not by hallucinating, but
because overwriting Berlin destroyed the only record that Berlin was ever the answer.

```bash
pip install memvara
```

**numpy and nothing else.** Runs offline, no API key, no Docker, no vector database.

| | |
|---|---|
| **Bitemporal** | Two time axes that move independently: when a fact was true, and when this store was told. |
| **Deterministic** | A contradiction in a predicate declared single-valued is resolved on write by a keyed lookup, not by a model's judgement. |
| **Auditable** | A claim carries the episodes cited for it and the claim it superseded, and `why()` returns both. |
| **Historical** | `search`, `get_all`, `history` and five more reads take `as_of=`, `valid_at=` or `known_at=`, so *what was true then* is a query rather than a reconstruction. |
| **LLM-light** | `remember()` never calls a model. `add()` filters with model-free tiers first and sends whatever survives in one extraction call. |

---

## The problem

AI agents don't just forget. **They remember the wrong thing.**

A customer says in March that invoices go to Bramble Cottage, not Coldharbour Road. In
August they write again, annoyed, because an invoice went to Coldharbour Road. Ask a
store that ranks by similarity where invoices go, and the August message wins: it is the
more recent one and it says the wrong address twice.

**That is not the model hallucinating.** The memory layer held both answers and marked
neither one current. Nothing in "an embedding and an `updated_at` column" can represent
*this value replaced that one in March*.

```text
Values ranked by similarity          Values with the interval each held

  Berlin                               Berlin    Jan 10 → Mar 15  ended
  London                               London    Mar 15 → Jun 02  ended
  New York                             New York  Jun 02 → now     live

  ↓                                    ↓
  whichever embeds closest             whichever was true at the
  to the question                      instant you asked about
```

Retrieval asks one question — *what is relevant?* Memory has to answer five:

> *What do I know?* · *When was it true?* · *When did I learn it?* ·
> *What replaced it?* · *Why do I believe it?*

Memvara answers all five **structurally**, which means none of them costs a model call.
A contradiction is an indexed lookup. A historical query is a range condition on two
columns. Provenance is a join.

The long version of that story, question by question, is
[Why Memvara?](https://github.com/memvara/memvara/blob/main/docs/concepts/why-memvara.md).
Memvara is neither a vector database nor a replacement for RAG —
[where it fits](https://github.com/memvara/memvara#rag-vector-stores-and-where-this-fits)
is further down.

---

## The 90-second demo

**Temporal memory, contradiction resolution and provenance, in ninety seconds.**

![Alice moves from Berlin to London to New York. get_all() answers New York, get_all(as_of=March) answers London, and history() shows every value with the interval it was true for.](https://github.com/memvara/memvara/releases/latest/download/demo.gif)

```bash
pip install memvara
git clone https://github.com/memvara/memvara && cd memvara
python3 examples/temporal_memory_demo/demo.py
```

Six beats: the problem, three writes, what is true now, what was true then, the record
behind both answers, and the close. Ninety seconds, held to a published schedule rather
than to whatever its pauses happen to add up to. It runs against a real store with no key
and no network — every value on screen is read back out of it.

`--fast` removes the pauses and changes nothing else, which is why the transcript is a
[golden file](https://github.com/memvara/memvara/blob/main/examples/temporal_memory_demo/expected-output.txt) the test suite asserts
on. The GIF above is not checked in either: it is a build product, recorded by CI from
this repository and attached to each release, so the URL never changes and the image
never goes stale. [How it is recorded](https://github.com/memvara/memvara/blob/main/examples/temporal_memory_demo/README.md), including
the deterministic replay that makes the same source produce the same bytes.

---

## Quickstart

**Two ways to run it, and the API is the same either way.** `Memvara("memory.db")` and
`Memvara(api_key="mv_…")` return objects with the same methods, so a function that takes a
`Memvara` and calls `search()` cannot tell which it was handed.

<table>
<tr>
<td width="50%" valign="top">

### Run it yourself

```bash
pip install memvara
```

```python
mem = Memvara("memory.db", user="alice")
```

A SQLite file you own. No API key, no network on the write path, no Docker, no vector
database — numpy is the only hard dependency. Apache-2.0, and everything in this
repository is in it.

As an MCP server on the same machine:

```bash
MEMVARA_DB=~/memory.db memvara-mcp
memvara-mcp init --agent claude
```

JSON-RPC 2.0 over stdio, fourteen tools, no SDK dependency. Claude Code, Claude Desktop,
Cursor, VS Code, Windsurf and Zed each have their own one-liner at
[memvara.dev/docs/self-hosted](https://memvara.dev/docs/self-hosted).
[MCP](https://github.com/memvara/memvara/blob/main/docs/integrations/mcp.md) ·
[Deploying](https://github.com/memvara/memvara/blob/main/docs/DEPLOY.md)

</td>
<td width="50%" valign="top">

### Hosted, at memvara.dev

Nothing to install and no key to paste. Give any MCP client this address:

```text
https://app.memvara.dev/mcp
```

Approve it once in a browser. That is OAuth, so the client holds a grant you can revoke
rather than a secret it has to store. In Claude Code the plugin wires the same URL and the
skill in one step:

```
/plugin marketplace add memvara/claude-memvara
/plugin install memvara
```

Claude, Claude Code, ChatGPT, Codex, Cursor, Grok, VS Code, OpenCode and OpenClaw each
have their own page at [memvara.dev/docs/cloud](https://memvara.dev/docs/cloud).

From your own code instead — the same client, against the hosted `/v1` API:

```bash
pip install 'memvara[cloud]'
```

```python
mem = Memvara(api_key="mv_…", user="alice")     # or Memvara.connect()
```

**A bare `Memvara()` never becomes remote.** The dispatch reads the explicit `api_key=` or
`base_url=` argument and never the environment, so a script that has always written to a
local file cannot start posting to a hosted store because somebody ran `memvara-mcp login`
on that machine.
[A hosted deployment](https://github.com/memvara/memvara/blob/main/docs/API.md#a-hosted-deployment)

Not a Python shop? `/v1` is a REST API and a bearer key.
[Cloud quickstart](https://memvara.dev/docs/cloud/quickstart) ·
[REST](https://memvara.dev/docs/cloud/api) ·
[Python client](https://memvara.dev/docs/cloud/python)

**Free: 12,000 memories and one project**, held rather than granted monthly, plus 2,000
recalls a month that do refill. Past either, the next call is refused and names the plan
that carries more — no card, and no way to be billed by surprise.
[Pricing](https://memvara.dev/pricing)

</td>
</tr>
</table>

The walkthrough below runs against either. It is written for the local store because that
is the one you can start in ten seconds.

```python
from datetime import datetime, timezone
from memvara import Memvara, NullLLM

UTC = timezone.utc
def at(m, d): return datetime(2026, m, d, tzinfo=UTC)

mem = Memvara("memory.db", user="alice", llm=NullLLM())

mem.remember("Alice", "lives_in", "Berlin",
             valid_from=at(1, 10), recorded_at=at(1, 10))
mem.remember("Alice", "lives_in", "London",
             valid_from=at(3, 15), recorded_at=at(3, 15))
mem.remember("Alice", "lives_in", "New York",
             valid_from=at(6, 2), recorded_at=at(6, 2))

[c.object for c in mem.get_all()]                  # ['New York']
[c.object for c in mem.get_all(as_of=at(3, 20))]   # ['London']
[c.object for c in mem.get_all(as_of=at(1, 20))]   # ['Berlin']

[r.text for r in mem.search("where does Alice live?")]
# ['Alice lives in New York']
```

Two dates per fact, because they answer different questions. **`valid_from` is when it
became true in the world; `recorded_at` is when this store was told.** They are equal here
because Alice told us on the day she moved — a fact that arrives late about the past is
where they differ, and that is the case a single `updated_at` column cannot represent.

That is [`examples/temporal_memory.py`](https://github.com/memvara/memvara/blob/main/examples/temporal_memory.py), which the test suite
runs and asserts on. Full walkthrough:
[Quickstart](https://github.com/memvara/memvara/blob/main/docs/getting-started/quickstart.md).

### Or hand it the conversation

```python
mem.add("I live in Berlin and work at Acme")
mem.add("Actually, I moved to Lisbon last month")

[r.text for r in mem.search("where do they live?")][:1]
# ['user lives in Lisbon']

[(c.object, c.state) for c in mem.history("user", "lives_in")]
# [('Berlin', 'ended'), ('Lisbon', 'live')]
```

`add()` takes a string, a list of strings, pre-built `Episode`s, or OpenAI/mem0-style
`{"role": ..., "content": ...}` transcripts, so an existing agent loop can pass its
messages straight through. It runs three model-free tiers first — hash dedupe,
near-duplicate detection, a salience gate and a rule-based extractor — and batches
whatever survives into a single extraction call.

**With no `llm=` configured there is no model tier at all**, so the two sentences above
work (they are recognised forms) and an employer mentioned in passing does not. Dropped
turns are counted on `WriteReceipt.unextracted` and the constructor warns once. That is the
qualifier on the offline claim: the library runs with no API key; extraction from arbitrary
prose does not. `remember()` is the offline way to get the full machine, and it is what a
real integration does.

---

## Use cases

**Coding agents are the case this repository leans on most.** Two weeks after a
migration, [`examples/coding_agent.py`](https://github.com/memvara/memvara/blob/main/examples/coding_agent.py)
answers the question git cannot:

```text
Q. What is checkout-service's auth strategy?
  OAuth 2.0 client credentials

Q. What was the old strategy, and when did it change?
  API keys                     2026-02-03 -> 2026-06-12  [ended]
  OAuth 2.0 client credentials 2026-06-12 -> now         [live]
  changed on 2026-06-12
```

`why()` then returns the June transcript turn the decision came from, and the claim it
replaced. That is a record rather than a note, and no model composes any of it.
[Guide: coding agents](https://github.com/memvara/memvara/blob/main/docs/guides/coding-agents.md)

The other four shapes this store is built for:

| | |
|---|---|
| **Support agents** | The customer corrected the address in March; the agent must not quote the old one in August. This is the corpus behind [`demo/`](https://github.com/memvara/memvara/blob/main/demo/README.md). |
| **Personal assistants** | The built-in vocabulary is this one: where somebody lives, works, what they are allergic to, and how they want to be spoken to (`memory_standing`). |
| **Research agents** | A finding that arrives late about the past is a `valid_from` in the past and a `recorded_at` of today, which is exactly what the two clocks are for. |
| **Multi-agent systems** | `tenant > user > agent > session`, with inheritance and fail-closed filters: a session sees that user's durable memory but never a sibling session's scratch space. |

```python
bob = mem.scope(user="bob")     # the whole API, with the scope bound
bob.add("I live in Oslo")
```

---

## Why Memvara

| | |
|---|---|
| 🕰️ **Two clocks, not one** | When it was true, and when you learned it — independently. Ask what you believed in March about June and get an answer, not a guess. |
| ⚖️ **Contradictions resolve without a model** | Cardinality is a schema property, so a conflict is an indexed lookup. Same two facts, same result, every run. |
| 🧾 **Nothing is silently lost** | A superseded fact is *ended*, never deleted, and every write returns a receipt saying what it did — including what it could *not* extract. |
| 🔍 **Retrieval that explains itself** | Vector and BM25 fused by rank, decayed per predicate, and every score inspectable rather than a ranking you have to trust. |
| 🗣️ **It answers the audit question in words** | `ask()` gives what is true now, what was true then, and what this store *would have told you* then — plus the day the record changed. No model composes it. |
| 🛡️ **A guess cannot quietly overwrite a statement** | A value worth less than half of what it would replace is kept beside it, and the receipt names both. Overwriting would record that the world changed, when nothing had. |
| 🧬 **Claims are a graph** | Walk relationships at a point in time — and optionally fuse that walk into search as a third retrieval leg. |
| 🔌 **Offline by default** | numpy and nothing else. No API key, no Docker, no vector database, no network on the write path. |

Longer form: [Why Memvara?](https://github.com/memvara/memvara/blob/main/docs/concepts/why-memvara.md)

---

## Temporal memory

Two axes means two clocks, and they move independently:

```python
mem.get_all(valid_at=T)   # what we believe TODAY about how the world was at T
mem.get_all(known_at=T)   # what we believed at T, about the world as it is now
mem.get_all(as_of=T)      # both clocks at T — what we believed at T, about T
```

The middle two are the ones a single instant cannot ask. A correction that arrives in
August about June is invisible to `as_of=June`, because that call rewinds the belief clock
past the correction; `valid_at=June` is how you see it. `as_of` is exact sugar for
`valid_at=known_at=T`, and passing it alongside either axis raises rather than quietly
picking one.

Eight reads take all three — `search`, `get_all`, `count`, `history`, `why`, `produced`,
`neighborhood`, `paths_between`. `recall()`, `get()` and `since()` take none of them, and
`ask()` spells it `at=`.

`ask()` composes the difference into an answer, which is the question the two clocks exist
for:

```python
mem.ask("where do they live?", at=datetime(2026, 3, 15, tzinfo=utc)).text

# user lives_in: Berlin.
#   On 2026-03-15 this store would have said Rome, and that is what anyone acting
#   on it then acted on. The difference was recorded 2026-03-22, 7 days after the
#   instant you asked about.
```

Three readings of every fact it touches — what is true now, what we believe today was true
then, and what this store *would have answered* then. The last two differing means the
record was corrected after the moment you asked about, so the answer somebody acted on is
not the answer they would get today. **No model is consulted**; every sentence is rendered
from a stored column.

[Bitemporal memory](https://github.com/memvara/memvara/blob/main/docs/concepts/bitemporal-memory.md) ·
[Temporal retrieval](https://github.com/memvara/memvara/blob/main/docs/concepts/temporal-retrieval.md)

---

## Contradiction resolution

```python
mem.remember("Alice", "lives_in", "Berlin")
mem.remember("Alice", "lives_in", "Lisbon")

[c.object for c in mem.get_all()]
# ['Lisbon']

[(c.object, c.state) for c in mem.history("Alice", "lives_in")]
# [('Berlin', 'ended'), ('Lisbon', 'live')]
```

Three steps, none of which involves a model. **The predicate is normalised** — `lives_in`,
`resides_in`, `based_in` and `moved_to` are one slot. **The entity is folded** —

```python
from memvara import entity_key
entity_key("Acme Corp.") == entity_key("ACME, Inc.") == entity_key("acme")   # True
```

— which is what makes the keyed lookup fire at all. **Then cardinality decides**:
`lives_in` is declared single-valued, so the new value closes the old one's interval.

The alternative design — embed, retrieve the nearest existing memories, ask a model
whether they conflict — fails two ways that have nothing to do with model quality. It can
**miss** (the conflicting value need not fall in top-k, and then both stay live), and it is
**not repeatable** (the same two facts can resolve differently on two runs, with nothing
downstream able to tell).

### Teach it your vocabulary

The built-in predicates are a personal-assistant vocabulary — where someone lives, where
they work. **A store of engineering facts matches none of them**, and an unknown predicate
takes the safe default twice over: multi-valued, so nothing supersedes, and slow-decaying,
so this morning's deploy still ranks as fresh in two years.

```bash
MEMVARA_PREDICATES=engineering memvara-mcp   # or: engineering,events,decisions,./ours.toml
```

Three packs ship: `engineering`, `events` and `decisions`. Naming several is normal and
they do not collide — loading all three declares 48 predicates against a 64-name ceiling
on what each extraction prompt carries.

```toml
[[predicate]]
name = "git_state"
cardinality = "one"     # supersedes; "many" accumulates
volatility = "fast"     # static | slow | fast -> 36500 | 730 | 7 day half-life
```

A declaration outranks a guess, so a pack corrects a store that already classified
something wrongly rather than only shaping a fresh one.

[Contradiction resolution](https://github.com/memvara/memvara/blob/main/docs/concepts/contradiction-resolution.md)

---

## Provenance

```python
turn = mem.add("Decision: migrate auth from API keys to OAuth 2.0.",
               role="system", ts=jun)
mem.remember("checkout-service", "auth_strategy", "OAuth 2.0",
             sources=turn.episode_ids, valid_from=jun, recorded_at=jun)

p = mem.why(claim_id)
[e.text for e in p.episodes]      # ['Decision: migrate auth from API keys to OAuth 2.0.']
[c.text for c in p.superseded]    # ['checkout-service auth strategy API keys']
p.derivation, p.extractor         # (<Derivation.USER: 'user'>, 'api')

[c.text for c in mem.produced(turn.episode_ids[0])]   # the same link, backwards
```

The turn, not a paraphrase of it — and `superseded`, which is the field that turns a note
into a record. *"We decided X"* becomes *"we decided X on 12 June, replacing Y, on this
evidence"*, which is the sentence an incident review actually needs.

**Three words, three different events, and using the wrong one is undetectable
afterwards:**

| Word | What happened | Written by |
|---|---|---|
| **ended** | The world changed. It was true, and then it wasn't. | a superseding write, or `forget(close="ended")` |
| **retired** | The record was wrong. It was never true. | `forget()`, `delete()` — the default |
| **erased** | The text itself is gone. Not recoverable. | `erase()`, `purge()` |

*Served a value that expired* and *served a value that was never true* are one column apart
and are not the same finding. Only the third deletes anything.

[Provenance](https://github.com/memvara/memvara/blob/main/docs/concepts/provenance.md)

---

## Architecture

```mermaid
flowchart TD
    A["Your agent<br/><i>Python, MCP client, or a framework adapter</i>"]
    B["<b>Memvara</b> — memvara/core.py<br/><i>add · remember · search · recall · ask ·<br/>history · why · forget · erase</i>"]
    W["Write path — memvara/write/<br/><i>dedupe → salience gate → rule extractor →<br/>(model, only if needed) → reconcile</i>"]
    R["Read path — memvara/retrieve/<br/><i>vector + BM25 + graph, fused by rank,<br/>decayed per predicate</i>"]
    S["Store — memvara/store/<br/><i>SQLite + FTS5 + an mmap vector sidecar</i>"]
    O["<b>Current state</b> + <b>history</b> + <b>provenance</b>"]

    A --> B
    B --> W
    B --> R
    W --> S
    S --> R
    R --> O
    B -.-> O
```

The write path and the read path meet only at the store, and **nothing on the read path
calls a model by default** — not even the optional reranker, which is a cross-encoder
rather than a generative model. The one opt-in exception is `search(ranked=True)` against
a retriever configured with a `read_selector`: one chat call per read, on the customer's
own key, naming which of the reranked turns actually bear on the question. See
`memvara.select`. History and provenance are not a separate subsystem: they fall out of
the store keeping intervals and supersession pointers instead of overwriting rows.

Everything replaceable is a protocol — `Store`, `Embedder`, `LLM`, `Redactor`, `Recorder` —
and each has a real second implementation in this repository rather than being a
hypothetical extension point.

[Architecture, in four diagrams](https://github.com/memvara/memvara/blob/main/docs/reference/architecture.md) ·
[How it works](https://github.com/memvara/memvara/blob/main/docs/DESIGN.md) · [Internals](https://github.com/memvara/memvara/blob/main/docs/INTERNALS.md)

### At a glance

| | |
|---|---|
| Unit of memory | a claim: `(subject, predicate, object)` |
| Temporal model | valid time and recorded time, queried independently |
| Conflict handling | predicate-aware and deterministic, decided on write |
| Provenance | the source episodes, the derivation, and the claim superseded |
| Retrieval | vector and BM25 fused by rank, optionally a graph leg, decayed per predicate |
| Storage | SQLite with FTS5, plus an mmap vector sidecar |
| Dependencies | numpy. Everything else is an extra |
| Model dependency | none for `remember()`; `add()` reaches for one only where no rule matches the prose |
| Python | 3.10 and later |

---

## Integrations

| | Keeps | Loses |
|---|---|---|
| **MCP** — Claude Code, Cursor, Codex, Grok, VS Code, OpenCode, Claude Desktop, ChatGPT | everything | — |
| **LangChain** `MemvaraRetriever` | everything, including `as_of=` | — |
| **LlamaIndex** `MemvaraRetriever`, `MemvaraMemoryBlock` | everything; the memory block keeps the write path too | — |
| **LangGraph** `MemvaraStore` | contradiction resolution, per-field claims | the predicate registry |
| **LangChain** `MemvaraChatMessageHistory` | the write path | supersession, intervals, source ids |
| **CrewAI** `MemvaraStorage` | storage and scope | the keyed lookup — its unit of memory has no subject or predicate |

The rule that decides it: **an interface that hands over the query text keeps everything;
one that hands over a pre-computed embedding, or a list of messages, cannot.** Each adapter
says which it is, out loud. [Frameworks](https://github.com/memvara/memvara/blob/main/docs/integrations/frameworks.md)

### Coming from somewhere else

```python
from memvara.compat import import_mem0, import_supermemory

import_mem0(mem, history_db="~/.mem0/history.db")   # replays mem0's own mutation log
import_supermemory(mem)                             # reads the Supermemory export API
```

mem0 records what changed and when, so that import rebuilds supersession and answers
`as_of` afterwards. Supermemory records current state, so its documents arrive as episodes
on their original timestamps and nothing invents a history it was never told. There is
also a method-level mem0 shim if you want its call surface on this store.

---

## Documentation

**[Start here → docs/](https://github.com/memvara/memvara/blob/main/docs/README.md)** — every page links to the next one.

| | |
|---|---|
| Getting started | [Installation](https://github.com/memvara/memvara/blob/main/docs/getting-started/installation.md) · [Quickstart](https://github.com/memvara/memvara/blob/main/docs/getting-started/quickstart.md) · [Your first memory](https://github.com/memvara/memvara/blob/main/docs/getting-started/first-memory.md) |
| Concepts | [Why Memvara?](https://github.com/memvara/memvara/blob/main/docs/concepts/why-memvara.md) · [Bitemporal memory](https://github.com/memvara/memvara/blob/main/docs/concepts/bitemporal-memory.md) · [Contradiction resolution](https://github.com/memvara/memvara/blob/main/docs/concepts/contradiction-resolution.md) · [Provenance](https://github.com/memvara/memvara/blob/main/docs/concepts/provenance.md) · [Temporal retrieval](https://github.com/memvara/memvara/blob/main/docs/concepts/temporal-retrieval.md) · [RAG and memory](https://github.com/memvara/memvara/blob/main/docs/concepts/rag-vs-memory.md) |
| Guides and integrations | [Coding agents](https://github.com/memvara/memvara/blob/main/docs/guides/coding-agents.md) · [MCP](https://github.com/memvara/memvara/blob/main/docs/integrations/mcp.md) · [Frameworks](https://github.com/memvara/memvara/blob/main/docs/integrations/frameworks.md) |
| Reference | [API](https://github.com/memvara/memvara/blob/main/docs/API.md) · [Architecture](https://github.com/memvara/memvara/blob/main/docs/reference/architecture.md) · [How it works](https://github.com/memvara/memvara/blob/main/docs/DESIGN.md) · [Internals](https://github.com/memvara/memvara/blob/main/docs/INTERNALS.md) · [Deploying](https://github.com/memvara/memvara/blob/main/docs/DEPLOY.md) |
| Also | [FAQ](https://github.com/memvara/memvara/blob/main/docs/FAQ.md) · [Benchmarks](https://github.com/memvara/memvara/blob/main/docs/BENCHMARKS.md) · [Limitations](https://github.com/memvara/memvara/blob/main/docs/LIMITATIONS.md) · [Upgrading](https://github.com/memvara/memvara/blob/main/docs/UPGRADING.md) · [Roadmap](https://github.com/memvara/memvara/blob/main/docs/ROADMAP.md) · [Open core](https://github.com/memvara/memvara/blob/main/docs/OPEN-CORE.md) |
| Examples | [Three runnable programs](https://github.com/memvara/memvara/blob/main/examples/README.md), asserted on by the suite |

---

## RAG, vector stores, and where this fits

**Memvara is not a replacement for RAG and it is not a vector database.** It uses vector
search internally, and swapping in pgvector is a protocol away.

```
RAG                                    Memory
 ↓                                      ↓
"Which documents are relevant?"        "What persistent state do I know about
                                        this entity, how has it changed, and
                                        what was true at a point in time?"
```

| | RAG | Memvara |
|---|---|---|
| Unit | a chunk of a document | a claim: `(subject, predicate, object)` |
| Addressed by | similarity to a query | the slot `(subject, predicate)` |
| Corpus | mostly static; documents are added | mostly mutable; values are replaced |
| A contradiction is | two chunks that both rank | a slot with two live values, resolved on write |
| "When" means | the document's date | two dates: when it was true, when you learned it |
| Good at | *what does the manual say about TLS errors* | *what is this customer's billing address, and what was it in March* |

They compose: **RAG answers from the corpus, memory supplies the state the corpus does not
know.** *"What is our refund window for this customer's plan?"* is two questions — which
plan they are on (one slot, one current value, with a history) and what the policy says for
that plan (a passage from a document).

Memvara ships retriever adapters for LangChain and LlamaIndex, so in an existing pipeline
it can be one more retriever rather than a separate call.
[RAG and memory](https://github.com/memvara/memvara/blob/main/docs/concepts/rag-vs-memory.md)

### Against mem0 specifically

mem0 and its descendants store a memory as an opaque string with an embedding, and every
`add()` costs a model call on the critical path. Retrieval is vector top-k.

> **Corrected against mem0 2.0.17.** An earlier version of this section said `add()` costs
> *two* LLM calls — extract, then adjudicate ADD/UPDATE/DELETE. That described mem0 1.x.
> 2.x makes **one** call, with existing memories passed into a single additive extraction
> prompt; `DEFAULT_UPDATE_MEMORY_PROMPT` is still in the source and no longer reached from
> the add path. The correction cuts against us, so it is stated rather than quietly
> dropped — but the contradiction problem it was cited for got *larger*, not smaller: 2.x's
> add path emits only `ADD` events, and its prompt says "Your sole operation is ADD".
> Conflicting values are **linked**, never retired. `update()` and `delete()` are calls
> your application has to know to make.

Four consequences that show up in production:

1. **Contradictions accumulate.** In 2.x this is explicit: nothing on the write path
   retires anything. Six months in, the store holds three cities for one person and
   returns whichever embeds closest to the question.
2. **Writes are slow and expensive.** A model call per turn, on the critical path,
   including for "ok, thanks."
3. **There is no time.** One `updated_at` column can't answer "where did she live in
   March?" or absorb a fact that arrives late about the past.
4. **Nothing explains itself.** When the agent says something wrong, you cannot ask which
   memory caused it, where that memory came from, or why it ranked first.

Memvara is built around the observation that **most of this doesn't need a model at all.**

---

## Agent Memory Benchmark

A public, reproducible benchmark for **memory systems in general**, not for memvara. It
measures what happens to a fact that changes: current state, historical state,
contradiction resolution, provenance, knowledge time, retrieval among distractors, cost
and latency. 262 events, 100 questions, 16 scenarios, no API key, about a second a system.

```bash
python -m benchmarks.agent_memory --system memvara --system naive --system vector-rag --compare
```

Any memory system can implement the adapter interface and be scored on the same dataset by
the same rules — including from another repository, without forking this one. Memvara does
not win every category, and the table reports the ones it loses.

| | |
|---|---|
| Run it, and the methodology | [`benchmarks/agent_memory/README.md`](https://github.com/memvara/memvara/blob/main/benchmarks/agent_memory/README.md) |
| The report, with results and limitations | [`docs/benchmarks/agent-memory-benchmark.md`](https://github.com/memvara/memvara/blob/main/docs/benchmarks/agent-memory-benchmark.md) |
| Add your memory system | [`benchmarks/agent_memory/CONTRIBUTING.md`](https://github.com/memvara/memvara/blob/main/benchmarks/agent_memory/CONTRIBUTING.md) |

---

## Measured

| | |
|---|---|
| Agent Memory Benchmark: changing facts, across systems | [`docs/benchmarks/agent-memory-benchmark.md`](https://github.com/memvara/memvara/blob/main/docs/benchmarks/agent-memory-benchmark.md#results) |
| Against the real `mem0ai` package | [`docs/BENCHMARKS.md`](https://github.com/memvara/memvara/blob/main/docs/BENCHMARKS.md#measured-against-the-real-mem0-package) |
| The two clocks, six question families | [`docs/BENCHMARKS.md`](https://github.com/memvara/memvara/blob/main/docs/BENCHMARKS.md#the-two-clocks-measured-synthetic-self-authored) |
| LOCOMO and LongMemEval, retrieval | [`docs/BENCHMARKS.md`](https://github.com/memvara/memvara/blob/main/docs/BENCHMARKS.md) |
| LongMemEval, judged answer accuracy (0.11.0's ranked recall) | [`docs/BENCHMARKS.md`](https://github.com/memvara/memvara/blob/main/docs/BENCHMARKS.md#answer-accuracy-judged-in-the-memorybench-harness) |
| Answer quality, end to end | [`docs/BENCHMARKS.md`](https://github.com/memvara/memvara/blob/main/docs/BENCHMARKS.md#answer-quality-end-to-end-an-authored-corpus-an-agent-as-the-reader) |

The harnesses are in [`bench/`](https://github.com/memvara/memvara/tree/main/bench) and [`demo/`](https://github.com/memvara/memvara/blob/main/demo/README.md), and every number is
reproducible from this repository. Where a result is synthetic or self-authored, its own
heading says so.

What these numbers do not cover, and every other limit this project knows about, is in
[Limitations](https://github.com/memvara/memvara/blob/main/docs/LIMITATIONS.md) — including why the LOCOMO and LongMemEval
retrieval figures are not answer accuracy, which is the one most often quoted wrongly. The
judged accuracy figure above is one run on one sample, with no same-harness comparison
against another vendor's hosted service.

---

## Development

```bash
python3 -m pytest -q                              # 4,172 passing, 10 skipped, no API key
python3 -m coverage run -m pytest && python3 -m coverage report   # gated at 100%
python3 -m benchmarks.agent_memory --system memvara --compare   # the agent memory benchmark
PYTHONPATH=. python3 bench/temporal.py            # the two clocks, six families
PYTHONPATH=. python3 bench/compare.py             # architecture comparison
PYTHONPATH=. python3 bench/perf.py                # throughput and scaling
```

**100% statement coverage, enforced** (`fail_under = 100`), and `mypy -p memvara` is
clean in CI. The suite runs in about 21
seconds with no network, no API key, and almost no sleeping — time is controlled by
passing explicit `datetime` values rather than patching the clock, and the handful of
tests that do sleep are measuring concurrency, where the wall clock is the thing under
test.

Coverage of the *lines* is the floor, not the goal. What the suite actually pins down:

- **Behavior** — contradictions resolve, history survives, users are isolated in all
  three directions (sibling session, sibling agent, other tenant), and the LLM stays idle.
  Fakes count their own calls, and the tests assert on those counts — the design claim is
  that the model is rarely consulted, so a test that doesn't count calls doesn't test it.
- **Failure paths** — dimension mismatches, transaction rollback (including nested),
  a classifier that raises, a store that loses rows mid-query, and model output that
  violates every field contract at once. These only run during an incident, which is
  exactly why they can't ship unexercised.
- **Adversarial input** — a fuzz corpus (SQL and FTS5 injection, path traversal, template
  injection, control characters, astral-plane codepoints, 5KB strings, combining marks)
  driven through every public method and a persistence round trip, plus randomized
  transcripts asserting the store never ends up internally inconsistent.
- **Executable docs** — the `Memvara` docstring runs as a doctest, the README walkthrough
  is mirrored in `tests/test_integration.py`, and `tests/test_examples.py` runs every
  program under `examples/` in a subprocess and asserts on what it prints. So the code a
  developer copies cannot drift from the code that ships.

The twelve remaining *branch* partials are verified-unreachable defensive guards — mostly
`if valid_to is None or valid_to > t`, where a live claim always satisfies the first
disjunct, so the second can never decide the branch. They are kept as guards rather than
deleted, and documented as such.

Design notes and the module-by-module contract live in [docs/INTERNALS.md](https://github.com/memvara/memvara/blob/main/docs/INTERNALS.md).
[docs/UPGRADING.md](https://github.com/memvara/memvara/blob/main/docs/UPGRADING.md) is the short list of changes that do not announce
themselves — read it before upgrading, starting with the one where `invalidated_at is
None` stopped meaning "live" without breaking anything.

## Contributing

Issues and pull requests are welcome. Two things to read first:

- **[CONTRIBUTING.md](https://github.com/memvara/memvara/blob/main/CONTRIBUTING.md)** — the bar a patch has to clear, and what will and
  will not be accepted. It is specific about scope: some things belong here, some belong
  in the commercial half, and it says how to tell which is which before you write the code.
- **[docs/ROADMAP.md](https://github.com/memvara/memvara/blob/main/docs/ROADMAP.md)** — what is done, what is still missing, and a
  *Deliberately deferred* list, which exists so that considered-and-declined stops reading
  as not-yet-done.

```bash
git clone https://github.com/memvara/memvara && cd memvara
python3 -m pip install -e ".[dev,cloud]"
python3 -m pytest -q
```

The gate is 100% statement coverage and a clean `mypy -p memvara`, both enforced in CI.
Documentation ships in the same commit as the code it describes — including the tool
descriptions in `memvara/server/tools.py`, which a model reads at runtime.

[SECURITY.md](https://github.com/memvara/memvara/blob/main/SECURITY.md) covers private vulnerability reporting. Do not open a public
issue for a vulnerability.

## License

Apache-2.0, for everything in this repository. See
[Open core](https://github.com/memvara/memvara/blob/main/docs/OPEN-CORE.md) for what is and is not in it.
