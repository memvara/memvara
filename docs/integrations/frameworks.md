# Framework adapters

Adapters for LangChain, LlamaIndex, LangGraph and CrewAI ship in `memvara/integrations/`.
**They do not all preserve what makes Memvara different, and this page says which loses
what** — because an adapter that quietly degrades into a vector store is worse than no
adapter at all.

The rule that decides it is simple: **an interface that hands over the query text keeps
everything; one that hands over a pre-computed embedding, or a list of messages, cannot.**
Memvara retrieves from a string — BM25 fused with vectors, rescored by a per-predicate
half-life — so a vector from somebody else's model is not even a point in its space.

| Adapter | Interface | Keeps | Loses |
|---|---|---|---|
| `langchain.MemvaraRetriever` | `BaseRetriever` | everything, including `as_of=` | — |
| `llamaindex.MemvaraRetriever` | retriever | everything, including `as_of=` | — |
| `llamaindex.MemvaraMemoryBlock` | `BaseMemoryBlock` | the whole write path *and* `recall()` | — |
| `langgraph.MemvaraStore` | `BaseStore` | contradiction resolution, per-field claims | the predicate registry |
| `langchain.MemvaraChatMessageHistory` | `BaseChatMessageHistory` | the write path | supersession, intervals, source ids |
| `crewai.MemvaraStorage` | `StorageBackend` | storage and scope | the headline feature — see below |

## The clean fits

**Retrievers.** `BaseRetriever` asks for *given a query, return documents*, which is what
`search()` already is. Everything survives the crossing: hybrid retrieval and its ranking,
scope inheritance, the retirement of contradicted values (they simply stop being
returned), the per-leg `Explanation`, the source turn ids, and `as_of` — a LangChain
retriever that can answer *what did we believe in March* is not something the interface
anticipated, and it works, because time travel is a property of the query rather than of
the response shape.

**LlamaIndex `MemvaraMemoryBlock`.** A memory block is asked *here are the recent
messages, produce what should be in the prompt* and *here are messages, absorb them* —
which is `recall()` and `add()` with the arguments already in the right order. It is the
only adapter where the **write** path is Memvara's: the hash dedupe, the salience gate,
the rule extractor and the batched model call all run.

## LangGraph loses least of the lossy ones, and instructively

`BaseStore.search(namespace_prefix, *, query: str, …)` hands over the text natively, *and*
`put(namespace, key, value)` supplies all three parts of a triple:

```
namespace + key    who this is about        ->  subject
a key of `value`   which question           ->  predicate
its value          the answer               ->  object
```

So an item is stored as **one claim per field**, and a later `put` changing `city` ends
exactly `city` and leaves `food` alone. That is contradiction resolution surviving a
foreign interface intact.

The two closures stay distinct across the interface, which is the part worth knowing: a
field whose **value changes** is an ordinary supersession and is stamped `ended` — the
world moved — while a field that **disappears** from the item is `delete()`d and so is
`retired`, with no `invalidated_by`, because nothing replaced it. `put` is an update
rather than a deletion request, and those are two different statements about the world.

What it loses is the **predicate registry**: a stored `home_city` does not contradict an
extracted `lives_in`, because the field name arrives as a bare string with no declaration
behind it.

```python
from memvara import Memvara
from memvara.integrations.langgraph import MemvaraStore   # pip install 'memvara[langgraph]'

store = MemvaraStore(Memvara("memory.db"), user="alice")
```

`memvara[langgraph]` names **`langgraph-checkpoint`**, not `langgraph`, and that is not a
typo: `pip install langgraph` produces a wheel with no `langgraph/store/` in it at all,
and depends on `langgraph-checkpoint>=4.1,<5`, which is where `langgraph.store.base`
actually lives. An application already using langgraph satisfies the floor without a
second resolve.

`on_delete="erase"` is worth knowing about: the default deletion is a retirement, which is
the wrong answer to a data-deletion request.

## LangChain chat history is lossy, deliberately not disguised

`BaseChatMessageHistory` models memory as **a list of messages**, and Memvara's unit is a
reconciled bitemporal claim. There is nowhere in a `list[BaseMessage]` to put a
supersession, a valid-time interval, a confidence, or the id of the turn a fact came from.

So `messages` returns the stored turns — the honest reading of the contract — and says
out loud both ways that list is smaller than it looks: the memory Memvara built is not in
it, and tier-0 hash dedupe means a repeated turn was stored once, so it is a deduplicated
corpus of source material rather than a verbatim log.

**`clear()` raises by default**, and that is the sharper edge. LangChain documents it as
*remove all messages from the store*; Memvara has two candidate meanings — retirement
(reversible, history intact) and `purge()` (irreversible erasure of claims, turns, vectors
and text index) — and those are not variants of one operation. A memory layer that guesses
between them at session teardown has picked the worst possible moment to be clever.

```python
MemvaraChatMessageHistory(mem, on_clear="ignore")   # "this session ended, keep what was learned"
MemvaraChatMessageHistory(mem, on_clear="purge")    # LangChain's literal meaning
```

Use `MemvaraRetriever` to put memory in a prompt. That is the supported shape.

## CrewAI loses the headline feature outright

`StorageBackend.search` is handed a `query_embedding` and **never the query text** —
CrewAI embeds the query with its own model before the backend is reached. Its unit of
memory is an opaque sentence with no subject or predicate, so the keyed lookup has nothing
to key on: *"Alice lives in Berlin"* and *"Alice moved to Lisbon"* both stay live.

```python
from crewai.memory import Memory
from memvara.integrations.crewai import MemvaraStorage   # pip install 'memvara[crewai]'

storage = MemvaraStorage(mem, user="alice")
memory = Memory(storage=storage, embedder=storage.embedder)
```

**`embedder=storage.embedder` is not optional.** The backend supplies the embedder,
remembers what it embedded, and recovers the query text from the vector it produced.
The two dishonest alternatives were both rejected: storing CrewAI's vectors and degrading
to cosine top-k (Memvara becomes a numpy matmul with extra steps), or re-embedding text
the interface never provides (impossible).

`crewai>=1.10.1` is a load-bearing floor. Releases 1.0.0 through 1.9.3 ship the previous
memory system with no `StorageBackend` protocol at all.

## Not a framework: the hosted store

`Memvara(api_key=…)` is the same class and the same methods, served by a deployment's
`/v1` API instead of a store on your machine, with every response hydrated back into the
same dataclasses. A function that takes a `Memvara` and calls `search()` cannot tell which
it was handed. See [A hosted deployment](../API.md#a-hosted-deployment) — including the
two places it diverges from the local engine on purpose.

---

Previous: [MCP](mcp.md) · Next: [Architecture](../reference/architecture.md)
