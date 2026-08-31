# Architecture

Four diagrams, in increasing detail. Every box below is a module in this repository —
nothing here is aspirational, and [Internals](../INTERNALS.md) is the module-by-module
contract behind it.

## The shape of it

```mermaid
flowchart TD
    A["Your agent<br/><i>Python, MCP client, or a framework adapter</i>"]
    B["<b>Memvara</b> — memvara/core.py<br/><i>the facade: add, remember, search, recall,<br/>ask, history, why, forget, end</i>"]
    W["Write path — memvara/write/<br/><i>dedupe → gate → extract → reconcile</i>"]
    R["Read path — memvara/retrieve/<br/><i>vector + BM25 + graph, time-aware</i>"]
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

Two things this picture is making a point of:

- **The write path and the read path meet only at the store.** Nothing on the read path
  calls a model, ever — not even the optional reranker, which is a cross-encoder rather
  than a generative model.
- **`history` and `provenance` are not a separate subsystem.** They fall out of the store
  keeping intervals and supersession pointers instead of overwriting rows.

## The write path, tier by tier

The design claim is that the model is rarely consulted. This is the mechanism, and each
tier's job is to stop a turn before it reaches the one that costs money.

```mermaid
flowchart TD
    E["Episode<br/><i>a turn, a document, a paste</i>"]
    T0{"<b>Tier 0</b> — no model<br/>content-hash duplicate?<br/>near-duplicate (cosine ≥ 0.97)?"}
    T1{"<b>Tier 1</b> — no model<br/>SalienceGate: does this<br/>carry a durable fact?"}
    T1B{"<b>Tier 1b</b> — no model<br/>FastExtractor: a<br/>recognised sentence form?"}
    T2["<b>Tier 2</b> — the only model call<br/>llm.extract(...), batched<br/>across the surviving turns"]
    REC["<b>Reconciler</b> — memvara/write/reconcile.py<br/><i>normalise predicate · fold entity ·<br/>look up (subject, predicate) · apply cardinality</i>"]
    ST[("Store")]
    SKIP["counted on the receipt<br/><i>skipped · reinforced · unextracted</i>"]

    E --> T0
    T0 -- "duplicate" --> SKIP
    T0 -- "new" --> T1
    T1 -- "chitchat" --> SKIP
    T1 -- "carries a fact" --> T1B
    T1B -- "matched" --> REC
    T1B -- "no match" --> T2
    T2 --> REC
    REC --> ST
```

With no `llm=` configured there is **no tier 2 at all**: turns that reach it are dropped
and counted on `WriteReceipt.unextracted`, and the constructor warns once. That is the
qualifier on the offline claim — the library runs with no API key; extraction from
arbitrary prose does not. `remember()` enters at the Reconciler and skips every tier
above it, which is why a real integration writing triples never needs a model.

`WriteReceipt` reports `llm_calls` on every write, so the claim is checkable rather than
asserted.

## The read path

```mermaid
flowchart TD
    Q["query string<br/>+ scope + time keywords"]
    V["vector leg<br/><i>exact numpy matmul over<br/>the candidate set</i>"]
    L["lexical leg<br/><i>BM25 via SQLite FTS5</i>"]
    G["graph leg — optional<br/><i>GraphTraverser walks<br/>claim-to-claim</i>"]
    F["Reciprocal Rank Fusion<br/><i>ranks, not scores — BM25 and cosine<br/>are not on comparable scales</i>"]
    SC["rescore<br/><i>recency (per-predicate half-life)<br/>× confidence × salience</i>"]
    RR["reranker — optional, off by default<br/><i>cross-encoder</i>"]
    RES["Result + Explanation<br/><i>every score inspectable</i>"]

    Q --> V
    Q --> L
    Q --> G
    V --> F
    L --> F
    G --> F
    F --> SC
    SC --> RR
    RR --> RES
```

The three time keywords are resolved **before anything else**, so `as_of` passed alongside
`valid_at` raises rather than quietly picking one. They reach the store as SQL predicates
over the four date columns, which is what makes a historical search a query rather than a
post-filter.

Decay is measured at `known_at`, not `valid_at`: recency asks how long ago we last heard
something, and that is a question about the belief clock.

## What a claim is

```mermaid
classDiagram
    class Claim {
        +id
        +subject
        +predicate
        +object
        +valid_from : when it became true
        +valid_to : when it stopped being true
        +recorded_at : when we learned it
        +invalidated_at : when we stopped believing it
        +confidence
        +salience
        +memory_type
        +scope
        +state() live | ended | retired
    }
    class Episode {
        +id
        +text
        +role
        +ts
        +scope
    }
    class Scope {
        +tenant
        +user
        +agent
        +session
    }
    class PredicateSpec {
        +name
        +cardinality : ONE | MANY
        +volatility : STATIC | SLOW | FAST
        +memory_type
        +aliases
    }
    Claim "1" --> "0..*" Episode : sources (why)
    Claim "1" --> "0..1" Claim : invalidated_by
    Claim --> Scope
    Claim ..> PredicateSpec : resolved through
```

**The four date columns are the whole design.** `valid_from`/`valid_to` are the world
clock; `recorded_at`/`invalidated_at` are the belief clock. Which pair a write closes is
what separates *ended* from *retired* — see
[bitemporal memory](../concepts/bitemporal-memory.md#the-three-states-and-why-there-are-three).

`Scope` is hierarchical — `tenant > user > agent > session`, with inheritance — so a query
at session scope also sees that user's durable memory but never a sibling session's
scratch space. Scope filters fail **closed**: a scope that resolves to nothing matches
nothing rather than degrading into an unfiltered query.

## Where the seams are

Everything replaceable is a protocol, and each one has a real second implementation in
this repository rather than being a hypothetical extension point:

| Protocol | Default | Also in this repo |
|---|---|---|
| `Store` | `SQLiteStore` | `RemoteStore` — partial on purpose, raising where the REST facade has no endpoint |
| `Embedder` | `HashingEmbedder` (offline, lexical) | `CachedEmbedder`, `LocalEmbedder` (sentence-transformers) |
| `LLM` | `NullLLM` | `AnthropicLLM`, `OpenAILLM` |
| `Redactor` | none | `PatternRedactor` |
| `Recorder` | none | `MemoryRecorder` |

The vector index is exact and in-process — a numpy matmul over the candidate set. Correct
and fast to roughly a million claims, at which point `Store` is where pgvector or Qdrant
goes.

## What is not in this repository

The REST API and the hosted control plane are the commercial half, and that is a decision
rather than a gap — [Open core](../OPEN-CORE.md) says where the line is and why it does
not move. What *is* here is the client half, twice over: `memvara/remote/` is what an
application calls (`Memvara(api_key=…)`), and `memvara/store/remote.py` is the low-level
`Store` the local engine calls.

---

Previous: [Frameworks](../integrations/frameworks.md) · Next: [API reference](../API.md) · [FAQ](../FAQ.md)
