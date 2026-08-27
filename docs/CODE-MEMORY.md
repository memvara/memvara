# Code memory

`memvara.code` is the first code-specific layer built on Memvara.

The goal is not to store a summary of every file. It is to maintain an auditable,
change-aware understanding of a codebase that can be reused by coding agents.

## P0

P0 supports Python and deliberately has no parser dependency. It uses Python's standard
`ast` module to identify:

- modules
- classes
- functions and async functions
- methods and async methods
- class variables
- module variables

Each symbol gets:

- a human-readable path and qualified name
- a deterministic symbol id
- a source hash
- a structural fingerprint
- its signature
- source range
- parent symbol

The path is metadata, not the semantic identity. When a symbol moves and its implementation
fingerprint is unchanged, the index can carry its previous id forward. A move therefore
updates `code_path` without throwing away the symbol's semantic context.

## Context lifecycle

Semantic context is stored as a normal Memvara claim:

```text
subject   = code:symbol:<id>
predicate = code_context
object    = LLM-generated context
```

`code_context` is single-valued. Writing a replacement context retires the old context
through Memvara's normal contradiction machinery. The old context remains auditable and
can be inspected historically.

A code edit therefore has three different outcomes:

```text
path changed, implementation unchanged
    -> update path; keep context

implementation changed
    -> retire old context; generate new context

symbol removed
    -> retire the symbol's context and structural claims
```

This distinction is intentional. A file rename should not cause an expensive LLM call,
and a source edit should not leave an old explanation live.

## LLM boundary

The P0 index is deterministic. Semantic generation is supplied through a
`ContextBuilder`:

```python
from memvara import CodeIndex, CodeMemory, Memvara

memory = Memvara("code-memory.db", user="repository")
code = CodeMemory(memory)

index = CodeIndex.from_directory(".")
code.sync(
    index,
    context_builder=lambda symbol, snapshot: generate_context(symbol, snapshot),
)
```

The builder is intentionally a protocol boundary rather than another LLM client. The
Memvara package already has provider-specific LLM integrations; code memory should not
force a second provider abstraction into the storage layer.

## What P0 does not do yet

P0 does not claim to understand a whole repository semantically. It does not yet build
cross-file dependency edges, ingest Git history, or automatically generate context with a
provider. Those are the next layers.

The important P0 invariant is smaller:

> **Never invalidate semantic context merely because a path changed; invalidate it when
the code that context describes actually changed.**

That invariant is what makes the later LLM layer economically viable.
