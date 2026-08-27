# Code memory P0 test plan

P0 is not ready for a semantic LLM layer until its deterministic lifecycle is predictable.
The test suite treats these as product invariants, not implementation details.

## Identity invariants

- A symbol's path is an address, not its durable identity.
- A unique symbol that moves keeps its symbol id.
- A unique symbol that is renamed can keep its symbol id when its implementation is unchanged.
- Ambiguous fingerprints must not silently merge two symbols.

## Change invariants

- Unchanged symbols produce no lifecycle work.
- A source implementation change produces a `changed` event.
- A move or rename with the same implementation does not require an LLM context rebuild.
- Removing a symbol retires its code claims rather than deleting history.
- A syntax-broken file remains visible to the file index, but its unavailable symbols are not fabricated.

## Context invariants

- Context is stored through normal Memvara claims.
- Replacing context retires the previous live context and preserves its history.
- A context builder is invoked only for symbols that need new semantic context.
- Context generation is not part of deterministic identity or invalidation decisions.

## Current P0 coverage

The tests cover:

1. Python modules, classes, functions, async functions, methods, async methods, module variables, and class variables.
2. File moves without semantic context invalidation.
3. Source changes and targeted context regeneration.
4. Symbol removal and claim retirement.
5. Invalid Python source.
6. Unrelated method changes not changing a class variable's fingerprint.
7. Unchanged checkouts producing no changes.

## Before P1

P1 should add tests for cross-file dependencies and relationship invalidation before introducing automatic semantic context generation. The important failure mode to prevent is a function retaining a locally valid explanation while its dependency contract has changed.
