# Consolidation, vocabularies and the graph

Two subsystems keep a store useful over a year rather than only over a week. Consolidation is
scheduled maintenance that fades what is stale, merges near-duplicates and promotes repeated
observations into patterns. The graph is the relationship structure that already exists
between claims — one claim's object is another claim's subject — which retrieval can walk to
answer a question two hops deep. Predicate vocabularies sit underneath both: they are what
tells the engine whether a slot holds one value or many, and how fast that kind of fact goes
stale.

## Where the code is

- Consolidation: `memvara/consolidate/__init__.py` — `Consolidator`, with `decay()`,
  `merge_duplicates()`, `promote()` and `run()`. `memvara/consolidate/decay.py` —
  `decay_pass()`, `decay_factor()`, `SALIENCE_FLOOR`. `memvara/consolidate/merge.py` —
  `merge_pass()`, `promote_pass()`, `survivor_rank()`. `memvara/consolidate/sweep.py` —
  `Sweep`, the bounded-transaction snapshot the pass reads and writes back through.
- Vocabularies: `memvara/schema.py` — `PredicateRegistry`, `PredicateSpec`, `Cardinality`,
  `Volatility`, `available_packs()`, `PredicatePackError`. The shipped packs are
  `memvara/packs/engineering.toml`, `memvara/packs/decisions.toml` and
  `memvara/packs/events.toml`.
- Entities: `memvara/entities.py` — `EntityRegistry`, `entity_key()`, `entity_id()`, which
  fold surface forms into the keys the graph joins on.
- Graph: `memvara/retrieve/traverse.py` — `GraphTraverser`, `Edge`, `Path`;
  `memvara/retrieve/spread.py` — `seed_keys()`, `rank_paths()`;
  `Store.adjacent()` and `Store.connectivity()` in `memvara/store/base.py`.
- Public entry points: `Memvara.consolidate()`, `Memvara.connectivity()`,
  `Memvara.neighborhood()`, `Memvara.paths_between()`, `Memvara.merge_predicate()` in
  `memvara/core.py`.
- Tests: `tests/test_decay.py`, `tests/test_merge.py`, `tests/test_traverse.py`,
  `tests/test_edges.py`, `tests/test_entities.py`, `tests/test_predicates.py`,
  `tests/test_predicate_packs.py`, `tests/test_predicate_backfill.py`.
- Documentation: [INTERNALS.md](../INTERNALS.md), sections *`memvara/consolidate/`*, *The
  store-level graph gate* and *`connectivity()`*. The measured value of the graph leg is in
  [BENCHMARKS.md](../BENCHMARKS.md).

## How the pieces fit

`Consolidator.run()` executes three stages and returns per-stage counts, where a count is
the number of claims that pass changed.

- `decay()` multiplies each claim's salience by its predicate's recency factor, floored so
  that nothing decays to zero and vanishes from ranking altogether.
- `merge_duplicates()` finds live claims sharing a slot whose embeddings exceed the
  threshold, keeps the one with the most observations, folds the others' sources and counts
  into it, and invalidates them with `invalidated_by` pointing at the survivor.
- `promote()` turns a repeatedly observed episodic claim into a semantic one. Seeing
  something happen once is an event; seeing it several times is a pattern. The promoted claim
  is marked `Derivation.CONSOLIDATION`.

On a settled store every count is zero, which is the signal that consolidation has converged.

The graph leg in `HybridRetriever` runs after the first fusion and re-fuses the whole list.
Its seeds are the folded entity keys of the best-scoring claims, from `spread.seed_keys()`,
so the walk needs no entity extractor over the free-text query and no second vocabulary that
could disagree with the store's.

Predicate vocabularies are declared in TOML and loaded through `MEMVARA_PREDICATES`, which
accepts shipped pack names, file paths, or a comma-separated mix, with later entries winning.
A declared spec outranks a persisted learned one, so a pack corrects a store that guessed
rather than only describing a fresh one. Loading a pack needs Python 3.11 or newer, where
`tomllib` arrives; on 3.10 the feature refuses with a message that names the cause.

## Invariants and assumptions

- **Consolidation is deterministic and calls no model.** No stage takes an `llm` parameter.
  Every stage is idempotent, because this runs on a schedule and a scheduler that fires twice
  must not leave a different store than one that fires once.
- **`Sweep` reads its snapshot once and writes back in bounded transactions.** Off the write
  path is not the same as out of the way; without this the pass would scan the table three
  times or hold the write lock for its own duration.
- **`now` is read once for the whole pass.** Pass it explicitly to evaluate two passes at the
  same instant, because the decay target depends on that instant and a claim near a rounding
  boundary would otherwise change on the second call.
- **The graph leg is closed when the store provably has no joins.** The condition is
  `joinable_claims == 0`, not a percentage threshold, because any percentage would be a
  constant fitted to the two corpora that exist. The gate sits after intent weighting so it
  can undo what the query classifier opened.
- **An empty connectivity result keeps the leg on.** A backend without `connectivity()`, or a
  facade too old to report the counts, has measured nothing, and reading that as "no joins"
  would switch a working graph leg off on every third-party store at once.
- **An unregistered predicate accumulates rather than superseding, and decays slowly.** That
  default is deliberate, because wrongly retiring a true fact is worse than keeping two
  competing ones. A vocabulary is how a deployment revises it.

## Read next

[INTERNALS.md](../INTERNALS.md)'s consolidate section states each stage's contract and its
constants. The comment block at the top of `memvara/packs/engineering.toml` explains
cardinality and volatility in two sentences and is the model for writing a new pack.

Next: [telemetry and benchmarks](telemetry-and-benchmarks.md).
