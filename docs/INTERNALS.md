# Engram internals — module contracts

This file is the interface contract between subsystems. `core.py` wires them together
against exactly these signatures, so treat them as fixed. Everything here is already
importable from the foundation modules:

- `engram/types.py` — `Claim`, `Episode`, `Scope`, `Result`, `Explanation`, `WriteReceipt`,
  `MemoryType`, `Derivation`, `utcnow()`, `content_hash()`
- `engram/schema.py` — `PredicateRegistry`, `PredicateSpec`, `Cardinality`, `Volatility`
- `engram/store/` — `Store` protocol, `SQLiteStore`
- `engram/embed/` — `Embedder` protocol, `HashingEmbedder`, `CachedEmbedder`, `default_embedder()`
- `engram/llm/base.py` — `LLM` protocol, `NullLLM`, `CLAIM_SCHEMA`, `PREDICATE_SCHEMA`,
  `EXTRACT_SYSTEM`, `PREDICATE_SYSTEM`

## Design invariants (do not violate)

1. **Deterministic paths never call an LLM.** Deduplication, contradiction resolution,
   ranking, decay, and time travel are all pure functions of stored state. Only
   `extract()` and `classify_predicate()` may touch a model.
2. **Unknown predicates default to `Cardinality.MANY`.** Wrongly retiring a true fact is
   worse than keeping two competing ones.
3. **Nothing is ever hard-deleted by the engine.** Superseding a claim sets
   `invalidated_at` / `invalidated_by`. History must stay queryable via `as_of`.
4. **Every claim carries provenance.** `sources` must be populated with the episode ids
   the claim came from, and `derivation` must reflect how it was produced.
5. **Everything must run with no API key and no network.** `NullLLM` + `HashingEmbedder`
   is the default configuration and the one the tests use.

---

## `engram/write/`

### `write/gate.py`

```python
class SalienceGate:
    def carries_fact(self, ep: Episode) -> tuple[bool, str]
```
Cheap, deterministic triage: does this turn plausibly contain a durable fact? Returns
`(should_extract, reason)`. Reason is a short slug used in receipts and tests
(`"no_content"`, `"ack_only"`, `"question"`, `"assistant_turn"`, `"has_declarative"`, ...).

Bias toward recall: a false positive costs one extraction call, a false negative loses a
memory permanently. Cheap negatives worth catching: empty/whitespace, pure
acknowledgements ("ok", "thanks", "sounds good"), bare questions with no declarative
clause. This gate is what removes most of the write-path LLM spend, because the majority
of conversational turns carry nothing durable.

### `write/fast.py`

```python
class FastExtractor:
    def __init__(self, registry: PredicateRegistry) -> None
    def extract(self, ep: Episode) -> list[Claim]
```
High-precision, zero-LLM pattern extraction for the handful of statement forms that are
both common and unambiguous — "my name is X", "I live in X", "I work at X", "I prefer X",
"I'm allergic to X", "I no longer work at X" (polarity -1). Precision over recall: emit
nothing rather than a wrong triple; the LLM tier is the fallback. Set
`derivation=Derivation.FAST_PATH`, `extractor="fast/v1"`, and `sources=[ep.id]`.

### `write/reconcile.py`

```python
class Reconciler:
    def __init__(self, store: Store, registry: PredicateRegistry) -> None
    def apply(self, claim: Claim, *, now: datetime | None = None) -> ReconcileResult
```
The contradiction engine. For a candidate claim:

1. **Exact duplicate** — a live claim with the same `value_key` exists: do not insert.
   Bump `observation_count`, raise `salience`, merge `sources`, return `action="reinforce"`.
2. **Conflict** — the predicate is `Cardinality.ONE` and live claims share the candidate's
   `fact_key` with a different `value_key`: insert the new claim, and for each superseded
   claim set `invalidated_at=now`, `invalidated_by=<new id>`, and `valid_to=now`.
   Return `action="supersede"` with the list.
3. **Retraction** — candidate has `polarity == -1`: invalidate matching live claims and do
   not store the negative claim as a live fact.
4. **Accumulate** — otherwise insert. `action="add"`.

```python
@dataclass
class ReconcileResult:
    action: str                  # "add" | "reinforce" | "supersede" | "retract" | "noop"
    claim: Claim | None          # the stored/updated claim
    invalidated: list[Claim]     # claims this one retired
```

### `write/pipeline.py`

```python
class WritePipeline:
    def __init__(self, store, embedder, registry, llm, *,
                 near_dup_threshold: float = 0.97,
                 reinforce_bump: float = 0.25) -> None

    def add(self, episodes: Sequence[Episode]) -> WriteReceipt
    def assert_claim(self, claim: Claim) -> WriteReceipt
```

`add()` runs the tiers in order and must populate every field of `WriteReceipt`,
including `llm_calls` (0 whenever the LLM is not consulted) and `latency_ms`:

- **Tier 0 (no LLM):** store the episode; skip content-hash duplicates
  (`store.find_episode_by_hash`). For surviving episodes, embed and check near-duplicates
  against existing claim embeddings — cosine >= `near_dup_threshold` reinforces the
  existing claim instead of writing a new one.
- **Tier 1 (no LLM):** `SalienceGate` drops turns carrying no durable fact
  (count them in `receipt.skipped`), then `FastExtractor` handles what it can.
- **Tier 2 (LLM):** only the turns that survived both and produced no fast-path claim are
  batched into a single `llm.extract(...)` call. Map `source_index` back to the
  originating episode for provenance. Unknown predicates trigger one
  `llm.classify_predicate(...)` per *new* predicate, cached via `registry.learn(...)` so
  it is never asked again.

Every produced claim goes through `Reconciler.apply()`, and every stored claim gets its
embedding written via `store.set_embedding()`.

---

## `engram/retrieve/`

### `retrieve/fusion.py`

```python
def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    *, k: int = 60, weights: Mapping[str, float] | None = None,
) -> dict[str, float]
```
Standard RRF: an item at rank `r` (0-based) in list `L` contributes
`weights[L] / (k + r + 1)`. Rank fusion rather than score fusion, because BM25 scores and
cosine similarities are not on comparable scales and normalizing them is guesswork.

### `retrieve/scoring.py`

```python
def recency_factor(claim: Claim, registry: PredicateRegistry, now: datetime) -> float
def final_score(fusion: float, *, recency: float, confidence: float, salience: float,
                w_recency: float, w_confidence: float, w_salience: float) -> float
```
`recency_factor` is exponential decay on the predicate's half-life:
`0.5 ** (age_days / half_life_days)`, age measured from `claim.valid_from`. A `STATIC`
predicate's 100-year half-life keeps its factor at ~1.0, so birthplaces do not decay out
of the ranking while "what I'm working on today" does.

### `retrieve/hybrid.py`

```python
class HybridRetriever:
    def __init__(self, store, embedder, registry, *,
                 w_vector: float = 1.0, w_lexical: float = 1.0, rrf_k: int = 60,
                 w_recency: float = 0.25, w_confidence: float = 0.15,
                 w_salience: float = 0.10, candidate_multiplier: int = 5) -> None

    def search(self, query: str, scope: Scope, *, k: int = 10,
               as_of: datetime | None = None, include_invalidated: bool = False,
               memory_types: Sequence[MemoryType] | None = None) -> list[Result]
```

Search must:
- expand `scope` via `scope.ancestors()` so a session query also sees user-level memory;
- run vector and lexical retrieval over `k * candidate_multiplier` candidates each;
- fuse with RRF, then rescore with recency/confidence/salience;
- pass `as_of` through to the store so **time travel returns what we believed then**,
  including claims later invalidated;
- populate `Explanation` on every `Result` — per-retriever rank and raw score, the fusion
  score, each scoring factor, and the final score. A result with no explanation is a bug.

---

## `engram/consolidate/`

```python
class Consolidator:
    def __init__(self, store, embedder, registry) -> None

    def decay(self, tenant: str | None = None, now: datetime | None = None) -> int
    def merge_duplicates(self, tenant: str | None = None, threshold: float = 0.97) -> int
    def promote(self, tenant: str | None = None, min_observations: int = 3) -> int
    def run(self, tenant: str | None = None) -> dict[str, int]
```

- `decay` multiplies `salience` by the predicate's recency factor, floored at `0.05` so
  nothing decays to zero and disappears from ranking entirely.
- `merge_duplicates` finds live claims sharing a `fact_key` whose embeddings exceed
  `threshold`, keeps the one with the highest `observation_count` (ties broken by earliest
  `recorded_at` for determinism), folds the others' `sources` and `observation_count` into
  it, and invalidates them with `invalidated_by` pointing at the survivor.
- `promote` turns a repeatedly-observed `EPISODIC` claim into a `SEMANTIC` one: seeing
  something happen once is an event, seeing it `min_observations` times is a pattern.
  The promoted claim gets `derivation=Derivation.CONSOLIDATION`.
- `run` executes all three and returns the per-stage counts.

All counts are "number of claims affected". These run off the write path.

---

## `engram/llm/anthropic.py`

```python
class AnthropicLLM:
    name: str                    # e.g. "anthropic/claude-opus-5"
    def __init__(self, model: str = "claude-opus-5", client=None,
                 effort: str = "low", max_tokens: int = 8192) -> None
    def extract(self, episodes, known_predicates) -> list[dict]
    def classify_predicate(self, predicate: str, example: str) -> dict
```

Hard API requirements — these are current and getting them wrong is a 400:

- Structured output goes in `output_config={"format": {"type": "json_schema", "schema": ...}}`.
  The top-level `output_format` parameter is deprecated; do not use it.
- **Never** pass `temperature`, `top_p`, or `top_k` — they are rejected on this model.
- Control depth with `output_config={"effort": "low"}` alongside `format`. Leave adaptive
  thinking on (the default); do not pass `thinking={"type": "disabled"}`.
- Import `anthropic` lazily inside `__init__` so `import engram` works without it, and
  raise a clear install hint if it is missing.
- Use `CLAIM_SCHEMA` / `PREDICATE_SCHEMA` / `EXTRACT_SYSTEM` / `PREDICATE_SYSTEM` from
  `llm/base.py` rather than redefining them.
- Validate and coerce the model's output before returning: drop claims with a missing or
  out-of-range `source_index`, clamp `confidence` to `[0, 1]`, and normalize predicates to
  snake_case. The engine trusts these dicts, so this is the trust boundary.

---

## Testing requirements

- `pytest`, no network, no API key, no sleeping. Use `SQLiteStore(":memory:")`,
  `HashingEmbedder`, and `NullLLM` or a local fake.
- Control time by passing explicit `datetime` values (`utcnow()` +/- `timedelta`) rather
  than patching the clock.
- Test behavior through the public surface, and cover the failure modes, not just the
  happy path: empty input, unicode, adversarial FTS5 query strings (`"a AND (b"`),
  concurrent writes, dimension mismatches, claims with no embedding.
- A fake LLM must assert on *call count*, not just output — the whole design claim is that
  the LLM is called rarely, so a test that does not count calls does not test the design.
