# Memvara internals — module contracts

This file is the interface contract between subsystems. `core.py` wires them together
against exactly these signatures, so treat them as fixed. Everything here is already
importable from the foundation modules:

- `memvara/types.py` — `Claim`, `Episode`, `Scope`, `Result`, `Explanation`, `WriteReceipt`,
  `MemoryType`, `Derivation`, `utcnow()`, `content_hash()`
- `memvara/schema.py` — `PredicateRegistry`, `PredicateSpec`, `Cardinality`, `Volatility`
- `memvara/store/` — `Store` and `SQLStore` protocols, `SQLiteStore`, `STATES`,
  `ClaimState`, `resolve_states()`, `state_predicate()`, `stored_state_predicate()`,
  `live_predicate()`
- `memvara/embed/` — `Embedder` protocol, `HashingEmbedder`, `CachedEmbedder`, `default_embedder()`
- `memvara/llm/base.py` — `LLM` protocol, `NullLLM`, `CLAIM_SCHEMA`, `PREDICATE_SCHEMA`,
  `EXTRACT_SYSTEM`, `PREDICATE_SYSTEM`

## Design invariants (do not violate)

1. **Deterministic paths never call an LLM.** Deduplication, contradiction resolution,
   ranking, decay, and time travel are all pure functions of stored state. Only
   `extract()` and `resolve_predicate()` may touch a model.
2. **Unknown predicates default to `Cardinality.MANY`.** Wrongly retiring a true fact is
   worse than keeping two competing ones.
3. **Nothing is ever hard-deleted by the engine, and end-of-life moves exactly one
   clock.** Closing valid time (`valid_to`) says *the world changed*; closing transaction
   time (`invalidated_at`) says *the record was wrong*. They are different events and no
   write may assert both. Superseding a claim therefore sets `valid_to` and
   `invalidated_by` and leaves `invalidated_at` unset — the old value stopped being true,
   and we were never mistaken about it. `Claim.state` names the outcome: `live`, `ended`,
   `retired`. History must stay queryable via `known_at` **and** `valid_at`; the second
   of those returned nothing on any history the engine wrote itself for as long as
   supersession closed both clocks. The correcting reading is reachable, never guessed:
   `close="retired"` on `remember`, `supersede`, `forget` and `delete`, defaulting to
   `"ended"` everywhere except `forget`/`delete`, which are belief operations by name.
4. **Every claim carries provenance.** `sources` must be populated with the episode ids
   the claim came from, and `derivation` must reflect how it was produced.
5. **Everything must run with no API key and no network.** `NullLLM` + `HashingEmbedder`
   is the default configuration and the one the tests use.
6. **A multi-hop answer is evaluated at one clock pair.** Every edge on a returned path
   must be checked against the same `(valid_at, known_at)`, pinned once before the walk.
   A path stitched from edges believed at different times is a connection that never
   simultaneously held, and reporting it as a fact is the worst thing traversal can do —
   it is invisible in any result that does not carry its timestamps. Two axes widened
   what has to be pinned; they did not weaken the rule.
7. **A filter and a limit may not live in different layers.** Whatever narrows rows has to
   run where the truncation runs, or the top-k is wrong: `Store.adjacent` shipped without
   a `scopes` argument and with the caller filtering afterwards, and on a shared tenant a
   question with 20 answers returned 8. This applies to any future store method that caps
   rows the caller is expected to authorize. It is why `states=` is a store parameter and
   not a comprehension in the facade: `search` over-fetches `k * candidate_multiplier` and
   ranks those, so a state filter applied afterwards finds a retired claim only when it
   happens to land inside that window — twelve live rows against `k=1` is a window of
   five, and the audit comes back empty with nothing saying it was truncated.

---

## `memvara/write/`

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
    def apply(self, claim: Claim, *, now: datetime | None = None,
              close: Closure = "ended") -> ReconcileResult
```
The contradiction engine. `close` decides which clock stops on whatever the candidate
displaces, and `"ended"` is the only answer this class could reach on its own: it is
told "here is the new value" and never "the old one was a mistake". For a candidate
claim:

1. **Exact duplicate** — a live claim with the same `value_key` exists: do not insert.
   Bump `observation_count`, raise the *storage* strength (`Claim.salience_base`) and
   stamp `last_observed`, merge `sources`, return `action="reinforce"`. The bump goes
   on the base, not on `salience`: the nightly pass recomputes `salience` from the
   base, so writing it there was erased once a claim aged past `0.415 * half_life`
   — 2.9 days for a FAST predicate, and permanently, since age only grows.
2. **Conflict** — the predicate is `Cardinality.ONE` and live claims share the candidate's
   `fact_key` with a different `value_key`: insert the new claim, and for each superseded
   claim set `invalidated_by=<new id>` plus `valid_to=<the new claim's valid_from>`,
   leaving `invalidated_at` unset. The old value stopped being true where the new one
   begins; it was not an error, so nothing on the belief clock moves and
   `get_all(valid_at=<back then>)` still returns it. Under `close="retired"` the axes
   swap: `invalidated_at=now` and `valid_to` untouched, because a correction witnessed
   no world event. Return `action="supersede"` with the list.
3. **Retraction** — candidate has `polarity == -1`: close out matching live claims and
   store the negative claim as a tombstone (invalidated *and* ended at `now`, so it can
   never be live) rather than as a live fact. The matches are **ended**, not retired:
   every negative form the write path produces is "no longer" / "used to" / "not any
   more", which is the world moving on. `close="retired"` is the caller saying the
   original was never true.
4. **Accumulate** — otherwise insert. `action="add"`.

```python
@dataclass
class ReconcileResult:
    action: str                  # "add" | "reinforce" | "supersede" | "retract" | "noop"
    claim: Claim | None          # the stored/updated claim
    invalidated: list[Claim]     # claims this one closed out — on whichever clock
                                 # `close=` stopped, so `ended` by default, not retired
```

`WritePipeline` copies that list onto `WriteReceipt.closed`, where `receipt.ended` and
`receipt.retired` split it by `Claim.state`. Anything rendering the list as one word is
wrong for one of the two closures; a supersession is `ended`.

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
  `llm.resolve_predicate(...)` per *new surface form*, cached via `registry.learn_alias`
  / `registry.learn` and persisted through `store.put_spec(spec, tenant)` so it is never
  asked again — including after a restart, and including by another process.

  Resolution, not classification, is the point. Asking "what cardinality is this?" lets
  `works_at`, `employed_by_company`, `job_employer` and `workplace` become four separate
  slots that can never contradict each other; a red-team simulation of 10k extractions
  over six concepts produced 41 predicates and four simultaneously-live employers that
  way. Asking "which existing predicate is this?" spends the same one-per-form call on
  merging instead, and a deterministic morphological pre-pass answers most of them for
  free before any model is consulted.

Every produced claim goes through `Reconciler.apply()`, and every stored claim gets its
embedding written via `store.set_embedding()`.

---

## `memvara/retrieve/`

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
def normalized_score(...) -> float   # Result.score, in [0, 1]
def final_score(fusion: float, *, recency: float, confidence: float, salience: float,
                w_recency: float, w_confidence: float, w_salience: float) -> float
```
`recency_factor` is exponential decay on the predicate's half-life:
`0.5 ** (age_days / half_life_days)`, age measured from `claim.trace_from`
(`max(valid_from, last_observed)`) — from `valid_from` alone, a fact restated daily
for ninety days still scored as ninety days stale. A `STATIC`
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
               as_of: datetime | None = None, valid_at: datetime | None = None,
               known_at: datetime | None = None,
               states: Collection[str] | None = None,
               include_invalidated: bool | None = None,
               memory_types: Sequence[MemoryType] | None = None,
               min_score: float = 0.0,
               include_episodes: bool = False) -> list[Result]
```

Search must:
- expand `scope` via `scope.ancestors()` so a session query also sees user-level memory;
- run vector and lexical retrieval over `k * candidate_multiplier` candidates each;
- fuse with RRF, then rescore with recency/confidence/salience;
- resolve the three time keywords through `types.time_axes` **before anything else**, so
  `as_of` + `valid_at` raises whatever else the call would have done;
- pass `valid_at` and `known_at` through to the store so **time travel returns what we
  believed then, and what we now believe was true then**, including claims later
  invalidated. Decay is measured at `known_at`, not `valid_at`: recency asks how long ago
  we last heard something, and that is a question about the belief clock;
- populate `Explanation` on every `Result` — per-retriever rank and raw score, the fusion
  score, each scoring factor, and the final score. A result with no explanation is a bug.

### `retrieve/traverse.py`

```python
class GraphTraverser:
    def __init__(self, store, registry, *, damping: float = HOP_DAMPING,
                 beam: int = 64, edge_limit: int = 1000) -> None

    def neighborhood(self, entity: str, scope: Scope, *, depth: int = 2, k: int = 10,
                     min_hops: int = 1, predicates: Sequence[str] | None = None,
                     as_of: datetime | None = None, valid_at: datetime | None = None,
                     known_at: datetime | None = None,
                     min_score: float = 0.0) -> list[Path]
    def paths_between(self, source: str, target: str, scope: Scope, *,
                      depth: int = 3, k: int = 3, ...) -> list[Path]
```

Traversal must:
- **pin one clock pair before the first hop** and pass that same
  `(valid_at, known_at)` to every `Store.adjacent` call. Neither axis may be forwarded as
  `None` — the store substitutes its own clock per call, so a 3-hop walk would evaluate 3
  hops at 3 instants and could return a path that was true at none of them. One clock read
  fills both defaults, so an argument-free walk is still a single coherent moment. This is
  invariant 6 below;
- drop `polarity <= 0` before a claim becomes an edge, so the guarantee holds for every
  store rather than for the ones that remembered;
- pass `scope.ancestors()` to `adjacent` **and** re-check `Scope.sees` on what comes
  back. The first is what makes the answer correct under a cap; the second is what makes
  the guarantee ours rather than a third-party store's;
- keep the score non-increasing along a path — a path may never outscore its own prefix,
  which is what makes `min_score` prunable mid-walk exactly rather than approximately.
  Salience is therefore excluded: it is unbounded above 1.0 by design;
- bound everything (depth, beam, per-hop `edge_limit`, cycle check) and order totally,
  with no `uuid4` deciding anything observable.

---

## `memvara/store/`

### The two time axes

`as_of` exists on the public facade and **nowhere below it**. `Memvara`, `ScopedMemvara`
and `AsyncMemvara` accept all three keywords and resolve them through
`types.time_axes(as_of, valid_at, known_at)`, which returns the pair and raises if
`as_of` was combined with either axis. Everything under the facade speaks in the pair.

Every `Store` method that used to take `as_of` now takes `valid_at` and `known_at`, both
**keyword-only**. That is deliberate: they replaced a positional argument, and a caller
still passing an instant third would otherwise be silently reinterpreted as `valid_at` —
a wrong answer with no error, which is the failure the split exists to remove.

```python
competing_claims(tenant, fact_key, *, valid_at=None, known_at=None)
adjacent(tenant, keys, *, outgoing=True, incoming=True, predicates=None,
         valid_at=None, known_at=None, scopes=None, limit=1000)
candidate_ids(scopes, *, valid_at=None, known_at=None, states=None,
              include_invalidated=None)
lexical_search(query, scopes, limit, *, valid_at=None, known_at=None, states=None,
               include_invalidated=None)
vector_search(qvec, scopes, limit, *, valid_at=None, known_at=None, states=None,
              include_invalidated=None)
episode_candidate_ids(scopes, *, valid_at=None, known_at=None)
lexical_search_episodes(query, scopes, limit, *, valid_at=None, known_at=None)
vector_search_episodes(qvec, scopes, limit, *, valid_at=None, known_at=None)
```

A SQL-backed store additionally implements `SQLStore`, a **second** protocol declared in
`store/base.py` and deliberately not folded into `Store` — the clause builders are SQL
generation, and `Store` promises a Qdrant or LanceDB backend can implement it without
them:

```python
_state_clause(valid_at, known_at, states=None, alias="")        -> tuple[str, list]
_live_clause(valid_at, known_at, include_invalidated, alias="") -> tuple[str, list]
_happened_clause(valid_at, known_at, alias="")                  -> tuple[str, list]
```

### The three states

A claim is `live` (neither clock closed), `ended` (valid time closed — the world moved
on) or `retired` (transaction time closed — the record was wrong). That is `Claim.state`,
and `store/base.py` exports the vocabulary and the SQL that selects it:

```python
STATES            # ("live", "ended", "retired") — also the canonical order
ClaimState        # Literal of the same three
resolve_states(states=None, include_invalidated=None, *, default=("live",))
state_predicate(at="?", *, states=None, alias="")   -> (sql, axes)
stored_state_predicate(states=None, *, prefix="")   -> sql
live_predicate(at="?", *, include_invalidated=False, alias="") -> sql
```

`resolve_states` is **the one place either spelling is interpreted**, so no surface can
invent its own reading of the older flag. It returns a canonical tuple in `STATES` order,
so one requested population compiles to one string however the caller spelled it.
Passing `states=` and `include_invalidated=` together raises: there is no reading of the
mix in which one of the two is not being ignored. Nothing is deprecated and nothing
warns — `filterwarnings = ["error::DeprecationWarning"]` would turn a warning into a
failure at every existing call site.

`default=` is what the method returns when neither argument is given, *and* what
`include_invalidated=False` means on it — one parameter names both, so they cannot drift
apart. It is `("live",)` on the read path and `("live", "ended")` on `iter_claims`.

`_state_clause` is the parameterised form of `state_predicate` and the method every read
filter in a SQL backend routes through. **It is the binding site** — the only place in
this repository that binds the state predicate's markers. `state_predicate` returns the
SQL *and* an axis list naming the clock behind each marker in order (`("known", "known",
"valid", "valid")` for the live-only case), so binding is a comprehension over that list
rather than a remembered order. That is what makes the one silent error unwritable: a
belief instant bound onto a world column answers identically to a correct one on every
`as_of` call, because those pass the two axes equal.

`_live_clause` is now `_state_clause` with the two-valued alias applied, and is neither
deprecated nor changed in meaning. `live_predicate` is likewise the two-state alias of
`state_predicate` — it drops the axis list, so a caller that binds markers should prefer
the general form. Both remain exported.

`stored_state_predicate` is the member of the family for a walk with no clock to read.
`iter_claims` pages over rows rather than answering a question about a moment, so its
filter is the *stored* state — what `Claim.state` reports — which differs from
`state_predicate` exactly where a timestamp is in the future: a claim retired next
October is `retired` here and still believed there. It returns `""` for the complete set,
meaning "no filter", so the caller drops it rather than emitting a tautology into a paged
scan. Its `prefix` is load-bearing, not decorative: `iter_claims` passes `"+"` to keep
the planner off `cl_live`, which would otherwise sort every page back into rowid order.

**`iter_claims`'s unflagged view is `("live", "ended")`, not live-only.** It has always
meant "every row we still believe", and it must stay that way: `reembed()` walks it, and
narrowing the default would silently stop re-encoding every superseded version in the
store. `include_invalidated=True` is unchanged and still means all three.
`include_invalidated` also stays *positional* there, because it always was.

### The three states do not tile the store

Asking for all three is not the union of the three parts. `Claim.state` is absolute while
`state_predicate` is as-of, so a claim recorded but not yet in force at `valid_at` — a
fact scheduled to start next month — is named by none of the three. The complete set
therefore collapses to the belief floor alone:

```python
state_predicate("?", states=STATES)
# ('(recorded_at <= ?)', ('known',))
```

which readmits that row and leaves `valid_at` with nothing to constrain. That is exactly,
and deliberately, the semantics `include_invalidated=True` has always had, and
`tests/test_bitemporal.py::test_asking_for_all_three_states_is_the_audit_view_valid_at_cannot_narrow`
pins it so it cannot drift into a surprise.

### The clauses themselves

`_live_clause` is four constraints, two per axis, each reading only its own clock:
`recorded_at <= known_at`, `invalidated_at > known_at`, `valid_from <= valid_at`,
`valid_to > valid_at`. `None` on an axis means that axis reads the clock, substituted
once per call — and one read fills both defaults so the two cannot land microseconds
apart.

The SQL itself is not written in either clause builder. `at` is the SQL *expression* for
the instant, substituted at every axis — a bind marker (`"?"`, `"%s"`) or a server clock
(`"now()"`) — so a counter with no store instance, on a raw connection, in another
repository, can still ask the same question. One expression rather than one per axis
because such a counter is always counting *now*, and two markers would let a caller bind
the pair transposed, which no `as_of` query can reveal. Where markers are used the four
bind **known, known, valid, valid**.

`include_invalidated=True` — equivalently `states=STATES` — lifts the **whole valid-time
interval** plus the retirement, leaving only `recorded_at <= known_at`. That is more than
the name promises, it is existing behaviour, and it must stay identical across backends:
under it, `valid_at` has no effect at all. The belief floor is the one clause that never
lifts, under any subset — returning something first heard in July when asked what we
believed in March is the only way a bitemporal read can actively lie.

A subset containing `retired` is written with the retired disjunct **first**, so its
belief marker stays ahead of the world markers and the axis discipline generalises rather
than changing shape per subset. `base._either` parenthesises each disjunction inside
*and* out, because `AND` binds tighter than `OR` and a bare disjunction dropped into a
conjunction re-associates silently — `floor AND retired OR in_force` is
`(floor AND retired) OR in_force`, which drops the belief floor in the direction that
answers "what did we believe in March" with something first heard in July.

`_happened_clause` is the episode form. A turn has no separate record time, so its single
`ts` is both its `valid_from` and its `recorded_at`; substituting that into the four
clauses above leaves `ts <= min(valid_at, known_at)`. There is no `include_invalidated`
and no `states` because nothing retires a turn.

`types.Claim.is_live(as_of=None, *, valid_at=None, known_at=None)` is the Python mirror
of `_live_clause` and is held to the same wording clause for clause. Three copies of one
predicate is three chances to disagree; `tests/test_bitemporal.py` checks the Python one
against the SQL one row for row.

### `stats()`

```python
{"episodes", "claims", "live_claims", "ended_claims", "invalidated", "embeddings"}
```

`live_claims` and `ended_claims` are both taken from `_state_clause`, and so from
`state_predicate`, rather than spelled out — a counter that writes its own copy of the
predicate is exactly how the cheap version got into three files. Neither is a column
test, and neither is derivable from the others. On a store holding one live claim, one
ended claim, one that ended and was *later* retired, and one recorded but not in force
until next year, `stats()` reports `claims=4, live_claims=1, ended_claims=1,
invalidated=1` — and:

| cheaper spelling | gives | truth | why |
|---|---|---|---|
| `invalidated_at IS NULL` | 3 | `live_claims` = 1 | counts every superseded version as live |
| `valid_to IS NOT NULL` | 2 | `ended_claims` = 1 | the ended-then-retired row is already inside `invalidated` |
| `claims - live_claims - invalidated` | 2 | `ended_claims` = 1 | the residual also holds the scheduled claim, which is in no state at all |

`ended_claims` and `invalidated` are **disjoint** — the state predicate excludes the
ended-then-retired row from the first, so it is counted once — which is why the key had
to be added rather than left to subtraction. It was the largest non-live population and
the only one with no key.

**The counts do not sum**, and the leftover is not the ended rows: `1 + 1 + 1` against
`claims = 4`. `claims` is the only total that covers everything, and a backend that
"corrects" the arithmetic has reintroduced the conflation.

That is the one change in this project that is wrong *silently* downstream: the old
one-column test is still valid SQL and still valid Python, it used to be right, and it
now over-counts with nothing to raise. [`docs/UPGRADING.md`](UPGRADING.md) carries the
grep list for finding copies of it.

`invalidate()` and `set_valid_to()` are in `Store` and **no engine path calls either**.
Closing a claim moves one clock; `invalidate` writes `invalidated_at` and
`invalidated_by` together, which is the conflation that was removed, and `set_valid_to`
writes no pointer. They remain as the protocol's only single-statement writes, and
because `set_valid_to(id, None)` reopens an interval — the one thing `close_out` cannot
do, deliberately, since a write path able to un-end a fact would do it by accident.

---

## `memvara/consolidate/`

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

## `memvara/llm/anthropic.py`

```python
class AnthropicLLM:
    name: str                    # e.g. "anthropic/claude-opus-5"
    def __init__(self, model: str = "claude-opus-5", client=None,
                 effort: str = "low", max_tokens: int = 8192) -> None
    def extract(self, episodes, known_predicates) -> list[dict]
    def resolve_predicate(self, surface: str, candidates: Sequence[str]) -> dict
    def classify_predicate(self, predicate: str, example: str) -> dict  # legacy fallback
```

Hard API requirements — these are current and getting them wrong is a 400:

- Structured output goes in `output_config={"format": {"type": "json_schema", "schema": ...}}`.
  The top-level `output_format` parameter is deprecated; do not use it.
- **Never** pass `temperature`, `top_p`, or `top_k` — they are rejected on this model.
- Control depth with `output_config={"effort": "low"}` alongside `format`. Leave adaptive
  thinking on (the default); do not pass `thinking={"type": "disabled"}`.
- Import `anthropic` lazily inside `__init__` so `import memvara` works without it, and
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
