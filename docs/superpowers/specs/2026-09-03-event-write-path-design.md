# Design: an event-capable write path

**Date:** 2026-09-03
**Status:** proposed
**Approach:** A (shared event-time resolver, both write tiers) together with B (quantities
as first-class claim fields)

## The problem, in two defects

**The world clock is fed the wrong time.** Both write paths hardcode `valid_from=ep.ts` —
`write/fast.py:283` and `write/pipeline.py:972` — and the extractor's output schema never
asks when the described event happened. So for every extracted claim, `valid_from` (world
clock) and `recorded_at` (belief clock) carry the same information: when we were told. "I
ran 30 minutes yesterday", said on the 20th, is stored as an event on the 20th.

Bitemporality is what this library is for. On the write path it is currently inert for
anything an extractor produced.

**The vocabulary cannot hold a quantity.** All 23 built-in predicates are personal-profile
attributes — `likes`, `prefers`, `goal`, `works_at`, `lives_in`. None carries an amount, a
unit, or a count. Running gpt-5.4 extraction instead of the fast path yields 97.3 claims per
question against 3.5, across 27 predicates against 9, and the shape does not change:
`user | goal | save for new camera`, `user | had_experience | surfing`. The turn each came
from can say how many hours were spent. The claim cannot.

`docs/superpowers/plans/2026-09-03-the-schema-mismatch.md` has the evidence.

## What is being built

Three pieces, in dependency order.

1. **`memvara/write/when.py`** — a pure resolver from a temporal expression plus an anchor
   to an instant. Shared by both write tiers.
2. **Event time on both write paths** — the fast path routes what it already strips; the
   LLM path gains an optional field in the extractor's output schema.
3. **`Claim.amount` and `Claim.unit`** — two optional fields, `SCHEMA_VERSION` 8 → 9, plus
   a `packs/events.toml` predicate pack.

### 1. The resolver

```python
def resolve(expression: str, anchor: datetime) -> datetime | None
```

Deterministic, pure, no I/O, no model. Returns `None` when it cannot resolve, which is a
first-class outcome rather than an error: the caller falls back to the anchor.

It handles the expressions the fast path already recognises, because that set was derived
from real writes rather than from a grammar: `yesterday`, `N days|weeks|months|years ago`,
`last|this|next week|month|year|summer|winter|spring|fall|autumn`, `in YYYY`.

**Resolution is deliberately coarse where the expression is coarse.** "Last month" resolves
to the start of the previous calendar month, not to a point 30 days back, and the claim
records that instant. A range type would be more honest and is out of scope — see
*Deliberately not in this design*.

**The model never does date arithmetic.** The LLM returns the expression it saw, as text;
this function turns it into an instant. Models are unreliable at date arithmetic and
reliable at spotting "three weeks ago", so the error surface stays in code that a unit test
can pin against a fixed anchor.

### 2. Event time on both paths

**Fast path.** `_FILLER` at `write/fast.py:80` already matches exactly these expressions and
discards them with `sub("")` at line 195, because leaving them in fragments slot identity —
"Lisbon" and "Lisbon last month" would be two facts. That reasoning is correct and does not
change. The regex becomes capturing, the matched text goes to `resolve()`, and the object is
normalised exactly as before. Slot identity, `fact_key` and `value_key` are untouched.

**LLM path.** The extractor's per-item output schema gains one optional string field, `when`,
holding the expression as seen in the turn. `pipeline.py` passes it through `resolve()`
against `ep.ts`.

**Both fall back to `ep.ts`** when nothing is stated, when the model omits the field, or when
the resolver returns `None`. That fallback is the current behaviour, so a turn that states no
time is stored exactly as it is today.

### 3. Quantities as claim fields

`Claim` gains two optional fields:

```python
amount: float | None = None
unit: str | None = None
```

Persisted as two nullable columns, `SCHEMA_VERSION` 8 → 9 with a `_migrate_to_v9` that adds
them. This is a smaller change than it looks: `put_claim` and `get_claim` already carry the
whole `Claim`, so **no `Store` protocol method is added or changed**. `docs/ROADMAP.md`
records what protocol changes cost here — three members added in #26 broke `mypy` in a
downstream repository whose CI was off, and nothing went red — and this design deliberately
does not pay that.

They stay out of `fact_key` and `value_key`. A quantity is a property of an observation, not
part of the slot it occupies, and putting it in the identity would make every distinct amount
a distinct fact that supersedes nothing.

`packs/events.toml` declares the vocabulary, loaded with `MEMVARA_PREDICATES=events`. It is a
pack and not a change to the built-in set for the reason `packs/decisions.toml` already
argues at length: a pack is reversible, a schema decision is not.

## What does not change

- **Claims with no stated time behave exactly as today.** `valid_from` is still `ep.ts`.
- **Slot identity, dedup and supersession keys.** `fact_key` and `value_key` are computed
  from the same inputs as now.
- **`recorded_at`.** The belief clock was always right and is not touched.
- **The `Store` protocol.** No method added, removed or re-signed.
- **Existing stored claims.** They are not reinterpreted or migrated. A claim written before
  this change means what it meant: `valid_from` is the conversation's timestamp. Event time
  cannot be recovered without re-extraction, and inventing it would be exactly the forged
  history this library exists to prevent.

## The risk this design carries

**Supersession outcomes change, and that is the point.** `write/reconcile.py` orders
competing claims by `valid_from` — line 516 splits them into newer and older on it, and line
282 takes `min(c.valid_from for c in newer)` as the boundary. Once `valid_from` is the event
time, "I moved to Lisbon last year", said today, correctly sorts before a claim recorded two
years ago and stating a move three years ago.

That is the behaviour bitemporality promises and it is not what the store does today. It also
means a store that begins writing event times will reconcile some slots differently from one
that has not. This must be stated in `docs/UPGRADING.md` rather than discovered.

**The failure mode to test for is a resolver that is confidently wrong.** A misresolved
expression writes a false world time that looks exactly like a true one — the class of
silent failure this repository's telemetry module exists for. Hence: the resolver returns
`None` rather than guessing, and every supported expression gets a test with a fixed anchor.

## Testing

- **Resolver unit tests** against a fixed anchor, one per supported expression form, plus
  the unresolvable cases that must return `None`.
- **Fast path**: a turn with a temporal tail gets the resolved `valid_from`; the object and
  `fact_key` are byte-identical to what the same turn produces today. The second assertion is
  the one that protects slot identity.
- **LLM path**: a stubbed extractor returning `when` produces the resolved `valid_from`; one
  omitting it falls back to `ep.ts`.
- **Reconciliation**: two claims whose event order differs from their conversation order
  supersede in event order. This is the test that would have failed before the change.
- **Round-trip**: `amount`/`unit` survive `put_claim` → `get_claim`, and a v8 file opened by
  this build migrates and reads back.
- Docstring examples run under `--doctest-modules`, so any example added here must execute.

## Documentation shipping with the code

`CHANGELOG.md`, `docs/UPGRADING.md` (the supersession change and the new fields),
`docs/INTERNALS.md`, and the `packs/events.toml` header. The MCP tool descriptions in
`memvara/server/tools.py` need review only if `amount`/`unit` become settable through
`memory_remember`, which this design does not propose.

## Deliberately not in this design

- **A range or interval type for coarse expressions.** "Last month" is a month, not an
  instant. Modelling that means an interval on the world clock, which is a far larger change
  to every comparison in `reconcile.py`. Resolving to the interval's start is a lossy but
  honest approximation, and the loss is recorded here so it does not read as an oversight.
- **SQL-side aggregation of amounts.** Summing hours or money in the database would need new
  `Store` methods. Retrieval hands the claims to a model that can add them. Revisit only if
  measurement shows the model's arithmetic is the limit.
- **Retroactive event times for existing claims.** Covered above: it would be forged history.
- **Changing the built-in predicate set.** The pack is the extension point.
