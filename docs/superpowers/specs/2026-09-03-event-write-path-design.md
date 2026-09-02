# Design: an event-capable write path

**Date:** 2026-09-03
**Status:** proposed, revised after review
**Approach:** a shared deterministic temporal resolver used by both write tiers, together
with quantities as first-class claim fields

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

## What `valid_from` means after this change

Not "the event time". **`valid_from` is the earliest resolved temporal boundary at which the
claim is asserted to hold.** The distinction matters because three things get conflated
otherwise: when an event happened, when a state began, and when the observation was made.
The third is `recorded_at` and always was. The first two are both boundaries, and this field
holds the earliest one known.

**The interval's end already exists and this design does not populate it.** `Claim.valid_to`
is a field (`types.py:527`), the store round-trips it, and `reconcile.py` sets it when a
claim is superseded. So "I worked at Google from 2019 to 2022" is *representable today* —
extraction simply never writes the end boundary. Teaching it to is a follow-up with its own
design, because a closed interval written at extraction time interacts with supersession in
ways a single boundary does not. Recording that here so the gap reads as deferred rather
than unnoticed.

## Temporal precision, and why it is not optional

Resolving "last month" to the first of the previous month turns a MONTH into an INSTANT and
tells nothing downstream that it did. Anything later reading that claim sees
`valid_from = 2026-08-01` and can only conclude the claim became true on the 1st of August.
That is false, and it is false in the silent way this repository's telemetry module exists to
catch.

So the resolver returns a boundary **and** the precision it was resolved at, and the claim
stores both:

```python
temporal_precision: Literal["instant", "day", "week", "month", "season", "year"] | None
```

`None` means the boundary was not resolved from an expression — the `ep.ts` fallback — and is
what every claim written before this change has.

| expression | `valid_from` | `temporal_precision` |
| --- | --- | --- |
| (none stated) | `ep.ts` | `None` |
| yesterday | previous day, 00:00 | `day` |
| three weeks ago | that day, 00:00 | `day` |
| last month | first of previous month | `month` |
| last year | first of previous year | `year` |
| in 2024 | 2024-01-01 | `year` |

The normalised boundary is still what orders claims. Precision records how much of that
boundary was invented by normalisation.

## Temporal ordering semantics

This is the section the first draft was missing, and it decides what supersession does.

- **`recorded_at`** orders belief. It is unaffected by this change and remains total.
- **`valid_from`** orders the world, **when the two boundaries are confidently comparable**.
- **`temporal_precision`** decides whether they are.

Two boundaries are **not** confidently orderable when the coarser one's interval contains
the finer one's boundary — a claim resolved to "2025" could be anywhere in 2025, so it
cannot be said to precede a claim at 2025-12-01. In that case reconciliation falls back to
`recorded_at` for precedence.

The worked example from review, which is exactly the case that motivates the rule:

```
2025-12-01  "I live in London."             valid_from=2025-12-01  precision=None
2026-09-03  "I moved to Lisbon last year."  valid_from=2025-01-01  precision=year
```

Ordering on `valid_from` alone puts Lisbon *before* London, so Lisbon supersedes nothing and
the store still believes London. With the rule, 2025-12-01 falls inside the year 2025, the
boundaries are not comparable, precedence falls to `recorded_at`, and Lisbon supersedes
London — which is what a person reading those two sentences concludes.

This is precision paying for itself on the first case that exercises it.

## What is being built

1. **`memvara/write/when.py`** — the resolver, pure and shared by both tiers.
2. **Event time on both write paths.**
3. **`Claim.amount`, `Claim.unit`, `Claim.temporal_precision`** — three optional fields,
   `SCHEMA_VERSION` 8 → 9.
4. **`packs/events.toml`** — the vocabulary.

### 1. The resolver

```python
def resolve(expression: str, anchor: datetime) -> tuple[datetime, Precision] | None
```

Deterministic, pure, no I/O, no model. Returns `None` when it cannot resolve, which is a
first-class outcome rather than an error: the caller falls back to the anchor with
`temporal_precision=None`. **It returns `None` rather than guessing** — a misresolved
expression writes a false world time indistinguishable from a true one.

It handles the expressions the fast path already recognises, because that set came from real
writes rather than from a grammar: `yesterday`, `N days|weeks|months|years ago`,
`last|this|next week|month|year|summer|winter|spring|fall|autumn`, `in YYYY`.

**Timezone contract.** Calendar arithmetic runs in the anchor's timezone when the anchor is
aware, and a naive anchor is treated as UTC — which is not a new rule but `types.as_utc`,
the convention every persisted timestamp in this library already follows. The resolver never
performs an implicit local-time conversion.

One consequence worth stating because it constrains what any future caller can expect: **the
store round-trips timestamps through epoch floats, so the anchor's timezone is not preserved
past the write.** Resolution happens before persistence, so a caller who needs local-day
semantics must pass an aware `ep.ts`; what lands in the store is the resulting instant. DST
transitions are therefore a property of the anchor's zone at resolution time and are handled
by `zoneinfo` arithmetic, not by day-length assumptions.

### 2. Event time on both paths

**Fast path.** `_FILLER` at `write/fast.py:80` already matches exactly these expressions and
discards them with `sub("")` at line 195, because leaving them in fragments slot identity —
"Lisbon" and "Lisbon last month" would be two facts. That reasoning is correct and does not
change.

The regex becomes capturing, but **the regex does not decide temporal semantics.** The fast
path returns the value and the mention separately:

```python
value, mention = _split_value_and_mention(object_text)
```

where `mention` is the matched expression or `None`. `when.resolve()` interprets it. Keeping
that boundary is what stops `fast.py` accumulating a temporal grammar the day someone wants
`from X until Y` or `since X`.

**LLM path.** The extractor's per-item output schema gains one optional string field, `when`,
documented in the prompt as:

> Optional. The temporal expression as it appears in the text, identifying the single
> earliest temporal boundary associated with this claim. Copy the expression; do not compute
> a date, and do not infer an exact date when the source gives only a coarse one. Omit it if
> the text states no time.

The contract is deliberately narrow: one boundary, source text only. It keeps date
arithmetic in testable code, and it stops the field growing into a prompt-side temporal
parser the first time someone meets an interval.

**Both fall back to `ep.ts`** with `temporal_precision=None` when nothing is stated, the
model omits the field, or the resolver returns `None`. That is today's behaviour, so a turn
stating no time is stored exactly as it is now.

### 3. Quantities

```python
amount: float | None = None
unit: str | None = None
```

**At most one quantity per claim.** "I ran 5 km in 30 minutes" carries two independent
quantities and must be emitted as two claims where predicate semantics allow, never as
`amount=5, unit="kilometer", object="5 km in 30 minutes"`. The extractor prompt states this;
a claim is one observation of one measurable thing.

**`unit` is canonical, singular, lowercase**: `minute`, `hour`, `kilometer`, `usd`. The
extractor normalises `mins`/`min`/`minutes` to `minute` on the way in. **Unit conversion is
out of scope** — nothing turns 120 minutes into 2 hours, and `amount=30, unit="minute"`
versus `amount=30, unit="usd"` are interpreted by whoever reads the predicate and unit
together. A measurement abstraction can arrive later if a measurement need does.

No `quantity_type`. `amount + unit` is sufficient for a first version and the alternative is
designing a type system before there is a second consumer.

**Quantities stay out of `fact_key` and `value_key`.** `amount` and `unit` are attributes of
a claim observation and so do not participate in claim identity; identity remains
subject / predicate / object / scope / polarity, and the quantity describes the particular
observation. Putting a quantity in the identity would make every distinct amount a distinct
fact that supersedes nothing — so `user | weight | ... | 70kg` followed by `71kg` would
accumulate rather than update.

### 4. The vocabulary pack

`packs/events.toml`, loaded with `MEMVARA_PREDICATES=events`. A pack rather than a change to
the built-in set, for the reason `packs/decisions.toml` argues at length: a pack is
reversible and a schema decision is not.

**Its scope is stated in the file header, because two different things want to live here.**
The pack declares **event predicates** — things that happened at a time, such as `ran`,
`visited`, `bought`, `attended`, `completed`. It does **not** declare quantity predicates
(`distance`, `duration`, `cost`, `weight`): a quantity is carried by `amount`/`unit` on an
event claim, not by a predicate of its own. Without that line the file becomes forty
unrelated measurements in six months.

## Persistence

Three nullable columns, `SCHEMA_VERSION` 8 → 9, with a `_migrate_to_v9` following the
existing idempotent, file-shape-driven pattern. **No `Store` protocol method is added or
changed**: `put_claim` and `get_claim` already carry the whole `Claim`. `docs/ROADMAP.md`
records what protocol changes cost here — three members added in #26 broke `mypy` in a
downstream repository whose CI was off, and nothing went red — and this design does not pay
that.

## What does not change

- **Claims stating no time behave exactly as today**: `valid_from = ep.ts`.
- **Slot identity, dedup and supersession keys** — `fact_key` and `value_key` are computed
  from the same inputs as now.
- **`recorded_at`** — the belief clock was always right.
- **The `Store` protocol.**
- **Existing stored claims are not reinterpreted or migrated.** A claim written before this
  change means what it meant: `valid_from` is the conversation's timestamp, and
  `temporal_precision` is `None`. Event time cannot be recovered without re-extraction, and
  inventing it is precisely the forged history this library exists to prevent. This belongs
  near the top of the `UPGRADING.md` entry, not in a footnote — an upgrade that silently
  rewrote world history would be the worst possible outcome of this change.

## Testing

- **Resolver units** against a fixed anchor, one per supported form, asserting both the
  boundary and the precision.
- **Adversarial temporal input**, as its own category, all of which must resolve to `None`
  rather than to a confident wrong answer: `not yesterday`, `I don't remember when`,
  `sometime last year`, `around March`, `a few weeks ago`, `recently`. Plus `now`, `today`,
  `tomorrow`, `next month`, which resolve but must not be mistaken for past boundaries.
- **Timezone and DST**: an aware anchor either side of a DST transition, and a naive anchor
  taking the `as_utc` path.
- **Fast path**: a turn with a temporal tail gets the resolved boundary, and its object and
  `fact_key` are byte-identical to what the same turn produces today. The second assertion
  is what protects slot identity.
- **LLM path**: a stubbed extractor returning `when` resolves it; one omitting it falls back.
- **Reconciliation integration**, which is where the design's risk lives:
  - two claims whose event order differs from their conversation order supersede in event
    order;
  - the Lisbon/London case above, asserting that incomparable precisions fall back to
    `recorded_at`;
  - "I used to live in London, but moved to Lisbon last year";
  - "I lived in Lisbon last year but now live in London" — the same facts in the opposite
    order, which must not produce the same answer.
- **Round-trip**: the three new fields survive `put_claim` → `get_claim`, and a v8 file
  opened by this build migrates and reads back.
- Docstring examples execute under `--doctest-modules`.

## Documentation shipping with the code

`CHANGELOG.md`, `docs/UPGRADING.md` (the supersession change, the new fields, and the
no-retroactive-rewrite guarantee), `docs/INTERNALS.md`, and the `packs/events.toml` header.
The MCP tool descriptions in `memvara/server/tools.py` need review only if the new fields
become settable through `memory_remember`, which this design does not propose.

## Deliberately not in this design

- **Populating `valid_to` at extraction.** The field exists and is unused by extraction; a
  closed interval written at extraction time interacts with supersession differently from a
  single boundary and deserves its own design.
- **A range type on the world clock.** Precision records that a boundary was coarse without
  requiring every comparison in `reconcile.py` to become interval arithmetic.
- **Unit conversion**, **`quantity_type`**, **SQL-side aggregation of amounts**, **new
  `Store` methods**, **retroactive event times**, and **expanding the built-in predicate
  set**. Each would turn a focused change into a temporal-model rewrite.
