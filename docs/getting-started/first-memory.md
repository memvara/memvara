# Your first memory

The [Quickstart](quickstart.md) wrote three facts with the minimum arguments. This page
writes one fact properly — with its source, its dates and its confidence — and then
corrects it three different ways, because *overtaken*, *stopped being true* and *was never
true* are three different events and the store records them differently.

Every block below is real output. The stores here are **in-memory** — `Memvara()` with no
path — so each section that says it starts fresh genuinely does; pass a filename to
persist. That distinction is load-bearing on this page rather than a detail: reopening the
same file would carry the previous section's claim forward, and a second write of the same
triple is not a second fact but a reinforcement, which changes what every receipt below
shows.

## A fact is a triple

```python
from memvara import Memvara, NullLLM

mem = Memvara(user="alice", llm=NullLLM())      # in-memory; pass a path to persist
receipt = mem.remember("Alice", "works_at", "Acme")
```

`subject`, `predicate`, `object`. The predicate is the part that carries meaning to the
engine: `works_at` is declared **single-valued** in the built-in schema, so a second
employer for Alice is a contradiction rather than an addition, and the store resolves it
without asking a model.

`remember()` returns a `WriteReceipt`, and reading it is how you find out what actually
happened:

```python
receipt
# <WriteReceipt +1 ~0 -0 skip=0 llm=0 1.5ms>

receipt.added[0].id
# 'cl_b9a...'
```

`+1` is one claim added; `llm=0` on every write is the design claim rather than a
coincidence of this example. **Read `added` rather than assuming it is non-empty**: write
the same triple twice and the second is a reinforcement, `<WriteReceipt +0 ~1 -0>`, with
`added` empty and `reinforced` holding the claim instead.

## Cite the turn it came from

A fact with no source is a fact nobody can check. Store the message first, then point the
claim at it. **Starting fresh**, so this is the first write of the triple:

```python
from datetime import datetime, timezone

UTC = timezone.utc
jan = datetime(2026, 1, 10, tzinfo=UTC)

mem = Memvara(user="alice", llm=NullLLM())      # in-memory; pass a path to persist
turn = mem.add("I've just started at Acme.", role="user", ts=jan)
receipt = mem.remember("Alice", "works_at", "Acme",
                       sources=turn.episode_ids,
                       valid_from=jan, recorded_at=jan,
                       confidence=0.9)
```

Now the belief traces back to text:

```python
p = mem.why(receipt.added[0].id)

[(e.ts.date().isoformat(), e.text) for e in p.episodes]
# [('2026-01-10', "I've just started at Acme.")]

p.derivation, p.extractor
# (<Derivation.USER: 'user'>, 'api')
```

And backwards, from the turn to everything it produced:

```python
[c.text for c in mem.produced(turn.episode_ids[0])]
# ['Alice works at Acme']
```

**`role=` decides what is extracted, not just who is credited.** Only `"user"` turns are
read by the extractor. Pass `role="system"` for a document, a log or a pasted file — it is
stored and citable, and nothing is extracted from it. This matters more than it looks: the
deterministic matcher strips quotation marks before matching, so a first-person sentence
quoted inside a log becomes a fact about whoever pasted it unless you say the role.

## The two dates

- **`valid_from`** — when it became true out in the world.
- **`recorded_at`** — when this store was told.

Omit both and each defaults to now, which is right for something you are learning as it
happens. Set them when you are backfilling: a finding from last week written with today's
`valid_from` records a claim that was never true across its own interval, and the store
will then answer historical questions wrongly with no symptom at write time.

**Dating one write and not the next is the same mistake wearing a disguise.** An undated
write takes `recorded_at=now`, which is later than any date you type — so a store whose
first fact is undated and whose second is stamped March will order them the other way
round from the way you wrote them. If any write in a slot carries dates, give them all
dates.

`valid_to` is the fourth field and you rarely set it by hand — a superseding write closes
it for you.

## Correcting it: three writes, three different reasons

This is the part worth getting right, because all three look identical in a store that
keeps one value per fact, and none of them is a deletion.

Each of the three starts from the same one-fact store — they are **alternatives, not a
sequence**. Run them in order against a single store and the second finds the slot already
closed:

```python
def fresh():
    mem = Memvara(user="alice", llm=NullLLM())
    mem.remember("Alice", "works_at", "Acme", valid_from=jan, recorded_at=jan)
    return mem

mar = datetime(2026, 3, 1, tzinfo=UTC)
may = datetime(2026, 5, 1, tzinfo=UTC)
```

### The value was right and has been overtaken

She changed employer. Write the new value; the old one's interval closes.

```python
mem = fresh()
mem.remember("Alice", "works_at", "Kovac Labs", valid_from=mar, recorded_at=mar)

[(c.object, c.state) for c in mem.history("Alice", "works_at")]
# [('Acme', 'ended'), ('Kovac Labs', 'live')]
```

`ended` — the world changed. Acme still answers `valid_at=` queries about January and
February, because it was true then.

### The value was right and has stopped being true, with no successor

She left Acme and has not said where she went. There is no new value to write, but the old
one has stopped:

```python
mem = fresh()
mem.forget("Alice", "works_at", at=may, close="ended")
# [Claim(object='Acme', state='ended', valid_to=2026-05-01)]
```

`close="ended"` closes **world time**: it says the fact finished being true on 1 May.

### The value was never true

Somebody misheard, or the wrong name was typed. Nothing about the world changed — the
*record* was wrong:

```python
mem = fresh()
mem.forget("Alice", "works_at", at=may)          # close="retired" is the default
# [Claim(object='Acme', state='retired', invalidated_at=2026-05-01)]
```

`retired` closes **belief time** and leaves the world interval untouched, which is exactly
right: we stop believing it from here on, at every world-time, and we do not assert that
anything out there happened. `forget()` defaults to this because forgetting is something
the holder of a memory does, not something the world does.

`delete(claim_id, close=…)` is the same pair of choices for one claim instead of a whole
slot.

**Give `at=` an instant after the fact began.** `valid_to` is clamped forward to
`valid_from`, so closing a March claim at January stores `valid_to == valid_from`: a
zero-length interval, which holds at no instant and is therefore absent from every answer
on either clock. Nothing raises. The commonest way to hit this is not typing a past date
but forgetting to date the *write* — an undated `remember()` takes `valid_from=now`, and
then any `at=` you name is in its past.

### None of the three deletes anything

All three leave the text, the sources and the embedding in place, which is what makes the
audit trail worth having. For a GDPR Article 17 request — where the text itself must cease
to exist — the calls are `erase(claim_id, sources=True)` and `purge()`, and they are
separate calls rather than a flag on these, because they are not variations of one
operation. See [Two meanings of "delete", kept apart](../API.md#two-meanings-of-delete-kept-apart).

## Teach it your vocabulary

The built-in predicates are a personal-assistant vocabulary. If you are storing
engineering facts, none of them match — and an undeclared predicate takes the safe default
twice over: multi-valued, so nothing supersedes, and slow-decaying, so this morning's
deploy still ranks as fresh in two years.

```python
from memvara import Cardinality, Memvara, PredicateRegistry, PredicateSpec, Volatility
from memvara.schema import BUILTIN_PREDICATES

registry = PredicateRegistry(BUILTIN_PREDICATES + (
    PredicateSpec(name="auth_strategy",
                  cardinality=Cardinality.ONE,      # supersedes; MANY accumulates
                  volatility=Volatility.SLOW),      # STATIC | SLOW | FAST
))
mem = Memvara(user="alice", registry=registry, llm=NullLLM())
```

Two vocabularies ship with the package — `engineering` and `decisions` — and
`load_all_specs("engineering,decisions")` reads them. A declaration outranks a guess, so a
pack corrects a store that already classified something wrongly rather than only shaping a
fresh one. The MCP server takes the same string as `MEMVARA_PREDICATES`.

Packs are TOML, parsed with `tomllib`, so **loading one needs Python 3.11**; it raises with
the reason on 3.10, which this package otherwise supports. Declaring the same predicates
inline as above works everywhere, and is what
[`examples/coding_agent.py`](../../examples/coding_agent.py) does for exactly that reason.

---

Previous: [Quickstart](quickstart.md) · Next: [Why Memvara?](../concepts/why-memvara.md)
