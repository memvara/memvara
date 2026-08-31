# Contradiction resolution

**A contradiction here is an indexed lookup, not a similarity search and not a model
call.** Same two facts, same result, every run — which is what "deterministic" means in
the sentence *deterministic contradiction resolution*.

```python
mem.remember("Alice", "lives_in", "Berlin")
mem.remember("Alice", "lives_in", "Lisbon")

[c.object for c in mem.get_all()]
# ['Lisbon']
```

One value, not two. Nothing was deleted:

```python
[(c.object, c.state) for c in mem.history("Alice", "lives_in")]
# [('Berlin', 'ended'), ('Lisbon', 'live')]
```

## How it decides

Three steps, none of which involves a model.

**1. The predicate is normalised.** `lives_in`, `resides_in`, `based_in`, `moved_to` and
`city` are aliases of one predicate in the built-in schema, so writing two of those
spellings for one person is still one slot.

```python
mem.remember("user", "lives_in", "Berlin")
mem.remember("user", "resides_in", "Lisbon")
[c.object for c in mem.get_all()]
# ['Lisbon']
```

**2. The entity is folded.** Surface forms collapse to one key before the lookup, so two
spellings of an employer are one employer:

```python
from memvara import entity_key
entity_key("Acme Corp.") == entity_key("ACME, Inc.") == entity_key("acme")
# True
```

This is the step that makes the keyed lookup fire at all. Without it, `Acme Corp.` and
`ACME, Inc.` are two slots and nothing contradicts anything.

**3. Cardinality decides.** Every predicate carries a declared cardinality, and it is a
schema property rather than an inference:

| Cardinality | Meaning | Examples |
|---|---|---|
| `ONE` | single-valued — a new value closes the old one | `lives_in`, `works_at`, `job_title`, `timezone` |
| `MANY` | multi-valued — values accumulate | `likes`, `speaks`, `allergic_to`, `owns_pet` |

So the whole resolution is: normalise the predicate, fold the entity, look up
`(subject, predicate)` in an index, and if the predicate is `ONE`, close the interval of
whatever is there. It is a database operation.

## Why this is not a top-k problem

The alternative design — embed the new memory, retrieve the nearest existing ones, ask a
model whether it conflicts with any of them — fails in two ways that have nothing to do
with model quality:

- **It can miss.** *"I'm in Lisbon now"* and *"Berlin"* need not be near each other in
  embedding space. If the conflicting memory falls outside top-k, the contradiction is
  never seen and both values stay live.
- **It is not repeatable.** The same two facts can resolve differently on two runs, and
  nothing downstream can tell.

Six months of that and the store holds three cities for one person and returns whichever
embeds closest to the question. The [README's *Against mem0 specifically*](../../README.md#against-mem0-specifically)
has the measured comparison against `mem0ai`, including the correction that mem0 2.x
makes one model call rather than two — and that its add path emits only `ADD` events, so
conflicting values are linked rather than retired.

## The guard: a guess cannot quietly overwrite a statement

Resolution being cheap does not make it safe to run on anything. A low-confidence
extraction that would displace a value somebody stated outright is kept **beside** it
rather than replacing it, and the receipt names both:

```python
receipt.disputed      # -> [Dispute(...)]  both values live, neither one silently won
```

Overwriting there would have recorded that the world changed, when nothing had. That
distinction — *the world changed* versus *the record was wrong* — is the one mistake in
this library that cannot be found by reading the data afterwards, which is why the write
path refuses to guess at it.

## Your domain needs its own vocabulary

The built-in predicates are a personal-assistant set: where somebody lives, where they
work, what they are allergic to. **A store of engineering facts matches none of them**,
and an undeclared predicate takes the safe default twice over — multi-valued, so nothing
supersedes, and slow-decaying, so this morning's deploy still ranks as fresh in two years.

The safe default is the right default (dropping a fact that turns out not to conflict
destroys information; keeping two that do only degrades ranking) and it is still not what
you want:

```python
from memvara import Cardinality, PredicateRegistry, PredicateSpec, Volatility
from memvara.schema import BUILTIN_PREDICATES, load_all_specs

registry = PredicateRegistry(
    BUILTIN_PREDICATES
    + load_all_specs("engineering,decisions")     # the two packs that ship
    + (PredicateSpec(name="auth_strategy",
                     cardinality=Cardinality.ONE,
                     volatility=Volatility.SLOW),))
```

Or, for the MCP server, the same string in the environment:

```bash
MEMVARA_PREDICATES=engineering,decisions memvara-mcp
```

**A declaration outranks a guess**, so a pack corrects a store that already classified
something wrongly rather than only shaping a fresh one. It is forward-only: it changes
what supersedes on the next write and retires nothing already stored.

Declared vocabularies are TOML and need Python 3.11 (`tomllib`); everything else here
works on 3.10.

## Where to see it

[`examples/temporal_memory.py`](../../examples/temporal_memory.py) is resolution on a
single-valued builtin. [`examples/coding_agent.py`](../../examples/coding_agent.py)
declares one and shows why the declaration is the difference between a supersession and
an accumulation.

---

Previous: [Bitemporal memory](bitemporal-memory.md) · Next: [Provenance](provenance.md)
