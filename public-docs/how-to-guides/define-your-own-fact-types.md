# How to define your own fact types

Memvara ships with a vocabulary built for personal-assistant facts — where someone lives,
where they work, what they're allergic to. If you're storing something else — engineering
decisions, product data, anything domain-specific — none of the built-in kinds will match
what you're writing, and by default every unrecognized kind of fact is treated as "can
have many values at once, and never goes stale." This guide shows you how to declare your
own.

## Why this matters

Every kind of fact (Memvara calls this a **predicate** — the "lives_in" part of "Alice
lives_in Berlin") has two properties attached to it:

- **Does a new value replace the old one, or add to it?** "Where someone lives" should
  have one current answer. "What languages someone speaks" should accumulate.
- **How quickly should this fact be treated as stale in search results?** A person's
  birthplace should never seem stale. What they're currently working on should stop
  outranking newer facts within a week or two.

If you never declare a predicate, Memvara has to guess, and it guesses conservatively:
every new value is kept alongside the old ones (nothing is thrown away, which is the safe
default), and every fact decays slowly (also the safe default — a false "this looks
stale" is worse than a false "this still looks fresh"). But conservative isn't the same
as correct. On a real store used for engineering facts, measured directly: 95% of the
facts written used a predicate outside the built-in vocabulary, and every one of them
accumulated instead of replacing anything it should have replaced.

## Declare a predicate inline

```python
from memvara import Cardinality, Memvara, PredicateRegistry, PredicateSpec, Volatility
from memvara.schema import BUILTIN_PREDICATES

registry = PredicateRegistry(BUILTIN_PREDICATES + (
    PredicateSpec(
        name="auth_strategy",
        cardinality=Cardinality.ONE,    # a new value replaces the old one
        volatility=Volatility.SLOW,     # decays over roughly two years
    ),
))

mem = Memvara("team.db", registry=registry, llm=NullLLM())
```

`Cardinality.ONE` means single-valued (a new value supersedes); `Cardinality.MANY` means
multi-valued (values accumulate). `Volatility` has three settings:

| Volatility | Roughly means | Example |
|---|---|---|
| `Volatility.STATIC` | Practically never goes stale | someone's birthplace |
| `Volatility.SLOW` | Stays fresh for around two years | where someone works, where they live |
| `Volatility.FAST` | Goes stale within about a week | what someone is currently working on |

## Use one of the built-in vocabularies

Memvara ships two ready-made vocabularies you can load instead of writing your own from
scratch:

- **`engineering`** — facts like `deploys_to`, `current_host`, `version`, `depends_on`,
  `known_defect`, and `rejected`, tuned for infrastructure and technical facts.
- **`decisions`** — `decided` and `observed`, for recording what an AI agent decided and
  why, both allowed to accumulate over time.

```python
from memvara.schema import BUILTIN_PREDICATES, load_all_specs

registry = PredicateRegistry(
    BUILTIN_PREDICATES + load_all_specs("engineering,decisions")
)
```

**This needs Python 3.11 or later**, because the vocabularies are stored as TOML files
and Python's built-in TOML reader (`tomllib`) only exists from 3.11 onward. If you're on
Python 3.10, declare the equivalent predicates inline using `PredicateSpec` as shown
above — that works on every version Memvara supports.

## For the MCP server

If you're connecting an AI assistant rather than writing Python directly, set the same
vocabulary names in an environment variable when you start the server:

```bash
MEMVARA_PREDICATES=engineering,decisions memvara-mcp
```

A comma-separated mix of shipped pack names and paths to your own TOML files also works.

## Declaring a predicate later corrects the past, not just the future

If you already have a store with facts under an undeclared predicate, adding a
declaration for it doesn't quietly change what's already there — it changes how future
writes to that predicate behave. This is deliberate: a declaration means "I know what
this kind of fact is now," and it should correct a store's understanding going forward
without silently retiring facts that were written under the old, looser rules.

## Next steps

- [Record and correct a fact](record-and-correct-a-fact.md) — writing facts once your
  vocabulary is set up.
- [How contradictions are resolved](../explanation/how-contradictions-are-resolved.md) —
  the full mechanics of why cardinality decides what happens on a conflicting write.
