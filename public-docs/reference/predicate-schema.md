# Predicate schema reference

This page is the full reference for declaring your own kinds of facts. For a
task-oriented walkthrough, see
[Define your own fact types](../how-to-guides/define-your-own-fact-types.md).

## `PredicateSpec`

```python
from memvara import Cardinality, PredicateSpec, Volatility

PredicateSpec(
    name="auth_strategy",
    cardinality=Cardinality.ONE,
    volatility=Volatility.SLOW,
    aliases=(),           # optional: other spellings that mean the same predicate
)
```

| Field | Meaning |
|---|---|
| `name` | The predicate's canonical name, e.g. `"lives_in"`. |
| `cardinality` | `Cardinality.ONE` (single-valued — a new value replaces the old one) or `Cardinality.MANY` (multi-valued — values accumulate). |
| `volatility` | `Volatility.STATIC`, `Volatility.SLOW`, or `Volatility.FAST` — how quickly a fact under this predicate should be treated as stale in search ranking. |
| `aliases` | Other names that should be treated as the same predicate — for example, `lives_in`, `resides_in`, and `based_in` can all be declared as aliases of one predicate. |

## `Cardinality`

| Value | Meaning | Typical use |
|---|---|---|
| `Cardinality.ONE` | A new value closes the interval of the previous one. | `lives_in`, `works_at`, `job_title` — things a person has exactly one of at a time. |
| `Cardinality.MANY` | New values are kept alongside existing ones. | `speaks`, `allergic_to`, `depends_on` — things that can have several true values at once. |

**If you never declare a predicate, it defaults to `MANY`.** This is a deliberately safe
default: it's better to keep two facts that turn out not to actually conflict than to
silently discard one that did matter.

## `Volatility`

| Value | Approximate half-life | Typical use |
|---|---|---|
| `Volatility.STATIC` | ~100 years (effectively never decays) | `born_in` — a ten-year-old fact should rank exactly as fresh as a new one. |
| `Volatility.SLOW` | ~2 years | `works_at`, `lives_in` — changes occasionally, stays relevant for a long time. |
| `Volatility.FAST` | ~7 days | `working_on` — last week's task shouldn't outrank this week's in search results. |

**If you never declare a predicate, it defaults to `SLOW`.** This means a fact about
something that changes daily — like which server is currently handling traffic — will
still rank as "fresh" in search results long after it's stopped being true, unless you
declare it as `FAST`.

## Registering predicates

```python
from memvara import PredicateRegistry
from memvara.schema import BUILTIN_PREDICATES

registry = PredicateRegistry(BUILTIN_PREDICATES + (
    PredicateSpec(name="auth_strategy", cardinality=Cardinality.ONE,
                  volatility=Volatility.SLOW),
))

mem = Memvara("memory.db", registry=registry)
```

`BUILTIN_PREDICATES` is the personal-assistant vocabulary Memvara ships with by default
— things like `lives_in`, `works_at`, `job_title`, `allergic_to`, and `speaks`. Include it
in your registry unless you specifically want to replace it entirely.

## Shipped vocabulary packs

Two ready-made vocabularies ship with the package as TOML files, loaded by name:

```python
from memvara.schema import load_all_specs

load_all_specs("engineering,decisions")
```

- **`engineering`** — `deploys_to`, `current_host`, `git_state`, `build_status`,
  `version`, `endpoint`, `owner` (all single-valued), plus `depends_on`, `rejected`,
  `known_defect`, `blocked_by` (multi-valued — a project having several dependencies, or
  several known defects, is normal, and a later one doesn't make an earlier one untrue).
- **`decisions`** — `decided` and `observed`, both multi-valued, for recording what an
  agent decided during its own work and why.

`load_all_specs()` reads TOML files and needs Python 3.11 or later (`tomllib`, the module
it uses, doesn't exist before that version). Declaring the equivalent predicates directly
with `PredicateSpec` in Python code, as shown above, works on every Python version Memvara
supports.

You can also pass a path to your own TOML file instead of a shipped pack name — the
format matches the two shipped packs.

## For the MCP server

The same pack names and file paths work as a comma-separated environment variable:

```bash
MEMVARA_PREDICATES=engineering,decisions memvara-mcp
```

## A declaration is forward-looking

Declaring a predicate that already has facts stored under it doesn't retroactively change
those existing facts — it changes how *future* writes to that predicate behave. This
means adding a missing declaration is always safe to do after the fact: it corrects the
store's understanding going forward without silently rewriting history that was recorded
under the previous, looser rules.

## Related pages

- [Define your own fact types](../how-to-guides/define-your-own-fact-types.md) — a
  task-oriented walkthrough of the same material.
- [How contradictions are resolved](../explanation/how-contradictions-are-resolved.md) —
  why cardinality is what decides what happens when two facts conflict.
