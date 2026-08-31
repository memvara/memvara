# Adding your memory system

The benchmark knows nothing about any particular system. Everything system-specific lives
behind one interface, and adding yours means writing one file.

We would rather have your adapter than not, including — especially — if it beats memvara
in a category. A benchmark that only its author's system does well on is a marketing
asset, not a benchmark, and the [Results](README.md#results) table already has a row where
a numpy baseline beats memvara.

## The interface

Five methods and two attributes. `benchmarks/agent_memory/adapters/base.py` is the
authority; this is the shape.

```python
from benchmarks.agent_memory.adapters.base import Ask, MemoryAnswer, Usage, wants_a_date


class MyMemory:
    name = "my-system"        # what --system takes, and what appears in the result file
    version = "2.1.0"         # the version of your system, not of this adapter

    def reset(self, predicates):
        """Discard all memory. `predicates` maps a relation name to a PredicateDecl,
        whose `.cardinality` is "one" or "many". Called once, before any event."""

    def remember(self, event):
        """Take delivery of one observation. Events arrive in `recorded_at` order.
        `event.valid_from` may be earlier than `event.recorded_at`, and sometimes is."""

    def query(self, ask) -> MemoryAnswer:
        """Answer one question. Return MemoryAnswer() to say you do not know."""

    def usage(self) -> Usage:
        """Cost counters, or Usage() if you count nothing. Leave a field None rather
        than reporting an unmeasured quantity as zero."""

    def close(self):
        """Release anything you opened."""


def build(**kwargs):
    return MyMemory()
```

Run it:

```bash
# from anywhere importable, without touching this repository
python -m benchmarks.agent_memory --system mypackage.adapters:build

# or add one line to registry.py and use a short name
python -m benchmarks.agent_memory --system my-system
```

## What you are given

**`reset(predicates)`** hands you the published schema. `cardinality == "one"` means a new
value replaces the old one; `"many"` means values accumulate. This is input, not something
to infer — inferring it is an interesting problem and not the one being measured.

**`remember(event)`** hands you a `MemoryEvent` with both a sentence and a
`(subject, predicate, object)` triple, plus:

| Field | What it is |
|---|---|
| `recorded_at` | when your system is being told |
| `valid_from` | when the fact became true in the world — sometimes much earlier |
| `valid_to` | set only where the dataset closes an interval outright; usually `None` |
| `source` | a label like `hr_directory` or `adr_027`. Gold answers to `provenance` are these |
| `confidence` | the reporter's reliability. Nothing is scored on it |
| `text` | the sentence, for whatever indexing you do |

Store the source label. Provenance questions are answerable by any system that keeps it
beside the value, and are not a memvara-shaped requirement.

**`query(ask)`** hands you an `Ask`:

| Field | What it is |
|---|---|
| `category` | one of the ten. Branch on it freely — every system gets the same one |
| `question` | the natural-language question |
| `probe` | `(subject, predicate)` when the dataset names the slot, else `None` and you have to find it |
| `at` | the **world** instant the question is about. Always set |
| `known_at` | the **belief** instant, when the question asks what you would have said then. `None` means "as you understand it today" |
| `about` | which value a `provenance` or date question is about |

There is no field carrying the gold answer, and there is no way to reach one.

## Answering

```python
MemoryAnswer(value="London")                       # a single value
MemoryAnswer(value="2026-03-15")                   # a date, ISO-8601 only
MemoryAnswer(values=("English", "German"))         # a set; order ignored
MemoryAnswer()                                     # you do not know
```

Return the **value**, not a sentence. Scoring accepts a value inside a short sentence, but
only when no competing value for the same slot also appears, so prose buys nothing and can
cost you a point.

`support=(...)` is optional: ids of whatever justified the answer. Never scored, printed
in the failure report, and excluded from the reproducibility check — so per-process ids
are fine.

**Abstain rather than guess.** Both score as failures outside the `negative` category, and
an abstention makes the failure report readable instead of misleading.

**Use `wants_a_date(ask)`** to tell a "when did this happen" question from "what would you
have said on this date". Both live in `knowledge_time`, they want different answer types,
and three adapters got the distinction wrong independently before it moved into `base.py`.

## The rules your adapter must follow

1. **No hardcoded answers**, and nothing keyed on a question id.
2. **Go through your system's public API.** Reaching past it produces a number nobody can
   reproduce with the published one.
3. **Disclose any external call.** If your system needs a model or a network service, say
   so in your pull request: which model, which version, which temperature. Note that a
   run with a model in it is not deterministic, and `--repeat-check` will say so.
4. **Report cost honestly.** `None` for what you do not measure. `0` is a claim.
5. **Implement the published supersession rules if your system can express them.** They
   are in `timeline.py` and in [README.md](README.md#the-supersession-rules-published).
   They are available to every adapter, which is what stops any one of them being an
   advantage.

## Before you open the pull request

```bash
python -m pytest tests/test_agent_memory_bench.py -q
python -m benchmarks.agent_memory --system my-system --repeat-check
python -m benchmarks.agent_memory --system my-system --system memvara --compare
```

Include in the body: the leaderboard row, the environment (Python, platform, your
system's version), and anything your adapter had to decide that the dataset did not decide
for it. If a category scores badly, say why you think so — a known limitation stated is
worth more than a number without one.

If your system beats memvara somewhere, that row goes in the README table as measured.
