# Quickstart

Five minutes, one file, no API key. At the end you will have asked one question at three
different instants and got three different correct answers.

```bash
pip install memvara
```

## 1. A store

```python
from datetime import datetime, timezone
from memvara import Memvara, NullLLM

UTC = timezone.utc
mem = Memvara("memory.db", user="alice", llm=NullLLM())
```

`"memory.db"` is a SQLite file; leave the path out for an in-memory store. `user=` sets
the default scope, so every read and write below is Alice's unless you say otherwise.
`llm=NullLLM()` asks for the offline configuration explicitly — see
[Installation](installation.md#what-you-get-with-no-model) for what that decides.

## 2. Three facts, on three days

```python
def at(month, day):
    return datetime(2026, month, day, tzinfo=UTC)

mem.remember("Alice", "lives_in", "Berlin",   valid_from=at(1, 10), recorded_at=at(1, 10))
mem.remember("Alice", "lives_in", "London",   valid_from=at(3, 15), recorded_at=at(3, 15))
mem.remember("Alice", "lives_in", "New York", valid_from=at(6, 2),  recorded_at=at(6, 2))
```

Two dates, because they answer different questions. **`valid_from` is when the fact
became true in the world.** **`recorded_at` is when this store was told.** They are equal
here because Alice told us on the day she moved. A fact that arrives late about the past
is where they differ, and that is the case a single `updated_at` column cannot represent.

Leave both out and `remember()` uses now for each, which is right for a fact you are
learning as it happens.

No model was called. `remember()` takes the triple already parsed, and `WriteReceipt`
reports `llm=0` to prove it.

## 3. What is true now

```python
[c.object for c in mem.get_all()]
# ['New York']

[r.text for r in mem.search("where does Alice live?")]
# ['Alice lives in New York']
```

One value, not three. `lives_in` is declared single-valued in the built-in schema, so
each new value closed the previous one's interval — an indexed lookup on
`(subject, predicate)`, not a similarity search and not a model call. See
[contradiction resolution](../concepts/contradiction-resolution.md).

## 4. What was true then

```python
[c.object for c in mem.get_all(as_of=at(3, 20))]
# ['London']

[c.object for c in mem.get_all(as_of=at(1, 20))]
# ['Berlin']
```

This is the part a vector store cannot do, and not for want of a feature: overwriting
Berlin with London destroys the only record that Berlin was ever the answer, and no
amount of retrieval quality recovers it afterwards.

`as_of=T` moves both clocks to `T`. There are two more, and they are the reason the axes
are separate:

```python
mem.get_all(valid_at=T)   # what we believe TODAY about how the world was at T
mem.get_all(known_at=T)   # what we believed at T, about the world as it is now
mem.get_all(as_of=T)      # both at T — what we believed at T, about T
```

A correction that arrives in August about June is invisible to `as_of=June`, because that
rewinds the belief clock past the correction. `valid_at=June` is how you see it. See
[bitemporal memory](../concepts/bitemporal-memory.md).

## 5. The record behind the answers

```python
[(c.object, c.state) for c in mem.history("Alice", "lives_in")]
# [('Berlin', 'ended'), ('London', 'ended'), ('New York', 'live')]
```

Each claim also carries the interval it held: `valid_from` and `valid_to` on the two ended
ones are 10 January to 15 March, and 15 March to 2 June. `valid_to` is `None` on the live
one.

Nothing was deleted. **`ended` means the world changed**; a value that was never true
reads `retired` instead, and `mem.forget()` is what writes that. Same timeline, different
reason, and an incident review needs to be able to tell them apart. See
[provenance](../concepts/provenance.md).

## 6. The narrated form

```python
mem.ask("where does Alice live?", at=at(3, 20)).text
```

```
where does Alice live?
  asked about 2026-03-20

Alice lives_in: London.
  It stopped being true 2026-06-02.
  Now: New York.
```

`ask()` composes three readings of every fact it touches — what is true now, what we
believe today was true then, and what this store *would have answered* then — and says so
when the last two differ. No model is consulted; every sentence is rendered from a stored
column.

## Run the whole thing

Everything above is [`examples/temporal_memory.py`](../../examples/temporal_memory.py),
which is run and asserted on by the test suite:

```bash
git clone https://github.com/memvara/memvara && cd memvara
python3 examples/temporal_memory.py
```

## Where to go next

- **You want the concepts.** [Bitemporal memory](../concepts/bitemporal-memory.md) is the
  one everything else rests on.
- **You want it in your editor rather than in your code.** [MCP](../integrations/mcp.md).
- **You want to write facts properly.** [Your first memory](first-memory.md) covers the
  source, the confidence, and the three different ways to correct a claim.
- **You want the whole surface.** [API reference](../API.md).

---

Previous: [Installation](installation.md) · Next: [Your first memory](first-memory.md)
