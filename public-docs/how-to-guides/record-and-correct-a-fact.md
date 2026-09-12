# How to record and correct a fact

This guide covers writing a fact properly — with its source and its dates — and then
shows the three different ways a fact can need correcting, because they mean three
different things and Memvara records them differently.

## Write a fact as a triple

Every fact in Memvara is a **claim**: a subject, a predicate, and an object. Think of it
as a small sentence — "Alice works at Acme" becomes `subject="Alice"`,
`predicate="works_at"`, `object="Acme"`.

```python
from memvara import Memvara, NullLLM

mem = Memvara("memory.db", user="alice", llm=NullLLM())
receipt = mem.remember("Alice", "works_at", "Acme")
```

`remember()` returns a `WriteReceipt`, which tells you exactly what happened:

```python
print(receipt)
# <WriteReceipt +1 ~0 -0 skip=0 llm=0 1.5ms>

print(receipt.added[0].id)
# 'cl_b9a...'
```

`+1` means one new claim was added. `llm=0` confirms no model was called — writing an
exact fact never needs one. If you write the exact same fact twice, the second write
doesn't add a new claim; it *reinforces* the existing one, and the receipt looks like
`<WriteReceipt +0 ~1 -0>` instead.

## Cite where a fact came from

A fact with no source is hard to trust later. Store the original message first, then
point the fact at it:

```python
from datetime import datetime, timezone

UTC = timezone.utc
january = datetime(2026, 1, 10, tzinfo=UTC)

turn = mem.add("I've just started at Acme.", role="user", ts=january)
receipt = mem.remember(
    "Alice", "works_at", "Acme",
    sources=turn.episode_ids,
    valid_from=january,
    recorded_at=january,
    confidence=0.9,
)
```

Now you can trace the belief back to the actual sentence it came from:

```python
provenance = mem.why(receipt.added[0].id)
print([e.text for e in provenance.episodes])
# ["I've just started at Acme."]
```

**The `role` you give a message decides whether Memvara tries to extract facts from it,
not just who gets credit for it.** Only messages marked `role="user"` are scanned for
facts. Use `role="system"` for a document, a log file, or anything you're pasting in
rather than something someone actually said — otherwise a quoted first-person sentence
inside a log ("the customer said 'I live in Toronto'") can be mistaken for the customer
actually telling *you* where they live.

## Set the two dates correctly

Every claim can carry two dates, and they answer different questions:

- **`valid_from`** — when the fact became true out in the world.
- **`recorded_at`** — when your system found out about it.

If you leave both out, Memvara uses the current time for both, which is correct when
you're recording something as it happens. Set them explicitly when you're entering
something you learned about *after the fact* — for example, backfilling last week's
support ticket into memory today. Get this wrong and Memvara will still accept the write,
but it will answer questions about "what was true last week" incorrectly, with nothing
in the output to tell you that happened.

```python
mem.remember(
    "api", "deploys_to", "Fly.io",
    valid_from=datetime(2026, 6, 12, tzinfo=UTC),   # became true in June
    # recorded_at defaults to now, which is fine if you're recording it today
)
```

If you set a date on one write to a fact, set dates on every write to that same fact.
Mixing dated and undated writes to the same fact can put them out of order — an undated
write is stamped "right now," which is later than any date you typed by hand.

## Correct a fact: three different situations, three different calls

This is the part worth getting right. All three corrections look similar if you only
glance at the result, but they mean different things, and getting them mixed up records a
false story about what actually happened.

Start from the same fact in all three examples:

```python
def fresh_store():
    mem = Memvara(user="alice", llm=NullLLM())
    mem.remember("Alice", "works_at", "Acme",
                 valid_from=january, recorded_at=january)
    return mem

march = datetime(2026, 3, 1, tzinfo=UTC)
may = datetime(2026, 5, 1, tzinfo=UTC)
```

### Situation 1: the fact was right, and the world changed

Alice changed jobs. The old fact was correct while it lasted; it just isn't current
anymore. Write the new value, and Memvara automatically closes out the old one:

```python
mem = fresh_store()
mem.remember("Alice", "works_at", "Kovac Labs", valid_from=march, recorded_at=march)

for c in mem.history("Alice", "works_at"):
    print(c.object, "-", c.state)
# Acme - ended
# Kovac Labs - live
```

`ended` means: this was true, and then the world moved on. Questions about January and
February still correctly say "Acme".

### Situation 2: the fact was right, and it has simply stopped, with no replacement yet

Alice left Acme and hasn't told you where she went next. There's no new value to write,
but the old one needs closing:

```python
mem = fresh_store()
mem.forget("Alice", "works_at", at=may, close="ended")
```

This closes out the fact — as of May, "Acme" is no longer current — without asserting
anything about what replaced it, because you don't know that yet.

### Situation 3: the fact was never true

Somebody mistyped the company name, or misheard what Alice said. Nothing about the world
changed here — the record itself was simply wrong from the moment it was written:

```python
mem = fresh_store()
mem.forget("Alice", "works_at", at=may)     # close="retired" is the default
```

This is `forget()`'s default behavior, because forgetting something is a decision you're
making now, not a claim that the world itself changed. If you already know a fact's exact
ID rather than its subject and predicate, `delete(claim_id, close=...)` does the same
thing for that one fact.

### None of these three delete anything

All three keep the original text and the record itself intact — you can always look back
and see what was believed, and why it changed. If you actually need to permanently erase
data — for example, to comply with a data-deletion request — that's a separate,
deliberate action:

```python
mem.erase(claim_id, sources=True)   # removes this claim's text and its source message
mem.purge()                         # removes everything in the current scope
```

See [Delete personal data](delete-personal-data.md) for the full picture, including how
to prove the data is actually gone.

## Teach Memvara your own kinds of facts

The facts Memvara understands out of the box — where someone lives, where they work, what
they like — are a personal-assistant vocabulary. If you're storing something else, like
engineering decisions, none of the built-in kinds will match, and every unrecognized kind
of fact defaults to "can have many values at once, and never goes stale." See
[Define your own fact types](define-your-own-fact-types.md) to fix that.

## Next steps

- [Ask about the past](ask-about-the-past.md) — how to ask what was true at a specific
  moment.
- [Delete personal data](delete-personal-data.md) — the difference between correcting a
  fact and actually erasing it.
