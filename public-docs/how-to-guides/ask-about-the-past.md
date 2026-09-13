# How to ask about the past

Memvara can answer questions not just about right now, but about any point in the past —
and it can even tell you what it *would have said* at a past moment, which can differ
from what was *actually true* at that moment. This guide shows how to ask each of these
questions correctly.

## The two clocks

Every fact in Memvara carries two independent dates:

- **When it was true in the world** — for example, the day someone actually moved house.
- **When your system found out about it** — which can be later, sometimes much later.

Most of the time these are the same day. They start to matter the moment somebody tells
you something *late*. If a customer emails in August saying "by the way, we moved offices
back in June," the fact became true in June but you only learned it in August. A system
that only tracks one date has to throw one of those two pieces of information away.

## Three ways to ask about time

Memvara gives you three keywords, and they answer three different questions:

```python
mem.get_all(valid_at=T)   # "What do we believe TODAY about how things stood at T?"
mem.get_all(known_at=T)   # "What did we believe AT THE TIME, about how things are now?"
mem.get_all(as_of=T)      # Both at once: "What did we believe AT THE TIME, about how things stood AT THE TIME?"
```

These three keywords work the same way across every read that supports time travel:
`search`, `get_all`, `count`, `history`, `why`, `produced`, `neighborhood`, and
`paths_between`.

### `valid_at` — what do we now believe was true then?

This is the one you reach for most often. It answers "using everything we know today,
what was actually the case on this date?" — including facts you learned about later that
apply to that date.

```python
june = datetime(2026, 6, 15, tzinfo=UTC)
mem.get_all(valid_at=june)
```

### `known_at` — what did we believe at the time?

This answers "if you had asked our system this question back then, using only what it
knew at that moment, what would it have said?" This is useful for reconstructing what an
agent or a person actually acted on at a given moment — even if that turns out to have
been outdated.

### `as_of` — both, at the same instant

`as_of=T` is shorthand for asking both questions at the same value of T — it tells you
what the system believed, at time T, about how things stood at time T. This is what
most people mean by "what did we think, back then" and it's the right default for most
historical questions.

## The tricky case: a late correction

Here is where the difference between the three actually matters. Suppose Alice moved to
Berlin on 1 March, but nobody told your system until 22 March.

```python
mem.remember("Alice", "lives_in", "Rome",   valid_from=jan1,  recorded_at=jan1)
mem.remember("Alice", "lives_in", "Berlin", valid_from=mar1,  recorded_at=mar22)

mid_march = datetime(2026, 3, 15, tzinfo=UTC)

mem.get_all(valid_at=mid_march)   # -> ['Berlin']   correct, using what we now know
mem.get_all(as_of=mid_march)      # -> []           see below
```

`valid_at` correctly says Berlin, because as of today, that's what we understand was true
on 15 March. `as_of`, on the other hand, rewinds the belief clock all the way back to 15
March — before the correction was even entered — and comes back empty. That's not a bug:
it's the honest answer to "what did the system actually know at that exact moment,"
which happens to be nothing useful, because the correction hadn't arrived yet. This is
exactly the ten-day window during which anyone asking your system would have gotten the
wrong (or no) answer, and it's worth being able to see that.

## Get a plain-English answer instead of raw data

If you want a sentence rather than a list of records, use `ask()`. It composes all three
readings above into a narrated answer, and it tells you when the "then" answer and the
"now" answer disagree:

```python
answer = mem.ask("where does Alice live?", at=mid_march)
print(answer.text)
```

```
where do they live?
  asked about 2026-03-15

user lives_in: Berlin.
  On 2026-03-15 this store would have said Rome, and that is what anyone
  acting on it then acted on. The difference was recorded 2026-03-22,
  7 days after the instant you asked about.
```

Nothing here calls a language model — every sentence is built directly from stored dates,
so it can't invent a date that isn't real.

## See a fact's full history

If you want every value a fact has ever held, rather than a single point in time, use
`history()`:

```python
for claim in mem.history("Alice", "lives_in"):
    print(claim.object, claim.state)
# Berlin  ended
# Oslo    retired
# Lisbon  live
```

`ended` means the fact was true and stopped being true because the world changed.
`retired` means the fact was never true — the record itself was wrong. See
[Record and correct a fact](record-and-correct-a-fact.md) for how each is written.

## Next steps

- [Search your memory](search-your-memory.md) — the same time-travel keywords work on
  search, not just on direct lookups.
- [What is bitemporal memory?](../explanation/bitemporal-memory-explained.md) — the
  reasoning behind having two clocks instead of one.
