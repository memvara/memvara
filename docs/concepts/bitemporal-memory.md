# Bitemporal memory

**Every fact carries two independent dates: when it was true in the world, and when this
store came to believe it.** That is the whole idea, and everything else in this library
follows from it.

```python
mem.remember("Alice", "lives_in", "Lisbon",
             valid_from=datetime(2026, 3, 15, tzinfo=UTC),   # world clock
             recorded_at=datetime(2026, 3, 22, tzinfo=UTC))  # belief clock
```

Alice moved on 15 March. We found out on 22 March. Those are different facts about the
same fact, and a store with one `updated_at` column has to throw one of them away.

## Why one date is not enough

Pick either column and a class of question becomes unanswerable:

- **Keep only "when it was true."** You can say Alice lived in Lisbon from 15 March. You
  cannot say what your agent would have told a customer on 18 March — which is the
  question an incident review asks.
- **Keep only "when we learned it."** You can say the record changed on 22 March. You
  cannot absorb a fact that arrives late about the past without lying about when it held.

Bitemporal is not a feature on top of temporal. It is the smaller of the two claims made
honest: with one clock, a late correction and a real change look identical.

## The three reads

Eight reads take the same three time keywords: `search`, `get_all`, `count`,
`history`, `why`, `produced`, `neighborhood` and `paths_between`. `recall()`, `get()`
and `since()` take none of them, and `ask()` spells it `at=`.

```python
mem.get_all(valid_at=T)   # what we believe TODAY about how the world was at T
mem.get_all(known_at=T)   # what we believed at T, about the world as it is now
mem.get_all(as_of=T)      # both clocks at T — what we believed at T, about T
```

`as_of=T` is exact sugar for `valid_at=known_at=T`. Passing it alongside either axis
raises rather than quietly picking one.

The middle one is the one a single instant cannot ask, and it is worth an example.

### The correction that `as_of` cannot see

```python
jan  = datetime(2026, 1,  4, tzinfo=UTC)
mar1 = datetime(2026, 3,  1, tzinfo=UTC)
mar22= datetime(2026, 3, 22, tzinfo=UTC)

mem.remember("user", "lives_in", "Rome",   valid_from=jan,  recorded_at=jan)
mem.remember("user", "lives_in", "Berlin", valid_from=mar1, recorded_at=mar22)
```

Alice moved to Berlin on 1 March. Nobody told the store until 22 March. Now ask about
15 March:

```python
t = datetime(2026, 3, 15, tzinfo=UTC)

[c.object for c in mem.get_all(valid_at=t)]
# ['Berlin']   — what we believe today about how 15 March was

[c.object for c in mem.get_all(as_of=t)]
# []           — see below; this is not the same as "we believed nothing"
```

`valid_at` is the useful one here, and it is the answer `as_of` cannot give: rewinding the
belief clock to 15 March puts it before the 22 March correction, so the Berlin claim is
not yet believed — while Rome's row already carries the end date the *later* write stamped
on it, so it is out of its world interval too. Both are excluded and the read is empty.

That is a real property of a row-level predicate, not a bug, and it is why the next
section exists: **`ask()` is the read that answers "what would this store have said on
15 March", and it is the only one that does.** It walks the supersession chain, which the
row carries, and dates each ending at the instant the replacement was recorded rather than
at the instant it took effect. `get_all(as_of=…)` is a scope-wide predicate over columns
with no timeline in front of it, and making it walk the chain would turn every read into a
per-slot join. The two answer different questions and both are documented as doing so —
see [Internals](../INTERNALS.md#ask-reconstructs-an-ending-the-row-cannot-date).

The window between 1 March and 22 March is the thing worth naming: it is exactly the
period in which your system was confidently wrong, and both reads above are needed to see
it.

## `ask()` composes all three into a sentence

```python
mem.ask("where do they live?", at=datetime(2026, 3, 15, tzinfo=UTC)).text
```

```
user lives_in: Berlin.
  On 2026-03-15 this store would have said Rome, and that is what anyone acting
  on it then acted on. The difference was recorded 2026-03-22, 7 days after the
  instant you asked about.
```

Three readings of every fact it touches — what is true now, what we believe today was
true then, and what this store *would have answered* then — plus the day the record
changed. **No model composes that.** Every sentence is rendered from a stored column, so
it costs a query and cannot hallucinate a date.

`Answer` carries one `Reading` per fact slot, each holding `now`, `then` and `stated`.
The last two differing is the finding; `text` is the narration.

## The three states, and why there are three

Closing a claim is not one operation. Which clock stops records *why* it stopped:

| State | valid time | transaction time | Means |
|---|---|---|---|
| `live` | open | open | currently believed, currently true |
| `ended` | closed at `valid_to` | open | **the world changed** — it was true, and then it wasn't |
| `retired` | untouched | closed at `invalidated_at` | **the record was wrong** — it was never true |

```python
[(c.object, c.state) for c in mem.history("Alice", "lives_in")]
# [('Berlin', 'ended'), ('Oslo', 'retired'), ('Lisbon', 'live')]
```

Berlin was where she lived until she moved. Oslo was a mistake — a mishearing, a typo,
somebody else's record. Lisbon is where she lives. A store with one "deleted" flag reports
the same thing for Berlin and Oslo, and *served a value that expired* is a different
incident from *served a value that was never true*.

Nothing is deleted in any of the three. `history()` and every `as_of` read still see all
of them. Deletion — the kind where the text itself ceases to exist — is `erase()` and
`purge()`, deliberately separate calls. See
[Two meanings of "delete", kept apart](../API.md#two-meanings-of-delete-kept-apart).

## The states do not tile the store

Worth knowing before you write a report against `stats()`: `live`, `ended` and `retired`
are not a partition. A claim can be both ended and retired — the world moved on *and* the
record turns out to have been wrong — so `live + ended + retired` does not equal
`claims`. `claims` is the only total. [Internals](../INTERNALS.md#the-three-states-do-not-tile-the-store)
has the query-level detail.

## Writing the dates correctly

The one mistake with no symptom at write time is **backfilling without `valid_from`**.

```python
# Wrong: records that the deploy target has been Fly.io since right now,
# which was never true.
mem.remember("api", "deploys_to", "Fly.io")

# Right.
mem.remember("api", "deploys_to", "Fly.io",
             valid_from=datetime(2026, 6, 12, tzinfo=UTC))
```

Both writes succeed. Both look identical in `get_all()`. Only the second answers
`as_of=July` correctly, and nothing tells you which one you made except reading the
timeline back.

---

Previous: [Why Memvara?](why-memvara.md) · Next: [Contradiction resolution](contradiction-resolution.md)
