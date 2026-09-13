# How contradictions are resolved

When Memvara receives a new fact that conflicts with one it already has, it decides what
to do without ever asking a language model. This page explains how, and why that
deliberately avoiding a model call is the point, not a shortcut.

## The behavior

```python
mem.remember("Alice", "lives_in", "Berlin")
mem.remember("Alice", "lives_in", "Lisbon")

print([c.object for c in mem.get_all()])
# ['Lisbon']
```

Only the current value comes back — but nothing was deleted:

```python
for c in mem.history("Alice", "lives_in"):
    print(c.object, c.state)
# Berlin - ended
# Lisbon - live
```

## The three steps, none of which involve a model

**1. The kind of fact gets normalized.** Several different words can mean the same
underlying fact — `lives_in`, `resides_in`, `based_in`, and `moved_to` are all built-in
aliases of one predicate, so writing any of them for the same person updates the same
slot.

**2. The subject gets folded to a canonical form.** "Acme Corp.", "ACME, Inc.", and
"acme" are recognized as referring to the same entity before any comparison happens. This
step matters more than it might seem — without it, two spellings of an employer's name
would simply be two entirely separate slots, and nothing would ever conflict with
anything.

**3. Cardinality decides what happens.** Every kind of fact has a declared property
called *cardinality* — either single-valued (a new value replaces the old one) or
multi-valued (values accumulate). This is a property of the *kind* of fact, decided in
advance, not something Memvara infers on the spot for each write.

So the entire process is: normalize the kind of fact, fold the subject to a canonical
identity, look up the existing value for that exact slot, and — if the kind of fact is
single-valued — close the old value's time interval. Every step here is a lookup or a
comparison. There's nothing for a model to interpret.

## Why not use a model for this instead?

The obvious alternative design is: take the new fact, search for similar existing facts
using embeddings, and ask a model whether any of the top matches actually conflict with
it. This fails in two ways that have nothing to do with how good the model is:

- **It can miss entirely.** "I'm in Lisbon now" and "Berlin" don't have to be close to
  each other in embedding space — nothing guarantees the conflicting fact even shows up
  among the candidates the model gets to look at.
- **It isn't repeatable.** The exact same two facts, run through the exact same process
  twice, can come back with different resolutions on different runs. Nothing downstream
  can tell that happened.

Over time, that combination means a store can end up holding several conflicting values
for the same fact, each one surfacing depending on which phrasing happens to embed
closest to whatever question is being asked. Memvara's approach — deterministic,
structural resolution — guarantees the same two facts resolve the same way every time.

## One deliberate guard against overreach

Being fast and deterministic doesn't mean this process should be trusted blindly. If a
low-confidence guess (say, something extracted loosely from a passing remark) would
overwrite a value someone stated directly and confidently, Memvara doesn't let the guess
silently win. Instead, both values are kept side by side, and the write result explicitly
flags this as a **dispute** rather than a resolved conflict.

This matters because *the world changed* and *the record was wrong* are two very
different explanations for the same-looking situation, and getting that distinction wrong
by accident isn't something you can detect just by looking at the stored data afterward.
So the write path refuses to guess at it when confidence is low.

## Your own facts need their own rules

The built-in vocabulary Memvara ships with is aimed at personal-assistant facts — where
someone lives, works, or what they're allergic to. If you're storing something else
entirely, like engineering decisions, none of the built-in kinds of fact will match what
you're writing, and by default every undeclared kind of fact is treated conservatively:
multi-valued (so nothing ever automatically replaces anything) and slow to go stale (so a
fact from months ago still ranks as fresh). Neither is wrong exactly — they're the safe
defaults — but they're rarely what you actually want. See
[Define your own fact types](../how-to-guides/define-your-own-fact-types.md) to fix this
for your own domain.

## Related pages

- [Bitemporal memory, explained](bitemporal-memory-explained.md) — how the two clocks
  fit into this.
- [Provenance and trust](provenance-and-trust.md) — how a correction is recorded so the
  reason for it stays visible.
