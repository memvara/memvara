# How to delete personal data

Memvara distinguishes between *correcting* a fact and *permanently erasing* it, because
they answer different real-world situations. This guide covers both, and shows how to
verify that a genuine deletion actually happened.

## First: is this a correction, or a deletion request?

If you're recording that a fact has changed, or that it was recorded incorrectly, you
want a **correction**, not a deletion. Corrections keep the historical record intact —
which is what lets you later explain why an agent said something, or what your system
believed at a given moment — and they're covered fully in
[Record and correct a fact](record-and-correct-a-fact.md).

You want an actual **deletion** only when the requirement is that the underlying text
itself must cease to exist — the clearest example is a data-subject deletion request
under privacy law (such as GDPR Article 17), where "we stopped believing it, but kept the
record" isn't sufficient.

## Delete one fact

```python
mem.erase(claim_id, sources=True)
```

This removes:

- the claim itself,
- its entry in the full-text search index (which stores the actual words, not just a
  pointer to them),
- its embedding (a raw vector can be reversed to recover something close to the original
  text, so it has to go too), and
- with `sources=True`, the original message the fact was extracted from — but only if no
  other surviving fact still cites that same message. One message can be the source for
  several facts, so this is checked rather than assumed.

## Delete everything in a scope

```python
mem.purge()
```

This erases every claim, message, and index entry for the current user/tenant/agent/
session scope. It returns a dictionary with counts of what was removed, so you can log
exactly what happened.

## Verify a deletion actually took effect

Don't just trust that `erase()` or `purge()` worked — ask Memvara to check for you:

```python
proof = mem.prove_erased(claim_id)
```

`prove_erased()` re-checks every table the content could theoretically still be sitting
in — the claim record, the search index, and the embedding store — and returns proof that
none of them still hold it. This is the call to use before telling a user, or an auditor,
that their data is actually gone.

## Why this is two separate calls rather than one flag

It would be simpler, on paper, to have one `delete()` call with a flag for "and also
erase the text." Memvara deliberately keeps these as separate operations, because they
answer genuinely different questions and one is reversible while the other isn't:

| | Correction (`forget`, `delete`, `end`) | Deletion (`erase`, `purge`) |
|---|---|---|
| Reversible | The old value is still visible in history | No — the text is gone |
| What it says | "We no longer believe this" or "this stopped being true" | "This data no longer exists anywhere in the store" |
| Right for | Fixing a mistake, recording that the world changed | A genuine data-erasure requirement |

Conflating the two would mean a routine correction could accidentally destroy the audit
trail that makes corrections useful in the first place. Keeping them apart means a
one-line typo fix never has to look like — or risk becoming — an irreversible deletion.

## Next steps

- [Record and correct a fact](record-and-correct-a-fact.md) — the three ways to correct a
  fact without erasing anything.
- [Provenance and trust](../explanation/provenance-and-trust.md) — why keeping a
  correction's history matters, and what it lets you answer later.
