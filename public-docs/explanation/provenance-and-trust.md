# Provenance and trust

Every fact Memvara stores traces back to the exact text it came from, and every source
message traces forward to everything it produced. This page explains why that link
matters, and the specific vocabulary Memvara uses to describe how a fact stopped being
current — because there is more than one reason a fact can stop being current, and mixing
them up hides real information.

## Tracing a fact back to its source

```python
provenance = mem.why(claim_id)

print([e.text for e in provenance.episodes])
# ["Decision: migrate service-to-service auth from API keys to OAuth 2.0..."]

print(provenance.derivation, provenance.extractor)
# (Derivation.USER, 'api')

print([c.text for c in provenance.superseded])
# ['checkout-service auth strategy API keys']
```

`why()` returns the claim, the exact source message it came from, how it was derived
(directly stated versus extracted from something more casual), and — if it replaced an
earlier fact — what that earlier fact was. And the reverse lookup exists too:

```python
mem.produced(episode_id)   # every fact that was extracted from this one message
```

When an AI agent says something wrong, this pair of lookups is how you find out exactly
which stored fact caused it, and where that fact came from in the first place. Without a
trace like this, "the agent said something wrong" and "the agent's memory is wrong" are
indistinguishable, and neither is fixable.

## Storing prose without extracting facts from it

Sometimes you want to keep a document, log, or transcript searchable and citable without
treating it as something a user is telling the system right now. Memvara's `role`
parameter controls exactly this: only messages marked as coming from the user are scanned
for facts. A message marked as a system document is stored, kept, and available as a
source for `why()` and `produced()` — but nothing is automatically extracted from it.

This distinction matters more than it might look like it does. Without it, a first-person
sentence quoted *inside* a pasted document — for example, a support log that includes "the
customer said 'I live in Toronto'" — could be mistakenly recorded as a fact about whoever
pasted the log in, rather than about the customer the quote was actually describing.

## Three different words, three different things that happened

This is the vocabulary the rest of Memvara is built on, and using the wrong one for what
actually happened records a false story about the change — one that nothing downstream
can detect and correct later, because the data itself looks the same either way.

| Word | What it means | How it's recorded |
|---|---|---|
| **ended** | The fact was true, and then the world changed. | A new value replacing an old one, or an explicit "this has stopped being true" correction. |
| **retired** | The fact was never actually true — the record itself was wrong. | An explicit "this was a mistake" correction — the default when you correct something. |
| **erased** | The underlying text has been permanently and irreversibly removed. | A deliberate, separate deletion action. |

The first two are the ones people mix up, and the mix-up is genuinely invisible
afterward if it happens: a system that reports "retired" for a fact that had simply
stopped being true — rather than one that was actually wrong from the start — leaves
anyone reading that record with no way to tell what actually occurred. **A correction is
never the same thing as a deletion**, and nothing in the first two rows above removes any
data — the full history stays visible through `history()` and through any query about a
past point in time.

## Erasure removes the actual bytes

Retiring or ending a fact is deliberately *not* the same as satisfying a genuine
data-deletion requirement, because the text itself is still sitting there, still
readable. That's exactly the wrong outcome for something like a GDPR Article 17 request,
where the requirement is that the data actually cease to exist.

`erase()` and `purge()` exist specifically for that case, and they remove the claim
itself, its entry in the search index (which stores the actual words, not just a
reference to them), and its embedding — because a raw embedding vector can, in practice,
be reversed to recover something close to the original text, so it has to go too, not
just the visible copy. `prove_erased()` re-checks every place the content could
theoretically still be sitting, and returns confirmation that it isn't. See
[Delete personal data](../how-to-guides/delete-personal-data.md) for the practical guide.

## Confidence travels with every claim

Alongside the source and the reason for any change, every claim also carries a confidence
score — how sure the system was when it wrote the fact. This is what allows the guard
described in
[How contradictions are resolved](how-contradictions-are-resolved.md#one-deliberate-guard-against-overreach)
to work: a low-confidence guess is never allowed to silently overwrite something stated
with high confidence.

## Related pages

- [How contradictions are resolved](how-contradictions-are-resolved.md)
- [Delete personal data](../how-to-guides/delete-personal-data.md)
