# How search works

Memvara's search combines exact keyword matching with meaning-based matching, adjusts
ranking based on how likely a fact is to have gone stale, and can explain exactly why any
result ranked where it did. This page explains the reasoning behind each of those three
properties.

## Why keyword matching, alongside meaning-based matching?

It's tempting to assume meaning-based (semantic, embedding-driven) search is simply
better than keyword matching, and for a lot of natural-language questions, it is. But
meaning-based matching specifically struggles with the exact things an AI agent often
needs to find verbatim: error codes, version numbers, IDs, and surnames. A search for
`ERR_7734_TLSHANDSHAKE` is a perfect match for keyword search and can easily be a
near-miss for something trying to match it by meaning, because there's no broader
"meaning" for a string like that to be close to.

So Memvara runs both approaches in parallel on every search and merges their results
using a technique called **Reciprocal Rank Fusion**, which combines the two result
*rankings* rather than trying to combine their raw scores. This matters because keyword
scores and meaning-based similarity scores aren't measured on comparable scales to begin
with — normalizing them onto a shared scale would mean guessing at a conversion rate that
has no principled answer.

## Why ranking accounts for how stale a fact is likely to be

A single, global "prefer recent results" rule is wrong in both directions at once. It
would push someone's decade-old birthplace down in the results, purely for being old, even
though it's exactly as true today as it was ten years ago. At the same time, it would let
last month's now-outdated task assignment keep outranking this week's current one, because
"last month" still counts as "recent" on any single, uniform scale.

Memvara avoids this by treating staleness as a property of the *kind* of fact, not of the
store as a whole — this is the `volatility` setting described in
[Define your own fact types](../how-to-guides/define-your-own-fact-types.md). A
birthplace can be declared to essentially never go stale, while "what someone is
currently working on" can be declared to go stale within about a week. This is also the
second reason declaring your own vocabulary matters: an undeclared kind of fact defaults
to a slow, two-year staleness period, which means something that changed this morning can
still rank as perfectly fresh months from now.

## Asking a search about the past

Search accepts the same time-travel keywords described in
[Bitemporal memory, explained](bitemporal-memory-explained.md) — `as_of`, `valid_at`, and
`known_at`. This isn't a filter applied after the fact to today's search results; it's
the search itself, re-run against the store exactly as it stood at that point in time.
This is what lets you answer "what would this system have surfaced, if someone had asked
this same question back then?"

## Why every result can explain its own score

A search result that can't explain why it ranked where it did is nearly impossible to
debug when something looks wrong, and a memory system that quietly returns the wrong fact
is one of the hardest categories of bug to track down, because nothing crashes and
nothing looks obviously broken. So every result Memvara returns carries a breakdown of
exactly how its score was built — the keyword-match rank and score, the meaning-match
rank and score, the recency multiplier applied, the fact's own stored confidence, and the
final combined score. Reading that breakdown tells you *why* something ranked highly (or
didn't), rather than requiring you to guess.

## Why the default meaning-based matching is limited

Out of the box, with no extra packages installed, Memvara's meaning-based matching is a
fast, fully offline, keyword-based fallback rather than a real semantic model — it will
not connect "doctor" with "physician," and it only understands Latin-script text. Text
written in Chinese, Japanese, Korean, Arabic, or Hebrew produces no usable meaning-based
signal at all under this default. This is a genuinely honest trade-off, not a hidden
gap: such text is still stored, and still fully searchable by exact keyword or by the
kind of fact it represents — it simply won't be found by meaning-based matching alone,
and Memvara warns you when this happens rather than silently dropping the fact.
Installing `memvara[local-embed]` and configuring a real embedding model removes this
limitation.

## Related pages

- [Search your memory](../how-to-guides/search-your-memory.md) — the practical guide to
  using search and getting a prompt-ready block of context.
- [Known limitations](known-limitations.md) — the full, honest list of what Memvara does
  and doesn't handle well.
