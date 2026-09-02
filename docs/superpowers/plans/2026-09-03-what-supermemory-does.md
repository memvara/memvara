# What Supermemory does, and why our retrieval work has stalled

**Date:** 2026-09-03
**Sources:** supermemory.ai/research/longmembench, plus offline replays of our own runs

## Their published number is 95%, not 81.6%

| category | Supermemory | memvara best (cap 15) |
| --- | ---: | ---: |
| single-session assistant | 100% | 87.5% |
| knowledge update | 99% | 62.8% |
| single-session user | 97% | 91.4% |
| multi-session | 93% | 56.8% |
| temporal reasoning | 91% | 84.7% |
| single-session preference | 90% | 53.3% |
| **overall** | **95%** | **70.5%** |

At roughly 720 mean tokens of injected context, against our 4,051 at cap 15. The gap is
not a few points of ranking. It is a different architecture.

## Their mechanism, in two stages

1. **Semantic search over memory titles.** They store atomic memories — single pieces of
   information, explicitly written so that ambiguous references are resolved — plus two
   dates per item, `documentDate` for when the conversation happened and `eventDate` for
   when the described event happened, plus links between memories marked `updates`,
   `extends` or `derives`.
2. **Injection of the original source chunk for each matched memory.** The memory is the
   index. The raw conversation behind it is what reaches the model.

Read against our results, that second stage explains the shape of everything we have
measured. We treat claims and episodes as two legs *competing for the same slots*: a claim
that matches spends a slot on its own normalised text, and the turn it came from has to win
a slot separately, on its own embedding. Supermemory spends the match on a claim and the
slot on the source.

Two of their three storage decisions are things memvara already has and does not use on
this benchmark. Bitemporality is the product's central claim, and `eventDate` versus
`documentDate` is precisely `valid_from` versus `recorded_at`. The `updates` / `extends` /
`derives` links are the graph leg, which sits at `w_graph=0.0` here because the join rate
on LongMemEval is 0.0%.

## The neighbour-window idea, tested and refuted

If what matters were the conversational context around a match, then injecting a match plus
its adjacent turns should buy cap-15's accuracy far more cheaply. Replaying the cap-15
ranked lists over the 193 questions whose gold answer string is literally present in the
haystack, so that "did the selected text contain the answer" is answerable:

| selection | answer present | est. tokens |
| --- | ---: | ---: |
| top-3 by rank | 42.5% | 650 |
| top-5 by rank | 54.4% | 1,226 |
| top-8 by rank | 62.7% | 2,156 |
| top-15 by rank | 69.4% | 4,379 |
| top-3 + 1 neighbour | 58.0% | 1,946 |
| top-5 + 1 neighbour | 66.8% | 3,007 |
| top-3 + 2 neighbours | 61.1% | 2,582 |
| top-5 + 2 neighbours | 67.4% | 3,969 |

Windows sit on the same token curve as simply taking more turns, within a point or two in
both directions. Three turns plus their neighbours reaches 58.0% for 1,946 tokens, where
eight turns by rank reach 62.7% for 2,156. **Adjacency is not the missing ingredient**, and
this cost nothing to establish.

So the difference is not which raw turns we select, at any budget or in any arrangement.
Six selection interventions have now been tested and none has beaten the token curve.

## The one thing that has never been tested properly

Ingest quality — and our single attempt at it is the weakest measurement in the whole set.

The `memvara-llmingest` arm ran gpt-5.4 extraction against the fast path and returned
exactly 0.0 difference, 68.8% against 68.8%. It ran on 16 questions taken as the first
eight by sorted id in each of two categories. Those 16 have a baseline accuracy of 68.8%,
against category averages of 26.3% for multi-session and 44.4% for temporal reasoning. The
sample is not merely small, it is drawn from the questions the baseline already answers,
where there is almost no headroom for any intervention to show up in.

That null result should not be relied on, and it is the result standing between us and the
one architectural difference we can actually name.

## What follows

1. **Re-run the ingest comparison on a random, adequately sized sample**, stratified across
   categories rather than taken by sorted id. This is the measurement that decides whether
   the gap is extraction or retrieval, and the current answer to it is unsound.
2. **Use claims as an index into episodes rather than as competing results.** A matched
   claim should pull its source turns into the context instead of spending a slot on its
   own text. That is their stage 2, it is a read-path change, and `why()` already returns
   `Provenance.episodes` so the link exists.
3. Not selection. Six arms is enough.
