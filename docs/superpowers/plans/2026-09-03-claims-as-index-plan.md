# Plan: claims as an index into the turns behind them

**Date:** 2026-09-03
**Branch:** `claude/event-write-path`
**Status:** proposed, not yet implemented

## The one difference we named and never built

Supermemory publishes 95% on LongMemEval-S at roughly 720 tokens of injected context.
Their read path is two stages: semantic search over atomic memories, then **injection of
the original source chunk behind each matched memory**. memvara's best is 70.5% at 4,051
tokens, and its read path ranks claims and turns as competitors for the same slots.

Four arms say the claim loses that competition:

| arm | claims' share of slots | median ctx | accuracy |
| --- | ---: | ---: | ---: |
| fast path | 53% | 775 | 48.1% |
| gpt-5.4 profile claims | 72% | 1,758 | 46.2% |
| event schema | 68% | 1,679 | 44.2% |
| turns only | 0% | 1,169 | 46.2% |

And across the retrieval arms accuracy tracked **gold turns retrieved** — 2.2 turns at
41.8%, 6.1 at 61.0%, 10.0 at 70.5% — not claims, not sessions reached.

So a matched claim is plausibly worth more as a pointer than as a result. This change
spends the match on the claim and the slot on its sources.

## What changes

One option on `HybridRetriever`, `claims_as_index: bool = False`, off by default.

When on, and only when `include_episodes` is set:

1. Claims are retrieved and ranked exactly as now — same legs, same fusion, same `_rank`.
2. Each ranked claim is replaced by the episodes its `sources` names, hydrated through the
   existing `_hydrate_episodes`, each inheriting the claim's score so the ranking is
   preserved rather than recomputed.
3. Those are merged with the episode leg's own results, deduplicated by episode id, and
   cut to `k`.
4. The claim itself does not appear in the output. It was the index entry.

Nothing else moves. No `Store` protocol change: `Claim.sources` and `_hydrate_episodes`
already exist, and `why()` is built on the same link.

## Why this and not a bigger change

It is the smallest change that tests the hypothesis. The alternative reading of the four
arms — that claims are simply worthless on this corpus — is already covered by the
turns-only arm, which scored 46.2%, no better than leaving them in. If claims are worth
anything, it is as a route to the right turns, and that is exactly what this measures.

## Risks, and what each would look like

- **Fewer distinct turns, not more.** Several claims often cite one turn, so N claims can
  collapse to far fewer than N episodes and the prompt ends up shorter than before.
  Mitigation is measurement, not code: report distinct turns retrieved alongside accuracy.
  If turns fall, this hurts for the reason the arms already established.
- **Score inheritance is a guess.** A claim's score measures how well the *claim* matched,
  and it is being used to rank a turn. The alternative — re-scoring each sourced turn
  against the query — is a second embedding pass per result and defeats the point.
  Recorded as a known approximation.
- **Dangling sources.** `erase` removes a turn and leaves claims citing it. A missing id
  must drop out silently rather than raise inside ranking.
- **It interacts with `max_per_source`.** Both reshape the episode head. They are measured
  separately; combining them is a later question.

## How it gets measured, and the mistake not to repeat

The 52-question arms could only detect a difference of about 19 points, and produced three
results within noise that were twice written up as conclusions. This one is sized to an
effect first:

- **200 questions per arm**, stratified across all six categories. At a ~50% base rate that
  detects roughly a 10-point difference, which is the smallest gap worth acting on here.
- **Against cap-15 (70.5%)**, which is memvara's real best, not against the fast path.
- **Retrieval config held identical across arms**, verified by probing the live container
  before the run rather than by reading the source — three arms were confounded today by a
  patch left set from an earlier experiment.
- **Report distinct gold turns retrieved** next to accuracy, since that is the quantity
  every arm so far has moved with.
- **Reuse the existing ingest.** This is a read-path change, so arms need search, answer
  and evaluate only — no re-ingest, and no extraction spend.

Cost: roughly one gateway key per arm, two arms.

## Order of work

1. `claims_as_index` in `HybridRetriever`, with the four tests already written: it
   contributes the source turn, it ships disabled, it deduplicates against the episode
   leg, and a dangling source contributes nothing.
2. Full suite green, and the option verified live inside the container.
3. Two arms at 200 questions: cap-15 as the control, cap-15 plus the index as the arm.
4. Report, including the case where it does not move — which would say the claim layer
   does not earn its slots on this benchmark, and turn the question into a product one.
