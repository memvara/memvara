# The write-path fix was built, measured, and did not work

**Date:** 2026-09-03
**Sample:** 52 questions, stratified random across all six LongMemEval-S categories, seed
`20260903`. Every figure below is on the same 52.

## Result

| arm | ingest | accuracy |
| --- | --- | ---: |
| fast path | rule-based, ~3.5 claims/question | **51.9%** |
| gpt-5.4, profile predicates | ~97 claims/question | 46.2% |
| gpt-5.4 + event time + quantities + `events` pack | ~46 claims/question | **44.2%** |

The write-path change made accuracy worse, and it is the worst of the three arms.

| type | fast path | gpt-5.4 profile | event schema |
| --- | ---: | ---: | ---: |
| multi-session | 28.6% | 42.9% | 42.9% |
| temporal-reasoning | 57.1% | 42.9% | 42.9% |
| knowledge-update | 62.5% | 37.5% | 37.5% |
| single-session-user | 66.7% | 50.0% | 50.0% |
| single-session-assistant | 80.0% | 80.0% | 80.0% |
| single-session-preference | 40.0% | 40.0% | 20.0% |

**Five of the six categories are identical between the two model-ingest arms.** The event
fields changed nothing that the judge could see; the single moving category is
preference, on five questions, which is noise.

## Why it changed nothing

Event data is a rounding error in what gets retrieved. Of 2,403 claims written across the
52 questions, **4.1% carry an event time and 9.9% carry a quantity**. The rest are the
same profile attributes as before, and claims still take 68% of the retrieval slots. A
handful of correctly-dated events cannot outweigh five hundred `likes` and `goal` rows
sharing the prompt with them.

The conditional durability rule did work as intended — event predicates went from 2.4% of
claims to 9.4% in the probe, and per-question claim volume fell from 97 to 46, so the rule
made extraction more selective rather than more prolific. It simply was not the binding
constraint.

## What the three arms actually show

| arm | turns retrieved | claims retrieved | claims' share | median ctx tokens | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast path | 156 | 179 | 53% | 775 | 51.9% |
| gpt-5.4 profile | 260 | 665 | 72% | 1,758 | 46.2% |
| event schema | 260 | 564 | 68% | 1,679 | 44.2% |

Both model-ingest arms retrieve **more turns** and **more than twice the context** of the
fast path, and both score lower. So the earlier reading — that claims crowd turns out of
the prompt — is not right either: the turns are there. What correlates with losing is the
*presence of many claims alongside them*.

The most defensible statement the data supports: **on this benchmark, extracted claims
compete with the source turns and lose, and adding more of them hurts whatever shape they
are in.** The fast path wins by extracting almost nothing and leaving the conversation to
speak for itself.

## What this does and does not refute

**Refuted:** the thesis in `2026-09-03-the-schema-mismatch.md` that memvara's LongMemEval
gap is caused by its predicates being unable to hold events, and that giving them event
times and quantities would close it. It was implemented in full, verified writing real
event rows (`played | tennis | 1.5 hour`, `paid | mortgage | 2023-04-01 | month`), and did
not move the number.

**Not refuted, and worth separating:** that `valid_from` should carry the time a turn
stated rather than the time it was said. That is a correctness fix to the product's
central claim — a store whose world clock and belief clock record the same instant is not
bitemporal in any useful sense — and it stands on its own merits whatever this benchmark
does. The same goes for the Postgres columns and the three-way ordering rule.

## What to try next, in order

1. **Stop adding claims and test removing them.** The one configuration never measured is
   the model-ingest arm with claims excluded from retrieval — turns only. If that scores
   near the fast path, claims are the problem outright and the question becomes what they
   are *for*, which is a product question rather than a benchmark one.
2. **Ask whether this benchmark measures the job.** LongMemEval rewards retrieving raw
   conversation. memvara's design bet is that a curated claim beats a raw turn, and three
   arms now say the opposite on this corpus. Either the bet is wrong, or the benchmark
   does not measure it — and deciding which is worth more than another arm.

## Cost

Four gateway keys, roughly $40. Two were spent on runs that turned out to be invalid: one
because the Postgres store silently dropped the new columns, one because `shape_claims`
silently dropped them from the model's reply. Both were explicit field lists that fail
without erroring. A three-question probe found the second for nothing, and should have
been run before the first.
