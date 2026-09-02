# memvara extracts a user profile; LongMemEval asks for an event ledger

**Date:** 2026-09-03
**Evidence:** claims written by the `memvara-llmingest2` arm (gpt-5.4 extraction) against
the `memvara-baseline-18e3626` arm (fast path), read directly from `memvara.claims`

## The finding

memvara's extraction vocabulary has 23 built-in predicates and **every one of them is a
personal-profile attribute**:

```
allergic_to  born_in  born_on  communication_style  dietary_restriction  dislikes
goal  job_title  likes  lives_in  located_now  mood  name  never_do  owns_pet
prefers  prefers_tool  pronouns  relationship_status  speaks  timezone
working_on  works_at
```

Not one carries a quantity, an amount, or the date of an event. The questions memvara loses
on need exactly those: "how many total hours did I spend running", "how much did I spend on
a gift for my sister", "which of these happened first". A store that can say what a user
*is like* cannot answer what a user *did, when, and how much*.

This is a schema mismatch, and a better extraction model does not fix it.

## A better model produces far more claims of the same wrong shape

| | fast path | gpt-5.4 |
| --- | ---: | ---: |
| claims per question | 3.5 | **97.3** |
| distinct predicates | 9 | 27 |

A 28-fold increase in extraction, which on its own says the fast path was doing almost
nothing on this corpus — 3.5 claims from roughly 420 turns. The registry can learn
predicates and gpt-5.4 added a few (`owns`, `had_experience`, `has_symptom`), so the
vocabulary is not a hard ceiling. It is a strong attractor, and what comes out is still a
profile:

```
user             goal            prepare for 20-mile bike ride
user             had_experience  surfing
user             goal            save for new camera
justin mcdonald  goal            develop copywriting skills and stay connected to advertising roots
transform        owns            team of 5 people
```

`user | had_experience | surfing` cannot answer how many hours were spent surfing. The turn
it came from can. That is the whole gap in one line.

## What it explains

Every result from this session falls out of it.

- **Claims contributed 0.0 points.** They are the wrong shape for the questions, so a slot
  spent on a claim is a slot wasted, whatever the claim's quality.
- **Only raw turn volume ever moved the number.** The turns contain the amounts and the
  dates. Nothing else retrieved does.
- **Reranking and session diversity did nothing.** Both reorder a candidate set whose useful
  members are raw turns; neither changes how much answer-bearing text arrives.
- **Supermemory reaches 95% at ~720 tokens.** Their memories are atomic facts with
  ambiguous references resolved, indexed for search and backed by the source chunk. Ours
  are profile attributes, indexed and returned as themselves.

## What follows

The fix is not a bigger extraction model. It is a predicate vocabulary that can hold an
event: something with a subject, an action, a quantity and a time, written at ingest and
carrying `valid_from` from the event rather than from the conversation. memvara already has
the bitemporal machinery to store that correctly — `valid_from` versus `recorded_at` is the
product's central claim, and it is currently recording profile attributes with it.

Two things to decide before building:

1. **Whether LongMemEval is the right target at all.** It rewards an event ledger over a
   conversation history. If memvara's intended job is remembering what a user is like
   across sessions, this benchmark measures something adjacent to it, and the honest move
   is to say so rather than to reshape the product around the test.
2. **If it is the target**, the work is a schema change plus an extraction prompt that fills
   it, and it is much larger than any retrieval tuning attempted this session. Nothing on
   the read path will close a 25-point gap while the write path stores the wrong thing.
