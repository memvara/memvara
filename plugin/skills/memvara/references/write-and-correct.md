# Writing, and what to do when they say it is wrong

Tool descriptions already say when to call each tool. This file is the
order, and the judgment no single description can make.

## What is worth storing

Would being wrong about this next week be embarrassing? Preferences,
constraints, decisions, and corrections pass. A stack trace, a function you
just explained, or a restatement of the last five turns does not. The
transcript already has those, and padding the store hides the facts that
matter.

## Dispute sequence

Four steps. Skipping to the write records the wrong history.

1. `memory_recall` — you should already have done this at the top of the turn.
2. `memory_search` for the disputed fact. Recall has no ids; search does.
3. `memory_why` on that id. Put the excerpt in front of them. Do not argue
   for the claim. They will recognise what they said, or the excerpt will
   show it was pulled from the wrong place. Those two endings need different
   writes.
4. Write from the excerpt, not from the complaint.

   - Excerpt was accurate at the time, world has moved → `memory_remember`
     the new value. That is the whole correction.
   - Excerpt was never right about this person → `memory_forget` the id,
     then `memory_remember` if you now have a fact.
   - Fact was right and has stopped being true, and they named when →
     `memory_end` (optional `at`) or `memory_remember` with `true_since` /
     `true_until`.

"That's wrong" about an excerpt they recognise is a change. The same words
about an excerpt that misquotes them is a mistake. Their wording does not
pick the tool.

Retiring cannot be undone from here. Arrive at it with the excerpt shown.

Worked turns: `examples.md`.

## No extraction model

`memory_stats` reports the extractor. `fast-path-only` means
`MEMVARA_LLM=none`: prose is matched against a fixed set of English sentence
forms, and a turn that fits none of them is accepted and quietly not stored.
The write receipt mentions it after the fact, one write at a time.

Check once at the start. On that server, write anything you actually want
kept as `memory_remember` subject / predicate / object. Do not hand a
paragraph to `memory_add` and hope.

## Valid time on a write

`memory_remember` takes `true_since` and `true_until` — when it was true in
the world. Default is now, which is wrong for a backfill. Nothing here
moves belief time. Do not invent `recorded_at`. See `time.md`.
