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

## "1 already live, 2 now"

A receipt saying this means the slot accumulated where it might have been
meant to replace. Nothing is wrong with the row that was written; the
predicate has no spec, so it is multi-valued by default and the new value
landed beside the old one.

Read it, do not reflex-fix it. Two live values are a **contradiction** for
something like `quota_gate/status` and completely **correct** for something
like `project/rejected` — a project rejects many things. The rows are
identical and the difference is intent, which is not a property of the row,
so the note offers both readings rather than picking.

- Genuinely one value: the old one is stale, so `memory_end` it. Use
  `memory_search` first to get the id.
- Genuinely several: nothing to do. It is already right.

If it keeps happening on the same predicate, the fix is not per-write. The
server can declare its vocabulary with `MEMVARA_PREDICATES` — a shipped pack
name, a TOML file, or a comma-separated mix — and a declared predicate
supersedes or accumulates because someone said so, instead of defaulting.
`engineering` and `decisions` ship with the package. Tell them; you cannot
set it yourself, it is server configuration read at startup.

Declaring also sets **volatility**, which has no note of its own. An
undeclared predicate decays at the slow default — a two-year half-life — so
a fact that changed this morning still ranks as fresh long after it stopped
being true, and nothing ever reports it.

## Carry the turn ids forward

Half the dispute sequence above runs on the excerpt: step 3 puts it in front
of them, step 4 writes from it rather than from the complaint. `memory_why`
has an excerpt to show only if one was attached when the claim was written, so
a claim stored without that leaves you at step 2 with nothing to do but argue.

Attaching it spans two tools, which is why neither tool's own description can
tell you to do it:

1. `memory_add` reports the turn ids it created.
2. `memory_remember` takes those ids in `sources`.

Lose them between the two and nothing errors. The write succeeds, the claim
looks ordinary, and the cost lands weeks later on the one occasion it matters —
someone challenges the fact and the honest reply is that there is nothing to
show them. Extraction wires this up on its own; a triple you compose by hand
does not, and by hand is the normal case on a `fast-path-only` deployment.

Ids, not prose. The turn is already stored; handing back its text writes a
duplicate of something this store is holding.

Nothing backfills. Claims written before you started doing this stay
unexplainable, so the sooner it becomes habit the smaller that set is.

## Valid time on a write

`memory_remember` takes `true_since` and `true_until` — when it was true in
the world. Default is now, which is wrong for a backfill. Nothing here
moves belief time. Do not invent `recorded_at`. See `time.md`.
