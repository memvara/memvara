---
name: memvara
description: How to use the memvara memory tools across a whole conversation — the sequence to follow when a user disputes something you remembered, which scope your writes land in, what is worth storing at all, and what changes on a server with no extraction model. Use whenever the memory_* tools are available.
---

# memvara: the parts that span tools

Each `memory_*` tool's own description already says when to call it and when not to, and
none of that is repeated here. What follows is only what no single description can carry:
sequences that cross several tools, and facts about the particular server this
conversation is connected to.

## Read before you assert

Anything you say about what is remembered — "you told me X", "I have nothing on file about
that", "you have mentioned this before" — must come from a tool result **in the current
turn**. Not from earlier in this conversation, not from a summary of it, not from your
sense of what you probably stored. Another conversation can supersede or retire a value
between one of your turns and the next, and a confident claim about memory you did not
just read is the exact failure this system exists to prevent.

If you have not looked, say so, then look.

## When the user says a memory is wrong

Four steps, in this order. Jumping straight to the last one is the common mistake, and it
records the wrong history.

1. **`memory_recall`** — which you should already have done at the top of the turn.
2. **`memory_search`** for the disputed fact, to get its claim id. Recall's output carries
   no ids; search's does, and both remaining steps need one.
3. **`memory_why`** on that id, and then do the thing its description asks of you: put the
   excerpt in front of the user instead of arguing for the claim. This is the load-bearing
   step, and it usually settles the matter either way — they recognise what they said, or
   the excerpt shows the fact was misread out of something else. Those two endings need
   different writes in step 4, which is why this comes before it and not after.
4. **Write the correction**, and let step 3 decide the shape rather than the wording of
   the complaint. Storing a replacement on its own says *the world changed*. Only
   `memory_forget` says *the record was wrong*, and only `memory_end` says a fact that
   was right has stopped being true. Each tool's description defines its own case; what
   none of them can tell you is that **the source turn from step 3 is your evidence for
   which case this is**. "That is wrong" about an excerpt the user recognises is a
   change; the same words about an excerpt that misquotes them is a mistake.

   So: where the excerpt shows a value that was accurate at the time, `memory_remember`
   alone is the whole correction. Where it shows one that was never accurate, add a
   `memory_forget` call — a replacement by itself cannot make that statement, whatever
   the user's wording implied. Skip step 3 and you are guessing, and a guess writes a
   history saying the world moved when the truth is that nobody was ever right about it.

The second call is the point rather than an inconvenience: retiring cannot be undone from
here, so it is worth arriving at with the excerpt already in front of you.

## Which scope you are writing into

Call `memory_stats` once, early, in any conversation where you expect to write. It reports
the scope this server is bound to, spelled `tenant/user/agent/session`, where `*` is an
unbound field.

If the session field is **not** `*`, the server was launched with `MEMVARA_SESSION` set,
and everything you write is invisible to the next conversation. Nothing in an ordinary
write result will tell you this, and no argument you can pass to a tool changes it — the
scope is fixed when the server starts. So:

* Say it plainly at the moment you store something: "noted for this session, it will not
  carry over" is honest. Letting someone believe a durable fact was kept is not.
* If it needs to persist, that is a one-line change to the client's env block — unset
  `MEMVARA_SESSION` — and an operator's job, not something to work around by writing the
  fact again.

## What is worth writing at all

The test: **would being wrong about this next week be embarrassing?** Preferences,
constraints, decisions and corrections pass it. Anything you could re-derive from the
conversation in front of you does not — the transcript already holds it, and a store
padded with restated context makes the facts that matter harder to retrieve.

## On a server with no extraction model

`memory_stats` also reports the extractor. If it says `fast-path-only`, this server was
started with `MEMVARA_LLM=none`: prose is matched against a fixed set of sentence forms,
and a turn that fits none of them is accepted and quietly not stored. A note on the write
receipt does report it, but only one write at a time and only afterwards.

So check once, at the start, and on such a server write anything you actually want kept as
an explicit `memory_remember` triple rather than handing prose to `memory_add` and hoping
the shape was recognised.
