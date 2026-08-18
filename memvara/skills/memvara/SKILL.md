---
name: memvara
description: >
  Use Memvara memory tools correctly across a conversation: read before
  asserting, correct a disputed fact in the right order, pick forget vs
  end vs remember from evidence, check the bound scope, and store only
  durable facts. Use when the memory_* tools are available, or when the
  user says remember this, you forgot, that's wrong, what do you know
  about me, delete that memory, or asks the agent to use Memvara.
---

# memvara

Each `memory_*` tool already says when to call that one tool. This file is only
what no single description can carry: sequences that cross several of them, and
facts about the server this conversation is bound to.

If you need a worked turn, read `references/examples.md`. If the user asks to
erase data, explain a past belief, or handle a legal deletion request, read
`references/governance.md` before you write.

## Read before you assert

Anything you say about what is remembered — "you told me X", "I have nothing on
file about that", "you have mentioned this before" — must come from a tool
result **in the current
turn**. Not from earlier in this conversation, not from a summary of it, not
from your sense of what you probably stored. Another conversation can replace
or withdraw a value between one of your turns and the next.

If you have not looked, say so, then look.

## When the user says a memory is wrong

Four steps, in this order. Skipping to the write records the wrong history.

1. **`memory_recall`** — which you should already have done at the top of the turn.
2. **`memory_search`** for the disputed fact, to get its claim id. Recall's output
   carries no ids; search's does, and both remaining steps need one.
3. **`memory_why`** on that id. Put the excerpt in front of the user instead of
   arguing for the claim. They will either recognise what they said, or the
   excerpt will show the fact was pulled from the wrong place. Those endings
   need different writes in step 4, which is why this comes before it.
4. **Write**, and let step 3 decide the shape rather than the wording of the
   complaint. A replacement on its own says *the world changed*. Only
   `memory_forget` says the stored value was never right. Only `memory_end`
   says a fact that was right has stopped being true. What no tool can tell
   you is that **the source turn from step 3 is your evidence for which case
   this is**. "That is wrong" about an excerpt they recognise is a change; the
   same words about an excerpt that misquotes them is a mistake.

   Where the excerpt shows a value that was accurate at the time,
   `memory_remember` alone is the whole correction. Where it shows one that
   was never accurate, add a `memory_forget` call — a replacement by itself
   cannot make that statement. Skip step 3 and you guess, and a guess writes
   a history saying the world moved when nobody was ever right about it.

Retiring cannot be undone from here, so arrive at it with the excerpt already
shown.

## Which scope you are writing into

Call `memory_stats` once, early, in any conversation where you expect to write.
It reports the scope this server is bound to, spelled `tenant/user/agent/session`,
where `*` is an unbound field.

If the session field is **not** `*`, the server was launched with
`MEMVARA_SESSION` set, and everything you write is invisible to the next
conversation. Nothing in an ordinary write result will tell you this, and no
argument you can pass to a tool changes it.

- Say it at the moment you store something: "noted for this session, it will
  not carry over" is honest. Letting someone believe a durable fact was kept
  is not.
- If it needs to persist, unset `MEMVARA_SESSION` in the client's env block.
  That is an operator's job, not something to work around by writing the fact
  again.

## What is worth writing at all

The test: **would being wrong about this next week be embarrassing?**
Preferences, constraints, decisions and corrections pass. Anything you could
re-derive from the conversation in front of you does not. The transcript
already holds it, and a store padded with restated context hides the facts
that matter.

## On a server with no extraction model

`memory_stats` also reports the extractor. If it says `fast-path-only`, this
server was started with `MEMVARA_LLM=none`: prose is matched against a fixed
set of sentence forms, and a turn that fits none of them is accepted and
quietly not stored. A note on the write receipt reports it, but only one write
at a time and only afterwards.

Check once, at the start. On such a server, write anything you actually want
kept as an explicit `memory_remember` triple rather than handing prose to
`memory_add` and hoping the shape was recognised.
