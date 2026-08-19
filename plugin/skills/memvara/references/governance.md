# Deletion, audit, and questions this server cannot answer

Read this when they ask why the agent said something, want data removed, or
ask what was believed on a past date. None of that is a reason to invent a
tool.

## "Why did you say that?"

`memory_search` for the claim, then `memory_why` on its id. Show the
excerpt. Do not reconstruct a reason from the current chat. If `why` has no
source turn, say that — an unsourced claim is itself the answer.

`memory_history` on the same subject and predicate is what you use when they
ask what the value used to be, not why it was written.

This is the product. An agent that asserts a memory without this read is
skipping the thing Memvara exists to make possible.

## "Delete me" / "delete that" / a legal deletion request

`memory_forget` retires a value. The row stays visible to
`memory_history`. That is a correction, not erasure.

Irreversible erase lives on the console and the REST API. It is
deliberately not a tool: a model that can be talked into a call must not
reach a button that shreds a scope.

When they want a real deletion:

1. Do not claim you deleted anything if you only retired a claim.
2. Tell them an org admin can erase from the console. The guide spells the
   difference: https://memvara.dev/docs/guide#erasure
3. If you are not sure they have that role, say so rather than walking them
   through a path you cannot complete.

## "What did we believe on 3 June?"

MCP search can take `as_of` — both clocks at one instant. The two clocks
separately (`valid_at` / `known_at`) live on the library and REST surfaces,
not on these tools. If they need one clock and not the other, say you cannot
do that from here and point them at the API. Do not guess which clock they
meant and silently pick `as_of`. See `time.md`.

## Legal hold, retention, residency

Out of reach of this skill. There is no tool for a hold and no tool that
moves data between regions. Escalate to an operator. Do not store a note
that pretends the hold is in force.

Do not call this a certification, a DPA, or "compliance-grade". Erasure
counts and `why()` exist. That is what you can show.
