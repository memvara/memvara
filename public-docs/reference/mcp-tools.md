# MCP tools reference

When Memvara is connected to an AI assistant over MCP, it exposes fourteen tools. This
page lists each one. See
[Use Memvara in an AI coding assistant](../how-to-guides/use-memvara-in-an-ai-coding-assistant.md)
for how to connect.

## Reading

| Tool | What it does |
|---|---|
| `memory_recall` | Look up what's already known about the current user, formatted as ready-to-use context before answering a question. |
| `memory_search` | Search memory and get back individual results with relevance scores — also the tool for asking about a past point in time. |
| `memory_standing` | List every standing preference the user has stated, with no search query involved — everything that always applies, not just what matches a specific question. |
| `memory_since` | Report what has changed in memory since a given point in time — useful when resuming a conversation after a gap. |
| `memory_history` | Show every value a specific fact has ever held, and when each one began and ended. |
| `memory_why` | Explain why a specific stored fact is believed: which message it came from, and what it replaced, if anything. |
| `memory_ask` | Answer a question about a past moment, and explicitly say whether the record has since changed. |
| `memory_neighborhood` | Show what is connected to a given entity, by walking the stored facts that reference it. |
| `memory_paths` | Find how two specific things are connected, if anything stored links them. |
| `memory_stats` | Report what this server is bound to (whose memory, and which scope), how much it holds, and whether it can extract facts from free text. |

## Writing

| Tool | What it does |
|---|---|
| `memory_add` | Store what the user just said, in their own words, and extract any facts it recognizes. |
| `memory_remember` | Record one exact fact directly, as a subject/predicate/object triple, skipping extraction entirely. |
| `memory_forget` | Retire a fact **because the record was wrong** — it was never true. |
| `memory_end` | Close out a fact **because it has stopped being true** — the world changed. |

## The two that are easy to confuse

`memory_forget` and `memory_end` are not interchangeable, and mixing them up records a
false story about what actually happened, in a way that's invisible afterward just by
looking at the data:

- Use **`memory_end`** when the fact *was* true and has since stopped being true — for
  example, someone moved house, or left a job. The old value still correctly answers
  questions about the period when it held.
- Use **`memory_forget`** when the fact was *never* true — a mishearing, a typo, or a
  record that was simply wrong from the start.

## Two things to check before writing anything

**Check what the server can extract.** Call `memory_stats` first. If it reports the
server has no extraction model available, then `memory_add` will only recognize a small
fixed set of very simple sentence patterns. For anything more nuanced, use
`memory_remember` with the fact spelled out explicitly — it never needs a model and can't
be misread.

**The `role` on `memory_add` decides what gets scanned for facts, not just who is
credited.** Facts are only extracted from turns marked as coming from the user. If you're
handing the tool a document, a transcript, or a pasted log — rather than something a
person actually said to the assistant right now — mark it accordingly so a quoted
sentence inside that text isn't mistaken for a live statement from the user.

## Related pages

- [Run your own MCP server](../how-to-guides/run-your-own-mcp-server.md) — self-hosting
  the server these tools are exposed from.
- [CLI and configuration reference](cli-and-configuration.md) — the environment variables
  that control server behavior, including a read-only mode.
