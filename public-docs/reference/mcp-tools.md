# MCP tools reference

When Memvara is connected to an AI assistant over MCP, it gives the assistant a set of
memory tools. This page lists every one of them, one row per tool. See
[Use Memvara in an AI coding assistant](../how-to-guides/use-memvara-in-an-ai-coding-assistant.md)
for how to connect.

## Reading

| Tool | What it does |
|---|---|
| `memory_recall` | Look up what's already known about the current user, formatted as ready-to-use context before answering a question. |
| `memory_search` | Search memory and get back individual results with relevance scores — also the tool for asking about a past point in time. |
| `memory_standing` | List every standing preference the user has stated, with no search query involved — everything that always applies, not just what matches a specific question. |
| `memory_profile` | Give a session everything it should start with in one call: standing preferences, what was stored recently, memories grouped by kind, and, when given a question, the memories most relevant to it. |
| `memory_since` | Report what has changed in memory since a given point in time — useful when resuming a conversation after a gap. |
| `memory_history` | Show every value a specific fact has ever held, and when each one began and ended. |
| `memory_why` | Explain why a specific stored fact is believed: which message it came from, and what it replaced, if anything. |
| `memory_ask` | Answer a question about a past moment, and explicitly say whether the record has since changed. |
| `memory_neighborhood` | Show what is connected to a given entity, by walking the stored facts that reference it. |
| `memory_paths` | Find how two specific things are connected, if anything stored links them. |
| `memory_stats` | Report what this server is bound to (whose memory, and which scope), how much it holds, and whether it can extract facts from free text. |
| `memory_get_document` | Show one stored document's record: its title, file path, where it came from, how many passages it is stored as, and whether it has been processed (`queued`, `extracting`, `done`, `stored`, or `failed` with the reason). It takes the document's `id`. It does not return the text; recall with passages turned on does that. |
| `memory_list_documents` | List the stored documents, newest first, with each one's id, status, passage count, title and file path. `filepath_prefix` keeps one folder, `status` keeps one processing state, and a long list comes a page at a time: pass the returned `cursor` back to get the next page. |

## Writing

| Tool | What it does |
|---|---|
| `memory_add` | Store what the user just said, in their own words, and extract any facts it recognizes. |
| `memory_remember` | Record one exact fact directly, as a subject/predicate/object triple, skipping extraction entirely. Give `expires_at`, a date and time in the future, for a fact that must not be kept past it, such as a temporary door code: once that time passes the fact stops being returned at once, is deleted within the hour, and cannot be found or restored. `expire_reason` records why. This is different from a fact that stops being true, which is kept with its history. |
| `memory_forget` | Retire a fact **because the record was wrong** — it was never true. |
| `memory_end` | Close out a fact **because it has stopped being true** — the world changed. |
| `memory_end_matching` | Close out a whole group of facts that have all stopped being true, such as everything about a project that has shipped. It works in two steps so nothing changes by surprise. The first call takes a search `query` and changes nothing: it lists the matching facts and returns a confirmation token. Calling again with that token as `confirm` ends exactly the facts that were listed, and none that started matching since. `k` caps how many matches are considered, and `reason` records why they ended. An expired or mismatched token is refused and nothing is changed. |
| `memory_forget_matching` | Retire a whole group of facts that were all wrong, or that the user asked to forget, such as everything stored about one topic. It works in the same two steps as `memory_end_matching`: a first call with a `query` lists the matches and returns a token, and a second call with that token as `confirm` retires exactly those facts. `k` and `reason` work the same way. Retired facts stay visible in their history; nothing is erased. |
| `memory_link` | Record that one stored fact adds detail to another (`extends`), or was worked out from another (`derives`), so that asking why either one is believed shows the connection. It takes the two facts' ids, `from_id` and `to_id`, and the `relation`. Both facts must be visible to the assistant, a fact cannot be linked to itself, and recording the same link twice changes nothing. |
| `memory_add_document` | Keep a whole document so passages from it can be found later: a policy, a README, meeting notes, a web page. Give exactly one of `content`, the text itself, or `url`, a page the server fetches (it refuses one on a private network). The text is split into passages of about 1,000 characters. Give `custom_id`, such as the file path or URL, when the same document may be sent again: sending it again updates the stored copy and keeps every unchanged passage. `title`, `filepath`, `mime` and `metadata` are optional. The reply gives the document's id and status; a document the server cannot read is refused with the reason, and nothing is stored. |
| `memory_delete_document` | Delete one stored document by its `id`. Its text is erased and cannot be searched or restored. No memory is erased: a memory whose only source was the document is retired with the reason "source document deleted", and one that also came from elsewhere keeps its other source. To replace a document with a newer version, add it again with the same `custom_id` instead. |

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
