---
name: memvara
description: >
  Wire Memvara into an agent or editor, pick the right surface (hosted MCP,
  Python library, REST), and use memory without forging history. Use when the
  user wants persistent memory, says remember this, that's wrong, you forgot,
  what do you know about me, delete that, why did you say that, migrate from
  mem0, connect MCP, or asks to use Memvara. Also use when the memory_* tools
  are already available.
---

# memvara

Pick the surface first. Then do the job. Do not invent an API, an `npx`
installer, or a JavaScript client — there isn't one.

## Which surface

Read `references/integrate.md` before you write code or config.

- **Editor / coding agent** (Claude Code, Grok, Cursor, Copilot): install the
  plugin, or paste the hosted MCP URL. Details in `references/hosted-mcp.md`.
- **A loop you are writing in Python**: `pip install memvara`, then
  `from memvara import Memvara`. The plugin does not install into LangChain.
- **A loop in any other language**: hosted MCP as a client, or the commercial
  REST API. The npm package named `memvara` is a name reservation and does
  nothing.
- **Air-gapped / the file must stay on this machine**: local stdio MCP. Last
  resort, not the default.

Hosted MCP URL: `https://app.memvara.dev/mcp`. Approve in the browser. It lasts
90 days.

## If the memory_* tools are already connected

Read before you assert. Anything you say about what is remembered — "you told
me X", "I have nothing on file" — must come from a tool result **in the current
turn**. If you have not looked, say so, then look.

When they say a memory is wrong, do this order: `memory_recall`,
`memory_search` (you need the claim id), `memory_why` (put the excerpt in front
of them). The excerpt is the **evidence for** which write comes next, not their
wording. A value that was accurate then and is different now is
`memory_remember`. A value that was never right also needs `memory_forget`.
A value that was right and has stopped is `memory_end`. Full sequence:
`references/write-and-correct.md`.

Call `memory_stats` once before you write. If the session field is not `*`, the
server was launched with `MEMVARA_SESSION` set and the note will not carry over
— say so. If stats say `fast-path-only`, write triples with `memory_remember`;
prose handed to `memory_add` is often accepted and not stored.

Store what would be **embarrassing** to get wrong next week. Do not restate the
transcript. See `references/scopes.md`.

## Other jobs

| They asked | Open |
|---|---|
| What was true then / what we believed then | `references/time.md` |
| Why did you say that / delete me / DSAR | `references/governance.md` |
| Moving off mem0 | `references/migrate-mem0.md` |
| A worked turn | `references/examples.md` |
| No skill folder on this client | `references/project-instructions.md` |

`memory_forget` is not erasure. Real deletion is an operator action on the
console or REST, and is deliberately not a tool. Never say you deleted data if
you only retired a claim.
