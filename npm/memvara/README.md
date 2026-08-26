# memvara

**Long-term memory for AI agents, one command away.**

```bash
npx memvara
```

That is an MCP server. Point any MCP client at it and your agent gets a memory that
survives the session — one that resolves contradictions instead of accumulating them, and
that can tell you what it believed last March.

There is nothing to configure. No URL to paste, no client to register, and if you have
signed in before, no browser either.

> **This package changed kind at `0.1.0`.** `0.0.x` was a name reservation that exposed no
> API. It is now a CLI. There is still no JavaScript library to `import` — the memory
> engine is Python, and this is the bridge to the hosted one.

## Use it

Most clients take a command. This is the block:

```json
{
  "mcpServers": {
    "memvara": { "command": "npx", "args": ["-y", "memvara"] }
  }
}
```

Claude Code:

```bash
claude mcp add memvara -- npx -y memvara
```

On first run it opens a browser once to sign you in, caches the token in
`~/.memvara/oauth.json` at mode 600, and refreshes it silently after that.

### If your client speaks to remote servers itself

**Then you do not need this package.** `https://app.memvara.dev/mcp` advertises standard
MCP OAuth — dynamic registration, PKCE, refresh — so a remote-capable client can connect
to it directly. This bridge exists for clients that only know how to spawn a command over
stdio, which is still most of them.

## Credentials

Looked for in this order; the first one found wins.

| | source | how it gets there |
| --- | --- | --- |
| 1 | `MEMVARA_API_KEY` | you set it — for CI, containers, anywhere without a browser |
| 2 | `~/.memvara/credentials.json` | `memvara-mcp login --project NAME`, the Python CLI |
| 3 | `~/.memvara/oauth.json` | `npx memvara login` |

The second is why this is not `mcp-remote`: if you already use the Python package, the
bridge finds that key and **never opens a browser at all**.

`~/.memvara/credentials.json` is read and never written — `memvara/server/config.py` owns
that file's schema, and our token pair does not fit in it.

```bash
npx memvara login     # sign in, cache the token
npx memvara logout    # forget it
npx memvara --server https://your-console.example.com
```

## What memvara is

**Bitemporal memory.** Every fact carries two independent clocks: when it was true in the
world, and when you learned it. Most stores have one `updated_at` column and therefore
cannot answer "where did she live in March?", nor absorb a fact that arrives late about
the past.

|  |  |
| --- | --- |
| 🕰️ **Two clocks, not one** | Ask what you believed in March about June, and get an answer rather than a guess. |
| ⚖️ **Contradictions resolve without a model** | Cardinality is a schema property, so a conflict is an indexed lookup. Same two facts, same result, every run. |
| 🧾 **Nothing is silently lost** | Every write returns a receipt saying what it did — including what it could *not* extract. |
| 🔍 **Retrieval that explains itself** | Vector and BM25, time-aware, every score inspectable rather than a ranking you have to trust. |
| 🧬 **Claims are a graph** | Walk relationships at a point in time, and fuse that walk into search as a third retrieval leg. |

A correction never destroys history. It closes the old value at the instant it stopped
being true, so that value goes on answering questions about the period it held.

## The tools

The set follows the library, and **a hosted deployment can be a release behind it** —
this bridge shows whatever the server it connects to advertises. `tools/list` is the
authority for a given day; the table below is what `app.memvara.dev` serves now.

| tool | for |
| --- | --- |
| `memory_recall` | Answer from memory. Call it *before* answering, not after being corrected. |
| `memory_search` | Inspect the store — ids, scores, record types. Also the time-travel entry point. |
| `memory_add` | Write prose and let extraction find the facts. |
| `memory_remember` | Write an exact subject/predicate/object triple, skipping extraction. |
| `memory_neighborhood` | What is connected to an entity, walked through facts rather than matched as text. |
| `memory_paths` | Chains between two entities, each hop a fact you can check. |
| `memory_ask` | What is true now, what was true then, and what this store *would have told you* then. |
| `memory_since` | What changed after an instant. |
| `memory_standing` | Standing instructions, ordered so a stated rule outranks an inferred one. |
| `memory_history` | Every value a fact has held, including closed ones. |
| `memory_why` | The turns a claim came from, and what it replaced. |
| `memory_end` | Close a fact that *was* true and has stopped being true. |
| `memory_forget` | Retire a value that was **never** right. |
| `memory_stats` | What the store holds, and which extractor is running. |

`memory_end` and `memory_forget` are not interchangeable: picking the wrong one records a
false reason for the change that nothing downstream can detect. Neither is erasure — both
stay visible to `memory_history`.

## What this package is not

- **Not a JavaScript library.** `require("memvara")` returns a signpost, not an API. The
  engine is Python: `pip install memvara`.
- **Not a local store.** This bridges to the hosted service. For an offline store that
  needs no account and no network, use the Python server directly:
  `MEMVARA_DB=~/memory.db memvara-mcp`.
- **Not a reimplementation.** A second engine would have to re-derive memvara's temporal
  invariants exactly, and getting one wrong is not hypothetical — a published paper
  conflated the two clocks on supersession and measured its own time-travel retrieval
  scoring *worse* than plain search as a result.

## Zero dependencies

This process holds a bearer token, so it has no dependency tree to audit. Everything is
Node stdlib: `fetch`, `node:crypto` for PKCE, `node:http` for the loopback redirect. The
Python MCP server is hand-rolled against the wire format for the same reason.

Requires Node 20 or newer.

## Links

| | |
| --- | --- |
| Source | [github.com/memvara/memvara](https://github.com/memvara/memvara) |
| Python package | [pypi.org/project/memvara](https://pypi.org/project/memvara/) |
| Hosted service | [memvara.dev](https://memvara.dev) |
| Client setup | [memvara.dev/docs/agents](https://memvara.dev/docs/agents) |
| Issues | [github.com/memvara/memvara/issues](https://github.com/memvara/memvara/issues) |

## License

Apache-2.0, the same as the library.
