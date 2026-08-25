# memvara

**This npm package is a name reservation. It exposes no API — `require("memvara")`
returns a notice object and nothing else.** There is no JavaScript client. If you are
here to `npm install` a library and call it, stop here; the rest of this page is about
what memvara is and how to reach it from JavaScript, which does not go through npm.

```js
require("memvara").implemented;   // false, always
```

You can still use memvara from a JavaScript or TypeScript agent today. It speaks **MCP**,
which is the interface an agent already has — see [Using it from
JavaScript](#using-it-from-javascript) below.

---

## What memvara is

**Bitemporal memory for AI agents.** Structured facts with deterministic contradiction
resolution, hybrid retrieval, and a write path that mostly doesn't call an LLM.

The idea it is built on: a memory has **two independent clocks**. When something was true
in the world, and when you found out. Most stores have one `updated_at` column and
therefore cannot answer "where did she live in March?", nor absorb a fact that arrives
late about the past.

|  |  |
| --- | --- |
| 🕰️ **Two clocks, not one** | When it was true, and when you learned it — independently. Ask what you believed in March about June and get an answer rather than a guess. |
| ⚖️ **Contradictions resolve without a model** | Cardinality is a schema property, so a conflict is an indexed lookup. Same two facts, same result, every run. |
| 🔌 **Offline by default** | numpy and nothing else. No API key, no Docker, no vector database, no network on the write path. |
| 🧾 **Nothing is silently lost** | Every write returns a receipt saying what it did — including what it could *not* extract. |
| 🔍 **Hybrid retrieval that explains itself** | Vector and BM25, time-aware, every score inspectable rather than a ranking you have to trust. |
| 🧬 **Claims are a graph** | Walk relationships at a point in time, and optionally fuse that walk into search as a third retrieval leg. |

A correction never destroys history. It closes the old value at the instant it stopped
being true, so the old value goes on answering questions about the period it held.

## Using it from JavaScript

Two routes, neither of which needs anything from npm. A client library is one way to reach
a service and it is not the way an agent usually does it — an agent reaches tools over
MCP, and memvara ships an MCP server, so a JS binding would sit between two things that
are already connected.

### Hosted

The console at [app.memvara.dev](https://app.memvara.dev) serves MCP over HTTP at
`https://app.memvara.dev/mcp`. One endpoint and a token; the same store from every machine
and every session. Client-by-client setup is at
[memvara.dev/docs/agents](https://memvara.dev/docs/agents).

### Local, offline, no account

The Python package ships the same server over stdio:

```bash
pip install memvara
MEMVARA_DB=~/memory.db memvara-mcp        # JSON-RPC 2.0 over stdio
memvara-mcp init --agent claude           # writes the client block for you
```

Point your MCP client at that command. Python has to be present to run the server;
**nothing about your own project has to be Python.**

### If you use an AI coding tool

Nothing to install and nothing to run:

```
/plugin marketplace add memvara/claude-memvara
/plugin install memvara
```

Cursor, Codex, Grok, VS Code and OpenCode have their own one-liners at
[memvara.dev/docs/agents](https://memvara.dev/docs/agents); Claude Desktop and ChatGPT
paste the same URL.

## The tool surface

Twelve tools, hand-rolled against the MCP wire format rather than taking an SDK
dependency — which is how "numpy and nothing else" survives the server too.

| tool | what it is for |
| --- | --- |
| `memory_recall` | Answer from memory. Plain notes, ready to read; call it *before* answering, not after being corrected. |
| `memory_search` | Inspect the store itself — ids, scores, record types. Also the time-travel entry point. |
| `memory_add` | Write prose and let extraction find the facts. |
| `memory_remember` | Write an exact subject/predicate/object triple, skipping extraction. |
| `memory_neighborhood` | What is connected to an entity, walked through stored facts rather than matched as text. |
| `memory_paths` | Chains between two entities, each hop a fact you can check. |
| `memory_since` | What changed after an instant. |
| `memory_history` | Every value a fact has held, including closed ones. |
| `memory_why` | The turns a claim was derived from, and what it replaced. |
| `memory_end` | Close a fact that *was* true and has stopped being true. |
| `memory_forget` | Retire a value that was **never** right. |
| `memory_stats` | What the store holds, and which extractor is in use. |

`memory_end` and `memory_forget` are not interchangeable, and picking the wrong one
records a false reason for the change that nothing downstream can detect. Neither is
erasure — both stay visible to `memory_history`.

## Teaching it your vocabulary

The built-in predicates are a personal-assistant vocabulary — where someone lives, where
they work. A store of engineering facts matches none of them, and an unknown predicate
takes the safe default twice over: multi-valued, so nothing supersedes, and slow-decaying,
so this morning's deploy still ranks as fresh in two years.

```bash
MEMVARA_PREDICATES=engineering memvara-mcp        # or: engineering,./ours.toml
```

```toml
[[predicate]]
name = "git_state"
cardinality = "one"     # supersedes; "many" accumulates
volatility = "fast"     # static | slow | fast -> 36500 | 730 | 7 day half-life
```

## Honest limitations

- **The default embedder is lexical, not semantic.** `HashingEmbedder` is the default so
  the library runs offline in milliseconds with no download. It will not put "physician"
  near "doctor" — install `memvara[local-embed]` or pass your own embedder for real
  semantic recall.
- **Extraction needs a model.** Running with no LLM configured keeps every turn and
  derives facts only from a fixed set of high-precision sentence forms; anything else is
  counted as unextracted and reported, not silently dropped. Write triples with
  `memory_remember` when you need certainty.
- **There is no JavaScript client**, which is the whole subject of the top of this page.

## Why publish an empty package at all

Because npm reserves nothing otherwise. An organisation reserves the `@memvara/*` scope
and not the bare name, exactly as a PyPI organisation reserves no project name — only a
publish does. A project that exists, is being used, and has a name worth protecting has a
legitimate claim to that name; that is different from registering names you have no
relationship to, which is what npm's policy is actually against.

If you wanted this name for something else: sorry, and genuinely — open an issue. If the
project is ever abandoned, the right thing is to hand it over rather than sit on it.

## If you want a real JS client

Say so on the [issue tracker](https://github.com/memvara/memvara/issues). Whether to build
one is an open question and the number of people who ask is most of the answer — MCP
covers the agent case, so what a client would add is the non-agent case: calling memvara
from ordinary application code. The REST API it would wrap is documented in the commercial
layer; the Python library needs no server and runs offline.

## Links

| | |
| --- | --- |
| Source | [github.com/memvara/memvara](https://github.com/memvara/memvara) |
| Python package | [pypi.org/project/memvara](https://pypi.org/project/memvara/) |
| Hosted service | [memvara.dev](https://memvara.dev) |
| MCP client setup | [memvara.dev/docs/agents](https://memvara.dev/docs/agents) |
| Issues | [github.com/memvara/memvara/issues](https://github.com/memvara/memvara/issues) |

## License

Apache-2.0, the same as the library.
