# MCP

Memvara speaks the Model Context Protocol, so an agent that supports MCP gets memory as
fourteen tools without you writing any code. **There are three ways to reach them, and
picking the right one takes one question.**

| You want | Use | Setup |
|---|---|---|
| Memory in your editor or coding agent, nothing to run | The plugin, or the hosted MCP URL | one line |
| Memory on your own machine, in a file you control | The local stdio server | one environment variable |
| Memory for other people, or from a non-Python client | A deployment | [Deploying](../DEPLOY.md) |

## In an editor or coding agent

Claude Code:

```
/plugin marketplace add memvara/claude-memvara
/plugin install memvara
```

Cursor, Codex, Grok, VS Code and OpenCode have their own one-liners at
[memvara.dev/docs/agents](https://memvara.dev/docs/agents). Claude Desktop and ChatGPT
paste the same hosted URL: `https://app.memvara.dev/mcp`, approved in the browser, good
for 90 days.

With no Python at all, `npx memvara` bridges a stdio MCP client to the hosted service and
signs you in on first run. It is a way *in*, not a second implementation — the engine is
this library, and the npm package named `memvara` is a name reservation with no client of
its own.

## On your own machine

```bash
pip install memvara
MEMVARA_DB=~/.memvara/memory.db memvara-mcp
```

JSON-RPC 2.0 over stdio, no SDK dependency — the server frames one JSON object per line in
about a hundred lines rather than pulling the reference SDK's dozen-package tree, which is
how the "numpy and nothing else" claim survives the server.

It **refuses to start without `MEMVARA_DB`** and prints the client configuration block
instead, so if your client says the server failed, run the command by hand and read what
it says. That printed block is the one to trust if any document drifts from the code.

```bash
memvara-mcp init --agent claude    # writes the client block, the skill tree and a note
```

[Deploying](../DEPLOY.md#2-as-an-mcp-server) has the per-client configuration, the full
environment table, and the two `command` traps that cost people an afternoon each.

### The scope is bound at startup

`MEMVARA_USER`, `MEMVARA_TENANT`, `MEMVARA_AGENT` and `MEMVARA_SESSION` are read from the
environment when the process starts and **cannot be changed by a tool call**. That is the
security property of the stdio transport: the process is the user, because the client
launched it with the user's environment, so there is no caller-supplied scope string for a
model to be talked into changing.

`MEMVARA_READ_ONLY=1` hides every tool that writes.

## The fourteen tools

| Tool | What it does |
|---|---|
| `memory_recall` | Look up what is already known about this user, rendered to read before answering |
| `memory_search` | Search and get back claim ids, scores and record types — and the tool for time travel on either clock |
| `memory_neighborhood` | What is connected to one entity, walked through stored facts rather than searched for |
| `memory_paths` | How two things are connected, if anything stored connects them |
| `memory_ask` | Answer about a *past* instant, and say whether the record has changed since |
| `memory_since` | What changed in this user's memory while you were away |
| `memory_standing` | Every standing preference recorded, with no query and no ranking |
| `memory_add` | Store what the user just said, in their own words |
| `memory_remember` | Record one exact fact as a triple, skipping extraction entirely |
| `memory_forget` | Retire a fact **because the record was wrong** |
| `memory_end` | Close out a fact that **has stopped being true** |
| `memory_history` | Every value one fact has ever held, with when each began |
| `memory_why` | Why one claim is believed: the turns it came from, and what extracted it |
| `memory_stats` | What this server is bound to, how much it holds, and whether it can extract |

`tests/test_docs.py` pins that list against `memvara/server/tools.py` — every name, once,
in the order the server declares them — so a tool added or renamed fails the suite rather
than leaving this table quietly wrong.

### The two that get confused, and the one that matters

`memory_forget` and `memory_end` are not synonyms and the difference is not recoverable
from the data afterwards:

- **`memory_end`** — the world changed. The value was true and has stopped being true.
  It keeps answering questions about the period it held.
- **`memory_forget`** — the record was wrong. The value was never true.

A write receipt once reported `retired 1` for a fact that had merely stopped being true,
which left a model reading its own memory tool with three names for two events. See
[provenance](../concepts/provenance.md#ended-retired-erased-three-words-three-different-events).

## Two things to check before writing

**`memory_stats` first.** If it reports `fast-path-only`, the deployment runs with no
extraction model — so a paragraph handed to `memory_add` that matches none of the fixed
sentence forms yields no fact. Use `memory_remember` with an explicit subject, predicate
and object: it needs no model and cannot mis-parse.

**`role` on `memory_add` decides what is extracted**, not just who is credited. The
deterministic matcher runs on every `role="user"` turn whatever the model configuration
is, and it strips quotation marks before it looks — so a first-person sentence quoted
inside a log or a pasted document is written down as a fact about whoever pasted it, at a
confidence above what they stated themselves. Pass `role="system"` for a transcript, a log
or a paste.

## Teach it your vocabulary

```bash
MEMVARA_PREDICATES=engineering,decisions memvara-mcp
```

Two vocabularies ship — `engineering` for infrastructure facts, `decisions` for what an
agent records about its own work — and a path to your own TOML file works in the same
list. Without one, every predicate outside the personal-assistant builtins falls to the
multi-valued, slow-decaying default, so nothing supersedes. See
[contradiction resolution](../concepts/contradiction-resolution.md#your-domain-needs-its-own-vocabulary).

## The packaged skill

`pip install memvara` ships a skill at `memvara/skills/memvara/`, which `memvara-mcp init`
writes into your agent's skill tree. It covers picking a surface, the correction sequence,
and what is worth storing — the judgement a tool description has no room for. It
deliberately does not repeat what a tool description already says.

---

Previous: [RAG and memory](../concepts/rag-vs-memory.md) · Next: [Frameworks](frameworks.md)
