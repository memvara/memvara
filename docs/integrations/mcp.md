# MCP

Memvara speaks the Model Context Protocol, so an agent that supports MCP gets memory as
twenty-two tools without you writing any code. **There are three ways to reach them, and
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

Claude, Claude Code, ChatGPT, Codex, Cursor, Grok, VS Code, OpenCode and OpenClaw each
have their own page at [memvara.dev/docs/cloud](https://memvara.dev/docs/cloud), and all
of them paste the same hosted URL: `https://app.memvara.dev/mcp`, approved once in the
browser. **The grant lasts until you revoke it, or ten years, whichever comes first.**

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

`MEMVARA_PROJECT` is the fifth part of that scope: the repository, as `host/owner/repo`.
Leave it unset and the server works it out at startup from the git remote of the directory
the client started it in, so every clone and worktree of one repository shares one
project. A fact about the codebase is then filed under that repository and is not
recalled in another one, while a preference whose predicate is declared global is written
without a project and is recalled everywhere. `MEMVARA_FEATURE_PROJECT_SCOPE=0` turns the
derivation off.

`MEMVARA_READ_ONLY=1` hides every tool that writes. `MEMVARA_FEATURE_PROFILE=0` hides the
profile tool listed below, `MEMVARA_FEATURE_FORGET_MATCHING=0` hides the two `_matching`
tools, `MEMVARA_FEATURE_LINKS=0` hides the link tool, `MEMVARA_FEATURE_DOCUMENTS=0`
hides the four document tools, and `MEMVARA_FEATURE_END_REASON=0` removes the `reason`
and `until_reason` arguments from every tool that has them.
`MEMVARA_FEATURE_RETRIEVAL_CHUNKS=0` makes a local server store each document as one
chunk instead of passages of about 1,000 characters.
`MEMVARA_FEATURE_EXTRACTION_CHUNKS=1` (off by default) makes the extraction model read a
turn over 6,000 characters in pieces, one call per piece. A feature marked (off by default)
stays off until its variable says `1`; every other feature is on until its variable says `0`.

With `MEMVARA_LLM` set to a model, the search and recall tools also ask that model for a
few other phrasings of each query and for the dates it names, before searching. That is one
model call of up to 10 seconds per read; a failed call serves the ordinary search.
`MEMVARA_FEATURE_QUERY_REWRITE=0` turns it off. The recall tool can also put a model's
summary above its notes when called with `synthesize`, and `MEMVARA_FEATURE_SYNTHESIS=0`
removes that argument. With `MEMVARA_LLM=none`, which is the default, neither happens.

`MEMVARA_ANCHORED=1` makes the three read tools answer only from memories the question is
demonstrably about, so a question about an entity this store has never heard of returns
nothing rather than the nearest memory about somebody else. `MEMVARA_READ_W_GRAPH=1.0`
switches on the retrieval leg that walks out of the entities a question names, which is
what lets an anchored read still reach a fact the question reaches only through another
one. Both ship off; [`docs/DEPLOY.md`](../DEPLOY.md#choosing-how-this-server-reads) has the
measurements and the case for each.

## The twenty-two tools

| Tool | What it does |
|---|---|
| `memory_recall` | Look up what is already known about this user, rendered to read before answering |
| `memory_search` | Search and get back claim ids, scores and record types — and the tool for time travel on either clock |
| `memory_neighborhood` | What is connected to one entity, walked through stored facts rather than searched for |
| `memory_paths` | How two things are connected, if anything stored connects them |
| `memory_ask` | Answer about a *past* instant, and say whether the record has changed since |
| `memory_since` | What changed in this user's memory while you were away |
| `memory_standing` | Every standing preference recorded, with no query and no ranking |
| `memory_profile` | Standing preferences, recent arrivals, memories grouped into buckets and, with a query, relevant memories, in one call |
| `memory_add` | Store what the user just said, in their own words |
| `memory_remember` | Record one exact fact as a triple, skipping extraction entirely. With `replaces`, it ends one named fact in the same write and records why |
| `memory_forget` | Retire a fact **because the record was wrong**, with an optional reason |
| `memory_end` | Close out a fact that **has stopped being true**, with an optional reason |
| `memory_end_matching` | End every live fact that matches a query: a preview first, then a confirming call that ends exactly the facts it listed |
| `memory_forget_matching` | Retire every live fact that matches a query, with the same preview and confirmation |
| `memory_link` | Record that one fact adds detail to another (`extends`) or was inferred from another (`derives`) |
| `memory_history` | Every value one fact has ever held, with when each began and any reason it was closed |
| `memory_why` | Why one claim is believed: the turns it came from, what extracted it, the facts it is linked to, and any reason it was closed |
| `memory_stats` | What this server is bound to, how much it holds, and whether it can extract |
| `memory_add_document` | Store a whole document, split into passages that `memory_recall` with `include_episodes` returns. Sending it again with the same `custom_id` updates it |
| `memory_get_document` | One stored document's record: title, file path, chunk count and processing status |
| `memory_list_documents` | The stored documents, newest first, filtered by file path prefix or status, a page at a time |
| `memory_delete_document` | Erase one document's text. A memory whose only source was the document is **retired**, not erased |

`tests/test_docs.py` pins that list against `memvara/server/tools.py` — every name, once,
in the order the server declares them — so a tool added or renamed fails the suite rather
than leaving this table quietly wrong.

**On the hosted URL, a release that adds a tool does not reach a session already
connected.** The tool list is negotiated once, at the handshake, and the server declares
`listChanged: false` — accurate for a stdio server, which your client starts and stops
itself, and not something the hosted deployment can honour across a redeploy. So an
editor or agent session left open across a release keeps the list it was given, and a
tool added by that release is simply absent rather than erroring.

Reconnecting is the whole fix: restart the client, or remove and re-add the server.
If a tool documented above is missing, check that first — it reads exactly like a tool
that has not shipped yet, which is the way this costs an afternoon.

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

It also ships one script, `scripts/memvara_auth.py`: the device-code flow, standard
library only, for the case where the browser grant will not finish and the agent has no
`memvara-mcp` to fall back on. The plugin repositories vendor the skill tree whole, so
that is the one copy every host gets.

---

Previous: [RAG and memory](../concepts/rag-vs-memory.md) · Next: [Frameworks](frameworks.md)
