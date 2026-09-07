# CLI and configuration reference

## `memvara-mcp` — run the MCP server

```bash
memvara-mcp
```

Installed automatically by `pip install memvara`. Serves memory as an MCP server over
standard input and output. Takes no flags for ordinary operation — every setting is an
environment variable, listed below.

```bash
memvara-mcp --version    # print the installed version
memvara-mcp --help       # print usage and the full environment variable list
```

### `memvara-mcp init` — generate a client configuration

```bash
memvara-mcp init --agent claude
```

Writes the connection settings a specific AI assistant needs (with `MEMVARA_DB` already
set to an absolute path), plus a packaged skill that teaches the assistant how to use
memory correctly.

| Flag | Meaning |
|---|---|
| `--agent NAME` | Which assistant to write configuration for — for example `claude`, `cursor`, or `grok`. |
| `--skill-only` | Write the skill files, but skip the connection configuration. |

Run `memvara-mcp init --help` for the full, current list of options.

### `memvara-mcp login` — sign in to a hosted deployment

```bash
memvara-mcp login --project my-project
```

Requires `pip install "memvara[cloud]"`. Runs a device-code sign-in flow in your browser
and writes an API key to `~/.memvara/credentials.json`, used automatically by
`MEMVARA_MODE=cloud`. Run `memvara-mcp login --help` for the full options.

## Environment variables

All of these configure `memvara-mcp`. They're read once, when the server process starts,
and can't be changed afterward by anything the connected assistant sends — this is what
keeps a model from being able to reach into a different user's memory.

| Variable | Meaning | Default |
|---|---|---|
| `MEMVARA_MODE` | `local` or `cloud`. Local opens a file on this machine; cloud talks to a hosted deployment over its API. | `local` |
| `MEMVARA_DB` | **Required in local mode.** Path to the SQLite file. Created automatically on first use. Use `:memory:` for a store that disappears when the process ends. | — |
| `MEMVARA_USER` | Whose memory this server serves. Leave unset to see the whole tenant. | — |
| `MEMVARA_TENANT` | The top-level isolation boundary above the user. | `default` |
| `MEMVARA_AGENT` | Narrows scope further, for setups with multiple agents. | — |
| `MEMVARA_SESSION` | Narrows scope to a single session. Memory written here is invisible to other sessions — leave this unset for anything meant to persist. | — |
| `MEMVARA_LLM` | `none` (offline, only fixed sentence patterns are recognized) or `anthropic` (needs `ANTHROPIC_API_KEY`). Local mode only — in cloud mode this is refused, because extraction runs inside the deployment instead. | `none` |
| `MEMVARA_EMBEDDER` | `hashing` (offline, keyword-based, 512 dimensions), `hashing:<dim>`, `local` or `local:<model>` (needs `memvara[local-embed]`, real semantic search), or `auto`. A store can only be opened with the same embedder that wrote it — the server tells you exactly which value to use if there's a mismatch. | `hashing` |
| `MEMVARA_PREDICATES` | Which fact vocabularies to load: shipped pack names (`engineering`, `decisions`), paths to your own TOML files, or a comma-separated mix. Needs Python 3.11+. | Built-in vocabulary only |
| `MEMVARA_READ_ONLY` | `1` to hide every tool that writes, exposing only reads. | unset |

## Optional install extras

| Extra | Adds | Install |
|---|---|---|
| `anthropic` | Fact extraction from free text, using Claude | `pip install "memvara[anthropic]"` |
| `openai` | Fact extraction from free text, using GPT | `pip install "memvara[openai]"` |
| `local-embed` | A real semantic embedding model, run locally | `pip install "memvara[local-embed]"` |
| `rerank` | A more accurate cross-encoder re-ranking step for search | `pip install "memvara[rerank]"` |
| `cloud` | Connecting to a hosted Memvara deployment, and `memvara-mcp login` | `pip install "memvara[cloud]"` |
| `langchain` | The LangChain retriever and chat history adapters | `pip install "memvara[langchain]"` |
| `llama-index` | The LlamaIndex retriever and memory block adapters | `pip install "memvara[llama-index]"` |
| `langgraph` | The LangGraph store adapter | `pip install "memvara[langgraph]"` |
| `crewai` | The CrewAI storage backend adapter | `pip install "memvara[crewai]"` |

## Connecting to a hosted deployment

Instead of opening a local file, pass credentials directly when creating a store, and
Memvara returns a `RemoteMemvara` object with the exact same methods as the local
`Memvara` class:

```python
from memvara import Memvara

mem = Memvara(api_key="your-key")             # or base_url="https://your-deployment/..."
```

Because it's the same interface, code written against a local store works unchanged
against a hosted one — a function that takes a `Memvara` object and calls `search()` on
it can't tell which kind it was actually given.

## Related pages

- [Run your own MCP server](../how-to-guides/run-your-own-mcp-server.md) — a full
  walkthrough using these settings.
- [Install Memvara](../how-to-guides/install-memvara.md) — choosing which extras to
  install.
