# How to run your own MCP server

If you want memory available to a team, or to a client that isn't running Python, run the
`memvara-mcp` server yourself and connect to it over standard input/output using your MCP
client's own process-launching configuration.

## The minimal setup

```bash
pip install memvara
MEMVARA_DB=~/.memvara/memory.db memvara-mcp
```

The server takes no command-line flags for its ordinary operation — everything is
configured through environment variables, because that's what an MCP client's own
settings file lets a user actually control per machine.

## The full configuration

| Variable | Meaning | Default |
|---|---|---|
| `MEMVARA_MODE` | `local` (open a file on this machine) or `cloud` (talk to a hosted deployment) | `local` |
| `MEMVARA_DB` | Path to the SQLite file. Required in local mode. Use `:memory:` for a throwaway store. | — |
| `MEMVARA_USER` | Whose memory this server reads and writes. Leave unset to see the whole tenant. | — |
| `MEMVARA_TENANT` | The top-level isolation boundary. | `default` |
| `MEMVARA_AGENT` | Narrows scope further, if you have multiple agents sharing a tenant. | — |
| `MEMVARA_SESSION` | Narrows scope to one session. Memory here isn't visible to other sessions — leave unset for anything you want to persist. | — |
| `MEMVARA_LLM` | `none` (offline, recognizes only fixed sentence forms) or `anthropic` (needs `ANTHROPIC_API_KEY`). Local mode only. | `none` |
| `MEMVARA_EMBEDDER` | `hashing` (offline, keyword-based), `local` (needs `memvara[local-embed]`, real semantic search), or `auto`. | `hashing` |
| `MEMVARA_PREDICATES` | Which fact vocabularies to load — shipped pack names, paths to your own TOML files, or a comma-separated mix. | Built-in vocabulary only |
| `MEMVARA_READ_ONLY` | Set to `1` to hide every tool that writes, leaving only reads. | unset |

In cloud mode, `MEMVARA_DB` is ignored, and `MEMVARA_LLM`/`MEMVARA_EMBEDDER` are refused
outright — extraction and embedding run inside the hosted deployment, so a value set here
would never actually be used.

## Point your client at it

Most MCP clients want a command, an argument list, and an environment block in their own
configuration file. The exact format differs by client, so generate it rather than typing
it by hand:

```bash
memvara-mcp init --agent claude
```

This writes the connection block with `MEMVARA_DB` already set to an absolute path (a
relative path in an MCP client's config is a common source of "it can't find the file"
errors, since the client may launch the process from a different working directory than
you expect). `--agent cursor` and `--agent grok` write configuration for those clients
instead. `memvara-mcp init --help` lists every option.

## Connect to a hosted deployment instead

If you're pointing this server at Memvara's hosted API rather than a local file, sign in
first:

```bash
pip install "memvara[cloud]"
memvara-mcp login --project my-project
```

This walks you through a device-code sign-in flow in your browser and writes an API key
to `~/.memvara/credentials.json`. Then start the server with:

```bash
MEMVARA_MODE=cloud memvara-mcp
```

## Diagnosing a startup failure

If your MCP client reports that the server failed to start, run the exact command by hand
in your own terminal:

```bash
MEMVARA_DB=~/.memvara/memory.db memvara-mcp
```

The server prints a clear, specific reason and exits — for example, if the store was
created with a different embedding model than the one you've now configured, it tells you
exactly which `MEMVARA_EMBEDDER` value would match. This message is more reliable than
whatever your MCP client shows you, since the client is usually just reporting that the
process exited.

## Next steps

- [Use Memvara in an AI coding assistant](use-memvara-in-an-ai-coding-assistant.md) — the
  full picture, including the hosted option if you don't need to self-host.
- [Configuration reference](../reference/cli-and-configuration.md) — every environment
  variable and command, in one place.
