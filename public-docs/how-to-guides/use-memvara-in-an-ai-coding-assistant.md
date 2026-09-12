# How to use Memvara in an AI coding assistant

Memvara speaks the Model Context Protocol (MCP), so any assistant that supports MCP —
Claude, Claude Code, Cursor, ChatGPT, Codex, and others — can read and write memory
directly, without you writing any integration code. This guide covers the three ways to
connect, and helps you pick the right one.

## Which option do you want?

| You want... | Use |
|---|---|
| Memory to just work in your editor, nothing to install or run yourself | The hosted service |
| Memory stored on your own machine, in a file you control | The local server |
| Memory for a team, or for a client that isn't Python | Your own deployment |

## Option 1: The hosted service (easiest)

If you're using Claude Code, install the plugin:

```
/plugin marketplace add memvara/claude-memvara
/plugin install memvara
```

For other clients — Claude, ChatGPT, Codex, Cursor, Grok, VS Code, and more — see
[memvara.dev/docs/cloud](https://memvara.dev/docs/cloud) for the specific setup for each
one. They all point at the same hosted address:

```
https://app.memvara.dev/mcp
```

You approve access once in your browser, and the connection stays authorized until you
revoke it (or ten years pass, whichever comes first).

If you don't have Python installed at all, `npx memvara` connects any MCP-capable client
to the hosted service and signs you in on first run.

## Option 2: Run the server yourself, locally

```bash
pip install memvara
MEMVARA_DB=~/.memvara/memory.db memvara-mcp
```

This runs entirely on your machine, storing memory in the SQLite file you name. It speaks
plain JSON over standard input and output — there's no separate SDK dependency, which is
part of why the whole install stays as small as "numpy and nothing else."

The server refuses to start unless `MEMVARA_DB` is set, and prints the configuration your
MCP client needs instead of starting silently. If your assistant says the connection
failed, run the command by hand in a terminal first and read what it prints — that
message is the most reliable source of what went wrong.

To generate the configuration automatically for a specific assistant:

```bash
memvara-mcp init --agent claude
```

This writes the connection settings your client needs, along with a packaged skill that
teaches the assistant how to use memory correctly — when to write a new fact, and which
of the three ways to correct one applies.

### The scope is fixed when the server starts

Four environment variables — `MEMVARA_USER`, `MEMVARA_TENANT`, `MEMVARA_AGENT`, and
`MEMVARA_SESSION` — decide whose memory this server instance can see, and they're read
once, when the process starts. They cannot be changed by anything the assistant sends
afterward. This is a deliberate security boundary: it means a model can never be talked
into reaching into a different user's memory, because there's no way for it to even ask.

Set `MEMVARA_READ_ONLY=1` if you want the assistant to be able to read memory but never
write to it.

## What the assistant can do

Once connected, your assistant gets fourteen tools, covering searching memory, recalling
what it knows about you, recording new facts, correcting old ones, and asking what was
true at a past moment. See [MCP tools reference](../reference/mcp-tools.md) for the
complete list.

## Teach it your own vocabulary

The facts Memvara understands by default are personal — where someone lives, works, and
so on. If you're using memory for something else, like tracking engineering decisions,
tell the server which vocabulary to use:

```bash
MEMVARA_PREDICATES=engineering,decisions memvara-mcp
```

See [Define your own fact types](define-your-own-fact-types.md) for what this changes and
why it matters.

## Next steps

- [Run your own MCP server](run-your-own-mcp-server.md) — deploying the server for a team
  or for non-Python clients.
- [MCP tools reference](../reference/mcp-tools.md) — the full list of tools and what each
  one does.
