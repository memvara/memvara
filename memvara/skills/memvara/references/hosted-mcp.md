# Hosted MCP

Paste a URL, click Allow, the ten tools appear. Nothing to install on the
machine. This is the default path.

URL: `https://app.memvara.dev/mcp`

That Allow screen is OAuth. You are granting the client a project, not handing
it a password. The grant lasts **90 days**. After that, open the approval page
and click Allow again. A forgotten connector does not stay authorized forever.

## Plugin (skill + this URL together)

Clients that have a plugin format. The repo `memvara/memvara` *is* the
marketplace.

Claude Code:

```
claude plugin marketplace add memvara/memvara
claude plugin install memvara@memvara
```

Grok:

```
grok plugin marketplace add memvara/memvara
grok plugin install memvara --trust
```

Cursor and Copilot/VS Code: add the same marketplace, then install `memvara`.
The plugin ships this URL and the skill. It does not ship a local Python
command.

After install the client opens a browser. That is the product. There is no
API key in the plugin files.

## Paste the URL yourself

Only listed where that client's own docs describe a hosted URL plus browser
sign-in.

| Client | Where |
|---|---|
| Claude (Desktop / claude.ai) | Settings → Connectors → Add custom connector |
| ChatGPT | Developer mode, then a custom connector. On Team/Enterprise, admins only. |
| Claude Code | `claude mcp add --transport http memvara https://app.memvara.dev/mcp` |
| Cursor | `"url"` under `mcpServers` in `.cursor/mcp.json` |
| VS Code | MCP: Add Server → HTTP, or `"type": "http"` under `servers` (not `mcpServers`). Needs 1.101+. |

Windsurf and Zed are not on this list. They stay on the local command path.

Per-client clicks: https://memvara.dev/docs/agents

## Local process (fallback)

When the store file must stay on a laptop, or the client can only launch a
command:

```
memvara-mcp init --agent claude
```

`--agent` is `claude`, `cursor`, or `grok`. `--skill-only` writes the skill
and leaves `.mcp.json` alone, for a client that already has the hosted URL.

`MEMVARA_DB` must be an absolute path. `command` is an interpreter that
imports `memvara`, not whichever `python3` a GUI `PATH` finds.

No `npx`. The npm package is a placeholder.

## The ten tools

`memory_recall`, `memory_search`, `memory_since`, `memory_add`,
`memory_remember`, `memory_forget`, `memory_end`, `memory_history`,
`memory_why`, `memory_stats`.

Same list on hosted and local. `erase`, `purge`, `reset`, `consolidate` are
not tools. A read-only server hides the four write tools.
