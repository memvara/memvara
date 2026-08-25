# Hosted MCP

Paste a URL, click Allow, the twelve tools appear. Nothing to install on the
machine. This is the default path.

URL: `https://app.memvara.dev/mcp`

That Allow screen is OAuth. You are granting the client a project, not handing
it a password. The grant lasts **90 days**. After that, open the approval page
and click Allow again. A forgotten connector does not stay authorized forever.

## Plugin (skill + this URL together)

Claude Code uses a dedicated marketplace, not this library repo:

```
/plugin marketplace add memvara/claude-memvara
/plugin install memvara
```

Other coding agents:

- Codex / ChatGPT desktop marketplace: `memvara/codex-memvara`
- Cursor: `memvara/cursor-memvara`
- Grok: `memvara/grok-memvara`
- VS Code: `memvara/vscode-memvara`
- OpenCode: `memvara/opencode-memvara` (remote MCP in `opencode.json`)
- OpenClaw: `memvara/openclaw-memvara` (`mcp add` Streamable HTTP)

This package's `plugin/` directory is the source those repos copy. ChatGPT in
the browser still pastes the URL until OpenAI's public directory lists us.

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

`npx memvara` bridges a stdio client to the hosted server and signs you in on
first run, for a machine with no Python at all.

## The thirteen tools

`memory_recall`, `memory_search`, `memory_neighborhood`, `memory_paths`,
`memory_since`, `memory_standing`, `memory_add`, `memory_remember`,
`memory_forget`, `memory_end`, `memory_history`, `memory_why`,
`memory_stats`.

That is what this library serves. **A hosted deployment can be behind it**, and
saying so is more useful than a number that is wrong for one of the two: a
server upgrades on its own schedule, so the newest tool is the one most likely
to be missing. Ask the connection rather than this page — `tools/list`, or
simply whether the tool you want is one you can see. A tool that is absent is a
deployment that has not caught up, not a tool that was removed.

`erase`, `purge`, `reset`, `consolidate` are not tools. A read-only server
hides the four write tools.
