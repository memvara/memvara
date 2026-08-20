# memvara plugin

The hosted MCP server and the skill, in one install. After this, the agent
can remember facts across sessions and has the rules for doing it without
forging the history.

This is for a **coding agent** — Claude Code, Grok, Cursor, Copilot. If you
are writing the agent loop yourself, skip to [Your own agent](#your-own-agent).

## Install

Claude Code uses the dedicated marketplace
[memvara/claude-memvara](https://github.com/memvara/claude-memvara):

```
/plugin marketplace add memvara/claude-memvara
/plugin install memvara
```

This `plugin/` directory is the source layout those repos copy. Do not
`marketplace add memvara/memvara` — this repository is the library.

- Claude Code: https://github.com/memvara/claude-memvara
- Codex: https://github.com/memvara/codex-memvara
- Cursor: https://github.com/memvara/cursor-memvara
- Grok: https://github.com/memvara/grok-memvara
- VS Code: https://github.com/memvara/vscode-memvara

The plugin points at `https://app.memvara.dev/mcp`. The first connection
opens a browser so you can click Allow. That grant lasts 90 days.

It does not start a local Python process and it does not ship a Node
installer.

Claude Desktop, claude.ai, and ChatGPT do not install plugins this way.
Paste the same URL into that client's connector settings instead:
https://memvara.dev/docs/agents

Windsurf and Zed are local-command clients. Use `memvara-mcp init`, not this
plugin.

## What you get

- The ten `memory_*` tools, on the hosted store.
- The `memvara` skill (`skills/memvara/`), which is the judgment the tool
  descriptions cannot carry: which surface to use, the dispute sequence,
  scope, clocks, erasure.

The skill files in this directory are a copy of `memvara/skills/memvara/`
in the Python package. A test fails if they drift.

## Your own agent

A plugin does not install into LangChain, CrewAI, or a loop you wrote.

- **Python:** `pip install memvara`, then `from memvara import Memvara`.
  See https://memvara.dev/docs/quickstart
- **Anything else, including JS:** speak MCP as a client against the URL
  above, or call the commercial REST API. The npm package `memvara` is a
  name reservation and does nothing.
- The skill is markdown. You can paste `skills/memvara/SKILL.md` (and the
  files under `references/`) into a system prompt. It is not a PyPI extra.

`skills/memvara/references/integrate.md` is the decision between those
paths.

## Team

Check the marketplace into the project so a clone plus trust is the setup.
One hosted project, scope by user. Do not commit a `MEMVARA_DB` that is a
path on your laptop.

## License

Apache-2.0, same as the library.
