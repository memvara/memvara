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
- OpenCode: https://github.com/memvara/opencode-memvara
- OpenClaw: https://github.com/memvara/openclaw-memvara

The plugin points at `https://app.memvara.dev/mcp`. The first connection
opens a browser so you can click Allow. That grant lasts until you revoke
it, or ten years, whichever comes first.

It does not start a local Python process and it does not ship a Node
installer.

Claude Desktop, claude.ai, and ChatGPT do not install plugins this way.
Paste the same URL into that client's connector settings instead:
https://memvara.dev/docs/cloud

Windsurf and Zed are local-command clients. Use `memvara-mcp init`, not this
plugin.

## What you get

- The ten `memory_*` tools, on the hosted store.
- The `memvara` skill (`skills/memvara/`), which is the judgment the tool
  descriptions cannot carry: which surface to use, the dispute sequence,
  scope, clocks, erasure.

The skill files in this directory are a copy of `memvara/skills/memvara/`
in the Python package. A test fails if they drift.

## Hooks

`hooks/` is the canonical hook tree. It is what makes memory automatic —
recall on every prompt, capture when a turn ends — and every plugin
repository vendors it from here, byte for byte, at the commit its own
`hooks.lock` names. Two tests over there compare the copy: one against that
sha, one against this branch's tip, because a lock and a copy frozen
together agree with each other forever.

It sits at the top level rather than inside the `memvara` package on
purpose. `pyproject.toml` says `packages = ["memvara"]`, so `memvara/hooks/`
would ship an importable `memvara.hooks.lib` to everyone who runs
`pip install memvara`. Here it stays in the sdist, out of the wheel, and the
canonical path is the same string as the vendored one — so the sync is a
copy with no rewriting in it to get wrong.

`hooks/core/` is host-neutral; `hooks/hosts/<id>.py` is one client's
protocol written down as data — its event names, stdin keys, reply keys,
timeouts. The registration file a client actually reads is generated from
that record, not vendored:

```
python3 plugin/hooks/tools/generate.py claude
```

Seven repositories vendor this one tree and each registers a different
client, so a `hooks.json` committed here would be one of them shipped to all
of them. This repository ignores that path for exactly that reason.

Five hook features can be switched off in `~/.memvara/settings.json`, a
flat object of `feature_name: true|false` where a missing key means the
feature's default. Each of these five is on by default.
`MEMVARA_FEATURE_<NAME>=0|1` overrides the file for one process.

- `project_scope`: the hooks work out the project from the repository's
  `origin` remote and send it to the server as a `Memvara-Project` header.
  Every clone and worktree of one repository gives the same project.
- `recall_mark`: every memory line the hooks inject starts with `⋈ `, so a
  reader can tell recalled memory from the rest of the context. Capture never
  mines a marked line, with the switch on or off.
- `status_line`: the hooks count, per session, the memory lines recalled,
  the read-only memory tools called and the facts captured, in
  `~/.memvara/.hooks/counts/<session>.json`.
- `query_rewrite`: lets the recall hook ask the local store's model to
  rewrite each prompt's query before it searches. On its own the switch
  does nothing. The hook asks for a rewrite only after
  `/memvara:setup verify-key` has made one test call to the configured
  model and the model answered it, and only while that same model is
  configured. A rewrite is one model call per prompt, billed by your
  provider. The hook waits at most 5 seconds for it and otherwise uses the
  plain result, so a slow or failing model never costs you your memories.
  A hosted install is always asked for a plain read, because the hosted
  service would use your organisation's key, which setup cannot check.
  The result of the check is kept in `~/.memvara/.hooks/read_model.json`.
  If the provider rejects the key during a later prompt, that prompt gets
  the plain result and the hook stops rewriting until the key is checked
  again.
- `agentic_capture`: when a turn ends, the capture hook lets the headless
  agent command search your memory with read-only tools before it
  suggests changes: a new fact, a new value for a stored fact, the end of
  a stored fact, or a link between two facts. The hook checks each
  suggestion and writes the ones that pass. The model cannot write
  anything itself, and it can use only your memvara server. It runs under
  your own login, like the single extraction call it replaces. Switched
  off, or when the agentic run fails, capture makes that single call
  instead. It runs only where the headless agent command is the host's
  own extractor. Measured on one machine, a turn cost a mean 19,400
  input and 1,200 output tokens and 17 seconds, with up to four searches.
  The single call cost 45,300 input tokens there, because it loads your
  instruction files and plugins; with small ones it costs about 21,000,
  which is then about the same as an agentic turn. The numbers and how
  they were measured are in `CHANGELOG.md`. On the hosted service, one
  capture turn counts as one recall, however many searches it makes.
  That takes effect once the hosted service supports the
  `Memvara-Capture-Run` header the hook sends; until then each search
  counts as a recall.

The file lists the library's other switches too, with the same defaults as
the MCP server. `extraction_chunks` and `agentic_extraction` are the two that
are off by default. No hook reads those; the MCP server reads them from
`MEMVARA_FEATURE_<NAME>`.

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
