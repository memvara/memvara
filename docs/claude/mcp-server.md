# The MCP server

`memvara-mcp` is the console script that serves this library's memory to a coding agent over
the Model Context Protocol. It speaks JSON-RPC on stdin and stdout, is meant to be launched
by an MCP client rather than run by hand, and is configured entirely from the environment.
Fourteen tools are exposed, from `memory_recall` and `memory_search` through the writes to
`memory_stats`.

The server has two modes. In local mode it opens a SQLite file on this machine and runs the
whole engine in process. In cloud mode it opens no file at all and serves the same fourteen
tools from a hosted deployment over that deployment's `/v1` API.

## Where the code is

- Entry point: `memvara/server/cli.py` — `main()`, and the `USAGE` text that documents every
  environment variable. `memvara/server/__main__.py` makes the package runnable.
- Tools: `memvara/server/tools.py` — `TOOLS`, `Tool`, `ToolContext`, `safe_line()`,
  `safe_detail()`, `STORED_HEADER`. Each tool's description is documentation a model reads
  at runtime; `.claude/rules/tool-descriptions.md` loads when you open this file.
- Protocol: `memvara/server/mcp.py` — `MemvaraMCPServer`, `SUPPORTED_PROTOCOLS`,
  `INSTRUCTIONS`; `memvara/server/protocol.py` — `serve_stdio()`, `success()`, `failure()`,
  and the JSON-RPC error codes.
- Configuration: `memvara/server/config.py` — `ServerConfig` and `ConfigError`. Every
  `MEMVARA_*` variable is read here and nowhere else.
- Argument checking: `memvara/server/validate.py` — `validate()` and `ToolError`, which turn
  a bad tool argument into a message that names the part that was wrong.
- Setup: `memvara/server/init.py` — what `memvara-mcp init` writes, including `AGENTS`,
  `MARKER` and `client_entry()`. `memvara/server/login.py` — `login()`, the device-code flow
  that writes a cloud credential.
- The interface both engines satisfy: `memvara/server/memory_api.py` — the `MemoryAPI`
  protocol, which is what lets one server serve either a local `Memvara` or a
  `RemoteMemvara`.
- Tests: `tests/test_server.py`, `tests/test_init.py`, `tests/test_login.py`,
  `tests/test_config_cloud.py`, `tests/test_memory_api_protocol.py`.
- Documentation: [the MCP integration page](../integrations/mcp.md) lists the fourteen tools
  and the three ways to reach them; [DEPLOY.md](../DEPLOY.md) has the environment table.

## How the pieces fit

`cli.main()` reads the environment into a `ServerConfig`, builds either a `Memvara` or a
`RemoteMemvara` from it, wraps that in a `MemvaraMCPServer`, and hands the server to
`serve_stdio()`. A tool call arrives as JSON-RPC, `validate()` checks its arguments against
the tool's declared schema, the tool runs against the `MemoryAPI`, and the result comes back
as text.

`memvara-mcp init --agent claude` writes the three files a client looks for: the server block
in the client's .mcp.json, the packaged skill tree, and a short note appended to the project's
`CLAUDE.md` between marker comments. The `--agent cursor` and `--agent grok` variants write
the same skill into those clients' skill directories, and `--skill-only` skips the .mcp.json step.
`memvara-mcp login --project NAME` is the separate path for cloud mode; it writes an API key
that `MEMVARA_MODE=cloud` then uses.

## Configuration

`docs/DEPLOY.md` has the full table, with defaults and which combinations are refused; keep
that table as the source and this paragraph as the map. Every setting is an environment
variable read by `memvara/server/config.py`.
`MEMVARA_MODE` chooses local or cloud. `MEMVARA_DB` is the SQLite path, required in local
mode. `MEMVARA_TENANT`, `MEMVARA_USER`, `MEMVARA_AGENT` and `MEMVARA_SESSION` bind the scope.
`MEMVARA_LLM` chooses the extraction backend from `none`, `anthropic` and `openai`, with the
`MEMVARA_LLM_MODEL`, `MEMVARA_LLM_MAX_TOKENS`, `MEMVARA_LLM_MAX_CLAIMS`,
`MEMVARA_LLM_TERSE_CLAIMS`, `MEMVARA_LLM_EXTRACT_SYSTEM` and `MEMVARA_LLM_EXTRA_BODY`
variables tuning it. `MEMVARA_EMBEDDER` chooses the embedder. `MEMVARA_PREDICATES` loads
declared vocabularies. `MEMVARA_READ_ONLY` hides every tool that writes.
`MEMVARA_ADVISE_REPLACEMENTS` turns on replacement advice. `MEMVARA_ANCHORED` makes
`anchored` the default on the three read tools, and `MEMVARA_READ_W_GRAPH` sets the weight
on the graph leg of retrieval — the two settings that decide how this server reads, both
off by default. `MEMVARA_API_KEY` and `MEMVARA_SERVER_URL` are the cloud credentials.

## Invariants and assumptions

- **The scope is bound at startup and no tool call can change it.** That is what stops a
  model reaching another user's memory, and it is why `tenant` is not a tool argument.
- **No MCP client can backdate the transaction clock.** This is invariant 8 in
  [INTERNALS.md](../INTERNALS.md). A client may state when a fact was true; it may not state
  when this store came to believe it.
- **Cloud mode refuses `MEMVARA_LLM` and `MEMVARA_EMBEDDER`.** Extraction and embedding run
  inside the deployment, so a value set here would never be used, and silently ignoring it
  would be worse than refusing to start.
- **A configuration error is a refusal that names the fix.** `ConfigError` messages say which
  variable was wrong and what a valid value looks like, because the server is launched by a
  client whose only feedback channel is the startup failure.
- **The documented example and the written entry may not drift.** `tests/test_init.py`
  compares what `init` writes against the example in the help text.
- **Tool text and the packaged skill do not repeat each other.**
  `tests/test_init.py::test_the_skill_does_not_restate_a_tool_description` holds that line.
  See `.claude/rules/packaged-skill.md`.

## Read next

`memvara/server/cli.py`'s `USAGE` string is the authoritative list of environment variables
and is what a user sees on `--help`. The module docstring in `memvara/server/tools.py`
explains how a tool description is written and why the wording matters.

Next: [the hosted client](remote-and-cloud.md).
