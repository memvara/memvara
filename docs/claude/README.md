# Claude context index

This directory holds the context that Claude Code does not load at the start of a session.
The root `CLAUDE.md` carries only the rules a reviewer has to enforce on every pull request.
Everything else lives here and in `.claude/rules/`, and is read when a task actually needs
it.

Find your task in the first column, read the file in the second, then go to the code in the
third.

| If the task touches | Read first | Primary code |
|---|---|---|
| Claims, slots, the two clocks, and the difference between ending, retiring and erasing | [memory-model.md](memory-model.md) | `memvara/core.py`, `memvara/types.py`, `memvara/store/` |
| Writing memory: the gate, extraction, contradiction handling | [write-pipeline.md](write-pipeline.md) | `memvara/write/`, `memvara/llm/` |
| Reading memory: search, recall, ranking, reranking | [retrieval.md](retrieval.md) | `memvara/retrieve/`, `memvara/rerank/`, `memvara/embed/` |
| The MCP server, its tools, and how a deployment is configured | [mcp-server.md](mcp-server.md) | `memvara/server/` |
| The hosted client and its relationship to memvara-cloud | [remote-and-cloud.md](remote-and-cloud.md) | `memvara/remote/` |
| Background maintenance, predicate vocabularies, the graph leg | [consolidation-and-graph.md](consolidation-and-graph.md) | `memvara/consolidate/`, `memvara/packs/`, `memvara/retrieve/traverse.py` |
| Counters, benchmark scripts, the demo harness | [telemetry-and-benchmarks.md](telemetry-and-benchmarks.md) | `memvara/telemetry.py`, `bench/`, `demo/` |
| Cutting a release, the npm bridge, the seven plugin repositories | [release-and-plugins.md](release-and-plugins.md) | `release/`, `npm/`, `plugin/`, `scripts/sync_plugin_repos.py` |
| How work is done here, with the incident behind each rule | [working-here.md](working-here.md) | none — this is process, not code |

## Rules that load themselves

A file under `.claude/rules/` carries a `paths:` list in its front matter and is loaded only
when Claude reads a file matching one of those globs. You do not need to open these; they
arrive when they are relevant.

| Rule file | Loads when you touch |
|---|---|
| `.claude/rules/tool-descriptions.md` | `memvara/server/tools.py` |
| `.claude/rules/packaged-skill.md` | anything under `memvara/skills/` or `plugin/` |
| `.claude/rules/doctests.md` | any Python file under `memvara/` |

## What this directory is not

It is not a second copy of the documentation. Where a subject already has a written
explanation — the module contracts in [INTERNALS.md](../INTERNALS.md), the design
decisions in [DESIGN.md](../DESIGN.md), the measurements in
[BENCHMARKS.md](../BENCHMARKS.md) — the page here points at the section rather than
repeating it. If you find a page here that contradicts the code, the code is right; fix the
page in the same commit.

Next: [How work is done here](working-here.md), or pick the row above that matches your task.
