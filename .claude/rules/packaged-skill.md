---
paths:
  - "memvara/skills/**"
  - "plugin/**"
---

# The packaged skill is vendored downstream

`memvara/skills/memvara/` is the source of truth for the memvara skill. It ships inside the
wheel, it is what `memvara-mcp init` writes into a project, and it is vendored into seven
dedicated plugin repositories named in `plugin-repos.txt`. Each of those pins the commit sha
it copied and diffs its copy against that sha in CI.

An edit here is therefore an edit in eight repositories. Treat it that way.

## The two trees must stay identical

`plugin/skills/memvara/` is the copy the marketplace plugin ships, and it has to match
`memvara/skills/memvara/` byte for byte. `tests/test_plugin.py` compares them. Change one and
you change both, in the same commit.

`scripts/sync_plugin_repos.py` is what copies the canonical tree into a plugin-repository
checkout. `.github/workflows/skill-notify.yml` pings those repositories when
`memvara/skills/**` or `plugin-repos.txt` changes.

There is exactly one sanctioned transform in any downstream copy: in the Claude plugin
repository the front-matter `name: memvara` becomes `name: memory`, so the client renders the
command as `/memvara:memory`. The drift test compensates for that single line and no other
byte.

## Converting its prose is its own commit

The writing rules in `CLAUDE.md` apply to this tree, but a rewrite here is a real change in
seven other repositories. Do it deliberately, in a commit that does nothing else, and never
as a drive-by while editing something nearby.

## The skill does not restate a tool description

The skill carries what no single tool description can: the sequences that span several tools,
the bound scope, what is worth storing, how to correct a disputed memory. A second copy of a
tool description is not redundancy — it is a second place that can go stale, and the tool
list in this tree rotted exactly that way once.

`tests/test_init.py::test_the_skill_does_not_restate_a_tool_description` is the guard, and
the tests either side of it check that the skill still carries the five things only it can.

## `plugin/hooks/` is the second vendored tree

Same relationship, no sanctioned transform at all: the canonical path and the vendored path
are the same string, so the sync is a plain copy and the gate is a plain byte comparison.
The generated hooks manifest under `plugin/hooks/` is the exception. It is built per host
rather than vendored, which is why it is ignored in this repository: a canonical copy would
be one client's registration file shipped to all of them.
