# Releasing, and the plugin repositories

This repository publishes two packages and feeds seven others. The Python package goes to
PyPI, an npm bridge goes to the npm registry, and the packaged skill plus the client hooks
are vendored into seven dedicated install-surface repositories that each ship memvara for one
coding agent.

The two halves share one rule, and every guard here follows from it: **a published version
can never be republished.** On PyPI that is absolute, because deleting a release does not
free the number. On npm, `unpublish` exists but is narrower and shorter-lived than people
expect, and `deprecate` leaves the version installable.

## Where the code is

- Publishing: `release/publish_pypi.py`, `release/publish_npm.py`,
  `release/rehearse_npm.py`, with the reasoning in `release/README.md`.
- The npm bridge: `npm/memvara/` — `npm/memvara/bin/memvara.js` is the entry point,
  `npm/memvara/lib/bridge.js`, `npm/memvara/lib/transport.js`, `npm/memvara/lib/creds.js` and
  `npm/memvara/lib/oauth.js` are the implementation, and `npm/memvara/test/` holds its tests.
- The plugin the agents install: `plugin/` — manifests under `plugin/.claude-plugin/`,
  `plugin/.cursor-plugin/` and `plugin/.github/`, the hosted server block in
  `plugin/.mcp.json`, the vendored skill under `plugin/skills/memvara/`, and the client-side
  hooks under `plugin/hooks/`.
- The canonical skill: `memvara/skills/memvara/`, which is the source of truth for both the
  wheel and every vendored copy.
- Fan-out: `plugin-repos.txt` names the seven repositories,
  `.github/workflows/skill-notify.yml` pings them when `memvara/skills/**` or that list
  changes, and `scripts/sync_plugin_repos.py` copies the skill into one checkout.
- The canonical instructions file for those repositories: `plugin-claude.md`. It is copied
  whole as each repository's own `CLAUDE.md`.
- CI and release automation: `.github/workflows/ci.yml`, `.github/workflows/release.yml`,
  `.github/workflows/release-npm.yml`.
- Tests: `tests/test_packaging.py`, `tests/test_npm_release.py`, `tests/test_plugin.py`,
  `tests/test_init.py`.
- Documentation: [RELEASING.md](../RELEASING.md) is the checklist, including the two places
  the version has to be bumped and the npm train.

## How the pieces fit

A release starts by bumping the version in both places, closing out `CHANGELOG.md`, and
tagging the commit that CI went green on. The publishing scripts delete and rebuild the
build directory every run, because `twine upload dist/*` ships whatever is in it and a stale
artifact is a wrong release nobody can take back. Credentials are read from the environment
and never written, prompted for, or logged.

The plugin fan-out is a different mechanism. `memvara/skills/memvara/` is the canonical
skill. Each of the seven repositories vendors a copy, pins the commit sha it copied, and
diffs its copy against that sha in CI. `plugin/skills/memvara/` in this repository is the
copy the marketplace plugin ships, and it has to stay byte-identical to the canonical tree.
`.claude/rules/packaged-skill.md` loads whenever you touch either tree and states the rule in
full.

`plugin-claude.md` is the same idea for prose. It is one file here and seven `CLAUDE.md`
files out there. Each downstream repository keeps its own runtime facts and, for the one
plugin that ships hooks, its hook rules, in a delimited block that the sync preserves; the
marker is `@@LOCAL@@` and `tests/test_plugin.py` asserts that there is exactly one of it.
Because the copy is verbatim, `plugin-claude.md` cannot rely on rule files or on any other
file in this repository: everything a plugin repository needs has to be inside it.

## Invariants and assumptions

- **A published version is final.** Every guard in `release/` exists because it is cheaper
  than the alternative, and each one carries its reason in the source.
- **The build directory is rebuilt from scratch on every run.**
- **The vendored skill is not edited downstream.** Fix it here, then let the sync carry it.
  There is exactly one sanctioned local transform, in the Claude plugin repository, and the
  drift test compensates for that single line and no other byte.
- **`plugin/hooks/` has no sanctioned transform at all.** The canonical path and the vendored
  path are the same string, so the sync is a plain copy and the gate is a plain subtree byte
  comparison.
- **The hooks manifest under `plugin/hooks/` is generated, not vendored.** Every repository
  registers a different client, so a canonical copy would be one repository's manifest
  shipped to all of them. It is ignored here so that it can never become canonical.
- **`plugin-claude.md` opens with its exact first line and carries exactly one `@@LOCAL@@`
  marker.** `tests/test_plugin.py` checks both, and also checks that the file still says
  where it is copied to, since the copy is what people will read.
- **The seven names in `plugin-repos.txt` are pinned by a test.** Adding an install surface
  means changing that list and that test together.

## Read next

[RELEASING.md](../RELEASING.md) is the procedure, and `release/README.md` explains why each
refusal in the scripts is there. `tests/test_plugin.py`'s docstrings explain what the plugin
tree has to keep true and what fails quietly when it does not.

Next: [the context index](README.md).
