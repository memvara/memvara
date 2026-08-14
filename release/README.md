# Release

**Releases are automated.** Two workflows do it, and this directory holds the Python they
run. [Automated releases](#automated-releases) is the part to read first; everything below
it documents `publish_pypi.py` and `publish_npm.py`, which are now the manual fallback and
the place most of the reasoning was first written down.

| you want to | do this |
|---|---|
| cut a release | run **Version bump** from the Actions tab, review the pull request, merge, tag, create a GitHub Release |
| publish | nothing — creating the release does it |
| publish by hand anyway | `python3 release/publish_pypi.py --test` first, and read the whole of this file |

## Why they live here

With the package. A release procedure that ships alongside the thing it releases is one a
contributor can read, audit and run — and it resolves the repository from its own location
rather than a configured path, so there is nothing to get wrong on a different machine.

**These are public, because this repository is**, and that is deliberate rather than
tolerated. Nothing here is a secret: credentials are read from the environment and never
written, prompted for, or logged. What is here is the reasoning behind each refusal, which
is worth more in the open than in a private repository where one person ever reads it.

That reasoning now applies to the workflows too, which is why the guards moved into them
rather than being written afresh: `.github/workflows/` is public for the same reason this
directory is.

## Automated releases

Two workflows, and the split between them is the design. **Version bump** proposes a
release and a human merges it; **Release** publishes what a human then tagged. Neither one
can do the other's half.

### 1. `version-bump.yml` — run from the Actions tab

Inputs: `part` (patch/minor/major), or `version` for an exact one such as `0.2.0rc1`;
`npm` to move the placeholder as well; `allow_empty_changelog` for a release with no
entries.

1. **Refuses to run off the default branch.** Not stylistic: the next version is computed
   from the version on the branch it runs from, so a stale branch produces a number that
   has already been used, and every guard downstream would pass.
2. **Refuses a version that does not go forwards, or that PyPI already has.** The registry
   question belongs here, one HTTP request before anything is written — at publish time
   the number is already in a tag, a release and a changelog heading, and the only remedy
   is to bump again.
3. **Writes the version to both places that state it** — `pyproject.toml` and
   `memvara/__init__.py` — matching an anchored pattern confined to the `[project]` table,
   and asserting exactly one hit in each file. A bump that runs, matches nothing and exits
   0 is the failure being designed out.
4. **Closes `## [Unreleased]` into `## [X.Y.Z] — YYYY-MM-DD`** and refuses if it is empty.
   That section becomes the GitHub Release body verbatim, so the changelog entry *is* the
   release notes and the pull request review is the review of them.
5. **Runs the suite and mypy on the bumped tree**, then opens a pull request.

Nothing is committed to `main`, nothing is tagged, nothing is published.

> **The pull request will show no CI checks.** GitHub does not trigger workflows on events
> raised by the built-in `GITHUB_TOKEN` — a documented guard against recursive runs, not a
> misconfiguration. The suite and mypy already ran on that exact tree in the job that
> opened it, and `release.yml` runs the full matrix on the tag before publishing. If the
> missing checkmark matters, swap in a GitHub App token
> (`actions/create-github-app-token`); do **not** swap in a personal access token, which is
> a long-lived credential of the kind this pipeline exists to stop needing.

After merging, on the default branch:

```bash
git tag -a v0.2.0 -m "memvara 0.2.0" && git push origin v0.2.0
```

then create a GitHub Release for that tag. That is the trigger.

### 2. `release.yml` — on `release: published`

| job | what it does | what it prevents |
|---|---|---|
| `guard` | tag ↔ version ↔ changelog ↔ registries | **a release tagged `v0.2.0` that ships `0.1.0`** |
| `test` | calls `ci.yml` — the full matrix, on the tag | publishing from a red commit |
| `build` | clean build, `twine check`, packaging tests, clean-venv install | a wheel that builds and cannot be installed |
| `notes` | overwrites the release body from `CHANGELOG.md` | notes written twice and drifting |
| `pypi` | Trusted Publishing, no token | a stored PyPI credential existing at all |
| `npm` | trusted publishing, provenance, no token | ditto for npm |
| `verify` | `pip install memvara==X` from the real index | a release that resolves for nobody |
| `assets` | attaches the sdist and wheel to the release | the release page being only a pointer |

`workflow_dispatch` re-runs it. Dispatch it **on the tag** — the dropdown lists tags — and
`guard` refuses a dispatch standing on a branch.

### The two packages do not share a version number

`memvara` on PyPI is the library. `memvara` on npm is a name reservation that exports
`{implemented: false}` and contains no client. So:

- an ordinary release bumps the Python version and **leaves npm alone**;
- the placeholder stays on a **`0.0.x`** line, which is reserved for "there is no
  implementation", and moves only via the `npm` input;
- `release.yml` publishes to npm only when that number changes, so a Python release does
  not touch the registry at all.

Three reasons, spelled out at length in [`versions.py`](versions.py). A matching number
would state that `npm install memvara@0.4.0` and `pip install memvara==0.4.0` are the same
software — one of them is an empty object. Every npm version is permanent and would be
spent on nothing. And PEP 440 and semver disagree exactly where releases are most
delicate: `0.2.0rc1` and `0.2.0-rc.1` are one intent spelled two ways, so a coupled
pipeline needs a translation layer exercised only on pre-releases.

A real JavaScript client, if one is written, starts its own semver line at `0.1.0`, and the
coupling question gets asked again then — with an actual client to reason about.
`tests/test_release.py` asserts the `0.0.x` rule, so changing it is a decision rather than
a drift.

### What has to be configured once, in each registry's UI

Neither workflow can do this, and both fail loudly until it is done. **No secret is
created by either procedure** — that is the point.

**PyPI.** pypi.org → Your projects → **memvara** → Manage → **Publishing** → *Add a new
publisher* → GitHub Actions:

| field | value |
|---|---|
| Repository owner | `memvara` |
| Repository name | `memvara` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The environment is optional and worth setting: it is part of the publisher's identity, so
with it configured no other workflow in the repository can mint a token for this project —
and it is where a required-reviewer rule goes if publishing should need a second pair of
eyes. Create a matching environment under repository Settings → Environments.

**npm.** npmjs.com → Packages → **memvara** → **Settings** → **Trusted publishing** →
GitHub Actions:

| field | value |
|---|---|
| Organization or user | `memvara` |
| Repository | `memvara` |
| Workflow filename | `release.yml` (case-sensitive, extension included) |
| Environment | `npm` |
| Allowed actions | `npm publish` |

Then, on the same settings page, **Publishing access → "Require two-factor authentication
and disallow tokens"**. That closes token publishing entirely while OIDC keeps working —
which is the whole return on this pipeline, and the reason to do it in the same sitting.

Two things that will bite otherwise:

- **`release.yml` is a security-relevant filename.** Renaming it invalidates both
  publishers; renaming some other workflow *to* it makes that workflow indistinguishable
  from this one.
- **npm trusted publishing cannot make a package that does not exist.** Both the UI and
  `npm trust` require the package to be there already. `memvara@0.0.1` was published by
  hand, so this is settled for this project — but a future `@memvara/*` package needs one
  bootstrap publish before it can be configured.

### On the tokens this replaces

npm **permanently revoked all classic tokens** in December 2025 — they cannot authenticate,
be recreated or be recovered — and granular write tokens now expire in at most 90 days. So
the leaked npm token this project has been carrying is already dead rather than merely
deprecated, and there is no maintenance-free token option left to choose instead of OIDC.
The account-wide PyPI token in `~/.pypirc` is the one still worth deleting by hand: once
the publisher above is configured, nothing needs it.

## The rule both scripts are built around

**A published version can never be republished.** On PyPI that is absolute — deleting a
release does not free the number. On npm, `unpublish` exists but is 72 hours and narrower
than people expect; `deprecate` is what actually applies afterwards, and it leaves the
version installable.

So every guard in these scripts is there because it is cheaper than the alternative, and
each one has a reason written next to it in the source. The one worth repeating: **`dist/`
is deleted and rebuilt every run.** `twine upload dist/*` ships whatever is in that
directory, and on the day this was written that was an sdist built four days and roughly
forty commits earlier. It would have uploaded cleanly, and `0.1.0` would have been spent
on it.

## PyPI

```bash
read -rs TWINE_PASSWORD && export TWINE_PASSWORD && export TWINE_USERNAME=__token__

python3 release/publish_pypi.py --test     # TestPyPI first, always
python3 release/publish_pypi.py            # the real thing
```

It refuses to run on an uncommitted tree, on an unpushed commit, without a credential *for
the service being uploaded to*, or when the version already exists on the index. It then
builds from clean, runs
`twine check` (which renders the README the way PyPI will — a description that fails to
render is *accepted* and displayed as raw text forever), asks you to type the version to
confirm, uploads, and installs the result into a throwaway virtualenv to prove that what
a user gets actually imports.

**First upload only:** the token must be scoped to the *entire account*, because PyPI
cannot scope a token to a project that does not exist yet. Replace it with a
project-scoped token straight afterwards.

**TestPyPI is a separate service** with its own account, its own token and its own
namespace. A `.pypirc` holding only a `[pypi]` section will not authenticate `--test`, and
what comes back is a bare `403 Forbidden` with **no message at all** — after the build,
the checks and half a megabyte of upload have already happened.

The script now refuses that combination up front, in under a second, and names the fix. It
reads the issuing domain out of the token's macaroon to do it, so a pypi.org token aimed at
TestPyPI is caught even when the section name looks right. The token itself is never
printed, logged, or sent anywhere; the only value derived from it is a domain name.

This is worth stating plainly, because the prose above already said "separate account,
separate token" before any of it happened, and that made no difference whatsoever: the
guard passed, so nobody read the warning. **A rule the code does not enforce is not in
effect**, however clearly it is written down.

### Sequencing that matters for the first release

1. **Claim the name before making the core repo public.** Pushing a public repo announces
   the name before it is secured.
2. This does **not** wait on the PyPI organisation request. An organisation reserves no
   project name — only the first upload does. Publish from a personal account and transfer
   the project into the organisation later.
3. `pyproject.toml`'s `Homepage` points at `github.com/memvara/memvara`, which is private
   until step 1 is done. The PyPI page will link to a 404 for as long as that gap lasts.

## npm

```bash
python3 release/publish_npm.py --package PATH --dry-run
python3 release/publish_npm.py --package PATH
```

**npm is a name-reservation question here, not a release process.** There is no JavaScript
client. The two other `package.json` files in this project are applications marked
`"private": true` — the console and the marketing site — and neither belongs in a
registry; the script refuses them by name rather than letting `npm` produce a vaguer error.

`memvara` on npm is unclaimed, and an npm organisation reserves only `@memvara/*`, exactly
as a PyPI organisation reserves no project name. Claiming the bare name requires publishing
something real, so [`npm/memvara/`](../npm/memvara) is a placeholder that names the
project, links to it, and does nothing:

```bash
python3 release/publish_npm.py --package npm/memvara --dry-run
```

It exports `{implemented: false, notice, python, homepage}`. It has no side effects and
**does not throw on import** — a placeholder that throws breaks a bundler's module graph
and turns "you installed the wrong thing" into a build failure several layers from its
cause. For TypeScript callers the protection is that the type has four properties and no
methods, so using it as a client — `memvara.recall(...)` — is a compile error (TS2339).
`implemented` is the literal `false` rather than `boolean` as a secondary signal.

An earlier version of this file claimed that `if (memvara.implemented)` errors because the
branch narrows to `never`. It does not: TypeScript does not report type-dead branches, and
`allowUnreachableCode: false` does not change it, being a syntactic check. Checked against
tsc 5 after publishing, which is the wrong order — the claim was plausible, which is
exactly the kind that survives unverified.

`npm/` is excluded from the Python sdist (`[tool.hatch.build.targets.sdist]` in
`pyproject.toml`). Hatch ships every top-level directory by default, so without that the
placeholder would have been bundled inside `pip install memvara`.

When a JS client for the REST API exists, point `--package` at it and nothing changes.

**`--dry-run` needs no npm account.** It runs every check, packs the tarball and calls
`npm publish --dry-run`, all anonymously; only a real publish needs `npm login` or
`NPM_TOKEN`. A rehearsal you cannot run until you are already set up is a rehearsal
nobody does.

`--access public` is passed always: a scoped package publishes as restricted by default,
and a restricted package on a free account is refused with a billing error rather than a
permissions one.

## A note on the network check

Both scripts query the registry with `requests` rather than `urllib`. A python.org build
on macOS does not use the system trust store, so `urllib` raises
`CERTIFICATE_VERIFY_FAILED` against ordinary HTTPS hosts while `curl` on the same machine
succeeds. That was found the first time the PyPI script ran, and it mattered: the version
check would have been unrunnable exactly where it is needed, and **a guard that has to be
bypassed to get work done stops being a guard**, because the bypass becomes the habit.
