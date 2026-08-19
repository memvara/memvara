# Release

Publishing scripts for the packages this project ships. Run them from the repository
root: `python3 release/publish_pypi.py --test`.

## Why they live here

With the package. A release procedure that ships alongside the thing it releases is one a
contributor can read, audit and run — and it resolves the repository from its own location
rather than a configured path, so there is nothing to get wrong on a different machine.

**These are public, because this repository is**, and that is deliberate rather than
tolerated. Nothing here is a secret: credentials are read from the environment and never
written, prompted for, or logged. What is here is the reasoning behind each refusal, which
is worth more in the open than in a private repository where one person ever reads it.

When one person releasing stops being enough, the next step is a GitHub Actions workflow
using **PyPI Trusted Publishing** — OIDC, with no token existing anywhere. Every check
below moves into it unchanged; only the upload step is replaced.

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

**The first publish happened.** `memvara@0.0.1` is on the registry as a name
reservation; there is still no JavaScript client. The release *process* is now the
same tagged-commit workflow as PyPI (`.github/workflows/release.yml`): `check-npm`,
`build-npm`, `publish-npm`. This script is the fallback for when Actions cannot
run, and it still refuses the two `"private": true` applications — the console and
the marketing site — rather than letting `npm` produce a vaguer error.

An npm organisation still only reserves `@memvara/*`. The placeholder is what
claimed the bare name. [`npm/memvara/`](../npm/memvara) is that package:

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
