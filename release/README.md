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

It refuses to run on an uncommitted tree, on an unpushed commit, without credentials, or
when the version already exists on the index. It then builds from clean, runs
`twine check` (which renders the README the way PyPI will — a description that fails to
render is *accepted* and displayed as raw text forever), asks you to type the version to
confirm, uploads, and installs the result into a throwaway virtualenv to prove that what
a user gets actually imports.

**First upload only:** the token must be scoped to the *entire account*, because PyPI
cannot scope a token to a project that does not exist yet. Replace it with a
project-scoped token straight afterwards.

**TestPyPI is a separate service** with its own account and its own token. A PyPI token
fails against it with nothing more helpful than "invalid credentials".

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

**There is nothing to publish yet, and that is the first thing to know.** Both
`package.json` files in this project are applications marked `"private": true` — the
console and the marketing site — and neither belongs in a registry. The script refuses
them by name rather than letting `npm` produce a vaguer error.

So npm is currently a *name reservation* question, not a release process. `memvara` is
unclaimed and an npm organisation reserves only `@memvara/*`, exactly as a PyPI
organisation reserves no project name. Claiming the bare name requires publishing
something real. A minimal placeholder that names the project and links to it is a
legitimate reservation for a project that genuinely exists — npm's policy is against
claiming names you have no relationship to — but the first publish is public and
effectively permanent, so this script will not invent one.

When a JS client for the REST API exists, point `--package` at it and nothing changes.

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
