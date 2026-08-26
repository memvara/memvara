# Releasing memvara

PyPI has `memvara` 0.6.0 (first upload 2026-08-14, `0.1.0`). npm has `memvara` 0.1.0.

**The two release on separate tags and this document covers both.** `v*` publishes the
Python package through `release.yml`; `npm-v*` publishes the npm package through
`release-npm.yml`. They were coupled until 0.4.1, which was a PyPI release containing no
Python changes at all, cut only because npm serves the README of the *published* version
and there is no way to edit one in place. One code-less release is a curiosity; a JS
package with its own cadence would have produced one per fix.

Since `.github/workflows/release.yml` exists, most of what follows is automated: pushing
a `v*` tag runs the whole gate on the tagged commit, builds in a clean runner, checks the
artifact rather than the tree, and then stops and waits for a human. The manual procedure
is kept below as the fallback, because a release process that cannot run when Actions is
down is not a release process.

**Publishing to PyPI or npm is out of scope for any agent working in this repository.**
It is outward-facing, effectively irreversible, and belongs to whoever owns the project.
The workflow does not change that; it moves the decision to one place — approving the
`pypi` or `npm` environment — instead of removing it. An agent may open the pull request
that bumps a version and may push a branch. Approving a publish is a maintainer's. Every
manual step below stops at TestPyPI / `npm publish --dry-run`.

npm versions are independent of the Python tag. `v0.2.1` does not imply npm `0.2.1`.
Whether npm publishes at all is decided by one comparison: `check-npm` reads the version
out of `npm/memvara/package.json`, asks the registry, and skips `publish-npm` if it is
already there. A tag pushed while that file still said `0.0.1` would have been a green
release that published nothing to npm, which was the expected first run.

**Check that comparison before assuming a tag ships JavaScript.** As of 0.7.0 the file
and the registry both say `0.1.0`, so `check-npm` logs `SKIP` and `publish-npm` does not
run. Bumping `npm/memvara/package.json` is the only thing that changes that.

This matters more than a skipped job, because **npm serves the README of the published
version and there is no way to edit one in place.** A Python release that improves the
README leaves the npm page exactly as it was. Cutting an npm version is a separate
decision with its own tag, and 0.4.1 exists because it was once made late.

---

## Version policy

`0.1.0`, and the leading zero is load-bearing: **the `Store`, `Embedder` and `LLM`
protocols may change in a minor release** until `1.0`. That is what `0.x` is for, and it
is stated in `CHANGELOG.md` so a downstream implementor of `Store` knows what they signed
up for.

`1.0` means exactly one thing: **those three protocols are stable.** Not that the feature
list is complete, not that the benchmarks are in — that everything behind
`Memvara(store=, embedder=, llm=)` is a contract we will not break in a minor version.
That is the promise a hosted layer or a third-party pgvector store needs before it can
exist, and it is the only promise `1.0` should be read as making.

> `docs/ROADMAP.md`'s Phase 8 names only `Store` and `Embedder`. `CHANGELOG.md` names all
> three. The three-protocol reading is the right one — `LLM` is injectable and
> third-party-implementable on exactly the same terms — so the ROADMAP line is the one to
> correct.

Between releases:

| change | bump |
|---|---|
| a bug fix, no signature change | patch |
| a new method, a new extra, a new adapter | minor |
| a protocol method added, removed or re-signed | minor **before** 1.0, major after |
| a stored-format change | minor, **and** `SCHEMA_VERSION` in `memvara/store/sqlite.py` |

That last row is the one that bites. The store refuses to open a database written by a
newer schema version, which is the correct behaviour and only helps if the version was
actually bumped.

---

## The checklist

### 1. Bump the version in both places

```
pyproject.toml        version = "0.4.0"
memvara/__init__.py    __version__ = "0.4.0"
```

Nothing in the build keeps these equal.
`tests/test_packaging.py::test_the_version_is_the_same_string_in_both_places_that_state_it`
does, so a one-sided bump fails the suite rather than shipping a wheel whose
`memvara-mcp --version` disagrees with `pip show`.

### 2. Close out the changelog

Move everything under `## [Unreleased]` into `## [0.4.0] — YYYY-MM-DD`, and leave
`[Unreleased]` empty behind it. Keep the *Fixed* entries specific — "a backdated
supersession left two live values for a single-valued predicate" is the entry someone
searches for; "bug fixes" is not.

### 3. Tag the commit CI went green on

```bash
git tag -a v0.4.0 -m "memvara 0.4.0"
git push origin v0.4.0
```

Tag the commit CI went green on, not the one you are standing on.

That push is the whole trigger. `.github/workflows/release.yml` then runs, in order:

| job | what it does, and which manual step it replaces |
|---|---|
| `version` | Refuses unless the tag, `pyproject.toml`'s `version` and `memvara/__init__.py`'s `__version__` are the same string. First and fastest, so a one-sided bump fails in seconds rather than after the matrix — or, the failure it really exists for, not at all, leaving a wheel on PyPI whose metadata disagrees with its tag. |
| `ci` | Calls `.github/workflows/ci.yml` **on the tagged commit**: 3.10–3.13 on Linux plus macOS and Windows, coverage gated at 100%, mypy, and the no-extras import job. It calls rather than restates, so there is one matrix in this repository and it cannot drift. A tag push starts nothing else, so without this job the release would be gated on whatever CI last happened to run. |
| `build` | `python -m build`, `twine check`, then the whole suite again *after* the build, which is the only run in which the four dist-gated tests execute at all — three in `tests/test_packaging.py` and one in `tests/test_init.py`. Then it installs the wheel into a fresh venv outside the repository and checks that the dependency set is exactly memvara and numpy, that the library works, and that `reveal_type` reports `str` rather than `Any`. |
| `check-npm` | Reads `npm/memvara/package.json` and asks the registry whether that version already exists. Writes `npm_version` and `npm_exists`. Logs `SKIP` or `PUBLISH` in those words, so a skip is readable rather than an absent job. Does not compare the npm version to the Python tag. |
| `build-npm` | `npm pack` once, hashes the tarball, refuses a file list that is not `package.json`'s `files`, uploads `npm-dist`. The publish job is not allowed to pack. |
| `publish-pypi` | Uploads the artifact `build` produced — those bytes, not a rebuild — over PyPI trusted publishing. Waits for a human first; see step 4. |
| `publish-npm` | Runs on a tag push when `npm_exists == false`. Downloads `npm-dist`, checks the SHA-256, and `npm publish`es **the tarball**. No `npm pack`, no reviewer wait. A TestPyPI dispatch does not publish npm. |

The runner has no `dist/`, no second checkout and no earlier build, and that is the point
rather than a convenience. The release attempted by hand before this workflow existed ran
`twine upload dist/*` in a second checkout whose `dist/` still held `0.1.0` wheels, and
uploaded those while releasing `0.2.0`; `twine` ships the directory, not the release. PyPI
refused them only because the filenames already existed. **In a clean runner building from
a tagged commit, that mistake has nowhere to come from** — there is no stale artifact to
prefer, and no source but the tag.

### 4. Approve the publish, or do not

`publish-pypi` runs in the `pypi` GitHub Environment, which has a required reviewer, so the
run stops there and waits. This is step 9 of the old checklist — *the real publish is a
decision, not a step* — expressed as something the machinery enforces rather than something
a document asks for.

Approving is the irreversible act for PyPI. Read the `build` job's log first: it lists
the files it built, and the version in those filenames is the one about to become
permanent. npm does not wait here: a tag push that `check-npm` marked `PUBLISH` uploads
after the hash check. If `check-npm` said `SKIP`, `publish-npm` does not run. Everything
under "Before a real publish" below still applies and none of it is checked by any job.

A TestPyPI rehearsal is available and never automatic: **Actions → Release → Run workflow**,
select the tag, set the target to `testpypi`. Opt-in because **a version number on TestPyPI
is spent** the moment it is used and cannot be re-uploaded even after the release is
deleted. A rehearsal wired into every tag push would burn the release's own number on the
rehearsal index, and would then fail the *second* run of that tag — turning "re-run the
release after the upload dropped" into a red build for a reason that has nothing to do with
the release. Rehearse on an `rc` version and keep the real number clean.

---

## One-time setup, by a human, once

None of the above can publish until a trusted publisher exists on the index. Nothing in
this repository can create one and nothing should: it is the registration that says *this
repository, this workflow, this environment may upload memvara*, and it is made from an
account this repository has no access to.

**PyPI: this is done.** It was the pending-publisher form when the project did not exist;
now that `memvara` is on PyPI the same registration lives under the project's own
*Publishing* tab, and the values below are what it should read. Kept because a publisher
can be removed or edited, and because a wrong value here fails at upload with a message
that does not say which field. Check against this table rather than against memory:

| field | value |
|---|---|
| PyPI Project Name | `memvara` |
| Owner | `memvara` |
| Repository name | `memvara` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

`release.yml` is the filename, not a path: the form expects a file in `.github/workflows/`
of the repository named above. A pending publisher **does not reserve the name** — it
converts into an ordinary publisher on the first successful upload, and is invalidated if
someone else registers `memvara` first. Afterwards the same values live under the project's
own *Publishing* tab.

The environment name is optional to PyPI, and setting it is deliberate. Once registered,
PyPI checks it: an OIDC token minted by a job that is not running in the `pypi` environment
is rejected at upload. So that one field does two jobs — it is where the human approval
lives, and it narrows what the publisher will trust from "any workflow run in this
repository" to "a run that got past the environment's rules".

The rehearsal job needs its own. **TestPyPI is a separate service** with a separate account
and a separate publisher registry, so repeat the same form at
<https://test.pypi.org/manage/account/publishing/> with `testpypi` as the environment name.
Skipping this does not break a release — it breaks the rehearsal, at the upload, which is
the last place anyone thinks to look for a missing registration.

### npm trusted publisher

The package already exists on npm, so there is no chicken-and-egg: the form is on the
package's Settings → Trusted Publisher, not a pending registration.

**Preflight, before filling the form.** `version` is what the registry serves, which is
the number *before* any pending publish, not the one in this tree.

```
npm view memvara name          # memvara
npm view memvara version       # what is live now
npm view memvara description   # the bridge, not the old reservation
```

Then, on npmjs.com → package `memvara` → Trusted Publisher → GitHub Actions:

| field | value |
|---|---|
| Organization or user | `memvara` |
| Repository | `memvara` |
| Workflow filename | `release-npm.yml` |
| Environment name | `npm` |
| Allowed actions | `npm publish` |

**`release-npm.yml`, not `release.yml`** — including the extension. The registration
names one workflow file, and npm's OIDC exchange compares it against the token's
`job_workflow_ref`. Moving the publish job to a different file therefore invalidates the
registration even though the repository, the environment and the package are unchanged.

That is not hypothetical: `npm-v0.1.0` failed on exactly this, hours after `0.0.3`
published cleanly from `release.yml`. **And the error does not say so.** It is neither
`ENEEDAUTH` nor a 403:

```
npm error code E404
npm error 404 Not Found - PUT https://registry.npmjs.org/memvara - Not found
npm error 404  The requested resource 'memvara@0.1.0' could not be found
npm error 404  or you do not have permission to access it.
```

A 404 on `PUT`, for a package that plainly exists and that you can `npm view`. Read it as
*unauthorized*; npm declines to distinguish "no such package" from "not yours to write"
so that a probe cannot enumerate private names. Nothing is published when it happens, so
the recovery is to fix the registration and **re-run the failed job on the original run**
— a `workflow_dispatch` re-run will not publish, because `publish` requires
`github.event_name == 'push'`, and that is deliberate.

npm CLI ≥ 11.5.1 and Node ≥ 22.14 are required for the OIDC exchange. The job installs
that CLI; do not pin an older one to "what the runner happened to have".

Then, in GitHub, **Settings → Environments**:

* **`pypi`** — add at least one *Required reviewer*. That is the approval gate; without it
  the publish job runs unattended and step 4 above is a comment rather than a control.
  Limiting the environment's deployment branches to tags matching `v*` is worth adding.
* **`testpypi`** — for the rehearsal. A reviewer is optional here: a rehearsal is not
  irreversible, only unrepeatable for a given version.
* **`npm`** — exists so the trusted publisher can name it. **No required reviewer.**
  A tag push that finds a new `package.json` version uploads. There is no Test-npm:
  a version number on the real registry is spent.

A wrong value in any of the fields fails at upload time with a message that does not say
which one, so they are worth checking against this table rather than against memory.

**No token appears anywhere in this.** That is the point of it: the workflow has no
`password:`, no `NPM_TOKEN`, no `NODE_AUTH_TOKEN`, and nothing in the repository or in
anyone's shell history that could publish memvara. The account-wide token the manual path
needs can be revoked once the first upload has gone through the workflow.

---

## The manual fallback

**The workflow above is the release process. This is for when it cannot run** — Actions is
down, the OIDC exchange is broken, or the release is unusual enough that the workflow
refuses it and you have concluded that it is wrong and you are right. It is also the source
the workflow was written from: every check up there exists down here first.

Two things it does not give you, and they are why it is second. It builds from your working
tree rather than from the tagged commit, and it uploads whatever happens to be in `dist/`.
Those are the two halves of the failure described in step 3. `release/publish_pypi.py`
guards exactly them — it deletes and rebuilds `dist/` every run and refuses a dirty or
unpushed tree — so prefer it to a raw `twine upload`.

### Green on every interpreter the package claims

```bash
python3 -m pytest -q
python3 -m coverage run -m pytest && python3 -m coverage report      # gated at 100%
```

Locally that is one interpreter. `requires-python = ">=3.10"` is a promise about four, so
the release gate is a **green CI run on the release commit** — 3.10–3.13 on Linux plus
3.13 on macOS and Windows. Do not cut a release off a local pass alone; the matrix exists
because 3.10 through 3.12 had never executed a line of this library until it did.

### Build, and let the packaging tests arm themselves

```bash
rm -rf dist
python3 -m build                      # wheel and sdist
python3 -m pytest -q                  # not just tests/test_packaging.py — see below
```

Four tests skip while `dist/` is empty and run once it is not: three in
`tests/test_packaging.py` — the `py.typed` marker is in the wheel, every module in the
tree is in the wheel, and the wheel's metadata version matches `memvara.__version__` — and
`tests/test_init.py::test_the_skill_reaches_the_wheel`, which is the one that catches the
skill `memvara init` writes out failing to ship. This step named only the packaging file
until the release workflow was written and the fourth one was found still skipping, so run
the suite rather than a file list. This ordering is the whole point: build first, then test
the artifact rather than the tree.

```bash
python3 -m pip install twine
python3 -m twine check dist/*         # README renders on PyPI, metadata is well-formed
```

### Install the artifact somewhere clean and use it

The `offline` CI job does the import half on every push. Do the rest by hand, from a
directory that is not the repository, because a source tree on `sys.path` will happily
hide a module the wheel forgot:

```bash
python3 -m venv /tmp/rel && cd /tmp
/tmp/rel/bin/pip install dist/memvara-0.2.0-py3-none-any.whl
/tmp/rel/bin/pip list                      # must be exactly: memvara, numpy, pip
/tmp/rel/bin/python -c "
from memvara import Memvara
mem = Memvara(':memory:', user='alice')
mem.remember('user', 'lives_in', 'Berlin')
print([r.text for r in mem.search('where do they live?')])
"
```

Then check that the types survived the trip, which is the other thing only an installed
artifact can tell you:

```bash
/tmp/rel/bin/pip install mypy
cd /tmp && printf 'from memvara import Memvara\nreveal_type(Memvara(":memory:").recall("x"))\n' > t.py
/tmp/rel/bin/python -m mypy t.py           # Revealed type is "str" — not "Any"
```

`Revealed type is "Any"` together with `missing library stubs or py.typed marker` means
the marker did not ship, and every annotation in the library is invisible to every user of
that wheel. It is a two-line check for a failure that is otherwise completely silent.

### Smoke-test the MCP server and the image

Against the venv from step 5, so this exercises the artifact rather than the source tree:

```bash
MEMVARA_DB=:memory: /tmp/rel/bin/python -m memvara.server <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"rel","version":"0"}}}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
EOF

docker build -t memvara-mcp:0.2.0 .
docker run --rm -i -e MEMVARA_DB=:memory: memvara-mcp:0.2.0 < /dev/null   # expect exit 0
```

Both should be quiet. The server writes protocol messages to stdout and nothing else —
anything on stdout that is not JSON-RPC desynchronises a real client.

### Tag

```bash
git tag -a v0.2.0 -m "memvara 0.2.0"
git push origin v0.2.0
```

Tag the commit CI went green on, not the one you are standing on.

Note that this push still starts `release.yml`, even when you are down here because that
workflow is the thing that is broken. It will either fail on its own or sit waiting for an
approval; leave it waiting, and do not approve it after uploading by hand. Two paths that
both think they are publishing `0.2.0` is how one of them discovers the number is spent.

### TestPyPI dry run

TestPyPI is a separate index with separate accounts and separate API tokens; register at
<https://test.pypi.org/> and mint a token scoped to the project.

```bash
python3 -m twine upload --repository testpypi dist/*
```

Then install it back from there, in a fresh venv, with the real index still available for
dependencies — TestPyPI's numpy is a stale mirror and resolving against it proves nothing
about the real thing:

```bash
python3 -m venv /tmp/testpypi && cd /tmp
/tmp/testpypi/bin/pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  memvara==0.2.0
/tmp/testpypi/bin/python -c "import memvara; print(memvara.__version__)"
```

Two things this catches that nothing earlier does: a `long_description` that fails to
render, and a metadata field the index rejects on upload rather than on build.

**A version number on TestPyPI is spent.** It will not accept a re-upload of the same
version even after a delete, so use `0.2.0` for the real thing only after the dry run has
passed on it — or burn `0.2.0rc1` on TestPyPI and keep `0.2.0` clean.

### Stop here

The real publish is a decision, not a step. What has to be true first is below.

---

## Before a real publish

### The name was available, and that was not free

The project was called `engram` until Phase 8 prep checked. `pip download --no-deps
engram` resolves today to an unrelated MIT library ("Shared research utilities for
differentiable rendering, vision-model training, and perceptual analysis") — so
`pip install engram` installed someone else's code, and `twine upload` under that name
would have been rejected. That is not a problem discoverable at upload time and worked
around; it decides what the package is called.

`engram` was a weak mark for a second reason worth recording: it is the standard
neuroscience term for a memory trace, so as a name for a memory product it is
*descriptive* — the hardest class to register and the easiest for a competitor to work
around.

`memvara` is coined, means nothing in any language, and is therefore a **fanciful mark**,
the strongest trademark class. Both bare names were claimed on 2026-08-14: PyPI by
uploading 0.1.0, npm by publishing the placeholder at `npm/memvara` as 0.0.1.

**The org is not the name.** That was the reason the placeholder had to exist at all.
`github.com/memvara` and the PyPI/npm organizations were already registered, but a PyPI
organization does not reserve a project name — the first upload does — and an npm org
reserves `@memvara/*`, not the bare `memvara`. We already lost `engram` by assuming a
name was ours. The uploads closed that gap. An organization still does not protect a
name you have not published.

### The rest

- **Phase 4 evidence.** `docs/ROADMAP.md` gates everything on it, and a published package
  is the most public possible commitment to the claims in the README. Publishing before
  there is one external number turns "we benchmarked against something we wrote" from an
  internal caveat into a public one.
- **A CLA, before the first outside contribution.** Once an external patch lands without
  one, relicensing needs that contributor's agreement, forever. This is cheaper to do the
  week before the first release than the week after.
- **Trusted publishing, not a long-lived token.** Done on this side:
  `.github/workflows/release.yml` publishes over GitHub Actions' OIDC publisher and holds
  no credential at all. PyPI's publisher and the `pypi` / `testpypi` environments exist.
  The npm half is the same shape and is **not** done — the trusted publisher on the
  package, and the `npm` GitHub Environment, above. Until they exist `publish-npm` fails
  at the upload the first time the version is actually new, which is the right order:
  machinery should not be able to grant itself permission to publish. The `npm`
  environment now exists; the trusted publisher on the package is the outstanding half,
  and `npm/memvara/package.json` is at `0.0.2`, so "the first time the version is
  actually new" is the next tag. Register it before tagging.
- **Nothing from the closed side, ever.** `docs/ROADMAP.md` puts governance (PII,
  encryption, the audit chain, RBAC) and the Postgres/pgvector store in a private
  repository, and "never committed" is the actual requirement — git history is public
  forever, so a commit-then-revert is a publication.
- **`docs/DEPLOY.md` matches the artifact.** It names image tags, config blocks and file
  layouts. Re-read it against the release rather than assuming.

---

## After a release

- Open `## [Unreleased]` in `CHANGELOG.md` again.
- Publish the image under the same version tag as the package, so
  `memvara-mcp:0.2.0` and `memvara==0.2.0` are never two different builds.
- `python3 -m build` leaves `dist/` populated. Clear it before the next release, or the
  wheel tests in `tests/test_packaging.py` will be checking the previous one — they filter
  on `memvara-<current version>-*.whl`, so a stale wheel makes them silently skip rather
  than silently pass, but the skip is still not the answer you wanted. This is a hazard of
  building on a machine that has a history. The workflow's runner has none, which is why
  those same three tests can be trusted there without anyone remembering to clean up.
