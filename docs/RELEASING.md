# Releasing memvara

`memvara` **0.1.0** is on PyPI and `memvara@0.0.1` is on npm — both since 14 August 2026,
both put there by hand. The pipeline described below exists so that nobody has to do it
that way again.

**The mechanics are in [`release/README.md`](../release/README.md).** That is the
operating manual: what each workflow checks and refuses, what has to be configured once in
each registry's own UI, and how the two publish scripts behave. It is not repeated here.
Two documents describing one pipeline is how the first sentence of this file came to read
"nothing has been published" on a day when two things had been.

What is here is what the pipeline cannot decide: what a version number is allowed to mean,
what `1.0` will commit us to, why the project has the name it has — and what to do on the
day the pipeline itself is unavailable.

---

## How a release happens

1. Run **Version bump** from the Actions tab, on the default branch. It bumps
   `pyproject.toml` and `memvara/__init__.py`, closes `## [Unreleased]` into a dated
   version heading in `CHANGELOG.md`, runs the suite and mypy on the bumped tree, and
   opens a pull request.
2. **Review and merge that pull request.** The changelog section in it becomes the GitHub
   Release body verbatim, so reviewing the pull request *is* reviewing the release notes.
3. Tag the merge commit and push the tag.
4. **Create a GitHub Release for that tag.** That is what publishes.

Nothing before step 4 uploads anything. Merging does not publish; tagging does not
publish; `release.yml` runs on `release: published` and on a deliberate re-run dispatched
from the tag.

The split is the design: a workflow *proposes* a version and a human approves it by
merging, and the moment a number becomes permanent is a decision someone made rather than
a side effect of a workflow they triggered. `release/README.md` has the job-by-job
breakdown, the once-only registry configuration, and the reason the npm placeholder moves
on its own `0.0.x` line instead of following the Python version.

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

> **Open before 1.0** (`docs/ROADMAP.md`, Phase 8): `Recorder` and `Redactor` are
> injectable on exactly the same terms — `Memvara(telemetry=…, redactor=…)` — and are not
> currently named in that promise. Either they are in it or the promise says why not.
> Ambiguity is the one outcome to avoid, because a closed layer and a third-party backend
> both build against them.

Between releases:

| change | bump |
|---|---|
| a bug fix, no signature change | patch |
| a new method, a new extra, a new adapter | minor |
| a protocol method added, removed or re-signed | minor **before** 1.0, major after |
| a stored-format change | minor, **and** `SCHEMA_VERSION` in `memvara/store/sqlite.py` |

That last row is the one that bites. The store refuses to open a database written by a
newer schema version, which is the correct behaviour and only helps if the version was
actually bumped. Nothing in the release pipeline checks it: the workflows compare version
numbers to each other and to the registries, and none of them can see that a migration
landed without its `SCHEMA_VERSION`.

---

## Releasing by hand

The checklist below is the **fallback**, for a day the pipeline cannot run: a GitHub
Actions outage, a registry that is up but whose OIDC exchange is not, a trusted-publisher
configuration that has been invalidated, a release that has to go out from a laptop. It is
also the reasoning the workflows were built from, which is why it is kept rather than
deleted.

`release/publish_pypi.py` and `release/publish_npm.py` are the scripts for this path and
their guards are documented in [`release/README.md`](../release/README.md) — read that
before running either.

### What you give up by taking this path

The pipeline enforces four things that a careful human does not reliably enforce, and all
four failures are permanent, because **a published version can never be republished**:

- **Tag ↔ version agreement.** Nothing else in the tooling connects a git tag to a version
  — the tag is typed into GitHub's release form and the version lives in two files — so
  `release.yml`'s `guard` job compares them and refuses. A release tagged `v0.2.0` that
  ships `0.1.0` is the failure that job exists for, and by hand there is no such
  comparison.
- **Publishing only from a green commit.** `release.yml` calls `ci.yml` as a reusable
  workflow on the released commit, so a release is held to the same matrix a pull request
  is: 3.10 through 3.13 on Linux plus 3.13 on macOS and Windows, coverage gated at 100%,
  mypy, and the no-extras import job. Locally you have one interpreter, and
  `requires-python = ">=3.10"` is a promise about four.
- **The pre-release dist-tag rule.** A pre-release version must be marked as a pre-release
  on the GitHub Release or `guard` refuses, and that label is what keeps the build off
  npm's `latest` dist-tag — pre-releases take `next`. A separate check refuses a
  pre-release aimed at `latest` even if the computation is bypassed. By hand,
  `npm publish` defaults to `latest`, which is what a bare `npm install memvara` resolves
  to.
- **Idempotency across a half-finished release.** Each registry is asked separately
  whether it already has the version; one that does is skipped and reported rather than
  failing the run, so re-running finishes a partial publish instead of dead-ending. By
  hand, "PyPI succeeded and npm failed" is a state you have to reason your way out of.

Two more things the pipeline does that are worth replicating by hand: it builds from a
deleted `dist/` every time, and it publishes with no token at all — both registries use
OIDC trusted publishing, so there is no stored credential for a manual release to fall
back on and no reason to create one.

### 1. Bump the version in both places

```
pyproject.toml        version = "0.2.0"
memvara/__init__.py    __version__ = "0.2.0"
```

*The Version bump workflow does this.* Nothing in the build keeps the two equal.
`tests/test_packaging.py::test_the_version_is_the_same_string_in_both_places_that_state_it`
does, so a one-sided bump fails the suite rather than shipping a wheel whose
`memvara-mcp --version` disagrees with `pip show`.

### 2. Close out the changelog

*The Version bump workflow does this too, and refuses an empty section.* Move everything
under `## [Unreleased]` into `## [0.2.0] — YYYY-MM-DD`, and leave `[Unreleased]` empty
behind it. Keep the *Fixed* entries specific — "a backdated supersession left two live
values for a single-valued predicate" is the entry someone searches for; "bug fixes" is
not. `release.yml` reads this section back out as the release body, so by hand it is still
the release notes.

### 3. Green on every interpreter the package claims

```bash
python3 -m pytest -q
python3 -m coverage run -m pytest && python3 -m coverage report      # gated at 100%
```

*`release.yml`'s `test` job does this on the tag, across the full matrix.* Locally it is
one interpreter, so by hand the gate is a green CI run on the commit you are about to
release. Do not cut a release off a local pass alone; the matrix exists because 3.10
through 3.12 had never executed a line of this library until it did.

### 4. Build, and let the packaging tests arm themselves

```bash
rm -rf dist
python3 -m build                      # wheel and sdist
python3 -m pytest tests/test_packaging.py -q
```

*`release.yml`'s `build` job does this, from a clean `dist/`.* Three tests in that file
skip when `dist/` is empty and run once it is not — the `py.typed` marker is in the wheel,
every module in the tree is in the wheel, and the wheel's metadata version matches
`memvara.__version__`. This ordering is the whole point: build first, then test the
artifact rather than the tree.

```bash
python3 -m pip install twine
python3 -m twine check dist/*         # README renders on PyPI, metadata is well-formed
```

A description that fails to render is not rejected at upload. It is accepted and displayed
as raw text forever, on the page that is the project's front door.

### 5. Install the artifact somewhere clean and use it

*The `build` job installs the wheel into a clean venv, imports it, and does a write and a
read.* The type check below is **not** automated anywhere; it is manual-only.

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

### 6. Smoke-test the MCP server and the image

*Not automated. Neither the MCP stdio handshake nor the Docker image is exercised by
`release.yml`, on either path.* Run this against the venv from step 5, so it exercises the
artifact rather than the source tree:

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

### 7. Tag

```bash
git tag -a v0.2.0 -m "memvara 0.2.0"
git push origin v0.2.0
```

*Manual on both paths, deliberately.* Tag the commit CI went green on, not the one you are
standing on. On the automated path this is where you stop and create the GitHub Release
instead of continuing below.

### 8. TestPyPI dry run

*The pipeline does not use TestPyPI at all* — its rehearsal is the clean-venv install in
`build`, and its dress rehearsal is that every guard runs before the first upload. On the
manual path TestPyPI is the only rehearsal available, and `release/publish_pypi.py --test`
is the supported way to do it.

TestPyPI is a separate index with separate accounts and separate API tokens; register at
<https://test.pypi.org/> and mint a token scoped to the project. A `.pypirc` holding only
a `[pypi]` section will not authenticate against it and returns a bare `403` with no
message.

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

### 9. Publish, then prove it resolves

`release/publish_pypi.py` and `release/publish_npm.py`, in that order and never with
`skip-existing`. Afterwards, install the *published* version from the real index into a
fresh venv and import it from a directory that is not a checkout — `release.yml`'s
`verify` job exists because building a wheel that installs is not the same fact as an
index serving one that resolves. Then attach the sdist and wheel to the GitHub Release, so
the exact bytes stay checkable later.

---

## Why the project is called memvara

The project was called `engram` until Phase 8 prep checked. `pip download --no-deps
engram` resolves to an unrelated MIT library ("Shared research utilities for
differentiable rendering, vision-model training, and perceptual analysis") — so
`pip install engram` installed someone else's code, and `twine upload` under that name
would have been rejected. That is not a problem discoverable at upload time and worked
around; it decides what the package is called.

`engram` was a weak mark for a second reason worth recording: it is the standard
neuroscience term for a memory trace, so as a name for a memory product it is
*descriptive* — the hardest class to register and the easiest for a competitor to work
around.

`memvara` is coined, means nothing in any language, and is therefore a **fanciful mark**,
the strongest trademark class. It is what stops someone selling a competing "Memvara
Cloud"; the license does not.

**Both bare names are now claimed** — `memvara` 0.1.0 on PyPI, `memvara@0.0.1` on npm —
and that was the point of publishing them when we did. **The org was never the
reservation.** `github.com/memvara` exists and PyPI/npm organizations are registered, but
on PyPI an organization reserves no project name (the project namespace is flat and is
claimed by the first upload or a PEP 541 request) and an npm org reserves `@memvara/*`,
not the bare `memvara`. Publishing the repository was a public mention of a name that was
still takeable on two registries, and we had already lost `engram` by assuming a name was
ours — so both were claimed in the same sitting as the first public push. Keep the record:
the next time this project reserves a name, that sequence is the one to repeat.

---

## Standing rules, independent of any one release

- **Nothing from the closed side, ever.** `docs/ROADMAP.md` puts governance (PII,
  encryption, the audit chain, RBAC) and the Postgres/pgvector store in a private
  repository, and "never committed" is the actual requirement — git history is public
  forever, so a commit-then-revert is a publication. Now that the package is on PyPI, an
  sdist is a second permanent copy of whatever the tree contained.
- **A CLA before the first outside contribution.** `CONTRIBUTING.md` states the
  requirement and is honest about why; the signing process is not automated, so it is
  handled per pull request. Once an external patch lands without one, relicensing needs
  that contributor's agreement, forever.
- **`docs/DEPLOY.md` matches the artifact.** It names image tags, config blocks and file
  layouts. Re-read it against the release rather than assuming.
- **`release.yml` is a security-relevant filename.** Both trusted publishers identify the
  workflow by name, so renaming it breaks publishing and renaming another workflow *to* it
  makes that workflow indistinguishable from this one.

---

## After a release

- Publish the image under the same version tag as the package, so `memvara-mcp:0.2.0` and
  `memvara==0.2.0` are never two different builds. No workflow does this.
- Reopening `## [Unreleased]` in `CHANGELOG.md` is no longer a manual step: the Version
  bump workflow leaves it empty behind the section it closed.
- If you built locally, `python3 -m build` leaves `dist/` populated. Clear it before the
  next release, or the wheel tests in `tests/test_packaging.py` will be checking the
  previous one — they filter on `memvara-<current version>-*.whl`, so a stale wheel makes
  them silently skip rather than silently pass, but the skip is still not the answer you
  wanted.
