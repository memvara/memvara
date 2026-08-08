# Releasing engram

Nothing has been published. This is the checklist for the day something is, plus the
things that have to be true first — one of which is a hard blocker nobody had checked.

**Publishing to PyPI is out of scope for any agent working in this repository.** It is
outward-facing, effectively irreversible, and belongs to whoever owns the project. Every
step below stops at TestPyPI.

---

## Version policy

`0.1.0`, and the leading zero is load-bearing: **the `Store`, `Embedder` and `LLM`
protocols may change in a minor release** until `1.0`. That is what `0.x` is for, and it
is stated in `CHANGELOG.md` so a downstream implementor of `Store` knows what they signed
up for.

`1.0` means exactly one thing: **those three protocols are stable.** Not that the feature
list is complete, not that the benchmarks are in — that everything behind
`Engram(store=, embedder=, llm=)` is a contract we will not break in a minor version.
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
| a stored-format change | minor, **and** `SCHEMA_VERSION` in `engram/store/sqlite.py` |

That last row is the one that bites. The store refuses to open a database written by a
newer schema version, which is the correct behaviour and only helps if the version was
actually bumped.

---

## The checklist

### 1. Bump the version in both places

```
pyproject.toml        version = "0.2.0"
engram/__init__.py    __version__ = "0.2.0"
```

Nothing in the build keeps these equal.
`tests/test_packaging.py::test_the_version_is_the_same_string_in_both_places_that_state_it`
does, so a one-sided bump fails the suite rather than shipping a wheel whose
`engram-mcp --version` disagrees with `pip show`.

### 2. Close out the changelog

Move everything under `## [Unreleased]` into `## [0.2.0] — YYYY-MM-DD`, and leave
`[Unreleased]` empty behind it. Keep the *Fixed* entries specific — "a backdated
supersession left two live values for a single-valued predicate" is the entry someone
searches for; "bug fixes" is not.

### 3. Green on every interpreter the package claims

```bash
python3 -m pytest -q
python3 -m coverage run -m pytest && python3 -m coverage report      # gated at 100%
```

Locally that is one interpreter. `requires-python = ">=3.10"` is a promise about four, so
the release gate is a **green CI run on the release commit** — 3.10–3.13 on Linux plus
3.13 on macOS and Windows. Do not cut a release off a local pass alone; the matrix exists
because 3.10 through 3.12 had never executed a line of this library until it did.

### 4. Build, and let the packaging tests arm themselves

```bash
rm -rf dist
python3 -m build                      # wheel and sdist
python3 -m pytest tests/test_packaging.py -q
```

Three tests in that file skip when `dist/` is empty and run once it is not — the
`py.typed` marker is in the wheel, every module in the tree is in the wheel, and the
wheel's metadata version matches `engram.__version__`. This ordering is the whole point:
build first, then test the artifact rather than the tree.

```bash
python3 -m pip install twine
python3 -m twine check dist/*         # README renders on PyPI, metadata is well-formed
```

### 5. Install the artifact somewhere clean and use it

The `offline` CI job does the import half on every push. Do the rest by hand, from a
directory that is not the repository, because a source tree on `sys.path` will happily
hide a module the wheel forgot:

```bash
python3 -m venv /tmp/rel && cd /tmp
/tmp/rel/bin/pip install dist/engram-0.2.0-py3-none-any.whl
/tmp/rel/bin/pip list                      # must be exactly: engram, numpy, pip
/tmp/rel/bin/python -c "
from engram import Engram
mem = Engram(':memory:', user='alice')
mem.remember('user', 'lives_in', 'Berlin')
print([r.text for r in mem.search('where do they live?')])
"
```

Then check that the types survived the trip, which is the other thing only an installed
artifact can tell you:

```bash
/tmp/rel/bin/pip install mypy
cd /tmp && printf 'from engram import Engram\nreveal_type(Engram(":memory:").recall("x"))\n' > t.py
/tmp/rel/bin/python -m mypy t.py           # Revealed type is "str" — not "Any"
```

`Revealed type is "Any"` together with `missing library stubs or py.typed marker` means
the marker did not ship, and every annotation in the library is invisible to every user of
that wheel. It is a two-line check for a failure that is otherwise completely silent.

### 6. Smoke-test the MCP server and the image

Against the venv from step 5, so this exercises the artifact rather than the source tree:

```bash
ENGRAM_DB=:memory: /tmp/rel/bin/python -m engram.server <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"rel","version":"0"}}}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
EOF

docker build -t engram-mcp:0.2.0 .
docker run --rm -i -e ENGRAM_DB=:memory: engram-mcp:0.2.0 < /dev/null   # expect exit 0
```

Both should be quiet. The server writes protocol messages to stdout and nothing else —
anything on stdout that is not JSON-RPC desynchronises a real client.

### 7. Tag

```bash
git tag -a v0.2.0 -m "engram 0.2.0"
git push origin v0.2.0
```

Tag the commit CI went green on, not the one you are standing on.

### 8. TestPyPI dry run

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
  engram==0.2.0
/tmp/testpypi/bin/python -c "import engram; print(engram.__version__)"
```

Two things this catches that nothing earlier does: a `long_description` that fails to
render, and a metadata field the index rejects on upload rather than on build.

**A version number on TestPyPI is spent.** It will not accept a re-upload of the same
version even after a delete, so use `0.2.0` for the real thing only after the dry run has
passed on it — or burn `0.2.0rc1` on TestPyPI and keep `0.2.0` clean.

### 9. Stop here

The real publish is a decision, not a step. What has to be true first is below.

---

## Before a real publish

### The name on PyPI is taken

`pip download --no-deps engram` resolves today and fetches **`engram 0.1.0a1`** —
"Shared research utilities for differentiable rendering, vision-model training, and
perceptual analysis", MIT, an unrelated project by a different author. So:

- `pip install engram` currently installs someone else's library. Every "pip install
  engram" in our documentation is wrong until this is settled — `README.md` says
  `pip install -e .`, which is right; anything that says otherwise needs fixing.
- `twine upload` under that name will be rejected. This is not a step that can be
  discovered at upload time and worked around; it decides what the package is called.

Three options, none of them free. **Pick a different distribution name** — `engram-memory`
or the commercial brand `docs/ROADMAP.md` says to choose deliberately — and keep the
import name `engram`, which nothing on PyPI controls. **Ask the owner**, which is a real
conversation with a real person and may simply be no. **PEP 541 name transfer**, which
applies to abandoned projects and is unlikely to succeed against a project with a release
using current metadata.

This also sharpens the ROADMAP's own trademark note. It says "engram" is probably weak as
a mark because it is an established neuroscience term; the fact that the PyPI name went to
an unrelated project in a different field is that weakness, already realised.

### The rest

- **Phase 4 evidence.** `docs/ROADMAP.md` gates everything on it, and a published package
  is the most public possible commitment to the claims in the README. Publishing before
  there is one external number turns "we benchmarked against something we wrote" from an
  internal caveat into a public one.
- **A CLA, before the first outside contribution.** Once an external patch lands without
  one, relicensing needs that contributor's agreement, forever. This is cheaper to do the
  week before the first release than the week after.
- **Trusted publishing, not a long-lived token.** GitHub Actions' OIDC publisher means
  there is no PyPI token in the repository or in anyone's shell history to leak.
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
  `engram-mcp:0.2.0` and `engram==0.2.0` are never two different builds.
- `python3 -m build` leaves `dist/` populated. Clear it before the next release, or the
  wheel tests in `tests/test_packaging.py` will be checking the previous one — they filter
  on `engram-<current version>-*.whl`, so a stale wheel makes them silently skip rather
  than silently pass, but the skip is still not the answer you wanted.
