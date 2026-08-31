# Contributing

## Before you write code

Open an issue first for anything beyond a bug fix or a typo. This library makes narrow,
deliberate trade-offs — a permissive default, a conservative fallback, a call not made —
and most of them are documented at the place they are made. A change that looks like an
obvious improvement is often one of those trade-offs, argued the other way. The issue
saves you the work, and it is also where you find out whether the thing you want to build
is one this repository is going to accept at all (see [Scope](#scope)).

## Running it

Nothing here needs an API key, a network, a Docker daemon or a database. That is the
point of the project and it is also the development setup:

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e ".[dev]"

python3 -m pytest -q                                              # 4,064 tests
python3 -m coverage run -m pytest && python3 -m coverage report    # gated at 100%
python3 -m mypy -p memvara                                         # must be clean
```

**Pass `embedder=` at every `Memvara()` you construct in a test.** `tests/conftest.py`
fails the run otherwise, naming the file and line. `default_embedder()` returns a
sentence-transformers model as soon as that package is importable — and it is importable
if you installed `memvara[local-embed]` *or* `memvara[rerank]`, since a cross-encoder is
one — so an unpinned construction makes that test's embedding space, and the suite's
runtime, a property of what happens to be installed on the machine running it.

`HashingEmbedder(dim=512)` is exactly what `default_embedder()` returns with
sentence-transformers absent, so pinning it changes nothing about what a test measures.
Under `bench/`, use `evalkit.build_embedder("hashing")`.

A handful of doctests are exempt, because zero-config `Memvara()` is the thing they
document; `conftest.py` says which and why. The server tests used to be exempt too — they
build a `Memvara` through `build_memvara()` and had no keyword to pass it — and are not
any more, now that `MEMVARA_EMBEDDER` gives them one.

`[dev]` is pytest, pytest-asyncio, coverage and mypy — no provider SDKs. The suite runs
entirely offline against `HashingEmbedder` and `NullLLM`; a test that needs a model uses a
fake that counts its own calls. **If a test you add reaches the network, it is wrong.**

The optional extras (`anthropic`, `openai`, `local-embed`, `langchain`, `llama-index`,
`crewai`, `langgraph`) are only needed to work on those adapters, and their tests skip
without them. Install them one at a time rather than all at once, so you notice when
something has quietly become a hard dependency.

CI runs 3.10–3.13 on Linux plus 3.13 on macOS and Windows, a separate coverage job, a
mypy job, and a fourth that installs the package with **no extras** and imports every
module — a top-level `import anthropic` anywhere in the tree fails that job, which is the
whole reason it exists.

## The bar

Three hard requirements, all enforced in CI.

**1. 100% statement coverage** (`fail_under = 100` in `pyproject.toml`). This is not a
vanity number and it is worth saying why, because a blanket coverage gate is usually a bad
idea. In a memory layer, the lines that are hard to cover are the lines that only execute
during an incident: a transaction rolling back, a classifier raising, a store losing rows
mid-query, a model returning a shape that violates every field contract at once. Those are
exactly the paths nobody exercises by hand and exactly the paths whose failure is
expensive. A gate at 95% is a gate that lets you skip precisely them.

The gate is on **statements**, not branches. Twelve branch partials remain and are
verified-unreachable defensive guards — mostly `if valid_to is None or valid_to > t`,
where a live claim always satisfies the first disjunct. They stay as guards, and they are
documented as such rather than deleted to make a number look better.

If you genuinely cannot reach a line, say so in the PR rather than adding a
`# pragma: no cover`. Usually it means the line should not exist.

**2. The suite runs offline and does not sleep.** No network, no API key, no `time.sleep`
except in the handful of concurrency tests where the wall clock is the thing under test.
Time is controlled by passing explicit `datetime` values, never by patching the clock — a
bitemporal store has two time axes and a patched clock hides which one you got wrong.

**3. `mypy -p memvara` is clean.** The package ships `py.typed`, so these annotations are
what a downstream user's type checker consumes. `search()` is `@overload`ed, which is the
kind of declaration that can be wrong in a way no runtime test notices: a bad overload
type-checks a call that then fails at runtime.

## House rules

These are the ones that actually decide whether a patch reads as belonging here.

**A test names the failure it prevents.** Coverage of the lines is the floor, not the
goal. What we want from a test is a statement about behaviour that would be *wrong* if the
code changed —
`test_a_backdated_supersession_closes_valid_time_where_the_new_value_begins`, not
`test_retire`. The docstring says why it matters. The name is long on purpose: when it
fails at 2am in someone else's CI, the name is the whole bug report.

**Assert on the thing the design claims.** The claim here is that the model is rarely
consulted, so a test that does not count model calls does not test it. The fakes count
their own calls; assert on the counts.

**Comments explain why, not what.** If a line is subtle enough to need a comment, the
comment should say what goes wrong without it. `# increment the counter` is noise;
`# after redaction, or the hash is a confirmation oracle for the text just removed` is the
reason the line is where it is.

**A result that flatters us is distrusted.** This is the one that has cost the most and
saved the most. The first version of the mem0 benchmark reported a result three times
better than the true one, because its oracle had a bug that only ever hurt the other
system. It was found by asking why a good number was good, not by asking why a bad number
was bad. So: if a change makes memvara look better, the PR explains the mechanism, and the
explanation has to survive someone trying to break it. Benchmarks come with the harness,
the configuration and the losses. A number without a reproduction is not evidence.

**Correct a document that has become untrue, in the same PR.** The README, `CHANGELOG.md`,
`docs/INTERNALS.md` and `docs/DEPLOY.md` make specific, checkable claims — test counts,
timings, file layouts, which tools exist. If your change falsifies one, fix it. If you find
one already false, fix that too and say so; a correction that cuts against the project is
kept in the text rather than quietly dropped, and there are several in the README already.

## Scope

### Likely to be accepted

- Bug fixes, with a regression test that fails before the fix.
- `Store`, `Embedder`, `LLM`, `Recorder` and `Redactor` implementations — these protocols
  exist to be implemented, and a third-party backend is the intended use of them.
- Model and embedding backends, as long as they add no import to the core install.
- Framework adapters, with an honest statement of what the adapter loses. Every existing
  one has that statement in its module docstring, because "works with LangChain" without
  it is a claim that quietly means four different things.
- Performance work that comes with a measurement, including a measurement of what got
  slower.
- Documentation that corrects something untrue.
- Test coverage of a path that only runs during an incident.

### Open an issue first

Changes to bitemporal semantics, the predicate schema defaults, the retrieval scoring,
scope resolution, or anything that alters what `why()` reports. Those are the load-bearing
decisions and they are the ones most likely to look wrong until you know why they are that
way. Also: anything that adds a runtime dependency to the core. `dependencies =
["numpy>=1.24"]` is a headline claim with a test pinning it by exact equality, so adding
one is a product decision, not a packaging detail.

### Likely to be declined

- **A feature that belongs in the commercial layer.** The
  [open-core boundary](docs/OPEN-CORE.md) is real and this
  is the honest part: a Postgres/pgvector store, a REST API, a multi-tenant control plane,
  usage metering or quotas, and the governance layer (retention policy, tamper-evident
  audit chain, RBAC) are a separate paid product, and a PR that adds one here is likely to
  be declined **even if it is good**. Being told that after you have written it is a bad
  experience, so the test is written down below rather than applied privately.
- A dependency added to the core install to make something 5% faster.
- A benchmark result without the harness that produced it.
- Marketing language, badges we cannot back, or a comparison that omits where we lose.
- A `# pragma: no cover` on a line that is reachable.

### How to tell whether something is on the commercial side

Ask these in order. The first `yes` decides it.

1. **Does it only make sense when there is more than one machine?** Shared storage,
   coordination, a network protocol between processes. → commercial.
2. **Does it only make sense when the operator is not the developer?** Roles, quotas,
   billing, an admin surface, policy enforced *on* the person writing the code rather than
   *by* them. → commercial.
3. **Is it a policy, or a seam?** A `Redactor` protocol is a seam and is here. A PII
   ruleset is a policy and is not. Same for `Recorder` (here) and the dashboard that
   consumes it (not). The rule of thumb: a seam is worth nothing to a competitor and
   everything to a deployment; a policy is the opposite.
4. **Does it change what a claim is, how a contradiction resolves, what `why()` returns,
   or what `search()` finds?** → core, always, no matter who benefits. Nothing in the paid
   layer is allowed to alter the semantics of the open one, which is the property that
   makes the split trustworthy rather than a lever.

If it is still unclear after those four, it is genuinely unclear, and the issue thread is
the right place — not the PR.

**One deliberate exception, made by the maintainer, so you do not file it as a policy
violation:** `memvara/store/remote.py` and the `memvara-mcp login` device-code flow are a
thin HTTP *client* for the hosted console, living in this repo behind the optional `cloud`
extra with a lazy `httpx` import — no import in the core install, no runtime dependency
added to it. By question 1 above that is a "yes": it only makes sense when there is more
than one machine, which is normally the whole test. It stays here anyway because it is a
caller of somebody else's REST API, not the REST API itself — the server, the multi-tenant
control plane and the auth backend it talks to remain entirely in the commercial product —
and because it changes nothing about what a claim is, how a contradiction resolves, or
what `search()` finds, which is the one thing question 4 makes non-negotiable regardless
of who benefits. A new client for some *other* hosted service is not covered by this
exception and should still open an issue first.

A declined feature is not a claim on your work. This is Apache-2.0: you can carry the
patch in a fork, publish it as a separate package against the `Store` protocol, and we
will link to it from the README if it is good. That option is deliberately left open,
because a boundary that also blocks the workaround is a different and worse thing than a
boundary.

## Contributor License Agreement

Contributions require a signed CLA before they can be merged.

This is not a formality, and it is worth being straight about the reason: the project is
Apache-2.0 and is intended to stay that way, but commercial products are built on top of
it, and a CLA is what keeps the copyright position unambiguous. Without one, every
external patch is an independent veto on any future licensing decision, including ones
made in the project's interest.

The CLA grants us the right to license your contribution, including in proprietary
products. **It does not take your copyright.** You keep it, and your contribution remains
available to you and to everyone else under Apache-2.0, permanently and irrevocably —
Apache-2.0 has no take-backs, so nothing we do later can withdraw what has already
shipped.

The signing process is not yet automated. Open your PR; we will sort the CLA out with you
before merge. If you would rather not sign one, say so in the issue — a bug report with a
reproduction is valuable on its own, and we would rather write the fix ourselves than lose
the report.

## Security

Do not open a public issue for a vulnerability. [SECURITY.md](SECURITY.md) has the private
reporting flow and the classes of issue that are in scope.
