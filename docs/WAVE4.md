# Wave 4 — parallel workstreams

Four agents, exclusive file ownership. Read this whole file before writing anything.

## What memvara is

A bitemporal memory layer for AI agents, at `/Applications/workstation/agent-memory`.
Facts are `Claim` triples carrying two independent time axes (valid time: when it was true;
transaction time: when we believed it). Contradictions resolve by a keyed lookup on
`(subject, predicate)` using declared predicate cardinality, not by top-k similarity plus
LLM adjudication. Retrieval is BM25 + vector fused by RRF. The write path avoids the model
via hash dedupe → near-dup → salience gate → rule extraction → batched LLM.

Start by reading `README.md` and `docs/INTERNALS.md`. `docs/ROADMAP.md` has the phase plan
and the open/closed licensing boundary — **read the boundary section before adding
anything**, because some work is deliberately excluded from this repository.

## House standards — these are enforced, not aspirational

**100% statement coverage** (`fail_under = 100` in `pyproject.toml`). Not negotiable. An
untested line in a memory layer is a line that only runs during an incident.

```bash
python3 -m pytest -q                                        # currently 1680 passing
python3 -m coverage run -m pytest && python3 -m coverage report
```

**Tests run offline.** No network, no API key, no sleeping except where concurrency is the
thing under test. Control time by passing explicit `datetime` values, never by patching the
clock. If your work needs a third-party SDK, it must be an *optional* import behind a lazy
`__getattr__` or a function-local import, tested with an injected fake — see
`memvara/llm/openai.py` and `tests/test_llm_openai.py` for the exact pattern to copy.

**A test states a behaviour that would be wrong if the code changed.** Name the failure:
`test_a_backdated_supersession_closes_valid_time_where_the_new_value_begins`, not
`test_retire`. The docstring says why it matters.

**Comments explain why, not what.** If a line needs a comment, the comment says what breaks
without it. Match the surrounding density and voice — read a few modules first. Em dashes
and plain prose; no bullet-point comment blocks.

**Do not overstate.** If you measure something, report the number and the conditions. If a
result flatters us, distrust it first and go find the bug — the previous wave shipped a
benchmark whose bug favoured us and it was caught by disbelief, not by tests. If you cannot
do part of the task, say so plainly in your report rather than shipping something weaker
and describing it as done.

## Exclusive file ownership

Touch **only** your files. If you need a change in someone else's, put it in your report
and I will apply it.

| workstream | owns |
|---|---|
| **A — evaluation** | `bench/locomo.py`, `bench/longmemeval.py`, `bench/evalkit.py`, `tests/test_bench_eval.py` |
| **B — framework adapters** | `memvara/integrations/**` (new package), `tests/test_integrations.py` |
| **C — privacy seams** | `memvara/redact.py`, `tests/test_redact.py`, `memvara/core.py`, `memvara/write/pipeline.py` |
| **D — deploy & release** | `Dockerfile`, `.dockerignore`, `docs/DEPLOY.md`, `docs/RELEASING.md`, `memvara/py.typed`, `tests/test_packaging.py` |

**Mine, do not edit:** `pyproject.toml`, `README.md`, `CHANGELOG.md`, `docs/*.md` except the
two D owns, `memvara/__init__.py`, `memvara/compat/**`, `memvara/server/**`, `bench/compare.py`,
`bench/baseline.py`, `bench/mem0_real.py`.

**If you need a new dependency or extra**, do not edit `pyproject.toml`. Name the exact
package and version specifier in your report. Verify it actually exists and installs first
(`pip download --no-deps <pkg>`); PyPI is reachable from this environment.

## Reporting

End with: what you built, what you measured (numbers and conditions), what you decided and
why, what you could not do, and anything you need from me or found wrong in a file you do
not own. Be specific about the last one — the previous wave's most valuable output was
agents catching each other's bugs.
