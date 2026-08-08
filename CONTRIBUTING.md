# Contributing

## Before you write code

Open an issue first for anything beyond a bug fix or a typo. This library makes narrow,
deliberate trade-offs — a permissive default, a conservative fallback, a call not made —
and most of them are documented at the place they are made. A change that looks like an
obvious improvement is often one of those trade-offs, argued the other way. The issue
saves you the work.

## The bar

Two hard requirements, both enforced in CI:

1. **100% statement coverage** (`fail_under = 100`). An untested line in a memory layer is
   a line that only runs during an incident.
2. **The suite runs offline.** No network, no API key, no sleeping except where
   concurrency is the thing under test. Time is controlled by passing explicit `datetime`
   values, never by patching the clock.

Coverage of the lines is the floor. What we actually want from a test is a statement about
behaviour that would be wrong if the code changed — `test_a_backdated_supersession_closes_
valid_time_where_the_new_value_begins` rather than `test_retire`. Name the failure, and say
in the docstring why it matters.

```bash
python3 -m pytest -q
python3 -m coverage run -m pytest && python3 -m coverage report
```

Comments explain *why*, not *what*. If a line is subtle enough to need a comment, the
comment should say what goes wrong without it.

## Contributor License Agreement

Contributions require a signed CLA before they can be merged.

This is not a formality and it is worth being straight about the reason: the project is
Apache-2.0 and is intended to stay that way, but commercial products are built on top of
it. A CLA keeps the copyright position unambiguous — without one, every external patch is
an independent veto on any future licensing decision, including ones made in the
project's interest.

The CLA grants us the right to license your contribution, including in proprietary
products. **It does not take your copyright.** You keep it, and your contribution remains
available to you and to everyone else under Apache-2.0.

The signing process is not yet automated. Open your PR; we will sort the CLA out with you
before merge.

## Licensing

The core library is Apache-2.0 and will remain so.

Some components — hosted APIs, dashboards, multi-tenant infrastructure, and governance
features — are developed as separate proprietary products. That boundary is documented in
[docs/ROADMAP.md](docs/ROADMAP.md). Contributions here are to the open core; nothing you
contribute is required for, or entitles you to, the closed components.

## Scope

Things that will likely be accepted: bug fixes with a regression test, `Store` and
`Embedder` implementations, model backends, performance work with a measurement, and
documentation that corrects something untrue.

Things to open an issue about first: changes to the bitemporal semantics, the predicate
schema defaults, the retrieval scoring, or anything that alters what `why()` reports.
Those are the load-bearing decisions, and they are the ones most likely to look wrong
until you know why they are that way.
