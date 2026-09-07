---
paths:
  - "memvara/**/*.py"
---

# Docstrings in this package run as tests

`pyproject.toml` sets `testpaths = ["tests", "memvara"]` and
`addopts = "--doctest-modules --ignore=memvara/skills"`. Every `>>>` example in a docstring
under `memvara/` is collected and executed by the suite.

Two consequences.

**A stale example fails the suite rather than rotting quietly.** This is the one corner of
documentation in this repository that defends itself. Everywhere else, nothing will catch
you.

**Rewriting the prose around an example does not exempt the example.** If you improve a
docstring, run the doctests for that module before you commit. A prose-only change is still a
change to a file the doctest collector reads.

## Practical notes

- Keep an example deterministic. A doctest that depends on the wall clock, on dictionary
  ordering, or on a network call will fail somewhere that is not your machine. This package
  pins `now=` in its own code for the same reason.
- An example that is illustration rather than something to run does not belong in a `>>>`
  block. Write it as plain text or as a fenced block outside the docstring.
- `--doctest-modules` imports every module it collects. That is why `memvara/skills/` is
  ignored: it holds data and a helper script rather than package code, and importing it would
  run something that was never meant to be imported.
- The offline default matters here. The suite runs with no API key and no network, so an
  example that reaches a model will not pass. Use `NullLLM` and `HashingEmbedder`, which are
  what a bare `Memvara(...)` already gives you.

To run just the doctests for one module:

```bash
python3 -m pytest -q --doctest-modules memvara/write/reconcile.py
```
