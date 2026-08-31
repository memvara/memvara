# Examples

Three runnable programs, in the order they are worth reading. Each one needs
`pip install memvara` and nothing else — no API key, no network, no database server —
and each prints its answers out of a real store rather than out of a string literal.

`tests/test_examples.py` runs all three on every CI run and asserts on what they print,
so an example here cannot drift from the library the way a snippet in a document can.

```bash
pip install memvara
git clone https://github.com/memvara/memvara && cd memvara

python3 examples/temporal_memory.py
python3 examples/coding_agent.py
python3 examples/temporal_memory_demo/demo.py
```

No `PYTHONPATH` and no clone-specific setup: the examples import `memvara` and nothing
else, so they run from any directory once the package is installed. (`bench/` and `demo/`
at the repository root do need `PYTHONPATH=.` — they import each other.)

---

## 1. [`temporal_memory.py`](temporal_memory.py) — start here

One person moves twice. Three questions, three different correct answers:

| Question | Answer |
|---|---|
| Where does Alice live now? | New York |
| Where did she live on 20 March? | London |
| Where did she live on 20 January? | Berlin |

A store that keeps one value per fact answers the first and gets the other two wrong —
not by hallucinating, but because overwriting Berlin destroyed the only record that
Berlin was ever the answer. Memvara stores an interval instead of a value, so all three
are lookups.

The example also prints the timeline behind them and the narrated form `ask()` composes.
Concepts: [bitemporal memory](../docs/concepts/bitemporal-memory.md),
[contradiction resolution](../docs/concepts/contradiction-resolution.md).

## 2. [`coding_agent.py`](coding_agent.py) — the engineering-decision case

A team migrates service-to-service auth from API keys to OAuth in June. Two weeks later
somebody asks why, and the four questions that follow are the ones a note in a vector
store cannot answer:

- **What is the strategy now?** — `get_all()`
- **What was it before, and when did it change?** — `history()`, which returns the
  superseded value with the day its interval closed
- **Why, and on what evidence?** — `why()`, which returns the actual transcript turn the
  belief came from
- **What would we have said on 1 April?** — `get_all(as_of=…)`

It is also the example that shows the one piece of setup a domain other than
personal-assistant needs: declaring `auth_strategy` single-valued, so the OAuth write
retires the API-keys claim instead of accumulating beside it.
Concepts: [provenance](../docs/concepts/provenance.md),
[guide: coding agents](../docs/guides/coding-agents.md).

## 3. [`temporal_memory_demo/`](temporal_memory_demo/) — the 90-second demo

The same story as example 1, paced for a screen recording, with the problem statement in
front of it and a close at the end. [Its README](temporal_memory_demo/README.md) has the
six beats, the exact transcript, and the recording procedure — including a
[VHS tape](temporal_memory_demo/demo.tape) that records a GIF deterministically.

---

## What is *not* here

**`demo/` at the repository root is not one of these.** It is the answer-quality
measurement harness — a 64-turn authored support history, five context-building arms, and
a blinded scoring run — and it exists to produce a number, not to teach the API. See
[`demo/README.md`](../demo/README.md) and
[`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md).

**No example calls a model.** Every one of them runs with `llm=NullLLM()` and writes
facts as triples through `remember()`. That is not a simplification for the sake of the
examples: it is how a real integration writes, and it is why the write path here does not
cost an API call. Extraction from arbitrary prose *does* need a model — see
[Installation](../docs/getting-started/installation.md#what-you-get-with-no-model) for
what the offline configuration will and will not store.

---

Next: [Quickstart](../docs/getting-started/quickstart.md) · [Why Memvara?](../docs/concepts/why-memvara.md) · [API reference](../docs/API.md)
