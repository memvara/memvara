# Guide: coding agents

A coding agent forgets the reasoning, not the code. The code is in git; *why* the team
chose OAuth over API keys, what it replaced, and what evidence was on the table when the
decision was made are in a Slack thread nobody can find.

This guide is the shape that works, and it is three decisions long.

The runnable version is
[`examples/coding_agent.py`](../../examples/coding_agent.py) — run it before reading on
if you would rather see the output first.

## 1. Declare your vocabulary, or nothing supersedes

The built-in predicates are a personal-assistant set. **An engineering store matches none
of them**, and an undeclared predicate takes the safe default twice over: multi-valued, so
a new value accumulates beside the old one instead of replacing it, and slow-decaying, so
this morning's deploy still ranks as fresh in two years.

Measured on a real store: 387 live claims, **95% of them using a predicate outside the
declared vocabulary** — `known_defect`, `deploy_gotcha`, `version`, `rejected` — all
invented at write time, none of them superseding anything.

Two vocabularies ship with the package:

- **`engineering`** — `deploys_to`, `current_host`, `git_state`, `build_status`,
  `version`, `endpoint`, `owner` (all single-valued), plus `depends_on`, `rejected`,
  `known_defect`, `blocked_by` (multi-valued on purpose: a project depends on many things
  and a later dependency does not make an earlier one untrue).
- **`decisions`** — `decided` and `observed`, both multi-valued and both `episodic`,
  for what an agent records about its own work.

```python
from memvara import Cardinality, Memvara, PredicateRegistry, PredicateSpec, Volatility
from memvara.schema import BUILTIN_PREDICATES, load_all_specs

registry = PredicateRegistry(
    BUILTIN_PREDICATES
    + load_all_specs("engineering,decisions")
    + (PredicateSpec(name="auth_strategy",
                     cardinality=Cardinality.ONE,       # a service authenticates one way
                     volatility=Volatility.SLOW),))

mem = Memvara("team.db", user="platform-team", registry=registry)
```

For the MCP server it is one environment variable, and the same string:

```bash
MEMVARA_PREDICATES=engineering,decisions memvara-mcp
```

A declaration outranks a guess, so adding a pack **corrects** a store that already
classified something wrongly rather than only shaping a fresh one. It is forward-only: it
changes what supersedes on the next write and retires nothing already stored.

**Packs are TOML, parsed with `tomllib`, so `load_all_specs` needs Python 3.11** and
raises with the reason on 3.10. Declaring the same predicates as `PredicateSpec`s in
Python — which is all a pack file is — works on every interpreter this package supports,
and is what [`examples/coding_agent.py`](../../examples/coding_agent.py) does. Reach for
the pack when you are on 3.11 or driving the MCP server, and for the inline form when the
code has to run anywhere.

## 2. Store the turn, then the fact, and link them

```python
jun = datetime(2026, 6, 12, tzinfo=UTC)

turn = mem.add(
    "Decision: migrate service-to-service auth from API keys to OAuth 2.0 client "
    "credentials. Manual key rotation does not work for the three third-party "
    "integrators onboarding in Q3, and a leaked key today has no expiry.",
    role="system", ts=jun)

mem.remember("checkout-service", "auth_strategy", "OAuth 2.0 client credentials",
             sources=turn.episode_ids, valid_from=jun, recorded_at=jun)

mem.remember("checkout-service", "decided",
             "migrate service-to-service auth to OAuth 2.0 client credentials",
             sources=turn.episode_ids, valid_from=jun, recorded_at=jun)
```

Three things are doing work here.

**`role="system"`.** The turn is a transcript being filed, not somebody speaking, so
nothing should be extracted from it. This matters more than it looks: the deterministic
matcher runs on every `role="user"` turn and strips quotation marks before it looks, so a
first-person sentence quoted inside a pasted log becomes a fact about whoever pasted it.

**`valid_from=jun`, not the default.** Backfilling a decision from last week without it
records a claim that was never true across its own interval, and the store then answers
historical questions wrongly with no symptom at write time. Both writes succeed; only the
dated one answers *what was the strategy in April* correctly.

**Two writes, because they are two different things.** The `auth_strategy` slot is
single-valued and *moves* — the API-keys claim's interval closes. The `decided` claim is
multi-valued and *accumulates* — a later decision does not make this one untrue, it makes
it no longer current, which is what the slot above records. Getting this backwards is the
common mistake: a `ONE` cardinality on `decided` would have every new decision silently
end the last one.

## 3. Answer the four questions

Two weeks later, somebody asks *why are we using OAuth?* That is four questions, and each
is one call:

```python
# What is it now?
[c.object for c in mem.get_all() if c.predicate == "auth_strategy"]
# ['OAuth 2.0 client credentials']

# What was it before, and when did it change?
[(c.object, c.valid_to, c.state) for c in mem.history("checkout-service", "auth_strategy")]
# [('API keys', 2026-06-12, 'ended'), ('OAuth 2.0 client credentials', None, 'live')]

# Why, and on what evidence?
p = mem.why(claim_id)
[e.text for e in p.episodes]      # the actual turn, not a paraphrase
[c.text for c in p.superseded]    # what it replaced

# What would we have said on 1 April?
[c.object for c in mem.get_all(as_of=datetime(2026, 4, 1, tzinfo=UTC))
 if c.predicate == "auth_strategy"]
# ['API keys']
```

`state == "ended"` is the answer to a fifth question nobody asks out loud: **the world
changed.** A value that had never been true reads `retired` instead, and `forget()` is
what writes that. An incident review needs to tell those apart, and nothing in the data
distinguishes them afterwards if the write got it wrong.

## Wiring it into an actual agent

**In an editor.** Install the MCP plugin and the agent gets fourteen tools; the packaged
skill tells it when to write and how to correct. See [MCP](../integrations/mcp.md).

**In a loop you are writing.** Two calls per turn:

```python
context = mem.recall(user_message, k=8)        # into the system prompt
...
mem.add(user_message, role="user")             # or remember(), for a fact you parsed
```

`recall()` returns a block framed as *reference data, not instructions*, with each claim
flattened to one line so a stored memory cannot forge prompt structure around itself.

**In LangGraph.** `MemvaraStore` is a `BaseStore`, and it loses least of the four
adapters: `put(namespace, key, value)` supplies all three parts of a triple, so changing
`city` retires exactly `city`. See [frameworks](../integrations/frameworks.md).

## Three habits worth having

1. **Close a note your own work disproved.** The commoner case is not somebody saying a
   memory is wrong — it is a note coming back in recall that this turn's work makes false.
   Nothing else notices. Correct it in the turn that falsified it, and say which of the
   three writes you used.
2. **Check the claim against the thing it describes, not against another note.** Two
   records agreeing with each other is what let the stale one stand.
3. **Store what would be embarrassing to get wrong next week.** Not the transcript.

## What does not work

**Handing prose to `add()` and expecting facts.** With no `llm=`, only a fixed set of
high-precision sentence forms is recognised — and none of them are engineering sentences.
Measured on a 64-turn support history: **64 episodes, 0 claims**. It is loud rather than
silent (`WriteReceipt.unextracted` counts every dropped turn, and the constructor warns
once) but it is real. `remember()` with a declared spec is the offline way to get the full
machine, and it is what a real integration does.

---

Previous: [RAG and memory](../concepts/rag-vs-memory.md) · Next: [MCP](../integrations/mcp.md)
