# Upgrading

What breaks, and what to do about it. `CHANGELOG.md` is the full record; this file is
the short list of things that will not announce themselves.

Entries are newest first, and each one says how you find your own instances of it.

---

## `valid_from` now carries the time a turn stated, not the time it was said

### What changed

Both write paths used to set `valid_from` to the episode's timestamp. They now resolve any
temporal expression the turn carried — "yesterday", "last month", "three weeks ago" — and
store that instead, together with a new `Claim.temporal_precision` recording how coarse it
was. A turn stating no time is unchanged: `valid_from` is still the episode's timestamp and
the precision is `None`.

`Claim.amount` and `Claim.unit` are new and default to `None`. `SCHEMA_VERSION` moves from
8 to 9, which adds three nullable columns; an older file upgrades in place on open.

### Who this changes, and in which direction

**Your existing claims are not touched.** Nothing backfills an event time, because it cannot
be recovered without re-extraction and inventing one would be forging history. Every claim
written before this upgrade keeps `valid_from` meaning the conversation's timestamp, and
`temporal_precision IS NULL` is the honest record of that.

**A build older than this cannot open a file this one has upgraded.** That is the usual
one-way schema door and the store refuses rather than corrupting. Take a copy first if you
may need to roll back.

**Supersession outcomes change for claims written after the upgrade.** Two boundaries now
order confidently only when their intervals do not overlap, and overlapping ones fall back
to `recorded_at`. Two claims with no precision compare exactly as they did before, so a
store that never records an event time sees no difference at all.

**Ranking changes with it.** `recency_factor` measures age from `valid_from`, so a claim
whose turn stated a past time is now scored as that old rather than as new. That is the
documented meaning of the field, and it is a real change in what comes back first for a
store that starts recording event times.

**If you need the old behaviour**, do not set event times: nothing resolves an expression
the extractor did not report, and the fast path only resolves tails it already stripped. To
be certain, run with `MEMVARA_LLM=none` and no `events` pack; a store that records no
precisions behaves as it did.

---

## `MEMVARA_MODE=cloud` now starts a server, and refuses two variables it used to accept

### What changed

A cloud-mode `memvara-mcp` used to exit 2 at startup with "cannot start a server yet". It
starts. It builds a `RemoteMemvara` — a client of the `/v1` facade — and serves the same
fourteen tools from a hosted deployment. Nothing about how the credential is found has
changed: `MEMVARA_API_KEY`, then the file `memvara-mcp login` writes.

The engine is still never run against a remote store, which is what the refusal protected.
`docs/OPEN-CORE.md` records why that is a decision rather than a gap.

### Who this changes, and in which direction

**If you configured cloud mode and were refused, delete nothing and try again.** The same
environment block now works, provided `httpx` is installed: `pip install "memvara[cloud]"`.

**If your cloud environment also sets `MEMVARA_LLM` or `MEMVARA_EMBEDDER`, the server now
refuses to start.** Unset them. Extraction and embedding run inside the deployment, so this
process would read the value and never use it — and the refusal is deliberately louder than
ignoring it, because an operator who sets `MEMVARA_LLM=anthropic` and sees a server start
has been told their writes are being extracted by a model that was never loaded. Only a
non-default value is refused; an unset variable is fine. `memory_stats` reports the
deployment's own extractor.

**If your hosted API key is read-only, the server now hides its write tools.** That is the
fix rather than a regression: it used to list them and let the deployment refuse them
mid-conversation as a 403. `MEMVARA_READ_ONLY` and the credential are OR-ed — a server
configured read-only stays read-only whatever the token allows.

**If you called `config.cloud_gap()`, `config._ENGINE_NEEDS` or `config._CLOUD_NOT_WIRED`,
they are gone.** Two were private. `cloud_gap()` was public and its whole purpose was to
answer "can cloud mode start", which is now "is `httpx` importable" — `memvara-mcp init`
asks exactly that, and `memvara.remote.client.install_hint()` is the message.

**If you type-annotated against `ToolContext.memory`, it is now `MemoryAPI`.** A protocol in
`memvara/server/memory_api.py`, satisfied by `ScopedMemvara` and `ScopedRemoteMemvara`
both. A parameter annotated `ScopedMemvara` still accepts what it always did; one that
*returns* `ToolContext.memory` as a `ScopedMemvara` no longer type-checks.

**If you implemented `MemoryAPI` yourself, `search`, `recall` and `ask` now take
`anchored: bool = False`.** The `memory_search`, `memory_recall` and `memory_ask` handlers
pass it on every call, so an implementation written against the previous protocol still
satisfies `isinstance` (a protocol checks names) and then raises `TypeError: unexpected
keyword argument 'anchored'` on the first call from any of the three tools. Accept the
keyword; honouring it means returning only results the query names an entity of, which
`memvara/retrieve/anchor.py` defines, and ignoring it is a documented lie to the model
that set it, so raise if you cannot honour it.

### How to find your own instances

```bash
grep -rn "MEMVARA_LLM\|MEMVARA_EMBEDDER" --include="*.json" ~/.claude .  # cloud env blocks
grep -rn "cloud_gap\|_CLOUD_NOT_WIRED\|_ENGINE_NEEDS" .
```

---

## Every recalled note that nobody stated now ends " (inferred)"

### What changed

`recall()` marks a note it did not get from the caller asserting it. A row is marked when
its `derivation` is anything other than `USER`, or when its `extractor` is anything other
than `api`. The marker is the literal string in `Memvara.RECALL_INFERRED`.

### Who this changes, and in which direction

**Anything that parses or asserts on `recall()` output.** The text of a marked row is no
longer the claim's text. If you compare a rendered line against a known string, strip the
suffix first — `line.removesuffix(Memvara.RECALL_INFERRED)` — rather than matching on
`endswith`.

**Anything budgeting the block.** A marked row costs about three more tokens. `budget=` is
still honoured exactly, because the fit loop measures the assembled block and the marker is
inside it — but a budget that used to hold eight notes may now hold seven. Two tests in
this repository had to raise their budgets for that reason.

**Stores built by extraction, most of all.** Nothing is marked on a store of facts a caller
asserted through `remember()`. On a store built by `add()`, or by a capture hook naming
itself in `extractor`, every row is marked. `demo/`'s corpus is the second kind, and its
prompt grew from 430 to 440 tokens.

### Which surfaces this reaches, and which it does not

Stated as surfaces rather than as a function name, because "`recall()` marks rows" does not
answer the question a client actually has. A hosted client never calls the method.

| surface | marks, and how |
|---|---|
| `Memvara.recall()` | **yes** — ` (inferred)` after the claim's text |
| the `memory_recall` MCP tool | **yes** — it returns `recall()`'s output verbatim |
| `memory_standing` | **yes** — ` inferred` INSIDE the bracket |
| `memory_since` | **yes** — both halves, added and gone |
| every other MCP surface that renders claims | no |

The last two arrived **after `0.8.0`**, not with it. On `0.8.0` exactly, only the two
`recall()` rows mark, which is what the first version of this table said and why it is
worth reading the table against the server you are actually running rather than against
this file's newest entry.

**The two spellings are deliberate and a parser needs both.** `recall()` rows carry no
metadata, so a suffix cannot be confused with anything; `_delta_lines` rows put metadata
first and the untrusted span last precisely so nothing trusted follows text a claim could
impersonate, so its marker is a bracket field. A consumer matching only `" (inferred)"`
will not see a marked standing row.

`memory_recall` is the one that surprises people, since a client reading a library-API note
reasonably concludes it is not about them — and a per-prompt hook calling the tool over the
hosted transport gets marked rows either way.

### If you parse `memory_standing` or `memory_since`, read this before upgrading the server

The bracket gained a field, and it is **not** a fixed-arity structure:

```
+ [id=cl_1 procedural live] user prefers tabs                      three tokens
+ [id=cl_2 procedural live inferred] user prefers spaces           four
- [id=cl_3 semantic ended 2026-08-26 14:09Z inferred] user …       six
```

Six, not five: `_stamp` renders `2026-08-26 14:09Z`, so **the instant itself contains a
space**. Even the metadata is not one token per field, which is the sharpest reason not to
read this bracket by counting.

`_state` has appended an instant for `ended` and `retired` since `7985c24` (2026-08-09),
so a consumer pinning a count was already wrong for those rows on any build after that
date. Read the bracket as a **set of tokens**.

This matters more than most format changes because of how such a parser usually fails. A
regex that pins three fields does not raise on a fourth — it fails to match, and a reader
that skips what it cannot match drops those rows **silently** while the block still looks
whole and its own count line agrees. The rows lost are exactly the derived ones, which are
the rows the marker exists to point at.

Measured on a real store while this shipped: a client pinning three fields rendered 31 of
37 standing rows and dropped the 6 machine-derived ones, reporting no error.

So the order is: **upgrade the clients that parse these rows, then the server.** For
`claude-memvara` that is `0.2.2` or later — and a published tag is not the check, since it
updates nobody. Read the version off the installed copy:

```bash
grep '"version"' ~/.claude/plugins/marketplaces/claude-memvara/plugin/.claude-plugin/plugin.json
```

### How to find your own instances

```python
from memvara import Derivation

sum(1 for c in mem.store.iter_claims(states=("live",))
    if c.derivation is not Derivation.USER or c.extractor not in ("", "api"))
```

That count is how many of your rows will gain the suffix. If it is zero, this entry does
not reach you.

---

## `remember(memory_type=...)` re-files a fact this store already holds

### What changed

Re-asserting a triple that already exists is a re-observation and reinforces the record,
which has not changed. What has changed is that an **asserted** `memory_type` now moves the
stored claim to that type, stamps `meta["retyped_from"]`, and reports a `Retype` on
`WriteReceipt.retyped`. It used to be dropped, so the claim kept its old type and gained
confidence — correcting a filing made the wrong filing more strongly believed.

### Who this changes, and in which direction

**Callers that pass `memory_type` when re-asserting known facts.** Their claims will move,
where before nothing happened. That is the point of the change, and it is worth knowing
before it surprises you: the type decides which population a claim is in, and
`memory_standing` returns the `procedural` one, which most clients inject at the top of
every session. A claim entering or leaving `procedural` changes what every later
conversation opens with.

**Nobody who omits it.** `remember()` with no `memory_type` takes the predicate's declared
default, which is nobody's opinion, and re-files nothing. Extraction never reaches this
path. That asymmetry is deliberate: agents re-assert known facts constantly without a view
about filing, and treating any difference as a correction would let the last writer win
when the last writer is usually the one who said nothing.

**`derivation` is untouched.** Only the filing moved. Where the fact came from is unchanged,
so an audit of provenance is unaffected.

### The hazard if your `memory_type` comes from a table

Worth stating because it is invisible at the call site. A writer that derives the type from
a fixed predicate-to-type map — rather than choosing it per write — turns **any edit to that
map into a bulk re-filing**, applied one claim at a time as each predicate is next
mentioned. No write looks like a re-filing; claims simply migrate between populations over
days. If any of the moved predicates are `procedural`, they enter or leave what
`memory_standing` returns, and so what every later session is given.

Nothing here prevents that, and it is the right behaviour once the map is the intended
source of truth. But change such a map deliberately, not incidentally.

### How to find your own instances

Search your own code for `remember(` calls that pass `memory_type` and are not creating a
new fact. Those are the writes whose behaviour changed. Afterwards,
`memory_why` on any moved claim shows `retyped_from` in its meta.

---

## `ask()` says more about a slot it cannot render as a simple list

### What changed

Two rendering corrections, both to `Answer.text`. A slot holding more than one value no
longer prints an unscoped provenance line — the dates now name the value they belong to.
And a **single-valued** slot holding two live values, which is what `AUTHORITY_SHARE`
leaves behind when it refuses a displacement, now says which value holds the slot instead
of joining both with a comma as though they were simultaneously true.

### Who this changes, and in which direction

**Anything asserting on `ask().text`.** The strings changed. `ask()` shipped in `0.7.0`, so
this reaches only code written against that one release.

Nothing about the stored data changed, and `why()` and `history()` were correct throughout.

---

## A write worth less than half of what it would replace no longer replaces it

### What changed

Contradiction resolution reads `confidence`. A candidate closes a live claim only if it is
worth at least half of it (`write.reconcile.AUTHORITY_SHARE`); below that the incumbent
stays live, the candidate is stored beside it, and the write reports a `Dispute` on
`WriteReceipt.disputed`. `remember()` also refuses `valid_to` at or before `valid_from`,
which used to store a claim no query returns.

### Who this changes, and in which direction

**Nobody writing at the confidences the shipped paths produce.** Those are 1.00
(`remember()` and `memory_remember`), 0.95 (the fast path), 0.70 (an extraction whose
model returned no figure) and 0.50 (one that ignored the schema). Every one of them clears
half of every other, so ordinary traffic supersedes exactly as before.

**Deployments that pass a low `confidence` deliberately** — an extraction model that
scores implied facts down, or an importer marking uncertain rows. Those writes used to win
and now do not. Two live values in a single-valued slot is the visible cost, and it is the
recoverable direction: keeping two competing facts degrades ranking, and ending a true one
destroys information. Retrieval already prefers the more confident of the two.

**`supersede()`, `forget()` and `delete()` are unchanged.** Each closes a claim the
caller named, before the reconciler weighs anything, so the rule does not reach them —
it arbitrates an inference the write path drew, and naming the row to close is not one.
Worth knowing before auditing a store on the strength of this entry.

**It catches a marked guess, not the extraction tier.** 0.70 is the default for an
extraction whose model gave no figure, and `0.70 >= 0.5 * 1.00` — so a mined paraphrase
still closes a fact a person stated at 1.00. That is deliberate: blocking it would stop
the store learning from conversation. If what you are worried about is a paraphrase
outranking something the user said outright, that is a *ranking* question and lives in
issue #62, not here.

**What you were losing before is worse than what you lose now.** The displaced claim was
stamped `ended`, which in this library asserts that the world changed. A guess that
collided with a known fact recorded a world event that never happened, on the axis whose
whole purpose is answering "what do we now believe was true then".

### How you find out it applies to you

`receipt.disputed` is non-empty, `repr(receipt)` shows `disputed=N`, the `write.disputed`
counter climbs, and `memory_remember` prints a note naming both values and both
confidences. A series that climbs from zero on this upgrade is not a new problem — it is
how often the old behaviour was firing.

For history already written this way, the pairs are gone: an `ended` claim displaced by a
guess is indistinguishable from one displaced by a fact, which is exactly why this was
worth fixing rather than migrating. What you can find is the population worth re-reading:

```python
for c in mem.get_all(states=["ended"]):
    successor = mem.store.get_claim(c.invalidated_by) if c.invalidated_by else None
    if successor is not None and successor.confidence < 0.5 * c.confidence:
        print(c.id, c.object, c.confidence, "→", successor.object, successor.confidence)
```

### If you want the old behaviour

There is no flag. The old behaviour recorded a reason it had not established.

---

## `remember()` raises on `true_since`, where it used to store it as metadata

### What changed

`memory_remember` calls the valid interval `true_since`/`true_until`; `Memvara.remember`
calls it `valid_from`/`valid_to`. Passing the tool's spelling to the method used to land
in `**meta`. It is now a `TypeError` naming the keyword it meant. The same call also
rejects any `meta` value `json.dumps` cannot serialize.

### Who this changes, and in which direction

**Anyone whose code passed `true_since=` a string and believed the interval was set.**
This is the case worth finding: it never raised. The claim was stored dated from the
instant of the write, with `true_since` filed beside it in `Claim.meta` — so a store
backfilled that way holds facts whose valid time is the import, not the history, and
every `valid_at` query about the period they cover answers nothing.

**Anyone who passed a `datetime` there** already had a hard failure, four frames down in
`put_claim`. Same for a non-JSON `meta` value. Those calls now fail at the call site with
the key named; nothing that used to succeed stops succeeding.

### How you find out it applies to you

Search your store for claims carrying the annotation, and read the gap between the two
axes:

```python
for c in mem.get_all(states=["live", "ended", "retired"]):
    if "true_since" in c.meta or "true_until" in c.meta:
        print(c.id, c.subject, c.predicate, c.object, "|",
              "meant", c.meta.get("true_since"), "| stored", c.valid_from)
```

Each one is a claim whose valid time is its import instant. Rewrite it with
`valid_from=`, which is the honest backfill this library documents.

### If you want the old behaviour

There is no flag. The old behaviour was the argument being dropped, and the argument
named an instant.

---

## Model-extracted claims with no tie to their cited turn are now rejected

### What changed

`WritePipeline` gained `reject_ungrounded`, defaulting to `"auto"`: a claim the
extraction model proposes is refused when its object shares not one content word with
the episode it cites as its source **and** the configured embedder finds no semantic
tie either (best chunk-cosine below 0.40). Refusals are counted on
`WriteReceipt.ungrounded` and reported in `memory_add`'s receipt as
`note: N proposed claim(s) had no support in the turn they cited as their source`.

### Who this changes, and in which direction

**Nobody running the shipped defaults.** The default `NullLLM` proposes no claims, so
there is nothing to filter. `remember()` and the deterministic fast path are never
checked at all — nothing a caller asserts directly is affected.

**Deployments with an extraction model configured** (`MEMVARA_LLM=anthropic`, or an
`llm=` passed in). Claims the model invents out of whole cloth — measured at 18–36% of
usable output for 4B-class local models, typically a placeholder like
`works_at: "Acme"` — no longer reach the store. Before this, such a claim did not sit
harmlessly beside the truth: on a ONE-cardinality predicate it superseded and *ended*
the true fact in the slot.

**The direction that can cost you:** a genuine claim whose object is a paraphrase
sharing zero vocabulary with its source, on a deployment whose embedder is the
lexical `HashingEmbedder` (where the semantic rescue cannot fire). That combination
was observed zero times in the 144 real claims measured, but it is possible, and it
costs the one claim — the episode itself is already stored and retrievable.

### How you find out it applies to you

`receipt.ungrounded` is non-zero, `repr(receipt)` shows `ungrounded=N`, and the
`memory_add` note above appears on the MCP transport. If the rescue's embedder fails,
the pipeline warns once (`RuntimeWarning`, "embedding failed during the grounding
rescue") and keeps the claims it could not check.

### If you want the old behaviour

`Memvara(write_reject_ungrounded=False, ...)` restores it exactly. `True` is a third
mode: the lexical check alone, no embedding rescue, for callers who have measured
their extractor and want the hard line.

---

## `include_episodes` now requires a real boolean, where a string used to be accepted

### What changed

`memory_recall` declares `include_episodes` as `boolean`, and the tool-call validator had
no branch for that type. Two things followed, and only the second one can break you.

A caller sending the argument the way the schema asks — `true` or `false` — got an
unhandled `KeyError: 'boolean'` raised out of the error path itself. That never worked, so
nothing depended on it.

A caller sending the **string** `"true"` was accepted, because a boolean fell through to
the validator's "must be a string" check and passed it. The handler then read the flag
through `bool(...)`, where every non-empty string is truthy — so `"true"` turned episodes
on, and so did `"false"`. Both are now rejected with a normal tool error.

### Who this changes, and in which direction

**Anyone whose client stringifies arguments.** Since the correctly-typed call raised, a
caller who was successfully getting episodes back was necessarily sending a string, and
that call now returns an error instead of results.

**Anyone sending `"false"` and expecting it to mean false.** That call was turning
episodes on. It now fails loudly rather than doing the opposite of what it says.

Callers sending real JSON booleans are unaffected, except that the call now works.

### How you find out it applies to you

The rejection names the argument and what arrived:

```
memory_recall.include_episodes must be a boolean, got a string ('true')
```

It arrives as a tool result with `isError: true`, the way every other argument rejection
does, so a model reading it can correct itself on the next turn.

### If you want the old behaviour

There is none to restore: one half raised `KeyError` and the other read `"false"` as true.
Send `true` or `false` as JSON booleans, not as strings.

---

## The graph leg stops running on a store where nothing chains

### What changed

If you set `w_graph > 0` (or `read_w_graph`), the leg now checks the store before it walks
and does not run when no live claim's object is another live claim's subject. On a store
with joins nothing changes: measured on 2WikiMultihopQA, the gate closed the leg on 0 of
3,000 searches and every returned row is identical.

`w_graph` still defaults to `0.0`, so a deployment that never turned the leg on is
unaffected.

### Who this changes, and in which direction

**Anyone running `w_graph > 0` against a store built from one person's own sentences.**
Extraction from a user's turns produces claims that all take that user as their subject,
so their objects are leaves and nothing chains — and the leg was returning other facts
about the hub, ranked by a near-uniform path score, into a fusion that reads positions.
Measured on LongMemEval that cost 1.6 points of its strongest category. You will now get
the two-leg result, which is the same result `w_graph=0.0` gives.

If you were relying on the third leg as a recall booster rather than as a walk, this
removes it. That was tested: `graph_depth=1` on the same store gains nothing in any
category, so there was no recall to boost.

### How you find out it applies to you

It says so, once per retriever:

```
UnjoinedStoreWarning: graph retrieval is configured (w_graph=1.0) and nothing in this
store chains: none of its 78 live claim(s) have an object that is another claim's
subject, so a walk has nowhere to go and the leg is not running.
```

It is a subclass of `DegradedRetrievalWarning`, so an existing `filterwarnings` on the
parent already catches it. `memory_stats` reports the same thing as a join rate, and
`Memvara.connectivity()` returns the two counts.

### If you want the old behaviour

There is no flag, deliberately: it would be a switch whose only setting is "make retrieval
worse in a way I have measured". The condition is a property of your data, so the way out
is to write facts whose subject is not the hub everything else hangs off — one is enough
to lift the gate, and it lifts within `GATE_RECHECK_EVERY` searches without a restart.

---

## `Store` gains `connectivity`, so `isinstance(x, Store)` flips for a third-party backend

### What changed

`memory_stats` reports a **join rate** — the share of live claims whose object is the
subject of another live claim — and the counts come from a new optional `Store` method,
`connectivity`. `Memvara`, `ScopedMemvara`, `AsyncMemvara` and `AsyncScopedMemvara` all
gained a `connectivity()` of their own.

Nothing in memvara requires it. Capability checks here are `getattr` per member, the
method is listed in `store.base.OMITTABLE`, and a backend without it costs the
`memory_stats` line and nothing else. Retrieval is untouched and no default moved.

### The one thing that will not announce itself

**`isinstance(your_store, Store)` was `True` and is now `False`**, if your backend
implemented all 43 members and not this one. `Store` is `@runtime_checkable` and
`isinstance` on a Protocol is all-or-nothing: it asks whether every member is present, so
it has never been able to answer "can this store walk a graph", and it is not the check to
gate on. Find your instances with:

```bash
grep -rn "isinstance(.*, Store)" .
```

Replace each with the capability you actually need — `getattr(store, "adjacent", None)`
for the graph leg, `getattr(store, "connectivity", None)` for the join rate. That is what
this codebase does at every call site, and each one degrades in a way it names out loud.

To keep `isinstance` passing, implement the method. `SQLiteStore.connectivity` is the
reference; `memvara_cloud`'s `PostgresStore` is the second, and the two differ only in how
each spells an empty endpoint (`''` against `NULL`).

### If you call `connectivity()` yourself

**`{}` is not `{"live_claims": 0, "joinable_claims": 0}`.** The first is a backend that
cannot measure it; the second is a store that was measured and has no joins in it — a
*star*, which is what a memory built from one user's own sentences looks like, and which
is a real finding about the write path. Treating a missing key as zero reports the finding
without the measurement, so branch on the empty mapping before dividing.

---

## `erase()` can now raise, and the schema is version 8

### What changed

`Memvara.erase()` used to report success from the store's return code. It now re-queries
the disk afterwards (`prove_erased`) and raises `ErasureIncomplete` if anything survived,
or if the store cannot be asked. The two ordinary outcomes are unchanged: `True` means
proved gone, `False` still means there was nothing to erase.

**The exception is reachable from a store you already have.** `RemoteStore` (cloud mode)
cannot count rows, so every `erase()` against it now raises instead of returning `True`.
That is the intended behaviour — it was returning `True` for an erasure it could not
verify — but it is a behaviour change on a working configuration.

The SQLite schema goes 7 → 8, adding an `erasures` audit table. The migration is the
`CREATE TABLE` and nothing else: there is no data to backfill, and an upgraded file starts
with an empty table, which means "nothing erased since the upgrade" and never "nothing was
ever erased here". **A file opened by this build cannot be opened by an older one** —
`_migrate` refuses a store stamped newer than the build reading it, which is deliberate
and is the usual one-way door.

### What to do about it

If you call `erase()` in a loop over a legal erasure request, catch the exception and
treat the erasure as incomplete:

```python
from memvara import ErasureIncomplete

try:
    mem.erase(claim_id)
except ErasureIncomplete as exc:
    log.error("half-erased: %s still holds rows", exc.proof.surviving)
```

To check an erasure that happened months ago, or in another process:

```python
mem.prove_erased(claim_id).proven
```

**It is not scope-checked**, and that is stated rather than fixed: the claim is gone, so
there is nothing to scope-check against. It reveals whether any row with a given id
survives, for an id the caller must already hold. Treat erasure verification as an
operator action.

---

## `Explanation` gained four fields and two more retrieval legs exist


### What changed

`Explanation` now carries `graph_rank`, `graph_score`, `temporal_rank`, `temporal_score`
and `intent`, and `HybridRetriever.__init__` takes `w_graph`, `graph_seeds`,
`graph_depth`, `w_temporal`, `traverser` and `intent_weighting`. All are additive and
every default reproduces the previous behaviour exactly: `w_graph=0.0` and
`w_temporal=0.0` mean no walk and no time query run, nothing extra is fused, and both
pairs of fields stay `None` on every result.

`Store` gains one optional method, `episodes_near`. A store without it does not run the
temporal leg, exactly as one without `vector_search_episodes` does not run the vector half
of the episode search.

Two things will announce themselves anyway.

**`Explanation.summary()` and `repr(Result)` gained fields.** A test asserting on the
whole string will see `graph#2(0.750)` and `time#1(0.500)` appear once those legs are on;
neither is emitted while they are off, and both ship off.

**`intent=` is different: it is emitted by default.** `intent_weighting` ships **on**, so
a stock build already prints `intent=lookup` on an ordinary search — it is the one new
field a test can meet without turning anything on. It is absent only when
`intent_weighting=False`.

**`Memvara` now hands its `GraphTraverser` to its retriever.** It is the same object
`neighborhood()` walks, so the two cannot disagree about what the graph is. Pass
`read_traverser=` to wire a differently-bounded walk into retrieval than the one the
public method exposes.

### What to do about it

Nothing, unless you want the leg. To turn it on:

```python
mem = Memvara("memory.db", read_w_graph=1.0)
```

Read `docs/BENCHMARKS.md` first. Neither public retrieval benchmark can measure it — both
run the offline write path over conversational data it extracts almost nothing from — so
the default is 0.0 and the only measured gain is on a synthetic multi-hop workload.

**If your `Store` is a third-party one**, the leg needs `adjacent()`. A store without it
degrades to the two legs it had, with a `DegradedRetrievalWarning` raised once per
retriever rather than silently. `RemoteStore` (cloud mode) is in that category: the method
is present and raises.

## Erasure now actually removes the text, and the schema is version 7

### What changed

`erase()`, `purge()` and `reset()` left the erased words readable in the database file.
The store now sets `PRAGMA secure_delete=ON` and FTS5's `secure-delete`, so the bytes are
overwritten rather than freed, and opening an existing store scrubs what is already on
disk once.

**If you have run an erasure on any earlier version, the text may still be in that file.**
Opening it with this release cleans the text index. It does not rewrite pages that were
freed before the upgrade — for those, one `VACUUM` after the first open finishes the job:

```python
from memvara import Memvara
mem = Memvara("memory.db")      # migrates and scrubs the text index
mem.store._db.execute("VACUUM") # reclaims pages freed by pre-upgrade deletes
mem.close()
```

Check a file yourself — the point is to look at the file, not to ask the store, which
answered correctly all along:

```bash
grep -c 'something-you-erased' memory.db || echo "not present"
```

### What to do about it

Nothing, for most callers. Three things are worth knowing.

**Writes cost about 6% more and `erase_claim` about 9% more.** Measured on a 5,000-claim
run. That is the price of the bytes being overwritten.

**The one-time `optimize` runs on first open.** 0.01 s over a 20,000-claim index, and
bounded by segment count rather than row count, so a normally-written store has little to
merge.

**Schema 6 → 7, and it is a one-way door.** A file opened by this build is refused by an
older one, which is deliberate: the FTS5 option is durable state in the file, and an older
build would write to a text index whose format it does not understand. The option needs
SQLite 3.35 — already the store's minimum, so nothing that could open the file before is
locked out now.

**Still not scrubbed: the `-wal`.** An erased claim's bytes can remain in the write-ahead
log until it checkpoints. A clean `close()` or a checkpoint clears it; `SECURITY.md` now
records this as the remaining residue rather than the claim it used to make.

---

## A blank part of a triple is now an error, not a quiet no-op

### What changed

`memory_remember` refuses an empty or whitespace-only `subject`, `predicate` or `object`.
It used to accept the call, store nothing, and return every counter at zero with
`isError` false.

Three nearby messages changed text in the same release, all of them cases where the old
wording was true-but-useless or self-contradictory:

- `memory_end` on an **already-retired** claim now says so, instead of reporting it as
  ended;
- `memory_since` with a **future** instant says the instant has not arrived, instead of
  "what you knew then still stands";
- `recall(budget=)`'s cut notice no longer says "n further notes *matched*".

### How the mistake shows up

The refusal is the only one that changes a call's outcome, and it surfaces as a tool
result with `isError: true` where there used to be a zero-count success. Anything that
treated that success as "written" was already wrong — nothing was stored either way —
but a caller that never checked will now see an error where it previously saw none.

The other three are text. A log rule or assertion matching on `still stands`, `ended, not
retired`, or `further notes matched` stops matching.

### What to grep for

```
memory_remember
further notes matched
still stands
```

...in fixtures, assertions, and anything that builds a triple from interpolated values —
an empty variable is where a blank part comes from.

### What to replace it with

Check the value before writing it. If a field can legitimately be empty, the fact is not
ready to store: a triple missing one of its three parts is not a partial fact, it is not
a fact.

The library's `remember()` is unchanged — it does not raise on a blank part, and it does
not store one either, returning a receipt with `added 0`. That is the same silent no-op,
and it is left alone deliberately: a caller holding a `WriteReceipt` can read the zero and
decide, whereas a model reading rendered text cannot tell that zero from any other. The
guard belongs where the ambiguity is.

---

## Failure messages are flattened and cut

### What changed

Two error paths that used to pass text through whole:

- a tool that raises returns `<name> failed: <ExceptionClass>: <message>`, and the
  *message* is now flattened to one line and cut at 300 characters;
- `memvara-mcp login` cuts an upstream error body at 200 characters.

The exception class name is untouched, and so is any message already inside the cap —
which is nearly all of them. A cut is marked with `…`.

### How the mistake shows up

While debugging. A long exception — a multi-line traceback repr, a driver error quoting a
whole statement, an HTML error page from a gateway — is no longer complete in a tool
result or in the login output, and the missing half is the half that used to matter to
someone reading it. Newlines inside a message become spaces, so a message that was laid
out to be read no longer is.

Nothing is swallowed: the failure still surfaces, in the same place, with the same class
name and status code.

### What to grep for

```
failed: 
isError
```

...in anything that parses tool results, and in log-scraping rules that match on error
text. A rule anchored on a phrase deep inside a long message may stop matching.

### What to replace it with

For the real detail, read the process's own logs or run the failing call from the library,
where exceptions are untouched — this cap is on what is handed to a *model*, and on what a
CLI writes to a build log, not on Python's exception itself. A `try`/`except` around a
library call still sees the whole thing.

---

## Storing non-Latin text now emits a warning

### What changed

A claim whose text embeds to an all-zero vector raises `UnembeddableTextWarning` (once
per pipeline) and increments `write.embedding_unusable` (per claim, tagged by script).
With the default `HashingEmbedder` that is any claim containing no `[a-z0-9']`
characters — Han, Kana, Hangul, Arabic, Hebrew.

Nothing about the write changed. The claim is stored, the vector is stored, and
retrieval behaves exactly as before. This is a diagnostic for something that was already
happening silently.

### How the mistake shows up

Only two ways, and both are about warnings rather than about memory:

1. **A test suite or service running under `-W error`** — or
   `filterwarnings = ["error"]` in `pytest.ini` — turns this into an exception on a write
   that used to pass. That is the intended signal if you did not know your vectors were
   empty, and a false alarm if you did.
2. **Log volume**, if you knowingly store text your embedder cannot read. It is
   warn-once per pipeline instance, so a process building one `Memvara` sees one line;
   a server constructing one per request sees one per request.

### What to grep for

```
UnembeddableTextWarning
write.embedding_unusable
```

...after upgrading, in whatever collects your warnings or metrics. If the counter is
non-zero, that is the share of your store vector search cannot reach.

### What to replace it with

If the warning is telling you something true, install a real embedder:

```bash
pip install 'memvara[local-embed]'
```

That produces non-zero vectors for those scripts. Genuine *cross-language* retrieval —
querying in English for a fact stored in Chinese — needs a multilingual model and is not
claimed by either option.

If you have accepted the limitation and want the warning gone, it has its own category
precisely so you can silence it alone:

```python
warnings.filterwarnings("ignore", category=UnembeddableTextWarning)
```

The counter keeps counting either way, which is the point of it being separate.

---

## `subject` and `predicate` are now length-bounded on the MCP tools

### What changed

`subject` is capped at 128 characters and `predicate` at 64. Both were previously
unbounded — a 2,000-character subject was accepted — because the tool validator had no
`maxLength` support and no schema declared one. Over the limit is now a normal tool error
naming the limit, the length sent, and where the text should have gone.

`object` is **not** capped. It carries the fact itself, and a long one is a legitimate
value rather than a misuse.

### How the mistake shows up

A call that used to succeed now returns `isError: true`. In practice this only bites
something writing a sentence into `subject` or `predicate` — using the slot name as if it
were the value — which is the shape the cap exists to stop. Real predicates are far
inside the bound: the longest built-in is 21 characters.

Nothing already stored is affected. The cap is on new arguments, not on existing claims,
and no read path filters on length.

### What to grep for

```
memory_remember
memory_forget
memory_end
memory_history
```

...in anything that builds a `subject` or `predicate` by interpolation rather than from a
fixed vocabulary. Those are the calls that can exceed a bound without anyone intending it.

### What to replace it with

Put the detail in `object`, which is where a value belongs, and keep the predicate a
short snake_case relation. If you genuinely need a longer slot name, the library's
`remember()` is unchanged and applies no cap — this bound is on the MCP surface, where
the argument is filled in by a model.

---

## `memory_history` rows gained a `true from` field

### What changed

Each row used to read:

```
1. [id=cl_… recorded 2026-08-21 07:33Z ended 2026-08-21 09:00Z] user lives in Berlin
```

and now reads:

```
1. [id=cl_… recorded 2026-08-21 07:33Z true from 2024-01-01 00:00Z ended …] user lives in Berlin
```

The header changed with it, to name which clock "oldest first" refers to. Row order is
**unchanged** — still `recorded_at` ascending, which is the declared protocol behaviour
for every backend.

### How the mistake shows up

Only for something parsing the rendered text. A regex anchored on `recorded <stamp>]`
— that is, expecting the state word or the closing bracket immediately after the recorded
instant — no longer matches, because `true from <stamp>` now sits between them. A fixed
field-count split on the bracketed span comes out two tokens longer.

Nothing about the ordering or the set of rows moved, so a test asserting *which* values
come back, or in what order, is unaffected.

### What to grep for

```
memory_history
recorded 
```

...in anything that consumes tool output rather than the library.

### What to replace it with

Read the claim rather than the render: `history()` on the library returns `Claim` objects
with `recorded_at`, `valid_from`, `valid_to` and `invalidated_at` as fields, which is
where anything programmatic should have been reading them from. The rendered row is for a
model to read.

If you were reconstructing chronology from row order, that was never reliable and is the
reason for this change — a backfilled value is listed last while being the earliest. Sort
on `valid_from` if you want the world's order.

---

## Square brackets in stored text now render as `［` and `］`

### What changed

`Memvara._safe_line` — and so `recall()` and every line the MCP server emits — maps `[`
and `]` to U+FF3B and U+FF3D anywhere in a claim, not just at the head. A stored value
containing `[id=cl_… relevance=0.99] …` used to render as something that read like a
second, higher-scoring result row; the brackets are what made it parse, so the brackets
are what stopped being passed through. `SECURITY.md` has the reasoning.

Storage is unchanged. `Claim.text` on disk still holds exactly what was written, and
`search()` and `history()` still return the claim objects verbatim — this is a rendering
change, and only the rendering methods are affected.

### How the mistake shows up

Anything that parses the *rendered* text rather than the claim objects. A scraper reading
`recall()` output for `[...]` spans finds none where a note contained brackets; a golden
file or snapshot test over `recall()` or a `memory_*` tool result goes red on any fixture
with a bracket in it; a diff of two stores rendered before and after upgrading shows
changes in rows nobody edited.

An exact-match assertion is where this bites. Substring checks for the claim's words are
unaffected — the text is all still there, and still in the same order.

### What to grep for

```
recall(
_safe_line
safe_line
```

...in your own tree, then in whatever consumes their return value. Fixtures are the ones
worth checking by eye: `grep -l '\[' tests/**/*.txt` over any snapshot of rendered output.

### What to replace it with

If you need the original characters, read the claim rather than the render — `search()`
and `history()` hand back `Claim` objects whose `.text` is untouched. Rendered output is
for a model to read, and has never been a parsing target; this change is the reason that
distinction now matters in practice.

---

## The packaged skill moved, and `init` writes a directory

### What changed

The skill `memvara-mcp init` writes used to live at
`memvara/skills/claude/SKILL.md` and land as a single file under
`.claude/skills/memvara/SKILL.md`. It now lives at `memvara/skills/memvara/` —
`SKILL.md` plus a `references/` directory — and `--agent` chooses where that
tree is written (`claude`, `cursor`, `grok`). `--skill-only` writes the tree
and the project note, and leaves `.mcp.json` alone.

### How the mistake shows up

An older `.claude/skills/memvara/SKILL.md` still loads. What it will not have
is `references/examples.md` or `references/governance.md`, so an agent that
follows the new body and tries to open those files finds nothing. A script
that greps the old package path, or that treated `--agent cursor` as a usage
error, is looking at a layout that is gone.

### What to grep for

```
memvara/skills/claude
.claude/skills/memvara/SKILL.md
memvara-mcp init --agent
```

### What to replace it with

```
memvara-mcp init --agent claude --force
```

`--force` replaces a drifted `SKILL.md` and fills in the missing reference
files. Without it, `init` keeps a file you edited and only writes the
references that are absent. `--skill-only` if the client is already connected
and you do not want a new `.mcp.json`.

Coding agents that can install plugins can skip `init` for the hosted path:

```
/plugin marketplace add memvara/claude-memvara
/plugin install memvara
```

---

## `memvara-mcp init`'s default output changed, if you installed `memvara[cloud]`

### What changed

`memvara-mcp init` used to write one thing, always: a local server configuration pointed
at a file on disk. With the optional `cloud` extra installed (`pip install
memvara[cloud]`), it now defaults to the hosted path instead — it runs `memvara-mcp
login`, a device-code flow against the console at `https://app.memvara.dev`, and the
`.mcp.json` block it writes configures the server for `mode: cloud` rather than a local
`MEMVARA_DB`. Without the `cloud` extra, nothing about `init` changed: same files, same
local-only output, same as every prior release.

### How the mistake shows up

A CI job, a container build, or a teammate's machine that runs `pip install
memvara[cloud] && memvara-mcp init` non-interactively now hits a device-code prompt where
it used to finish silently — `login` waits on browser approval, which nothing headless can
give it. The failure mode is a hang or a timeout, not a wrong answer, but it is easy to
mistake for the package being broken rather than for the default having moved.

### What to grep for

```
memvara[cloud]                 # anywhere in requirements/pyproject/CI config
memvara-mcp init                # invocations with no --mode flag, in scripts or CI
```

Any hit combining an installed `cloud` extra with an unattended `init` call is a
candidate.

### What to replace it with

Pin the mode explicitly rather than relying on which extras happen to be installed:

```bash
memvara-mcp init --mode local        # unchanged local-file behavior, on any install
# or, in the server's own environment:
MEMVARA_MODE=local
```

`--mode local` (or `MEMVARA_MODE=local` on the server itself) is fully supported
regardless of which extras are installed and does not require a network call at any
point. See [docs/OPEN-CORE.md](OPEN-CORE.md)
for what the `cloud` extra does and does not add.

---

## `invalidated_at is None` no longer means "live"

**This is the one to read.** It is the only change in this project's history that is
wrong *silently*: no exception, no migration error, no red test, no deprecation warning.
The expression is still valid Python and still valid SQL. It used to be right.

### What changed

Ending a claim closes **one** of two clocks:

| you write | clock that closes | `Claim.state` | still believed? |
|---|---|---|---|
| a new value for a single-valued fact | valid time (`valid_to`) | `ended` | **yes** |
| `close="retired"`, `forget()`, `delete()` | transaction time (`invalidated_at`) | `retired` | no |

Superseding used to close both. It closes valid time alone now — the world changed, the
record was never wrong — so a superseded claim has `valid_to` set and `invalidated_at`
still `None`.

That claim is `ended`: **neither live nor invalidated**. The two conditions used to
select the same rows, and they no longer do.

### How the mistake shows up

Always in the same direction: too many. A store where one person's address has changed
four times reports **five** live claims instead of one. Nothing else in the data moves
with it, so the step is unfalsifiable from the inside — and on a metered or billed
surface, it is money.

### What to grep for

In application code, dashboards, saved queries, alert thresholds, notebooks, and any
third-party `Store` implementation:

```
invalidated_at is None        # Python
invalidated_at is not None    # Python — the mirror, and no longer the complement
invalidated_at IS NULL        # SQL, also `is null`, `ISNULL(`, `= NULL`
invalidated_at IS NOT NULL
```

Every hit is one of three things:

1. **A liveness test.** Now wrong. Replace it (below).
2. **A retirement test** — "which records did we stop believing?" Still exactly right,
   and now selects a strictly smaller set than "not live".
3. **An audit view** that wants everything ever displaced, either way. Use
   `invalidated_by IS NOT NULL`: the pointer is written under both closures, which is
   the whole reason it is a separate column.

Three copies of the wrong test existed across this project's own repositories when the
change landed, one of them a billing gauge, and finding them was manual.

### What to replace it with

```python
# one claim, in Python
claim.is_live()                       # now
claim.is_live(valid_at=T)             # in force at T, as we understand things today
claim.is_live(known_at=T)             # as we understood things at T
```

```python
# SQL, without needing a store instance
from memvara.store import live_predicate

sql = f"SELECT count(*) FROM claims WHERE {live_predicate('?')}"
# four binds, in the order: known, known, valid, valid
```

`live_predicate(at="?", *, include_invalidated=False, alias="")` takes the SQL
*expression* for the instant and substitutes it at every axis, so `"?"`, `"%s"` and
`"now()"` all work. Spelled out, it is:

```sql
SELECT count(*) FROM claims
 WHERE recorded_at   <= now()
   AND (invalidated_at IS NULL OR invalidated_at > now())
   AND valid_from    <= now()
   AND (valid_to     IS NULL OR valid_to     > now())
```

Two clocks, four columns, both bounds on each. `stats()["live_claims"]` is that same
predicate at the wall clock, and `stats()["ended_claims"]` is the population the old
idiom was quietly folding into it. `claims` is the only total that covers everything,
because the claim counts a store reports **no longer sum** — see the next entry.

---

## `stats()` gained `ended_claims`, and the counts still do not sum

`live_claims`, `ended_claims` and `invalidated` are three *disjoint* populations, and
their sum is not `claims`. A claim recorded but not yet in force — scheduled to start
next month — is in none of them.

Take a store with four claims: one live, one ended, one that ended and was *later*
retired, and one scheduled for next year.

```python
mem.stats()
# {'episodes': 0, 'claims': 4, 'live_claims': 1, 'ended_claims': 1,
#  'invalidated': 1, 'embeddings': 0}
```

`ended_claims` was added because it was the largest non-live population, it had no key,
and **it is not derivable**. Every cheaper way of getting it is wrong on that store:

| you might write | gives | truth | why |
|---|---|---|---|
| `valid_to IS NOT NULL` | 2 | 1 | the ended-then-retired row is already inside `invalidated`; this counts it twice |
| `claims - live_claims - invalidated` | 2 | 1 | the residual also holds the scheduled claim, which is in no state at all |

If you derived an "ended" or "not live" number by subtraction, re-derive it from the key.
If you are implementing a third-party `Store`, add `ended_claims` and resist the urge to
make the arithmetic close: a backend that "corrects" it has put the conflation back.

---

## Read filters take `states=`, and `include_invalidated` is its alias

Additive — every existing call keeps working — but it is the parameter to reach for now.
`Memvara.search` / `get_all` / `count`, their `ScopedMemvara` mirrors and the
`AsyncMemvara` / `AsyncScopedMemvara` ones all take `states=`, any non-empty subset of
`("live", "ended", "retired")`, defaulting to `["live"]`.

```python
mem.get_all(states=["retired"])          # the correction audit — everything we stopped
                                         # believing, and nothing that merely stopped
                                         # being true
mem.get_all(states=["ended"])            # the other half: still believed, no longer true
mem.get_all(include_invalidated=True)    # unchanged: exactly states=("live","ended","retired")
```

`include_invalidated` is a permanent alias, **not** deprecated, and emits no warning —
`filterwarnings = ["error::DeprecationWarning"]` would make a warning here fail every
existing call site rather than notify anyone. `False` means `["live"]`, `True` means all
three. Passing both raises `ValueError`; there is no reading of the mix in which one of
them is not being ignored.

Two things that will otherwise surprise you:

**Asking for all three states makes `valid_at` inert.** The three do not tile the store —
`Claim.state` is absolute while the query is as-of, so a claim recorded but not yet in
force at `valid_at` is named by none of them. The complete set therefore compiles to the
belief floor alone rather than to the union of its parts, which readmits that row and
leaves the world clock nothing to constrain. This is not new behaviour: it is exactly
what `include_invalidated=True` has always meant, now stated.

**`iter_claims` is the exception, and its default is `("live", "ended")`.** It filters the
*stored* state, not the state at an instant — it is a walk over rows with no moment to
ask about — and its unflagged view has always meant "every row we still believe" rather
than "every row in force right now". `include_invalidated=False` there is *not*
live-only, and narrowing it would silently stop `reembed()` re-encoding every superseded
version in the store. `include_invalidated` also stays positional there, because it
always was.

**Third-party `Store` implementations must widen their read-path signatures** —
`candidate_ids`, `lexical_search`, `vector_search` and `iter_claims` now take
`states: Collection[str] | None = None` alongside a widened
`include_invalidated: bool | None = None`. Route both through
`memvara.store.resolve_states`, which is the single place either spelling is
interpreted, and build the SQL with `state_predicate` / `stored_state_predicate` rather
than writing the clause again. `state_predicate` returns the axis behind every bind
marker, so binding becomes a comprehension over that list rather than a remembered order.

---

## `Store.erase_claim` returns counts, not a bool

```python
store.erase_claim(claim_id, sources=False)
# {'claims': 1, 'episodes': 0, 'embeddings': 1, 'entities': 1}
store.erase_claim("cl_does_not_exist")
# {'claims': 0, 'episodes': 0, 'embeddings': 0, 'entities': 0}
```

The same four keys `purge` returns, so the two erasure paths evidence themselves the same
way — and the per-claim path is the one an erasure request naming a single memory
actually takes. A missing id returns **all zeroes rather than an absent key**, so a caller
totalling an erasure campaign never special-cases it. `counts["claims"]` is 0 or 1 and
carries exactly what the boolean carried.

**`Memvara.erase()` still returns `bool`, deliberately.** Widening it would change a
published signature from a flag to a mapping, and every `if mem.erase(id):` in existence
would start taking the branch unconditionally — a dict of zeroes is truthy. A caller who
wants the evidence calls `store.erase_claim` or `purge()`.

```python
if mem.erase(claim_id):          # still correct
    ...
if store.erase_claim(claim_id):  # ALWAYS true — check ["claims"] instead
    ...
```

---

## `WriteReceipt.invalidated` is now `WriteReceipt.closed`

Same list, better name, and two new derived views that answer the question the single
field could not.

```python
receipt = mem.add(transcript)

receipt.closed        # claims this write closed out, on either clock
receipt.ended         # ... the ones the world moved past   (close="ended")
receipt.retired       # ... the ones we stopped believing   (close="retired")
receipt.invalidated   # the old name. Same list object. Still works.
```

`invalidated` is a deliberate alias and raises **no** `DeprecationWarning` — this
package sets `filterwarnings = ["error::DeprecationWarning"]`, so a warning would break
callers rather than notify them. It will be removed at `1.0.0`.

The rename is worth making at your call sites for the reason the split exists: a receipt
holds whichever closure the write applied, so code that renders `invalidated` with the
word "retired" is wrong for every supersession — which is almost all of them.

## The MCP server distinguishes the two closures

`memory_add` and `memory_remember` used to report `added N, retired N, ...` where the
second number counted every closure. A supersession is not a retirement, so a model
reading its own memory tool was told the record had been wrong when only the world had
moved on — while `memory_history` rendered the same claim as `ended` and `memory_forget`
used "retired" for the thing that genuinely is one. Three names, two events.

Captured from a live server, `MEMVARA_LLM` unset — two `memory_add` calls and a
`memory_forget`, verbatim:

```
> memory_add {"text": "I live in Berlin"}
added 1, ended 0, retired 0, already-known 0, no-fact 0 (0 model call(s))
+ [cl_047bac579e6d4ed680cc] user lives in Berlin

> memory_add {"text": "I live in Lisbon"}
added 1, ended 1, retired 0, already-known 0, no-fact 0 (0 model call(s))
+ [cl_44b2c5ad486f491d9d43] user lives in Lisbon
- [cl_047bac579e6d4ed680cc ended 2026-08-13 06:58Z] user lives in Berlin

> memory_forget {"subject": "user", "predicate": "lives_in"}
Retired 1 value(s) of user/lives_in. They no longer answer questions; memory_history still shows them.
- [cl_44b2c5ad486f491d9d43] user lives in Lisbon
```

Both counts always appear, and each displaced claim carries its own closure — so the
second call says `ended`, the third says `Retired`, and they mean different things. This
is tool output, not an API: nothing to migrate, but a transcript or an eval fixture that
pins the old string needs updating.

## Turns filed under tenant `"default"` by `remember(sources=[Episode(...)])`

`Episode.scope` defaults to `Scope()`, whose tenant is the literal `"default"`. So the
documented way to attach provenance — building an `Episode` yourself and passing it as a
source — wrote the raw user text into that tenant while the claim it supported landed in
the right one. `get_episode` is unscoped, so `why()` went on resolving and nothing
surfaced it; what did surface it was an erasure reporting `episodes: 0` with the sentence
still on disk. A caller-built episode that names no scope now adopts its claim's.

**Only stores that use an explicit tenant can be affected.** If every write went through a
`Memvara(...)` left at the default tenant, the episode and its claim both landed in
`"default"` and there is nothing to find — the mismatch needs a second tenant to be a
mismatch. Verified both ways against the shipped schema before writing this.

### Detecting it

Read-only, and it names the tenant the turn should have been under:

```sql
SELECT e.id     AS episode_id,
       e.tenant AS episode_tenant,   -- where the turn landed, usually 'default'
       c.tenant AS claim_tenant,     -- where it should have been
       count(*) AS claims_affected
FROM episodes e
JOIN claim_sources s ON s.episode_id = e.id
JOIN claims        c ON c.id         = s.claim_id
WHERE e.tenant <> c.tenant
GROUP BY e.id, e.tenant, c.tenant
ORDER BY e.tenant, e.id;
```

No rows is the healthy answer. It looks for the mismatch rather than for `'default'`
specifically, so it also catches a turn misfiled under any other tenant.

### What it means for an erasure you have already run

This is the part worth acting on. `erase()` and `purge()` are scoped, so a request served
while the turn sat in another tenant deleted the claim and reported `episodes: 0` — a
truthful count of what it found, about a sentence that is still on disk. If you have
answered a deletion request for anyone whose turns this query returns, the text of those
turns was not erased. Re-run the erasure for the affected tenant, or delete the rows the
query names, and note the correction wherever the original erasure was recorded.

Moving a turn is not just an `UPDATE`: `episodes_fts` and `episode_embeddings` are keyed
by `episode_id` and carry no tenant of their own, so re-filing a row leaves its index
entries reachable from the tenant they were written under. Deleting the affected episodes
and re-attaching provenance is the safer repair.

---

Previous: [Documentation index](README.md) · Next: [Roadmap](ROADMAP.md) · [Changelog](../CHANGELOG.md)
