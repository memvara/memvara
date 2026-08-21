# Upgrading

What breaks, and what to do about it. `CHANGELOG.md` is the full record; this file is
the short list of things that will not announce themselves.

Entries are newest first, and each one says how you find your own instances of it.

---

## `Explanation` gained two fields and a third retrieval leg exists

### What changed

`Explanation` now carries `graph_rank`, `graph_score` and `intent`, and
`HybridRetriever.__init__` takes `w_graph`, `graph_seeds`, `graph_depth`, `traverser` and
`intent_weighting`. All are additive and every default reproduces the previous behaviour
exactly: `w_graph=0.0` means no walk runs, no leg is fused, and `graph_rank` stays `None`
on every result.

Two things will announce themselves anyway.

**`Explanation.summary()` and `repr(Result)` gained fields.** A test asserting on the
whole string will see `graph#2(0.750)` and `intent=lookup` appear once the leg is on;
neither is emitted while it is off, and `intent=` is absent whenever
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
