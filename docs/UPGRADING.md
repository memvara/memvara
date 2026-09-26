# Upgrading

What breaks, and what to do about it. `CHANGELOG.md` is the full record; this file is
the short list of things that will not announce themselves.

Entries are newest first, and each one says how you find your own instances of it.

---

## Creating or upgrading a store needs permission to write `<db>.lock`

### What changed

Only one process at a time now creates a store or upgrades it to a newer schema, so two
processes that open it at once no longer get in each other's way (#281). The process doing
it holds a write lock on `<db>.lock` while it works, and it needs permission to write that
file to take the lock. An open of a store that needs neither only reads the file, as before.

If `<db>.lock` exists and the account that opens the store may not write it, the open that
would create or upgrade the store raises `PermissionError`. The message names the lock file,
for `memory.db` the file `memory.db.lock`, and says what to do. Nothing has been created or
upgraded when it is raised.

### Who this changes

**If `<db>.lock` belongs to another account,** for example because an MCP server once ran
as another user, and the store needs an upgrade. The open that upgrades a store is the first
open of it after you install a memvara version with a newer schema, or the first open of a
store an older version wrote, so that is when you meet the refusal.

### What to do

Make `<db>.lock` writable by the account that opens the store, or delete it while nothing
has the store open; the next open creates it again. It holds no data.

### How to find it in your code

This is about files, not code. `ls -l memory.db.lock` shows the file's owner and mode; the
account that runs memvara must be able to write it.

---

## `forget()` also closes a value stored to begin later

### What changed

`forget(subject, predicate)` closes every value in the slot that the store believes and
that has not ended. Before this release it closed only the values in force at the moment
of the call, so a value stored with a `valid_from` in the future was left alone and began
answering when its start arrived, in a slot the caller had just forgotten. That value is
now closed with the others and returned with them. `forget()` returns the values in the
order they were recorded.

Under `close="ended"`, a value that has not begun by `at` is ended at its own start, so it
is true at no instant and never begins. `memory_forget` and `memory_end` given a
`predicate` call `forget()`, so on a local server they do the same. Closing one value by
its id, with `delete()` or with the tools' `claim_id`, is unchanged.

### Who this changes

**If you store a value ahead of time and then close its slot,** the value you stored
ahead is now closed too. For example, you record that someone starts at Globex next month
with `valid_from` set to that date, and later call `forget("user", "works_at")` or
`memory_end` with `predicate` to close the current employer. Before this release, Globex
survived and started answering next month. Now `forget()` retires it, and
`close="ended"` or `memory_end` ends it at its own start, so it never answers. To close
only the current value, close it by id instead: `delete(claim_id)`,
`delete(claim_id, close="ended")`, or `memory_forget` or `memory_end` with `claim_id`.

**If you read the list `forget()` returns,** it now includes the values stored to begin
later, in recorded order.

**`RemoteMemvara` does not change with your client.** A hosted deployment runs its own
`forget()`, so this reaches a hosted store when the deployment moves to this release. A
server started with `MEMVARA_MODE=cloud` therefore describes `memory_forget` and
`memory_end` as closing every current value and does not promise the rest.

**If you implement your own `Store`,** it gains an optional member, `unended_claims`.
`forget()` asks the store for the values to close with it, so a slot that has held many
values is not read whole. A backend without the method keeps working and closes the same
values: `forget()` reads the slot with `slot_history` and filters it with
`Claim.is_unended`, which costs a read of every value the slot has held. It is listed in
`store.base.OMITTABLE`. As with `connectivity` below, `isinstance(your_store, Store)`
turns `False` for a backend that implemented every other member, because `isinstance` on
a Protocol asks for all of them; check the capability with `getattr` instead. To
implement it, select with `store.unended_predicate()`, which returns the SQL and the clock
behind each marker, and order by `recorded_at` and then `id`, as `slot_history` does.
`SQLiteStore.unended_claims` is the reference.

### How to find it in your code

Search for `forget(` and for `memory_forget` and `memory_end` calls that pass a
`predicate`, and check whether the slot can hold a value stored to begin later: one
written by `remember(..., valid_from=<a future instant>)`, or by `memory_remember` with a
`true_since` in the future. This lists the values stored to begin later that a store holds
now, each of which would be closed by a `forget()` of its slot:

```python
from memvara import Memvara
from memvara.types import utcnow

mem = Memvara("memory.db")
now = utcnow()
for c in mem.store.iter_claims():
    if c.valid_from > now and (c.valid_to is None or c.valid_to > c.valid_from):
        print(c.id, c.subject, c.predicate, repr(c.object), "begins", c.valid_from)
```

---

## A restatement with an earlier start is added, not reinforced

### What changed

Writing a value the store already holds live, with a `valid_from` earlier than the start
of the stored claim, used to count as a repeat. The receipt named the stored claim under
`reinforced`, and the earlier start was dropped. Now the write stores the earlier period
as a claim of its own, ending where the stored claim begins, and the receipt names that
new claim under `added` instead of naming the stored claim under `reinforced`. The store
holds one more row than it did before. The stored claim itself is not changed, and its
observation count and salience do not rise.

This applies to `remember()`, `supersede()`, `memory_remember`, and every claim that
`add()` extracts from a turn. A few related cases:

- Only a stored claim the writer can see counts: one in its own scope or in a broader
  scope it reads, never one in a sibling project, agent or session.
- Writing the same earlier start a second time is a repeat of the claim for the earlier
  period. The receipt names that claim under `reinforced`, and nothing is stored.
- A write that starts even earlier stores only the part before the stored periods
  begin.
- A write that names `expires_at` is still a repeat of the stored claim, as before, so
  the expiry lands on it.
- `memory_remember`'s reply for such a write reads `added 1 ... already-known 0` where
  it read `added 0 ... already-known 1`, followed by a note that the same value is
  already stored from where the new claim ends.
- A turn that `add()` takes for a repeat in tier 0, before extraction, is not covered
  yet and still reinforces the stored claim (#318).

### Who this changes

**If you read `receipt.reinforced` to decide whether a fact was already known,** a
restatement with an earlier start now shows up in `receipt.added` instead. The claim
there is already over: its `valid_to` is where the stored claim begins, so it is not a
new current value. To tell it apart from a value the store did not hold, check whether
its `valid_to` is set and `history(subject, predicate)` holds the same object from that
instant.

**If you count stored rows**, such as `stats()["claims"]` or the length of `history()`,
a restatement with an earlier start adds one. A plain `count()` is unchanged, because
the added claim is already over; a `count(valid_at=...)` inside the earlier period now
includes it.

**If you import history with `remember(valid_from=...)`**, a fact restated from an older
record now keeps the period it covers, so reads at those earlier instants return it.

### How to find it in your code

Search for `.reinforced` and `.added` in code that reads the receipt of `remember(`,
`supersede(` or `add(`, and for the text `already-known` in code that parses
`memory_remember` replies, where the write can carry a `valid_from` or `true_since` in
the past.

---

## `reembed()` needs the store to itself, and every store keeps a `<db>.lock` file

### What changed

`SQLiteStore.clear_embeddings()`, which `Memvara.reembed()` and `Memvara(...,
reembed=True)` call first, now raises `StoreInUseError` while another process, or another
`SQLiteStore` in the same process, has the store open. It changes nothing when it
refuses. Before, it truncated the vector file under them, and a process that still mapped
the file died with SIGBUS on its next vector search, with no exception anyone could catch.
An encrypted store did not crash, but went on returning the vectors that had been
cleared.

To make that visible, every store that opens a file-backed database now keeps
`<db>.lock` beside it: for `memory.db`, `memory.db.lock`. It holds no data. Each open store
holds a shared lock on it until `close()`, and a clear takes it exclusively. A store that
opens while another store is clearing waits for the clear to finish, for up to 60 seconds,
and then raises `StoreInUseError`.

### Who this changes

**If you re-embed a store that an MCP server, a worker or a notebook has open,** stop them
first, then run `reembed()` and restart them with the new embedder. They had to restart
anyway: each process embeds new writes with the model it started with.

**If a test or a script opens a second `SQLiteStore` on the same file and clears the
vectors through one of them,** close the other first.

**If the refusal names a store you thought was closed,** look for one that was opened and
never closed and that something still refers to, such as a variable in a notebook. A store
that nothing refers to any more does not count, because a refused clear collects garbage
once and asks again. Neither does a store whose construction failed, or the store that a
failed `Memvara(...)` had opened, unless what failed was its expired-claims sweep.

**If you back up or copy a store's directory,** `memory.db.lock` can be copied or left
out; the next open recreates it. Do not delete it while anything has the store open,
because a clear would then stop seeing that store.

### How to find it in your code

Search for `reembed(`, `reembed=True` and `clear_embeddings(` in code that runs while
another process may have the store open, and catch `StoreInUseError` there if a refusal
should not end the program.

---

## "C++", "C#" and "C" are three values, and a store re-keys its claims on first open

### What changed

`entity_key()` used to drop every punctuation mark, so "C++", "C#" and "C" folded to one
key, and so did the blood types "A+" and "A-". It now keeps a `+` at the end of any word,
and a `#` or a single `-` at the end of a word of one or two letters. The minus sign U+2212
counts as `-`. Inside a word the symbols still separate, so "x-ray" and "X ray" are one
value, as before. The fold also strips every leading "the" instead of only the first, so
"The The Band" folds to `band`, where it folded to `the band`, a key that did not fold to
itself.

Every claim's `subject_key`, `object_key`, `fact_key` and `value_key` is built from that
fold, so `SQLiteStore` moves to schema version 16. The first open of an older file
re-derives all four for every claim from the text you wrote. Text keys exactly as before
unless a word ends in one of these symbols, or it starts with "the" twice once legal forms
such as "Inc" are dropped.

### Who this changes, and in which direction

**If your store holds such values, some may already have been lost to a repeat.** Before
this release, a second value that folded to the same key as a live claim was recorded as
another observation of that claim, not stored. So "C#" after "C++" counted as a second
observation of "C++", and in a slot that holds one value, "A-" after "A+" counted as a
second observation of "A+". Re-keying cannot bring such a value back, because it was
never a claim. What remains is the turn it came from, among the sources of the claim it
was folded into, and the listing below finds those claims. A value written by
`remember()` without `sources` left no turn behind. To restore a value, write it again
with `remember()`, which now stores it as a value of its own. In a slot that holds one
value, that write also ends the value it was folded into, so first decide which of the
two is current.

**If you call `entity_key()` yourself, its output changes for these names.**
`entity_key("C#")` returns `"c#"` where it returned `"c"`, so a key you saved from an
earlier version no longer equals the key this version computes for the same text.

**Aliases the entity registry learned stay filed under the old key.** A store whose
registry learned aliases, through a model or `EntityRegistry.learn_alias()`, keeps each
under the key its spelling folded to when it was learned. An alias learned for "C#" was
filed under `c`, so it still answers for "C" and no longer answers for "C#".

**If a stored name starts with "the" twice**, such as "The The Band", its key changes
from `the band` to `band`, and it becomes one value with "The Band". Its slot does not
move, because the slot was already computed from `band`. The query at the end of this
entry finds such claims.

**`RemoteMemvara` computes no keys.** The hosted service does, so a hosted store changes
when the service moves to this release, not when your client does.

### How to find your instances

After upgrading, this lists the live claims that absorbed a repeat whose source turn
contains a word ending in `+`, `#` or `-`. It over-reports, since a turn can mention such
a word without it being the value, so read each source before writing anything back:

```python
import re
from memvara import Memvara

# A word followed by `+`, `#` or `-`, where the fold now keeps the symbol.
NAME_SYMBOL = re.compile(r"\w[+#\-\u2212]+(?!\w)")

mem = Memvara("memory.db")
for c in mem.store.iter_claims(states=["live"]):
    if c.observation_count < 2 or not c.sources:
        continue
    turns = [ep.content for ep in mem.store.get_episodes(c.sources).values()]
    if any(NAME_SYMBOL.search(t) for t in turns):
        print(f"{c.id} {c.subject} {c.predicate} {c.object!r}")
        for t in turns:
            print("    ", t[:160])
```

Claims whose subject or object starts with "the" twice, which re-key to the name without
them. `LIKE` ignores ASCII case in SQLite. A spelling with punctuation or a legal form
between the two, such as "The, the Band" or "The Inc The Band", is not caught:

```sql
SELECT id, subject, predicate, object FROM claims
WHERE subject LIKE 'the the %' OR object LIKE 'the the %';
```

---

## The write path reads fewer turns as restatements

### What changed

Before extracting from a turn, `add()` compares it with the nearest live claim. A turn
close enough counts as a restatement: the claim gains an observation and the turn as a
source, and nothing is extracted from the turn. That check used a cosine of 0.97 under
every embedder and ignored numbers. It now uses the merge's threshold for the embedder's
space, 0.985 for all-MiniLM-L6-v2, 0.99 for bge-small-en-v1.5 and 0.97 for any other
embedder, and it never counts a turn whose numbers differ from the claim's text.

### Who this changes, and in which direction

**If your turns are worded like claims**, as `subject predicate object`, more of them now
reach extraction. That is the wording an agent or an importer produces when it sends
"user has appointment on 2024-03-15" instead of a sentence a person wrote. A turn stating
another value is now extracted instead of reinforcing the claim it resembles. A close
restatement that scores between 0.97 and the new threshold is extracted too, so
`receipt.skipped` is lower and, with a model configured, `receipt.llm_calls` can be
higher.

**Turns written by people are rarely affected.** None of the 16 first-person turns in
`bench/embedder_calibration.py` comes near either threshold: the highest scores 0.944
against its claim, so the check read none of them as a restatement before this release
either.

**If you pass `write_near_dup_threshold=`**, or `near_dup_threshold=` to `WritePipeline`,
your number is still used, and the rule on numbers now applies under it as well. If you
read `WritePipeline.near_dup_threshold`, it is now `None` unless a number was passed; the
threshold in use is then `calibration_of(pipe.embedder).merge`, from
`memvara.embed.calibration`.

**Values lost before this release stay lost.** A turn the check read as a restatement was
stored, and it became a source of the claim it was read as repeating, so its text is
still there. This lists every source turn that the check read as a restatement at 0.97
and would not read that way now. It embeds each claim's source turns again, one call per
claim. To restore a value, write it again with `remember()`.

### How to find your instances

```python
import numpy as np
from memvara import Memvara
from memvara.embed.calibration import calibration_of, numbers

mem = Memvara("memory.db")
threshold = calibration_of(mem.embedder).merge

def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)

for c in mem.store.iter_claims(states=["live"]):
    stored = mem.store.get_embedding(c.id)
    if stored is None or not c.sources:
        continue
    turns = [ep.content for ep in mem.store.get_episodes(c.sources).values()]
    for turn, vector in zip(turns, mem.embedder.encode(turns)):
        cosine = float(unit(vector) @ unit(stored))
        if cosine >= 0.97 and (cosine < threshold or numbers(turn) != numbers(c.text)):
            print(f"{c.id} {c.text!r} <- {turn[:160]!r} ({cosine:.3f})")
```

---

## Consolidation never merges two values whose numbers differ, and merges less under MiniLM

### What changed

The duplicate merge in consolidation folds two live claims in one slot when their
embeddings are close enough, and retires the one it folds. It now refuses two claims whose
values hold different numbers, whatever their cosine and whatever the embedder:
"2023-05-01" and "2023-05-02", "numpy 1.26.4" and "numpy 1.26.2", "85,000" and "85000".
Numbers are compared in order and without leading zeros, so "09:30" and "9:30" can still
merge. A store written with `sentence-transformers/all-MiniLM-L6-v2` also merges at a
cosine of 0.985 instead of 0.97.

### Who this changes, and in which direction

**If your store was written with MiniLM**, as every store `LocalEmbedder()` wrote through
0.15 was, consolidation merges less from its next run. On the pairs in
`bench/embedder_calibration.py`, MiniLM at 0.97 folded 22 of 69 pairs of different values
and 4 of 24 restatements that hold the same numbers. Now it folds none of the 69 and 1 of
the 24. Restatements it used to fold, such as "NYC" and "New York City", stay two live
claims, and `recall()` can return both.

**Under any embedder**, two values that write one number differently, such as "85,000"
and "85000", now stay two claims. The two such pairs in the measurement scored below 0.96
in all three spaces measured, so the default thresholds did not merge them either; only a
lower `threshold=` did.

**If you pass `threshold=` to `merge_duplicates()`**, the threshold still decides, but
never for two values whose numbers differ.

**Merges already made stay made.** A merge retires the claim it folds and does not delete
it. The retired claim's `invalidated_by` names the claim it was folded into, and
`mem.store.iter_claims(states=["retired"])` returns it. This lists every merge in a store,
and marks the ones this release would refuse under any embedder:

```python
import re
from memvara import Memvara

def numbers(text):
    return [n.lstrip("0") or "0" for n in re.findall(r"\d+", text)]

mem = Memvara("memory.db")
for c in mem.store.iter_claims(states=["retired"]):
    # forget() and replaces= record a retirement under meta["closure"]; a merge does not.
    if not c.invalidated_by or any(e.get("close") == "retired" for e in c.meta.get("closure", [])):
        continue
    kept = mem.store.get_claim(c.invalidated_by)
    # A backfill also folds claims, but only two that hold one value.
    if kept is None or kept.value_key == c.value_key:
        continue
    note = "  numbers differ" if numbers(kept.object) != numbers(c.object) else ""
    print(f"{c.id} {c.object!r} was folded into {kept.id} {kept.object!r}{note}")
```

In a MiniLM store, an unmarked line may also be a merge this release refuses. Only a
reader can tell a restatement, such as "NYC" and "New York City", from a different value,
such as "left knee" and "right knee". To bring a value back, write it again with
`remember()`, which stores it as a new live claim.

---

## A process that searches a large scope keeps that scope's claim list in memory, and claim writes maintain one more index

### What changed

The vector leg over claims keeps, per scope and per set of states, the ids and matrix rows
of the claims a read of the present can see, and ranks those instead of asking SQLite for
the list on every search. It returns the same claims. A list is kept until the next commit,
or until the clock reaches the next instant at which a claim of the same tenant starts,
ends, is retired, expires or becomes known, whichever comes first. Each claim costs about
90 bytes, so 100,000 claims in one scope hold 8.9 MB, and building them peaks at 14.5 MB.
The lists across all scopes are capped at 1,000,000 claims, about 90 MB, and the least
recently used scope goes first.

The store finds that next instant through a new index on the claims table,
`cl_last_change`, created on open like the store's other late indexes. A store written by
an earlier version builds it the first time this version opens it, in about 0.1 s, and the
file grows by about 2.5 MB, per 100,000 claims. Every claim write maintains it, which
costs about 5%: +3.5 µs per claim written inside `batch()` and +24 µs per claim committed
on its own, over 20,000 claims. An earlier version that opens the file afterwards keeps
the index and maintains it too.

### Who this changes, and in which direction

**If you run memvara where memory is tight and one scope holds hundreds of thousands of
claims**, budget about 90 bytes per claim per process on top of what it used before.

**If claim writes are your bottleneck**, expect them about 5% slower.

Nothing else changes. A read pinned to an instant, a filtered search, a search inside
`batch()` and a store with no vectors yet read exactly as before.

---

## A search embeds its query the way bge expects a query

### What changed

`LocalEmbedder` puts the instruction bge's English models are trained to see before a
search query, `"Represent this sentence for searching relevant passages: "`, before each
query a search embeds, when its model is one of them: `BAAI/bge-small-en-v1.5`, the
default, or its base or large sibling. It embeds what a store keeps exactly as before, so
no store needs re-embedding. Every other model, `HashingEmbedder`, and an embedder of your
own embed a query as they always did.

### Who this changes, and in which direction

**If you pass `min_score` to `search()` or `recall()` with bge-small**, check it again. A
result's score reads the cosine between the query and the row, and the instruction lowers
that cosine by about 0.03: over LOCOMO's questions, the median cosine to the best turn went
from 0.746 to 0.713, so a threshold that used to pass a row may no longer.
`episode_score_floor` is a fraction of the best result's score, so it moves less.

**If you wrap `LocalEmbedder` in an embedder of your own**, give the wrapper an
`encode_queries` method that calls `memvara.embed.encode_queries` on the embedder it
wraps, as `CachedEmbedder` does. Without one, a search through the wrapper embeds its
query through `encode`, without the instruction, and finds what it found before.

---

## A search on a store with a file uses up to three more threads

### What changed

`HybridRetriever` runs each stage's vector leg on a pool thread beside its lexical leg,
when the store is a `SQLiteStore` with a file and the calling thread is not inside
`batch()`. The pool is made on the first such search and kept: up to three threads per
retriever, and so per `Memvara`, each holding its own SQLite read connection, as the
threads a rewritten read uses already do. Results are unchanged, and the embedder is still
called only from the thread that called `search()`.

### Who this changes, and in which direction

**If you count threads or open file handles per process**, allow three more of each per
`Memvara` that searches a file store. `close()` closes the connections.

**If you pass your own `Store`**, nothing changes: a store without `SQLiteStore`'s private
`_parallel_reads` keeps both legs on the calling thread.

---

## A process that searches a large scope keeps that scope's turn list in memory

### What changed

The vector leg over turns keeps, per scope, the ids, times and matrix rows of the turns
that have a vector, and ranks those instead of asking SQLite for the list on every search.
It returns the same turns. Each turn costs about 100 bytes, so 199,499 turns in one scope
hold 19.3 MB, and building them peaks at about twice that. The lists across all scopes are
capped at 1,000,000 turns, about 100 MB, and the least recently used scope goes first. Every
commit empties them, so the first search after a write rebuilds the list it needs and costs
a little more than a search did before: over 199,499 turns, 238 ms for the vector leg
against 220 to 230 ms.

### Who this changes, and in which direction

**If you run memvara where memory is tight and one scope holds hundreds of thousands of
turns**, budget about 100 bytes per turn per process on top of what it used before. A
store whose scopes hold a few thousand turns each will not notice.

**If you count open file handles per process**, allow one more SQLite connection per store
that searches turns. The store uses it only to ask whether another connection has
committed, and `close()` closes it.

Nothing else changes. A filtered search, a search inside `batch()` and a store with no
vectors yet read exactly as before.

---

## `LocalEmbedder()` loads bge-small-en-v1.5

### What changed

`LocalEmbedder()` with no model now loads `BAAI/bge-small-en-v1.5`. Through 0.15 it loaded
`sentence-transformers/all-MiniLM-L6-v2`. The two have the same width, 384, so no
dimension check can tell their vectors apart; the name in the store's fingerprint,
`<db>.embedder.json`, is what does. The grounding rescue and the duplicate merge read
bge-small's cosines with thresholds measured for it, 0.65 and 0.99. MiniLM keeps 0.40 and
merges at 0.985 (see the consolidation entry above), and every other embedder keeps 0.40
and 0.97.

### Who this changes, and in which direction

**If you construct `Memvara()` with no `embedder=`** and sentence-transformers is
installed, an existing store opens with the local model its fingerprint names, so nothing
changes for it. A new store gets bge-small. A store with 384-wide vectors and no
fingerprint keeps MiniLM, the model a default configuration wrote it with.

**If you run the MCP server with `MEMVARA_EMBEDDER=local`**, the same holds. The server
reads the model off the store's fingerprint before it starts, so a deployment with an
existing store neither changes model nor fetches bge-small.

**If you construct `LocalEmbedder()` yourself** and open a store MiniLM wrote, `Memvara`
now raises `EmbedderMismatchError`. Through 0.15 that opened, because `LocalEmbedder()`
was MiniLM then. The message names the fix: pass
`LocalEmbedder("sentence-transformers/all-MiniLM-L6-v2")` to keep the store as it is, or
migrate it once with `Memvara(..., embedder=LocalEmbedder(), reembed=True)`. Find your
instances by searching for `LocalEmbedder()`.

**If you call `merge_duplicates()` without `threshold=`**, its default is now the
calibrated value, which is 0.97, as before, for every embedder but bge-small and MiniLM. A
threshold you pass is used as it is.

**If ingest time matters**, bge-small encodes at about half MiniLM's speed on a CPU: 192 s
against 98 s for LOCOMO's 5,882 turns, and 12 ms more for the median read.

---

## The first open of an existing store builds one index

### What changed

`SQLiteStore` has a new index on the episodes table, `ep_cover`, which the vector leg's
turn list reads instead of the table. It is created on open, like the store's other late
indexes, so a store written by an earlier version builds it the first time this version
opens it. That open is slower once, by about 1.7 s per 190,000 turns, and the file grows
by about 10 MB per 190,000 turns. Every open after that is unchanged. An earlier version
that opens the file afterwards keeps the index and uses it.

The lexical legs now join the text index to its table on rowid. That is only correct while
each text index row sits at the rowid of the row it indexes, which every write in this
library keeps true, and which `VACUUM`, `VACUUM INTO` and SQLite's backup API preserve.

### Who this changes, and in which direction

**If you open a large store where a pause matters**, open it once after upgrading, at a
time a slow open costs nothing: `SQLiteStore(path).close()` builds the index.

**If you have copied a store by re-inserting its rows into a new file**, such as a SQL
dump replayed into an empty database, the text index may no longer line up with the rows.
Erasure already relied on that, and now lexical search does too: such a store can miss a
lexical match or return another row in its place. Rebuild both text indexes from their
tables with the store closed:

```sql
DELETE FROM claims_fts;
INSERT INTO claims_fts (rowid, claim_id, text) SELECT rowid, id, text FROM claims;
DELETE FROM episodes_fts;
INSERT INTO episodes_fts (rowid, episode_id, content) SELECT rowid, id, content FROM episodes;
```

A store that has only ever been written by this library, and copied as a file, with
`VACUUM INTO` or with the backup API, needs nothing.

---

## `recall()` dates each turn, and shows the part of a long turn the question names

### What changed

Every line under `recall()`'s episode header, `RECALL_EPISODE_HEADER`, now starts with
the day the turn was said, in brackets: `- [8 May 2023] I went to the support group
yesterday`. A turn longer than `RECALL_EPISODE_CHARS` (280) is no longer cut to its first
280 characters. It is cut to the sentence that shares the most words with the question,
plus as many neighbouring sentences as fit, with `…` on each side where text was left
out. A turn that shares no word with the question is still cut from its start. Nothing
changes for claims, for the headers, or for which turns are returned.

### Who this changes, and in which direction

**If you parse `recall()` output**, an episode line is now `- [<day>] <text>`, and the
day is written like the dated header's: `8 May 2023`. Bullets still start with `- `, so
code that counts lines that way counts the same number. Find your instances by searching
for `RECALL_EPISODE_HEADER` or for comparisons against a rendered turn.

**If you compare a long turn's rendering with its first 280 characters**, compare with
`memvara.retrieve.excerpt.excerpt(text, query, Memvara.RECALL_EPISODE_CHARS)` instead;
with no word shared between text and query it returns the old head cut.

**If you pass `budget=`**, each episode line is longer by its date, 13 to 20 characters,
so a tight budget holds slightly fewer turns. Claims are placed first as before.

**If you use `RemoteMemvara`**, nothing changes with this release: `recall()` there returns
the block the hosted service renders.

---

## A second feature switch is off by default: `agentic_extraction`

### What changed

`agentic_extraction` is a new feature name, off unless
`MEMVARA_FEATURE_AGENTIC_EXTRACTION=1` turns it on, because its release bar has not been
measured (`docs/ROADMAP.md`, the "Reversed" list). Nothing about extraction changes unless
you turn it on. `WriteReceipt` gained two fields, `agentic_fallback` (`None` unless the
switch was on and a batch fell back to one call) and `proposals_refused` (empty unless the
switch was on), and `repr(receipt)` shows each only when it is set.

### Who this changes, and in which direction

**If you compare `FEATURES_OFF_BY_DEFAULT`, `ServerConfig().features_off` or a default
`MemvaraMCPServer`'s `features_off` with a literal set**, it is now
`{"agentic_extraction", "extraction_chunks"}`. Compare with `FEATURES_OFF_BY_DEFAULT`.

**If you read `memory_stats` output**, a server with no feature variables now says
`features switched off: agentic_extraction, extraction_chunks`.

**If you keep a copy of the feature names**, as the plugin hooks do in
`plugin/hooks/lib/settings.py`, add `agentic_extraction` with the default `False`; the
MCP server refuses a `MEMVARA_FEATURE_*` name it does not know.

**If you turn it on**, a write costs several model calls instead of one, and
`receipt.llm_calls` counts every one of them. It also takes longer. `add()`, and so
`memory_add` over MCP, gives the tool loop 25 seconds, because a caller is waiting and an
MCP client gives up on a tool call after a limited time; a loop that runs out falls back
to one extraction call, so a write can take 25 seconds plus that call. `reextract()`,
which a background worker runs, gives the loop 180 seconds. If your client's tool timeout
is shorter than about 40 seconds, either leave the switch off or extract later with
`reextract()`. With a backend that does not implement
`llm.ToolChat`, every write falls back to one call and says `agentic_fallback=unsupported`,
so the switch does nothing but add that line; switch it off again.

**If you implemented a `Reconciler` subclass that overrides `_retire` or `_retract`**,
both now take a `reason` keyword argument, which `apply` passes.

---

## The plugin's capture hook searches your memory before it writes

### What changed

`agentic_capture` is a new feature switch, on by default. It belongs to the plugin: the
MCP server only checks the name. With it on, the capture hook on Claude Code runs the
headless agent command with read-only access to your memory, lets it search up to four
times, and applies the facts, replacements, ends and links it proposes after checking
each one. With it off, capture makes one extraction call per turn, as before.

### Who this changes, and in which direction

**If you keep a copy of the feature names**, as the plugin hooks do in
`plugin/hooks/lib/settings.py`, add `agentic_capture`. The MCP server refuses a
`MEMVARA_FEATURE_<NAME>` it does not know, so an older server refuses
`MEMVARA_FEATURE_AGENTIC_CAPTURE`.

**If you generate the plugin's hooks file**, regenerate it: the capture hook's timeout
on Claude Code is now 180 seconds rather than 120, to cover an agentic run followed by the
single-call extraction when the agentic run fails.

**If you run a hosted organisation with a model key**, the searches the agentic run makes
may each be a rewritten search until the hosted service is updated: a call on the
organisation's key, and about 145 rate-limit units instead of about 66. The run sends a
`Memvara-Read-Stages: plain` header asking for plain reads, and the hosted service does
not read that header yet; the change on the cloud side is separate. A local store is not
affected: the run's local server always starts with `MEMVARA_FEATURE_QUERY_REWRITE=0` and
`MEMVARA_FEATURE_SYNTHESIS=0`, whatever your own configuration says.

**If you watch a hosted plan's recall allowance**, on the hosted service one capture turn
counts as one recall, however many searches it makes. Each run sends a
`Memvara-Capture-Run` header with a new random id, so the service can tell which searches
belong to one run. This takes effect once the hosted service supports the header, which is
a separate change on the cloud side. Until then each of a run's searches, up to four,
counts as a recall.

**If your headless login comes from a settings file** rather than from the normal login,
the agentic run cannot sign in, because it loads no settings files. Every turn then falls
back to the single-call extraction, and `capture.log` says so on each turn. Switch it off
with `MEMVARA_FEATURE_AGENTIC_CAPTURE=0` or `"agentic_capture": false` in
`~/.memvara/settings.json` to skip the failed attempt.

---

## Claims can expire and be erased, stores move to schema 15, and extraction takes guidance

### What changed

Claims have two new fields, `expires_at` and `expire_reason`, which only a caller sets,
with `remember(expires_at=..., expire_reason=...)` or the `memory_remember` tool. Once
`expires_at` passes, `Memvara.erase_expired()` **erases** the claim: the row, its text
index entry and its vector are deleted, an erasure record is written, and a proof is
checked against the disk. It runs when a `Memvara` opens a store and hourly in the MCP
server, but not on a read-only server. Reads leave the claim out from the instant it
expires, before the sweep deletes it. This is the one case in which the engine deletes a
claim on its own, and it reverses the old wording of invariant 3 in `docs/INTERNALS.md`
for that case only. A claim written without `expires_at` is never erased by the engine,
and ending or superseding a claim still deletes nothing.

Opening a store written by an earlier version upgrades it to schema 15, which adds the two
nullable columns and an index. Nothing is backfilled. A store stamped 15 cannot be opened
by an earlier version, which refuses rather than guessing.

`LLM.extract()` takes a new keyword, `guidance`, which appends per-project rules to the
extraction prompt. The MCP server reads them from a TOML file named by
`MEMVARA_EXTRACT_GUIDANCE`.

Two feature switches are new, both on by default: `extraction_guidance` and
`expiry_erasure`.

### Who this changes, and in which direction

**If you implement `Store`**, add `expired_claims(now)` to have expiring claims erased. It
is optional: without it the sweep at open is skipped and `erase_expired()` raises
`NotImplementedError` naming your store. If your store persists `Claim` fields one by one,
persist `expires_at` and `expire_reason` too, or an expiry a caller set is lost.

**If you implement `LLM`**, nothing breaks: the write path passes `guidance=` only to a
backend with `accepts_guidance = True`, and only when guidance is configured. Configuring
guidance for a backend without it raises `TypeError` when the `Memvara` is built. To
support it, take `guidance=` in `extract()` and pass your system message through
`memvara.llm.with_guidance`.

**If you keep a copy of the feature names**, as the plugin hooks do in
`plugin/hooks/lib/settings.py`, add `extraction_guidance` and `expiry_erasure`.

**If you downgrade**, an earlier version refuses a schema 15 store. Keep a copy of the file
from before the upgrade if you may need to roll back. Find your stores with
`grep -rn MEMVARA_DB ~/.claude.json .mcp.json`.

**If you run the MCP server on Python 3.10**, `MEMVARA_EXTRACT_GUIDANCE` makes it refuse to
start, because the file is TOML and Python's TOML reader arrives in 3.11. Everything else
works on 3.10, and a guidance built in Python (`Memvara(write_guidance=Guidance(...))`)
works there too.

**If you use the hosted client**, `remember(expires_at=...)` is sent to the deployment,
and a deployment from before expiry refuses the field with 422.

---

## The MCP server creates new stores encrypted, and needs the `encrypt` extra to do it

### What changed

`encryption` is a new feature switch, on by default. With it on, the MCP server creates a
new store encrypted: the database with SQLCipher, and every row of the vector file with
AES-256-GCM. That needs `pip install 'memvara[encrypt]'`, and without it the server refuses
to start rather than create the store unencrypted. The key is read from the OS keychain
(service `memvara`, account `db-key`), then `MEMVARA_DB_KEY`, then `~/.memvara/db.key`,
where one is generated with mode 0600 if none exists. **A store whose key is lost cannot
be read**, so back the key up with `memvara encrypt --export-key`.

Nothing changes for a store that already exists. It opens as it is, with a warning at
start and a `storage: NOT encrypted` line in `memory_stats`, until you convert it with
`memvara encrypt <path>` (stop every process that uses it first). The library's
`Memvara(path)` and `SQLiteStore(path)` are unchanged; `encryption=True` asks for it there.

Separately, `erase_claim()` and `purge()` now blank a vector's row in `<db>.vecs` even when
the process has not searched yet. They used to leave it, which kept the erased text
recoverable from the vector.

### Who this changes, and in which direction

**If you start the MCP server against a path that does not exist yet**, such as a new
machine or a test, install the extra or set `MEMVARA_FEATURE_ENCRYPTION=0`. The server's
message names both. Find your launch configurations with
`grep -rn MEMVARA_DB ~/.claude.json .mcp.json`.

**If you run the server in a container**, pass `MEMVARA_DB_KEY` from a secret manager. The
image this repository builds installs the extra and sets `HOME=/data`, so without the
variable the key is generated onto the volume beside the store. An image of your own that
leaves `HOME` inside the container loses a generated key with the container, and the
store on the volume cannot be read on the next run.

**If you read the store file with other tools** (the `sqlite3` shell, a backup script
using `.backup`), they cannot open an encrypted store. Stop the server and copy the files,
and keep the key with the backup, stored apart from it.

**If you keep a copy of the feature names**, as the plugin hooks do in
`plugin/hooks/lib/settings.py`, add `encryption`.

**If you run several processes against one encrypted store**, each holds its own
decrypted copy of the vector matrix in memory, where an unencrypted store's memory-mapped
matrix was shared between them.

---

## Seven store methods take a filter, and the two read tools take two new arguments

### What changed

`search()` and `recall()` take `filters` and `filepath_prefix`, which narrow a read by
metadata and by the file path of the document a memory came from. The filter runs inside
the store, so seven `Store` methods take a new keyword argument, `where`, a
`memvara.filters.SearchFilter` or `None`: `candidate_ids`, `lexical_search`,
`vector_search`, `episode_candidate_ids`, `lexical_search_episodes`,
`vector_search_episodes` and `episodes_near`. `memory_search` and `memory_recall` take
the same two arguments, and a new feature switch, `metadata_filters`, is on by default.

### Who this changes, and in which direction

**If you implement `Store` yourself**, nothing breaks for a read that does not filter:
the retriever passes `where` only when the caller filtered. A filtered read against your
store raises `TypeError` naming `where` until you add it. Add it to all seven methods and
apply it in the same query as your limit, or a filtered search returns fewer than `k`
matches while more exist. `SQLiteStore` in `memvara/store/sqlite.py` is the reference:
`_where_clause` builds the condition. Find your implementations with
`grep -rn "def lexical_search" your_package/`.

**If you call a hosted deployment with `RemoteMemvara`**, a call that passes either
argument fails with `InvalidRequest` (HTTP 422) until the deployment is updated, because
its request models refuse a field they do not know. It is never answered unfiltered. A
call without them is unchanged.

**If you compare a tool's accepted arguments**, `memory_search` and `memory_recall` list
`filepath_prefix` and `filters`. With `MEMVARA_FEATURE_METADATA_FILTERS=0` both stay in
the schema with a description saying they are refused, and a call carrying one is
refused with a message naming the switch.

**If you list feature names** (the plugin hooks' `FEATURES`, or a check of
`MEMVARA_FEATURE_*` variables), add `metadata_filters`.

**If you pass a class of your own as `ToolContext.memory`**, its `search` and `recall`
need `filters` and `filepath_prefix`: `MemoryAPI` declares both and the tools pass both
on every call, as `None` unless the model asked for a filter.

---

## A hosted purge with a project bound is refused instead of erasing every project

### What changed

`RemoteMemvara.purge()` and `AsyncRemoteMemvara.purge()`, and their scoped views, used to
send `POST /v1/erasures` with the user, agent and session only. From a client bound to a
project, for example `Memvara(api_key=..., project=...)` or `mem.scope(project=...)`, that
request erased the user's memory in every project, not just the bound one. They now raise
`ValueError` and send nothing while a project is bound, as `RemoteStore.purge()` already
did. The erasure route has no project field yet; memvara-cloud #267 adds one. A client with
no project bound purges exactly as before.

### How you find your instances

**A hosted `purge()` called on a client or view that has a project.** Search your code for
`purge(` on a `RemoteMemvara`, an `AsyncRemoteMemvara` or a scoped view of either. To erase
the user's memory in every project, call it from a client with no project bound, and say
so in the code, because that is what the call does.

---

## A `Memvara` with a chat-capable `llm=` now calls it on every read, and `MemoryAPI` gained `query_rewrite` and `synthesize`

### What changed

**If you configured `llm=` only for extraction, every read now makes a model call too.**
`search()` and `recall()` gained `query_rewrite: bool = True`, and `recall()` gained
`synthesize: bool = False`. When `llm=` is a backend that implements `Chat`, which
`OpenAILLM` and `AnthropicLLM` both do, every `search()` and every `recall()` now sends one
request to that model before retrieving anything, on your key, and waits up to 10 seconds
for it. It then runs up to four retrievals instead of one, and may read the store at a
`valid_at` taken from the question's dates. Before this, such a store called its model
only on writes and on `ranked=True` reads.

What that costs, measured on this change: the extra retrieval work took a read from a
median of 5.6 ms to 21.0 ms (200 reads, `k=10`, turns included, a local store of 1,000
claims and 1,000 turns, `HashingEmbedder`, a model stub that answers instantly, Python
3.13 on macOS). The model call itself comes on top of that, is billed by your provider,
and was not measured here because no key was available; it is at most the 10-second
deadline, after which the plain read is served. `retrieval.rewrite_ms` records it on
every call once you configure `telemetry=`.

To switch it off for every read, pass `query_rewrite=False` to `Memvara(...)`, or set
`MEMVARA_FEATURE_QUERY_REWRITE=0` on a server. To keep one read model-free, pass
`query_rewrite=False` to that call, or `**memvara.select.PLAIN_READ`.

Nothing changes for a store opened with the default `NullLLM`, or with any backend that
cannot chat: no read calls a model, and `.rewrite.outcome` reports `unconfigured`.

`MemoryAPI`, the protocol the MCP tools are written against, declares `query_rewrite` on
`search` and `recall`, and `synthesize` on `recall`. The tools pass both on every call.

### How you find your instances

**A store opened with a model.** Search your code for `Memvara(` with `llm=OpenAILLM`,
`llm=AnthropicLLM`, or a backend of your own that has a `chat` method. If you need reads
to stay model-free, for cost, latency, or a benchmark whose numbers were measured without
a rewrite, pass `query_rewrite=False` to the constructor. On a server, set
`MEMVARA_FEATURE_QUERY_REWRITE=0`.

**A benchmark or test that counts model calls.** A fake backend with a `chat` method now
receives rewrite calls on reads, with the system prompt
`memvara.select.stages.REWRITE_SYSTEM`. Pass `**PLAIN_READ` (or `query_rewrite=False`)
where the count should stay as it was. The benchmark harnesses in `bench/` and the demo
already do.

**A `recall(synthesize=True)` block you parse.** A block without a summary starts with
`(summary not written — <outcome>.)`, and for a fallback the reason follows the outcome,
as in `(summary not written — fallback: timeout.)`.

**An `AsyncMemvara` under heavy read load.** `search()` and `recall()` now run on their
own pool of `memvara.aio.READ_THREADS` (8) threads instead of the loop's default executor,
so a read waiting on a model cannot hold a thread other `to_thread` work needs. A burst of
more than eight reads queues.

**A class of your own used as `ToolContext.memory`.** Search your code for `ToolContext(`
or `MemoryAPI`. Its `search` needs `query_rewrite`, and its `recall` needs `query_rewrite`
and `synthesize`, or the tools raise `TypeError: unexpected keyword argument`.
`ScopedMemvara` and `ScopedRemoteMemvara` already take them.

---

## The schema is version 14, four MCP tools are new, and deleting a document erases its text

### What changed

`SCHEMA_VERSION` moves from 13 to 14. The migration adds two tables: `documents`, one row
per stored document, and `document_chunks`, one row per chunk, each naming the episode
the chunk was stored as. Nothing is copied into them, because no earlier version stored a
document. As with every schema bump, a file opened by this build is refused by an older
build rather than written to. Take a copy first if you may need to go back.

`Episode.hash` now mixes in `meta["document_id"]` when an episode has one. Only document
chunks carry that key, so the hash of every episode you already have is unchanged.

The MCP server serves twenty-two tools instead of eighteen: `memory_add_document`,
`memory_get_document`, `memory_list_documents` and `memory_delete_document` are new, and
the `documents` feature switch hides all four. `memory_delete_document` is the first
tool that erases stored text. What it erases is one document's own chunks; it erases no
memory, and a memory whose only source was the document is retired with the reason
"source document deleted".

### Who this changes, and in which direction

**If you implement `Store` yourself**, nine document methods are new: `put_document`,
`get_document`, `find_document`, `list_documents`, `document_chunks`,
`put_document_chunks`, `delete_document`, `claims_citing_any` and `erase_episodes`. They
are optional as a group, and a store that implements them sets the class attribute
`holds_documents = True`; without it `add_document()` and the methods beside it raise
`NotImplementedError` naming your store, and nothing else changes. An episode a document
still lists must survive `erase_episode` and `erase_claim(sources=True)`, or a document
can lose part of its text behind its own back.

**If you compare `purge()` output against a fixed set of keys**, add `documents` and
`document_chunks`. `erase_claim()` keeps its four keys.

**If you use `SalienceGate` directly**, an episode with `meta["document_id"]` now passes
the role check whatever its role, and one that also has `meta["extract"] = False` is
refused with the reason `document_not_extracted`.

**If you serve MCP to a model with a fixed tool budget**, set
`MEMVARA_FEATURE_DOCUMENTS=0` to keep the list at eighteen.

**If you list a store's episodes**, document chunks appear among them with
`role="system"` and `meta["document_id"]` set. Filter on that key to leave them out.

---

## One feature switch is now off by default: `extraction_chunks`

### What changed

`extraction_chunks` is a new feature name. With it on, a turn over 6,000 characters is sent
to the extraction model in pieces, one call per piece. Unlike every earlier feature it is
off unless `MEMVARA_FEATURE_EXTRACTION_CHUNKS=1` turns it on, because its release bar has
not been met (`docs/ROADMAP.md`, the "Reversed" list). Nothing about extraction changes
unless you turn it on.

### Who this changes, and in which direction

**If you build `ServerConfig` in Python and compare `features_off` with an empty set**, it
is now `{"extraction_chunks"}` by default, the same value `ServerConfig.from_env()` returns
for an environment that sets nothing. Compare with `FEATURES_OFF_BY_DEFAULT` instead.
`MemvaraMCPServer(features_off=...)` defaults to the same set, so a server you build in
Python without that argument lists `extraction_chunks` as off. If you pass `features_off`
yourself, what you pass is the whole set: add `extraction_chunks` to it to keep that
feature reported as off.

**If you read `memory_stats` output**, a server started from the environment with no
feature variables now says `features switched off: extraction_chunks` where it said
`features: all on`.

**If you keep a copy of the feature names**, as the plugin hooks do in
`plugin/hooks/lib/settings.py`, add `extraction_chunks`; the MCP server refuses a
`MEMVARA_FEATURE_*` name it does not know, and a copy without it will not match the library.

---

## The schema is version 13, three MCP tools are new, erasure proofs count a fifth table, and `supersede()` ends where the new value begins

### What changed

`SCHEMA_VERSION` moves from 12 to 13. The migration adds one table, `claim_links`, which
holds typed links between memories (`extends` and `derives`, written by `Memvara.link()`
and the new `memory_link` tool). Nothing is copied into it: an
upgraded store starts with no links, and an empty table means "no links recorded since
the upgrade", not "no memory was ever derived from another". As with every schema bump,
a file opened by this build is refused by an older build rather than written to. Take a
copy first if you may need to go back.

`Store.residue()` on `SQLiteStore` now returns a fifth key, `claim_links`, because erasing
a claim also erases every link that names it, and a proof has to be able to see a link
that survived.

The MCP server serves seventeen tools instead of fourteen: `memory_end_matching`,
`memory_forget_matching` and `memory_link` are new. `memory_end`, `memory_forget` and
`memory_remember` take new optional arguments (`reason`, and on `memory_remember` also
`until_reason` and `replaces`). No existing argument changed meaning.

### Who this changes, and in which direction

**If you implement `Store` yourself**, `put_link()` and `claim_links()` are new protocol
methods. Both are optional: without them `Memvara.link()` raises `NotImplementedError`
naming your store, and `links()` and `why().links` report none. If your store can erase a
claim, delete its links in the same transaction, or `prove_erased()` on your store has no
way to notice a link left behind. `put_link()` returns the row it kept, which is the
earlier one when the same link was already recorded.

**If you compare `residue()` output against a fixed set of keys**, add `claim_links`.

**If you call `supersede()` without `at`**, a `close="ended"` supersession now closes the
old claim at the new claim's `valid_from`, where it used to use the new claim's
`recorded_at`. The two are the same unless you set them apart, which a replay of history
usually does: the old value now ends where the new one began rather than where it was
recorded. `close="retired"` is unchanged. **If you call `supersede()` on a claim that is
already retired, or with `close="ended"` on one that is not live**, it now raises
`ValueError` and writes nothing. Retiring an ended claim still works.

**If you run more than one server process against one store**, set
`MEMVARA_CONFIRM_SECRET` to the same value on every one of them. The matching tools hand
out a confirmation token with each preview, signed with that key. Without it each process
generates its own key, and a preview served by one process is refused when another one
receives the confirmation. A single stdio server needs nothing.

**If a client pins the tool list**, it now sees three more tools. A read-only server
hides all three, as it hides every write tool.

### How to find your instances

Your own stores: `grep -rn "class .*Store" your_code/` and check each for `put_link` and
`claim_links`. Your deployments: any launch configuration that starts more than one
`memvara-mcp` process, or more than one worker, against the same `MEMVARA_DB`. Your
replays: `grep -rn "\.supersede(" your_code/` and check each call that builds the new
claim with a `valid_from` different from its `recorded_at` and passes no `at`.

---

## The local MCP server files memory under the repository it starts in

### What changed

`memvara-mcp` in local mode now binds a project, the fifth part of the scope, unless you
switch it off. When `MEMVARA_PROJECT` is unset, the server works the project out at startup
from the git remote of the directory the client started it in, for example
`github.com/acme/app`. Outside a git repository nothing changes.

With a project bound, a new fact is filed under that repository when its predicate is
project-relative, which is the default for any predicate not declared otherwise. It is then
recalled in that repository and not in another one. A predicate declared global, which
includes every built-in person fact such as `prefers` or `lives_in`, is written without a
project, exactly as before, and is recalled everywhere.

Memories written before this release have no project, so they stay visible from every
repository. A new value for a project-relative predicate that holds one value at a time is
filed in the repository's own slot and does not end the older value written without a
project, because that value is still the answer everywhere else. Inside the repository,
reads return the repository's value and leave the older one out; in every other repository
the older value still answers. If the repository's value is later ended, the older one
answers there again.

The hosted clients are unaffected unless a project is given. `memvara-mcp` in cloud mode
derives the project the same way and sends it as the `Memvara-Project` header; a deployment
that does not read the header yet ignores it.

### How you find your instances

**A local `memvara-mcp` launched from inside a git repository.** Call `memory_stats`: the
scope line reads `tenant/user/project/agent/session`, and a project part other than `*`
means the server bound one. To keep the old behaviour, set
`MEMVARA_FEATURE_PROJECT_SCOPE=0` in the server's environment block. To pin a project
instead of deriving it, set `MEMVARA_PROJECT` to a name such as `github.com/acme/app`; a
value that is not in that form stops the server at startup.

**Any `MEMVARA_FEATURE_*` variable already in your environment.** The server now reads
these and refuses to start on a name it does not know or a value that is not a boolean.

**A call that passes `project=` as metadata.** `remember(..., project="x")` used to store
`project` as an annotation on a claim filed without a project. It now raises `TypeError`,
on the local engine and the hosted clients alike, and the fix is to bind the project with
`Memvara(project=...)` or `mem.scope(project=...)`. Search your code for `project=` on a
`remember(` or `supersede(` call.

---

## `memory_standing` asks the engine, and `MemoryAPI` gained `standing` and `profile`

### What changed

`MemoryAPI`, the protocol the MCP tools are written against, now declares `standing` and
`profile` and no longer declares `get_all`, which the tools stopped calling. A third-party
view passed to `MemvaraMCPServer` through `ToolContext` has to provide the two new
members.

### How you find your instances

**A class of your own used as `ToolContext.memory`.** Search your code for `ToolContext(`
or `MemoryAPI`. The two concrete views that ship, `ScopedMemvara` and
`ScopedRemoteMemvara`, already have both methods.

---

## `--judge-model` is honoured with `--reader openai`, where it used to be ignored

### What changed

The benchmark runners and `demo/harness.py` build the model judge from the reader — same
provider and same pinned settings, with `--judge-model` swapped in. Before, an OpenAI run
that passed `--judge-model` got an Anthropic judge on its own defaults, and the flag had
no effect on it. A run that named a judge model now grades with that model, on the same
provider as the reader.

This is a change in what a command does, not in what it accepts. Nothing starts failing,
but a number produced by a run that passed `--judge-model` beside `--reader openai` was
graded by a different model than the command said, so it is not comparable with the same
command run today.

### How you find your instances

**A script or a recorded run that passes `--judge-model` with `--reader openai`.** Grep
your run logs for a report header naming a judge; the header has always printed the judge
it used, so a stored report says which model actually graded. Re-run anything you intend
to compare against a new number.

---

## A `memvara` console script, and `login --project` takes an id or nothing

### What changed

Installing the package now puts a second command on your `PATH`, `memvara`, with three
subcommands: `login`, `logout` and `whoami`. `memvara login` is the device-code sign-in
`memvara-mcp login` already ran, and `memvara-mcp login` keeps working.

`--project` is now optional on both, and a project *name* is refused. The hosted
console's authorize route has no session, so it cannot look a name up; it takes a project
id or nothing, and it already answered a name with 400 `bad_request`. Leave `--project`
out and choose the project in the browser.

### How you find your instances

**A script or a setup document that runs `memvara-mcp login --project NAME`.** It has been
failing against the hosted console with a 400; it now fails before a browser opens, with a
message saying to drop the flag. Remove `--project NAME`, or pass the project's id.

**A machine with the npm package installed globally.** `npm install -g memvara` installs a
bin named `memvara` as well — the stdio bridge to the hosted MCP server — and whichever of
the two comes first on `PATH` wins. The two `login`s are different programs: the bridge's
writes an OAuth token to `~/.memvara/oauth.json`, this one's writes an API key to
`~/.memvara/credentials.json`. Run `which -a memvara` to see which you have. Use `npx
memvara` for the bridge and `python3 -m memvara` for this command wherever it matters; both
spellings reach the right program whatever `PATH` says. The bridge reads
`~/.memvara/credentials.json` first, so a key from either `login` serves it.

**A second project on one machine.** `memvara login --credentials PATH` writes the key
somewhere other than `~/.memvara/credentials.json`, so the file every other caller reads
still names the project it did. Nothing reads the other file unless you point it there.

---

## A question that says one relation two ways is a lookup again

### What changed

The intent gate counts the distinct predicates a question names, and two of them is a
chain that opens the graph leg. It counted each word on its own. A word that answers to
two predicates once the prepositions are gone — `work` is the content of both `works_at`
and `job_title`'s alias `works_as` — could therefore add a second name from one relation,
and "what company does Ada work at" read as a chain. The count is now the fewest
predicates one greedy pass needs to account for everything the question said, so that
question is one slot
and a lookup.

Separately, `intent.predicate_refs` no longer memoises a registry's vocabulary. The memo
could answer with a vocabulary the registry no longer held after `register()` or
`learn_alias()`, and with a garbage-collected registry's vocabulary when a new registry
reused its `id()`.

### Who this changes, and in which direction

**Only reads with `w_graph > 0` on a store whose vocabulary declares edges.** With the leg
off, nothing changes. With it on, a question that says one relation two ways now skips
the walk, which is what the gate is for, and `Explanation.intent` reports `lookup` where
it reported `relational`. A question that names two relations still opens it: "who
founded the company that Ada works at" names `works_at` and `founded_by`. If you register
predicates after the first search, the gate now sees them.

---

## A `project:<key>` subject keeps `procedural`, and standing sets list stated rules first

### What changed

Since 0.14.0 the rule in the next entry has one exemption: a subject typed `project:<key>`
(`types.PROJECT_SUBJECT_TYPE`, read through `Claim.subject_type` like every typed entity, so
`Project:` folds and `project: some prose` does not count) keeps `procedural`. A
preference that holds only in one checkout is filed under `project:<absolute path>`, stays
a standing rule, and a client that knows its working directory can show it there and
nowhere else. A bare repository name (`memvara`) is not the prefix and still moves.

In the same release `memory_standing` and the plugin's session-start block order every
row a caller asserted (`extractor` `""` or `"api"`) before every row a model or a hook
derived, and only then by confidence, recency and id. Confidence used to come first, and a
model's own confidence in its paraphrase outranked the sentence the user typed.

### How you find your instances

A `project:`-scoped claim that was retyped by 0.13.0 stays `semantic` until the same triple
is written again; `memory_history` shows `retyped_from`. Re-assert the triple with
`memory_type="procedural"` to move it back. Standing sets that a client truncates by `k`
change composition, not size: the stated rows now survive a cut.

---

## `procedural` is for the subject `user` only

### What changed

A claim about any subject other than `user` that arrives as `procedural` is filed as
`semantic`. It does not matter where the type came from: `remember(memory_type=...)`, a
model's extraction, or a predicate declared `procedural` because it is usually about the
user (`prefers_tool`, `never_do`). The receipt says it happened: `retyped` carries a
`Retype` whose `reason` is `"subject"`, and the MCP write tools print a note naming the
claim and the rule.

`procedural` means how the user wants work done. It is the population `memory_standing`
returns and nothing else, and clients inject that set at the top of every session, so a
project fact filed there is carried on every turn of every later conversation. A
repository, a service or a file cannot want anything, so such a filing was wrong however it
was produced; this release makes the store say so instead of relying on every caller and
every prompt to know it.

### Who this changes, and in which direction

**If every `procedural` claim you write is about `user`, nothing changes.** Verbatim notes
(the `note` predicate: the mem0-compatible `infer=False` path and the importer) written
with `memory_type="procedural"` keep their type too, and keep it when restated: a note is
the owner's own text, not a claim about a thing.

**If you wrote `procedural` claims about other subjects, they move.** A new one is filed as
`semantic` at write. One already on record moves the next time the same triple is seen,
even by a write that asserts no type; the safety property that an unopinionated write
cannot undo a deliberate correction still holds, because moving such a claim out of
`procedural` is never a correction anyone could have wanted to keep. Nothing else about the
claim changes: not its confidence, its sources, its `derivation` or its `extractor`. Until
a claim is seen again it stays where it is, so a store that wants the move now re-asserts
the triples, and `memory_standing` shrinks to the user's own instructions as they go.

**If you asserted `procedural` for another subject on purpose**, for example to keep a
team rule under a project subject in the standing set, that no longer works and there is no
switch for it. File it under `user` (`user / never_do / git add -A`), which is where a
standing instruction belongs whichever repository it is about.

## An entity can declare what kind of thing it is

### What changed

A subject or object may carry a `type:` namespace, and the namespace is part of the
entity's identity:

```python
mem.remember("company:apple", "founded_in", "Cupertino")
mem.remember("fruit:apple", "grows_in", "orchards")
```

Those are two entities. Neither is the bare `apple`. A fact about one never retires a fact
about the other, and a graph walk does not cross between them.

The namespace is optional and nothing has to adopt it. A surface form without one is an
entity of no declared type, which is an identity of its own rather than a wildcard.

### Who this changes, and in which direction

**If you never write a colon in a subject or object, nothing changes.** Ordinary text that
happens to contain one is not read as a namespace: a URL is excluded by the two slashes
after the colon, prose by the space after it, and a time by a namespace having to begin
with a letter. Everything else keys exactly as it did.

**If you already write `type:name` subjects, your keys change and the migration fixes
them.** `company:apple` used to fold to `company apple`, treating the namespace as a word
of the name; it now folds to `company:apple`. The version 12 migration re-derives every
stored key from the text you wrote and rehashes both `fact_key` and `value_key` from the
new keys, so nothing is left addressing a slot no one else computes.

**Two folds behave differently now, and both are narrower than before.** Stripping a
corporate form is confined to one namespace, so `company:Apple Inc.` and `company:apple`
are one entity while `company:apple` and `fruit:apple` can never reach each other. And
`EntityRegistry.learn_alias` raises `ValueError` if the two surface forms have different
namespaces, where it previously performed the merge. If you call it in a loop over
untrusted pairs, catch that. A surface form that folds to nothing at all — `"..."`, a bare
emoji — is still the silent no-op it always was, rather than a namespace mismatch: there is
no entity there to have a namespace.

### How to find your instances

Claims whose ends carry a namespace, after upgrading:

```sql
SELECT subject_type, object_type, count(*) FROM claims
WHERE subject_type != '' OR object_type != '' GROUP BY 1, 2;
```

---

## A claim can belong to a repository, and reads no longer cross between them

### What changed

`Scope` gains a fifth element, `project`, holding a `host/owner/repo` identity, and you pass
it once when you open the store:

```python
mem = Memvara("memory.db", user="alice", project="github.com/you/repo")
```

The project is mixed into the slot key, so two repositories that both record
`postgresql version` hold two separate facts instead of one that keeps overwriting itself.
Reads are filtered by it as well: a handle opened on one project does not enumerate another
project's claims.

Not everything partitions. `PredicateSpec` gains `project_scoped`, and a predicate declared
`project_scoped = false` has its claims written with the project cleared, so they sit at user
level and stay visible from inside every project. All 23 builtin predicates are declared
global, so a personal assistant's memory behaves exactly as it did.

`SCHEMA_VERSION` moves from 11 to 12. It adds a nullable `project` column to `claims` and to
`episodes`, and then rehashes every `fact_key` in `claims`, because the hash takes a fifth
input now.

### Who this changes, and in which direction

**If you never pass `project`, nothing changes.** Every scope has `project=None`, every claim
is written with it unset, and the read filter matches. The rehash changes the bytes of your
keys but not which claims compete for a slot, so contradiction handling behaves identically
before and after.

**If you start passing `project`, older claims do not move into it.** Claims written before
the upgrade keep `project` NULL, which means "not recorded against a repository". Nothing
infers one for them, because choosing a repository for a fact after the event would be
inventing where it was learned. A handle opened with `project=` still sees them, because
visibility widens upward from a project to that user's project-less claims — but a *new*
claim on the same subject and predicate lands in a different slot and will not supersede the
old one. If you want the old fact superseded, write the new one with `project=None`, or
declare its predicate `project_scoped = false`.

**If your own vocabulary declares predicates, decide which of them partition.** An undeclared
predicate partitions by project, which is the safe direction but not always the one you want.
A predicate that records something about the person rather than the codebase should be
declared global:

```toml
[[predicate]]
name = "prefers"
cardinality = "many"
project_scoped = false
```

### How to find your instances

Claims written before the upgrade, and therefore not in any project:

```sql
SELECT count(*) FROM claims WHERE project IS NULL;
```

Predicates your vocabulary leaves partitioned, which is every one that does not say otherwise:
grep your pack files for `project_scoped` and assume `true` wherever it is absent.

---

## Graph traversal now follows only declared relations

### What changed

A claim's object carries an `ObjectKind` — `ENTITY` or `VALUE` — set at write time from the
predicate's declared `object_type`. Only an entity object can be one end of a graph edge.
A predicate nobody has declared produces a value, so it carries no edge.

`SCHEMA_VERSION` moves from 10 to 11, adding one nullable column to `claims`.

### Who this changes, and in which direction

**Read this one if you use `neighborhood()`, `paths_between()`, or the graph leg of
`search()`.** It is the change in this release that can quietly do less than it did before.

**Claims you already have are unaffected.** They keep `object_kind IS NULL`, which means
"written before this rule", and both the traversal gate and the walker admit them. Nothing
backfills the column: the answer depends on which vocabulary a deployment loads rather than
on anything in the row, so a backfill would make two machines disagree about one file. Your
existing edges survive the upgrade.

**Claims written after the upgrade need a vocabulary.** If you have not declared
`object_type` for a predicate, its new claims are value-valued and carry no edge. A store
that walks today and is still being written to will therefore see its graph thin out over
time rather than break at once, which is the more confusing failure of the two — so if you
rely on traversal, declare your relations now:

```toml
[[predicate]]
name = "depends_on"
cardinality = "many"
volatility = "slow"
object_type = ["software"]
graph = true
```

**This is deliberate and it is the point.** Connectivity used to be whatever string
collisions produced, which is how a version number came to be connected to an age.
Connectivity is now something you declare and can measure.

**Watch out when declaring a predicate that already exists.** A declared spec *replaces* a
builtin of the same name rather than extending it, so `PredicateSpec("lives_in",
object_type=("place",))` silently drops that builtin's `ONE` cardinality and its aliases.
Repeat what you want to keep, or build the new spec with `dataclasses.replace` from the
builtin.

---

## Predicates can declare their graph behaviour

### What changed

`PredicateSpec` gains six declaration-only fields: `subject_type`, `object_type`, `graph`,
`inverse`, `inverse_cardinality` and `traversal_cost`. A TOML vocabulary can set all six,
and `PredicateSpec.objects_are_entities` reads `object_type` to say whether this
predicate's objects name things or are scalars.

`SCHEMA_VERSION` moves from 9 to 10, adding six columns to the `predicates` table. An older
file upgrades in place on open.

### Who this changes, and in which direction

**Nothing about an existing store's behaviour changes when you add these fields alone.**
The defaults mean "takes values, walks nowhere", which is exactly what every predicate
meant before they existed. No claim is touched and no migration backfills anything — these
are declared by a vocabulary, not derived from data, so there is nothing to derive.

`object_type` and `graph` are read by the object-kind rule described in the entry above:
a claim can carry a graph edge only when its predicate declares entity objects **and**
declares the relation walkable. Declaring an entity `object_type` without `graph = true`
is legitimate and means "these objects are things, but walking this relation does not help
answer anything" — `mentions` is the example. The remaining four fields are declared and
not yet read.

**Your existing packs keep loading.** Every new key is optional. The shipped `engineering`,
`decisions` and `events` packs declare none of them.

**A pack with a key this version does not recognise now fails to load**, where it used to
be ignored. That is the one behaviour change and it is deliberate: a pack is read once, at
startup, by nobody, so a misspelled key is a declaration that silently does nothing. If a
vocabulary of yours carried an extra key as a note to a reader, move it to a `#` comment.

**A build older than this cannot open a file this one has upgraded.** The usual one-way
schema door; the store refuses rather than corrupting. Take a copy first if you may need to
roll back.

---

## `MemoryAPI.recall` gains `valid_at`, the world clock

### What changed

`recall()` takes a keyword-only `valid_at: datetime | None = None`. It reads the block as
things were on that day, as far as we know today, under a header that names the day.
`MemoryAPI.recall`, the protocol `server/tools.py` is typed against, declares it, and the
`memory_recall` handler passes it on every call, as `None` when the model sent none.

### Who this changes, and in which direction

**Nothing changes for a caller of `Memvara.recall()` that never passes it.** The default
is `None`, and a call without it renders exactly what it rendered before.

**If you implemented `MemoryAPI` yourself, `recall` now takes `valid_at`.** An
implementation written against the previous protocol raises `TypeError: unexpected
keyword argument 'valid_at'` on the first `memory_recall` call. Accept the keyword. If
your backend cannot read at a past day, raise `ValueError` when the value is not `None`
rather than ignoring it: a block about the present returned for a question about the
past is wrong with nothing in the output that says so. `RemoteMemvara.recall` does
exactly that, because `POST /v1/recall` has no time axis.

### How to find your own instances

```bash
grep -rn "def recall" --include="*.py" . | grep -v "memvara/"
```

---

## A truncated model answer now fails the write instead of extracting nothing

### What changed

`OpenAILLM` and `AnthropicLLM` now read the provider's reason for stopping and raise
`memvara.llm.TruncatedResponse` when the model ran out of token budget mid-answer. Before,
neither backend read that field at all.

### What you will see

Writes that used to come back with no claims now come back marked `deferred`, if and only
if the model was cut off. `WriteReceipt.unextracted` counted those turns before and still
does; what is new is `deferred=True` beside it, which is the field that separates "the
model was cut off" from "this turn held no facts". Both looked the same before.

A caller that uses `Memvara.add()` sees no exception — `WritePipeline` catches it, exactly
as it already catches a provider timeout. A caller that invokes `OpenAILLM.extract()` or
`AnthropicLLM.extract()` directly now gets `TruncatedResponse` where it used to get `[]`.
`bench/` scripts and any harness of your own that calls a backend straight are the cases to
check.

`import_mem0(..., extract=True)` is the one such caller inside memvara, and it gained the
same guard in this change: a chunk whose extraction call fails is counted in the new
`ImportReceipt.unextracted` and the import carries on to the next chunk. Without that, a
single truncated answer part-way through a long history would have ended the whole import
and returned no receipt at all. If you see `unextracted` above zero after an import, the
notes are stored and those turns still need phase 2 run over them.

The tokens a truncated call burned are still reported on the receipt. The check runs after
the usage is recorded, so a truncation stays visible on the bill.

### How to find your own instances

If writes start reporting `deferred` after upgrading, the model is hitting its budget and
was hitting it before — you were just not being told. Two things fix it, and they pull in
opposite directions:

```python
OpenAILLM(model="...", max_claims=12)   # ask for a shorter answer
OpenAILLM(model="...", max_tokens=16384)  # give it more room
```

`MEMVARA_LLM_MAX_CLAIMS` is the server-side name of the first. Prefer it: an unbounded
claims array is what produces a runaway in the first place, because a grammar has no legal
way to end a response that keeps restating itself. Raising `max_tokens` buys room for a
longer answer and does nothing about a model that will not stop.

---

## Entity keys are bounded at 512 characters

### What changed

`entity_key` never returns more than `ENTITY_KEY_MAX` (512) characters. A longer key is cut
to the words that fit and finished with a 16-character digest of the whole key. Keys that
fit under the bound are returned exactly as before, so no existing entity id or `fact_key`
moves.

### What you will see

Nothing, unless a store already holds an entity whose folded key is longer than 512
characters. That is a value of a few kilobytes used as a subject or object, and on the
hosted store such a write failed outright, so a store there cannot hold one. A SQLite store
can. The next write of that value gets the bounded key, which is a new identity: the claim
is stored as a new row beside the old one instead of reinforcing it, and the old row keeps
answering under its old key. Nothing warns at write time.

### How to find your own instances

Scan the `entities` table for ids whose key part is longer than 512 characters, before
upgrading:

```sql
SELECT id FROM entities WHERE length(id) - instr(id, char(31)) > 512;
```

An empty result means this entry does not apply to you. A row in it is a value that will
get a new identity on its next write; re-observe it once after upgrading if you want the
history to continue under the new key, or leave both rows if the old value will not be
written again.

---

## Model-proposed claims now pass a pollution guard, on by default

### What changed

`WritePipeline` gained `reject_polluted`, defaulting to `True`. It sits beside
`reject_ungrounded` and catches what that guard says it cannot: a **real** value filed under
a slot it does not belong to. Measured on a small self-hosted model given the predicate
vocabulary — `gate / lives_in / "Port 61434"` beside `endpoint` and `build_status` for the
same value, one found fact forced into every available slot. `Port 61434` is in the turn,
so grounding passes; `lives_in` is ONE-cardinality, so storing it would have ended the true
fact there.

Two rules refuse and one discounts. Within one turn, a (subject, object) pair under
several predicates keeps every predicate the registry knows and drops the unknown ones
beside them, or the first when it knows none. `lives_in`, `born_in` or `located_now` with a
digit or URL in the object is refused, on any subject. And a claim under a
novel predicate, or under a ONE-slot builtin (other than `born_on` and `timezone`) with a
digit or URL in it, is stored at `min(confidence, 0.4)`, which the reconciler's half rule
stores beside an incumbent rather than over it. Refusals count on the new
`WriteReceipt.polluted`; the discount is not counted.

### What you will see

On a frontier model, almost nothing: the fixture that measured this is a 3.8B model's
output, and the rules are shaped to its failure. `polluted` in a receipt is the number to
watch. If it is non-zero on a model you trust, the claim most likely refused is an
invented predicate beside a known one for the same value in the same turn — `employer_of_
record` next to `works_at: Acme` — which the reconciler would otherwise have stored as a
second slot that can never contradict the first. Facts about a named third party are not
refused: `alice / lives_in / Porto` is Alice's own slot.

### If you want the old behaviour

`Memvara(write_reject_polluted=False, ...)` restores it exactly. `remember()` and the fast
path never passed through this guard, so nothing a caller asserts directly has changed.
`docs/INTERNALS.md` and `memvara/write/pollution.py` carry the rules and the measurement;
`tests/test_pollution.py` holds the numbers.

---

## `MemoryAPI.recall` gains `ranked`, an opt-in model-ranked read mode

### What changed

`search()` and `recall()` gain a keyword-only `ranked: bool = False` — see
`memvara.select` and each method's docstring for what it does. `MemoryAPI.recall` (the
protocol `server/tools.py` is typed against) declares it; `MemoryAPI.search` does not,
because `memory_search` pins `include_episodes` to `False`, which `ranked=True` refuses
outright. `Explanation` gains `selected` and `span`; `RecallResult` gains `selection`;
`search()` now returns a `SearchResults` (a `list` subclass with one extra attribute,
`.selection`) rather than a plain `list` — every existing caller that indexes, iterates
or serializes the result is unaffected.

### Who this changes, and in which direction

**Nothing changes for a caller that never sets `ranked=True` or `read_selector=`.** The
default is `False`, no `read_selector` is configured unless you pass one, and
`SearchResults.selection` is `None` on every plain read — the same silence `anchored`'s
addition kept.

**If you implemented `MemoryAPI` yourself, `recall` now takes `ranked: bool = False`.**
The `memory_recall` handler passes it on every call, so an implementation written
against the previous protocol raises `TypeError: unexpected keyword argument 'ranked'`
on the first call. Accept the keyword; an implementation with nothing to rank should
raise `ValueError` if it is ever passed `True` rather than silently ignoring it, the same
rule `anchored` set.

**If you pattern-matched `search()`'s return value as `isinstance(x, list)` and stopped
there, nothing changes** — `SearchResults` is one. If you compared it to a list literal
with `==`, that still works too; only `repr()` differs if you relied on it printing
exactly `[]`, which it still does, since `SearchResults` uses the plain `list` repr.

### How to find your own instances

```bash
grep -rn "def search\|def recall" --include="*.py" . | grep -v "memvara/"
```

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
login`, a device-code flow against the console at `https://console.aurora-notes.dev`, and the
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
