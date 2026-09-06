# Render a turn's date into the text the retriever ranks

**Status: built, measured, and reverted on 2026-09-06.** The design below was implemented in
full and did not pay on LongMemEval-S. Overall evidence R@12 moved +0.5 with the reranker off
and −0.3 to −0.7 with it on, and dating the index *reduced* what the cross-encoder was worth
on temporal-reasoning from +7.1 to +5.8 (two-token form) and +4.3 (eight-token form) — a
dose-response in prefix length, which is the BM25 length-normalisation cost predicted in "The
cost, measured before the decision" below. The code was reverted; the numbers are in
[`docs/BENCHMARKS.md`](../../BENCHMARKS.md#dating-the-episode-index-which-did-not-pay) and the
reasoning in [`docs/ROADMAP.md`](../../ROADMAP.md).

**The document is kept as written**, including the four decisions taken at design time
(episodes only, the long date form, the string reaching the reranker but not the caller, and
the fingerprint guard), because the measurement only means something next to the argument it
tested. Two things it got wrong are corrected in place below: §2's `write/pipeline.py` row,
and the claim that `as_utc` alone renders in UTC.

**What survives the revert:** `--rerank` on `bench/longmemeval.py`, which had to be built to
run the arms, and the first measurement of the `--share-store` episode-retrieval baseline.

This is item 1 of "What is worth measuring here, in order" in
[`docs/ROADMAP.md`](../../ROADMAP.md), which the Hindsight paper prompted. It is the first of
those six to be built.

## The problem

Nothing that ranks on text can see time. `episodes_fts` is declared
`fts5(episode_id UNINDEXED, content, tokenize='porter unicode61')`, the episode vector is
`embedder.encode([ep.content])`, and `rerank/stage.py` scores `item.text`, which for an
episode is `content`. The turn's date lives in `episodes.ts`, a column — reachable by a SQL
filter, invisible to every leg that reads words.

So when LongMemEval asks what happened in June 2023, BM25 is matching "June" against turn
text that never contains it, and the cross-encoder is being asked whether an undated passage
answers a dated question. The date is in the store. It is not in the part of the store that
ranking reads.

The tokenizer confirms an ISO date alone cannot close the gap. Measured on 2026-09-06 against
SQLite's own `porter unicode61`:

| indexed | query `june` | query `"june 2023"` | query `2023` |
|---|---|---|---|
| `2023-06-15 I switched to the Pro plan.` | no match | no match | match |
| `Thursday, 15 June 2023 I switched…` | match | match | match |

`unicode61` splits `2023-06-15` into `2023`, `06` and `15`. The month name is not optional.

## The cost, measured before the decision

A date prefix is not free for BM25, because length normalisation penalises longer documents.
Measured on 2026-09-06 over a synthetic ten-turn corpus, query `pro plan`, one short match and
one long match (SQLite's `bm25()` returns negative scores; lower is better):

| indexed form | short match | long match | separation |
|---|---:|---:|---:|
| bare content | −2.2828 | −1.2029 | 1.08 |
| `June 2023 ` prefix | −2.3269 | −1.4040 | 0.92 |
| `Thursday, 15 June 2023 (2023-06-15) ` prefix | −2.3753 | −1.7043 | 0.67 |

Ranking order held in all three, but the separation between a strong and a weak match
compressed by **38%** at the long form. Ten documents is not a corpus and this is a direction
rather than a magnitude — what it establishes is that the change has a cost on every query,
not only on dated ones, and that the cost has to be read off the real benchmark rather than
assumed away. §6 turns that into the gate.

The user chose the long form with this table in front of them. The format is therefore
settled, and the measurement in §6 is what would reopen it.

## 1. The rendering

One property, one definition, on `Episode` in `memvara/types.py`:

```python
@property
def indexed_text(self) -> str:
    """`content` with the turn's date in front of it, for the legs that rank on words."""
    # "Thursday, 15 June 2023 (2023-06-15)\n<content>"
```

It sits beside `Episode.text`, which aliases `content`, so a reader meets both at once and
the docstring can state the distinction: **`.text` is what was said, `.indexed_text` is what
is searched.** Nothing renames `.text`; it is documented as an alias of `content` and callers
writing audit output rely on it.

Two constraints the implementation has to meet.

**Rendered in UTC**, from `as_utc(self.ts)`. The store's own clock discipline is UTC
everywhere else and an index that varied with the writer's timezone would put two spellings
of one day into one file.

**Month and weekday names come from a table in the module, never from `strftime`.** `%A` and
`%B` are locale-dependent: two machines writing to one store under different `LC_TIME`
settings would index two different languages, and nothing would report it. This is the kind
of defect the repository's telemetry module exists for — silent, and invisible in the data
afterwards.

Invariant 1 holds: this is a pure function of stored state with no model on any path.
Determinism holds: one `ts` yields one string, on every platform.

## 2. The call sites

| site | change |
|---|---|
| `store/sqlite.py:1746`, `add_episode` | index `indexed_text` rather than `content` |
| `core.py:1208` | encode `indexed_text` |
| `write/pipeline.py:616` | **unchanged** — see below |
| `core.py:3112`, `reembed` | `lambda e: e.indexed_text` |
| `store/sqlite.py:1241`, `_migrate_to_v3` | unchanged; v10 runs after it and rebuilds everything |

`episodes.content` is never written differently. `get_episode()`, `why()`, `Result.text` and
`EpisodeResult.text` all keep returning the raw turn, so no caller's output changes.

**Correction, made while implementing this: `write/pipeline.py:616` keeps encoding
`content`, so it is not one of the sites that changed.** Those vectors are not stored — they
are compared against *claim* vectors in `_tier0_near_dupes`, and a claim renders without a
date. Dating one side of that comparison moves the two apart. Measured 2026-09-06 with the
shipped `HashingEmbedder(dim=512)` against a `near_dup_threshold` of 0.97: turn text
identical to the claim's own rendering scores cosine 1.0000 from `content` and 0.5243 from
`indexed_text`; a realistic restatement — "I live in Berlin." against `user lives_in
Berlin` — goes 0.5990 to 0.2544. The prefix does not degrade tier 0, it switches it off,
and the only symptom would be `WriteReceipt.llm_calls` no longer being zero. The comment at
that line records the measurement.

## 3. The reranker sees it; the caller does not

Hindsight puts the same string into the reranker's input, and the cross-encoder is the
component here best able to use a date — it is the one measured at +4.5 R@12 and +14.4 R@1 in
[`docs/ROADMAP.md`](../../ROADMAP.md). Leaving it reading undated text would build half the
item.

`rerank()` gains an optional accessor rather than a widened protocol:

```python
def rerank(reranker, query, items, *, top_n, document=None) -> list[R]:
    ...
    docs = [(document or _text_of)(item) for item in head]
```

Default behaviour is unchanged, the existing doctest passes untouched, and a third-party item
exposing only `.text` keeps working. The two call sites in `retrieve/hybrid.py` — line 774
over mixed `Result`/`EpisodeResult` rows, line 1367 over `Episode` rows — pass an accessor
returning `indexed_text` for episode-shaped rows and `.text` for claims.

**The dated string stops there.** It does not become `Result.text` and does not reach the
reader's prompt. That was a deliberate choice against the larger change: letting the reader
see dates at the same time as the index would confound the judged MemoryBench run, because a
delta could then come from retrieval or from the prompt with no way to separate them. Claims
are untouched for the reason in §7.

## 4. The migration — schema version 10

`SCHEMA_VERSION` moves 9 → 10. `_migrate_to_v10` **drops and recreates `episodes_fts`**, then
repopulates it in one `INSERT ... SELECT` over `episodes`, with the renderer registered on the
connection through `create_function` so month names are computed in Python while the loop
stays in SQLite. Rowids are carried across from `episodes.rowid`, which is what `add_episode`
mirrors and what deletion depends on.

**Dropping rather than rewriting row by row is the point, not an optimisation.**
`_migrate_to_v7` turned on FTS5's `secure-delete` for the text indexes, so a delete rewrites
doclists inside existing segment pages rather than appending a marker; enough of those inside
one uncommitted transaction raises `SQLITE_CORRUPT_VTAB`, which surfaces as
`database disk image is malformed` from a write that is changing nothing. The comment above
`put_claim`'s FTS write records that failure and the store that reproduced it — 19,420 of
19,420 rewrites in the first transaction. A migration that rewrote every episode row would be
that exact shape. A freshly created table has no doclists to rewrite, so the hazard does not
arise and there is no batch size to get wrong.

Shape-driven and idempotent like every migration here: a new database arrives at version 0
with an empty `episodes` table and the repopulate does nothing.

**Vectors are not migrated.** The store has no embedder and cannot encode. Existing stores
need `reembed()`, which exists for exactly this and re-encodes episodes in the same pass.

## 5. The benchmark loader

`bench/longmemeval.py` substitutes `2023-01-01` for a session whose date will not parse
(`parse_instance`, line 222) and counts them in `Instance.undated`. Under this change that
fallback would be rendered into the text of every undated turn, so a question mentioning
January would match a cluster of turns the dataset never dated — a gain that would look real
and would not be.

The loader keeps the fallback for ordering, which is what it is for, and marks those sessions
so nothing is rendered for their turns. The honesty goes where the knowledge is: the library
renders whatever `ts` holds, because `ts` is already the turn's time for `as_of` filtering,
ordering and the belief floor, and a caller who stamps a wrong `ts` is already being answered
wrongly by time travel. Rendering it into text creates no new lie. Inventing one in a
benchmark loader would.

## 6. Success criteria

Not a green suite. The claim this change makes is about a benchmark row, so the benchmark is
the gate.

**Four runs** of `bench/longmemeval.py --score retrieval --share-store`: before and after,
each with and without the cross-encoder. Reported **per category, every category** — not the
headline, and not temporal-reasoning alone.

1. **temporal-reasoning must rise** from its current 66.6 R@12. That is the claim being made.
2. **No other category may fall.** This is the gate for the dilution measured in the table
   above. Single-session-user, multi-session and knowledge-update are where a 38% compression
   of BM25 separation would show up, and if one of them drops, the date format is the first
   thing to revisit rather than the last.

Report the numbers themselves, not "the gate passes". A latency or accuracy claim measured on
one surface and generalised to the others is the specific failure this project has already
paid for.

Unit tests alongside:

- the rendered string is byte-identical under a changed `LC_TIME`, which is the locale trap in
  §1;
- `_migrate_to_v10` rebuilds the index of a store written at version 9, and running it twice
  changes nothing;
- `reembed()` encodes the dated text;
- `get_episode()`, `Result.text` and `EpisodeResult.text` still return the raw turn;
- `rerank()` with no `document` argument behaves exactly as before.

## 7. The half-migrated store, and why it gets a guard

The migration rebuilds the text index and cannot rebuild the vectors. A store that upgrades
and never runs `reembed()` therefore holds a **dated FTS index and undated vectors**: the
lexical leg and the vector leg disagree about what is in the store, results degrade, and
nothing anywhere reports it.

`embed/fingerprint.py` already records who owns a store's vector space in a JSON sidecar. It
gains an `episode_text` version integer, read with a `.get()` default so an old sidecar stays
readable and a new one stays readable by an older build. A store whose recorded version is
behind the running build's is the half-migrated case, and it warns.

This is a fifth thing in a change that has four, and Karpathy §2 argues against it. It is
included because the failure it catches is silent, and this repository's telemetry module
exists because a red-team review classified six of eleven long-horizon failure modes that
way. A degraded index that reports nothing is worse than one that refuses to open.

## What this does not do, and why

**Claims are not touched.** `Claim.render()` keeps returning subject, predicate and object with
no date. Rendering `valid_from` into it is the same idea on the other unit and it is real work,
but it belongs in its own piece for two reasons that were checked rather than assumed:

- `consolidate/merge.py` depends on claim text twice. `_blocking_key` is
  `" ".join(claim.text.split()).casefold()`, which decides which claims are compared at all,
  and `_unit_vectors` compares embeddings of that same text at a **0.97** threshold. Two
  observations of one fact rendered at different precisions — `when.resolve` returns a
  precision precisely so that "last month" is not sharpened into a day — would sort apart and
  fall under the threshold. Consolidation would quietly stop folding duplicates, and nothing
  goes red when a merge does not happen.
- No instrument in this repository could measure the gain. The corpora that produce judged
  numbers barely extract: LOCOMO yields 0 claims from 5,882 turns, LongMemEval 78 from 10,866.

So the claim side waits until the episode side has produced a number, at which point the
precision question and the merge hazard get designed rather than carried along.

## Noticed and deliberately not changed

Both belong to somebody's later work, per Karpathy §3.

- **`add_episode` rewrites its FTS row unconditionally**, where `put_claim` skips the rewrite
  when the text has not moved — the same hazard class, and this change makes each rewrite
  slightly larger. Pre-existing and out of scope.
- **`bench/locomo.py` has the fallback §5 removes from LongMemEval, and it is still there.**
  `parse_sample` substitutes a synthetic monotonic date for a session whose
  `session_N_date_time` will not parse and counts it in `Sample.undated`, so under this
  change that invented date would be rendered into the text of every one of those turns —
  the same defect, on the other loader. It changes no number today: the file's own docstring
  records that all 272 session timestamps parse, and that the counter exists for the file
  changing under us. It needs the treatment `Instance.written` gives LongMemEval, and it did
  not get it here because that is a second loader's ingest, exclusion and reporting path with
  its own tests, and this spec was approved naming one loader.
- **`docs/ROADMAP.md` says the intent gate's relational vocabulary is a hand-written list and
  that deriving it from the predicate registry is "Not done, deliberately".** That is stale:
  `retrieve/intent.py` has `predicate_refs()` and `observed_refs()`, they fold on `word_stem`,
  and `retrieve/hybrid.py` calls them at lines 799 and 961. The roadmap's status header is
  also still `0.8.0` with 3,593 tests. Correcting both is its own commit.

## Order of work

1. `Episode.indexed_text`, with the locale test. Nothing reads it yet.
2. The four index call sites and the v10 migration, with the migration tests.
3. The `rerank()` accessor and the two `hybrid.py` call sites.
4. The fingerprint version and its warning.
5. The `bench/longmemeval.py` undated fix.
6. The four benchmark runs, reported per category, into `docs/BENCHMARKS.md`.

**Status, 2026-09-06:** steps 1 to 5 are built, with unit tests, and the documentation
named below shipped with them. **Step 6 has not been run.** No number in this document has
been confirmed against the benchmark, and `docs/BENCHMARKS.md` carries nothing from this
change. The claim in §6 — that temporal-reasoning rises and no other category falls — is
therefore still a claim, and the format stays open until it is measured.

Documentation ships in the same commit as the code it describes:
`CHANGELOG.md` for every user-visible change, `docs/UPGRADING.md` for the version bump and the
`reembed()` requirement, `docs/INTERNALS.md` where it describes what the episode index holds,
and `docs/BENCHMARKS.md` for the numbers from step 6.
