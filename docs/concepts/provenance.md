# Provenance

**Every claim traces back to the text it came from, and every source turn traces forward
to what it produced.** When an agent says something wrong, those two directions are how
you find out which memory caused it and where that memory came from.

```python
p = mem.why(claim_id)

[e.text for e in p.episodes]
# ["Decision: migrate service-to-service auth from API keys to OAuth 2.0 ..."]

p.derivation, p.extractor
# (<Derivation.USER: 'user'>, 'api')

[c.text for c in p.superseded]
# ['checkout-service auth strategy API keys']
```

`Provenance` carries the claim, the source episodes, how it was derived, which extractor
wrote it, and what it displaced. Backwards:

```python
[c.text for c in mem.produced(episode_id)]
```

## Writing the link

Provenance is not inferred afterwards — you supply it at write time, and it costs one
extra argument:

```python
turn = mem.add("Decision: migrate auth to OAuth 2.0 client credentials.",
               role="system", ts=jun)
mem.remember("checkout-service", "auth_strategy", "OAuth 2.0 client credentials",
             sources=turn.episode_ids, valid_from=jun, recorded_at=jun)
```

`add()` stores the turn and returns its ids on the receipt. `remember(sources=…)` points
the claim at them. When the extractor writes a claim out of a turn it does this itself,
so claims that came from `add()` alone are already linked.

**`role=` decides what is extracted, not just who is credited.** Only `"user"` turns are
read by the extractor; `"system"` stores and cites without extracting, which is what you
want for a document, a log or a paste. The deterministic matcher strips quotation marks
before it looks, so a first-person sentence quoted inside a log is otherwise written down
as a fact about whoever pasted it — at a confidence above what they stated themselves.

## Why it is a different claim from "we keep the source text"

Plenty of stores keep the source document. What `why()` adds is the third field:

```python
p.superseded
```

The claim this one **replaced**. That is the field that turns a note into a record — it
is how *"we decided X"* becomes *"we decided X on 12 June, replacing Y, on this
evidence"*, which is the sentence an incident review or a design discussion actually
needs.

And it is dated correctly, which is subtler than it sounds. A row's `valid_to` is written
in place by the write that displaces it, so the row knows its own ending but not when
that ending came to be believed. `why()` dates an ending at the **successor's**
`recorded_at` — the instant the pointer was written — rather than at the instant the
change took effect. Without that rule, a query about July reports an August replacement.

## Ended, retired, erased: three words, three different events

This is the vocabulary the rest of the library is built on, and using the wrong one
records a false reason for a change that nothing downstream can detect.

| Word | What happened | Call | Recoverable |
|---|---|---|---|
| **ended** | The world changed. It was true, and then it wasn't. | a superseding write, or `forget(close="ended")` | the value still answers `valid_at` queries about its interval |
| **retired** | The record was wrong. It was never true. | `forget()`, `delete()` — the default | still visible to `history()` and `as_of` |
| **erased** | The text itself is gone. | `erase()`, `purge()` | no |

The first two are the ones that get confused, and the confusion is invisible in the data
afterwards: a write receipt once reported `retired 1` for a fact that had merely stopped
being true, leaving a reader with three names for two events.

**Never report a retirement as a deletion.** Nothing in the first two rows deletes
anything.

## Erasure removes the bytes, not just the row

`erase()` and `purge()` exist because *retire* is the wrong answer to "delete my data":
the text stays readable, which does not satisfy a GDPR Article 17 request. Erasure
removes the claim, the FTS entry (which stores the tokens directly), the embedding (which
leaks content under inversion) and — with `sources=True`, or always for `purge` — the
source turns.

```python
mem.erase(claim_id, sources=True)
mem.prove_erased(claim_id)     # re-queries every table the content could survive in
```

`erase(sources=True)` only removes turns that no surviving claim still cites, because one
turn can source several claims.

## The confidence and salience fields

Two more things travel with a claim and both are inspectable rather than opaque:

- **`confidence`** — how sure the writer was. A low-confidence extraction that would
  displace a high-confidence statement is kept beside it and reported in
  `WriteReceipt.disputed` rather than silently winning. See
  [contradiction resolution](contradiction-resolution.md#the-guard-a-guess-cannot-quietly-overwrite-a-statement).
- **`salience`** — how much this claim matters, which feeds ranking. See
  [temporal retrieval](temporal-retrieval.md).

## Where to see it

[`examples/coding_agent.py`](../../examples/coding_agent.py) writes the sources, then
answers *why did we change it, and on what evidence* out of `why()`.

---

Previous: [Contradiction resolution](contradiction-resolution.md) · Next: [Temporal retrieval](temporal-retrieval.md)
