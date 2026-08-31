# Why Memvara?

**Retrieval and memory answer different questions, and most agent memory layers are
retrieval systems being asked to do memory's job.**

Retrieval asks one question:

> *What information is relevant to this?*

Memory has to answer five:

> *What do I know?*
> *When was it true?*
> *When did I learn it?*
> *What replaced it?*
> *Why do I believe it?*

An embedding plus a timestamp answers the first. It cannot answer the other four, and no
amount of retrieval quality gets you there, because the information those questions need
was thrown away at write time.

## The failure, concretely

A customer emails support in March:

> *We've moved. Invoices should go to Bramble Cottage from now on, not Coldharbour Road.*

You store that. In August the same customer writes again, annoyed:

> *An invoice went to Coldharbour Road again.*

Now ask the store where to send invoices. It holds two sentences, both mentioning the
customer and an address, and it ranks them by similarity to the question. The August
message is more recent and mentions Coldharbour Road twice. A reasonable retriever
returns it first, an agent reads it, and the customer gets a third wrong invoice.

**That is not the model hallucinating.** The memory layer held both answers and marked
neither one current. It had no way to: nothing in "an embedding and an `updated_at`
column" can represent *this value replaced that one in March*.

## What each of the five questions needs

### *What do I know?*

A fact has to be addressable, not merely findable. Memvara stores a **claim** — a
`(subject, predicate, object)` triple — so `(customer, billing_address, ?)` is a slot you
can look up, rather than a phrase you can search for. That is what makes the next four
questions answerable at all.

It is also why contradiction resolution here is not a similarity problem. See
[contradiction resolution](contradiction-resolution.md).

### *When was it true?* and *When did I learn it?*

Two different dates, and collapsing them into one loses real answers. Every claim carries
both:

- **valid time** — the interval the fact held in the world.
- **transaction time** — when this store came to believe it.

A correction that arrives in August about June has an August transaction time and a June
valid time. With one column you can record when you learned it or when it was true, and
either choice makes a class of question unanswerable. See
[bitemporal memory](bitemporal-memory.md).

```python
mem.get_all(valid_at=june)   # what we believe today about how June was
mem.get_all(known_at=june)   # what we believed in June, about the world as it is now
mem.get_all(as_of=june)      # both — what we believed in June, about June
```

The third one is what an auditor asks: *what would this system have told somebody on that
day?* It is a different question from *what actually was the case*, and a store that
cannot separate them cannot answer either honestly. `ask(question, at=T)` is the read that
composes all three into a sentence, and it is the one to reach for when the auditor's
question is the question — see
[bitemporal memory](bitemporal-memory.md#ask-composes-all-three-into-a-sentence).

### *What replaced it?*

Coldharbour Road did not become wrong. It stopped being current, on a specific day,
because Bramble Cottage took over. Memvara closes the old claim's interval and keeps it:

```python
# `billing_address` is not in the built-in vocabulary, so declare it single-valued —
# otherwise it takes the safe default, MANY, and both values stay live.
registry = PredicateRegistry(BUILTIN_PREDICATES + (
    PredicateSpec(name="billing_address", cardinality=Cardinality.ONE,
                  volatility=Volatility.SLOW),))

[(c.object, c.state) for c in mem.history("customer", "billing_address")]
# [('Coldharbour Road', 'ended'), ('Bramble Cottage', 'live')]
```

`ended` is one of three states, and the distinctions are the product:

| State | Means | Written by |
|---|---|---|
| `live` | currently believed and currently true | any write |
| `ended` | was true, the world changed | a superseding write, or `forget(close="ended")` |
| `retired` | never true, the record was wrong | `forget()`, `delete()` |

*Served a value that expired* and *served a value that was never true* are one column
apart and they are not the same finding. A store with a single "deleted" flag reports the
same thing for both.

### *Why do I believe it?*

```python
p = mem.why(claim_id)
[e.text for e in p.episodes]
# ["We've moved. Invoices should go to Bramble Cottage from now on, not Coldharbour Road."]
```

The turn, not a paraphrase of it. When an agent says something wrong, this is how you
find out which memory caused it and where that memory came from. See
[provenance](provenance.md).

## The consequence that is easy to miss

Because the answers above are **structural** rather than inferred, none of them costs a
model call. A contradiction is an indexed lookup on `(subject, predicate)`. A historical
query is a range condition on two columns. Provenance is a join.

So the write path does not need a model to be correct, and the read path does not need
one to be honest. In this library the model is an *extractor* — the thing that turns
prose into triples — and it is the only part that is optional, batched, and reported on
every receipt as `llm_calls`.

That is the actual claim, and it is narrower than "AI memory":

> **Memvara is persistent temporal memory with history, provenance, and deterministic
> state evolution.**

Not a vector database — though it uses vectors, and swapping in pgvector is a protocol
away. Not a RAG wrapper — see [RAG and memory](rag-vs-memory.md), which is about how the
two compose rather than which one wins.

## What this costs you

Honesty is part of the argument, so:

- **You have to have triples.** Memvara gets them from `remember()` for free, and from
  prose with a rule-based extractor that recognises a fixed set of sentence forms. Beyond
  those forms it needs a model — and with no `llm=` configured, turns it does not
  recognise are dropped rather than stored. It counts them (`WriteReceipt.unextracted`)
  and warns once, but it is a real limit.
- **You have to declare vocabulary for your domain.** The built-in predicates are a
  personal-assistant set. Engineering facts need a pack or a few `PredicateSpec`s, or
  everything falls to the multi-valued default and nothing supersedes.
- **Entity resolution folds surface forms, not the world.** `Acme Corp` and `acme, inc.`
  collapse. `Big Blue` and `IBM` do not, unless you say so.

[Limitations](../LIMITATIONS.md) is the full list, and it is considerably longer than
this one.

---

Previous: [Your first memory](../getting-started/first-memory.md) · Next: [Bitemporal memory](bitemporal-memory.md)
