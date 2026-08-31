# RAG and memory

**They answer different questions, and the useful arrangement is both.** This page is not
an argument against retrieval-augmented generation — RAG is the right tool for the job it
does, and Memvara uses retrieval internally to find claims.

```
RAG
 ↓
"Which documents are relevant to this question?"

Memory
 ↓
"What persistent state do I know about this entity,
 how has that state changed,
 and what was true at a particular point in time?"
```

## The difference in one table

| | RAG | Memory |
|---|---|---|
| Unit | a chunk of a document | a claim: `(subject, predicate, object)` |
| Addressed by | similarity to a query | the slot `(subject, predicate)` |
| Corpus | mostly static; documents are added | mostly mutable; values are replaced |
| A contradiction is | two chunks that both rank | a slot with two live values, resolved on write |
| "When" means | the document's date | two dates: when it was true, when you learned it |
| Deleting means | dropping a chunk | three different things — see [provenance](provenance.md#ended-retired-erased-three-words-three-different-events) |
| Good at | *what does the manual say about TLS errors* | *what is this customer's billing address, and what was it in March* |

## Why a document store cannot just be used as memory

Not because it is worse — because the *shape* is wrong for mutable state.

A document corpus grows by addition and its facts are stable: the 2024 handbook still
says what it said. Agent memory is the opposite. A customer's address is one slot whose
value gets replaced, and every replacement leaves behind a chunk that is still perfectly
retrievable and now wrong.

So the failure is not a ranking failure. When you index *"invoices go to Coldharbour
Road"* in January and *"invoices go to Bramble Cottage"* in March, both are in the corpus,
both are relevant to *where do invoices go*, and nothing in the index says the second
replaced the first. Retrieval returns whichever embeds closest to the phrasing of the
question. Adding a recency boost helps until the customer mentions the old address again
in August, which they will, because that is what people complain about.

Memvara's answer is not better retrieval. It is that the March write **closed the January
claim's interval**, so the old value is out of the live set by construction, and still
there when you ask about March. See
[contradiction resolution](contradiction-resolution.md).

## Why memory does not replace RAG either

A claim is a triple. Triples are the wrong representation for most of what an agent needs
to read:

- **Prose that has to stay prose** — a policy document, a runbook, a legal clause. You
  want the paragraph, not a decomposition of it.
- **Things with no stable subject** — a support article, a paper, a log file.
- **Volume.** A corpus of a hundred thousand documents is a retrieval problem. A hundred
  thousand claims about one user is a data-modelling mistake.

Memvara stores source turns as **episodes** and returns them alongside claims when you
ask (`include_episodes=True`), so verbatim text has a home here — but it is a home for the
evidence behind a fact, not a document store.

## Using both

The composition that works is: **RAG answers from the corpus, memory supplies the state
the corpus does not know.**

```python
# 1. What does this agent persistently know about this user, right now?
context = mem.recall(question, k=8)

# 2. What does the corpus say?
docs = your_vector_store.search(question, k=5)

# 3. Both into the prompt, labelled as different kinds of thing.
prompt = f"{context}\n\nRetrieved documents:\n{render(docs)}\n\nUser: {question}"
```

`recall()` returns a block already framed as *reference data, not instructions*, with
each claim flattened to one line — see
[temporal retrieval](temporal-retrieval.md#recall-is-the-one-you-put-in-a-prompt).

Two places the seam is worth thinking about:

- **Let memory decide the entity, and RAG decide the passage.** *"What is our refund
  window for this customer's plan?"* is two questions: which plan they are on (memory —
  one slot, one current value, with a history) and what the refund policy says for that
  plan (RAG — a passage from a document).
- **Write the durable conclusions back.** The point of the previous step is that next
  time, *which plan they are on* costs a lookup rather than a retrieval.

Memvara ships retriever adapters for LangChain and LlamaIndex, so in an existing RAG
pipeline it can also be *one more retriever* rather than a separate call. That is the
cheapest way to try it, and it does give up the time keywords in some of the adapters —
see [frameworks](../integrations/frameworks.md) for exactly which.

## What Memvara does not claim

It is not a knowledge graph over your documents, it does not do entity extraction at
corpus scale, and its vector index is exact and in-process rather than an ANN service.
Point it at a hundred thousand documents and you have used the wrong tool; point it at
what your agent needs to remember about the people and systems it works with, and the
five questions in [Why Memvara?](why-memvara.md) become lookups.

---

Previous: [Temporal retrieval](temporal-retrieval.md) · Next: [Guide: coding agents](../guides/coding-agents.md)
