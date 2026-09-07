# Memory vs. retrieval-augmented generation

Retrieval-augmented generation (RAG) and memory answer genuinely different questions, and
the useful way to think about them is as complements, not competitors. This page is not
an argument against RAG — Memvara itself uses retrieval internally to find facts — it's
an explanation of where the boundary between the two actually sits, and why crossing it
in either direction causes problems.

## The difference in one sentence each

- **RAG answers:** "Which documents are relevant to this question?"
- **Memory answers:** "What persistent state do I currently know about this specific
  thing, how has that state changed over time, and what was true at a particular point in
  the past?"

## The difference, side by side

| | RAG | Memory |
|---|---|---|
| The basic unit | A chunk of a document | A claim: subject, kind of fact, value |
| How it's found | Similarity to the question being asked | The exact slot it belongs to |
| How the underlying data grows | Mostly by addition — documents pile up | Mostly by replacement — values change |
| A contradiction looks like | Two chunks that both rank highly for the same question | One slot with a conflicting new value, resolved automatically when it's written |
| "When" means | The document's own date | Two separate dates — when it was true, and when it was learned |
| "Deleting" means | Removing a chunk from the index | Three genuinely different things — see [Provenance and trust](provenance-and-trust.md) |
| Good for | *"What does the manual say about TLS errors?"* | *"What is this customer's billing address, and what was it back in March?"* |

## Why you can't just use a document store as memory

Not because a document store is worse at its job — because the *shape* of the data is
wrong for state that changes.

A document corpus grows mostly by addition, and its individual facts tend to be stable:
last year's handbook still accurately says what it said last year. Agent memory is the
opposite case — a customer's billing address is a single slot whose *value* gets
replaced, and every time it's replaced, the old value is still sitting there in the
corpus, perfectly retrievable, and now simply wrong.

So this isn't a search-quality problem you can fix by improving the retrieval algorithm.
If you index "invoices go to Coldharbour Road" in January and "invoices go to Bramble
Cottage" in March, both sentences are genuinely relevant to the question "where do
invoices go," and nothing in a plain document index says the second one replaced the
first. Retrieval will return whichever one happens to match the phrasing of the question
best. Adding a general recency boost only delays the failure — it resurfaces the moment
the customer mentions the old address again later, which, in practice, happens.

Memvara's actual answer to this isn't "better retrieval." It's that writing the March fact
**closes the January fact's time interval** as a structural act, at write time — so the
old value is out of the current answer set by construction, and still available if
someone specifically asks about February. See
[How contradictions are resolved](how-contradictions-are-resolved.md).

## Why memory doesn't replace RAG either

A claim is a triple: subject, kind of fact, value. That's a genuinely bad fit for most of
what an AI agent actually needs to read:

- **Prose that needs to stay prose.** A policy document, a runbook, a legal clause — you
  want the actual paragraph, not a forced decomposition of it into subject-predicate-value
  triples.
- **Things with no single, stable subject.** A support article, a research paper, a raw
  log file.
- **Sheer volume.** A hundred thousand documents is a retrieval problem to be solved with
  a document index. A hundred thousand individual claims about a single user, on the
  other hand, usually means something went wrong in how the data was modeled.

Memvara does store the original source text — as **episodes**, the messages a fact was
extracted from — and you can retrieve that verbatim text alongside the facts it produced.
But that's a home for *evidence supporting a fact*, not a general-purpose document store.

## Using both together

The arrangement that actually works: **RAG answers questions from your document corpus,
and memory supplies the persistent state that corpus was never meant to track.**

```python
# 1. What does the agent persistently know about this specific user or entity?
context = mem.recall(question, k=8)

# 2. What does the document corpus say?
documents = your_document_search.search(question, k=5)

# 3. Combine both into the prompt, clearly labeled as different kinds of information.
prompt = f"{context}\n\nRetrieved documents:\n{render(documents)}\n\nUser: {question}"
```

One useful way to find the seam: a question like "what is our refund window for this
customer's plan?" is actually two separate questions bundled together — *which plan is
this customer on* (a memory question — a single current value with a full history behind
it) and *what does the refund policy say for that plan* (a RAG question — an actual
passage from an actual document). Answering them with the right tool for each, rather
than forcing one system to do both, is usually the better design.

It's also worth writing durable conclusions back into memory once you've worked them out:
the whole benefit of separating the two is that the next time someone asks *which plan is
this customer on*, it costs a lookup instead of a fresh retrieval and re-interpretation.

## Related pages

- [Search your memory](../how-to-guides/search-your-memory.md) — the mechanics of
  `recall()`, the call built specifically for putting memory into a prompt.
- [What problem does Memvara solve?](what-problem-memvara-solves.md) — the bigger picture
  this page zooms into.
