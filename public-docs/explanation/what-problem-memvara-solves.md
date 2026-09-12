# What problem does Memvara solve?

Most systems built to give AI agents "memory" are actually retrieval systems, and
retrieval systems only know how to answer one question: *what information is relevant to
this?* Memory has to answer five, and the fifth-hardest one — *what replaced this?* — is
exactly the one that causes the most visible, embarrassing failures in production.

## A concrete failure

Imagine a customer support system with memory. In March, a customer emails:

> "We've moved. Invoices should go to Bramble Cottage from now on, not Coldharbour Road."

The system stores this. In August, the same customer writes again, annoyed:

> "An invoice went to Coldharbour Road again."

Now ask the memory system where to send invoices. It's holding two pieces of text, both
mentioning this customer and an address, and — if it's built on similarity search — it
ranks them by how closely they match the question. The August message is more recent and
mentions "Coldharbour Road" twice. A perfectly reasonable retrieval system returns it
first. The agent reads it, and the customer gets a third wrong invoice.

**This is not the AI model hallucinating.** The memory system genuinely held both
answers and had no way to mark either one as the current one. There was nothing in
"a stored sentence plus a timestamp" that could represent the fact *this value replaced
that one, in March.*

## The five questions

Retrieval systems are built to answer one question well. A real memory system has to
answer five, and they're genuinely different questions:

1. **What do I know?** — Not "what text is similar to this," but "what is the current
   value of this specific fact?"
2. **When was it true?** — The interval during which a fact actually held, in the real
   world.
3. **When did I learn it?** — A separate date from the above, because information often
   arrives late, and conflating the two loses real information.
4. **What replaced it?** — Not just "this fact is gone," but a link to what took its
   place, and when.
5. **Why do I believe it?** — A trace back to the actual message or event the belief came
   from, not a system's paraphrase of it.

An embedding plus a single timestamp can answer the first question and nothing else. No
amount of tuning retrieval quality gets you the other four, because the information those
questions need was thrown away the moment the fact was written down as an undifferentiated
blob of text.

## How Memvara answers each one

**"What do I know?"** Memvara stores facts as **claims** — a subject, a kind of fact
(called a predicate), and a value, like `(customer, billing_address, "Bramble Cottage")`.
This makes a fact something you can look up by its slot, not just something you can
happen to find by searching for similar wording. See
[How contradictions are resolved](how-contradictions-are-resolved.md).

**"When was it true?" and "When did I learn it?"** Every claim carries two independent
dates — see [Bitemporal memory, explained](bitemporal-memory-explained.md).

**"What replaced it?"** When a new value for a single-valued fact arrives, the old one's
time interval is closed, not deleted — the record stays, marked as no longer current, and
linked to what took its place.

**"Why do I believe it?"** Every claim traces back to the actual message it came from —
see [Provenance and trust](provenance-and-trust.md).

## The part that's easy to miss

Because Memvara's answers to these questions come from stored structure rather than from
asking a model to figure it out again each time, none of them costs a model call.
Deciding whether two facts conflict is a database lookup. Answering "what was true on this
date" is a range check on two stored columns. Explaining where a fact came from is a
simple lookup, not a summarization task.

This is why an AI model is only ever needed for one thing in Memvara: turning loose,
unstructured text into a structured fact in the first place. Once a fact is structured,
everything else — contradiction handling, time travel, search, explaining a decision — is
mechanical, deterministic, and fast.

## What this is not

Memvara is not a vector database, though it uses vector search internally as part of how
it finds facts. It's not a replacement for retrieval-augmented generation (RAG) either —
see [Memory vs. retrieval-augmented generation](memory-vs-retrieval-augmented-generation.md)
for how the two actually fit together, which is as complements rather than competitors.

## Related pages

- [Bitemporal memory, explained](bitemporal-memory-explained.md)
- [How contradictions are resolved](how-contradictions-are-resolved.md)
- [Provenance and trust](provenance-and-trust.md)
- [Known limitations](known-limitations.md) — the honest cost of getting these five
  answers right.
