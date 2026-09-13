# Known limitations

Every meaningful limit Memvara has, stated plainly, and checked against the actual code.
A limit you discover on your own, at the exact moment it breaks something you were
relying on, costs far more than a paragraph here would have. This page exists so that
doesn't happen to you.

## Search is keyword-based by default, not meaning-based

`HashingEmbedder`, the default with no extra packages installed, is a fast, fully
offline, keyword-based fallback rather than a genuine semantic embedding model. It will
not connect "doctor" with "physician." It also only understands Latin-script text — text
written in Chinese, Japanese, Korean, Arabic, or Hebrew produces no usable
meaning-matching signal, though it's still stored and still fully searchable by keyword.
Memvara warns you the first time this happens on a write, rather than silently dropping
anything. Install `memvara[local-embed]` and configure a real embedding model for genuine
semantic and multilingual search.

## Extracting facts from free-form text needs a language model

With no model configured, Memvara only recognizes a small, fixed set of very
high-confidence sentence patterns — things like "I live in X" or "I work at X." Anything
more nuanced or indirect is not understood, and is dropped rather than guessed at. This
is loud, not silent: Memvara warns you once when this happens, and every write reports
exactly how many messages it had to skip. Recording facts you already know precisely,
using `remember()`, never needs a model at all, regardless of configuration.

## Recognizing that two names refer to the same thing has limits

Memvara folds obviously equivalent spellings together — "Acme Corp" and "acme, inc." are
recognized as the same entity — but it does not understand the world well enough to know
that "Big Blue" and "IBM" are the same company unless you explicitly tell it so. This
matching can also be too generous in the other direction: two different people who happen
to share a name are treated as one entity by default, which means, on a fact that's
declared single-valued, a job change recorded for one of them can look identical to a job
change for the other. There's a repair tool for exactly this situation
(`split_entity()`), but nothing warns you proactively when it happens, because there's
genuinely no way to detect it automatically.

## The vector search index does not scale to very large corpora

Memvara's search index is exact and runs entirely in-process — there's no approximate
nearest-neighbor index and no separate search service to run. This is correct and fast up
to roughly a million stored facts. Past that point, a pluggable storage backend (such as
pgvector or a dedicated vector database) is the right next step, and Memvara's storage
interface is designed to allow that.

## Personal-assistant facts and English are the built-in starting points

The built-in vocabulary of recognized facts is aimed at personal-assistant use cases —
where someone lives, works, and so on — and the rules used to automatically recognize
sentence patterns are written for English. Text in other languages falls through to
needing a language model for extraction, which is the correct behavior, but it is a real,
measurable cost rather than something Memvara claims to handle for free.

## Published benchmark numbers measure retrieval, not final answer accuracy

Public benchmark results Memvara reports (from datasets like LOCOMO and LongMemEval)
measure how well the system retrieves the right underlying facts — not whether a
downstream language model, given those facts, produces the fully correct final answer to
a question. These are genuinely useful and cheap to reproduce, but they are not the exact
metric those benchmark papers themselves report, and shouldn't be quoted as if they were.
Because both benchmarks are public, any language model used to grade a full end-to-end
answer may already have encountered them during its own training, which is a separate
reason to treat a very strong end-to-end score with some skepticism, while treating a weak
one as fairly solid evidence of a real gap.

## No encryption of stored data by default

Memvara handles data deletion and redaction, but it does not encrypt the database file
itself — that's left to however you deploy it (full-disk encryption is the straightforward
answer for most self-hosted setups). This is a deliberate scope decision rather than an
oversight: encrypting the searchable text while leaving the raw vector embeddings
unencrypted would provide very little real protection, since a plaintext embedding can be
used to confirm a guess about the underlying text with very high accuracy.

## The built-in redaction tool is a starting point, not a compliance guarantee

Memvara ships a basic tool for stripping obviously sensitive content (like credit card
numbers) before storage, but it is explicitly not certified or guaranteed to meet any
specific compliance standard. Treat it as a sensible default you can build on, not as a
finished answer to a regulatory requirement — a serious deployment should bring its own,
more thorough redaction logic where the stakes require it.

## Related pages

- [What problem does Memvara solve?](what-problem-memvara-solves.md) — the underlying
  design tradeoffs these limitations follow from.
- [Install Memvara](../how-to-guides/install-memvara.md) — which optional extras remove
  which limitations above.
