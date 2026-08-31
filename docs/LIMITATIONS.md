# Limitations

Every limit this project knows about, stated in full. The list is longer than most
projects publish, and it is long on purpose: a limit a reader finds for themselves, at
the point it breaks their work, costs far more than a paragraph here.

Four of these catch people first — the default embedder is lexical and Latin-only,
extraction from arbitrary prose needs a model, entity resolution folds surface forms
rather than the world, and the published LOCOMO and LongMemEval numbers are retrieval
rather than answer accuracy. The [FAQ](FAQ.md#what-are-the-honest-limitations) is the
short version if that is all you came for.

- **`HashingEmbedder` is a lexical fallback, not a semantic model.** It's the default so
  the library runs offline in milliseconds with no download, and it makes tests
  deterministic. It will not put "physician" near "doctor". Install
  `memvara[local-embed]` or pass your own embedder for real semantic recall.
- **Two harnesses compare against mem0, and only one of them runs the real thing.**
  `bench/mem0_real.py` drives the actual `mem0ai` package; `bench/compare.py` drives
  `bench/baseline.py`, a
  reimplementation of mem0's documented architecture, and is kept because it can vary
  parameters (top-k, threshold, chitchat ratio) that the real package does not expose.
  Both share one extraction oracle, so both isolate architecture from model quality — and
  neither says anything about end-to-end answer quality. That is what `demo/` is for, and
  its one run is a sanity check with an agent as the reader, not a benchmark.
- **The LOCOMO / LongMemEval numbers above are retrieval, not accuracy.** They are real
  and they run free, but they are not the metric those papers report and must never be
  quoted as if they were. Closing that gap needs a reader model. Measured, on
  `claude-opus-5`: **$7–$31 for LOCOMO** and **$3–$9 for LongMemEval**, the spread being
  thinking tokens rather than answers, plus a few dollars for `--judge llm`; a
  stratified `--shuffle 7 --limit 200` sample is about a tenth of that and finishes in
  twenty minutes rather than hours. The full procedure — flags, key variable, order of
  operations, worked example, and where each number came from — is in one place, the
  module docstring of `bench/evalkit.py`. It is deliberately not restated here, because
  it was previously stated in four places and three of them drifted: the "$17.50" this
  bullet used to carry assumed twice the input tokens the harness actually sends.
  The harness reports a `none` / `memory` / `full` triple when a reader *is* configured,
  on purpose: a memory score with no reader-only floor and no whole-haystack ceiling
  beside it is uninterpretable, and stuffing the transcript into the reader is measurable
  as `full`, labelled a reader ceiling rather than a result.
- **LOCOMO and LongMemEval are public, and a good end-to-end score on them proves less
  than it looks.** Any reader model may have seen them in training, and nothing in the
  harness can distinguish a retrieved answer from a remembered one. The asymmetry is
  the usable part: a *strong* score is weak evidence (contamination inflates), and a
  *weak* score is strong evidence against us. This is why the `--context none` floor is
  reported beside every score, and why a purpose-written scenario with no such confound
  is built separately rather than instead.
- **The vector index is exact and in-process.** A numpy matmul over the candidate set —
  correct and fast to roughly a million claims, at which point the `Store` protocol is
  where pgvector or Qdrant goes.
- **Predicate schema, the salience gate and the fast extractor are English-centric.** The
  schema grows by learning, but the seed set is small on purpose, and the gate's and
  extractor's rules are English sentence forms. On other scripts they fall through to the
  model — which is correct behavior and a real cost. This is the limitation the telemetry
  measures directly: `gate.drop` and `fast.miss` are tagged by script, so the gap is
  visible rather than assumed.
- **The default embedder is worse than English-centric — it is Latin-only.**
  `HashingEmbedder`, what you get with no extras installed, tokenises `[a-z0-9']+` and
  builds its character n-grams over the rejoined word list, so text in Han, Kana, Hangul,
  Arabic or Hebrew produces an **all-zero vector**. Retrieval handles that honestly — it
  abstains on a zero norm rather than inventing a rank — so such a claim is stored, is
  reachable by predicate, and is never returned by meaning. A write like that now warns
  (`UnembeddableTextWarning`) and counts (`write.embedding_unusable`, tagged by script).
  Mixed text is affected without being caught: `user lives in 里斯本` embeds fine, from
  the Latin half alone. Installing `memvara[local-embed]` gets a real model and non-zero
  vectors; genuine cross-language retrieval needs a multilingual model and is not claimed.
- **Entity resolution folds surface forms, it does not know the world.** `Acme Corp` and
  `acme, inc.` collapse; `Big Blue` and `IBM` do not, unless you enable the opt-in model
  path or declare the alias. `Stark` versus `Stark Industries` is genuinely ambiguous and
  is left that way. The fold is confident in the other direction too — two people who
  share a name are one entity, so on a single-valued predicate the later one's job
  retires the earlier one's. Nothing in the data can tell that from an ordinary job
  change, so there is no warning for it; `split_entity()` is the repair, dated and
  dry-run by default, for when a person knows what the store cannot.
- **`AsyncMemvara` is a thread-pool wrapper, not an async rewrite.** It keeps an asyncio
  event loop unblocked, which is what it is for; it does not make the store itself async.
- **With no `llm=`, `add()` keeps only what its rules recognise — and on some corpora that
  is nothing.** The default `NullLLM` runs tiers 0, 1 and 1b and then stops, so
  high-precision sentence forms ("I live in X", "I work at X") are extracted for nothing
  and an employer mentioned in passing is dropped. Measured on `demo/`'s 64-turn support
  history: **64 episodes, 0 claims** — the rules matched not one turn, so that store does
  no supersession and no bitemporal reasoning at all. It is loud rather than silent —
  `Memvara()` warns once with a `DegradedExtractionWarning`, and
  `WriteReceipt.unextracted` counts the dropped turns on every write — but it is the
  qualifier on the offline claim: the *library* runs with no API key, extraction from
  arbitrary prose does not. `remember()` with a declared `PredicateSpec` is the offline
  way to get the full machine, and it is what a real integration does; see
  [What the fast path does not catch](DESIGN.md#what-the-fast-path-does-not-catch-measured).
  Retrieval, contradiction resolution and consolidation never needed a model.
- **No REST server in the open core, and there is not going to be one.** MCP over stdio is
  the shipped remote surface here. The REST API is a component of the commercial product
  rather than a gap in this one — see
  [Open core](OPEN-CORE.md), which says where that line is and
  why it does not move. What this repository does ship is the *client* half, twice over.
  `memvara/remote/` is the one an application calls: `Memvara(api_key=...)`, the library's
  own API served by a deployment. `memvara/store/remote.py` is the low-level `Store` the
  local engine calls, and it is partial on purpose: it implements what the REST facade
  actually exposes and raises `NotImplementedError`, with a docstring, everywhere it does
  not. A `put_claim` that quietly wrote through `POST /v1/facts` would reinterpret every
  field the caller set, and a `competing_claims` returning `[]` for want of an endpoint
  would tell every write path a slot was empty. Both are worse than an exception.
- **The framework adapters do not all preserve what makes memvara different.** LangChain
  and LlamaIndex *retrievers* keep everything, including `as_of=`, because "query in,
  documents out" is what `search()` already is. A LangChain `ChatMessageHistory` keeps
  the write path and loses the rest: a `list[BaseMessage]` has nowhere to put a
  supersession, a valid-time interval or a source id, and tier-0 dedupe means it is not
  a faithful transcript either. CrewAI loses the headline feature outright — its unit of
  memory is an opaque sentence with no subject or predicate, so the keyed lookup has
  nothing to key on and "Alice lives in Berlin" and "Alice moved to Lisbon" both stay
  live. **LangGraph loses least of the four**, and instructively: `BaseStore` is the only
  interface that hands over the query text natively, *and* `put(namespace, key, value)`
  supplies all three parts of a triple — so an item is stored as one claim per field and
  changing `city` ends exactly `city`, which is contradiction resolution surviving a
  foreign interface intact. What it loses is the predicate registry: a stored `home_city`
  does not contradict an extracted `lives_in`. Each adapter says which it is; see
  `memvara/integrations/`.
- **No encryption at rest.** `purge()`, `erase()` and the redaction hook cover the
  deletion and ingestion halves of a privacy story; the storage half is the deployment's
  problem, and full-disk encryption is the honest answer today. It is not laziness:
  SQLCipher works here — measured, +43–48% on writes, search unchanged, and FTS5 keeps
  working because page-level encryption sits *beneath* SQLite — but the mmap-backed
  `.vecs` sidecar stays plaintext outside that boundary, and a plaintext vector is a
  confirmation oracle. Encoding a guess and taking the cosine against that file returns
  exactly 1.0000 for the right text and 0.87 for a one-digit-different phone number, so
  it is not merely confirmable, it is hill-climbable. Encrypting the text and not the
  vectors would be theatre.
- **The built-in redactor is not compliance-grade** and says so in its own docstring. It
  is a default, not a product: the seam is the deliverable, and a serious deployment
  brings its own `Redactor`.

---

Previous: [Benchmarks](BENCHMARKS.md) · Next: [Roadmap](ROADMAP.md) · [Open core](OPEN-CORE.md)
