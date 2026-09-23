# Parity phase 2: documents and retrieval

**Date:** 2026-09-23. **Status:** design approved in conversation on 2026-09-23, not built.
Line numbers are from agent-memory `origin/main` at `b57ef714` (**core**) and memvara-cloud
`origin/main` at `c1ab3ab` (**cloud**). Phase 1 is
`2026-09-23-parity-phase-1-claude-code-experience-design.md`; its switch store (§3 there) is
reused here.

## 1. What this phase delivers, and which decisions it reverses

| # | Feature | Reverses |
|---|---|---|
| 1 | A document store with caller-supplied ids | the positioning line "A policy document should stay a document" (`docs/FAQ.md:30`, `docs/concepts/rag-vs-memory.md:105`) |
| 2 | Ingestion of URLs, HTML, PDF, images, audio and video | same |
| 3 | Chunking for retrieval, and chunking before extraction | ROADMAP deferred entry "Splitting a long turn into chunks before extraction" (`docs/ROADMAP.md:485-514`) |
| 4 | Metadata filters and a file-path filter in search | — |
| 5 | Query rewrite and result synthesis with a model on the read path | invariant 1 (`docs/INTERNALS.md:34-53`) and "The anchor is given, never parsed" (`memvara/retrieve/temporal.py:17-23`) |
| 6 | Encryption at rest for the local store, vectors included | ROADMAP deferred entry "Encryption at rest" (`docs/ROADMAP.md:587-594`, `docs/LIMITATIONS.md:129-138`) |

Decisions taken on 2026-09-23: all on by default with a switch in setup; the per-prompt recall
hook uses the read-path model only when setup found a key, and setup shows the cost; deleting a
document retires memories whose only source it was (§4.1).

Each reversal is a decision page in Confluence and changes the text it reverses in the same
commit: the ROADMAP entry moves to a "Reversed" list with the date and the reason, and the
invariant text says precisely what is now allowed. The old text is not deleted, because the
measurements in it are still true.

Switch names: `documents`, `ingest_urls`, `ingest_media`, `retrieval_chunks`,
`extraction_chunks`, `metadata_filters`, `query_rewrite`, `synthesis`, `encryption`.

## 2. Constraints every workstream keeps

- Phase 1's §2 constraints apply.
- **Lazy optional imports** (invariant 5, `docs/INTERNALS.md:156-174`, enforced by
  `tests/test_packaging.py`). New dependencies go in a new `ingest` extra and an `encrypt` extra
  in `pyproject.toml:47-96`; nothing new becomes a hard dependency.
- **Filters run where the limit runs** (invariant 7, `docs/INTERNALS.md:197-218`): metadata and
  path filters go into the store methods, not a post-filter.
- **`why()` keeps working**: every claim still cites whole episodes through `Claim.sources`
  (`memvara/types.py:709`) and `claim_sources`.

## 3. The data model

New tables in both stores. SQLite goes from schema v13 (phase 1) to v14; Postgres from v3 to v4.

**`documents`**: `tenant`, `id` (`doc_…`), `custom_id` (nullable, unique per tenant and scope
key, at most 255 characters), `scope_key`, `title`, `filepath` (nullable, `/`-separated),
`source_uri` (nullable), `mime`, `content_hash` (blake2b-16 of the normalised text),
`status` (`queued`, `extracting`, `done`, `failed`), `error` (nullable), `meta` (JSON),
`created_at`, `updated_at`.

**`document_chunks`**: `tenant`, `document_id`, `position`, `text`, `hash`, `episode_id`
(the episode this chunk became), primary key `(tenant, document_id, position)`.

**Chunk vectors** reuse the episode vector path: each chunk is stored as an episode with
`role="system"` and `meta.document_id`, so the existing `episode_embeddings`, FTS index and
`include_episodes` search leg serve chunks without a second index (core `store/sqlite.py:300`,
:319; `retrieve/hybrid.py:160`). The `Episode.hash` dedupe (`types.py:598-599`) is extended with
the document id, so two documents with a shared paragraph keep separate chunks.

**Metadata index**: SQLite `json_extract` expressions are generated per filter key; Postgres
gains a GIN index on `claims.meta`, `episodes.meta` and `documents.meta`.

## 4. The features

### 4.1 The document store

**Interface (library; MCP and REST mirror it).**

```python
Memvara.add_document(content: str | bytes | None = None, *, url: str | None = None,
                     custom_id: str | None = None, title: str | None = None,
                     filepath: str | None = None, mime: str | None = None,
                     meta: dict | None = None, extract: bool = True, **scope) -> Document
Memvara.get_document(id_or_custom_id, **scope) -> Document | None
Memvara.list_documents(*, filepath_prefix=None, status=None, limit=50, cursor=None, **scope) -> Page[Document]
Memvara.update_document(id_or_custom_id, *, content=..., title=..., meta=..., filepath=...) -> Document
Memvara.delete_document(id_or_custom_id, **scope) -> DeleteResult
Memvara.delete_documents(ids_or_custom_ids, **scope) -> list[DeleteResult]
Memvara.document_status(id_or_custom_id, **scope) -> DocumentStatus
```

- Exactly one of `content` and `url` is given.
- **Re-ingest by `custom_id`.** Adding with an existing `custom_id` is an update. The new text is
  chunked, and each new chunk is matched to an old chunk **by content hash, not by position**, so
  an edit near the top of a document does not invalidate every later chunk. A matched chunk keeps
  its episode and vector and only has its `position` updated; an unmatched new chunk becomes a new
  episode; an old chunk with no match is handled like a delete of that chunk (next bullet). Only
  new episodes go to extraction. The chunker's boundaries depend only on the text around them, so
  an edit changes the chunks it touches and at most the one after it.
- **Delete.** The document row, its chunk rows and its chunk episodes are erased, using the
  existing `erase_episode` path extended to documents. Each claim whose every source was one of
  those episodes is **retired** with reason `"source document deleted"` (phase 1 §4.8) before its
  sources are removed. A claim with any other source keeps its other sources. No memory is
  erased by a document delete.
- **Status.** `queued` on write, `extracting` while the write pipeline runs over the new
  episodes, `done` or `failed` with `error`. Extraction failures leave the document stored and
  searchable.
- **Surfaces.** MCP tools `memory_add_document`, `memory_get_document`, `memory_list_documents`,
  `memory_delete_document`; cloud `POST/GET /v1/documents`, `GET/PATCH/DELETE
  /v1/documents/{id}`, `POST /v1/documents/delete` (bulk), `GET /v1/documents/{id}/status`;
  `RemoteMemvara` methods of the same names.

### 4.2 Ingestion

A new package `memvara/ingest/` with one module per source type and one entry point,
`ingest(content | url, mime) -> Extracted(text, title, mime, pages?)`.

- **URLs.** `http` and `https` only. The host is resolved and refused if any address is
  private, loopback, link-local, multicast or reserved; the check repeats after every redirect
  (at most 5). Limits: 10 MB body, 20 s total, a fixed User-Agent. The fetcher is the only
  network code in `ingest/` and is injected, so tests pass a fake.
- **HTML.** Standard-library `html.parser` extraction of the main text and title; scripts,
  styles and navigation dropped.
- **PDF.** `pypdf` from the `ingest` extra; one text block per page, page numbers kept in chunk
  metadata.
- **Images, audio, video.** A new runtime-checkable protocol in `memvara/llm/base.py`:

  ```python
  class Multimodal(Protocol):
      def describe_image(self, data: bytes, mime: str) -> str: ...
      def transcribe(self, data: bytes, mime: str) -> str: ...
  ```

  It is a separate protocol, like `Chat` and `ReplacementJudge` (`base.py:142-180`), so older
  backends keep passing `isinstance` checks. Implemented on `AnthropicLLM` (images) and
  `OpenAILLM` (images and audio; video as audio track plus sampled frames when the backend
  accepts them). Without a backend that supports the media type, the upload is refused with
  `media_unsupported` and the reason.

### 4.3 Chunking

**Retrieval chunks** (`retrieval_chunks`). Sentence-aware splitting to about 1,000 characters
with a 150-character overlap, never splitting inside a sentence unless one sentence exceeds the
limit. Used for every document.

**Chunking before extraction** (`extraction_chunks`). For a single turn or document section
over 6,000 characters, extraction runs per chunk inside core, not the worker, so
`source_index` keeps pointing at the whole episode (the ROADMAP entry's own condition). Claims
from different chunks with the same `fact_key` and value are merged before reconciliation.

**Release bar, from the ROADMAP entry.** On the longest episode in
`tests/fixtures/phi4_spike/`, chunked extraction must recover **5 of 5** key facts with **0**
duplicates. A test pins this with recorded model output. If the bar is not met, the feature
ships switched off, visibly, with the measurement in the pull request body, per the repository
rule on scope that cannot be finished.

### 4.4 Metadata and file-path filters

`search()` and `recall()` gain `filters: dict[str, str | int | float | bool | list] | None` and
`filepath_prefix: str | None`.

- A filter is an equality test on a top-level key of `meta`; a list means "any of". Keys are
  restricted to `[A-Za-z0-9_.-]{1,64}` so they are safe inside a JSON path.
- The filter reaches `candidate_ids`, `lexical_search`, `vector_search` and the episode legs in
  `store/base.py:662-737`, so the limit applies after filtering (invariant 7).
- `filepath_prefix` matches documents' `filepath` and the episodes and claims that cite them.
- MCP `memory_search` and `memory_recall` gain `filters` and `filepath_prefix`; cloud
  `SearchRequest` and `RecallRequest` (`rest/models.py:1172-1256`) gain the same fields.

### 4.5 A model on the read path

**Query rewrite** (`query_rewrite`). Before retrieval, one chat call receives the query and
today's date and returns JSON: up to 3 alternative queries and an optional date range. Each
query is retrieved; the lists are fused with the existing reciprocal-rank fusion; the date range
becomes a `valid_at` window. The explicit `valid_at` argument, when given, wins over the model's
range.

**Synthesis** (`synthesis`). `recall(synthesize=True)` sends the recalled rows to one chat call
and returns a short synthesis above the rows. The rows are still returned, so nothing the model
drops is lost to the caller.

**Behaviour shared with `ranked=True`.** Both reuse the selector's pattern
(`memvara/select/model.py:143-158`, `retrieve/hybrid.py:1340-1460`): the caller's `Chat` backend,
a 10 s timeout, and outcomes `applied`, `fallback`, `key_rejected`, `disabled`, `unconfigured`
reported on the result. A fallback serves the plain result. The cloud uses the per-organisation
key (`control/api/selector_settings.py`).

**Per-prompt hook.** The recall hook turns rewrite on only when setup recorded a working key;
`/memvara:setup` shows the added latency and cost before the user confirms.

**Invariant text.** Invariant 1 in `docs/INTERNALS.md` is rewritten to: deterministic stages
(deduplication, contradiction resolution, ranking, decay, time travel) never call a model; the
read path may call one only through the named stages `ranked`, `query_rewrite` and `synthesis`,
each with a recorded outcome and a model-free fallback. `retrieve/temporal.py` keeps its rule for
the deterministic leg and points to `query_rewrite` as the model-backed alternative.

### 4.6 Encryption at rest (local store)

- **Database:** SQLCipher through the `encrypt` extra (`sqlcipher3` binding). `SQLiteStore`
  (`store/sqlite.py:1117`) opens with the key when `encryption` is on.
- **Vectors:** the `.vecs` sidecar (`_vec_path`, :766-772; `np.memmap`, :926) is encrypted with
  AES-256-GCM in fixed-size blocks and decrypted into an in-memory array on open; writes append
  encrypted blocks. This removes the objection in the ROADMAP entry, that a plaintext vector is a
  confirmation oracle. The cost is memory: the matrix is resident instead of memory-mapped.
- **Key:** the OS keychain (macOS Keychain, Linux Secret Service), else `MEMVARA_DB_KEY`, else a
  generated key in `~/.memvara/db.key` with mode 0600 and a warning at start. Losing the key makes
  the store unreadable; `memvara encrypt --export-key` prints it for backup.
- **Existing stores:** `memvara encrypt <db>` converts in place through a temporary copy and an
  atomic rename. New stores are encrypted by default.
- **Postgres:** unchanged; `docs/LIMITATIONS.md` states that database and disk encryption belong
  to the operator.

**Measurement required in the pull request:** write and search latency, and resident memory,
before and after, on the store used for `bench/`.

## 5. Workstreams for the fan-out

| Stream | Repository | Features | Depends on |
|---|---|---|---|
| P2-A | core | 3 data model, 4.1 store and library API, 4.3 retrieval chunks | phase 1 P1-A merged (schema v13) |
| P2-B | core | 4.2 ingestion package and `Multimodal` | P2-A's `add_document` signature (can start on a stub) |
| P2-C | core | 4.3 extraction chunks with the release bar | — |
| P2-D | core | 4.4 filters | P2-A merged (documents' `filepath`) |
| P2-E | core | 4.5 rewrite and synthesis, invariant rewrite | — |
| P2-F | core | 4.6 encryption | P2-A merged (last schema change of the phase) |
| P2-G | cloud | Postgres v4, document routes, filter and rewrite fields, per-org key use | P2-A, P2-D, P2-E merged |
| P2-H | core `plugin/hooks` + plugin repo | hook uses rewrite when keyed, setup entries | P2-E merged |

## 6. Out of scope for phase 2

Connectors that feed the document store are phase 3. Encryption of the cloud database is an
operator task, not this phase.
