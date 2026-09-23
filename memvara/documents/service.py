"""The document store behind `Memvara.add_document` and the methods beside it.

A document is stored as a row that describes it and a list of chunks that hold its text.
Each chunk is an ordinary episode with `role="system"` and `meta["document_id"]` set, so
the episode text index, the episode vectors and `search(include_episodes=True)` serve
documents without a second index, and `why()` on a claim extracted from a document quotes
the chunk it came from.

Three behaviours are the reason this module exists rather than a loop over `add()`.

**Adding a document whose `custom_id` already exists in the same scope updates it.** The
new text is chunked and each new chunk is matched to an old chunk by the digest of its
text, not by its position. A matched chunk keeps its episode, its vector and every claim
that cites it, and only its position changes. An unmatched new chunk becomes a new
episode, and only new episodes are read by the write pipeline. An old chunk that nothing
matched is removed the way a deleted document's chunks are (next paragraph). The chunker
places boundaries by local content (see `memvara.documents.chunk`), so an edit near the
top of a long document leaves most of its chunks matched.

**Deleting a document erases its text and retires, never erases, the memories it was the
only source of.** The document row, its chunk rows and its chunk episodes are erased. A
claim whose every source was one of those episodes is retired with the reason "source
document deleted", which `history()` and `why()` show; a claim that also had another
source keeps that source. Both kinds lose the erased episodes from `sources`, so no claim
cites a turn that no longer exists.

**Extraction failing does not lose the document.** The chunks are stored and indexed
before the write pipeline reads them. If extraction raises or is deferred, the document is
kept with `status="failed"` and an `error`, and its chunks stay searchable.

**Content that is not plain text goes through one seam.** A URL, `bytes`, or a mime type
that is not plain text is handed to `memvara.ingest.extract(content, url=..., mime=...)`,
which returns an object with `text`, `title` and `mime`. That package is built
separately; until it is installed, such a call raises `NotImplementedError` saying so.
"""

from __future__ import annotations

import importlib
import json
from contextlib import nullcontext
from datetime import datetime
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..redact import EPISODE
from ..types import (CUSTOM_ID_CHARS, DOCUMENT_DELETED_REASON, DOCUMENT_META,
                     DOCUMENT_STATES, Claim, DeleteResult, Document, DocumentChunk,
                     DocumentStatus, Episode, Page, Scope, close_out, content_hash, utcnow)
from .chunk import normalise, split

if TYPE_CHECKING:  # pragma: no cover
    from ..core import Memvara

__all__ = ["DocumentService", "PLAIN_TEXT", "MAX_LIST"]

#: The mime a `str` document is stored under when the caller names none.
PLAIN_TEXT = "text/plain"

#: `text/*` types that are markup rather than text, so they go through ingestion even when
#: they arrive as a `str`: their tags are not what a reader searches for.
_MARKUP = frozenset({"text/html", "text/xml"})

#: The most documents one `list_documents` page returns.
MAX_LIST = 1000

#: The longest `error` a failed document records. Enough for a provider's message, not so
#: much that a stack dump ends up in every listing.
_ERROR_CHARS = 500

#: The document store methods a backend must have to hold documents. They are optional
#: as a group; see `Store.put_document`.
_REQUIRED = ("put_document", "get_document", "find_document", "list_documents",
             "document_chunks", "put_document_chunks", "delete_document")


def _needs_ingest(content: str | bytes | None, mime: str | None) -> bool:
    if not isinstance(content, str):
        return True
    if mime is None:
        return False
    base = mime.split(";", 1)[0].strip().lower()
    return not base.startswith("text/") or base in _MARKUP


def _ingest(content: str | bytes | None, url: str | None,
            mime: str | None) -> tuple[str, str | None, str]:
    """Text, title and mime for content that is not plain text, through the seam.

    Looked up by name at call time rather than imported, because the ingestion package is
    built separately and is not in every build. Missing, it is a refusal that says what
    to pass instead, rather than an `ImportError` from inside this module.
    """
    try:
        module = importlib.import_module("memvara.ingest")
    except ImportError:
        what = (f"the URL {url!r}" if url is not None
                else "bytes" if not isinstance(content, str)
                else f"mime type {mime!r}")
        raise NotImplementedError(
            f"adding a document from {what} needs the ingestion package "
            "(memvara.ingest), which is not installed in this build. Pass the text as a "
            f"str with mime {PLAIN_TEXT!r} or 'text/markdown' instead.") from None
    got = module.extract(content, url=url, mime=mime)
    return (str(got.text), getattr(got, "title", None),
            str(getattr(got, "mime", None) or mime or PLAIN_TEXT))


def _check_custom_id(custom_id: str | None) -> None:
    if custom_id is None:
        return
    if not isinstance(custom_id, str) or not custom_id.strip():
        raise ValueError("custom_id must be a non-empty string, or None")
    if len(custom_id) > CUSTOM_ID_CHARS:
        raise ValueError(f"custom_id is {len(custom_id)} characters; the most a document "
                         f"accepts is {CUSTOM_ID_CHARS}")


def _check_filepath(filepath: str | None) -> None:
    if filepath is not None and (not isinstance(filepath, str) or not filepath.strip()):
        raise ValueError("filepath must be a non-empty '/'-separated path, or None")


def _check_meta(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    """A copy of `meta`, refused unless it is a mapping with string keys that encodes as
    JSON. Refused here rather than at the store, so nothing is half-written."""
    if meta is None:
        return {}
    if not isinstance(meta, Mapping) or not all(isinstance(k, str) for k in meta):
        raise ValueError("meta must be a mapping with string keys")
    try:
        json.dumps(dict(meta))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta must be JSON-serialisable: {exc}") from None
    return dict(meta)


def _encode_cursor(doc: Document) -> str:
    return f"{doc.created_at.isoformat()}|{doc.id}"


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    at, sep, doc_id = cursor.partition("|")
    try:
        if not sep or not doc_id:
            raise ValueError
        return datetime.fromisoformat(at), doc_id
    except ValueError:
        raise ValueError(f"cursor {cursor!r} is not one list_documents returned") from None


class DocumentService:
    """The document operations of one `Memvara`. Scope is resolved by the caller."""

    def __init__(self, mem: "Memvara") -> None:
        self.mem = mem

    @property
    def store(self) -> Any:
        """The store, refused with a message naming it when it cannot hold documents."""
        store = self.mem.store
        missing = [name for name in _REQUIRED if getattr(store, name, None) is None]
        if missing:
            raise NotImplementedError(
                f"{type(store).__name__} does not implement {', '.join(missing)}, so it "
                "cannot store documents")
        return store

    def _batch(self) -> Any:
        batch = getattr(self.mem.store, "batch", None)
        return batch() if batch is not None else nullcontext()

    # -- reading ---------------------------------------------------------------

    def resolve(self, scope: Scope, ref: str) -> Document | None:
        """The document `ref` names, by id or by `custom_id`, if this scope sees it.

        An id is tried first. A `custom_id` is looked up at this scope and then at each
        broader one, narrowest first, which is the same visibility every other read has:
        a session sees its user's documents, and never a sibling session's.
        """
        store = self.store
        doc = store.get_document(scope.tenant, ref)
        if doc is not None and scope.sees(doc.scope):
            return doc
        for s in scope.ancestors():
            doc = store.find_document(s, ref)
            if doc is not None:
                return doc
        return None

    def require(self, scope: Scope, ref: str) -> Document:
        doc = self.resolve(scope, ref)
        if doc is None:
            # One message for a missing document and one in another scope, so the error
            # cannot be used to test whether an id exists elsewhere.
            raise KeyError(f"no document {ref!r} in scope {scope.key()}")
        return doc

    def status(self, scope: Scope, ref: str) -> DocumentStatus:
        doc = self.require(scope, ref)
        return DocumentStatus(doc.id, doc.status, doc.error, doc.chunks, doc.updated_at)

    def listing(self, scope: Scope, *, filepath_prefix: str | None, status: str | None,
             limit: int, cursor: str | None) -> Page[Document]:
        if not 1 <= limit <= MAX_LIST:
            raise ValueError(f"limit must be between 1 and {MAX_LIST}, not {limit}")
        if status is not None and status not in DOCUMENT_STATES:
            raise ValueError(f"status {status!r} is not one of {DOCUMENT_STATES}")
        after = _decode_cursor(cursor) if cursor is not None else None
        # One more than asked for, so the last page is known to be the last one rather
        # than followed by an empty page.
        rows = self.store.list_documents(scope.ancestors(), filepath_prefix=filepath_prefix,
                                         status=status, limit=limit + 1, after=after)
        items = rows[:limit]
        more = len(rows) > limit
        return Page(items, _encode_cursor(items[-1]) if more else None)

    # -- writing ---------------------------------------------------------------

    def _text(self, scope: Scope, content: str | bytes | None, url: str | None,
              mime: str | None) -> tuple[str, str | None, str]:
        """The normalised, redacted text of what the caller passed, its title and mime."""
        if _needs_ingest(content, mime) or url is not None:
            text, title, mime = _ingest(content, url, mime)
        else:
            text, title, mime = str(content), None, mime or PLAIN_TEXT
        redactor = self.mem.redactor
        if redactor is not None:
            # The whole text, before it is chunked or hashed. Every chunk, every chunk
            # digest and the document digest are then computed over redacted text, so no
            # stored digest can confirm a value the redactor removed.
            text = redactor.redact(text, field=EPISODE, scope=scope)
            if title is not None:
                title = redactor.redact(title, field=EPISODE, scope=scope)
        text = normalise(text)
        if not text:
            raise ValueError("the document has no text to store")
        return text, title, mime

    def add(self, scope: Scope, content: str | bytes | None, *, url: str | None,
            custom_id: str | None, title: str | None, filepath: str | None,
            mime: str | None, meta: Mapping[str, Any] | None, extract: bool) -> Document:
        if (content is None) == (url is None):
            raise TypeError("add_document() needs exactly one of content and url")
        if url is not None and not isinstance(url, str):
            raise TypeError("url must be a string")
        _check_custom_id(custom_id)
        _check_filepath(filepath)
        fields = _check_meta(meta)
        store = self.store
        given = mime
        text, found_title, mime = self._text(scope, content, url, mime)
        redactor = self.mem.redactor
        if title is not None and redactor is not None:
            title = redactor.redact(title, field=EPISODE, scope=scope)
        existing = store.find_document(scope, custom_id) if custom_id is not None else None
        if existing is not None:
            # An update. Fields the caller did not pass keep their stored values.
            old = store.document_chunks(scope.tenant, existing.id)
            existing.title = title if title is not None else (found_title or existing.title)
            existing.filepath = filepath if filepath is not None else existing.filepath
            existing.source_uri = url if url is not None else existing.source_uri
            # Plain text sent with no mime keeps a stored plain-text type such as
            # text/markdown, which is what "fields you do not pass keep their stored
            # values" means for the one field `_text` always fills in.
            if given is not None or not isinstance(content, str) or _needs_ingest(
                    "", existing.mime):
                existing.mime = mime
            if meta is not None:
                existing.meta = fields
            return self._write(existing, text, old, extract)
        doc = Document(scope=scope, custom_id=custom_id, title=title or found_title,
                       filepath=filepath, source_uri=url, mime=mime, meta=fields)
        return self._write(doc, text, [], extract)

    def update(self, scope: Scope, ref: str, *, content: str | bytes | None,
               title: str | None, meta: Mapping[str, Any] | None,
               filepath: str | None, extract: bool) -> Document:
        _check_filepath(filepath)
        fields = _check_meta(meta)
        doc = self.require(scope, ref)
        redactor = self.mem.redactor
        if title is not None:
            doc.title = (title if redactor is None
                         else redactor.redact(title, field=EPISODE, scope=doc.scope))
        if filepath is not None:
            doc.filepath = filepath
        if meta is not None:
            doc.meta = fields
        if content is None:
            doc.updated_at = utcnow()
            self.store.put_document(doc)
            return doc
        # New content is read under the stored mime. When it goes through ingestion, the
        # mime ingestion reports is the one kept, because bytes say nothing about their
        # own type and the extractor is what found out.
        text, found_title, doc.mime = self._text(doc.scope, content, None, doc.mime)
        if title is None and found_title:
            doc.title = found_title
        return self._write(doc, text, self.store.document_chunks(doc.scope.tenant, doc.id),
                           extract)

    def _write(self, doc: Document, text: str, old: Sequence[DocumentChunk],
               extract: bool) -> Document:
        """Chunk `text`, match it against `old`, store the difference, then extract."""
        store = self.store
        now = utcnow()
        pieces = split(text) if self.mem.retrieval_chunks else [text]
        unused: dict[str, list[DocumentChunk]] = {}
        for chunk in old:
            unused.setdefault(chunk.hash, []).append(chunk)
        episode_of: dict[str, str] = {}
        fresh: list[Episode] = []
        chunks: list[DocumentChunk] = []
        for position, piece in enumerate(pieces):
            digest = content_hash(piece)
            if unused.get(digest):
                # The same text as an old chunk: keep its episode, vector and citations.
                episode_id = unused[digest].pop(0).episode_id
            elif digest in episode_of:
                # The same text twice in this document: one episode serves both.
                episode_id = episode_of[digest]
            else:
                ep = Episode(content=piece, scope=doc.scope, role="system", ts=now,
                             meta={DOCUMENT_META: doc.id})
                fresh.append(ep)
                episode_id = ep.id
            episode_of[digest] = episode_id
            chunks.append(DocumentChunk(position, piece, digest, episode_id))
        kept = {c.episode_id for c in chunks}
        dropped = list(dict.fromkeys(c.episode_id for c in old if c.episode_id not in kept))

        doc.content_hash = content_hash(text)
        doc.status, doc.error, doc.updated_at = "queued", None, now
        with self._batch():
            store.put_document(doc)
            for ep in fresh:
                store.add_episode(ep)
            store.put_document_chunks(doc.scope.tenant, doc.id, chunks)
            self._release(doc.scope.tenant, dropped, now)
            for episode_id in dropped:
                store.erase_episode(episode_id, cited=True)
        doc.chunks = len(chunks)
        # A transaction of its own, after the rows are durable, as `Memvara.add` does:
        # encoding must not hold the write lock, and a turn without a vector is still
        # found by text search until a retry fills it in.
        with self._batch():
            self.mem._index_episodes([ep.id for ep in fresh])
        self._extract(doc, fresh if extract else [])
        return doc

    def _extract(self, doc: Document, fresh: Sequence[Episode]) -> None:
        """Run the write pipeline over the new chunks and record how it went."""
        if fresh:
            doc.status = "extracting"
            self.store.put_document(doc)
            try:
                receipt = self.mem.writer.reextract(list(fresh))
            except Exception as exc:
                # Deliberately broad. The chunks are already stored and indexed, and a
                # document is worth keeping whatever the model did; the failure is
                # recorded on the document instead of being raised past it.
                doc.status = "failed"
                doc.error = f"{type(exc).__name__}: {exc}"[:_ERROR_CHARS]
            else:
                if receipt.deferred:
                    doc.status = "failed"
                    doc.error = ("extraction was deferred because the model call did not "
                                 "complete. The chunks are stored and searchable, and "
                                 "Memvara.reextract() can retry them.")
                else:
                    doc.status = "done"
        else:
            doc.status = "done"
        doc.updated_at = utcnow()
        self.store.put_document(doc)

    def _release(self, tenant: str, episode_ids: Sequence[str],
                 at: datetime) -> tuple[list[str], list[str]]:
        """Detach every claim from episodes about to be erased.

        A claim every one of whose sources is among them is retired first, with the reason
        "source document deleted". A claim with another source keeps that one. Either way
        the erased episodes leave `sources`, so no claim cites a turn that is gone.
        Returns the ids retired and the ids that kept another source.
        """
        doomed = set(episode_ids)
        citing: dict[str, Claim] = {}
        for episode_id in episode_ids:
            for claim in self.mem.store.claims_citing(tenant, episode_id):
                citing.setdefault(claim.id, claim)
        retired: list[str] = []
        unlinked: list[str] = []
        for claim in citing.values():
            rest = [s for s in claim.sources if s not in doomed]
            if rest:
                unlinked.append(claim.id)
            elif claim.invalidated_at is None:
                close_out(claim, at, None, "retired", DOCUMENT_DELETED_REASON)
                retired.append(claim.id)
            claim.sources = rest
            self.mem.store.put_claim(claim)
        return retired, unlinked

    def delete(self, scope: Scope, ref: str) -> DeleteResult:
        doc = self.resolve(scope, ref)
        if doc is None:
            return DeleteResult(id=ref, deleted=False)
        store = self.store
        tenant = doc.scope.tenant
        episode_ids = list(dict.fromkeys(
            c.episode_id for c in store.document_chunks(tenant, doc.id)))
        with self._batch():
            retired, unlinked = self._release(tenant, episode_ids, utcnow())
            # The rows first: erasing an episode also removes the chunk row that repeats
            # its text, and the count reported is the document's own chunk rows.
            chunks = store.delete_document(tenant, doc.id)
            episodes = sum(bool(store.erase_episode(e, cited=True)) for e in episode_ids)
        return DeleteResult(id=doc.id, deleted=True, custom_id=doc.custom_id,
                            chunks=chunks, episodes=episodes, retired=tuple(retired),
                            unlinked=tuple(unlinked))
