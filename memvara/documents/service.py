"""The document store behind `Memvara.add_document` and the methods beside it.

A document is stored as a row that describes it and a list of chunks that hold its text.
Each chunk is an ordinary episode with `role="system"` and `meta["document_id"]` set, so
the episode text index, the episode vectors and `search(include_episodes=True)` serve
documents without a second index, and `why()` on a claim extracted from a document quotes
the chunk it came from.

Four behaviours are the reason this module exists rather than a loop over `add()`.

**New chunks are read for facts.** With `extract=True`, the default, every new chunk goes
through the write pipeline's extraction. The salience gate accepts a document chunk
whatever its role; the fast path, which reads first-person sentences as the user's own,
does not run on it, because it reads only user turns. So facts come from the model tier,
and a store with no model stores and indexes the document and extracts nothing from it.

**Adding a document whose `custom_id` already exists in the same scope updates it.** The
new text is chunked and each new chunk is matched to an old chunk by the digest of its
text, not by its position. A matched chunk keeps its episode, its vector and every claim
that cites it, and only its position changes. An unmatched new chunk becomes a new
episode, and only new episodes are read by the write pipeline, except after a failed or
skipped extraction, when kept chunks that no claim cites yet are read again. An old chunk
that nothing matched is removed the way a deleted document's chunks are (next paragraph).
The chunker places boundaries by local content (see `memvara.documents.chunk`), so an
edit near the top of a long document leaves most of its chunks matched.

**Deleting a document erases its text and retires, never erases, the memories it was the
only source of.** The document row, its chunk rows and its chunk episodes are erased. A
claim whose every source was one of those episodes is retired with the reason "source
document deleted", which `history()` and `why()` show; a claim that also had another
source keeps that source. Both kinds lose the erased episodes from `sources`, so no claim
cites a turn that no longer exists.

**The status says what happened.** `queued` when written, `extracting` while the pipeline
runs, then `done` when every chunk has been read, `stored` when some chunk was kept
unread because the caller passed `extract=False`, or `failed` with an `error`. A failed
document is still stored and its chunks are still searchable.

**Content that is not plain text goes through one seam.** A URL, `bytes`, or a mime type
that is not plain text is handed to `memvara.ingest.extract(content, url=..., mime=...)`,
which returns an object with `text`, `title` and `mime`. That package is built
separately; until it is installed, such a call raises `NotImplementedError` saying so.
"""

from __future__ import annotations

import importlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple, Sequence

from ..redact import EPISODE
from ..store import transaction
from ..types import (CUSTOM_ID_CHARS, DOCUMENT_DELETED_REASON, DOCUMENT_EXTRACT,
                     DOCUMENT_META, DOCUMENT_STATES, REASON_CHARS, Claim, DeleteResult,
                     Document, DocumentChunk, DocumentState, DocumentStatus, Episode,
                     Page, Scope, close_out, content_hash, one_source, utcnow)
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


def is_plain_text(mime: str) -> bool:
    """Whether a mime names text a document can store as it is, such as `text/plain` or
    `text/markdown`, rather than something ingestion has to read.

    >>> is_plain_text("text/markdown; charset=utf-8"), is_plain_text("text/html")
    (True, False)
    """
    base = mime.split(";", 1)[0].strip().lower()
    return base.startswith("text/") and base not in _MARKUP


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


class _Stored(NamedTuple):
    """What `_store` wrote, and what `_finish` needs to index and extract it."""

    doc: Document
    fresh: list[Episode]
    kept: list[str]
    previous: DocumentState | None
    previous_error: str | None
    extract: bool


class DocumentService:
    """The document operations of one `Memvara`. Scope is resolved by the caller."""

    def __init__(self, mem: "Memvara") -> None:
        self.mem = mem

    @property
    def store(self) -> Any:
        """The store, refused with a message naming it when it cannot hold documents.

        Asked through `holds_documents`, a marker a store sets when it implements the
        document methods, rather than through the methods' presence. `RemoteStore` has
        every method, each raising, so a presence check would pass and the caller would
        meet the stub's message about REST routes instead of this one.
        """
        store = self.mem.store
        if getattr(store, "holds_documents", False) is not True:
            raise NotImplementedError(
                f"{type(store).__name__} cannot store documents: it does not set "
                "holds_documents. Use a store that does, such as SQLiteStore, or "
                "RemoteMemvara.add_document() against a hosted deployment.")
        return store

    def _redact(self, text: str | None, scope: Scope) -> str | None:
        """`text` through the configured redactor, or unchanged when there is none."""
        redactor = self.mem.redactor
        if text is None or redactor is None:
            return text
        return redactor.redact(text, field=EPISODE, scope=scope)

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
        return DocumentStatus.of(self.require(scope, ref))

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
        """The normalised, redacted text of what the caller passed, its title and mime.

        The whole text is redacted before it is chunked or hashed, so every chunk, every
        chunk digest and the document digest are computed over redacted text, and no
        stored digest can confirm a value the redactor removed.
        """
        if url is not None or not isinstance(content, str) or (
                mime is not None and not is_plain_text(mime)):
            text, title, mime = _ingest(content, url, mime)
        else:
            text, title, mime = content, None, mime or PLAIN_TEXT
        text = normalise(self._redact(text, scope) or "")
        if not text:
            raise ValueError("the document has no text to store")
        return text, self._redact(title, scope), mime

    def add(self, scope: Scope, content: str | bytes | None, *, url: str | None,
            custom_id: str | None, title: str | None, filepath: str | None,
            mime: str | None, meta: Mapping[str, Any] | None, extract: bool) -> Document:
        one_source(content, url)
        if url is not None and not isinstance(url, str):
            raise TypeError("url must be a string")
        _check_custom_id(custom_id)
        _check_filepath(filepath)
        fields = _check_meta(meta)
        store = self.store
        text, found_title, found_mime = self._text(scope, content, url, mime)
        title = self._redact(title, scope)
        # The lookup and the rows it decides between are written in one transaction, so
        # two concurrent adds of one `custom_id` cannot both miss the lookup and race to
        # insert; the second waits, finds the first, and becomes an update of it.
        # Indexing and extraction run after it, so no model call holds the write lock.
        with transaction(store):
            stored = self._store_add(scope, text, custom_id=custom_id, title=title,
                                     found_title=found_title, filepath=filepath, url=url,
                                     mime=mime, found_mime=found_mime, meta=meta,
                                     fields=fields, extract=extract)
        return self._finish(stored)

    def _store_add(self, scope: Scope, text: str, *, custom_id: str | None,
                   title: str | None, found_title: str | None, filepath: str | None,
                   url: str | None, mime: str | None, found_mime: str,
                   meta: Mapping[str, Any] | None, fields: dict[str, Any],
                   extract: bool) -> _Stored:
        """`add`'s lookup and its rows. The caller holds a transaction around both."""
        store = self.store
        existing = store.find_document(scope, custom_id) if custom_id is not None else None
        if existing is None:
            doc = Document(scope=scope, custom_id=custom_id,
                           title=title if title is not None else found_title,
                           filepath=filepath, source_uri=url, mime=found_mime,
                           meta=fields)
            return self._store(doc, text, [], extract)
        # An update. Fields the caller did not pass keep their stored values.
        if title is not None or found_title is not None:
            existing.title = title if title is not None else found_title
        if filepath is not None:
            existing.filepath = filepath
        if url is not None:
            existing.source_uri = url
        # Plain text sent with no mime keeps a stored plain-text type such as
        # text/markdown; anything else takes the mime it was read under.
        if (mime is not None or found_mime != PLAIN_TEXT
                or not is_plain_text(existing.mime)):
            existing.mime = found_mime
        if meta is not None:
            existing.meta = fields
        return self._store(existing, text,
                           store.document_chunks(scope.tenant, existing.id), extract)

    def update(self, scope: Scope, ref: str, *, content: str | bytes | None,
               title: str | None, meta: Mapping[str, Any] | None,
               filepath: str | None, mime: str | None, extract: bool) -> Document:
        _check_filepath(filepath)
        fields = _check_meta(meta)
        doc = self.require(scope, ref)
        if title is not None:
            doc.title = self._redact(title, doc.scope)
        if filepath is not None:
            doc.filepath = filepath
        if meta is not None:
            doc.meta = fields
        if content is None:
            if mime is not None:
                doc.mime = mime
            doc.updated_at = utcnow()
            self.store.put_document(doc)
            return doc
        # The mime the new content is read under: the one passed; otherwise the stored
        # one for text when that is a text type; otherwise none, so ingestion detects it.
        # Reading new bytes under a stale stored type would hand a PDF's replacement,
        # say an image, to the PDF reader.
        if mime is None and isinstance(content, str) and is_plain_text(doc.mime):
            mime = doc.mime
        text, found_title, doc.mime = self._text(doc.scope, content, None, mime)
        if title is None and found_title is not None:
            doc.title = found_title
        return self._finish(self._store(
            doc, text, self.store.document_chunks(doc.scope.tenant, doc.id), extract))

    def _store(self, doc: Document, text: str, old: Sequence[DocumentChunk],
               extract: bool) -> _Stored:
        """Chunk `text`, match it against `old`, and store the difference, in one
        transaction. `_finish` indexes and extracts what this wrote."""
        store = self.store
        now = utcnow()
        previous, previous_error = (doc.status, doc.error) if old else (None, None)
        pieces = split(text) if self.mem.retrieval_chunks else [text]
        unused: dict[str, list[DocumentChunk]] = {}
        for chunk in old:
            unused.setdefault(chunk.hash, []).append(chunk)
        episode_of: dict[str, str] = {}
        fresh: list[Episode] = []
        chunks: list[DocumentChunk] = []
        meta: dict[str, Any] = {DOCUMENT_META: doc.id}
        if not extract:
            meta[DOCUMENT_EXTRACT] = False
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
                             meta=dict(meta))
                fresh.append(ep)
                episode_id = ep.id
            episode_of[digest] = episode_id
            chunks.append(DocumentChunk(position, piece, digest, episode_id))
        kept = {c.episode_id for c in chunks}
        dropped = list(dict.fromkeys(c.episode_id for c in old if c.episode_id not in kept))

        doc.content_hash = content_hash(text)
        doc.status, doc.error, doc.updated_at = "queued", None, now
        with transaction(store):
            store.put_document(doc)
            for ep in fresh:
                store.add_episode(ep)
            store.put_document_chunks(doc.scope.tenant, doc.id, chunks)
            self._release(doc.scope.tenant, dropped, now)
            store.erase_episodes(dropped, cited=True)
        doc.chunks = len(chunks)
        fresh_ids = {ep.id for ep in fresh}
        return _Stored(doc, fresh, [c.episode_id for c in chunks
                                    if c.episode_id not in fresh_ids],
                       previous, previous_error, extract)

    def _finish(self, stored: _Stored) -> Document:
        """Index the new chunks, then extract, and record the outcome as the status."""
        doc, fresh, kept, previous, previous_error, extract = stored
        store = self.store
        # A transaction of its own, after the rows are durable, as `Memvara.add` does:
        # encoding must not hold the write lock, and a turn without a vector is still
        # found by text search until a retry fills it in.
        with transaction(store):
            self.mem._index_episodes([ep.id for ep in fresh])
        if not extract:
            # Nothing is read. A new chunk left unread makes the document `stored`; an
            # unchanged document keeps the outcome it already had, error included,
            # because nothing about it was tried again.
            if fresh or previous not in ("done", "failed"):
                doc.status = "stored"
            else:
                doc.status, doc.error = previous, previous_error
            doc.updated_at = utcnow()
            store.put_document(doc)
            return doc
        retry = [] if previous in (None, "done") else self._unread(doc, kept)
        self._extract(doc, fresh + retry)
        return doc

    def _unread(self, doc: Document, episode_ids: Sequence[str]) -> list[Episode]:
        """Kept chunks that no claim cites yet, readied to be read again.

        Called when the document's last extraction failed or was skipped, so a re-add of
        the same text retries what was not read rather than reporting `done` over it. A
        chunk stored with `extract=False` loses that mark, because this call asked for
        extraction. A chunk the model already read and found nothing in is read once
        more, which is the cost of not recording attempts on the episode.
        """
        store = self.store
        cited = {s for c in store.claims_citing_any(doc.scope.tenant, episode_ids)
                 for s in c.sources}
        found = store.get_episodes([e for e in dict.fromkeys(episode_ids)
                                    if e not in cited])
        out: list[Episode] = []
        with transaction(store):
            for ep in found.values():
                if DOCUMENT_EXTRACT in ep.meta:
                    ep.meta = {k: v for k, v in ep.meta.items() if k != DOCUMENT_EXTRACT}
                    store.add_episode(ep)
                out.append(ep)
        return out

    def _extract(self, doc: Document, episodes: Sequence[Episode]) -> None:
        """Run the write pipeline over `episodes` and record how it went. With nothing
        left to read, the document is `done`."""
        doc.status = "done"
        if episodes:
            doc.status = "extracting"
            self.store.put_document(doc)
            try:
                receipt = self.mem.writer.reextract(list(episodes))
            except Exception as exc:
                # Deliberately broad. The chunks are already stored and indexed, and a
                # document is worth keeping whatever the model did; the failure is
                # recorded on the document instead of being raised past it.
                doc.status = "failed"
                doc.error = f"{type(exc).__name__}: {exc}"[:REASON_CHARS]
            else:
                if receipt.deferred:
                    doc.status = "failed"
                    doc.error = ("extraction was deferred because the model call did not "
                                 "complete. The chunks are stored and searchable; adding "
                                 "the document again retries them.")
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

        One `claims_citing_any` query finds every citer. The claims are then written one
        `put_claim` each inside the caller's transaction, which commits once; the store
        protocol has no bulk claim write, and one would run the same statements.
        """
        if not episode_ids:
            return [], []
        doomed = set(episode_ids)
        retired: list[str] = []
        unlinked: list[str] = []
        for claim in self.mem.store.claims_citing_any(tenant, episode_ids):
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
        with transaction(store):
            retired, unlinked = self._release(tenant, episode_ids, utcnow())
            # The rows first: a chunk episode is protected while its document lists it.
            chunks = store.delete_document(tenant, doc.id)
            episodes = store.erase_episodes(episode_ids, cited=True)
        return DeleteResult(id=doc.id, deleted=True, custom_id=doc.custom_id,
                            chunks=chunks, episodes=episodes, retired=tuple(retired),
                            unlinked=tuple(unlinked))

