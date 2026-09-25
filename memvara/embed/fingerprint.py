"""Which embedder wrote a store's vectors, and whether yours can read them.

Vectors are only comparable to vectors from the same model. A store therefore has an
embedder baked into it, and swapping the embedder silently invalidates every vector in
it. That happens by accident more or less as designed today: `default_embedder()`
prefers a 384-dimensional sentence-transformers model when one is installed and falls
back to the 512-dimensional `HashingEmbedder` when it is not, so `pip install
memvara[local-embed]` — advice the README gives — changes the embedder of an existing
store on the next process start.

Two failure shapes, one detectable and one not:

* **Different dimension.** Every read raises `query dim 384 != index dim 512`, while
  writes keep succeeding, so the store keeps growing and none of it is searchable.
  Loud, but only at read time and only after the damage is done.
* **Same dimension, different model.** Nothing raises at all. Retrieval just returns
  the wrong claims forever, because the query vector and the stored vectors are in
  unrelated spaces that happen to have the same width.

The second is why this module records a *name* and not only a dimension.

Where it is recorded: a small JSON file next to the store file. The `Store` protocol has
no metadata surface, and inventing one belongs to whoever owns storage — so this is
deliberately advisory. A store with no file beside it, such as an in-memory store, has no
record, and identity checking degrades to the dimension check, which is derived from the
stored vectors themselves and cannot be lost. When a store file's record is missing or
unreadable (the database was copied without it, or a crash tore it while an older
version wrote it), nothing can tell whether the store's vectors came from the embedder in
use. When such a store holds vectors, `Memvara` warns as it opens it, and records the
embedder in use so that the next change is noticed.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import _name_of

__all__ = ["EmbedderFingerprint", "embedder_name", "fingerprint_of", "local_model",
           "read_fingerprint", "read_fingerprint_at", "write_fingerprint", "stored_dim",
           "SIDECAR_SUFFIX"]

SIDECAR_SUFFIX = ".embedder.json"

# How many claims to look at before giving up on finding one with a vector. A store can
# legitimately hold claims that were never embedded, but it cannot hold many of them, so
# a bounded scan is the difference between an O(1) construction check and one that reads
# the whole claims table on a store with no vectors at all.
_PROBE_LIMIT = 32

embedder_name = _name_of


@dataclass(frozen=True, slots=True)
class EmbedderFingerprint:
    """The identity of the embedder a set of vectors was produced by."""

    name: str
    dim: int

    def __str__(self) -> str:
        return f"{self.name} (dim {self.dim})"

    def __repr__(self) -> str:
        return f"<EmbedderFingerprint {self}>"


def fingerprint_of(embedder: Any) -> EmbedderFingerprint:
    """This embedder's identity. Named for the module so `memvara.embed.fingerprint`
    keeps meaning the module and not a function shadowing it."""
    return EmbedderFingerprint(embedder_name(embedder), int(embedder.dim))


def sidecar_path(store: Any) -> str | None:
    """Where this store's fingerprint lives, or None if it cannot have one.

    An in-memory store has nothing to persist alongside, and neither does a third-party
    store that does not expose a path — in both cases the answer is "no identity on
    record", which the caller handles by falling back to the dimension check.
    """
    path = getattr(store, "path", None)
    if not isinstance(path, str) or not path or path.startswith(":memory:"):
        return None
    return path + SIDECAR_SUFFIX


def read_fingerprint(store: Any) -> EmbedderFingerprint | None:
    """The recorded identity of whatever wrote this store's vectors, if known."""
    path = sidecar_path(store)
    if path is None:
        return None
    return _read_sidecar(path)


def read_fingerprint_at(db_path: str) -> EmbedderFingerprint | None:
    """`read_fingerprint` for a store file that is not open yet.

    The MCP server builds its embedder before it opens the store, and `local` there means
    the local model this store was written by, so it reads the record by path. A record
    beside no store file is a leftover from a deleted one and describes nothing, so it
    reads as none.
    """
    if not db_path or db_path.startswith(":memory:") or not os.path.exists(db_path):
        return None
    return _read_sidecar(db_path + SIDECAR_SUFFIX)


def local_model(fp: EmbedderFingerprint | None) -> str | None:
    """The sentence-transformers model a `local:MODEL` fingerprint names, else None.

    >>> local_model(EmbedderFingerprint("local:BAAI/bge-small-en-v1.5", 384))
    'BAAI/bge-small-en-v1.5'
    >>> local_model(EmbedderFingerprint("hashing:512:3-5", 512)) is None
    True
    """
    if fp is None or not fp.name.startswith("local:"):
        return None
    return fp.name[len("local:"):] or None


def _read_sidecar(path: str) -> EmbedderFingerprint | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return EmbedderFingerprint(str(data["embedder"]), int(data["dim"]))
    except (OSError, ValueError, KeyError, TypeError):
        # Unreadable, absent or corrupt: all the same answer. A damaged advisory file
        # must never be the reason a memory store refuses to open.
        return None


def write_fingerprint(store: Any, fp: EmbedderFingerprint) -> bool:
    """Record who owns this store's vector space. Best-effort by construction.

    The record is written to a temporary file beside it, which is then renamed over it,
    so a crash or a full disk during the write leaves the old record or the new one and
    never half of one. This matters most when `reembed()` rewrites the record, because
    the store then already holds vectors, and a torn record reads as no record at all. A
    crash in the middle of the write can leave the temporary file behind; nothing reads
    it.

    Returns whether it was written, which is information for tests rather than for
    callers: a read-only directory is a fine place to keep a memory store, and losing
    the identity check there is a smaller harm than refusing to run.
    """
    path = sidecar_path(store)
    if path is None:
        return False
    # A name of its own for each write, so two processes writing at once never write
    # into one temporary file.
    staged = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        fh = open(staged, "x", encoding="utf-8")
    except OSError:
        return False
    try:
        with fh:
            json.dump({"embedder": fp.name, "dim": fp.dim}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(staged, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.remove(staged)
        return False
    return True


def stored_dim(store: Any) -> int | None:
    """Width of the vectors already in `store`, or None if it holds none.

    Unlike the recorded name this cannot be lost or falsified — it is read back from
    the vectors themselves — so it is the check that actually protects the upgrade
    path.
    """
    # Fast path: the shipped store keeps an index that already knows. Private, and
    # guarded accordingly: it is an optimization over the protocol-only probe below,
    # not a requirement.
    index = getattr(store, "_vec", None)
    dim = getattr(index, "dim", None)
    # `index is not None` is implied by `dim` being an int — a missing index yields a
    # missing `dim` — but implied is not stated, and the `len()` below is what a reader
    # (and a checker) has to justify without leaving the line.
    if index is not None and isinstance(dim, int) and len(index) > 0:
        return dim

    get_embedding = getattr(store, "get_embedding", None)
    iter_claims = getattr(store, "iter_claims", None)
    if get_embedding is None or iter_claims is None:
        return None
    try:
        for scanned, claim in enumerate(iter_claims(include_invalidated=True)):
            vec = get_embedding(claim.id)
            if vec is not None:
                return int(np.asarray(vec).reshape(-1).shape[0])
            if scanned + 1 >= _PROBE_LIMIT:
                break
    except NotImplementedError:
        # A `Store` whose `iter_claims`/`get_embedding` exist as methods (so the
        # `getattr` checks above pass) but whose backing surface has no way to answer —
        # `RemoteStore`, whose vectors and claim listing are both server-internal, in
        # particular. The probe exists to protect a store this process can actually
        # read from a silent width mismatch; a store that flatly cannot be probed is
        # not that case, so this falls back to "unknown" exactly like a store with no
        # vectors yet, rather than making every `Memvara(store=RemoteStore(...))`
        # unconstructible.
        return None
    return None
