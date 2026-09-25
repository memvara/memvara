"""Stores in the test process, built the way every test in this repository builds one:
with the hashing embedder, and with no model."""

from __future__ import annotations

import pathlib
from typing import Any

from memvara import Memvara, NullLLM
from memvara.embed import HashingEmbedder


def memory(**options: Any) -> Memvara:
    """An in-memory store."""
    return Memvara(embedder=HashingEmbedder(dim=512), llm=NullLLM(), **options)


def file(path: pathlib.Path, **options: Any) -> Memvara:
    """A store in the SQLite file at `path`. A server started with MEMVARA_DB=`path` and
    the child environment's MEMVARA_EMBEDDER=hashing opens it in the same vector space."""
    return Memvara(str(path), embedder=HashingEmbedder(dim=512), llm=NullLLM(), **options)
