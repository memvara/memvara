"""Stores this code cannot read are refused, with a message and without a traceback.

Two kinds: a store written by a newer version than this code knows, and a store opened
with an embedder that did not write it. The newer store is made by raising the schema
version of the newest committed store, as a later release's migration would.
"""

from __future__ import annotations

import pathlib
import sqlite3
import subprocess
import sys

import pytest

from memvara import EmbedderChangedWarning, EmbedderMismatchError, Memvara, NullLLM
from memvara.embed import HashingEmbedder
from memvara.store.sqlite import SCHEMA_VERSION

from harness import stores
from harness.env import child_env

from . import golden

NEWER = SCHEMA_VERSION + 1


def newer_store(tmp_path: pathlib.Path) -> pathlib.Path:
    """The newest committed store, stamped with a schema version past this code's."""
    db = golden.unpack(golden.TAGS[-1], tmp_path)
    raw = sqlite3.connect(db)
    raw.execute(f"PRAGMA user_version = {NEWER}")
    raw.commit()
    raw.close()
    return db


def serve(db: pathlib.Path, home: pathlib.Path,
          **extra: str) -> subprocess.CompletedProcess[str]:
    """Start the MCP server on `db` with nothing on its input, and wait for it to exit.

    The server is told to write UTF-8. Otherwise, on Windows, it writes the em dash in
    its messages in the console's code page, and reading that as UTF-8 fails."""
    env = child_env(home, {"MEMVARA_DB": str(db), "PYTHONIOENCODING": "utf-8", **extra})
    return subprocess.run([sys.executable, "-m", "memvara.server"], input="",
                          capture_output=True, text=True, encoding="utf-8", env=env,
                          timeout=60, check=False)


def test_a_store_from_a_newer_version_is_refused_and_left_as_it_was(
        tmp_path: pathlib.Path) -> None:
    db = newer_store(tmp_path)
    before = golden.snapshot(db)
    for attempt in ("first", "second"):
        with pytest.raises(RuntimeError) as refused:
            stores.file(db)
        message = str(refused.value)
        assert f"schema version {NEWER} was written by a newer Memvara" in message, attempt
        assert f"this build understands {SCHEMA_VERSION}" in message, attempt
    assert golden.changes(before, golden.snapshot(db)) == []


@pytest.mark.parametrize("tag", golden.TAGS)
def test_an_old_store_is_refused_by_an_embedder_of_another_width(
        tag: str, tmp_path: pathlib.Path) -> None:
    db = golden.unpack(tag, tmp_path)
    with pytest.raises(EmbedderMismatchError,
                       match="512-dimensional vectors, written by hashing:512"):
        Memvara(str(db), embedder=HashingEmbedder(dim=256), llm=NullLLM())


@pytest.mark.parametrize("tag", golden.TAGS)
def test_an_old_store_warns_when_another_embedder_of_its_width_opens_it(
        tag: str, tmp_path: pathlib.Path) -> None:
    """Documented: an embedder of the same width but another vector space is warned
    about, not refused, because nothing can raise on it (`EmbedderChangedWarning` and
    `Memvara._check_embedder` in memvara/core.py). The old release's embedder record
    must still be read for the warning to fire."""
    db = golden.unpack(tag, tmp_path)
    with pytest.warns(EmbedderChangedWarning, match="unrelated vector spaces"):
        Memvara(str(db), embedder=HashingEmbedder(dim=512, ngram=(2, 4)),
                llm=NullLLM()).close()


def test_the_server_refuses_an_old_store_with_an_embedder_of_another_width(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    db = golden.unpack(golden.TAGS[0], tmp_path)
    done = serve(db, home, MEMVARA_EMBEDDER="hashing:256")
    assert done.returncode == 2, done.stderr
    assert done.stderr.startswith("memvara-mcp: ")
    assert "512-dimensional" in done.stderr
    assert "Traceback" not in done.stderr
