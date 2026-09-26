"""Three bugs the upgrade tests found, each pinned as a strict expected failure that
accepts only its own symptom (#299, #300, #301)."""

from __future__ import annotations

import gc
import pathlib
import sys
import warnings

import pytest

from memvara import EmbedderMismatchError, Memvara, NullLLM
from memvara.embed import HashingEmbedder
from memvara.store.sqlite import SCHEMA_VERSION

from harness import known_bugs, stores

from . import golden
from .test_adv_upgrade_refusals import NEWER, newer_store, serve


@known_bugs.xfail("B22")
def test_the_server_refuses_a_store_from_a_newer_version_without_a_traceback(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """The server should exit with status 2 and one `memvara-mcp:` line, as it does for
    an embedder mismatch. Today it exits with status 1 and a traceback that ends in the
    RuntimeError from the store, because the command line does not catch it."""
    db = newer_store(tmp_path)
    done = serve(db, home)
    if (done.returncode == 1 and "Traceback" in done.stderr
            and "RuntimeError: " in done.stderr and f"schema version {NEWER}" in done.stderr):
        raise known_bugs.Reproduced(done.stderr.splitlines()[-1])
    assert done.returncode == 2, done.stderr
    assert done.stderr.startswith("memvara-mcp: ")
    assert f"schema version {NEWER}" in done.stderr
    assert "Traceback" not in done.stderr


@known_bugs.xfail("B23")
@pytest.mark.parametrize("tag", [t for t, v in golden.RELEASES.items() if v < SCHEMA_VERSION])
def test_a_store_refused_for_its_embedder_is_left_at_its_old_version(
        tag: str, tmp_path: pathlib.Path) -> None:
    """A refused open should leave the file as it was. Today the constructor migrates the
    store and commits before the embedder check refuses it, so the release that wrote
    the store can no longer open it."""
    db = golden.unpack(tag, tmp_path)
    with pytest.raises(EmbedderMismatchError):
        Memvara(str(db), embedder=HashingEmbedder(dim=256), llm=NullLLM())
    after = golden.schema_version(db)
    if after == SCHEMA_VERSION:
        raise known_bugs.Reproduced(f"{tag}: version {golden.RELEASES[tag]} became {after}")
    assert after == golden.RELEASES[tag]


def refuse(db: pathlib.Path) -> None:
    """Open `db` and expect the newer-version refusal, keeping no reference to it."""
    try:
        stores.file(db)
    except RuntimeError:
        return
    raise AssertionError("the store from a newer version was opened")


@pytest.mark.skipif(sys.version_info < (3, 13),
                    reason="Python before 3.13 does not report an unclosed SQLite connection")
@known_bugs.xfail("B24")
def test_a_refused_open_closes_the_connection_it_opened(tmp_path: pathlib.Path) -> None:
    """A failed open must not hold the database file until garbage collection. Today the
    constructor releases its presence lock but never closes its connection, which
    Python 3.13 reports as an unclosed database when the connection is collected."""
    db = newer_store(tmp_path)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        refuse(db)
        gc.collect()
    leaked = [str(w.message) for w in seen if issubclass(w.category, ResourceWarning)]
    if leaked and all("unclosed database" in message for message in leaked):
        raise known_bugs.Reproduced(leaked[0])
    assert leaked == []
