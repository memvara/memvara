"""Three bugs the upgrade tests found, all fixed (#299, #300, #301). Each test that pinned
one runs as a normal test."""

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


def test_the_server_refuses_a_store_from_a_newer_version_without_a_traceback(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """The server exits with status 2 and one `memvara-mcp:` line, as it does for an
    embedder mismatch. It used to exit with status 1 and a traceback that ends in the
    RuntimeError from the store, because the command line did not catch it (#299)."""
    db = newer_store(tmp_path)
    done = serve(db, home)
    if (done.returncode == 1 and "Traceback" in done.stderr
            and "RuntimeError: " in done.stderr and f"schema version {NEWER}" in done.stderr):
        raise known_bugs.Reproduced(done.stderr.splitlines()[-1])
    assert done.returncode == 2, done.stderr
    assert done.stderr.startswith("memvara-mcp: ")
    assert f"schema version {NEWER}" in done.stderr
    assert "Traceback" not in done.stderr


@pytest.mark.parametrize("tag", [t for t, v in golden.RELEASES.items() if v < SCHEMA_VERSION])
def test_a_store_refused_for_its_embedder_is_left_at_its_old_version(
        tag: str, tmp_path: pathlib.Path) -> None:
    """A refused open leaves the file as it was. The constructor used to migrate the store
    and commit before the embedder check refused it, so the release that wrote the store
    could no longer open it (#300). The check now runs before the upgrade."""
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
def test_a_refused_open_closes_the_connection_it_opened(tmp_path: pathlib.Path) -> None:
    """A failed open must not hold the database file until garbage collection. The
    constructor used to release its presence lock but not close its connection, which
    Python 3.13 reports as an unclosed database when the connection is collected (#301)."""
    db = newer_store(tmp_path)
    # Collected first, so that a connection an earlier test in this process left open is
    # not reported here: the warning cannot say which store its connection belonged to.
    gc.collect()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        refuse(db)
        gc.collect()
    leaked = [str(w.message) for w in seen if issubclass(w.category, ResourceWarning)]
    if leaked and all("unclosed database" in message for message in leaked):
        raise known_bugs.Reproduced(leaked[0])
    assert leaked == []
