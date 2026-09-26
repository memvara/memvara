"""The first open of a new store, while another process is creating the same store.

Creating a store switches the new file to WAL mode, and that switch needs a stronger lock.
While another connection holds the write lock, SQLite refuses the switch at once instead
of waiting. The open used to fail in a millisecond, where every other write waits out the
five-second busy timeout (#281). It now tries the switch again within that timeout. The
nightly tier checks the same case between real processes that open one new store at the
same moment.
"""

from __future__ import annotations

import pathlib
import sqlite3
import threading
import time

from harness import stores

HOLD = 0.5      # how long the other connection keeps the write lock, in seconds


def test_opening_a_store_another_connection_is_writing_waits_for_it(
        tmp_path: pathlib.Path) -> None:
    db = tmp_path / "s.db"
    other = sqlite3.connect(db, isolation_level=None, check_same_thread=False)
    other.execute("CREATE TABLE t (x)")      # a new file, still in rollback-journal mode
    other.execute("BEGIN IMMEDIATE")
    release = threading.Timer(HOLD, lambda: other.execute("COMMIT"))
    release.start()
    started = time.monotonic()
    try:
        stores.file(db).close()
    finally:
        release.join()
        other.close()
    assert time.monotonic() - started >= HOLD
