"""File modes on POSIX: the files that hold or serve memories are owner-only.

A store's contents, its encryption key, and the daemon socket that serves recall are all
things another account on the machine must not read. The vector sidecar, the generated key
file and the daemon socket are created owner-only rather than left to the umask; the
database and its lock file take the umask, which `SECURITY.md` documents as out of scope
(the store is a file with the filesystem's permissions and nothing more), asserted here so
a change in either direction is noticed.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import time

import pytest

from harness import stores
from harness.env import REPO, child_env
from harness.hooks import short_dir

pytestmark = pytest.mark.skipif(os.name != "posix",
                                reason="no POSIX permission bits to check")

needs_extra = pytest.mark.skipif(
    importlib.util.find_spec("sqlcipher3") is None,
    reason="needs the encrypt extra: pip install 'memvara[encrypt]'")


def _mode(path: pathlib.Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_the_vector_file_is_owner_only_and_the_database_takes_the_umask(
        tmp_path: pathlib.Path) -> None:
    """The `.vecs` sidecar is created 0600 whatever the umask, because it holds vectors
    that leak content under inversion. The database and its lock file take the umask —
    documented in SECURITY.md as out of scope — so this pins both directions under a known
    umask."""
    old = os.umask(0o022)
    try:
        db = tmp_path / "store.db"
        with stores.file(db) as mem:
            mem.remember("user", "lives_in", "Lisbon", user="alice")
        assert _mode(pathlib.Path(str(db) + ".vecs")) == 0o600
        assert _mode(db) == 0o644           # takes the umask; documented, not a defect
        assert _mode(pathlib.Path(str(db) + ".lock")) == 0o644
    finally:
        os.umask(old)


@needs_extra
def test_a_generated_key_file_is_owner_only(tmp_path: pathlib.Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """A store key generated into `~/.memvara/db.key` is written 0600, so a copy of the
    store file alone does not carry a world-readable key beside it."""
    from memvara.store.encryption import EncryptionWarning, key_file

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    old = os.umask(0o022)
    try:
        # Generating the first key warns that it must be backed up; that is expected here.
        with pytest.warns(EncryptionWarning):
            with stores.file(tmp_path / "e.db", encryption=True):
                pass
    finally:
        os.umask(old)
    assert _mode(key_file()) == 0o600


def test_the_daemon_socket_is_owner_only_inside_an_owner_only_directory(
        tmp_path: pathlib.Path) -> None:
    """The recall daemon's unix socket is a read interface to everything stored, so it is
    bound 0600 inside its 0700 runtime directory rather than left to the umask."""
    # A short HOME on purpose: the socket path is HOME/.memvara/.hooks/run/recall-*.sock,
    # and macOS refuses a unix socket path of 104 bytes or more, which pytest's own long
    # tmp_path would pass. short_dir measures the path the daemon will get. The store file
    # can live under the long tmp_path.
    home = short_dir("home")
    db = tmp_path / "store.db"
    with stores.file(db) as mem:
        mem.remember("user", "lives_in", "Lisbon", user="u")
    env = child_env(home, {"MEMVARA_DB": str(db), "MEMVARA_USER": "u"})
    run_dir = home / ".memvara" / ".hooks" / "run"
    proc = subprocess.Popen([sys.executable, str(REPO / "plugin" / "hooks" / "daemon.py")],
                            env=env)
    try:
        socket_file: "pathlib.Path | None" = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            found = list(run_dir.glob("recall-*.sock")) if run_dir.exists() else []
            if found:
                socket_file = found[0]
                break
            time.sleep(0.1)
        assert socket_file is not None, "the daemon did not create its socket"
        assert _mode(socket_file) == 0o600
        assert _mode(run_dir) == 0o700
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        shutil.rmtree(home, ignore_errors=True)
