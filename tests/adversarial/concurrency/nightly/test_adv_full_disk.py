"""A disk that fills up: the child lowers the largest file it may write and writes until
a write fails. The failure must be loud, and the store must recover when it is opened
again without the limit. POSIX only; the nightly tier runs on the maintainer's Mac."""

from __future__ import annotations

import pathlib
import sys

import pytest

from harness import stores
from harness.crash import Child, CrashHarnessError, after_crash

USER = "u1"

pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="RLIMIT_FSIZE exists only on POSIX")


def test_a_full_disk_fails_loudly_and_the_store_recovers(tmp_path: pathlib.Path) -> None:
    db = tmp_path / "s.db"
    home = tmp_path / "home"
    home.mkdir()
    with stores.file(db) as mem:
        mem.remember("user", "likes", "green tea", user=USER)
    limit = max(p.stat().st_size for p in tmp_path.iterdir() if p.is_file()) + 256 * 1024
    spec = {"db": str(db), "user": USER, "setup": [], "point": None, "hold": False,
            "action": ["fill", {"limit": limit}]}
    with Child(spec, home=home, timeout=120) as child:
        with pytest.raises(CrashHarnessError) as failed:
            child.wait_for("DONE")
        code = child.proc.wait(timeout=30)
    assert code == 3, f"the child exited with {code}, not with a reported failure"
    message = str(failed.value)
    assert any(word in message for word in ("full", "too large", "File too large",
                                            "disk")), message
    live = {a["ids"][0]: f"q{a['index']:06d}x" for a in child.acked if a["ids"]}
    assert live, "no write succeeded before the disk filled"
    after_crash(db, USER, live).close()
