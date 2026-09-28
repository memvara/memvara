"""A disk that really fills up: the store lives on a small RAM disk, so a write fails with
"no space left on device", as on a full laptop disk, rather than with the "file too
large" that the nightly tier's `RLIMIT_FSIZE` version produces. SQLite can treat the two
differently, so both are tested.

macOS only, because the RAM disk is made with `hdiutil` and `diskutil`; it is detached
when the test ends, whatever happens.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
from collections.abc import Iterator

import pytest

from harness import stores
from harness.crash import Child, CrashHarnessError, after_crash

USER = "u1"
#: Size of the RAM disk: 16 MB, in 512-byte sectors.
SECTORS = 16 * 1024 * 1024 // 512
#: Free space left on the disk before the child starts writing: room for the vector
#: file's first 256 rows (512 KB at this embedder's width) and some claims after them.
HEADROOM = 2 * 1024 * 1024

pytestmark = pytest.mark.skipif(sys.platform != "darwin",
                                reason="a RAM disk is made with hdiutil, which is macOS only")


@pytest.fixture()
def ramdisk() -> Iterator[pathlib.Path]:
    device = subprocess.run(["hdiutil", "attach", "-nomount", f"ram://{SECTORS}"],
                            check=True, capture_output=True, text=True).stdout.strip()
    name = f"memvara-full-{os.getpid()}"
    try:
        subprocess.run(["diskutil", "erasevolume", "HFS+", name, device], check=True,
                       capture_output=True, text=True)
        yield pathlib.Path("/Volumes") / name
    finally:
        subprocess.run(["hdiutil", "detach", device, "-force"], capture_output=True)


def test_a_real_full_disk_fails_loudly_and_the_store_recovers(
        ramdisk: pathlib.Path, home: pathlib.Path) -> None:
    db = ramdisk / "s.db"
    stores.file(db).close()
    filler = ramdisk / "filler"
    with filler.open("wb") as fh:
        fh.truncate(max(0, shutil.disk_usage(ramdisk).free - HEADROOM))
    # A limit far above the disk's size, so the disk runs out before the limit applies.
    spec = {"db": str(db), "user": USER, "setup": [], "point": None, "hold": False,
            "action": ["fill", {"limit": 1 << 40}]}
    with Child(spec, home=home, timeout=120) as child:
        with pytest.raises(CrashHarnessError) as failed:
            child.wait_for("DONE")
        code = child.proc.wait(timeout=30)
    assert code == 3, f"the child exited with {code}, not with a reported failure"
    message = str(failed.value)
    assert any(word in message for word in ("full", "space")), message
    filler.unlink()
    live = {a["ids"][0]: f"q{a['index']:06d}x" for a in child.acked if a["ids"]}
    assert live, "no write succeeded before the disk filled"
    after_crash(db, USER, live).close()
