"""The builder's two ways of failing: a writer that runs too long, and a copy of the
finished store into tests/fixtures/stores/ that fails partway.

Both must end in a `BuildError` or the original error, and neither may leave the
committed files half replaced. Neither test needs a release: `git archive` is the only
step replaced, so they run in a clone with no tags.
"""

from __future__ import annotations

import pathlib
import shutil
from typing import Any

import pytest

from harness.env import child_env

from . import build_stores

#: The committed files of one store, with contents that tell old from new.
OLD = {"golden.json": b"old golden", "store.db.embedder.json": b"old record",
       "store.db.gz": b"old database", "store.db.vecs.gz": b"old vectors"}


def test_a_writer_that_runs_too_long_is_reported_as_a_build_error(
        tmp_path: pathlib.Path, home: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The writer runs as a real child process, with a time limit it cannot meet."""
    monkeypatch.setattr(build_stores, "extract",
                        lambda tag, dest, *, env: dest)
    monkeypatch.setattr(build_stores, "WRITE_SECONDS", 0.001)
    with pytest.raises(build_stores.BuildError, match="did not finish within"):
        build_stores.write_with_release("v0.16.0", tmp_path / "store.db",
                                        env=child_env(home))


def committed(out: pathlib.Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in out.iterdir()}


def test_an_install_that_fails_partway_leaves_the_committed_files_as_they_were(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "stores" / "v9.9.9"
    out.mkdir(parents=True)
    for name, data in OLD.items():
        (out / name).write_bytes(data)
    staged = tmp_path / "staged"
    staged.mkdir()
    for name in OLD:
        (staged / name).write_bytes(b"new " + name.encode())
    real = shutil.copyfile
    copied: list[Any] = []

    def third_copy_fails(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        copied.append(src)
        if len(copied) == 3:
            raise OSError("no space left on device")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(build_stores.shutil, "copyfile", third_copy_fails)
    with pytest.raises(OSError, match="no space left on device"):
        build_stores.install(staged, out)
    assert committed(out) == OLD
    assert sorted(item.name for item in out.parent.iterdir()) == ["v9.9.9"], (
        "the failed install left a directory beside the store")


def test_an_install_replaces_every_file_and_leaves_nothing_beside_the_store(
        tmp_path: pathlib.Path) -> None:
    out = tmp_path / "stores" / "v9.9.9"
    out.mkdir(parents=True)
    for name, data in OLD.items():
        (out / name).write_bytes(data)
    (out / "left over from an older layout").write_bytes(b"x")
    staged = tmp_path / "staged"
    staged.mkdir()
    new = {name: b"new " + name.encode() for name in OLD}
    for name, data in new.items():
        (staged / name).write_bytes(data)
    build_stores.install(staged, out)
    assert committed(out) == new
    assert sorted(item.name for item in out.parent.iterdir()) == ["v9.9.9"]
