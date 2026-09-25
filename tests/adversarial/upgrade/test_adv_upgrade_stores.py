"""Stores written by old releases, opened by this checkout's code.

Each store under `tests/fixtures/stores/<tag>/` was written by release `<tag>`'s own
code, one store for each schema version a release has shipped; `build_stores.py` says
how. Each is committed with `golden.json`, the dump `golden.dump` took of it before
anything else opened it. Every test unpacks a copy into its own temporary directory, so
no test can change a committed file.
"""

from __future__ import annotations

import pathlib

import pytest

from . import golden


@pytest.mark.parametrize("tag", golden.TAGS)
def test_a_committed_store_holds_what_its_golden_dump_says(
        tag: str, tmp_path: pathlib.Path) -> None:
    """A check on the fixtures themselves. A store and its golden dump are written
    together, so the unopened store must read exactly as the dump says."""
    db = golden.unpack(tag, tmp_path)
    record = golden.load(tag)
    assert record["tag"] == tag
    assert golden.schema_version(db) == record["schema_version"] == golden.RELEASES[tag]
    assert golden.compare(record["data"], golden.dump(db)) == []
