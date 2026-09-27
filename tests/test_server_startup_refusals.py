"""The stdio server refuses a store from a newer version with one line, not a traceback
(#299). The upgrade tests check the same over a real process; these run `main()` in this
process, where coverage reaches."""

from __future__ import annotations

import io
import sqlite3

import pytest

from memvara.server import cli
from memvara.server.cli import main
from memvara.store.sqlite import SCHEMA_VERSION


def _env(path) -> dict[str, str]:
    return {"MEMVARA_DB": str(path), "MEMVARA_EMBEDDER": "hashing",
            "MEMVARA_FEATURE_ENCRYPTION": "0", "MEMVARA_FEATURE_PROJECT_SCOPE": "0"}


def test_a_store_from_a_newer_version_is_refused_in_one_line(tmp_path) -> None:
    path = tmp_path / "s.db"
    assert main([], env=_env(path), stdin=io.StringIO(""), stdout=io.StringIO()) == 0
    con = sqlite3.connect(path)
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    con.close()
    err = io.StringIO()
    assert main([], env=_env(path), stdin=io.StringIO(""), stdout=io.StringIO(),
                stderr=err) == 2
    lines = err.getvalue().splitlines()
    assert lines[0].startswith(f"memvara-mcp: {path}: schema version {SCHEMA_VERSION + 1} "
                               "was written by a newer Memvara"), lines
    assert "Traceback" not in err.getvalue()


def test_any_other_runtime_error_at_startup_still_raises(tmp_path, monkeypatch) -> None:
    """Only the store's own refusal is a configuration problem. Anything else is a bug,
    and a traceback is what a bug should produce."""
    def broken(config):
        raise RuntimeError("something else went wrong")

    monkeypatch.setattr(cli, "build_memvara", broken)
    with pytest.raises(RuntimeError, match="something else"):
        main([], env=_env(tmp_path / "s.db"), stdin=io.StringIO(""), stdout=io.StringIO(),
             stderr=io.StringIO())
