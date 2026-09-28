"""Session start records the standing block it injected, where recall reads it (#343).

Without the record, the first prompt of every session found the standing check due and the
digest different, and injected the whole block a second time. The hook conformance test
runs both hooks against a store; these check the record itself.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import recall  # noqa: E402
from lib import mark, standing  # noqa: E402

BLOCK = "Memvara — how this user wants work done\n- tabs for indentation"


@pytest.fixture
def seen(monkeypatch, tmp_path) -> pathlib.Path:
    directory = tmp_path / "recalled"
    monkeypatch.setattr(recall, "SEEN_DIR", str(directory))
    return directory


def test_recall_reads_the_block_session_start_recorded_as_unchanged(seen) -> None:
    standing.record_injected("s1", BLOCK, 1000.0, directory=str(seen))
    assert recall._read_standing("s1") == (standing.digest(BLOCK), 1000.0)
    # Within the refresh interval, recall does not even look at the store.
    assert recall._standing_refresh("s1", 1000.0 + recall.STANDING_REFRESH_SECONDS - 1) == (
        "", None)


def test_the_record_keeps_what_recall_already_wrote_for_the_session(seen) -> None:
    recall._write_state("s1", ["h1", "h2"], "billing database")
    standing.record_injected("s1", BLOCK, 1000.0, directory=str(seen))
    data = json.loads((seen / "s1.json").read_text())
    assert data == {"seen": ["h1", "h2"], "query": "billing database",
                    "standing": standing.digest(BLOCK), "standing_at": 1000.0}


def test_the_record_reads_the_older_list_format(seen) -> None:
    seen.mkdir()
    (seen / "s1.json").write_text(json.dumps(["h1"]))
    standing.record_injected("s1", BLOCK, 1000.0, directory=str(seen))
    assert recall._read_state("s1") == (["h1"], "")
    assert recall._read_standing("s1") == (standing.digest(BLOCK), 1000.0)


@pytest.mark.parametrize("session", ["", "a/b", "a\0b", ".", ".."])
def test_a_session_id_that_cannot_name_a_file_records_nothing(seen, session) -> None:
    standing.record_injected(session, BLOCK, 1000.0, directory=str(seen))
    assert not seen.exists()


def test_the_digest_ignores_the_recall_mark_and_the_wrapping() -> None:
    marked = "\n".join(mark.marked(line) for line in BLOCK.split("\n"))
    assert marked != BLOCK
    assert standing.digest(marked) == standing.digest(BLOCK)
    assert standing.digest(BLOCK.replace("\n", "\n\n  ")) == standing.digest(BLOCK)
