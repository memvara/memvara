"""The one helper every hook state file goes through: atomic writes, a lock, pruning.

A hook must never fail a turn over a state file, so every test here also checks that a
failure came back as a return value rather than an exception.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import recall  # noqa: E402
from lib import state_file  # noqa: E402


def test_a_write_lands_whole_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "deep" / "state.json"
    assert state_file.write_json(str(path), {"a": 1}, ".x-") is True
    assert json.loads(path.read_text()) == {"a": 1}
    assert sorted(os.listdir(path.parent)) == ["state.json"]


def test_a_rename_that_fails_removes_its_temporary_file(monkeypatch, tmp_path):
    path = tmp_path / "state.json"

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    assert state_file.write_json(str(path), {"a": 1}, ".x-") is False
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("path", ["/tmp/a\0b/state.json", ""])
def test_a_path_the_os_refuses_is_a_failed_write_not_an_exception(path):
    assert state_file.write_json(path, {"a": 1}) is False
    assert state_file.read_json(path) == {}
    lock = path + ".lock" if path else "/tmp/a\0b/.lock"
    assert state_file.update_json(path, lambda d: d, lock_path=lock) is False
    state_file.prune(path, 1, time.time())


def test_data_that_is_not_json_is_a_failed_write(tmp_path):
    path = tmp_path / "state.json"
    assert state_file.write_json(str(path), {"a": object()}) is False
    assert os.listdir(tmp_path) == []


def test_an_update_whose_change_fails_leaves_the_old_file(tmp_path):
    path = tmp_path / "state.json"
    state_file.write_json(str(path), {"a": 1})

    def change(data):
        raise KeyError("missing")

    assert state_file.update_json(str(path), change,
                                  lock_path=str(tmp_path / ".lock")) is False
    assert json.loads(path.read_text()) == {"a": 1}


def test_an_update_creates_the_directory_it_needs(tmp_path):
    path = tmp_path / "new" / "state.json"
    assert state_file.update_json(str(path), lambda d: {"n": 1, "was": d},
                                  lock_path=str(tmp_path / "new" / ".lock")) is True
    assert json.loads(path.read_text()) == {"n": 1, "was": None}


def test_prune_removes_only_old_files_with_the_suffix(tmp_path):
    now = time.time()
    for name, age in (("old.json", 100), ("new.json", 1), ("old.lock", 100)):
        path = tmp_path / name
        path.write_text("{}")
        os.utime(path, (now - age, now - age))
    state_file.prune(str(tmp_path), 50, now)
    assert sorted(os.listdir(tmp_path)) == ["new.json", "old.lock"]


# -- recall's per-session state goes through it ----------------------------------------


@pytest.fixture
def seen_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(recall, "SEEN_DIR", str(tmp_path / "recalled"))
    return tmp_path / "recalled"


def test_a_concurrent_prompt_does_not_lose_the_hashes_another_one_added(seen_dir):
    """Two prompts answered at once each read the old seen-set and wrote their own.

    The second write used to replace the first, so the first prompt's memories were
    injected again on the next turn. Hashes already in the file are now kept.
    """
    recall._write_state("s", ["a"], "q")
    # Both prompts read ["a"]; the first adds "b", the second adds "c".
    recall._write_state("s", ["a", "b"], "q")
    recall._write_state("s", ["a", "c"], "q")
    seen, _ = recall._read_state("s")
    assert sorted(seen) == ["a", "b", "c"]
    assert seen[-2:] == ["a", "c"], "the latest writer's hashes are the newest"


def test_the_seen_set_is_still_bounded(seen_dir):
    recall._write_state("s", [f"h{i}" for i in range(recall.MAX_SEEN)], "q")
    recall._write_state("s", ["new"], "q")
    seen, _ = recall._read_state("s")
    assert len(seen) == recall.MAX_SEEN
    assert seen[-1] == "new"


def test_the_standing_keys_are_carried_forward(seen_dir):
    recall._write_state("s", [], "q", ("digest", 5.0))
    recall._write_state("s", ["x"], "q2")
    assert recall._read_standing("s") == ("digest", 5.0)


def test_an_old_bare_list_state_file_is_still_read_and_upgraded(seen_dir):
    seen_dir.mkdir()
    (seen_dir / "s.json").write_text(json.dumps(["old"]))
    recall._write_state("s", ["new"], "q")
    assert recall._read_state("s") == (["old", "new"], "q")


def test_a_session_id_with_a_nul_byte_gets_no_state_and_no_exception(seen_dir):
    recall._write_state("a\0b", ["x"], "q")
    assert recall._read_state("a\0b") == ([], "")
    assert not seen_dir.exists()


def test_the_write_leaves_only_the_state_file_and_the_lock(seen_dir):
    recall._write_state("s", ["x"], "q")
    assert sorted(os.listdir(seen_dir)) == [".lock", "s.json"]
