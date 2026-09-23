"""The per-session activity counts the status line reads, and the hooks that keep them.

`~/.memvara/.hooks/counts/<session>.json` holds how many memory lines were recalled into a
session, how many read tools the model called, and how many facts capture stored. The
status-line script in the plugin repository vendors `lib/counts.py` and prints these, so a
count that drifts from what the hooks did is a status line that lies about memory.
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

import approve  # noqa: E402
import capture  # noqa: E402
import recall  # noqa: E402
from lib import counts, project, settings  # noqa: E402

SESSION = "0a1b2c3d-0000-4000-8000-000000000001"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Every file a hook writes goes under `tmp_path`, and every switch starts at its default."""
    monkeypatch.setattr(counts, "COUNTS_DIR", str(tmp_path / "counts"))
    monkeypatch.setattr(settings, "SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(project, "CACHE_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(settings, "_LOADED", None)
    for name in list(os.environ):
        if name.startswith("MEMVARA_FEATURE_"):
            monkeypatch.delenv(name)
    # No git lookups in these tests; the project channel has its own file.
    monkeypatch.setenv("MEMVARA_FEATURE_PROJECT_SCOPE", "0")
    monkeypatch.delenv(project.ENV, raising=False)


# -- the file ---------------------------------------------------------------------------


def test_a_session_with_no_file_reads_as_zeros():
    assert counts.read(SESSION) == {"recalled": 0, "searched": 0, "captured": 0,
                                    "updated_at": None}


def test_bump_adds_and_stamps_the_time():
    counts.bump(SESSION, "recalled", 3, now=0.0)
    counts.bump(SESSION, "recalled", 2, now=0.0)
    counts.bump(SESSION, "searched", now=0.0)
    assert counts.read(SESSION) == {"recalled": 5, "searched": 1, "captured": 0,
                                    "updated_at": "1970-01-01T00:00:00Z"}


def test_bump_leaves_no_temporary_file_behind():
    counts.bump(SESSION, "captured", 4)
    names = sorted(os.listdir(counts.COUNTS_DIR))
    assert names == [".lock", f"{SESSION}.json"]


@pytest.mark.parametrize("session", ["", ".", "..", "../escape", "a/b", "a\\b", "a\0b"])
def test_a_session_id_that_could_name_another_file_is_ignored(session):
    counts.bump(session, "recalled", 1)
    assert counts.read(session)["recalled"] == 0
    assert not os.path.exists(counts.COUNTS_DIR) or os.listdir(counts.COUNTS_DIR) == [], (
        "nothing may be written for an id like that")


def test_an_unknown_field_or_a_non_positive_count_changes_nothing():
    counts.bump(SESSION, "forgotten", 1)
    counts.bump(SESSION, "recalled", 0)
    counts.bump(SESSION, "recalled", -2)
    assert not os.path.exists(counts.COUNTS_DIR)


@pytest.mark.parametrize("body", [
    "not json",
    "[1, 2, 3]",
    json.dumps({"recalled": "7", "searched": True, "captured": -1, "updated_at": 5}),
])
def test_an_unreadable_file_reads_as_zeros_and_is_repaired_by_the_next_bump(body):
    os.makedirs(counts.COUNTS_DIR)
    pathlib.Path(counts.COUNTS_DIR, f"{SESSION}.json").write_text(body, encoding="utf-8")
    assert counts.read(SESSION) == {"recalled": 0, "searched": 0, "captured": 0,
                                    "updated_at": None}
    counts.bump(SESSION, "recalled", 1)
    assert counts.read(SESSION)["recalled"] == 1


def test_files_untouched_for_fourteen_days_are_pruned():
    os.makedirs(counts.COUNTS_DIR)
    old = pathlib.Path(counts.COUNTS_DIR, "old-session.json")
    old.write_text("{}", encoding="utf-8")
    recent = pathlib.Path(counts.COUNTS_DIR, "recent-session.json")
    recent.write_text("{}", encoding="utf-8")
    stale = time.time() - counts.TTL_SECONDS - 60
    os.utime(old, (stale, stale))
    counts.bump(SESSION, "recalled", 1)
    assert old.exists(), "a bump does not prune; that is once per session"
    counts.prune()
    assert not old.exists()
    assert recent.exists()
    assert pathlib.Path(counts.COUNTS_DIR, f"{SESSION}.json").exists()


def test_session_start_prunes_the_counters_and_the_project_cache(monkeypatch, tmp_path):
    import session_start

    pruned: list[str] = []
    monkeypatch.setattr(counts, "prune", lambda: pruned.append("counts"))
    monkeypatch.setattr(project, "prune", lambda: pruned.append("projects"))
    monkeypatch.setattr(session_start, "payload", lambda: {"session_id": SESSION, "cwd": ""})
    monkeypatch.setattr(session_start, "write", lambda host, reply: None)
    monkeypatch.setattr(session_start, "due_capture_alert", lambda: "")
    monkeypatch.setattr(session_start, "open_writer", lambda: (None, None))
    assert session_start.main() == 0
    assert pruned == ["counts", "projects"]


def test_a_directory_that_cannot_be_created_fails_silently(monkeypatch, tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setattr(counts, "COUNTS_DIR", str(blocker / "counts"))
    counts.bump(SESSION, "recalled", 1)
    assert counts.read(SESSION)["recalled"] == 0


def test_the_reader_imports_nothing_from_the_rest_of_the_hooks(tmp_path):
    """The status-line script vendors this one file and calls `read`; it must stand alone.

    Proven by loading the file on its own, outside its package, and reading a real file.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("counts_alone", HOOKS / "lib" / "counts.py")
    alone = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alone)
    alone.COUNTS_DIR = str(tmp_path)
    (tmp_path / "s.json").write_text(json.dumps({"recalled": 4}), encoding="utf-8")
    assert alone.read("s")["recalled"] == 4
    top_level = [line for line in (HOOKS / "lib" / "counts.py").read_text().splitlines()
                 if line.startswith(("from ", "import "))]
    assert all(".settings" not in l and ".state_file" not in l for l in top_level)


def test_a_session_id_with_a_nul_byte_never_raises():
    """`os` calls raise `ValueError` on a NUL, which the old guard did not catch."""
    counts.bump("a\0b", "recalled", 1)
    assert counts.read("a\0b")["recalled"] == 0


def test_parallel_bumps_lose_no_count(tmp_path):
    """Two tool calls approved at once must both be counted."""
    import subprocess

    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from lib import counts\n"
        "counts.COUNTS_DIR = %r\n"
        "for _ in range(50): counts.bump(%r, 'searched')\n"
    ) % (str(HOOKS), counts.COUNTS_DIR, SESSION)
    procs = [subprocess.Popen([sys.executable, "-c", script]) for _ in range(4)]
    assert all(p.wait(timeout=60) == 0 for p in procs)
    assert counts.read(SESSION)["searched"] == 200


# -- the hooks that keep them -----------------------------------------------------------


class _Replies(list):
    def __call__(self, host, reply) -> None:
        self.append(reply)


def _approve(monkeypatch, tool: str) -> _Replies:
    replies = _Replies()
    monkeypatch.setattr(approve, "payload", lambda: {"session_id": SESSION, "tool_name": tool})
    monkeypatch.setattr(approve, "write", replies)
    assert approve.main() == 0
    return replies


@pytest.mark.parametrize("leaf", ["memory_standing", "memory_ask", "memory_profile"])
def test_the_read_tools_the_research_subagent_calls_are_approved(monkeypatch, leaf):
    replies = _approve(monkeypatch, f"mcp__plugin_memvara_memvara__{leaf}")
    assert [r.decision for r in replies] == ["allow"]


def test_every_approved_read_counts_as_one_search(monkeypatch):
    _approve(monkeypatch, "mcp__memvara__memory_search")
    _approve(monkeypatch, "mcp__memvara__memory_standing")
    assert counts.read(SESSION)["searched"] == 2


@pytest.mark.parametrize("leaf", ["memory_remember", "memory_forget", "memory_end",
                                  "memory_add"])
def test_a_write_tool_is_neither_approved_nor_counted(monkeypatch, leaf):
    replies = _approve(monkeypatch, f"mcp__memvara__{leaf}")
    assert replies == []
    assert counts.read(SESSION)["searched"] == 0


def test_no_count_is_kept_when_the_status_line_is_switched_off(monkeypatch):
    monkeypatch.setenv("MEMVARA_FEATURE_STATUS_LINE", "0")
    replies = _approve(monkeypatch, "mcp__memvara__memory_search")
    assert [r.decision for r in replies] == ["allow"], "approval does not depend on the switch"
    assert not os.path.exists(counts.COUNTS_DIR)


def _recall(monkeypatch, tmp_path, block: str, standing: str = "") -> _Replies:
    replies = _Replies()
    monkeypatch.setattr(recall, "SEEN_DIR", str(tmp_path / "recalled"))
    monkeypatch.setattr(recall, "payload", lambda: {
        "session_id": SESSION, "prompt": "which database does billing use",
        "cwd": str(tmp_path)})
    monkeypatch.setattr(recall, "write", replies)
    monkeypatch.setattr(recall, "log_line", lambda *a, **k: None)
    monkeypatch.setattr(recall, "due_capture_alert", lambda: "")
    monkeypatch.setattr(recall, "due_alert_for_model", lambda: "")
    monkeypatch.setattr(recall, "fast_recall", lambda *a, **k: (block, True, ""))
    monkeypatch.setattr(recall, "_standing_refresh",
                        lambda *a, **k: (standing, ("d", 1.0) if standing else None))
    assert recall.main() == 0
    return replies


def test_recall_counts_every_memory_line_it_injects(monkeypatch, tmp_path):
    block = f"{recall.HEADER}\n- billing uses postgres\n- deploys go to fly.io"
    standing = "Standing:\n⋈ - always open a PR"
    _recall(monkeypatch, tmp_path, block, standing)
    assert counts.read(SESSION)["recalled"] == 3, "two recalled memories and one standing rule"


def test_recall_does_not_count_a_memory_already_in_context(monkeypatch, tmp_path):
    block = f"{recall.HEADER}\n- billing uses postgres"
    _recall(monkeypatch, tmp_path, block)
    _recall(monkeypatch, tmp_path, block)
    assert counts.read(SESSION)["recalled"] == 1


def test_recall_counts_a_standing_update_on_a_turn_with_nothing_new(monkeypatch, tmp_path):
    _recall(monkeypatch, tmp_path, "", "Standing:\n⋈ - rule one\n⋈ - rule two")
    assert counts.read(SESSION)["recalled"] == 2


def _capture(monkeypatch, tmp_path, stored: int, failed: "list[str]") -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {
        "role": "user", "content": "remember that billing uses postgres"}}) + "\n",
        encoding="utf-8")
    monkeypatch.setattr(capture, "STATE", tmp_path / "capture-state.json")
    monkeypatch.setattr(capture, "payload", lambda: {
        "session_id": SESSION, "transcript_path": str(transcript), "cwd": str(tmp_path)})
    monkeypatch.setattr(capture, "log", lambda line: None)
    monkeypatch.setattr(capture, "open_writer", lambda: (object(), None))
    monkeypatch.setattr(capture, "_keep_turn", lambda *a: (False, []))
    monkeypatch.setattr(capture, "triples", lambda *a, **k: ["a fact"] * max(stored, 1))
    monkeypatch.setattr(capture, "store_facts", lambda *a, **k: (stored, failed))
    assert capture.main() == 0


def test_capture_counts_the_facts_a_turn_stored(monkeypatch, tmp_path):
    _capture(monkeypatch, tmp_path, 2, [])
    assert counts.read(SESSION)["captured"] == 2


def test_capture_counts_nothing_when_every_write_failed(monkeypatch, tmp_path):
    _capture(monkeypatch, tmp_path, 0, ["quota"])
    assert counts.read(SESSION)["captured"] == 0
