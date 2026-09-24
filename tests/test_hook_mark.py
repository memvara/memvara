"""The `⋈ ` mark on every memory line the hooks inject, and capture refusing to mine it.

Two properties matter more than the glyph. The recall hook's per-session dedup must hash the
line without the mark, or every memory a running session had already seen would be injected
again on the first prompt after the upgrade. And capture must drop marked lines, or the
store's own output is read back as conversation and stored a second time.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import capture  # noqa: E402
import recall  # noqa: E402
import session_start  # noqa: E402
from lib import counts, mark, project, settings, standing, transcript  # noqa: E402

SESSION = "0a1b2c3d-0000-4000-8000-000000000002"
MARK = "⋈ "


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(counts, "COUNTS_DIR", str(tmp_path / "counts"))
    monkeypatch.setattr(settings, "SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(project, "CACHE_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(settings, "_LOADED", None)
    for name in list(os.environ):
        if name.startswith("MEMVARA_FEATURE_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("MEMVARA_FEATURE_PROJECT_SCOPE", "0")
    monkeypatch.delenv(project.ENV, raising=False)


class _Replies(list):
    def __call__(self, host, reply) -> None:
        self.append(reply)


def _recall(monkeypatch, tmp_path, block: str) -> _Replies:
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
    monkeypatch.setattr(recall, "_standing_refresh", lambda *a, **k: ("", None))
    assert recall.main() == 0
    return replies


BLOCK = f"{recall.HEADER}\n- billing uses postgres\n- deploys go to fly.io"


# -- injection --------------------------------------------------------------------------


def test_the_mark_is_the_brand_glyph_and_a_space():
    assert mark.MARK == MARK


def test_every_recalled_memory_line_starts_with_the_mark(monkeypatch, tmp_path):
    reply = _recall(monkeypatch, tmp_path, BLOCK)[0]
    lines = reply.context.splitlines()
    assert lines[0] == recall.HEADER, "the header is not a memory and is not marked"
    assert lines[1:] == [f"{MARK}- billing uses postgres", f"{MARK}- deploys go to fly.io"]


def test_the_mark_can_be_switched_off(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMVARA_FEATURE_RECALL_MARK", "0")
    reply = _recall(monkeypatch, tmp_path, BLOCK)[0]
    assert reply.context.splitlines()[1:] == ["- billing uses postgres",
                                              "- deploys go to fly.io"]


def test_the_dedup_hash_is_taken_over_the_line_without_the_mark(monkeypatch, tmp_path):
    _recall(monkeypatch, tmp_path, BLOCK)
    state = json.loads((tmp_path / "recalled" / f"{SESSION}.json").read_text())
    assert state["seen"] == [recall._digest("- billing uses postgres"),
                             recall._digest("- deploys go to fly.io")]


def test_a_seen_set_written_before_the_mark_still_suppresses_repeats(monkeypatch, tmp_path):
    """A session running across the upgrade must not have everything injected again."""
    recalled = tmp_path / "recalled"
    recalled.mkdir()
    (recalled / f"{SESSION}.json").write_text(json.dumps({
        "seen": [recall._digest("- billing uses postgres"),
                 recall._digest("- deploys go to fly.io")],
        "query": "", "standing": "", "standing_at": 0.0}))
    reply = _recall(monkeypatch, tmp_path, BLOCK)[0]
    assert reply.context == ""
    assert "2 already in context" in reply.status


def test_a_clipped_line_is_marked_after_it_is_clipped(monkeypatch, tmp_path):
    long = "- " + "x" * (recall.MAX_INJECTED_CHARS + 50)
    reply = _recall(monkeypatch, tmp_path, f"{recall.HEADER}\n{long}")[0]
    line = reply.context.splitlines()[1]
    assert line.startswith(MARK + "- x")
    assert len(line) == len(MARK) + recall.MAX_INJECTED_CHARS + 1, "clip, then the mark"
    assert reply.context.splitlines()[-1] == recall.MORE


def test_every_standing_row_is_marked():
    notes = [standing.Note(text="always open a PR", subject="user", inferred=False,
                           confidence=1.0, recorded="2026-09-01", ident="c1"),
             standing.Note(text="prefers tabs", subject="user", inferred=True,
                           confidence=0.7, recorded="2026-09-02", ident="c2")]
    block = standing.render(notes, "Standing:", 1000)
    assert block.splitlines() == ["Standing:", f"{MARK}- always open a PR",
                                  f"{MARK}- prefers tabs{standing.MARKER}"]


def test_mark_block_marks_only_bullets_and_is_safe_to_apply_twice():
    block = "Header:\n- one\n(1 further note did not fit)\n- two"
    once = mark.mark_block(block)
    assert once == f"Header:\n{MARK}- one\n(1 further note did not fit)\n{MARK}- two"
    assert mark.mark_block(once) == once
    assert mark.mark_block(block, mark=False) == block
    assert mark.count(once) == mark.count(block) == 2


class _Hosted:
    def stats(self) -> str:
        return "scope: prj_x/*/*/*/*  (tenant/user/project/agent/session)\n" \
               "visible at this scope: 3 claim(s)"

    def recall(self, query, **kwargs) -> str:
        return f"{session_start.HEADER}\n- works on memvara\n- lives in Delhi"


def test_session_start_marks_every_section_and_counts_marked_lines(monkeypatch):
    replies = _Replies()
    monkeypatch.setattr(session_start, "payload", lambda: {"session_id": SESSION, "cwd": ""})
    monkeypatch.setattr(session_start, "write", replies)
    monkeypatch.setattr(session_start, "due_capture_alert", lambda: "")
    monkeypatch.setattr(session_start, "open_writer", lambda: (_Hosted(), lambda: None))
    # The legacy fallback returns the server's own block, which carries no mark.
    monkeypatch.setattr(session_start, "standing_block",
                        lambda *a, **k: f"{session_start.STANDING_HEADER}\n- always open a PR")
    assert session_start.main() == 0
    reply = replies[0]
    memory = [line for line in reply.context.splitlines() if mark.is_memory(line)]
    assert memory == [f"{MARK}- always open a PR", f"{MARK}- works on memvara",
                      f"{MARK}- lives in Delhi"]
    assert "session opened with 3 memories" in reply.status


def test_the_binding_line_names_all_five_parts_of_the_scope():
    line = session_start._binding_line("prj_x/*/*/*/*", "3 claim(s)")
    assert "(tenant/user/project/agent/session; '*' means unbound)" in line


# -- capture ---------------------------------------------------------------------------


def _jsonl(*entries: dict) -> bytes:
    return "\n".join(json.dumps(e) for e in entries).encode("utf-8")


def _user(text) -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}}


def _assistant(text: str) -> dict:
    return {"type": "assistant", "message": {"role": "assistant",
                                             "content": [{"type": "text", "text": text}]}}


def test_a_transcript_with_an_injected_block_mines_nothing_from_that_block():
    """The acceptance test in the spec: recalled memory is never extracted again.

    The block arrives three ways here: whole with its header, as marked lines with the
    header cut off, and quoted back by the assistant. None of its text may reach the turn
    that is mined, and all of it must reach the echo filter.
    """
    raw = _jsonl(
        _user(f"{recall.HEADER}\n{MARK}- billing uses postgres"),
        _user("which database should the new service use?"),
        _user([{"type": "text", "text": f"{MARK}- deploys go to fly.io\n{MARK}- lives in Delhi"}]),
        _assistant(f"From memory:\n{MARK}- billing uses postgres\nI will check the config."),
    )
    turn, injected = transcript.last_turn_with_injections(raw)
    assert "User: which database should the new service use?" in turn
    assert "Claude: From memory:\nI will check the config." in turn
    for text in ("billing uses postgres", "deploys go to fly.io", "lives in Delhi"):
        assert text not in turn, f"injected memory was mined: {text}"
        assert text in injected, f"the echo filter was not told about: {text}"


def test_a_turn_made_only_of_an_injected_block_has_nothing_to_mine():
    raw = _jsonl(_user(f"{MARK}- billing uses postgres\n{MARK}- deploys go to fly.io"))
    assert transcript.last_turn_with_injections(raw) == ("", [])


def test_the_glyph_inside_a_line_is_not_a_mark():
    raw = _jsonl(_user("rename the ⋈ icon in the status bar"))
    assert transcript.last_turn(raw) == "User: rename the ⋈ icon in the status bar"


def test_an_unmarked_list_outside_a_block_is_not_treated_as_injected():
    raw = _jsonl(_user("todo:\n- write the tests\n- ship it"))
    turn, injected = transcript.last_turn_with_injections(raw)
    assert "- write the tests" in turn
    assert injected == []


def test_capture_hands_the_extractor_a_turn_without_the_injected_block(monkeypatch, tmp_path):
    """End to end through `capture.main`: the extractor never sees recalled memory."""
    session = tmp_path / "session.jsonl"
    session.write_bytes(_jsonl(
        _user("which database should the new service use?"),
        _assistant(f"{MARK}- billing uses postgres\nThen postgres it is."),
    ) + b"\n")
    handed: list[str] = []
    monkeypatch.setattr(capture, "STATE", tmp_path / "capture-state.json")
    monkeypatch.setattr(capture, "payload", lambda: {
        "session_id": SESSION, "transcript_path": str(session), "cwd": str(tmp_path)})
    monkeypatch.setattr(capture, "log", lambda line: None)
    monkeypatch.setattr(capture, "open_writer", lambda: (object(), None))
    monkeypatch.setattr(capture, "_keep_turn", lambda *a: (False, []))
    # Agentic capture is on by default, so it is handed the turn first. It records what
    # it was given and reports that it could not run, which sends the turn on to the
    # single-call extractor as well: both must get it without the injected block.
    monkeypatch.setattr(capture.agentic, "capture",
                        lambda store, turn, context, *a, **k:
                        handed.extend([turn, context]) and None)
    monkeypatch.setattr(capture, "triples",
                        lambda turn, cwd, injected=(): handed.append(turn) or [])
    assert capture.main() == 0
    assert len(handed) == 3
    assert not any("billing uses postgres" in text for text in handed)
    assert "Then postgres it is." in handed[0] and "Then postgres it is." in handed[2]


# -- the standing digest ---------------------------------------------------------------


def test_switching_the_mark_does_not_make_the_standing_set_look_changed(monkeypatch, tmp_path):
    """The digest must be taken over what the block says, not how it is dressed.

    Hashing the marked block made every running session report "standing preferences
    updated" once after the upgrade, and again each time `recall_mark` was switched.
    """
    from lib import standing as standing_mod
    from lib import write as write_mod

    notes = [standing_mod.Note(text="always open a PR", subject="user", inferred=False,
                               confidence=1.0, recorded="2026-09-01", ident="c1")]
    monkeypatch.setattr(standing_mod, "standing_block",
                        lambda *a, **k: standing_mod.render(notes, "Standing:", 1000))
    monkeypatch.setattr(write_mod, "open_writer", lambda: (object(), None))
    monkeypatch.setattr(recall, "SEEN_DIR", str(tmp_path / "recalled"))

    # Before the upgrade: an unmarked block, digested as it was then.
    old_digest = recall._digest("Standing:\n- always open a PR")
    later = recall.STANDING_REFRESH_SECONDS + 10.0

    monkeypatch.setenv("MEMVARA_FEATURE_RECALL_MARK", "1")
    recall._write_state(SESSION, [], "", (old_digest, 0.0))
    block, state = recall._standing_refresh(SESSION, later)
    assert block == "", "the mark alone must not count as a change"
    assert state == (old_digest, later)

    monkeypatch.setenv("MEMVARA_FEATURE_RECALL_MARK", "0")
    block, _ = recall._standing_refresh(SESSION, later * 2)
    assert block == "", "switching the mark off must not count as a change either"


# -- the two ways capture recognises injected memory -----------------------------------


@pytest.mark.parametrize("name, text", [
    ("header and marks", f"{recall.HEADER}\n{MARK}- billing uses postgres\n"
                         f"{MARK}- deploys go to fly.io"),
    ("header without marks", f"{recall.HEADER}\n- billing uses postgres\n"
                             "- deploys go to fly.io"),
    ("marks without header", f"{MARK}- billing uses postgres\n{MARK}- deploys go to fly.io"),
    ("truncated header with marks", f"Recalled from Mem\n{MARK}- billing uses postgres\n"
                                    f"{MARK}- deploys go to fly.io"),
    ("standing header and marks", f"{session_start.STANDING_HEADER}\n"
                                  f"{MARK}- billing uses postgres\n"
                                  f"{MARK}- deploys go to fly.io"),
])
def test_header_and_mark_detection_agree_on_every_shape_of_block(name, text):
    """Whichever rule recognises a block, the echo filter and the mining filter must agree.

    `_injected_lines` tells the extractor what it was shown; `_clean` decides what is mined.
    A memory reported as injected but still mined, or mined without being reported, is the
    gap these two rules must not leave between them.
    """
    memories = ["billing uses postgres", "deploys go to fly.io"]
    assert transcript._injected_lines(text) == memories, name
    kept = transcript._clean(text)
    for memory in memories:
        assert memory not in kept, f"{name}: mined {memory!r}"
