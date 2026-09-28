"""Capture says what it decided.

Capture runs in the background on every host, so nothing it prints reaches anyone, and
its log is its whole account: plugin/hooks/capture.py says that "every path that reaches
a decision writes a line, including the ones that decide to do nothing". Each case here
runs capture once on Claude Code, against the fake agent CLIs, and looks for its line in
`capture.log`.

The last test covers the four paths on which capture decides to do nothing before it
reads a turn. Each used to return without a line (#342, B60).
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any, Callable

import pytest

from harness.fakes.cli import FakeClis
from harness.hooks import HookRunner

from . import support

Make = Callable[..., HookRunner]

_FACT = [(support.USER_TURN, support.ASSISTANT_TURN)]

#: Each case: the transcript's (user, assistant) turns, or None for a transcript that holds
#: only the assistant's reply; whether a store is configured; and the line capture must
#: log, as a regular expression.
CASES = {
    "a turn that only says to carry on": ([("ok", "Done.")], True,
                                          r"turn=\d+c skipped=continuation$"),
    "a transcript with no typed prompt": (None, True, r"no turn to mine$"),
    "no store and no login": (_FACT, False, r"turn=\d+c stored=0 failed=no store or login$"),
    "a turn that states a fact": (_FACT, True,
                                  r"turn=\d+c agentic searches=0 proposals=1 refused=0 "
                                  r"stored=1 "),
}


@pytest.mark.parametrize("case", CASES)
def test_capture_logs_what_it_decided(hooks: Make, clis: FakeClis, tmp_path: pathlib.Path,
                                      case: str) -> None:
    turns, configured, line = CASES[case]
    transcript = tmp_path / "t.jsonl"
    if turns is None:
        transcript.write_text(json.dumps({"type": "assistant", "message": {
            "content": [{"type": "text", "text": "Done."}]}}) + "\n")
    else:
        support.write_transcript("claude", transcript, turns)
    env = (support.store_env(support.make_store(tmp_path / "capture.db", memory=False))
           if configured else {})
    clis.script("claude", support.PROPOSALS_REPLY)
    result = hooks("claude", env=env, stubs=clis).run("capture", session="s",
                                                      transcript_path=str(transcript))
    assert result.exit_code == 0
    assert any(re.match(line, logged) for logged in result.log("capture")), result.logs


#: The Stops on which capture decides to do nothing before it reads a turn, each with the
#: line it must log.
NOTHING_TO_DO = {
    "a Stop that a hook itself triggered": r"skipped=stop triggered by a hook$",
    "no transcript path": r"skipped=no transcript path$",
    "a transcript path that is not a file": r"skipped=transcript is not a file$",
    "a transcript that has not grown since the last Stop":
        r"skipped=transcript unchanged since the last capture$",
}


@pytest.mark.parametrize("case", NOTHING_TO_DO)
def test_capture_says_so_when_it_decides_to_do_nothing(
        hooks: Make, clis: FakeClis, tmp_path: pathlib.Path, case: str) -> None:
    """capture.py, `main`, logs a line when the Stop was triggered by a hook
    (`stop_hook_active`), when the payload names no transcript, when the path names no
    file, and when the transcript is the size it was at the last capture. For the last,
    the first Stop mines the turn and the second is the one checked."""
    transcript = support.write_transcript("claude", tmp_path / "t.jsonl",
                                          [(support.USER_TURN, support.ASSISTANT_TURN)])
    fields: dict[str, Any] = {"transcript_path": str(transcript)}
    if case == "a Stop that a hook itself triggered":
        fields["stop_hook_active"] = True
    elif case == "no transcript path":
        fields = {}
    elif case == "a transcript path that is not a file":
        fields["transcript_path"] = str(tmp_path / "gone.jsonl")
    env = support.store_env(support.make_store(tmp_path / "capture.db", memory=False))
    clis.script("claude", support.PROPOSALS_REPLY)
    runner = hooks("claude", env=env, stubs=clis)
    stop = support.host_json("claude", "capture", session="s", cwd=tmp_path, **fields)
    if case == "a transcript that has not grown since the last Stop":
        mined = runner.run("capture", stdin=stop)
        assert any(" stored=1 " in line for line in mined.log("capture")), mined.logs
    result = runner.run("capture", stdin=stop)
    assert result.exit_code == 0
    assert support.crashes(result) == []
    line = NOTHING_TO_DO[case]
    assert any(re.search(line, logged) for logged in result.log("capture")), result.logs


def test_a_payload_that_is_not_valid_utf8_still_reaches_the_detached_capture(
        hooks: Make, clis: FakeClis, tmp_path: pathlib.Path) -> None:
    """run.py hands the payload to the capture child as the bytes it read. It used to
    decode them and encode them again, and a byte that is not UTF-8, read with
    surrogateescape as Python's UTF-8 mode and the C locale read it, made the encoding
    fail, so the child was never started and the turn was lost."""
    transcript = support.write_transcript("claude", tmp_path / "t.jsonl", [("ok", "Done.")])
    stop = json.dumps({"session_id": "s", "transcript_path": str(transcript),
                       "cwd": "@"}).encode().replace(b'"@"', b'"\xff"')
    runner = hooks("claude", env={"PYTHONIOENCODING": "utf-8:surrogateescape"}, stubs=clis)
    result = runner.run("capture", stdin=stop)
    assert result.detached_pid is not None, result.logs
    assert any(re.search(r"skipped=continuation$", line)
               for line in result.log("capture")), result.logs
