"""Capture says what it decided.

Capture runs in the background on every host, so nothing it prints reaches anyone, and
its log is its whole account: plugin/hooks/capture.py says that "every path that reaches
a decision writes a line, including the ones that decide to do nothing". Each case here
runs capture once on Claude Code, against the fake agent CLIs, and looks for its line in
`capture.log`.

On four paths capture decides to do nothing and logs nothing, which B60 (#342) pins.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any, Callable

import pytest

from harness import known_bugs
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


# -- known bugs --------------------------------------------------------------------------

#: The Stops on which capture decides to do nothing and logs nothing (B60).
SILENT = ("a Stop that a hook itself triggered", "no transcript path",
          "a transcript path that is not a file",
          "a transcript that has not grown since the last Stop")


@pytest.mark.parametrize("case", SILENT)
@known_bugs.xfail("B60")
def test_capture_says_so_when_it_decides_to_do_nothing(
        hooks: Make, clis: FakeClis, tmp_path: pathlib.Path, case: str) -> None:
    """capture.py, `main`, returns 0 without a line when the Stop was triggered by a hook
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
    if (result.log("capture"), result.log("hooks")) == ((), ()):
        raise known_bugs.Reproduced(f"B60: capture logged nothing for {case}")
    assert result.log("capture"), result.logs
