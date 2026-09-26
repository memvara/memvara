"""Capture says what it decided.

Capture runs in the background on every host, so nothing it prints reaches anyone, and
its log is its whole account: plugin/hooks/capture.py says that "every path that reaches
a decision writes a line, including the ones that decide to do nothing". Each case here
runs capture once on Claude Code, against the fake agent CLIs, and looks for its line in
`capture.log`.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Callable

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
    env = ({"MEMVARA_DB": str(support.make_store(tmp_path / "capture.db", memory=False)),
            "MEMVARA_USER": support.USER} if configured else {})
    clis.script("claude", support.PROPOSALS_REPLY)
    result = hooks("claude", env=env, stubs=clis).run("capture", session="s",
                                                      transcript_path=str(transcript))
    assert result.exit_code == 0
    assert any(re.match(line, logged) for logged in result.log("capture")), result.logs
