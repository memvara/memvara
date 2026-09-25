"""The fake agent CLIs are the ones a child process finds on PATH, record how they were
started, and print their scripted reply in the format the capture hook reads."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from harness.env import REPO, child_env
from harness.fakes.cli import EXHAUSTED, USAGE, CliCall, CliReply, FakeClis
from harness.hooks import host_record

pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="the fake agent CLIs are POSIX shell scripts")

HOOKS = REPO / "plugin" / "hooks"

#: A reply holding what a careless envelope would mangle: a newline, quotes, a backslash,
#: characters outside ASCII, and JSON inside the text.
REPLY = 'Line one\n"quoted" \\ back — ünïcode ✓\n{"facts": []}'

#: Runs in a child process. It binds a host the way `plugin/hooks/run.py` does, then asks
#: the capture hook's own extraction code to start that host's CLI and read its reply.
EXTRACT = r"""
import json, shutil, sys
sys.path.insert(0, sys.argv[1])
from core import host
host.use(__import__("hosts." + sys.argv[2], fromlist=["HOST"]).HOST)
from lib import extract
reply, usage, model = extract._payload("TURN", "PROMPT ")
print(json.dumps({"found": shutil.which(sys.argv[2]), "reply": reply, "usage": usage}))
"""

#: Runs in a child process: the agentic capture run, which reads `claude`'s stream-json.
AGENTIC = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
from core import host
from hosts import claude
host.use(claude.HOST)
from lib import agentic
command = agentic.argv("RULES", "/no/such/config.json", "DATA")
run = agentic._run(command, dict(os.environ))
print(json.dumps({"failure": run.failure, "argv": command[1:],
                  "result": (run.watch.result or {}).get("result")}))
"""


@pytest.fixture
def fakes(tmp_path: pathlib.Path) -> FakeClis:
    return FakeClis(tmp_path / "bin")


@pytest.fixture
def home(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return tmp_path_factory.mktemp("cli-home")


def _child(script: str, *args: str, fakes: FakeClis, home: pathlib.Path,
           cwd: pathlib.Path) -> Any:
    done = subprocess.run([sys.executable, "-c", script, str(HOOKS), *args],
                          capture_output=True, text=True, encoding="utf-8", timeout=60,
                          env=child_env(home, {"PATH": fakes.path()}), cwd=str(cwd),
                          stdin=subprocess.DEVNULL)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def _run(fakes: FakeClis, name: str, home: pathlib.Path, *argv: str,
         stdin: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(fakes.bin / name), *argv], input=stdin, capture_output=True,
                          text=True, encoding="utf-8", timeout=30, env=child_env(home))


def test_the_claude_capture_starts_is_the_fake_and_its_reply_arrives_exactly(
        fakes: FakeClis, home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    fakes.script("claude", REPLY)
    got = _child(EXTRACT, "claude", fakes=fakes, home=home, cwd=tmp_path)
    assert got["found"] == str(fakes.bin / "claude")
    assert (got["reply"], got["usage"]) == (REPLY, dict(USAGE))
    assert fakes.calls("claude") == [
        CliCall(argv=[*host_record("claude").extractor.argv[1:], "PROMPT TURN"], stdin="")]
    assert fakes.calls("codex") == []


def test_the_codex_capture_starts_is_the_fake_and_its_events_are_read(
        fakes: FakeClis, home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    fakes.script("codex", CliReply(text=REPLY, usage={"input_tokens": 3, "output_tokens": 4}))
    got = _child(EXTRACT, "codex", fakes=fakes, home=home, cwd=tmp_path)
    assert got["found"] == str(fakes.bin / "codex")
    assert (got["reply"], got["usage"]) == (REPLY, {"input_tokens": 3, "output_tokens": 4})
    assert fakes.calls("codex") == [
        CliCall(argv=[*host_record("codex").extractor.argv[1:], "PROMPT TURN"], stdin="")]
    assert fakes.calls("claude") == []


def test_the_agentic_run_reads_the_fake_s_stream_of_events(
        fakes: FakeClis, home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    fakes.script("claude", '{"proposals": []}')
    got = _child(AGENTIC, fakes=fakes, home=home, cwd=tmp_path)
    assert (got["failure"], got["result"]) == ("", '{"proposals": []}')
    (call,) = fakes.calls("claude")
    assert call == CliCall(argv=got["argv"], stdin="")
    # The flags are written out here rather than read from `agentic.argv`, so the command
    # the fake saw is compared with what the run has to ask for, not with itself. The
    # fake prints stream-json, which the run can read only when it asks for it, and the
    # other flags keep the run away from the user's own settings, tools and servers.
    argv = call.argv
    assert argv[0] == "-p"
    assert {("--output-format", "stream-json"), ("--mcp-config", "/no/such/config.json"),
            ("--setting-sources", ""), ("--tools", ""), ("--permission-mode", "dontAsk"),
            ("--system-prompt", "RULES")} <= set(zip(argv, argv[1:]))
    assert {"--verbose", "--strict-mcp-config", "--no-session-persistence"} <= set(argv)
    # The data comes last, straight after the one value `--system-prompt` takes.
    assert argv[-2:] == ["RULES", "DATA"]


def test_a_raw_reply_is_printed_byte_for_byte_with_its_stderr_and_status(
        fakes: FakeClis, home: pathlib.Path) -> None:
    fakes.script("claude", CliReply(stdout="not json {\n", stderr="login expired\n",
                                    exit_code=1))
    done = _run(fakes, "claude", home, "-p", "hello", stdin="piped text")
    assert (done.stdout, done.stderr, done.returncode) == ("not json {\n",
                                                           "login expired\n", 1)
    assert fakes.calls("claude") == [CliCall(argv=["-p", "hello"], stdin="piped text")]


def test_a_run_with_no_reply_left_fails_loudly(fakes: FakeClis, home: pathlib.Path) -> None:
    fakes.script("codex", "only one")
    assert _run(fakes, "codex", home, "exec").returncode == 0
    second = _run(fakes, "codex", home, "exec")
    assert second.returncode == EXHAUSTED
    assert "no scripted reply left for run 2" in second.stderr
    assert len(fakes.calls("codex")) == 2


def test_scripting_again_starts_from_the_first_new_reply(
        fakes: FakeClis, home: pathlib.Path) -> None:
    """The call log keeps every run, but which reply comes next starts over with each
    script, so a test can script, run, and script again."""
    fakes.script("claude", "old reply")
    assert json.loads(_run(fakes, "claude", home, "-p", "one").stdout)["result"] \
        == "old reply"
    fakes.script("claude", "new reply")
    second = _run(fakes, "claude", home, "-p", "two")
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["result"] == "new reply"
    assert [call.argv for call in fakes.calls("claude")] == [["-p", "one"], ["-p", "two"]]


def test_a_name_that_is_not_one_of_the_fakes_is_refused(fakes: FakeClis) -> None:
    with pytest.raises(ValueError, match="no fake 'cursor-agent'"):
        fakes.script("cursor-agent", "a reply")


def test_windows_is_refused_with_the_reason(monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: pathlib.Path) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(NotImplementedError, match="only as an .exe"):
        FakeClis(tmp_path / "bin")
