"""The feature switches the plugin's hooks read: `recall_mark`, `status_line`,
`agentic_capture` and `project_scope`. `project_scope` is read by the MCP server too, and
has a test for each side.

Each hook test runs the real hooks in their own processes with `HookRunner`, on Claude
Code, once with the switch at its default and once with `MEMVARA_FEATURE_<NAME>=0` in the
hook's environment, which plugin/hooks/lib/settings.py reads before the settings file. The
switches and what each one controls are listed in `FEATURE_DEFAULTS` in
plugin/hooks/lib/settings.py and memvara/server/config.py.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from typing import Callable

import pytest

from harness import stores
from harness.fakes.cli import NO_FAKES, FakeClis
from harness.fakes.hosted_mcp import PROJECT_HEADER, FakeHostedMcp
from harness.hooks import HookRunner
from harness.stdio import McpProcess

from ..hooks import support

Make = Callable[..., HookRunner]
Start = Callable[..., McpProcess]

#: The project a checkout whose origin is REMOTE belongs to. GitHub names are
#: case-insensitive, so the owner and repository are folded to lower case.
REMOTE = "git@github.com:Acme/App.git"
PROJECT = "github.com/acme/app"


def _git_checkout(path: pathlib.Path) -> pathlib.Path:
    """Make `path` a git repository whose `origin` is REMOTE."""
    path.mkdir(parents=True, exist_ok=True)
    for args in (["init", "-q"], ["remote", "add", "origin", REMOTE]):
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)
    return path


def _counts(home: pathlib.Path, session: str) -> dict[str, object] | None:
    """The status line counters the hooks keep for `session`, or None when they keep
    none (plugin/hooks/lib/counts.py)."""
    path = home / ".memvara" / ".hooks" / "counts" / f"{session}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@pytest.mark.covers("switch:recall_mark")
def test_recall_mark_puts_the_mark_on_every_recalled_line_only_while_switched_on(
        hook_runner: Make, tmp_path: pathlib.Path) -> None:
    """Every memory line the hooks put into a prompt starts with the mark "⋈ " while
    `recall_mark` is on, which is its default. Switched off, the same memory is recalled
    without the mark. plugin/hooks/lib/mark.py makes this promise."""
    env = support.store_env(support.make_store(tmp_path / "memory.db"))

    marked = hook_runner("claude", env=env).run("recall", session="on",
                                                 prompt=support.PROMPT)
    lines = support.context_of("claude", marked.reply).splitlines()
    assert f"⋈ - {support.MEMORY}" in lines, lines

    plain = hook_runner("claude", env={**env, "MEMVARA_FEATURE_RECALL_MARK": "0"}).run(
        "recall", session="off", prompt=support.PROMPT)
    lines = support.context_of("claude", plain.reply).splitlines()
    assert f"- {support.MEMORY}" in lines, lines
    assert not [line for line in lines if line.startswith("⋈")], lines


@pytest.mark.covers("switch:status_line")
def test_status_line_switched_off_stops_the_hooks_counting_for_the_status_line(
        hook_runner: Make, tmp_path: pathlib.Path) -> None:
    """While `status_line` is on, which is its default, the recall hook counts the
    memories it injects and the approve hook counts each read tool it approves, in a
    per-session file the status line reads. Switched off, counting stops and no file is
    written, while recall and approval work as before. plugin/hooks/lib/counts.py and
    `FEATURE_DEFAULTS` in plugin/hooks/lib/settings.py make this promise."""
    env = support.store_env(support.make_store(tmp_path / "memory.db"))
    search = support.tool_name("claude", "memory_search")

    on = hook_runner("claude", env=env)
    on.run("recall", session="on", prompt=support.PROMPT)
    on.run("approve", session="on", tool_name=search)
    counted = _counts(on.home, "on")
    assert counted is not None
    assert (counted["recalled"], counted["searched"]) == (1, 1)

    off = hook_runner("claude", env={**env, "MEMVARA_FEATURE_STATUS_LINE": "0"})
    recalled = off.run("recall", session="off", prompt=support.PROMPT)
    approved = off.run("approve", session="off", tool_name=search)
    assert support.MEMORY in support.context_of("claude", recalled.reply)
    assert support.decision_of("claude", approved.reply) == "allow"
    assert _counts(off.home, "off") is None, "nothing was counted with the switch off"


@pytest.mark.covers("switch:agentic_capture")
def test_agentic_capture_switched_off_makes_one_extraction_call_instead_of_a_tool_run(
        hook_runner: Make, tmp_path: pathlib.Path) -> None:
    """While `agentic_capture` is on, which is its default, the capture hook on Claude
    Code starts the headless agent command with read-only memory tools and applies the
    proposals it returns. Switched off, capture makes one extraction call per turn and
    stores the facts that call returns. `FEATURE_DEFAULTS` in memvara/server/config.py
    and plugin/hooks/capture.py make this promise.

    The two runs are told apart by what the fake `claude` was asked for: an agentic run
    asks for `stream-json` output and is given an MCP configuration, and the single call
    asks for `json` output."""
    if sys.platform == "win32":
        pytest.skip(NO_FAKES)
    # One transcript for each run: capture skips a transcript it has already mined.
    turns = [(support.USER_TURN, support.ASSISTANT_TURN)]
    first = support.write_transcript("claude", tmp_path / "first.jsonl", turns)
    second = support.write_transcript("claude", tmp_path / "second.jsonl", turns)

    on_store = support.make_store(tmp_path / "on.db", memory=False)
    on_clis = FakeClis(tmp_path / "on-clis")
    on_clis.script("claude", support.PROPOSALS_REPLY)
    agentic = hook_runner("claude", env=support.store_env(on_store), stubs=on_clis).run(
        "capture", session="on", transcript_path=str(first))
    assert agentic.exit_code == 0
    (run,) = on_clis.calls("claude")
    assert "stream-json" in run.argv and "--mcp-config" in run.argv, run.argv
    assert any(re.match(r"turn=\d+c agentic searches=0 proposals=1 refused=0 stored=1 ",
                        line) for line in agentic.log("capture")), agentic.logs

    off_store = support.make_store(tmp_path / "off.db", memory=False)
    off_clis = FakeClis(tmp_path / "off-clis")
    off_clis.script("claude", support.FACTS_REPLY)
    single = hook_runner("claude", env={**support.store_env(off_store),
                                        "MEMVARA_FEATURE_AGENTIC_CAPTURE": "0"},
                         stubs=off_clis).run("capture", session="off",
                                             transcript_path=str(second))
    assert single.exit_code == 0
    (call,) = off_clis.calls("claude")
    assert "stream-json" not in call.argv and "--mcp-config" not in call.argv, call.argv
    assert "json" in call.argv
    assert any(re.match(r"turn=\d+c facts=1 stored=1 ", line)
               for line in single.log("capture")), single.logs
    assert not any("agentic" in line for line in single.log("capture")), single.logs

    for path in (on_store, off_store):
        with stores.file(path) as mem:
            assert [(c.subject, c.predicate, c.object)
                    for c in mem.scope(user=support.USER).get_all()] == [
                ("user", "lives_in", "Lisbon")]


@pytest.mark.covers("switch:project_scope")
def test_project_scope_switched_off_stops_the_hooks_sending_the_checkouts_project(
        hook_runner: Make, tmp_path: pathlib.Path) -> None:
    """While `project_scope` is on, which is its default, a hook works out the project
    from the working directory's git remote and sends it to the hosted service in the
    `memvara-project` header, so memories are kept per repository. Switched off, no
    project is sent. plugin/hooks/lib/project.py makes this promise.

    `HookRunner` switches the project scope off unless a test switches it on, so the first
    run says "1" explicitly. The hosted service is `FakeHostedMcp`."""
    _git_checkout(tmp_path / "work")
    with FakeHostedMcp() as hosted:
        env = {"MEMVARA_API_KEY": hosted.api_key, "MEMVARA_SERVER_URL": hosted.serve()}

        hook_runner("claude", env={**env, "MEMVARA_FEATURE_PROJECT_SCOPE": "1"}).run(
            "recall", session="on", prompt=support.PROMPT)
        sent = {request.header(PROJECT_HEADER) for request in hosted.requests}
        assert sent == {PROJECT}, sent

        before = len(hosted.requests)
        hook_runner("claude", env={**env, "MEMVARA_FEATURE_PROJECT_SCOPE": "0"}).run(
            "recall", session="off", prompt=support.PROMPT)
        later = hosted.requests[before:]
        assert later, "the hook with the switch off reached the service too"
        assert {request.header(PROJECT_HEADER) for request in later} == {None}


@pytest.mark.covers("switch:project_scope")
def test_project_scope_switched_off_stops_the_server_deriving_the_project(
        mcp: Start, tmp_path: pathlib.Path) -> None:
    """While `project_scope` is on, the MCP server works out the project from the git
    remote of the directory it runs in and binds its scope to it, so a fact about the
    project is stored in that project. Switched off, it derives no project. The comment
    above `FEATURE_DEFAULTS` in memvara/server/config.py and docs/DEPLOY.md make this
    promise.

    `McpProcess` switches the project scope off unless a test switches it on. The fact
    uses a predicate no vocabulary declares, because such a predicate is kept per project;
    a builtin such as `lives_in` describes the user and is stored without a project."""
    checkout = _git_checkout(tmp_path / "checkout")

    on = mcp(tmp_path / "on.db", cwd=checkout, features={"project_scope": True})
    on.initialize()
    # memory_stats names the bound scope, with the slashes of the project escaped.
    escaped = PROJECT.replace("/", "%2F")
    assert f"scope: default/tester/{escaped}/" in on.call("memory_stats").text
    assert not on.call("memory_remember", subject="billing", predicate="deploys_to",
                       object="frankfurt").is_error
    assert on.close() == 0

    off = mcp(tmp_path / "off.db", cwd=checkout, features={"project_scope": False})
    off.initialize()
    assert "scope: default/tester/*/" in off.call("memory_stats").text
    assert not off.call("memory_remember", subject="billing", predicate="deploys_to",
                        object="frankfurt").is_error
    assert off.close() == 0

    with stores.file(tmp_path / "on.db") as mem:
        assert [c.scope.project for c in mem.scope(user="tester", project=PROJECT)
                .get_all()] == [PROJECT]
    with stores.file(tmp_path / "off.db") as mem:
        assert [c.scope.project for c in mem.scope(user="tester").get_all()] == [None]
