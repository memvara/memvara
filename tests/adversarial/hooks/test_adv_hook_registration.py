"""The hook registration a client reads, generated from each host's record.

`plugin/hooks/tools/generate.py <host>` writes `hooks.json`, which tells a
Claude-Code-shaped client which command answers which event. This repository commits no
`hooks.json`: plugin/README.md says the path is ignored here, because seven plugin
repositories each generate their own and diff it there. So these tests check what can be
checked here: generating is repeatable, what it writes agrees with the host's record,
and a host that has no shell hooks is refused without touching the file that exists.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
from types import ModuleType

import pytest

from harness.env import child_env
from harness.hooks import HOOKS_DIR, host_record

from . import support

#: The hosts whose hooks are shell commands. OpenCode's are JavaScript (hosts/opencode.py).
SHELL_HOSTS = ("claude", "codex", "copilot", "cursor")

#: What each shell host's registration declares, from the measurements in its record:
#: the plugin-root expression each command starts from, whether capture is registered
#: to run in the background (only Claude Code honours that), and the context limit that
#: Codex needs declared on every hook that can carry context.
DECLARED = {
    "claude": ("${CLAUDE_PLUGIN_ROOT}", True, None),
    "codex": ("${PLUGIN_ROOT:-${CLAUDE_PLUGIN_ROOT}}", False, 32000),
    "copilot": ("${PLUGIN_ROOT:-${COPILOT_PLUGIN_ROOT}}", False, None),
    "cursor": ("${CURSOR_PLUGIN_ROOT:-${CLAUDE_PLUGIN_ROOT}}", False, None),
}


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    """plugin/hooks/tools/generate.py, loaded from its file under a name of its own."""
    spec = importlib.util.spec_from_file_location("hook_registration_generator",
                                                  HOOKS_DIR / "tools" / "generate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_of_the_hooks(tmp_path: pathlib.Path) -> pathlib.Path:
    """A copy of plugin/hooks, so that generating never writes into this checkout."""
    tree = tmp_path / "hooks"
    shutil.copytree(HOOKS_DIR, tree, ignore=shutil.ignore_patterns("__pycache__", "hooks.json"))
    return tree


def _generate(tree: pathlib.Path, host: str,
              home: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(tree / "tools" / "generate.py"), host],
                          capture_output=True, text=True, env=child_env(home),
                          cwd=str(tree), timeout=60)


@pytest.mark.parametrize("host", SHELL_HOSTS)
def test_the_registration_declares_what_the_host_record_measured(
        generator: ModuleType, host: str) -> None:
    record = host_record(host)
    body = json.loads(generator.registration(record))
    root, background_capture, context_limit = DECLARED[host]
    assert body["description"] == record.description
    assert set(body["hooks"]) == set(support.EVENTS[host].values())
    for hook, event in support.EVENTS[host].items():
        (entry,) = body["hooks"][event]
        (command,) = entry["hooks"]
        expected: dict[str, object] = {
            "type": "command",
            "command": f'python3 "{root}/hooks/run.py" {hook} --host {host}',
            "timeout": support.LIMITS[host][hook],
        }
        if hook == "capture" and background_capture:
            expected["async"] = True
        if hook != "capture" and context_limit is not None:
            expected["additionalContextLimit"] = context_limit
        assert command == expected, (hook, command)
        if hook == "approve":
            assert entry["matcher"] == record.approve.matcher
        else:
            assert "matcher" not in entry


@pytest.mark.parametrize("host", SHELL_HOSTS)
def test_generating_twice_writes_the_bytes_the_record_builds(
        generator: ModuleType, host: str, tmp_path: pathlib.Path,
        tmp_path_factory: pytest.TempPathFactory) -> None:
    tree = _copy_of_the_hooks(tmp_path)
    home = tmp_path_factory.mktemp("home")
    written = []
    for _ in range(2):
        done = _generate(tree, host, home)
        assert (done.returncode, done.stdout.strip()) == (0, str(tree / "hooks.json")), done
        written.append((tree / "hooks.json").read_bytes())
    assert written[0] == written[1] == generator.registration(host_record(host))


def test_a_host_with_no_shell_hooks_is_refused_and_the_file_there_is_kept(
        tmp_path: pathlib.Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """OpenCode's hooks are JavaScript, so there is no shell registration to build.
    generate.py builds before it opens the file, so the refusal cannot empty it."""
    tree = _copy_of_the_hooks(tmp_path)
    kept = b'{"kept": true}\n'
    (tree / "hooks.json").write_bytes(kept)
    done = _generate(tree, "opencode", tmp_path_factory.mktemp("home"))
    assert done.returncode == 1
    assert "JavaScript" in done.stderr
    assert (tree / "hooks.json").read_bytes() == kept
