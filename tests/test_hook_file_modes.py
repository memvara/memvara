"""Everything the plugin keeps under `~/.memvara` is private to the account that runs it.

The hooks write logs that quote a person's prompts and a model's replies (`capture.log`,
`recall.log`, `recall-sample.log`, `hooks.log`), the token ledger `usage.jsonl`,
`capture-state.json`, which names every transcript the capture hook has mined, and small
state and lock files. They were created with the process's default modes, so under the
usual umask of 022 every account on the machine could read them, and `~/.memvara` itself
was 0755 whenever a hook, `memvara-mcp login` or `memvara-mcp init` created it.

Each test runs the writers under umask 022, in a child process whose home directory is the
test's own, and reads the modes back from the disk.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import stat
import subprocess
import sys

import pytest

from harness.env import REPO, child_env

pytestmark = pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX")

HOOKS = REPO / "plugin" / "hooks"

#: Every Python writer the hooks have, run once each. `capture` is imported for its state
#: writer only; its `main` does not run on import.
WRITERS = r"""
import os
import sys

os.umask(0o022)
sys.path.insert(0, sys.argv[1])

import capture
from lib import counts, ipc, read_model, usage, write

write.log("a capture line")
for name in ("recall", "hooks", "recall-sample"):
    ipc.log_line(name, "a line")
usage.JsonlRecorder().counter("write.tokens_in", 1)
capture._write_state({})
counts.bump("session-1", "recalled")
read_model.save({"checked": True})
ipc.runtime_dir()
"""

#: The files the writers above must have left behind, relative to `~/.memvara`.
WRITTEN = {".hooks/capture.log", ".hooks/recall.log", ".hooks/hooks.log",
           ".hooks/recall-sample.log", ".hooks/usage.jsonl", ".hooks/capture-state.json",
           ".hooks/counts/session-1.json", ".hooks/counts/.lock",
           ".hooks/read_model.json", ".hooks/read_model.json.lock"}


@pytest.fixture()
def umask_022():
    """The umask most accounts run with, for this process and every child it starts."""
    old = os.umask(0o022)
    yield
    os.umask(old)


def run_writers(home: pathlib.Path) -> None:
    done = subprocess.run([sys.executable, "-c", WRITERS, str(HOOKS)], env=child_env(home),
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def modes(root: pathlib.Path) -> dict[str, int]:
    """Every directory and file under `root`, and `root` itself, with its permission bits."""
    found = {".": stat.S_IMODE(root.stat().st_mode)}
    for path in root.rglob("*"):
        if path.is_symlink() or stat.S_ISSOCK(path.lstat().st_mode):
            continue
        found[str(path.relative_to(root))] = stat.S_IMODE(path.stat().st_mode)
    return found


def readable_by_others(root: pathlib.Path) -> dict[str, str]:
    return {name: oct(mode) for name, mode in modes(root).items() if mode & 0o077}


def test_the_hooks_create_their_directories_0700_and_their_files_0600(tmp_path, umask_022):
    home = tmp_path / "home"
    home.mkdir()
    run_writers(home)
    root = home / ".memvara"
    assert WRITTEN <= set(modes(root)), "a writer did not run"
    assert readable_by_others(root) == {}, "another account can read these"


def test_the_hooks_repair_what_an_older_version_left_readable(tmp_path, umask_022):
    """A directory or file an earlier version created with the default modes is made
    private the first time a hook in a process uses it."""
    home = tmp_path / "home"
    hooks = home / ".memvara" / ".hooks"
    (hooks / "counts").mkdir(parents=True)
    for directory in (home / ".memvara", hooks, hooks / "counts"):
        directory.chmod(0o755)
    for name in WRITTEN:
        path = home / ".memvara" / name
        path.write_text("{}" if name.endswith(".json") else "", encoding="utf-8")
        path.chmod(0o644)
    run_writers(home)
    assert readable_by_others(home / ".memvara") == {}, "an old mode was left as it was"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node for the JS shim")
def test_the_javascript_hooks_log_privately_too(tmp_path, umask_022):
    """The OpenCode plugin logs through `js/shim.mjs`, into the same directory."""
    home = tmp_path / "home"
    (home / ".memvara" / ".hooks").mkdir(parents=True)
    (home / ".memvara").chmod(0o755)
    (home / ".memvara" / ".hooks").chmod(0o755)
    shim = (HOOKS / "js" / "shim.mjs").as_uri()
    script = f"import {{ note }} from {shim!r}; note('hooks', 'a line')"
    done = subprocess.run(["node", "--input-type=module", "-e", script],
                          env=child_env(home), capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert (home / ".memvara" / ".hooks" / "hooks.log").exists()
    assert readable_by_others(home / ".memvara") == {}


def test_login_and_init_create_the_memvara_directory_0700(tmp_path, umask_022):
    """The two commands a person runs before any hook does, and each can be the one that
    creates `~/.memvara`."""
    from memvara.server.init import _store_directory
    from memvara.server.login import _write_credentials

    signed_in = tmp_path / "a" / ".memvara"
    _write_credentials(api_key="mv_test", project="p", server_url="https://example.test",
                       path=signed_in / "credentials.json")
    stored = tmp_path / "b" / ".memvara"
    _store_directory(stored / "memory.db")
    for directory in (signed_in, stored):
        assert oct(stat.S_IMODE(directory.stat().st_mode)) == "0o700"


def test_login_and_init_make_an_existing_memvara_directory_private(tmp_path, umask_022,
                                                                   monkeypatch):
    """`mkdir(mode=0o700, exist_ok=True)` sets the mode only on a directory it creates, so
    `memvara-mcp init` and `login` left a `~/.memvara` that an earlier version had created
    0755 as open as they found it. Each command now takes the permissions for group and
    others off it, as the hooks do, and leaves the home directory above it as it was."""
    import io

    from memvara.server.init import init
    from memvara.server.login import login
    from test_login import (AUTH_BODY, FakeResponse, _install_fake_httpx, _no_browser,
                            _no_loopback, approved)

    home = pathlib.Path(os.path.expanduser("~"))  # the test's own, from conftest
    home.chmod(0o755)
    root = home / ".memvara"
    root.mkdir()
    root.chmod(0o755)
    project = tmp_path / "project"
    project.mkdir()
    assert init(["--agent", "cursor", "--dir", str(project), "--mode", "local"], env={},
                stdout=io.StringIO(), stderr=io.StringIO()) == 0
    assert oct(stat.S_IMODE(root.stat().st_mode)) == "0o700", "init left it open"

    root.chmod(0o755)
    _no_browser(monkeypatch)
    _no_loopback(monkeypatch)
    _install_fake_httpx(monkeypatch, {"device/authorize": [FakeResponse(201, AUTH_BODY)],
                                      "device/token": [FakeResponse(200, approved())]})
    assert login(["--credentials", str(root / "credentials.json")], env={},
                 stdout=io.StringIO(), stderr=io.StringIO()) == 0
    assert oct(stat.S_IMODE(root.stat().st_mode)) == "0o700", "login left it open"
    assert oct(stat.S_IMODE(home.stat().st_mode)) == "0o755", "the home directory changed"


def test_init_makes_each_directory_from_memvara_down_to_the_store_private(tmp_path,
                                                                        umask_022):
    """A store named deeper under `~/.memvara` sits in directories an earlier version may
    have created 0755 too. `init` creates each missing one 0700 and takes the permissions
    for group and others off each one that exists, from `~/.memvara` down to the store's
    own directory."""
    import io

    from memvara.server.init import init

    root = pathlib.Path(os.path.expanduser("~")) / ".memvara"
    (root / "stores").mkdir(parents=True)
    for directory in (root, root / "stores"):
        directory.chmod(0o755)
    store = root / "stores" / "work" / "memory.db"
    project = tmp_path / "project"
    project.mkdir()
    assert init(["--agent", "cursor", "--dir", str(project), "--mode", "local",
                 "--db", str(store)], env={}, stdout=io.StringIO(),
                stderr=io.StringIO()) == 0
    assert {str(d.relative_to(root)): oct(stat.S_IMODE(d.stat().st_mode))
            for d in (root, root / "stores", store.parent)} == {
        ".": "0o700", "stores": "0o700", "stores/work": "0o700"}
