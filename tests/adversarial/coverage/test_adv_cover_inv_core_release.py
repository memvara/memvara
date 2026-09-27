"""Invariants of the release and plugin tooling that no older test checked.

The tooling is `release/publish_pypi.py`, which builds and uploads the Python package,
`scripts/sync_plugin_repos.py`, which copies the packaged skill into a plugin repository,
and `plugin/hooks/`, the client hooks that the plugin repositories vendor whole. Nothing
here uploads, pushes or reaches a network: each test runs the tool against a temporary
directory shaped like the checkout it would run in. The invariants are in the
"Invariants and assumptions" section of `docs/claude/release-and-plugins.md`.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable

import pytest

from harness import skips, stores
from harness.env import child_env
from harness.hooks import HOOKS_DIR, HookRunner, parse_reply

REPO = pathlib.Path(__file__).resolve().parents[3]
SKILL = REPO / "memvara" / "skills" / "memvara"
SYNC = REPO / "scripts" / "sync_plugin_repos.py"


def _tree(root: pathlib.Path, *, leave_out: tuple[str, ...] = ()) -> dict[str, bytes]:
    """Every file under `root` by its relative path, without build artifacts."""
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
            and path.name not in leave_out}


def _publish_pypi() -> ModuleType:
    """release/publish_pypi.py, loaded from its file under a name of its own."""
    spec = importlib.util.spec_from_file_location("publish_pypi_under_test",
                                                  REPO / "release" / "publish_pypi.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@skips.needs_toml
@pytest.mark.covers("inv:RP2")
def test_the_build_directory_is_deleted_before_every_build(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """docs/claude/release-and-plugins.md: "The build directory is rebuilt from scratch
    on every run." `twine upload dist/*` uploads whatever is in `dist/`, so an artifact
    left there by an earlier build would be published under the new version, and a
    published version can never be replaced.

    `build()` runs against a checkout whose `dist/` holds a stale sdist. The build step is
    replaced by a stand-in that writes a fresh wheel and sdist, as `python -m build`
    does, and records whether `dist/` was already gone when it started. The stale file
    must be gone and `build()` must return exactly the two fresh artifacts.
    """
    publish = _publish_pypi()
    monkeypatch.setattr(publish, "REPO", tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "memvara-0.0.9.tar.gz").write_bytes(b"built from an older commit")
    seen: list[bool] = []

    def build_step(*cmd: str, **_: Any) -> str:
        assert list(cmd[1:]) == ["-m", "build"], cmd
        seen.append(dist.exists())
        dist.mkdir()
        for name in ("memvara-1.0.0-py3-none-any.whl", "memvara-1.0.0.tar.gz"):
            (dist / name).write_bytes(b"fresh")
        return ""

    monkeypatch.setattr(publish, "run", build_step)
    made = publish.build()
    assert seen == [False], "dist/ still existed when the build started"
    assert [path.name for path in made] == ["memvara-1.0.0-py3-none-any.whl",
                                            "memvara-1.0.0.tar.gz"]
    assert sorted(path.name for path in dist.iterdir()) == [path.name for path in made]


def _sync(dest: pathlib.Path, home: pathlib.Path) -> None:
    done = subprocess.run([sys.executable, str(SYNC), str(dest)], capture_output=True,
                          text=True, env=child_env(home), timeout=60)
    assert done.returncode == 0, done.stderr


@pytest.mark.covers("inv:RP3")
def test_the_sync_replaces_a_downstream_edit_with_the_canonical_skill(
        tmp_path: pathlib.Path) -> None:
    """docs/claude/release-and-plugins.md: "The vendored skill is not edited downstream.
    Fix it here, then let the sync carry it." The sync is what carries it, so the sync
    must leave the plugin repository's copy byte for byte equal to the canonical skill in
    `memvara/skills/memvara/`, whatever the copy held before.

    A plugin checkout is given a vendored copy with a local edit to `SKILL.md` and a file
    the canonical tree does not have. After the sync the copy must be the canonical tree
    exactly: the edit undone, the extra file gone, and no transform applied to any byte.
    The one sanctioned transform, a single front-matter line in claude-memvara, belongs
    to that repository's own gate and never to the sync.
    """
    dest = tmp_path / "claude-memvara"
    vendored = dest / "plugin" / "skills" / "memvara"
    shutil.copytree(SKILL, vendored, ignore=shutil.ignore_patterns("__pycache__"))
    skill_md = vendored / "SKILL.md"
    skill_md.write_text(skill_md.read_text(encoding="utf-8").replace(
        "name: memvara", "name: memory", 1) + "\nA local note.\n", encoding="utf-8")
    (vendored / "references" / "local-only.md").write_text("Kept only here.\n",
                                                           encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()

    _sync(dest, home)

    assert _tree(vendored) == _tree(SKILL)


def _generate(tree: pathlib.Path, host: str, home: pathlib.Path) -> pathlib.Path:
    done = subprocess.run([sys.executable, str(tree / "tools" / "generate.py"), host],
                          capture_output=True, text=True, env=child_env(home),
                          cwd=str(tree), timeout=60)
    assert done.returncode == 0, done.stderr
    return pathlib.Path(done.stdout.strip())


@pytest.mark.covers("inv:RP4")
def test_a_plain_copy_of_the_hooks_tree_runs_in_a_plugin_checkout(
        tmp_path: pathlib.Path, hook_runner: Callable[..., HookRunner]) -> None:
    """docs/claude/release-and-plugins.md: "`plugin/hooks/` has no sanctioned transform
    at all." The canonical path and the vendored path are the same string, so the sync
    is a plain copy. That holds only while the tree needs nothing from this repository
    outside itself and nothing rewritten on the way.

    The tree is copied byte for byte to `plugin/hooks/` in a checkout that holds nothing
    else of this repository. The registration is generated there, as a plugin repository
    generates it, and the session-start command it names is run exactly as the client
    would run it. Its reply must equal the canonical tree's reply for the same store, and
    the copied tree must still equal the canonical one afterwards, apart from the
    generated manifest.
    """
    db = tmp_path / "memory.db"
    mem = stores.file(db, user="alice")
    mem.remember("user", "prefers", "tabs over spaces")
    mem.close()
    runner = hook_runner("claude", server_env={"MEMVARA_DB": str(db),
                                               "MEMVARA_USER": "alice"})
    canonical = runner.run("session_start")
    assert canonical.exit_code == 0, canonical.stderr

    plugin = tmp_path / "claude-memvara" / "plugin"
    copied = plugin / "hooks"
    shutil.copytree(HOOKS_DIR, copied,
                    ignore=shutil.ignore_patterns("__pycache__", "hooks.json"))
    manifest = json.loads(_generate(copied, "claude", tmp_path / "home").read_text(
        encoding="utf-8"))
    (entry,) = manifest["hooks"][runner.host.events["session_start"]]
    # The command is `python3 "<plugin root>/hooks/run.py" <hook> --host <host>`. It is
    # split on its quotes rather than by a shell parser, so a Windows path keeps its
    # backslashes.
    interpreter, script, rest = entry["hooks"][0]["command"].split('"')
    assert interpreter.strip() == "python3"
    script = script.replace("${CLAUDE_PLUGIN_ROOT}", str(plugin))
    done = subprocess.run([sys.executable, script, *rest.split()], capture_output=True,
                          env=runner.environment, cwd=str(runner.cwd),
                          input=json.dumps(runner.payload("session_start")).encode(),
                          timeout=60)
    assert done.returncode == 0, done.stderr.decode(errors="replace")
    reply = parse_reply(done.stdout.decode("utf-8"), what="the copied session_start")
    assert reply == canonical.reply
    assert "tabs over spaces" in json.dumps(reply)
    assert _tree(copied, leave_out=("hooks.json",)) == _tree(HOOKS_DIR,
                                                            leave_out=("hooks.json",))


@pytest.mark.covers("inv:RP5")
def test_the_hooks_manifest_is_generated_beside_the_tree_and_never_tracked_here(
        tmp_path: pathlib.Path) -> None:
    """docs/claude/release-and-plugins.md: "The hooks manifest under `plugin/hooks/` is
    generated, not vendored." Every plugin repository registers a different client, so
    a canonical copy would ship one client's manifest to all of them. It is ignored here
    so that it can never become canonical.

    Three things are checked: generating writes the manifest as `hooks.json` at the root
    of the hooks tree; git ignores exactly that path in this repository; and this
    repository tracks no `hooks.json` anywhere under `plugin/hooks/`.
    """
    copied = tmp_path / "hooks"
    shutil.copytree(HOOKS_DIR, copied,
                    ignore=shutil.ignore_patterns("__pycache__", "hooks.json"))
    written = _generate(copied, "claude", tmp_path / "home")
    assert written == copied / "hooks.json"
    relative = (HOOKS_DIR / written.relative_to(copied)).relative_to(REPO).as_posix()
    assert relative == "plugin/hooks/hooks.json"

    ignored = subprocess.run(["git", "-C", str(REPO), "check-ignore", "--no-index", "-v",
                              relative], capture_output=True, text=True, timeout=60)
    assert ignored.returncode == 0, "git does not ignore the generated hooks manifest"
    assert ignored.stdout.startswith(".gitignore:"), ignored.stdout
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "plugin/hooks"],
                             capture_output=True, text=True, check=True,
                             timeout=60).stdout.split()
    assert tracked, "git tracks nothing under plugin/hooks"
    assert not [path for path in tracked if path.endswith("hooks.json")], tracked
