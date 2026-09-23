"""The project scope the client hooks derive from the git remote, and how it reaches the server.

Every clone and every worktree of one repository must resolve to one project, and the hooks'
hosted calls must carry that project in the `Memvara-Project` header. The hooks cannot import
the library, so `plugin/hooks/lib/project.py` is a second copy of `memvara/project.py`; the
normalisation rows both copies must agree on live in `plugin/hooks/lib/project_vectors.json`.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

from lib import hosted, ipc, project, settings  # noqa: E402

VECTORS = json.loads((HOOKS / "lib" / "project_vectors.json").read_text(encoding="utf-8"))

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Keep every test away from the real `~/.memvara` and from the caller's switches."""
    monkeypatch.setattr(settings, "SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(project, "CACHE", str(tmp_path / "hooks" / "projects.json"))
    for name in list(os.environ):
        if name.startswith("MEMVARA_FEATURE_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv(project.ENV, raising=False)


def _git(*args: str, cwd: pathlib.Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main", *args],
        cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _repo(path: pathlib.Path, remote: "str | None" = None) -> pathlib.Path:
    path.mkdir(parents=True)
    _git("init", "-q", cwd=path)
    _git("commit", "-q", "--allow-empty", "-m", "first", cwd=path)
    if remote:
        _git("remote", "add", "origin", remote, cwd=path)
    return path


# -- normalisation ---------------------------------------------------------------------


@pytest.mark.parametrize("row", VECTORS["normalise"], ids=lambda r: r["rule"])
def test_every_remote_in_the_shared_vectors_normalises_as_recorded(row):
    """The rows are the contract with the library's copy, so they are read, not restated."""
    assert project.normalise_remote(row["remote"]) == row["project"], row["rule"]


def test_the_vectors_cover_every_rule_the_spec_names():
    """A vector file that lost its hard rows would still pass the test above."""
    rules = " ".join(row["rule"] for row in VECTORS["normalise"])
    for rule in ("credentials", "port kept", ".git stripped", "trailing slash",
                 "query and fragment", "scp-style", "lower-cased", "names no host"):
        assert rule in rules, f"no vector exercises: {rule}"


def test_the_case_insensitive_hosts_in_the_vectors_are_the_ones_the_code_folds():
    assert sorted(project.CASE_INSENSITIVE_HOSTS) == sorted(VECTORS["case_insensitive_hosts"])


@pytest.mark.parametrize("example", VECTORS["path_form"]["examples"])
def test_the_path_form_is_a_short_digest_of_the_repository_root(example):
    assert project.path_identity(example["root"]) == example["project"]
    digest = hashlib.sha256(example["root"].encode("utf-8")).hexdigest()
    assert example["project"] == (VECTORS["path_form"]["prefix"]
                                  + digest[:VECTORS["path_form"]["hex_chars"]])


# -- canonical_project against real repositories -------------------------------------


@needs_git
def test_a_worktree_resolves_to_the_same_project_as_its_main_repository(tmp_path):
    main = _repo(tmp_path / "main", "git@github.com:memvara/memvara.git")
    _git("worktree", "add", "-q", str(tmp_path / "linked"), cwd=main)
    assert project.canonical_project(str(main)) == "github.com/memvara/memvara"
    assert project.canonical_project(str(tmp_path / "linked")) == "github.com/memvara/memvara"


@needs_git
def test_an_https_clone_and_an_ssh_clone_resolve_to_one_project(tmp_path):
    https = _repo(tmp_path / "a", "https://github.com/memvara/memvara.git")
    ssh = _repo(tmp_path / "b", "git@github.com:memvara/memvara.git")
    assert project.canonical_project(str(https)) == project.canonical_project(str(ssh))


@needs_git
def test_a_subdirectory_resolves_to_its_repository(tmp_path):
    main = _repo(tmp_path / "main", "https://github.com/memvara/memvara")
    (main / "deep" / "er").mkdir(parents=True)
    assert project.canonical_project(str(main / "deep" / "er")) == "github.com/memvara/memvara"


@needs_git
def test_a_repository_without_a_remote_gets_the_path_form_of_its_main_root(tmp_path):
    """Worktrees of an unpushed repository still share one project."""
    main = _repo(tmp_path / "main")
    _git("worktree", "add", "-q", str(tmp_path / "linked"), cwd=main)
    expected = project.path_identity(os.path.realpath(str(main)))
    assert project.canonical_project(str(main)) == expected
    assert project.canonical_project(str(tmp_path / "linked")) == expected


@needs_git
def test_a_remote_that_names_no_host_falls_back_to_the_path_form(tmp_path):
    main = _repo(tmp_path / "main", "/srv/git/memvara.git")
    assert project.canonical_project(str(main)) == project.path_identity(
        os.path.realpath(str(main)))


def test_outside_a_repository_there_is_no_project(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert project.canonical_project(str(plain)) is None
    assert project.canonical_project("") is None
    assert project.canonical_project(str(tmp_path / "does-not-exist")) is None


def test_a_missing_git_binary_means_no_project(monkeypatch, tmp_path):
    """The hooks must fail silently: no git is the same as no repository."""
    monkeypatch.setenv("PATH", str(tmp_path))
    assert project.canonical_project(str(tmp_path)) is None


# -- the switch, the cache and the channel --------------------------------------------


def test_resolve_is_none_when_the_project_scope_switch_is_off(monkeypatch):
    monkeypatch.setattr(project, "canonical_project", lambda cwd: "github.com/o/r")
    monkeypatch.setenv("MEMVARA_FEATURE_PROJECT_SCOPE", "0")
    assert project.resolve("/anywhere") is None
    monkeypatch.setenv("MEMVARA_FEATURE_PROJECT_SCOPE", "1")
    assert project.resolve("/anywhere") == "github.com/o/r"


def test_resolve_caches_per_directory_so_a_prompt_does_not_pay_for_git(monkeypatch):
    """Two git calls cost about 30ms, which is the whole per-prompt budget of `recall.py`."""
    calls: list[str] = []

    def fake(cwd: str) -> "str | None":
        calls.append(cwd)
        return "github.com/o/r" if cwd == "/a" else None

    monkeypatch.setattr(project, "canonical_project", fake)
    assert project.resolve("/a", now=1000.0) == "github.com/o/r"
    assert project.resolve("/a", now=1001.0) == "github.com/o/r"
    assert project.resolve("/b", now=1002.0) is None
    assert project.resolve("/b", now=1003.0) is None, "an answer of None is cached too"
    assert calls == ["/a", "/b"]

    later = 1000.0 + project.CACHE_TTL_SECONDS + 10
    assert project.resolve("/a", now=later) == "github.com/o/r"
    assert calls == ["/a", "/b", "/a"], "a stale entry is recomputed"
    cached = json.loads(pathlib.Path(project.CACHE).read_text(encoding="utf-8"))
    assert "/b" not in cached, "entries past their lifetime are pruned on write"


def test_resolve_survives_a_corrupt_or_unwritable_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(project, "canonical_project", lambda cwd: "github.com/o/r")
    pathlib.Path(project.CACHE).parent.mkdir(parents=True)
    pathlib.Path(project.CACHE).write_text("not json", encoding="utf-8")
    assert project.resolve("/a") == "github.com/o/r"
    pathlib.Path(project.CACHE).write_text('["a list"]', encoding="utf-8")
    assert project.resolve("/a", now=5.0) == "github.com/o/r"
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setattr(project, "CACHE", str(blocker / "projects.json"))
    assert project.resolve("/a") == "github.com/o/r"


def test_resolve_with_no_directory_uses_the_process_directory(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(project, "canonical_project", lambda cwd: seen.append(cwd) or None)
    project.resolve("")
    assert seen == [os.getcwd()]


def test_bind_sets_the_channel_and_clears_it(monkeypatch):
    monkeypatch.setattr(project, "resolve", lambda cwd: "github.com/o/r" if cwd else None)
    assert project.bind("/a") == "github.com/o/r"
    assert os.environ[project.ENV] == "github.com/o/r"
    assert project.bind("") is None
    assert project.ENV not in os.environ, "a stale project must not leak into a later call"


def test_the_daemon_address_differs_by_project(monkeypatch, tmp_path):
    """One resident daemon serves one project, so it cannot answer for another repository.

    The daemon inherits the hook's environment, and `store_key` reads the channel from it,
    so the client and the daemon it spawns compute one address.
    """
    monkeypatch.delenv("MEMVARA_DB", raising=False)
    monkeypatch.setattr(ipc, "server_env", lambda: {})
    monkeypatch.setattr(ipc, "_HOME", str(tmp_path))
    monkeypatch.setenv(project.ENV, "github.com/o/a")
    first = ipc.store_key()
    monkeypatch.setenv(project.ENV, "github.com/o/b")
    second = ipc.store_key()
    monkeypatch.delenv(project.ENV)
    assert len({first, second, ipc.store_key()}) == 3


def test_a_local_store_keeps_one_daemon_whatever_the_project(monkeypatch, tmp_path):
    """The local route does not read the header, so splitting its daemon would buy nothing."""
    monkeypatch.setenv("MEMVARA_DB", str(tmp_path / "store.db"))
    monkeypatch.setattr(ipc, "server_env", lambda: {})
    monkeypatch.setenv(project.ENV, "github.com/o/a")
    first = ipc.store_key()
    monkeypatch.setenv(project.ENV, "github.com/o/b")
    assert ipc.store_key() == first


# -- the header on the wire -----------------------------------------------------------


class _Response:
    status = 200

    def __init__(self, body: dict) -> None:
        self._raw = json.dumps(body).encode()

    def read(self) -> bytes:
        return self._raw

    def getheader(self, name: str) -> "str | None":
        return None


class _Connection:
    def __init__(self) -> None:
        self.headers: list[dict] = []

    def request(self, method, path, body, headers) -> None:
        self.headers.append(dict(headers))

    def getresponse(self) -> _Response:
        return _Response({"jsonrpc": "2.0", "id": 1, "result": {}})

    def close(self) -> None:
        pass


def _client(monkeypatch) -> "tuple[hosted.HostedRecall, _Connection]":
    conn = _Connection()
    client = hosted.HostedRecall("key", "https://example.invalid")
    monkeypatch.setattr(client, "_connect", lambda: conn)
    return client, conn


def test_every_hosted_call_carries_the_project_header(monkeypatch):
    monkeypatch.setenv(project.ENV, "github.com/memvara/memvara")
    client, conn = _client(monkeypatch)
    client._rpc("initialize", {})
    client._rpc("tools/call", {"name": "memory_stats", "arguments": {}})
    assert [h.get("memvara-project") for h in conn.headers] == [
        "github.com/memvara/memvara", "github.com/memvara/memvara"]


def test_no_project_means_no_header(monkeypatch):
    client, conn = _client(monkeypatch)
    client._rpc("initialize", {})
    assert "memvara-project" not in conn.headers[0]


@pytest.mark.parametrize("value", ["github.com/o/r\r\nx-evil: 1", "github.com/ö/r"])
def test_a_value_that_cannot_travel_as_a_header_is_not_sent(monkeypatch, value):
    """A header with a line break would inject another header; a non-ASCII one would
    raise inside `http.client` and turn every call into a failure."""
    monkeypatch.setenv(project.ENV, value)
    client, conn = _client(monkeypatch)
    client._rpc("initialize", {})
    assert "memvara-project" not in conn.headers[0]


# -- the switch store -----------------------------------------------------------------


def test_every_phase_one_hook_feature_defaults_to_on():
    for name in ("project_scope", "status_line", "recall_mark"):
        assert settings.enabled(name) is True, name


def test_the_settings_file_switches_a_feature_off(tmp_path):
    pathlib.Path(settings.SETTINGS).write_text(
        json.dumps({"recall_mark": False, "status_line": True}), encoding="utf-8")
    assert settings.enabled("recall_mark") is False
    assert settings.enabled("status_line") is True
    assert settings.enabled("project_scope") is True, "a missing key means the default"


def test_an_environment_variable_overrides_the_file(monkeypatch):
    pathlib.Path(settings.SETTINGS).write_text(json.dumps({"recall_mark": False}),
                                               encoding="utf-8")
    monkeypatch.setenv("MEMVARA_FEATURE_RECALL_MARK", "1")
    assert settings.enabled("recall_mark") is True
    monkeypatch.setenv("MEMVARA_FEATURE_STATUS_LINE", "0")
    assert settings.enabled("status_line") is False


@pytest.mark.parametrize("raw", ["", "maybe", "2"])
def test_an_unreadable_override_falls_back_to_the_file(monkeypatch, raw):
    pathlib.Path(settings.SETTINGS).write_text(json.dumps({"recall_mark": False}),
                                               encoding="utf-8")
    monkeypatch.setenv("MEMVARA_FEATURE_RECALL_MARK", raw)
    assert settings.enabled("recall_mark") is False


@pytest.mark.parametrize("body", ["not json", "[1, 2]", '{"recall_mark": "no"}'])
def test_an_unreadable_settings_file_means_the_default(body):
    pathlib.Path(settings.SETTINGS).write_text(body, encoding="utf-8")
    assert settings.enabled("recall_mark") is True
