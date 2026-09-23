"""The project scope the client hooks derive from the git remote, and how it reaches the server.

Every clone and every worktree of one repository must resolve to one project, and the hooks'
hosted calls must carry that project in the `Memvara-Project` header. The hooks cannot import
the library, so `plugin/hooks/lib/project.py` is a second copy of `memvara/project.py`; the
normalisation rows both copies must agree on live in `plugin/hooks/lib/project_vectors.json`.
"""

from __future__ import annotations

import hashlib
import json
import ntpath
import os
import posixpath
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
    monkeypatch.setattr(project, "CACHE_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(settings, "_LOADED", None)
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


@pytest.mark.parametrize("row", VECTORS["check_project"]["rows"], ids=lambda r: r["rule"])
def test_every_value_in_the_shared_check_rows_is_judged_as_recorded(row):
    """The server's `check_project` rule, which the hooks must apply before sending."""
    assert project.is_canonical(row["value"]) is row["valid"], row["rule"]


@pytest.mark.parametrize("length, valid", [(512, True), (513, False)])
def test_a_project_name_is_at_most_512_characters(length, valid):
    value = "example.com/" + "a" * (length - len("example.com/"))
    assert project.is_canonical(value) is valid


@needs_git
@pytest.mark.parametrize("remote", ["https://example.com/team/../repo",
                                    "git@host_name:team/repo.git"])
def test_a_remote_the_server_would_refuse_falls_back_to_the_path_form(tmp_path, remote):
    """The library's copy does this, and the server answers a refused header with 400."""
    main = _repo(tmp_path / "main", remote)
    assert project.normalise_remote(remote) is not None, "it normalises; the check refuses it"
    assert project.canonical_project(str(main)) == project.path_identity(
        os.path.realpath(str(main)))


def test_the_vectors_cover_every_rule_the_spec_names():
    """A vector file that lost its hard rows would still pass the test above."""
    rules = " ".join(row["rule"] for row in VECTORS["normalise"])
    for rule in ("credentials", "port kept", ".git stripped", "trailing slash",
                 "query and fragment", "scp-style", "lower-cased", "names no host"):
        assert rule in rules, f"no vector exercises: {rule}"


def test_the_case_insensitive_hosts_in_the_vectors_are_the_ones_the_code_folds():
    assert sorted(project.CASE_INSENSITIVE_HOSTS) == sorted(VECTORS["case_insensitive_hosts"])


@pytest.mark.parametrize("example", VECTORS["path_form"]["examples"],
                         ids=lambda e: e["root"])
def test_every_path_form_row_hashes_as_recorded(example):
    """POSIX and Windows roots alike, checked on every platform, because the function is
    pure: `canonical_project` only puts `realpath` in front of it."""
    assert project.path_identity(example["root"]) == example["project"]


@pytest.mark.parametrize("root", ["C:\\Users\\dev\\memvara", "c:/Users/dev/memvara",
                                  "C:/Users/dev/memvara/", "C:\\Users\\dev\\memvara\\"])
def test_every_spelling_of_one_windows_root_hashes_one_string(root):
    """`realpath` on Windows gives backslashes and whichever drive-letter case the system
    reports. Hashed as given, one repository would get a different name from the library,
    or from itself after the drive letter changed case."""
    expected = hashlib.sha256(b"c:/Users/dev/memvara").hexdigest()[:16]
    assert project.path_identity(root) == f"path:{expected}"


def test_the_rest_of_a_windows_path_keeps_its_case():
    """Only the drive letter folds; folding the rest would merge two directories on a
    case-sensitive volume."""
    assert project.path_identity("C:\\Src\\App") != project.path_identity("C:\\src\\app")


def test_the_filesystem_root_is_not_emptied():
    assert project.path_identity("/") == f"path:{hashlib.sha256(b'/').hexdigest()[:16]}"
    assert project.path_identity("/srv/app/") == project.path_identity("/srv/app")


@pytest.mark.parametrize("paths, common, root", [
    (ntpath, "C:\\src\\app\\.git", "C:\\src\\app"),
    (ntpath, "C:\\srv\\app.git", "C:\\srv\\app.git"),
    (posixpath, "/src/app/.git", "/src/app"),
    (posixpath, "/srv/app.git", "/srv/app.git"),
])
def test_the_main_working_tree_is_found_the_same_way_on_both_platforms(paths, common, root):
    """The directory holding `.git`, or a bare repository's own directory, with the
    platform's path rules passed in so that Windows is pinned on any machine."""
    assert project.main_root(common, paths) == root


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


#: Directories the cache tests resolve. Made absolute the way `resolve` makes them, so the
#: keys are the same on Windows, where "/a" becomes "D:\\a".
A = os.path.abspath("/a")
B = os.path.abspath("/b")


# -- the switch, the cache and the channel --------------------------------------------


def test_resolve_is_none_when_the_project_scope_switch_is_off(monkeypatch):
    monkeypatch.setattr(project, "canonical_project", lambda cwd: "github.com/o/r")
    monkeypatch.setenv("MEMVARA_FEATURE_PROJECT_SCOPE", "0")
    assert project.resolve("/anywhere") is None
    monkeypatch.setenv("MEMVARA_FEATURE_PROJECT_SCOPE", "1")
    assert project.resolve("/anywhere") == "github.com/o/r"


def test_resolve_caches_per_directory_so_a_prompt_does_not_pay_for_git(monkeypatch):
    """A lookup costs up to 30ms, which is the whole per-prompt budget of `recall.py`."""
    calls: list[str] = []

    def fake(cwd: str) -> "str | None":
        calls.append(cwd)
        return "github.com/o/r" if cwd == A else None

    monkeypatch.setattr(project, "canonical_project", fake)
    assert project.resolve(A, now=1000.0) == "github.com/o/r"
    assert project.resolve(A, now=1001.0) == "github.com/o/r"
    assert project.resolve(B, now=1002.0) is None
    assert project.resolve(B, now=1003.0) is None, "an answer of None is cached too"
    assert calls == [A, B]

    later = 1000.0 + project.CACHE_TTL_SECONDS + 10
    assert project.resolve(A, now=later) == "github.com/o/r"
    assert calls == [A, B, A], "a stale entry is recomputed"


def test_each_directory_has_its_own_cache_file(monkeypatch):
    """One shared file meant a whole-file rewrite, unlocked, by every repository's hooks."""
    monkeypatch.setattr(project, "canonical_project", lambda cwd: f"example.com/o/{os.path.basename(cwd)}")
    project.resolve(A, now=1.0)
    project.resolve(B, now=1.0)
    files = sorted(pathlib.Path(project.CACHE_DIR).glob("*.json"))
    assert len(files) == 2
    entries = sorted(json.loads(f.read_text(encoding="utf-8"))["cwd"] for f in files)
    assert entries == sorted([A, B])


def test_a_cache_entry_for_another_directory_is_a_miss(monkeypatch):
    """The entry repeats its directory, so a hash collision cannot answer for another one."""
    monkeypatch.setattr(project, "canonical_project", lambda cwd: "example.com/o/right")
    path = pathlib.Path(project._cache_path(A))
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"cwd": "/elsewhere", "project": "example.com/o/wrong",
                                "at": 1.0}), encoding="utf-8")
    assert project.resolve(A, now=2.0) == "example.com/o/right"


def test_resolve_survives_a_corrupt_or_unwritable_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(project, "canonical_project", lambda cwd: "github.com/o/r")
    path = pathlib.Path(project._cache_path(A))
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")
    assert project.resolve(A) == "github.com/o/r"
    path.write_text('["a list"]', encoding="utf-8")
    assert project.resolve(A, now=5.0) == "github.com/o/r"
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setattr(project, "CACHE_DIR", str(blocker / "projects"))
    assert project.resolve(A) == "github.com/o/r"


def test_prune_removes_cache_files_past_their_lifetime(monkeypatch):
    monkeypatch.setattr(project, "canonical_project", lambda cwd: None)
    project.resolve(A)
    old = pathlib.Path(project._cache_path(A))
    stale = old.stat().st_mtime - project.CACHE_TTL_SECONDS - 60
    os.utime(old, (stale, stale))
    project.resolve(B)
    project.prune()
    assert not old.exists()
    assert pathlib.Path(project._cache_path(B)).exists()


@needs_git
def test_a_repository_with_a_remote_costs_one_git_process(monkeypatch, tmp_path):
    """The common-dir lookup is only needed for the path form."""
    main = _repo(tmp_path / "main", "https://github.com/memvara/memvara")
    calls: list[list[str]] = []
    real = project._git
    monkeypatch.setattr(project, "_git", lambda args: calls.append(args) or real(args))
    assert project.canonical_project(str(main)) == "github.com/memvara/memvara"
    assert len(calls) == 1 and "remote" in calls[0]


@needs_git
def test_a_remote_that_is_not_utf8_means_no_project_rather_than_a_crash(tmp_path):
    """Git hands back raw bytes, and decoding them as text inside `subprocess.run` raised
    `UnicodeDecodeError` from a place nothing caught: every hook in such a repository failed
    its turn. A hook must never fail a turn, so this degrades to no project scope."""
    main = _repo(tmp_path / "main")
    # Written into the config file as bytes rather than passed to `git config` as an
    # argument: Windows decodes a command-line argument before git sees it, and the test
    # would then fail in its own setup.
    with open(main / ".git" / "config", "ab") as config:
        config.write(b'[remote "origin"]\n\turl = https://h.example/\xff/r\n')
    assert project.canonical_project(str(main)) is None


def test_a_directory_with_a_nul_byte_means_no_project():
    assert project.canonical_project("/tmp/a\0b") is None


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


def test_the_settings_file_is_read_once_per_process(monkeypatch):
    """Three hooks each ask several switches; one small file should be parsed once."""
    pathlib.Path(settings.SETTINGS).write_text(json.dumps({"recall_mark": False}),
                                               encoding="utf-8")
    opened: list[str] = []
    real_open = open

    def counting_open(path, *args, **kwargs):
        if str(path) == settings.SETTINGS:
            opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)
    for name in ("recall_mark", "status_line", "project_scope", "recall_mark"):
        settings.enabled(name)
    assert len(opened) == 1
    assert settings.enabled("recall_mark") is False


@pytest.mark.parametrize("body", ["not json", "[1, 2]", '{"recall_mark": "no"}'])
def test_an_unreadable_settings_file_means_the_default(body):
    pathlib.Path(settings.SETTINGS).write_text(body, encoding="utf-8")
    assert settings.enabled("recall_mark") is True


# -- parity with the library's copy ----------------------------------------------------

import memvara.project as library  # noqa: E402
from memvara.server.config import FEATURES as LIBRARY_FEATURES  # noqa: E402

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "project_vectors.json"


def test_the_two_vectors_files_are_the_same_bytes():
    """The library's tests read `tests/fixtures/`, the hooks' tests read `plugin/hooks/lib/`.

    A row added to one and not the other would let the two copies drift while each still
    passed its own suite.
    """
    assert FIXTURE.read_bytes() == (HOOKS / "lib" / "project_vectors.json").read_bytes()


@pytest.mark.parametrize("row", VECTORS["normalise"], ids=lambda r: r["rule"])
def test_both_copies_normalise_every_shared_remote_the_same_way(row):
    assert project.normalise_remote(row["remote"]) == library.normalize_remote(row["remote"])


def _library_accepts(value: str) -> bool:
    try:
        library.check_project(value)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("row", VECTORS["check_project"]["rows"], ids=lambda r: r["rule"])
def test_both_copies_accept_the_same_project_names(row):
    assert project.is_canonical(row["value"]) is _library_accepts(row["value"])


@pytest.mark.parametrize("length", [511, 512, 513])
def test_both_copies_share_the_length_limit(length):
    value = "example.com/" + "a" * (length - len("example.com/"))
    assert project.is_canonical(value) is _library_accepts(value)


@pytest.mark.parametrize("example", VECTORS["path_form"]["examples"],
                         ids=lambda e: e["root"])
def test_both_copies_spell_and_hash_a_root_the_same_way(example):
    assert project.path_identity(example["root"]) == library.path_identity(example["root"])


@pytest.mark.parametrize("common", ["C:\\src\\app\\.git", "C:\\srv\\app.git"])
def test_both_copies_find_the_main_working_tree_the_same_way(common):
    assert project.main_root(common, ntpath) == library.main_root(common, ntpath)


def test_both_copies_fold_case_on_the_same_hosts():
    assert project.CASE_INSENSITIVE_HOSTS == library._CASE_INSENSITIVE_HOSTS


@needs_git
@pytest.mark.parametrize("remote", [
    "git@github.com:Memvara/Memvara.git",
    "https://example.com/team/../repo",
    "/srv/git/memvara.git",
    None,
])
def test_both_copies_name_a_real_worktree_the_same(tmp_path, remote):
    """The end-to-end answer, not only the parts: one repository, one project, both copies."""
    main = _repo(tmp_path / "main", remote)
    _git("worktree", "add", "-q", str(tmp_path / "linked"), cwd=main)
    for checkout in (main, tmp_path / "linked"):
        assert project.canonical_project(str(checkout)) == \
            library.canonical_project(str(checkout))


def test_the_hooks_know_the_same_features_as_the_library():
    """`ServerConfig` refuses an unknown `MEMVARA_FEATURE_<NAME>`; the hooks must accept
    exactly the names it accepts, or one side ignores a switch the other honours."""
    assert settings.FEATURES == LIBRARY_FEATURES


def test_asking_for_a_feature_that_does_not_exist_is_a_bug_the_tests_catch():
    with pytest.raises(ValueError, match="not a feature"):
        settings.enabled("recall_marks")
