"""One project name per repository, whichever clone, worktree or remote spelling is used.

The failure these tests prevent is quiet. If two checkouts of one repository resolve to two
project names, a fact learned in one is simply never recalled in the other, and nothing
reports it: each checkout sees a store that looks complete. The same is true in reverse
when two different repositories collide on one name, except that then one repository's
facts appear in the other.

Most tests drive `canonical_project` through a fake git runner, so they need no repository
on disk. Two tests run the real git binary against a repository made in `tmp_path`, because
the worktree case depends on what git actually prints, and they skip when git is absent.

The plugin hooks carry their own copy of the normaliser, because they run without the
library. `fixtures/project_vectors.json` is a byte-identical copy of the rows the hooks'
tests read, and this file runs the library against every one of them, so the two copies
cannot drift apart without a failure here.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import ntpath
import os
import posixpath
import pathlib
import shutil
import subprocess

import httpx
import pytest

from memvara import AsyncMemvara, HashingEmbedder, Memvara, NullLLM
from memvara import project as project_module
from memvara.project import canonical_project, check_project, normalize_remote
from memvara.remote import hydrate
from memvara.remote.aio import AsyncRemoteMemvara
from memvara.remote.api import PROJECT_HEADER, RemoteMemvara
from memvara.server import MemvaraMCPServer, ServerConfig, build_memvara, main
from memvara.server import config as config_module
from memvara.server.config import FEATURES, ConfigError

VECTORS = pathlib.Path(__file__).resolve().parent / "fixtures" / "project_vectors.json"


# -- normalising a remote ------------------------------------------------------

@pytest.mark.parametrize("remote", [
    "git@github.com:memvara/memvara-cloud.git",
    "https://github.com/memvara/memvara-cloud.git",
    "https://github.com/memvara/memvara-cloud",
    "https://user:token@github.com/memvara/memvara-cloud.git",
    "ssh://git@github.com/memvara/memvara-cloud.git",
    "https://github.com/memvara/memvara-cloud/",
    "https://GitHub.com/Memvara/Memvara-Cloud.git",
    "git+ssh://git@github.com/memvara/memvara-cloud.git",
    "https://github.com/memvara/memvara-cloud.git?ref=main#readme",
    "  git@github.com:memvara/memvara-cloud.git\n",
])
def test_every_spelling_of_one_github_remote_names_one_project(remote):
    """An https clone and an ssh clone of one repository must share memory. They are the
    two remotes a person is most likely to have side by side on one machine."""
    assert normalize_remote(remote) == "github.com/memvara/memvara-cloud"


def test_case_is_kept_on_a_forge_not_known_to_ignore_it():
    """Folding owner and repository on every host would merge two projects on a forge that
    treats `Team` and `team` as different. Only the host always folds, because DNS does."""
    assert normalize_remote("https://GIT.example.com/Team/Repo.git") == \
        "git.example.com/Team/Repo"


def test_a_port_written_in_the_remote_is_part_of_the_name():
    """Two forges on one host with different ports are two servers. The port is kept as
    written, default or not, because that is the rule the plugin hooks follow too."""
    assert normalize_remote("https://git.example.com:8443/team/repo") == \
        "git.example.com:8443/team/repo"
    assert normalize_remote("ssh://git@git.example.com:2222/team/repo.git") == \
        "git.example.com:2222/team/repo"


def test_gitlab_subgroups_keep_every_path_segment():
    assert normalize_remote("git@gitlab.com:Group/Sub/Repo.git") == "gitlab.com/group/sub/repo"


def test_percent_encoding_and_a_home_directory_are_kept_as_written():
    """Neither is decoded or rewritten, so the library and the hooks, which do not
    decode either, produce one name for one remote."""
    assert normalize_remote("https://git.example.com/team/my%2Drepo") == \
        "git.example.com/team/my%2Drepo"
    assert normalize_remote("git@server.lan:~alice/repo.git") == "server.lan/~alice/repo"


def test_the_library_and_the_hooks_agree_on_every_shared_row():
    """Every row of the vectors file the plugin hooks' tests also read."""
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    assert sorted(vectors["case_insensitive_hosts"]) == \
        sorted(project_module._CASE_INSENSITIVE_HOSTS)
    for row in vectors["normalise"]:
        assert normalize_remote(row["remote"]) == row["project"], row["rule"]
    for example in vectors["path_form"]["examples"]:
        # The pure half of the path form, so a Windows root is checked on every
        # platform. `canonical_project` adds only `realpath` in front of it.
        assert project_module.path_identity(example["root"]) == example["project"], \
            example["root"]


def test_the_library_and_the_hooks_agree_on_which_names_the_server_accepts():
    """The hooks fall back to the path form when a normalised remote fails this check,
    so the two copies must agree on it row for row, or one of them sends a header the
    server refuses."""
    rows = json.loads(VECTORS.read_text(encoding="utf-8"))["check_project"]["rows"]
    assert rows
    for row in rows:
        try:
            check_project(row["value"])
            valid = True
        except ValueError:
            valid = False
        assert valid is row["valid"], row["rule"]


def test_the_shared_vectors_are_the_hooks_own_file_when_both_are_here():
    """The fixture is a copy so that this suite does not depend on the hooks tree. Once
    both are in one checkout, the copy has to be the same bytes as the original."""
    hooks = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks" / "lib" \
        / "project_vectors.json"
    if not hooks.exists():
        pytest.skip("the hooks' copy of the vectors is not in this checkout yet")
    assert hooks.read_bytes() == VECTORS.read_bytes()


@pytest.mark.parametrize("remote", [
    "",
    "   ",
    "/srv/git/repo.git",
    "../sibling",
    "file:///srv/git/repo.git",
    "C:\\src\\repo",
    "https://",
    "https://github.com/",
    "https://github.com/.git",
    "https://github.com:notaport/o/r",
    "git@github.com:",
    "x:o/r",
])
def test_a_remote_naming_no_clonable_repository_is_not_a_project(remote):
    """A local path names a directory on this machine only, so it cannot be the shared
    identity of a repository. `canonical_project` falls back to the path form for it."""
    assert normalize_remote(remote) is None


# -- checking a name -------------------------------------------------------------

@pytest.mark.parametrize("value", [
    "github.com/memvara/memvara",
    "git.example.com:8443/team/repo",
    "gitlab.com/group/sub/repo",
    "server.lan/repo",
    "path:0123456789abcdef",
])
def test_every_name_the_resolver_produces_passes_the_check(value):
    assert check_project(value) == value


@pytest.mark.parametrize("value, reason", [
    ("", "cannot be empty"),
    (None, "cannot be empty"),
    ("x" * 600, "at most 512"),
    ("path:0123", "exactly 16"),
    ("path:0123456789ABCDEF", "exactly 16"),
    ("github.com/o/has space", "whitespace"),
    ("github.com/o/r\x00", "whitespace or a control"),
    ("my-project", "needs a host"),
    ("GitHub.com/o/r", "lower-case host"),
    ("-bad.com/o/r", "lower-case host"),
    ("github.com//r", "empty, '.' or '..'"),
    ("github.com/o/..", "empty, '.' or '..'"),
    ("github.com/o/r/", "empty, '.' or '..'"),
])
def test_a_malformed_name_is_refused_with_the_rule_it_broke(value, reason):
    """The hosted deployment returns this message as the reason for a 400, and the MCP
    server prints it at startup, so it has to say what to change."""
    with pytest.raises(ValueError, match=reason):
        check_project(value)


# -- resolving a directory, with a fake git --------------------------------------

class FakeGit:
    """Answers the two questions `canonical_project` asks, and records where it asked."""

    def __init__(self, common: str | None, remote: str | None) -> None:
        self.common, self.remote = common, remote
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def __call__(self, args, cwd):
        self.calls.append((tuple(args), cwd))
        return self.common if args[0] == "rev-parse" else self.remote


def test_a_directory_outside_any_repository_has_no_project():
    """`None` is what every caller had before project scope existed, so a server started
    outside a repository behaves exactly as it always did."""
    git = FakeGit(common=None, remote="git@github.com:a/b.git")
    assert canonical_project("/tmp/nowhere", run=git) is None
    assert len(git.calls) == 1, "no remote lookup without a repository"


def test_the_remote_is_read_from_the_common_git_directory(tmp_path):
    """Reading the remote from the worktree would also work today, because git shares
    remote configuration across worktrees. Reading it from the common directory makes the
    resolution independent of that detail."""
    common = tmp_path / "main" / ".git"
    common.mkdir(parents=True)
    git = FakeGit(common=str(common), remote="git@github.com:Acme/App.git")
    assert canonical_project(str(tmp_path / "wt"), run=git) == "github.com/acme/app"
    assert git.calls[1][0] == ("--git-dir", os.path.realpath(common), "remote", "get-url",
                               "origin")


def test_a_relative_common_directory_is_resolved_against_the_working_directory(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    sub = tmp_path / "repo" / "src"
    sub.mkdir()
    git = FakeGit(common="../.git", remote=None)
    expected = hashlib.sha256(
        os.path.realpath(tmp_path / "repo").encode()).hexdigest()[:16]
    assert canonical_project(str(sub), run=git) == f"path:{expected}"


def test_without_a_remote_a_worktree_and_its_main_checkout_share_the_path_name(tmp_path):
    """The path form hashes the main working tree, which is the directory holding the
    common `.git`, never the worktree's own directory. Hashing the working directory
    would give every worktree its own project."""
    common = tmp_path / "main" / ".git"
    common.mkdir(parents=True)
    main = canonical_project(str(tmp_path / "main"), run=FakeGit(str(common), None))
    worktree = canonical_project(str(tmp_path / "elsewhere"),
                                 run=FakeGit(str(common), None))
    assert main == worktree
    assert main.startswith("path:") and len(main) == len("path:") + 16
    check_project(main)


def test_a_bare_repository_is_named_by_its_own_directory(tmp_path):
    bare = tmp_path / "repo.git"
    bare.mkdir()
    name = canonical_project(str(bare), run=FakeGit(str(bare), None))
    expected = hashlib.sha256(os.path.realpath(bare).encode()).hexdigest()[:16]
    assert name == f"path:{expected}"


@pytest.mark.parametrize("remote", ["/srv/git/app.git", "https://git.lan/team/../app"])
def test_a_remote_that_cannot_be_a_project_name_falls_back_to_the_path(tmp_path, remote):
    """A local remote names no shared repository, and a remote whose decoded name the
    hosted deployment would refuse must not become a header it answers with a 400."""
    common = tmp_path / ".git"
    common.mkdir()
    assert canonical_project(str(tmp_path), run=FakeGit(str(common), remote)) \
        .startswith("path:")


# -- the default runner ------------------------------------------------------------

def test_the_runner_reports_a_missing_directory_as_no_answer(tmp_path):
    assert project_module._run_git(["rev-parse", "--git-common-dir"],
                                   str(tmp_path / "absent")) is None


def test_the_runner_reports_a_missing_git_binary_as_no_answer(monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(project_module.subprocess, "run", missing)
    assert project_module._run_git(["--version"], str(tmp_path)) is None


def test_the_runner_gives_up_on_a_git_that_hangs(monkeypatch, tmp_path):
    """A server should start without a project rather than wait on a hung filesystem."""
    def hangs(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=kwargs["timeout"])
    monkeypatch.setattr(project_module.subprocess, "run", hangs)
    assert project_module._run_git(["--version"], str(tmp_path)) is None


def test_the_runner_reports_a_failed_git_as_no_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(
        project_module.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 128, stdout=b"noise\n", stderr=b""))
    assert project_module._run_git(["--version"], str(tmp_path)) is None


# -- the real git binary -----------------------------------------------------------

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


@needs_git
def test_two_worktrees_of_one_real_repository_resolve_to_one_project(tmp_path):
    """The acceptance test from the design, against what git really prints. A linked
    worktree prints an absolute common directory and the main checkout a relative one, so
    this is the case the join in `canonical_project` exists for."""
    main = tmp_path / "main"
    main.mkdir()
    _git("init", "-q", cwd=main)
    _git("commit", "-q", "--allow-empty", "-m", "first", cwd=main)
    _git("worktree", "add", "-q", str(tmp_path / "wt"), cwd=main)
    (main / "sub").mkdir()

    unpushed = {canonical_project(str(p)) for p in (main, main / "sub", tmp_path / "wt")}
    assert len(unpushed) == 1 and unpushed.pop().startswith("path:")

    _git("remote", "add", "origin", "git@github.com:Acme/App.git", cwd=main)
    pushed = {canonical_project(str(p)) for p in (main, main / "sub", tmp_path / "wt")}
    assert pushed == {"github.com/acme/app"}


@needs_git
def test_a_real_directory_outside_any_repository_has_no_project(tmp_path, monkeypatch):
    # GIT_CEILING_DIRECTORIES stops git walking up out of tmp_path into a repository that
    # happens to contain the temporary directory.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert canonical_project(str(tmp_path)) is None


# -- the server configuration: switches and the project ------------------------------

class Derive:
    """Stands in for `canonical_project` inside `config.py` and records the directory."""

    def __init__(self, answer):
        self.answer, self.asked = answer, []

    def __call__(self, cwd):
        self.asked.append(cwd)
        return self.answer


@pytest.fixture()
def derive(monkeypatch):
    # Replaces the stub `tests/conftest.py` installs, with one that records its calls.
    fake = Derive("github.com/acme/app")
    monkeypatch.setattr(config_module, "canonical_project", fake)
    return fake


LOCAL = {"MEMVARA_DB": ":memory:"}


@pytest.mark.derives_project
def test_every_feature_is_on_unless_switched_off(derive):
    assert ServerConfig.from_env(LOCAL).features_off == frozenset()


@pytest.mark.parametrize("value, off", [("0", True), ("off", True), ("1", False),
                                        ("", False), ("  ", False)])
@pytest.mark.derives_project
def test_a_feature_variable_switches_its_feature(derive, value, off):
    config = ServerConfig.from_env({**LOCAL, "MEMVARA_FEATURE_PROFILE": value})
    assert config.features_off == (frozenset({"profile"}) if off else frozenset())


@pytest.mark.derives_project
def test_a_misspelt_feature_is_refused_with_the_name_it_probably_meant(derive):
    """Ignoring `MEMVARA_FEATURE_PROFLE=0` would leave the feature on while the operator
    believes it is off, which is the failure a refusal at startup prevents."""
    with pytest.raises(ConfigError, match=r"MEMVARA_FEATURE_PROFLE does not name a "
                                          r"feature: 'profle' \(did you mean 'profile'\?\)"):
        ServerConfig.from_env({**LOCAL, "MEMVARA_FEATURE_PROFLE": "0"})
    with pytest.raises(ConfigError, match="MEMVARA_FEATURE_LINKS='maybe' is not a boolean"):
        ServerConfig.from_env({**LOCAL, "MEMVARA_FEATURE_LINKS": "maybe"})


def test_the_server_and_the_environment_refuse_an_unknown_feature_the_same_way():
    """One check, so the two doors cannot come to disagree about what a feature is."""
    with pytest.raises(ValueError, match=r"'profle' \(did you mean 'profile'\?\)"):
        MemvaraMCPServer(Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM()),
                         features_off={"profle"})
    assert set(FEATURES) >= {"profile", "project_scope"}


def test_without_opting_in_a_test_never_derives_a_project():
    """`tests/conftest.py` switches the derivation off for every test that does not ask
    for it, so no test depends on which checkout it happens to run in."""
    assert ServerConfig.from_env(LOCAL).project is None


@needs_git
@pytest.mark.derives_project
def test_an_opted_in_test_derives_the_project_from_a_real_repository(tmp_path):
    _git("init", "-q", cwd=tmp_path)
    _git("remote", "add", "origin", "git@github.com:Acme/App.git", cwd=tmp_path)
    assert ServerConfig.from_env(LOCAL, cwd=str(tmp_path)).project == "github.com/acme/app"


@pytest.mark.derives_project
def test_the_project_is_derived_from_the_working_directory_by_default(derive, tmp_path):
    assert ServerConfig.from_env(LOCAL, cwd=str(tmp_path)).project == "github.com/acme/app"
    assert derive.asked == [str(tmp_path)]
    ServerConfig.from_env(LOCAL)
    assert derive.asked[-1] == os.getcwd()


@pytest.mark.derives_project
def test_outside_a_repository_there_is_no_project(monkeypatch):
    monkeypatch.setattr(config_module, "canonical_project", Derive(None))
    assert ServerConfig.from_env(LOCAL).project is None


@pytest.mark.derives_project
def test_switching_project_scope_off_stops_the_derivation_only(derive):
    """The switch turns off working the project out. An operator who also wrote a
    project name asked for that project, so it still applies."""
    off = {**LOCAL, "MEMVARA_FEATURE_PROJECT_SCOPE": "0"}
    assert ServerConfig.from_env(off).project is None
    assert derive.asked == []
    named = ServerConfig.from_env({**off, "MEMVARA_PROJECT": "gitlab.com/team/svc"})
    assert named.project == "gitlab.com/team/svc"


@pytest.mark.derives_project
def test_an_explicit_project_wins_and_must_be_canonical(derive):
    config = ServerConfig.from_env({**LOCAL, "MEMVARA_PROJECT": " path:0123456789abcdef "})
    assert config.project == "path:0123456789abcdef" and derive.asked == []
    assert config.scope_kwargs["project"] == "path:0123456789abcdef"
    with pytest.raises(ConfigError, match="MEMVARA_PROJECT: 'my-app' is not a project"):
        ServerConfig.from_env({**LOCAL, "MEMVARA_PROJECT": "my-app"})


@pytest.mark.derives_project
def test_a_local_server_opens_its_store_at_the_project(derive):
    memory = build_memvara(ServerConfig.from_env({**LOCAL, "MEMVARA_USER": "alice"}))
    try:
        assert memory.default_scope.project == "github.com/acme/app"
    finally:
        memory.close()


@pytest.mark.derives_project
def test_a_cloud_server_sends_the_project_as_a_header(derive):
    """The hosted deployment reads the project from `Memvara-Project` and never from a
    tool argument, so the header is the only channel and it must be on every request."""
    config = ServerConfig.from_env({"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": "k",
                                    "MEMVARA_SERVER_URL": "https://example.test"})
    memory = build_memvara(config)
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"standing": [], "recent": [], "relevant": [],
                                         "buckets": {}, "warnings": []})

    memory._http._client._transport = httpx.MockTransport(handler)
    memory.scope(user="alice").profile()
    memory.scope(project="github.com/acme/web").profile()
    assert [r.headers.get(PROJECT_HEADER) for r in seen] == \
        ["github.com/acme/app", "github.com/acme/web"]


def test_a_client_with_no_project_sends_no_header_and_writes_keep_their_key():
    mem = RemoteMemvara(api_key="k", base_url="https://example.test")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"id": "cl_1", "retired": True, "erased": False})

    mem._http._client._transport = httpx.MockTransport(handler)
    mem.delete("cl_1")
    mem.scope(project="github.com/acme/app").delete("cl_1")
    assert PROJECT_HEADER not in seen[0].headers
    assert seen[1].headers[PROJECT_HEADER] == "github.com/acme/app"
    assert all("Idempotency-Key" in r.headers for r in seen), \
        "the project header must not replace the key that makes a retried write safe"


def test_the_async_client_sends_the_project_header_too():
    mem = AsyncRemoteMemvara(api_key="k", base_url="https://example.test",
                             project="github.com/acme/app")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"memories": []})

    mem._http._client = httpx.AsyncClient(base_url="https://example.test",
                                          transport=httpx.MockTransport(handler))

    async def run():
        await mem.standing(k=1)
        await mem.scope(project="github.com/acme/web").standing(k=1)

    asyncio.run(run())
    assert [r.headers[PROJECT_HEADER] for r in seen] == \
        ["github.com/acme/app", "github.com/acme/web"]


def test_a_rendered_scope_keeps_its_project_and_an_older_one_has_none():
    base = {"tenant": "t", "user": "alice", "agent": None, "session": None}
    assert hydrate.scope(base).project is None
    assert hydrate.scope({**base, "project": "github.com/acme/app"}).project == \
        "github.com/acme/app"


# -- one project scope, end to end ----------------------------------------------------

def _server(memory, project):
    return MemvaraMCPServer(memory, user="alice", project=project)


def _tool(server, name, arguments):
    reply = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                   "params": {"name": name, "arguments": arguments}})
    return reply["result"]["content"][0]["text"]


def test_a_fact_about_one_repository_stays_there_and_a_preference_follows_the_user():
    """The acceptance test from the design, through the tools a model calls. `depends_on`
    is project-relative, so it is filed under repository A and not recalled in B.
    `prefers` is declared global, so it is written without a project and recalled in
    both."""
    memory = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice")
    a = _server(memory, "github.com/acme/app")
    b = _server(memory, "github.com/acme/web")
    _tool(a, "memory_remember", {"subject": "api", "predicate": "depends_on",
                                 "object": "postgres"})
    _tool(a, "memory_remember", {"predicate": "prefers", "object": "pytest",
                                 "memory_type": "procedural"})
    assert "postgres" in _tool(a, "memory_search", {"query": "postgres"})
    assert "postgres" not in _tool(b, "memory_search", {"query": "postgres"})
    assert "pytest" in _tool(b, "memory_standing", {})
    stats = _tool(a, "memory_stats", {})
    assert "default/alice/github.com%2Facme%2Fapp/*/*" in stats
    assert "(tenant/user/project/agent/session; '*' means unbound)" in stats
    assert "features: all on" in stats


def test_the_command_line_binds_the_project_and_the_switches(tmp_path):
    """`main()` is the path a client takes, so the project and the switches have to reach
    the server through it rather than only through `ServerConfig`."""
    lines = [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
             {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": "memory_stats", "arguments": {}}}]
    stdout = io.StringIO()
    status = main([], env={"MEMVARA_DB": str(tmp_path / "m.db"), "MEMVARA_USER": "alice",
                           "MEMVARA_PROJECT": "github.com/acme/app",
                           "MEMVARA_FEATURE_PROFILE": "0"},
                  stdin=io.StringIO("".join(json.dumps(m) + "\n" for m in lines)),
                  stdout=stdout)
    assert status == 0
    listed, stats = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert "memory_profile" not in [t["name"] for t in listed["result"]["tools"]]
    body = stats["result"]["content"][0]["text"]
    assert "github.com%2Facme%2Fapp" in body and "features switched off: profile" in body


def test_a_view_for_the_instances_own_project_needs_no_twin():
    memory = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(),
                     project="github.com/acme/app")
    assert memory.scope(user="alice").memvara is memory
    assert memory.scope(project="github.com/acme/app").memvara is memory
    other = memory.scope(project="github.com/acme/web")
    assert other.memvara is not memory and other.memvara.store is memory.store
    assert other.scope.project == "github.com/acme/web"


def test_the_async_facade_binds_a_project_the_same_way():
    memory = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice")
    amem = AsyncMemvara(memory)

    async def run():
        app = amem.scope(project="github.com/acme/app")
        await app.remember("api", "depends_on", "postgres")
        same = amem.scope(user="alice")
        return (await app.get_all(), await same.get_all(),
                await amem.scope(project="github.com/acme/web").get_all())

    in_app, unscoped, in_web = asyncio.run(run())
    assert len(in_app) == 1 and unscoped == [] and in_web == []



# -- the project is never metadata ------------------------------------------------------

def test_a_project_passed_to_remember_is_refused_rather_than_stored_as_metadata():
    """`project` is bound once. Accepted through `**meta` it would be stored as an
    annotation and the claim filed without a project, while the caller believed it had
    been filed under one."""
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM())
    with pytest.raises(TypeError, match=r"scope\(project=\.\.\.\)"):
        mem.remember("api", "depends_on", "postgres", project="github.com/acme/app")
    assert mem.get_all() == []
    remote = RemoteMemvara(api_key="k", base_url="https://example.test")
    with pytest.raises(TypeError, match=r"scope\(project=\.\.\.\)"):
        remote.remember("api", "depends_on", "postgres", project="github.com/acme/app")
    with pytest.raises(TypeError, match=r"scope\(project=\.\.\.\)"):
        remote.supersede("cl_1", "api", "depends_on", "mysql", project="x")
    aremote = AsyncRemoteMemvara(api_key="k", base_url="https://example.test")
    with pytest.raises(TypeError, match=r"scope\(project=\.\.\.\)"):
        asyncio.run(aremote.remember("api", "depends_on", "pg", project="x"))
    with pytest.raises(TypeError, match=r"scope\(project=\.\.\.\)"):
        asyncio.run(aremote.supersede("cl_1", "api", "depends_on", "pg", project="x"))


def test_project_is_one_of_the_keys_the_engine_owns():
    from memvara.types import RESERVED_META
    assert "project" in RESERVED_META


# -- the boundary predicate ---------------------------------------------------------------

def test_a_project_scope_contains_only_its_own_project():
    """`contains` is the downward predicate `forget` and `history` use. It must not rely
    on fact keys carrying the project to keep one repository out of another."""
    from memvara.types import Scope
    app = Scope("t", "alice", project="github.com/acme/app")
    assert app.contains(Scope("t", "alice", project="github.com/acme/app", session="s"))
    assert not app.contains(Scope("t", "alice", project="github.com/acme/web"))
    assert not app.contains(Scope("t", "alice"))
    assert Scope("t", "alice").contains(app), "an unset project is a wildcard"



@needs_git
def test_a_remote_that_is_not_utf8_falls_back_to_the_path_form(tmp_path):
    """Git stores a remote as bytes. One that is not UTF-8 must not raise out of server
    startup; it is treated as no usable remote, exactly as the hooks treat it.

    The remote is written into the config file as bytes rather than passed to `git
    config` as an argument, because Windows decodes a command-line argument before git
    ever sees it, and the test would then fail in its own setup.
    """
    _git("init", "-q", cwd=tmp_path)
    with open(tmp_path / ".git" / "config", "ab") as config:
        config.write(b'[remote "origin"]\n\turl = https://example.com/\xff/repo.git\n')
    name = canonical_project(str(tmp_path))
    assert name is not None and name.startswith("path:")


def test_the_runner_reads_output_that_is_not_utf8_as_no_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(
        project_module.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=b"\xff\xfe", stderr=b""))
    assert project_module._run_git(["--version"], str(tmp_path)) is None



# -- the path form, with Windows paths on any platform ------------------------------------

@pytest.mark.parametrize("root", ["C:\\Users\\dev\\memvara", "c:/Users/dev/memvara",
                                  "C:/Users/dev/memvara/", "C:\\Users\\dev\\memvara\\"])
def test_every_spelling_of_one_windows_root_hashes_one_string(root):
    """`realpath` on Windows gives backslashes and whichever drive-letter case the
    system reports. Hashed as given, one repository would get a different name from the
    hooks, or from itself after a drive letter changed case."""
    expected = hashlib.sha256(b"c:/Users/dev/memvara").hexdigest()[:16]
    assert project_module.path_identity(root) == f"path:{expected}"


def test_the_rest_of_a_windows_path_keeps_its_case():
    """Only the drive letter folds. Folding the whole path would merge two directories on
    a case-sensitive volume."""
    assert project_module.path_identity("C:\\Src\\App") != \
        project_module.path_identity("C:\\src\\app")


def test_the_filesystem_root_is_not_emptied():
    assert project_module.path_identity("/") == \
        f"path:{hashlib.sha256(b'/').hexdigest()[:16]}"


@pytest.mark.parametrize("paths, common, root", [
    (ntpath, "C:\\src\\app\\.git", "C:\\src\\app"),
    (ntpath, "C:\\srv\\app.git", "C:\\srv\\app.git"),
    (posixpath, "/src/app/.git", "/src/app"),
    (posixpath, "/srv/app.git", "/srv/app.git"),
])
def test_the_main_working_tree_is_found_the_same_way_on_both_platforms(paths, common, root):
    """The directory holding `.git`, or a bare repository's own directory, with the
    platform's own path rules passed in so Windows is pinned on any machine."""
    assert project_module.main_root(common, paths) == root
