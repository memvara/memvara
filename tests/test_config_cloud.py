"""`MEMVARA_MODE=cloud`: `ServerConfig.from_env`'s branch, and `memvara-mcp login`'s
dispatch from `cli.main`.

What `build_memvara` does with a cloud config is pinned in
`tests/test_remote_cloud_mode.py`. The one property asserted here is the one this file
used to assert from the other side: cloud mode builds a client of the REST facade and
never an engine over a `RemoteStore`.

Offline throughout. `build_memvara(mode="cloud")` constructs a real `RemoteMemvara`, which
builds a real `httpx.Client` — no request is ever made, so no transport needs mocking
here; the network-touching parts of the cloud path (the HTTP calls themselves) belong to
`tests/test_remote_client.py` and `tests/test_login.py`.
"""

from __future__ import annotations

import io
import json

import pytest

from memvara.server.cli import main
from memvara.core import Memvara
from memvara.remote.api import RemoteMemvara
from memvara.server.config import (CREDENTIALS_PATH, ConfigError, ServerConfig,
                                   build_memvara)


# -- mode resolution --------------------------------------------------------------------

def test_default_mode_is_local():
    assert ServerConfig.from_env({"MEMVARA_DB": ":memory:"}).mode == "local"


def test_an_unknown_mode_is_a_startup_error():
    with pytest.raises(ConfigError, match="is not a mode"):
        ServerConfig.from_env({"MEMVARA_MODE": "hybrid", "MEMVARA_DB": ":memory:"})


def test_cloud_mode_needs_no_memvara_db():
    config = ServerConfig.from_env({"MEMVARA_MODE": "cloud",
                                    "MEMVARA_API_KEY": "k-env"})
    assert config.mode == "cloud"
    assert config.path == ""


def test_cloud_mode_reads_the_api_key_from_the_environment():
    config = ServerConfig.from_env({"MEMVARA_MODE": "cloud",
                                    "MEMVARA_API_KEY": "k-env"})
    assert config.api_key == "k-env"
    assert config.server_url == "https://app.memvara.dev"


def test_cloud_mode_prefers_the_environment_key_over_the_credentials_file(tmp_path,
                                                                          monkeypatch):
    creds = tmp_path / "credentials.json"
    creds.write_text(json.dumps({"api_key": "k-file", "server_url": "https://f.example"}))
    monkeypatch.setattr("memvara.server.config.CREDENTIALS_PATH", creds)
    config = ServerConfig.from_env({"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": "k-env"})
    assert config.api_key == "k-env"
    assert config.server_url == "https://app.memvara.dev"  # not the file's URL either


def test_cloud_mode_falls_back_to_the_credentials_file(tmp_path, monkeypatch):
    creds = tmp_path / "credentials.json"
    creds.write_text(json.dumps({"api_key": "k-file", "server_url": "https://f.example"}))
    monkeypatch.setattr("memvara.server.config.CREDENTIALS_PATH", creds)
    config = ServerConfig.from_env({"MEMVARA_MODE": "cloud"})
    assert config.api_key == "k-file"
    assert config.server_url == "https://f.example"


def test_an_explicit_server_url_env_var_overrides_the_credentials_file_url(tmp_path,
                                                                           monkeypatch):
    creds = tmp_path / "credentials.json"
    creds.write_text(json.dumps({"api_key": "k-file", "server_url": "https://f.example"}))
    monkeypatch.setattr("memvara.server.config.CREDENTIALS_PATH", creds)
    config = ServerConfig.from_env({"MEMVARA_MODE": "cloud",
                                    "MEMVARA_SERVER_URL": "https://explicit.example"})
    assert config.api_key == "k-file"
    assert config.server_url == "https://explicit.example"


def test_cloud_mode_with_no_key_anywhere_names_login(tmp_path, monkeypatch):
    creds_path = tmp_path / "nope.json"
    monkeypatch.setattr("memvara.server.config.CREDENTIALS_PATH", creds_path)
    with pytest.raises(ConfigError) as caught:
        ServerConfig.from_env({"MEMVARA_MODE": "cloud"})
    message = str(caught.value)
    assert "MEMVARA_API_KEY" in message
    assert "memvara-mcp login" in message
    assert str(creds_path) in message  # names wherever CREDENTIALS_PATH actually is


@pytest.mark.parametrize("payload", ["not json", "[]", "{}", '{"api_key": ""}',
                                     '{"api_key": 5}'])
def test_every_unusable_credentials_file_is_treated_as_not_logged_in(tmp_path, monkeypatch,
                                                                      payload):
    creds = tmp_path / "credentials.json"
    creds.write_text(payload)
    monkeypatch.setattr("memvara.server.config.CREDENTIALS_PATH", creds)
    with pytest.raises(ConfigError, match="MEMVARA_API_KEY"):
        ServerConfig.from_env({"MEMVARA_MODE": "cloud"})


def test_a_missing_credentials_file_is_treated_as_not_logged_in(tmp_path, monkeypatch):
    monkeypatch.setattr("memvara.server.config.CREDENTIALS_PATH",
                        tmp_path / "does-not-exist.json")
    with pytest.raises(ConfigError, match="MEMVARA_API_KEY"):
        ServerConfig.from_env({"MEMVARA_MODE": "cloud"})


def test_credentials_path_constant_matches_logins_own():
    """`config.py`'s module docstring says this path is kept equal to `login.py`'s by
    construction; assert it rather than only asserting it in prose.

    Read from `conftest`, which captured both before the autouse redirect ran. This used
    to compare a module-level `from ... import` -- bound at collection, so never
    redirected -- against a call-time read of the other, which is. Once every test runs
    under the redirect those two are different by construction, and the test failed while
    the invariant it names was untouched.
    """
    from conftest import REAL_CONFIG_CREDENTIALS_PATH, REAL_LOGIN_CREDENTIALS_PATH

    assert REAL_CONFIG_CREDENTIALS_PATH == REAL_LOGIN_CREDENTIALS_PATH


# -- build_memvara(mode="cloud") ---------------------------------------------------------

def test_build_memvara_cloud_mode_builds_a_client_of_the_facade_and_not_an_engine():
    """The guarantee that replaced the refusal, and it is the same guarantee.

    `MEMVARA_MODE=cloud` used to build `Memvara(store=RemoteStore(...))`. That constructs
    fine — `RemoteStore.__init__` only needs a URL and a key — and then the engine calls
    `put_claim`, `lexical_search` and `competing_claims` on every turn, for which the REST
    facade has no endpoint. The server started, advertised twelve tools, and raised
    `NotImplementedError` on the first one a model reached for: a failure arriving
    mid-conversation, where the model cannot act on it and whoever configured the
    deployment is not in the room.

    Refusing to start was the right answer to that, and this is a better one. The engine
    is still never run against a remote store — what changed is that the server is now a
    client of the facade, which is what `docs/OPEN-CORE.md` said the answer was. So the
    thing to keep asserting is not the refusal but what makes the refusal unnecessary:
    what comes back is a `RemoteMemvara`, and it is not an `Memvara` at all.
    """
    config = ServerConfig.from_env({"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": "k",
                                    "MEMVARA_TENANT": "acme", "MEMVARA_USER": "alice"})
    client = build_memvara(config)
    try:
        assert isinstance(client, RemoteMemvara)
        assert not isinstance(client, Memvara), (
            "a subclass would put the engine back over the wire by the side door"
        )
        assert client.default_scope.tenant == "acme"
        assert client.default_scope.user == "alice"
    finally:
        client.close()


def test_no_cloud_path_anywhere_constructs_an_engine_over_a_remote_store():
    """`cloud_gap()` was a set difference built to empty out when `RemoteStore.WIRED`
    grew, and that day does not arrive under this design: the `Store` seam is bypassed
    rather than completed. Deleting the guard is therefore deliberate, and this is what
    has to stay true in its place.

    Read off `config.py`'s syntax rather than its behaviour, because a reintroduced
    `Memvara(store=RemoteStore(...))` would construct, start, and fail exactly as it did
    before — there is no call that returns the wrong answer for a test to catch. Two
    things are checked: nothing here imports `RemoteStore`, and no call here passes a
    `store=`, which is the only door the engine has onto one.
    """
    import ast
    from pathlib import Path

    import memvara.server.config as config_module

    tree = ast.parse(Path(config_module.__file__).read_text())
    imported = {alias.name for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    assert "RemoteStore" not in imported, (
        "cloud mode builds a client of the facade; importing RemoteStore here means the "
        "engine is being pointed at a store that cannot serve it"
    )
    assert "RemoteMemvara" in imported

    passes_a_store = [call for call in ast.walk(tree) if isinstance(call, ast.Call)
                      for kw in call.keywords if kw.arg == "store"]
    assert not passes_a_store, (
        "`store=` is how an engine is handed a backend, and the remote one cannot serve "
        "put_claim, lexical_search or competing_claims"
    )


def test_build_memvara_rejects_a_hand_built_cloud_config_with_no_api_key():
    """`ServerConfig.from_env` never produces `mode="cloud"` with `api_key=None`, but
    `ServerConfig` can be constructed directly in Python bypassing that check — mypy
    sees `api_key: str | None`, so `build_memvara` has to narrow it itself rather than
    hand `None` to `RemoteStore`."""
    config = ServerConfig(mode="cloud", api_key=None)
    with pytest.raises(ConfigError, match="no api_key"):
        build_memvara(config)


def test_cloud_mode_refuses_a_named_llm_instead_of_loading_one(monkeypatch):
    """Extraction runs inside the deployment, so `MEMVARA_LLM=anthropic` under cloud mode
    names a subsystem this process would never use. Accepting it silently is the failure:
    the operator sets it, sees a server start, and believes their writes are being
    extracted by a model that was never loaded. The local path still builds one, which is
    the same `_anthropic()` call on the same line of code."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    import sys
    import types as pytypes

    monkeypatch.setitem(sys.modules, "anthropic",
                        pytypes.SimpleNamespace(Anthropic=lambda: object()))
    cloud = ServerConfig.from_env({"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": "k",
                                   "MEMVARA_LLM": "anthropic"})
    assert cloud.llm == "anthropic"
    with pytest.raises(ConfigError, match="MEMVARA_LLM"):
        build_memvara(cloud)

    local = ServerConfig.from_env({"MEMVARA_DB": ":memory:", "MEMVARA_LLM": "anthropic"})
    memory = build_memvara(local)
    try:
        assert memory.extractor.startswith("fast-path+anthropic/")
    finally:
        memory.close()


# -- cli.py: the login subcommand dispatch -----------------------------------------------

def test_main_dispatches_login_to_the_login_module(monkeypatch):
    """`main()` imports `login.py` lazily, only on this branch — asserted by patching the
    function it calls after import, not by any earlier import of `login`."""
    calls = {}

    def fake_login(argv, *, env, stdout, stderr):
        calls["argv"] = argv
        calls["env"] = env
        print("fake login ran", file=stdout)
        return 0

    import memvara.server.login as login_module

    monkeypatch.setattr(login_module, "login", fake_login)
    out = io.StringIO()
    status = main(["login", "--project", "proj"], env={"X": "1"}, stdout=out)
    assert status == 0
    assert calls["argv"] == ["--project", "proj"]
    assert calls["env"] == {"X": "1"}
    assert "fake login ran" in out.getvalue()


def test_main_login_dispatch_forwards_the_exit_status(monkeypatch):
    import memvara.server.login as login_module

    monkeypatch.setattr(login_module, "login", lambda argv, **kw: 2)
    assert main(["login", "--help"], env={}, stdout=io.StringIO()) == 2
