"""`MEMVARA_MODE=cloud`: `ServerConfig.from_env`'s branch, `build_memvara`'s `RemoteStore`
branch, and `memvara-mcp login`'s dispatch from `cli.main`.

Offline throughout. `build_memvara(mode="cloud")` constructs a real `RemoteStore`, which
constructs a real `httpx.Client` — no request is ever made, so no transport needs
mocking here; the network-touching parts of the cloud path (the HTTP calls themselves)
belong to `tests/test_store_remote.py` and `tests/test_login.py`.
"""

from __future__ import annotations

import io
import json

import pytest

from memvara.server.cli import main
from memvara.server.config import CREDENTIALS_PATH, ConfigError, ServerConfig, build_memvara
from memvara.store.remote import RemoteStore


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


def test_credentials_path_constant_matches_logins_own(monkeypatch):
    """`config.py`'s module docstring says this path is kept equal to `login.py`'s by
    construction; assert it rather than only asserting it in prose."""
    from memvara.server.login import _CREDENTIALS_PATH

    assert CREDENTIALS_PATH == _CREDENTIALS_PATH


# -- build_memvara(mode="cloud") ---------------------------------------------------------

def test_build_memvara_opens_a_remote_store_in_cloud_mode():
    config = ServerConfig.from_env({"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": "k",
                                    "MEMVARA_TENANT": "acme", "MEMVARA_USER": "alice"})
    memory = build_memvara(config)
    try:
        assert isinstance(memory.store, RemoteStore)
        assert memory.store._base_url == "https://app.memvara.dev"
        assert memory.default_scope.tenant == "acme"
        assert memory.default_scope.user == "alice"
    finally:
        memory.close()


def test_build_memvara_cloud_mode_honours_a_custom_server_url():
    config = ServerConfig.from_env({"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": "k",
                                    "MEMVARA_SERVER_URL": "https://custom.example"})
    memory = build_memvara(config)
    try:
        assert memory.store._base_url == "https://custom.example"
    finally:
        memory.close()


def test_build_memvara_rejects_a_hand_built_cloud_config_with_no_api_key():
    """`ServerConfig.from_env` never produces `mode="cloud"` with `api_key=None`, but
    `ServerConfig` can be constructed directly in Python bypassing that check — mypy
    sees `api_key: str | None`, so `build_memvara` has to narrow it itself rather than
    hand `None` to `RemoteStore`."""
    config = ServerConfig(mode="cloud", api_key=None)
    with pytest.raises(ConfigError, match="no api_key"):
        build_memvara(config)


def test_build_memvara_cloud_mode_respects_the_llm_backend(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    import sys
    import types as pytypes

    monkeypatch.setitem(sys.modules, "anthropic",
                        pytypes.SimpleNamespace(Anthropic=lambda: object()))
    config = ServerConfig.from_env({"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": "k",
                                    "MEMVARA_LLM": "anthropic"})
    memory = build_memvara(config)
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
