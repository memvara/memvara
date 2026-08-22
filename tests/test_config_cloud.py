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
from memvara.server.config import (CREDENTIALS_PATH, ConfigError, ServerConfig,
                                   _ENGINE_NEEDS, build_memvara)
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

def test_build_memvara_refuses_cloud_mode_rather_than_starting_a_server_that_cannot_work():
    """It used to construct one, and that was the defect.

    `Memvara(store=RemoteStore(...))` builds fine: `RemoteStore.__init__` only needs a
    URL and a key. The engine then calls `put_claim`, `lexical_search` and
    `competing_claims` on every turn and the REST facade has an endpoint for none of
    them, so the server started, advertised twelve tools, and raised
    `NotImplementedError` on the first one a model reached for. A failure that arrives
    mid-conversation as a tool error is the worst place for it: the model cannot act on
    it, and whoever configured the deployment is not in the room.

    Refusing here puts the failure where the configuration was made, with the flag that
    fixes it. It is a **decision** rather than a stopgap — `docs/OPEN-CORE.md` records
    which side of the line each seam is on — and it un-refuses itself: the check is
    `_ENGINE_NEEDS - RemoteStore.WIRED`, so the day those endpoints exist and `WIRED`
    grows, this branch stops firing on its own.
    """
    config = ServerConfig.from_env({"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": "k",
                                    "MEMVARA_TENANT": "acme", "MEMVARA_USER": "alice"})
    with pytest.raises(ConfigError) as exc:
        build_memvara(config)
    message = str(exc.value)
    assert "put_claim" in message and "lexical_search" in message, (
        "the message has to name what is missing, or it is untraceable"
    )
    assert "MEMVARA_MODE=local" in message, "and what to do instead"


def test_the_cloud_guard_is_derived_from_the_store_rather_than_hardcoded():
    """The guard must not outlive the gap it names.

    A literal `raise` here would keep refusing after the endpoints landed, and the person
    who added them would have no reason to look in this file.
    """
    assert _ENGINE_NEEDS - RemoteStore.WIRED, (
        "RemoteStore now wires everything the engine needs — delete the guard in "
        "build_memvara and restore the cloud-mode construction tests above it"
    )


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
    # Cloud mode refuses before it reaches the LLM branch, so the backend choice is
    # asserted on the local path instead — same `_anthropic()` call, same line of code.
    local = ServerConfig.from_env({"MEMVARA_DB": ":memory:", "MEMVARA_LLM": "anthropic"})
    assert config.llm == "anthropic"
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
