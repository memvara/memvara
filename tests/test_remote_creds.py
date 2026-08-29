"""Where an api key and a base url come from, in what order.

The order matters more than it looks. An explicit argument beating the environment is
ordinary; the credentials file coming last is what makes `memvara-mcp login` usable
without making it authoritative over a key the caller passed in this process.
"""
import json

import pytest

from memvara.remote.creds import MissingCredential, resolve


def test_an_explicit_key_wins_over_everything():
    key, url = resolve("explicit", None, env={"MEMVARA_API_KEY": "from-env"})
    assert key == "explicit"


def test_the_environment_supplies_the_key_when_the_caller_did_not():
    key, _ = resolve(None, "https://example.test", env={"MEMVARA_API_KEY": "from-env"})
    assert key == "from-env"


def test_the_credentials_file_is_the_last_resort(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"api_key": "from-file"}))
    monkeypatch.setattr("memvara.remote.creds.CREDENTIALS_PATH", path)
    key, _ = resolve(None, "https://example.test", env={})
    assert key == "from-file"


def test_no_key_anywhere_names_the_command_that_writes_one(tmp_path, monkeypatch):
    monkeypatch.setattr("memvara.remote.creds.CREDENTIALS_PATH", tmp_path / "absent.json")
    with pytest.raises(MissingCredential) as caught:
        resolve(None, "https://example.test", env={})
    assert "memvara-mcp login" in str(caught.value)


def test_the_base_url_defaults_to_the_hosted_deployment():
    _, url = resolve("k", None, env={})
    assert url == "https://app.memvara.dev"


def test_the_environment_can_point_at_another_deployment():
    _, url = resolve("k", None, env={"MEMVARA_SERVER_URL": "https://self.hosted.test"})
    assert url == "https://self.hosted.test"


def test_a_trailing_slash_is_removed_so_paths_do_not_double_up():
    _, url = resolve("k", "https://example.test/", env={})
    assert url == "https://example.test"


def test_a_blank_environment_value_is_treated_as_unset_rather_than_as_a_url():
    _, url = resolve("k", None, env={"MEMVARA_SERVER_URL": "   "})
    assert url == "https://app.memvara.dev"
