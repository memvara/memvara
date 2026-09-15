"""`memvara` — the top-level console script: `login`, `logout`, `whoami`.

Everything here runs offline. `login` reaches the network through `httpx`, which
`tests/test_login.py` already replaces with an in-process fake, and this file uses the
same seam; `whoami` reaches it through `RemoteMemvara`, which is replaced the same way.
No test in this file writes to `~/.memvara/credentials.json`, and one of them asserts
that, because that file holds a live API key on a developer's machine and a test that
overwrote it would take somebody's hosted store away from them until they logged in
again.

The failures these tests exist to prevent:

1. **A login that writes over the credential already on the machine.** The demo in
   `demo/` needs a key for a *different* project from the one this machine already uses,
   and the two cannot share one file: whatever reads `~/.memvara/credentials.json`
   would silently start writing into the demo's project. `--credentials PATH` is what
   keeps them apart, and it is worth a test each way.
2. **A key printed.** `whoami` exists to say what a credential authorizes, so it holds
   one, and the one thing it must never do is echo it. The assertion is on the output,
   not on the code that builds it.
3. **A command that names something that does not exist.** The usage text is what a
   person retypes, so every command it names is dispatched here, and the console script
   `pyproject.toml` declares is imported and called rather than trusted.
"""

from __future__ import annotations

import builtins
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from memvara import __version__
from memvara.cli import USAGE, main
from memvara.server import login as login_module

# -- fakes ------------------------------------------------------------------------


class FakeResponse:
    """One canned `httpx` response. Mirrors `tests/test_login.py`'s own fake."""

    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self) -> Any:
        return self._payload


class FakeClient:
    def __init__(self, responses: dict[str, list[FakeResponse]]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict]] = []

    def post(self, url: str, json: dict) -> FakeResponse:  # noqa: A002 - httpx's keyword
        self.calls.append((url, json))
        for path, queue in self._responses.items():
            if url.endswith(path):
                return queue.pop(0) if len(queue) > 1 else queue[0]
        raise AssertionError(f"unexpected POST {url}")

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *exc: object) -> None:
        pass


AUTH_BODY = {
    "device_code": "dc-1", "user_code": "ABCD-1234",
    "verification_uri": "https://app.memvara.dev/device",
    "verification_uri_complete": "https://app.memvara.dev/device?code=ABCD-1234",
    "expires_in": 900, "interval": 0,
}
APPROVED = {"status": "approved", "api_key": "mv_secret-key", "project": "memvara-demo",
            "privilege": "read-write"}


def _offline_login(monkeypatch) -> FakeClient:
    """No browser, no loopback listener, no socket. Returns the client that answered."""
    import httpx

    client = FakeClient({"device/authorize": [FakeResponse(201, AUTH_BODY)],
                         "device/token": [FakeResponse(200, APPROVED)]})
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: client)
    monkeypatch.setattr(login_module.webbrowser, "open", lambda url: False)
    monkeypatch.setattr(login_module, "_bind_loopback_listener", lambda: None)
    return client


def _run(argv: list[str], **kwargs: Any) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    status = main(argv, stdout=out, stderr=err, **kwargs)
    return status, out.getvalue(), err.getvalue()


# -- the command surface ----------------------------------------------------------


def test_the_usage_names_every_command_the_script_dispatches() -> None:
    """A usage text is what a person retypes, so a command named there and not
    dispatched is a "command not found" at the one moment they were following
    instructions. Read off the dispatch table rather than written out again here, so a
    command added to one and not the other fails this."""
    from memvara.cli import COMMANDS

    for name in COMMANDS:
        assert re.search(rf"^\s*memvara {name}\b", USAGE, flags=re.MULTILINE), (
            f"{name} is dispatched and the usage does not show it")


def test_help_and_version_print_and_exit_zero() -> None:
    status, out, _ = _run(["--help"])
    assert status == 0 and "memvara login" in out
    status, out, _ = _run(["--version"])
    assert status == 0 and out.strip() == __version__


def test_no_arguments_prints_the_usage_rather_than_doing_something() -> None:
    """There is no default command. `memvara` with nothing after it is somebody finding
    out what the command does, and the worst possible answer is a device-code login."""
    status, out, err = _run([])
    assert status == 2 and "memvara login" in err and out == ""


def test_an_unknown_command_names_the_ones_that_exist() -> None:
    status, _, err = _run(["sync"])
    assert status == 2 and "sync" in err and "login" in err


def test_the_console_script_pyproject_declares_is_importable_and_callable() -> None:
    """`memvara` is generated by the installer and never run by this suite, so a typo in
    the entry point survives every test here and surfaces on a user's first command."""
    root = Path(__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    scripts = text.split("[project.scripts]")[1].split("[")[0]
    target = re.search(r"^memvara\s*=\s*\"([^\"]+)\"", scripts, flags=re.MULTILINE)
    assert target, f"pyproject declares no `memvara` script: {scripts!r}"
    module, _, attribute = target.group(1).partition(":")
    assert callable(getattr(__import__(module, fromlist=[attribute]), attribute))


def test_python_m_memvara_runs_the_same_entry_point() -> None:
    """`python3 -m memvara` is the spelling that cannot collide with the npm package's
    own `memvara` command, and the documentation offers it for exactly that reason."""
    import memvara.__main__ as module

    assert module.main is main


# -- login ------------------------------------------------------------------------


def test_login_without_a_project_lets_the_person_approving_choose_one(monkeypatch,
                                                                     tmp_path) -> None:
    """The hosted console refuses a project *name* here (400 `bad_request`) because the
    authorize route has no session to resolve a name against, and it takes a project id
    or nothing at all. Nothing is the better call: the grant then names no project, and
    the signed-in person who approves the code picks from the projects they hold. So the
    request must carry no `project` key at all rather than an empty one."""
    client = _offline_login(monkeypatch)
    path = tmp_path / "demo.json"
    status, out, err = _run(["login", "--credentials", str(path)], env={})

    assert status == 0, err
    assert "project" not in client.calls[0][1]
    assert json.loads(path.read_text())["project"] == "memvara-demo"
    assert "memvara-demo" in out


def test_login_refuses_a_project_name_here_rather_than_spending_a_request(monkeypatch,
                                                                         tmp_path):
    """`--project` takes the project's id. A name reaches the server as a 400 with a
    good message, but the message arrives after a browser has been opened, so the check
    is made here and says what to do instead."""
    _offline_login(monkeypatch)
    status, _, err = _run(["login", "--project", "memvara-demo",
                           "--credentials", str(tmp_path / "c.json")], env={})
    assert status == 2
    assert "project id" in err and "Omit --project" in err and "browser" in err


def test_login_writes_the_file_it_was_given_and_leaves_the_default_one_alone(monkeypatch,
                                                                            tmp_path):
    """The reason `--credentials` exists. `~/.memvara/credentials.json` on a developer's
    machine holds a live key for the project they work in; the demo needs a key for a
    different project, and one file cannot hold both. Writing the demo's key there would
    silently move every other caller — the MCP server in cloud mode, `bench/hosted.py`,
    the npm bridge — into the demo's project."""
    _offline_login(monkeypatch)
    home = Path.home() / ".memvara" / "credentials.json"
    before = home.read_bytes() if home.is_file() else None

    path = tmp_path / "nested" / "demo.json"
    assert _run(["login", "--credentials", str(path)], env={})[0] == 0

    assert json.loads(path.read_text())["api_key"] == "mv_secret-key"
    assert (home.read_bytes() if home.is_file() else None) == before
    if sys.platform != "win32":
        assert oct(path.stat().st_mode)[-3:] == "600"


# -- logout -----------------------------------------------------------------------


def test_logout_removes_the_named_file_and_says_the_key_still_works(tmp_path) -> None:
    """Deleting the file is not revoking the key, and a message that let somebody
    believe it was would be the one sentence in this command that matters. The key stays
    valid until it is revoked in the console, and the text says so."""
    path = tmp_path / "demo.json"
    path.write_text(json.dumps({"api_key": "mv_secret-key", "project": "memvara-demo"}))

    status, out, _ = _run(["logout", "--credentials", str(path)])

    assert status == 0 and not path.exists()
    assert "mv_secret-key" not in out
    assert "still valid" in out and "revoke" in out


def test_logout_with_nothing_to_remove_is_not_an_error(tmp_path) -> None:
    status, out, _ = _run(["logout", "--credentials", str(tmp_path / "absent.json")])
    assert status == 0 and "nothing to remove" in out


# -- whoami -----------------------------------------------------------------------


class FakeRemote:
    """Stands in for `RemoteMemvara`, recording what it was constructed with."""

    seen: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        FakeRemote.seen = dict(kwargs)

    def whoami(self) -> dict[str, Any]:
        return {"token_id": "tok_9", "granted_privilege": "read-write",
                "effective_privilege": "read-write", "read_only": False,
                "expires_at": "2027-01-01T00:00:00Z",
                "scope": {"tenant": "t-1", "user": None, "agent": None, "session": None}}

    def close(self) -> None:
        pass


def _credential(tmp_path: Path) -> Path:
    path = tmp_path / "demo.json"
    path.write_text(json.dumps({"api_key": "mv_secret-key", "project": "memvara-demo",
                                "server_url": "https://app.memvara.dev"}))
    return path


def test_whoami_reports_the_credential_without_ever_printing_it(monkeypatch, tmp_path):
    """The whole point of the command is to hold a key and describe it. `mv_secret-key`
    appearing anywhere in this output would put a live credential into a terminal
    transcript, a screenshot or a CI log."""
    monkeypatch.setattr("memvara.remote.api.RemoteMemvara", FakeRemote)
    path = _credential(tmp_path)

    status, out, err = _run(["whoami", "--credentials", str(path)], env={})

    assert status == 0, err
    assert "mv_secret-key" not in out and "mv_secret-key" not in err
    assert "memvara-demo" in out and "tok_9" in out and "read-write" in out
    assert str(path) in out, "say which credential answered, since there can be two"
    assert FakeRemote.seen["api_key"] == "mv_secret-key"
    assert FakeRemote.seen["base_url"] == "https://app.memvara.dev"


def test_whoami_falls_back_to_the_environment_when_no_file_was_named(monkeypatch):
    """`MEMVARA_API_KEY` is the credential every other part of this library reads first,
    so a `whoami` that ignored it would describe a different credential from the one a
    run is about to use."""
    monkeypatch.setattr("memvara.remote.api.RemoteMemvara", FakeRemote)
    status, out, _ = _run(["whoami"], env={"MEMVARA_API_KEY": "mv_from-env"})
    assert status == 0 and FakeRemote.seen["api_key"] == "mv_from-env"
    assert "MEMVARA_API_KEY" in out and "mv_from-env" not in out


def test_whoami_with_no_credential_at_all_names_the_command_that_makes_one(monkeypatch):
    status, _, err = _run(["whoami", "--credentials", "/nonexistent/creds.json"], env={})
    assert status == 1 and "memvara login" in err


def test_whoami_without_the_cloud_extra_names_the_extra(monkeypatch, tmp_path) -> None:
    """`httpx` is the `cloud` extra and a bare install does not have it.

    It is imported when the transport is *constructed*, not when `memvara.remote` is
    imported, which is the seam this command has to catch it at: constructing performs
    no network call, so the refusal costs nothing and arrives before anything is sent.
    The message names the install line rather than the missing module, because the
    module's own ImportError does not say which extra carries it."""
    real = builtins.__import__

    def no_httpx(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "httpx":
            raise ImportError("No module named 'httpx'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_httpx)
    status, _, err = _run(["whoami", "--credentials", str(_credential(tmp_path))],
                          env={})
    assert status == 2 and "memvara[cloud]" in err


def test_whoami_reports_a_refused_credential_rather_than_raising(monkeypatch, tmp_path):
    """An expired or revoked key is the ordinary reason to run this command, so the
    answer is a sentence, not a traceback with a bearer token in the request it echoes."""
    class Refusing(FakeRemote):
        def whoami(self) -> dict[str, Any]:
            from memvara.remote.errors import AuthError

            raise AuthError(401, "unauthorized", "this credential was refused")

    monkeypatch.setattr("memvara.remote.api.RemoteMemvara", Refusing)
    status, _, err = _run(["whoami", "--credentials", str(_credential(tmp_path))],
                          env={})
    assert status == 1 and "refused" in err and "mv_secret-key" not in err


def test_whoami_says_when_the_deployment_refuses_every_write(monkeypatch, tmp_path):
    """A read-only deployment accepts the credential and then refuses every write, which
    from the caller's side looks like a broken key. `whoami` is where that gets said."""
    class ReadOnly(FakeRemote):
        def whoami(self) -> dict[str, Any]:
            return {**super().whoami(), "read_only": True,
                    "effective_privilege": "read"}

    monkeypatch.setattr("memvara.remote.api.RemoteMemvara", ReadOnly)
    status, out, _ = _run(["whoami", "--credentials", str(_credential(tmp_path))],
                          env={})
    assert status == 0
    assert "read-only" in out and "(granted read-write)" in out


# -- the command lines, and their mistakes ------------------------------------------


@pytest.mark.parametrize("command", ["logout", "whoami"])
def test_each_command_has_its_own_help(command: str) -> None:
    status, out, _ = _run([command, "--help"], env={})
    assert status == 0 and out.startswith(f"memvara {command} —")


@pytest.mark.parametrize("argv, expected", [
    (["logout", "--bogus"], "unexpected argument '--bogus'"),
    (["whoami", "--server"], "--server needs a value"),
    (["whoami", "--credentials="], "--credentials needs a value"),
])
def test_a_malformed_option_is_a_usage_error_that_names_the_option(argv, expected):
    """Exit 2, as `memvara-mcp` uses for a command line that was wrong rather than a
    command that failed, and the message says which part was wrong."""
    status, _, err = _run(argv, env={})
    assert status == 2 and expected in err


def test_logout_accepts_the_equals_form_like_every_other_option(tmp_path) -> None:
    path = _credential(tmp_path)
    assert _run(["logout", f"--credentials={path}"])[0] == 0
    assert not path.exists()


def test_logout_reports_a_path_it_could_not_remove_rather_than_claiming_success(
        tmp_path) -> None:
    """A directory where the file should be cannot be unlinked, and "Removed" printed
    over it would tell somebody their key was gone when nothing had happened at all."""
    blocked = tmp_path / "is-a-directory"
    blocked.mkdir()
    status, out, err = _run(["logout", "--credentials", str(blocked)])
    assert status == 1 and "could not remove" in err and "Removed" not in out
    assert blocked.exists()


# -- the credentials file, read in one place --------------------------------------


def test_a_credentials_file_is_read_by_one_function_whatever_reads_it(tmp_path) -> None:
    """`whoami`, `logout` and `demo/harness.py`'s hosted arms all read a file `login`
    wrote, and three readers of one format is three chances to disagree about what an
    unreadable one means. A missing, unparsable or keyless file is "not signed in" here,
    which is what every caller does with it."""
    from memvara.remote.creds import read_credentials_file

    assert read_credentials_file(tmp_path / "absent.json") == {}
    (tmp_path / "torn.json").write_text("{not json")
    assert read_credentials_file(tmp_path / "torn.json") == {}
    (tmp_path / "keyless.json").write_text(json.dumps({"project": "p"}))
    assert read_credentials_file(tmp_path / "keyless.json") == {}
    (tmp_path / "list.json").write_text(json.dumps(["mv_secret-key"]))
    assert read_credentials_file(tmp_path / "list.json") == {}, (
        "valid JSON that is not an object is not a credential either")
    assert read_credentials_file(_credential(tmp_path)) == {
        "api_key": "mv_secret-key", "project": "memvara-demo",
        "server_url": "https://app.memvara.dev"}
