"""`memvara-mcp login` — the device-code flow, offline throughout.

`httpx.Client` never reaches a socket: every test monkeypatches `httpx.Client` (imported
inside `login()`, after the `cloud` extra boundary) with a fake that answers
`.post(url, json=...)` in-process, following the same seam `tests/test_llm_openai.py` uses
for `openai`. The loopback HTTP listener is real — it is a plain stdlib `HTTPServer` bound
to `127.0.0.1` on an OS-chosen port, which is not "the network" in the sense the gate cares
about (no DNS, no remote host, no sleep beyond what the polling loop itself needs) — but
`webbrowser.open` is always stubbed out so no test ever launches a real browser.

`_CREDENTIALS_PATH` is redirected to tmp_path for every test. A successful
login writes that file, and leaving it at ~/.memvara/credentials.json used
to overwrite a real 0600 API key with the fixture `key-123` whenever a test
forgot to opt in. Isolation of the network, the browser and the loopback
listener was never the missing piece.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pathlib
import subprocess
import sys

import pytest

from memvara.remote import creds as creds_module
from memvara.server import config as config_module
from memvara.server import login as login_module
from memvara.server.login import LOGIN_USAGE, login


class FakeResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self) -> Any:
        if isinstance(self._payload, str):
            raise ValueError("not JSON")
        return self._payload


class FakeClient:
    """Stands in for `httpx.Client()`: answers `.post()` from a queue of canned
    responses, keyed by path, and records every call."""

    def __init__(self, responses: dict[str, list[FakeResponse]]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict]] = []
        self.init_kwargs: dict = {}

    def post(self, url: str, json: dict) -> FakeResponse:  # noqa: A002 - matches httpx's kw
        self.calls.append((url, json))
        for path, queue in self._responses.items():
            if url.endswith(path):
                return queue.pop(0) if len(queue) > 1 else queue[0]
        raise AssertionError(f"unexpected POST {url}")

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def close(self) -> None:
        pass


def _install_fake_httpx(monkeypatch, responses: dict[str, list[FakeResponse]]) -> FakeClient:
    """`login()` does `import httpx` locally and then `httpx.Client(...)`; patching the
    module's own `Client` attribute is what a local `import httpx` sees too, since it is
    the same module object either way."""
    import httpx

    client = FakeClient(responses)

    def factory(*a, **kw):
        client.init_kwargs = kw
        return client

    monkeypatch.setattr(httpx, "Client", factory)
    return client


def _no_browser(monkeypatch) -> None:
    monkeypatch.setattr(login_module.webbrowser, "open", lambda url: False)


def _no_loopback(monkeypatch) -> None:
    """Simulate a sandboxed environment where binding 127.0.0.1 fails."""
    monkeypatch.setattr(login_module, "_bind_loopback_listener", lambda: None)


AUTH_BODY = {
    "device_code": "dc-1", "user_code": "ABCD-1234",
    "verification_uri": "https://app.memvara.dev/device",
    "verification_uri_complete": "https://app.memvara.dev/device?code=ABCD-1234",
    "expires_in": 900, "interval": 0,
}


def approved(**overrides) -> dict:
    body = {"status": "approved", "api_key": "key-123", "project": "proj",
            "privilege": "read-write"}
    body.update(overrides)
    return body


# -- argument parsing -----------------------------------------------------------------

def test_help_prints_usage_and_exits_zero():
    out = io.StringIO()
    assert login(["--help"], stdout=out) == 0
    assert out.getvalue() == LOGIN_USAGE + "\n"


def test_missing_project_is_a_usage_error():
    err = io.StringIO()
    assert login([], env={}, stderr=err) == 2
    assert "--project" in err.getvalue()


def test_unexpected_argument_is_a_usage_error():
    err = io.StringIO()
    assert login(["--bogus", "x"], env={}, stderr=err) == 2
    assert "unexpected argument" in err.getvalue()


def test_an_option_with_no_value_is_a_usage_error():
    err = io.StringIO()
    assert login(["--project"], env={}, stderr=err) == 2
    assert "needs a value" in err.getvalue()


def test_equals_form_is_accepted(monkeypatch):
    _no_browser(monkeypatch)
    _no_loopback(monkeypatch)
    client = _install_fake_httpx(monkeypatch, {
        "device/authorize": [FakeResponse(200, AUTH_BODY)],
        "device/token": [FakeResponse(200, approved())],
    })
    out, err = io.StringIO(), io.StringIO()
    status = login(["--project=proj"], env={}, stdout=out, stderr=err)
    assert status == 0
    assert client.calls[0][1]["project"] == "proj"


def test_the_httpx_client_sends_the_csrf_header_the_hosted_console_requires(monkeypatch):
    """Unauthenticated POSTs without `X-Memvara-CSRF` are 403 `csrf_failed` on
    the hosted console. Presence is the whole check when there is no session,
    which this process never has — login used to omit the header and never
    started."""
    _no_browser(monkeypatch)
    _no_loopback(monkeypatch)
    client = _install_fake_httpx(monkeypatch, {
        "device/authorize": [FakeResponse(200, AUTH_BODY)],
        "device/token": [FakeResponse(200, approved())],
    })
    assert login(["--project", "proj"], env={}, stdout=io.StringIO()) == 0
    headers = client.init_kwargs.get("headers") or {}
    assert headers.get("X-Memvara-CSRF")


# -- server URL resolution --------------------------------------------------------------

def test_server_url_falls_back_to_the_environment_then_the_default(monkeypatch):
    _no_browser(monkeypatch)
    _no_loopback(monkeypatch)
    client = _install_fake_httpx(monkeypatch, {
        "device/authorize": [FakeResponse(200, AUTH_BODY)],
        "device/token": [FakeResponse(200, approved())],
    })
    out = io.StringIO()
    status = login(["--project", "proj"],
                   env={"MEMVARA_SERVER_URL": "https://custom.example"}, stdout=out)
    assert status == 0
    assert client.calls[0][0].startswith("https://custom.example")


def test_explicit_server_flag_wins_over_the_environment(monkeypatch):
    _no_browser(monkeypatch)
    _no_loopback(monkeypatch)
    client = _install_fake_httpx(monkeypatch, {
        "device/authorize": [FakeResponse(200, AUTH_BODY)],
        "device/token": [FakeResponse(200, approved())],
    })
    status = login(["--project", "proj", "--server", "https://flagged.example"],
                   env={"MEMVARA_SERVER_URL": "https://env.example"}, stdout=io.StringIO())
    assert status == 0
    assert client.calls[0][0].startswith("https://flagged.example")


# -- authorize failures -----------------------------------------------------------------

def test_authorize_http_error_is_reported(monkeypatch):
    _no_browser(monkeypatch)
    _no_loopback(monkeypatch)

    class RaisingClient(FakeClient):
        def post(self, url, json):  # noqa: A002
            import httpx

            raise httpx.ConnectError("refused", request=None)

    import httpx

    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: RaisingClient({}))
    err = io.StringIO()
    status = login(["--project", "proj"], env={}, stdout=io.StringIO(), stderr=err)
    assert status == 1
    assert "could not reach" in err.getvalue()


def test_authorize_non_200_is_a_login_failure(monkeypatch):
    _no_browser(monkeypatch)
    _no_loopback(monkeypatch)
    _install_fake_httpx(monkeypatch, {
        "device/authorize": [FakeResponse(400, {"error": "unknown_project"})],
    })
    err = io.StringIO()
    status = login(["--project", "proj"], env={}, stdout=io.StringIO(), stderr=err)
    assert status == 1
    assert "the server refused to start a device login" in err.getvalue()
    assert "unknown_project" in err.getvalue()


def test_authorize_201_is_success(monkeypatch):
    """The hosted console answers 201 Created for a minted grant. The client
    used to treat anything other than 200 as refusal, so a successful
    authorize never reached the poll."""
    _no_browser(monkeypatch)
    _no_loopback(monkeypatch)
    _install_fake_httpx(monkeypatch, {
        "device/authorize": [FakeResponse(201, AUTH_BODY)],
        "device/token": [FakeResponse(200, approved())],
    })
    assert login(["--project", "proj"], env={}, stdout=io.StringIO()) == 0


def test_authorize_error_body_that_is_not_json_falls_back_to_text(monkeypatch):
    _no_browser(monkeypatch)
    _no_loopback(monkeypatch)
    _install_fake_httpx(monkeypatch, {
        "device/authorize": [FakeResponse(502, "upstream error")],
    })
    err = io.StringIO()
    status = login(["--project", "proj"], env={}, stdout=io.StringIO(), stderr=err)
    assert status == 1
    assert "upstream error" in err.getvalue()


def test_an_upstream_error_page_does_not_land_whole_in_the_log(monkeypatch):
    """A gateway answering with HTML is the ordinary shape of this failure.

    Unlike the tool surface this reaches a terminal rather than a model, so the risk is
    volume: a login run in CI writes stderr into a build log, and on a public repository
    that log is public. The cut keeps the part that says what went wrong, which is what
    the operator ran the command for — a short body, like the test above, is untouched.
    """
    _no_browser(monkeypatch)
    _no_loopback(monkeypatch)
    _install_fake_httpx(monkeypatch, {
        "device/authorize": [FakeResponse(502, "<html><body>" + "x" * 5000)],
    })
    err = io.StringIO()
    status = login(["--project", "proj"], env={}, stdout=io.StringIO(), stderr=err)

    assert status == 1
    body = err.getvalue()
    assert len(body) < 500, f"a 5,000-character error page reached the log: {len(body)}"
    assert "502" in body and "<html>" in body, "the diagnosis survives the cut"
    assert "…" in body, "and it says it was cut"


# -- browser handling ---------------------------------------------------------------

def test_browser_opened_successfully_prints_the_short_message(monkeypatch):
    _no_loopback(monkeypatch)
    monkeypatch.setattr(login_module.webbrowser, "open", lambda url: True)
    _install_fake_httpx(monkeypatch, {
        "device/authorize": [FakeResponse(200, AUTH_BODY)],
        "device/token": [FakeResponse(200, approved())],
    })
    out = io.StringIO()
    assert login(["--project", "proj"], env={}, stdout=out) == 0
    assert "Opened a browser" in out.getvalue()
    assert AUTH_BODY["user_code"] in out.getvalue()


def test_browser_launch_raising_is_treated_as_not_opened(monkeypatch):
    _no_loopback(monkeypatch)

    def boom(url):
        raise RuntimeError("no display")

    monkeypatch.setattr(login_module.webbrowser, "open", boom)
    _install_fake_httpx(monkeypatch, {
        "device/authorize": [FakeResponse(200, AUTH_BODY)],
        "device/token": [FakeResponse(200, approved())],
    })
    out = io.StringIO()
    assert login(["--project", "proj"], env={}, stdout=out) == 0
    assert f"Open {AUTH_BODY['verification_uri']}" in out.getvalue()


# -- the polling loop: every RFC 8628 status -----------------------------------------

def _run(monkeypatch, poll_responses, *, no_loopback=True):
    if no_loopback:
        _no_loopback(monkeypatch)
    _no_browser(monkeypatch)
    monkeypatch.setattr(login_module.time, "sleep", lambda s: None)
    client = _install_fake_httpx(monkeypatch, {
        "device/authorize": [FakeResponse(200, AUTH_BODY)],
        "device/token": poll_responses,
    })
    out, err = io.StringIO(), io.StringIO()
    status = login(["--project", "proj"], env={}, stdout=out, stderr=err)
    return status, out.getvalue(), err.getvalue(), client


def test_approved_on_the_first_poll_writes_credentials(monkeypatch, tmp_path):
    status, out, err, client = _run(monkeypatch, [FakeResponse(200, approved())])
    assert status == 0
    path = tmp_path / "credentials.json"
    assert "Signed in" in out
    data = json.loads(path.read_text())
    assert data == {
        "api_key": "key-123", "project": "proj",
        "server_url": data["server_url"], "issued_at": data["issued_at"],
    }
    # Windows has no POSIX permission bits: `os.chmod` there can only toggle the
    # read-only attribute, so group/other bits can never be cleared and `st_mode`
    # always reads back as 0o666 regardless of what login.py asked for. The chmod
    # call is still correct there (and harmless), it is just unobservable this way.
    if sys.platform != "win32":
        assert oct(path.stat().st_mode)[-3:] == "600"


def test_a_successful_login_does_not_rewrite_the_home_credentials_file(monkeypatch,
                                                                      tmp_path):
    """`~/.memvara/credentials.json` is 0600 and holds a live API key. This
    file used to write the fixture `key-123` over it whenever a test reached
    `_write_credentials` without an opt-in redirect — which was most of them.
    """
    home_creds = Path.home() / ".memvara" / "credentials.json"
    before = home_creds.read_bytes() if home_creds.is_file() else None
    _no_browser(monkeypatch)
    _no_loopback(monkeypatch)
    _install_fake_httpx(monkeypatch, {
        "device/authorize": [FakeResponse(200, AUTH_BODY)],
        "device/token": [FakeResponse(200, approved())],
    })
    assert login(["--project", "proj"], env={}, stdout=io.StringIO()) == 0
    written = tmp_path / "credentials.json"
    assert json.loads(written.read_text())["api_key"] == "key-123"
    after = home_creds.read_bytes() if home_creds.is_file() else None
    assert after == before
    assert login_module._CREDENTIALS_PATH == written


def test_authorization_pending_is_polled_again(monkeypatch):
    status, out, err, client = _run(monkeypatch, [
        FakeResponse(400, {"error": "authorization_pending"}),
        FakeResponse(200, approved()),
    ])
    assert status == 0
    assert len(client.calls) == 3  # authorize + 2 polls


def test_slow_down_increases_the_interval_and_keeps_polling(monkeypatch):
    status, out, err, client = _run(monkeypatch, [
        FakeResponse(400, {"error": "slow_down"}),
        FakeResponse(200, approved()),
    ])
    assert status == 0


def test_access_denied_is_a_terminal_failure(monkeypatch):
    status, out, err, client = _run(monkeypatch,
                                    [FakeResponse(400, {"error": "access_denied"})])
    assert status == 1
    assert "denied in the browser" in err


def test_expired_token_is_a_terminal_failure(monkeypatch):
    status, out, err, client = _run(monkeypatch,
                                    [FakeResponse(400, {"error": "expired_token"})])
    assert status == 1
    assert "expired" in err


def test_an_unexpected_response_shape_is_reported_verbatim(monkeypatch):
    status, out, err, client = _run(monkeypatch,
                                    [FakeResponse(200, {"status": "what"})])
    assert status == 1
    assert "unexpected response" in err


def test_a_poll_response_that_is_not_json_is_a_login_failure(monkeypatch):
    """Regression: `_poll_once` raising `_LoginFailed` used to escape `_poll`'s loop
    uncaught, crashing the process instead of reporting a normal exit-1 failure."""
    status, out, err, client = _run(monkeypatch, [FakeResponse(200, "not json")])
    assert status == 1
    assert "answer a poll" in err


def test_timeout_is_reported_and_stops_polling(monkeypatch, tmp_path):
    _no_loopback(monkeypatch)
    _no_browser(monkeypatch)
    monkeypatch.setattr(login_module.time, "sleep", lambda s: None)
    short_auth = dict(AUTH_BODY, expires_in=0)
    _install_fake_httpx(monkeypatch, {
        "device/authorize": [FakeResponse(200, short_auth)],
        "device/token": [FakeResponse(200, approved())],
    })
    err = io.StringIO()
    status = login(["--project", "proj"], env={}, stdout=io.StringIO(), stderr=err)
    assert status == 1
    assert "timed out" in err.getvalue()


# -- the loopback listener ------------------------------------------------------------

def test_a_real_loopback_redirect_is_caught_as_a_hint(tmp_path):
    """The listener binds for real. A thread drives an actual HTTP GET at it while the
    main thread calls `handle_request()` — the same call `_poll` makes — and the caught
    dict records the hit. This is the unit under test in `_poll`'s redirect-hint branch;
    the outcome there always still comes from the polled `device/token` response (see the
    module docstring), never from this signal alone.
    """
    import threading

    listener = login_module._bind_loopback_listener()
    assert listener is not None
    port = listener.server_address[1]

    def hit() -> None:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/callback", timeout=5)

    thread = threading.Thread(target=hit)
    thread.start()
    listener.timeout = 5.0
    listener.handle_request()
    thread.join(timeout=5)

    assert listener._memvara_caught.get("hit") == "1"
    listener.server_close()


def test_a_redirect_hint_caught_mid_poll_skips_the_sleep(monkeypatch):
    """Drives the real listener through `login()` itself: a background thread hits the
    loopback callback while `_authorize` is in flight, so by the time `_poll`'s first
    `listener.handle_request()` runs, the hit is already queued and `redirect_hint_used`
    flips to `True` in the same iteration that reads it.
    """
    import threading

    _no_browser(monkeypatch)
    monkeypatch.setattr(login_module.time, "sleep", lambda s: None)

    class RedirectingClient(FakeClient):
        def post(self, url, json):  # noqa: A002
            self.calls.append((url, json))
            if url.endswith("device/authorize"):
                port = int(json["redirect_uri"].rsplit(":", 1)[1].split("/")[0])
                threading.Thread(
                    target=lambda: urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/callback", timeout=5)).start()
                return FakeResponse(200, AUTH_BODY)
            return FakeResponse(200, approved())

    import httpx

    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: RedirectingClient({}))
    out = io.StringIO()
    status = login(["--project", "proj"], env={}, stdout=out)
    assert status == 0


def test_login_completes_with_a_real_loopback_listener_bound(monkeypatch):
    """End to end with the listener genuinely bound (no redirect ever sent to it) — the
    login still finishes from the poll alone, and the listener is closed afterwards."""
    _no_browser(monkeypatch)
    monkeypatch.setattr(login_module.time, "sleep", lambda s: None)
    _install_fake_httpx(monkeypatch, {
        "device/authorize": [FakeResponse(200, AUTH_BODY)],
        "device/token": [FakeResponse(200, approved())],
    })
    out = io.StringIO()
    status = login(["--project", "proj"], env={}, stdout=out)
    assert status == 0


def test_bind_loopback_listener_returns_none_when_binding_fails(monkeypatch):
    def raise_oserror(*a, **kw):
        raise OSError("address in use")

    monkeypatch.setattr(login_module, "HTTPServer", raise_oserror)
    assert login_module._bind_loopback_listener() is None


def test_no_test_in_this_repository_can_reach_the_real_credentials_file():
    """The redirect is on by default for every test, not opt-in per file.

    This is the guard on the fixture rather than on any one caller. `login.py` writes
    `_CREDENTIALS_PATH` on success, and for a long time only this file redirected it --
    so the hole was never a test that forgot, it was the next file that never knew. That
    happened three times, and each time a real 0600 key was replaced by `key-123`. The
    API returns a key exactly once; there was nothing to restore from.

    Asserted from inside an ordinary test, with no fixture requested by name, because
    that is the state every future test starts in.
    """
    # From `conftest`, captured before anything was redirected. Computing
    # `pathlib.Path.home()` here would read the *patched* HOME, so `real` would be the
    # tmp path and every assertion below would compare two disposable paths and pass
    # whatever the fixture did. That is exactly what happened when this was first
    # written: removing the redirect left it green.
    from conftest import REAL_CONFIG_CREDENTIALS_PATH as real

    for name, value in (("login._CREDENTIALS_PATH", login_module._CREDENTIALS_PATH),
                        ("config.CREDENTIALS_PATH", config_module.CREDENTIALS_PATH)):
        assert value != real, (
            f"{name} still points at the developer's own credentials ({real}); a test "
            "that writes it destroys a key that cannot be recovered")
        assert real.parent not in value.parents and value.parent != real.parent, (
            f"{name} is beside the developer's own credentials ({value}); a different "
            "name in the same directory is not isolation")


def test_the_read_side_is_redirected_too_not_only_the_write():
    """`remote/creds.py` from-imports the constant, so patching `config` misses it.

    A from-import binds the value at the importing module's import time. Redirecting
    `config.CREDENTIALS_PATH` therefore leaves `creds.CREDENTIALS_PATH` pointing at the
    developer's own file, and `creds._from_file()` reads it. Nothing is destroyed -- it is
    a read -- and that is what makes it worth a test: the failure is a suite that passes
    because whoever ran it happened to be logged in, and fails on a machine that is not.
    """
    from conftest import REAL_CONFIG_CREDENTIALS_PATH as real

    assert creds_module.CREDENTIALS_PATH != real, (
        f"creds.CREDENTIALS_PATH still reads {real}; a from-import binds the value at "
        "its own import time, so patching config does not reach it")


def test_a_child_process_inherits_the_redirect_through_HOME():
    """The monkeypatch is in-process; a subprocess re-imports and computes `Path.home()`.

    Six test files here spawn subprocesses. `Path.home()` reads HOME on POSIX, so setting
    it is what makes the redirect survive the fork -- without it a child that reaches
    `_write_credentials` writes the real file exactly as before, and only the session-end
    snapshot notices, after the key is gone.
    """
    written = subprocess.run(
        [sys.executable, "-c",
         "import pathlib; print(pathlib.Path.home())"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert "pytest-of" in written, (
        f"a child process resolved home to {written}; HOME is not redirected, so anything "
        "it writes lands in the developer's real home")
