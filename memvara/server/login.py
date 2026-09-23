"""`memvara login` — trade a device code for an API key, without a password to type.

Two console scripts reach this one function: `memvara login`, which is the command a
person runs, and `memvara-mcp login`, which is the same flow under the older name and is
kept working. `prog` below is which of the two was typed, and every message uses it.

This is the client half of the contract `config.py`'s module docstring names on the other
side: `POST /api/auth/device/authorize` mints a `device_code` (a secret, kept only here and
on the server's digest) and a `user_code` (short, meant for a human to read or click), a
browser tab lets a signed-in person approve or deny the sign-in, and this process polls
`POST /api/auth/device/token` until that decision lands. RFC 8628 is the shape of all
three steps; nothing here invents a new one.

**Which project the key is for is usually not decided here.** The authorize route is
unauthenticated, so it has no session and no organization, and it therefore takes a
project *id* or no project at all rather than a name. With no project named the grant is
unbound and the person approving it in the browser chooses from the projects they hold,
which is both the smaller surface and the easier command to type.

Two ways the "click and it just works" experience degrades, on purpose rather than by
accident:

* No loopback listener (a sandboxed environment that will not let this process bind a
  port). Login still finishes — it just costs the same polling loop, without the extra nudge
  a caught redirect gives it, and the browser is opened straight at
  `verification_uri_complete` either way.
* No browser (`webbrowser.open` returns `False`, or raises, or there simply is not one).
  The `user_code` and `verification_uri` are printed so a human can type or open them
  somewhere else — the fallback RFC 8628 exists to describe.

What a caught loopback redirect is *not* trusted for: proof that a key was issued. A
redirect is one browser navigation, sent best-effort by the approval page as an additional
signal — it can be lost to a network hiccup, a proxy, a browser that blocks the request, or
a user who closes the tab before it lands. Only `POST .../device/token` answering
`{"status": "approved", ...}` is the source of truth, and it is also the one call allowed to
write the credentials file: writing it from the redirect instead would let a spoofed local
request (anything can connect to `127.0.0.1`) hand this process an API key it never asked
the server to mint.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import uuid
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

__all__ = ["LOGIN_USAGE", "login"]

#: Where the flow starts absent MEMVARA_SERVER_URL — the same default `config.py` gives
#: `ServerConfig.server_url`, restated here because this module is the one that has to
#: reach it before any `ServerConfig` exists.
_DEFAULT_SERVER_URL = "https://app.memvara.dev"

#: Where the credentials this command obtains get written. Kept equal to
#: `config.CREDENTIALS_PATH` by construction — both are `~/.memvara/credentials.json` — but
#: not imported from there, so this module (the one the `cloud` extra pulls `httpx` in for)
#: never becomes a reason `config.py` has to know about `httpx` too.
_CREDENTIALS_PATH = Path.home() / ".memvara" / "credentials.json"

#: How long this process is willing to keep polling before giving up and telling the user
#: to try again — a ceiling independent of the server's own `expires_in`, in case a clock
#: or a network stall would otherwise spin forever.
_MAX_WAIT_SECONDS = 900

#: The hosted console refuses unauthenticated POSTs that lack this header. Presence is
#: the whole check when there is no session cookie, because a cross-site HTML form
#: cannot set one; this process has no session and never will, so any value works.
#: Without it, `POST /api/auth/device/authorize` answers 403 `csrf_failed` and login
#: never starts.
_CSRF_HEADERS = {"X-Memvara-CSRF": "cli"}

LOGIN_USAGE = f"""\
memvara login — sign in to a memvara-cloud deployment and store an API key.

Opens a browser to approve this device, then polls until the key is issued. Nothing to
type unless the browser cannot be opened automatically, in which case the code to enter
is printed here.

  --project ID    which project to issue the key for, as the project's id. Optional, and
                  leaving it out is the better call: the request then names no project
                  at all and the person approving it in the browser picks from the
                  projects they hold. A project *name* is refused, here and by the
                  server, because this request has no session to resolve a name against.
                  The id may be written with or without hyphens, in either case, in
                  braces or as urn:uuid:...; it is sent as lower-case with hyphens.
  --server URL    the memvara-cloud deployment to sign in to. Default: MEMVARA_SERVER_URL
                  if this shell has one, otherwise {_DEFAULT_SERVER_URL!r}.
  --credentials PATH
                  where to write the key. Default ~/.memvara/credentials.json, which is
                  the file everything else reads. Name another one to hold a key for a
                  second project without moving every other caller into it.

On success, writes the api key to that file, mode 0600. Run this once per project per
machine; MEMVARA_MODE=cloud picks the default file up automatically after that.
"""

_OPTIONS = ("--project", "--server", "--credentials")

def _project_id(value: str) -> str | None:
    """`value` as the authorize route takes a project id, or `None` if it is not one.

    The authorize route needs an id. Anything else is a name or a slug, which that route
    refuses 400 — it is unauthenticated, so it has no organization to resolve a name
    against and will not guess across every organization on the deployment. Checked here
    so the refusal arrives before a browser opens.

    Parsed with `uuid.UUID`, so every form that parser accepts is accepted: hyphenated or
    not, either case, in braces, or as a `urn:uuid:`. The value returned is always the
    lower-case hyphenated form, which is how the console shows a project's id, so the
    route sees one spelling of one project.

    >>> _project_id("{3F1C9B2E7A414D6E9C058B2D1F4A6E70}")
    '3f1c9b2e-7a41-4d6e-9c05-8b2d1f4a6e70'
    >>> _project_id("dev") is None
    True
    """
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return None


class _Usage(Exception):
    """The command line was wrong, and the message says which part."""


class _LoginFailed(Exception):
    """The flow reached the server and the server said no — not a usage error."""


@dataclass(frozen=True, slots=True)
class _Authorization:
    """What `POST .../device/authorize` handed back — the whole shape this flow needs."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int


def _parse(argv: Sequence[str]) -> dict[str, str]:
    """`--name value` and `--name=value`, matching `init.py`'s hand-written parser."""
    options: dict[str, str] = {}
    rest = list(argv)
    while rest:
        argument = rest.pop(0)
        name, joined, inline = argument.partition("=")
        if name not in _OPTIONS:
            raise _Usage(f"unexpected argument {argument!r}")
        value = inline if joined else (rest.pop(0) if rest else "")
        if not value.strip():
            raise _Usage(f"{name} needs a value")
        options[name] = value.strip()
    return options


def _bind_loopback_listener() -> HTTPServer | None:
    """Bind 127.0.0.1 on an OS-assigned port, or give up quietly.

    Called before `authorize` so the port can be named in `redirect_uri` — the server has
    to be told before it knows there is anywhere to redirect to. `None` on any failure
    (sandboxed environment, no loopback available) rather than raising: this listener is an
    optimization, and the polling loop below finishes the login without it.
    """
    caught: dict[str, str] = {}

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # noqa: D401 — silence stdout noise
            pass

        def do_GET(self) -> None:  # noqa: N802 — http.server's own method name
            caught["hit"] = "1"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<!doctype html><title>memvara</title>"
                b"<body>Signed in. You can close this tab and return to your terminal.</body>")

    try:
        server = HTTPServer(("127.0.0.1", 0), _Handler)
    except OSError:
        return None
    server.timeout = 0.0
    server._memvara_caught = caught  # type: ignore[attr-defined]
    return server


def _authorize(client: Any, server_url: str, project: str | None,
              redirect_uri: str | None) -> _Authorization:
    # No `project` key at all when none was given, rather than a null or an empty string.
    # The route reads the field's presence as "this caller has chosen a project", and an
    # empty value is a 400 rather than the unbound grant that was meant.
    body: dict[str, str] = {} if project is None else {"project": project}
    if redirect_uri is not None:
        body["redirect_uri"] = redirect_uri
    response = client.post(f"{server_url}/api/auth/device/authorize", json=body)
    # 201 is what the hosted console actually returns (`DeviceAuthorized` is a
    # created grant). Tests historically faked 200; both are success, anything
    # else is the server refusing to start.
    if response.status_code not in (200, 201):
        raise _LoginFailed(_server_error(response, "start a device login"))
    data = response.json()
    return _Authorization(
        device_code=data["device_code"], user_code=data["user_code"],
        verification_uri=data["verification_uri"],
        verification_uri_complete=data["verification_uri_complete"],
        expires_in=int(data["expires_in"]), interval=int(data["interval"]))


#: Longest upstream body this command will echo. The same cap the Supermemory importer
#: uses on the same kind of text, for the same reason.
_BODY = 200


def _server_error(response: Any, doing: str) -> str:
    """The message a failed step prints, with the upstream body bounded.

    Unlike the tool surface, this ends up on a terminal rather than in a model's context,
    so the risk is volume rather than forged structure: a gateway that answers with an
    HTML error page, or a JSON envelope carrying an infrastructure dump, otherwise lands
    whole in whatever is reading stderr — which for a login run in CI is a build log, and
    on a public repository that log is public. Two hundred characters keeps the part that
    says what went wrong, which is what the operator ran this for.
    """
    try:
        detail = str(response.json())
    except ValueError:
        detail = str(response.text)
    if len(detail) > _BODY:
        detail = detail[:_BODY - 1] + "…"
    return f"the server refused to {doing} ({response.status_code}): {detail}"


def _poll_once(client: Any, server_url: str, device_code: str) -> dict[str, Any]:
    """One `POST .../device/token`. Returns the parsed body either way.

    RFC 8628's error shape — `{"error": "..."}"` — comes back as an ordinary 400, which is
    why this does not raise on a non-200: `authorization_pending` and `slow_down` are the
    expected shape of "not yet," not a failure this command reports.
    """
    response = client.post(f"{server_url}/api/auth/device/token",
                           json={"device_code": device_code})
    try:
        return dict(response.json())
    except ValueError as exc:
        raise _LoginFailed(_server_error(response, "answer a poll")) from exc


def _write_credentials(*, api_key: str, project: str, server_url: str,
                       path: Path | None = None) -> Path:
    """Write the key, mode 0600, to `path` or to the default credentials file.

    `path` is what `--credentials` sets, and it is the whole of how one machine holds a
    key for two projects. The default file is the one `MEMVARA_MODE=cloud`, the npm
    bridge and `Memvara.connect()` all read, so writing a second project's key there does
    not add a credential — it moves every one of those callers into that project.
    """
    path = _CREDENTIALS_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "api_key": api_key,
        "project": project,
        "server_url": server_url,
        "issued_at": datetime.now(timezone.utc).isoformat(),
    }
    # Written new and chmod'd before any content lands, rather than chmod'd after: a
    # process that dies between write and chmod would otherwise leave a plaintext key
    # world-readable for however long the gap lasted.
    path.touch(mode=0o600, exist_ok=True)
    os.chmod(path, 0o600)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def login(argv: Sequence[str], *, env: Mapping[str, str] | None = None,
         stdout: TextIO | None = None, stderr: TextIO | None = None,
         prog: str = "memvara-mcp login") -> int:
    """Run the device-code flow to completion. Returns an exit status.

    `prog` is the command the person actually typed, and every message this flow prints
    uses it. Two console scripts reach this one function — `memvara login` and
    `memvara-mcp login` — and a failure that tells somebody to re-run the other one is a
    message they have to translate before they can act on it.
    """
    env = os.environ if env is None else env
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    if "--help" in argv or "-h" in argv:
        print(LOGIN_USAGE, file=out)
        return 0

    try:
        options = _parse(argv)
        project = options.get("--project")
        if project is not None and _project_id(project) is None:
            raise _Usage(
                f"--project takes a project id (a uuid), and {project!r} is not one. "
                "Omit --project: the sign-in then names no project at all, and you pick "
                "one in the browser from the projects you hold. A project's id is in "
                "the console, on that project's own settings page.")
        if project is not None:
            project = _project_id(project)
        credentials = Path(options["--credentials"]) if "--credentials" in options \
            else None
        server_url = (options.get("--server") or env.get("MEMVARA_SERVER_URL") or "").strip() \
            or _DEFAULT_SERVER_URL
    except _Usage as exc:
        print(f"{prog}: {exc}\n\n{LOGIN_USAGE}", file=err)
        return 2

    # Imported here, not at module scope: this file belongs to the `cloud` extra and the
    # `--no extras` CI job (see CONTRIBUTING.md) never calls it, but `cli.py` still imports
    # this module's name to dispatch to it, so the import boundary has to be here rather
    # than at the top of the file.
    import httpx

    listener = _bind_loopback_listener()
    redirect_uri = None
    if listener is not None:
        port = listener.server_address[1]
        redirect_uri = f"http://127.0.0.1:{port}/callback"

    try:
        with httpx.Client(timeout=10.0, headers=_CSRF_HEADERS) as client:
            try:
                authorization = _authorize(client, server_url, project, redirect_uri)
            except httpx.HTTPError as exc:
                print(f"{prog}: could not reach {server_url}: {exc}", file=err)
                return 1
            except _LoginFailed as exc:
                print(f"{prog}: {exc}", file=err)
                return 1

            named = ("the project you choose in the browser" if project is None
                     else project)
            print(f"{prog} — {named} on {server_url}", file=out)
            print("", file=out)
            opened = False
            try:
                opened = webbrowser.open(authorization.verification_uri_complete)
            except Exception:  # noqa: BLE001 — a browser launcher can fail in any shape
                opened = False
            if opened:
                print("Opened a browser to approve this device. If nothing appeared, go "
                     f"to {authorization.verification_uri} and enter this code:", file=out)
            else:
                print(f"Open {authorization.verification_uri} and enter this code:",
                     file=out)
            print(f"\n  {authorization.user_code}\n", file=out)
            print("Waiting for approval...", file=out)
            out.flush()

            return _poll(client, out, err, server_url=server_url,
                        authorization=authorization, listener=listener,
                        credentials=credentials, prog=prog)
    finally:
        if listener is not None:
            listener.server_close()


def _poll(client: Any, out: TextIO, err: TextIO, *, server_url: str,
         authorization: _Authorization, listener: HTTPServer | None,
         credentials: Path | None = None,
         prog: str = "memvara-mcp login") -> int:
    interval = max(authorization.interval, 1)
    deadline = time.monotonic() + min(authorization.expires_in, _MAX_WAIT_SECONDS)
    redirect_hint_used = False

    while True:
        if time.monotonic() >= deadline:
            print(f"{prog}: timed out waiting for approval. Run "
                 f"\"{prog}\" again.", file=err)
            return 1

        # A caught redirect is a hint to poll right away instead of sleeping out the full
        # interval — the module docstring's reason for why it is never trusted on its own
        # applies here too: the poll below is still what decides the outcome.
        if listener is not None and not redirect_hint_used:
            listener.timeout = min(interval, max(deadline - time.monotonic(), 0))
            listener.handle_request()
            if getattr(listener, "_memvara_caught", {}).get("hit"):
                redirect_hint_used = True

        try:
            result = _poll_once(client, server_url, authorization.device_code)
        except _LoginFailed as exc:
            print(f"\n{prog}: {exc}", file=err)
            return 1
        status = result.get("status")
        error = result.get("error")

        if status == "approved":
            path = _write_credentials(api_key=result["api_key"], project=result["project"],
                                     server_url=server_url, path=credentials)
            print(f"\nSigned in to project {result['project']}. Wrote {path} "
                 f"(privilege: {result.get('privilege')}).", file=out)
            if credentials is None:
                print("Set MEMVARA_MODE=cloud (MEMVARA_DB is not needed in that mode) "
                     "and this key is picked up automatically.", file=out)
            else:
                # A key outside the default file is picked up by nothing on its own, and
                # somebody who asked for one has a caller in mind that has to be told
                # where it is. Saying so here is cheaper than the silence that follows a
                # run which quietly used the other project's key.
                print("Nothing reads this file on its own, because it is not the "
                     "default one. Point whatever needs it at this path — "
                     "\"memvara whoami --credentials PATH\" is the check that it "
                     "works.", file=out)
            return 0
        if error == "authorization_pending":
            time.sleep(0 if redirect_hint_used else interval)
            redirect_hint_used = False
            continue
        if error == "slow_down":
            interval += 5
            time.sleep(interval)
            continue
        if error == "access_denied":
            print(f"\n{prog}: denied in the browser.", file=err)
            return 1
        if error == "expired_token":
            print(f"\n{prog}: this device code expired. Run \"{prog}\" again.", file=err)
            return 1
        print(f"\n{prog}: unexpected response from the server: {result}", file=err)
        return 1
