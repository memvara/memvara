"""How many recall requests one prompt sends to a hosted server, and what the banner says.

Each test runs the real recall hook in a child process against a small fake MCP server on
127.0.0.1, and counts the `memory_recall` calls the server receives. The hosted service
counts every `memory_recall` it answers with HTTP 200 against the plan's recall allowance,
including a call it refused because of an argument. So the number of calls is the number
that matters to a customer, and it is what these tests assert.

The child process gets a temporary home directory, credentials from `MEMVARA_API_KEY` and
`MEMVARA_SERVER_URL`, and `MEMVARA_DAEMON=1` so that it never starts a background daemon.
Nothing here reads or writes the real `~/.memvara` or `~/.claude`.

The fake server answers the way `app.memvara.dev` answers, as read from memvara-cloud's
`rest/limits.py` and `rest/mcp.py`:

* A spent daily allowance on a paid plan is HTTP 429 with code `rate_limited`, a
  `Retry-After` header in seconds, and a `detail` that names the metric, the limit, the
  count used, `resets_at` and `reason: over_period_allowance`.
* A spent monthly allowance on Free is HTTP 402 with code `quota_exhausted` and
  `detail.resets_at`, and no `Retry-After`.
* A plain rate limit is HTTP 429 with code `rate_limited` and a `detail` that names the
  rule (`credential`, `project` or `connection`), not a metric.
* An argument the tool does not declare is refused by the tool's validator. That refusal is
  a tool result with `isError: true` inside an HTTP 200, so it is counted.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
RECALL = REPO / "plugin" / "hooks" / "recall.py"

#: The arguments `memory_recall` declares on app.memvara.dev today, read from its
#: `tools/list` on 2026-09-24. `query_rewrite` is not among them.
LIVE = ("query", "k", "budget", "min_score", "include_episodes", "memory_types",
        "anchored", "ranked", "valid_at", "as_of")

#: What the fake server returns for a recall that finds something. Only the wider recall
#: (the one with `include_episodes`) finds anything, so the first recall comes back empty
#: and the hook always goes on to the wider one. That is the case the second half of these
#: tests is about.
FOUND = "Known about the user:\n- billing uses postgres"


class Fake:
    """The state behind one fake server: what it declares, how it answers, what it saw."""

    def __init__(self, accepts=LIVE, *, refuse=None, probe_fails=False) -> None:
        self.accepts = set(accepts)
        #: `None` to answer normally, or `(status, body, headers)` to refuse every recall.
        self.refuse = refuse
        #: When true, `tools/list` answers HTTP 500, so the client cannot learn the schema.
        self.probe_fails = probe_fails
        self.recalls: list[dict] = []
        self.lock = threading.Lock()

    def tools(self) -> list:
        return [
            {"name": "memory_recall",
             "inputSchema": {"type": "object",
                             "properties": {name: {} for name in sorted(self.accepts)}}},
            # Listed so that the standing-preferences refresh uses this tool and does not
            # fall back to a `memory_recall` of its own, which would muddle the count.
            {"name": "memory_standing",
             "inputSchema": {"type": "object", "properties": {"k": {}}}},
        ]

    def answer(self, message: dict) -> "tuple[int, dict | None, dict]":
        method = message.get("method")
        ident = message.get("id")
        if method == "initialize":
            return 200, {"jsonrpc": "2.0", "id": ident, "result": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "serverInfo": {"name": "fake", "version": "0"}}}, {"mcp-session-id": "s1"}
        if method == "notifications/initialized":
            return 202, None, {}
        if method == "tools/list":
            if self.probe_fails:
                return 500, {"error": {"code": "internal", "message": "boom"}}, {}
            return 200, {"jsonrpc": "2.0", "id": ident,
                         "result": {"tools": self.tools()}}, {}
        if method != "tools/call":
            return 200, {"jsonrpc": "2.0", "id": ident,
                         "error": {"code": -32601, "message": "unknown method"}}, {}
        params = message.get("params") or {}
        name = params.get("name")
        arguments = dict(params.get("arguments") or {})
        if name != "memory_recall":
            return 200, _result(ident, ""), {}
        with self.lock:
            self.recalls.append(arguments)
        if self.refuse is not None:
            status, body, headers = self.refuse
            return status, body, headers
        unknown = sorted(set(arguments) - self.accepts)
        if unknown:
            # The shape of `memvara.server.validate`'s refusal: a tool result, not an
            # HTTP error, which is why the hosted service counts it.
            text = (f"memory_recall: unknown argument(s) {', '.join(map(repr, unknown))}. "
                    f"Accepted: {', '.join(sorted(self.accepts))}.")
            return 200, _result(ident, text, error=True), {}
        return 200, _result(ident, FOUND if arguments.get("include_episodes") else ""), {}


def _result(ident, text: str, *, error: bool = False) -> dict:
    result: dict = {"content": [{"type": "text", "text": text}]}
    if error:
        result["isError"] = True
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def _serve(fake: Fake) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - the stdlib's name
            size = int(self.headers.get("content-length") or 0)
            message = json.loads(self.rfile.read(size) or b"{}")
            status, body, headers = fake.answer(message)
            raw = b"" if body is None else json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args) -> None:  # the test output is not the place
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _run_hook(fake: Fake, tmp_path: pathlib.Path) -> dict:
    """Run the recall hook once, as a harness would, and return what it printed."""
    server = _serve(fake)
    try:
        home = tmp_path / "home"
        home.mkdir()
        work = tmp_path / "work"
        work.mkdir()
        # The parent's environment, not a minimal one: on Windows a child without
        # SYSTEMROOT cannot open a socket, so the hook reached no server at all. Every
        # MEMVARA_ variable is removed, so nothing from the machine running the suite
        # decides what the hook reads, and both home variables point at the temporary
        # directory (`os.path.expanduser` reads USERPROFILE on Windows).
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("MEMVARA_")}
        env.update({
            "HOME": str(home),
            "USERPROFILE": str(home),
            "PYTHONPATH": str(REPO),
            "MEMVARA_API_KEY": "mv_test",
            "MEMVARA_SERVER_URL": f"http://127.0.0.1:{server.server_address[1]}",
            # Never start a background daemon: it would outlive the test and send a
            # warm-up recall of its own.
            "MEMVARA_DAEMON": "1",
        })
        event = {"session_id": "sess-requests", "prompt": "which database does billing use",
                 "cwd": str(work), "hook_event_name": "UserPromptSubmit"}
        done = subprocess.run([sys.executable, str(RECALL)], input=json.dumps(event),
                              capture_output=True, text=True, env=env, cwd=str(work),
                              timeout=60)
        assert done.returncode == 0, done.stderr
        out = done.stdout.strip()
        return json.loads(out) if out else {}
    finally:
        server.shutdown()
        server.server_close()


def _first_and_wider(fake: Fake) -> "tuple[list[dict], list[dict]]":
    """The recall calls the server saw, split into the first recall and the wider one.

    Told apart by `budget`: the hook asks for 300 on the first recall and 600 on the wider
    one (`recall.BUDGET` and `recall.EPISODE_BUDGET`).
    """
    first = [a for a in fake.recalls if a.get("budget") == 300]
    wider = [a for a in fake.recalls if a.get("budget") == 600]
    assert len(first) + len(wider) == len(fake.recalls), fake.recalls
    return first, wider


# -- one request per recall -----------------------------------------------------------


def test_a_server_like_todays_gets_one_request_per_recall(tmp_path):
    fake = Fake()
    reply = _run_hook(fake, tmp_path)
    [first], [wider] = _first_and_wider(fake)
    assert "include_episodes" not in first and first["min_score"] == pytest.approx(0.29)
    assert wider["include_episodes"] is True and wider["min_score"] == pytest.approx(0.29)
    # `query_rewrite` is not in this server's schema, so it is never sent.
    assert all("query_rewrite" not in call for call in fake.recalls)
    assert "1 memory recalled" in reply["systemMessage"]


def test_a_server_without_include_episodes_gets_one_request_for_the_wider_recall(tmp_path):
    """The wider recall used to be sent three times to a server like this.

    First with `min_score` and `include_episodes`, refused; again without `min_score`,
    refused, and logged as "hosted rejected min_score" although the server had not; and
    a third time without either. All three are HTTP 200, so all three were counted.
    """
    fake = Fake(accepts=set(LIVE) - {"include_episodes"})
    _run_hook(fake, tmp_path)
    first, wider = _first_and_wider(fake)
    assert (len(first), len(wider)) == (1, 1), fake.recalls
    assert all("include_episodes" not in call for call in fake.recalls)
    # The floor is kept: the server takes it, and nothing it refused was about it.
    assert all(call.get("min_score") == pytest.approx(0.29) for call in fake.recalls)


def test_a_server_without_min_score_gets_one_request_per_recall(tmp_path):
    fake = Fake(accepts=set(LIVE) - {"min_score"})
    _run_hook(fake, tmp_path)
    first, wider = _first_and_wider(fake)
    assert (len(first), len(wider)) == (1, 1), fake.recalls
    assert all("min_score" not in call for call in fake.recalls)
    log = (tmp_path / "home" / ".memvara" / ".hooks" / "recall.log").read_text()
    assert "UNFILTERED" in log, "a recall without its floor must still say so in the log"


def test_a_failed_probe_falls_back_to_dropping_what_the_server_refused(tmp_path):
    """When `tools/list` fails the client cannot know the schema, so it sends the opt-out
    from query rewrite anyway and drops it when the server refuses it by name. That costs a
    second request, and only in this case."""
    fake = Fake(probe_fails=True)
    reply = _run_hook(fake, tmp_path)
    first, wider = _first_and_wider(fake)
    assert len(first) == 2 and len(wider) == 2, fake.recalls
    assert "query_rewrite" in first[0] and "query_rewrite" not in first[1]
    assert first[1]["min_score"] == pytest.approx(0.29), (
        "the refusal named query_rewrite, so the floor must not be dropped with it")
    assert "1 memory recalled" in reply["systemMessage"]


# -- what a refused recall says -------------------------------------------------------


DAILY_SPENT = (
    429,
    {"error": {
        "code": "rate_limited",
        "message": "the 'retrieval.query' allowance is momentarily exhausted "
                   "(700 of 700 used). Retry in 12600s.",
        "detail": {"metric": "retrieval.query", "limit": 700, "used": 700,
                   "resets_at": "2026-09-25T00:00:00+00:00",
                   "reason": "over_period_allowance"}}},
    {"Retry-After": "12600"},
)


def test_a_spent_daily_allowance_says_so_and_when_it_resets(tmp_path):
    fake = Fake(refuse=DAILY_SPENT)
    reply = _run_hook(fake, tmp_path)
    assert len(fake.recalls) == 1, (
        f"a refusal about the allowance is not about an argument, and must not be "
        f"resent without one: {fake.recalls}")
    message = reply["systemMessage"]
    assert "recall failed" not in message
    assert "today's recall allowance is used up" in message
    assert "resets in 3 h 30 min" in message


def test_a_spent_daily_allowance_without_retry_after_names_the_reset_time(tmp_path):
    status, body, _ = DAILY_SPENT
    fake = Fake(refuse=(status, body, {}))
    reply = _run_hook(fake, tmp_path)
    assert "today's recall allowance is used up" in reply["systemMessage"]
    assert "resets at 00:00 UTC" in reply["systemMessage"]


def test_a_spent_monthly_allowance_keeps_its_own_message_and_is_sent_once(tmp_path):
    fake = Fake(refuse=(402, {"error": {
        "code": "quota_exhausted",
        "message": "the 'retrieval.query' allowance for this project is spent",
        "detail": {"metric": "retrieval.query", "limit": 2000, "used": 2000,
                   "resets_at": "2026-10-01T00:00:00+00:00",
                   "reason": "over_plan_allowance"}}}, {}))
    reply = _run_hook(fake, tmp_path)
    assert len(fake.recalls) == 1, fake.recalls
    assert "retrieval quota spent — resets 1 Oct" in reply["systemMessage"]


def test_a_plain_rate_limit_is_still_reported_as_a_failure(tmp_path):
    """Too many requests too fast is not an allowance being used up. It clears in
    seconds, and the banner must not tell the reader to wait until tomorrow."""
    fake = Fake(refuse=(429, {"error": {
        "code": "rate_limited",
        "message": "this request costs 65 units and the credential rate limit ...",
        "detail": {"rule": "credential", "limit": 1200, "remaining": 0, "cost": 65,
                   "retry_after": 3, "unit": "units of server work"}}},
        {"Retry-After": "3"}))
    reply = _run_hook(fake, tmp_path)
    assert len(fake.recalls) == 1, fake.recalls
    assert "recall failed" in reply["systemMessage"]
    assert "allowance" not in reply["systemMessage"]


def test_a_server_error_is_still_reported_as_a_failure(tmp_path):
    fake = Fake(refuse=(500, {"error": {"code": "internal", "message": "boom"}}, {}))
    reply = _run_hook(fake, tmp_path)
    assert len(fake.recalls) == 1, fake.recalls
    assert "recall failed" in reply["systemMessage"]
