"""Hosted recall over the MCP endpoint, using nothing but the standard library.

This exists so a hosted install needs no `pip install memvara`. The plugin's install story
is "paste a URL", and a hook that silently did nothing until someone also installed a
Python package would be worse than no hook: it fails the same way a working hook over an
empty store looks.

Three things here were found by measurement rather than reasoning, and each one is a
silent failure if you skip it.

**Set a User-Agent.** Cloudflare rejects the default `Python-urllib/3.13` with error 1010
before the request reaches the application at all. Measured side by side: the stock agent
gets 403/1010, and `curl/8.7.1`, a browser string and `memvara-hook/0.1` all get through to
a genuine 401. Nothing in that 403 hints that the client's name is the problem.

**Bring a CA bundle.** python.org's macOS build does not use the system trust store, so
verification fails with CERTIFICATE_VERIFY_FAILED on a certificate every other tool on the
machine accepts. `certifi` is used when present and the default context otherwise.

**Use `http.client`, not `urllib`.** `urlopen` builds a fresh connection per call, which
throws away the TLS handshake every prompt — about 170ms of the ~390ms a cold request
costs. `HTTPSConnection` is the stdlib object that can be held open, and holding it is the
entire reason the daemon pays for itself on a hosted install: ~390ms cold against
~162-287ms warm.

**Reads raise, and that is a reversal.** They used to answer `None` on any failure, on the
argument that a prompt without a memory block beats a prompt with an error in it. That rule
is still right, but it belongs in the *hook*, not here: collapsing failure into the same
value as "nothing relevant" is what let a dead client look like an empty store for thirty
minutes at a time. See `HostedError` and `daemon.Daemon._answer` — the caller decides what a
failure costs, and it can only decide if it is told.
"""

from __future__ import annotations

import json
import os
import os.path
import ssl

from .ipc import log_line

#: Anything but the stdlib default. See the module docstring: this single header is the
#: difference between reaching the application and being refused at the edge.
USER_AGENT = "memvara-hook/0.1"

#: Written by `memvara-mcp login`.
CREDENTIALS = os.path.join(os.path.expanduser("~"), ".memvara", "credentials.json")

DEFAULT_BASE = "https://app.memvara.dev"
MCP_PATH = "/mcp"

#: Long enough for a cold TLS handshake on a slow link, short enough that a wedged
#: endpoint does not hold a prompt hostage.
TIMEOUT_SEC = 6.0

PROTOCOL_VERSION = "2025-06-18"


#: Statuses that mean "the session id you are holding is not one I know" -- a server that
#: restarted, or a session that aged out. These and only these earn a re-handshake: the
#: call is replayed once and usually succeeds. Every other non-200 is a refusal the server
#: will give again, so replaying it spends a second round trip to learn nothing.
_STALE_SESSION = frozenset((401, 404))


def _refusal(status: int, raw: bytes) -> "HostedError":
    """A `HostedError` carrying whatever the server said about why it refused.

    The API answers a refusal with `{"error": {"code": ..., "message": ..., "detail": ...}}`
    and this is the only frame that still holds it. A body that will not parse is not an
    error here -- plenty of statuses arrive with none, or with HTML from something in
    front of the API -- so the status alone is the fallback.
    """
    code, message, detail = "", "", {}
    try:
        body = json.loads(raw.decode("utf-8"))
        error = body.get("error") or {}
        code = str(error.get("code") or "")
        message = str(error.get("message") or "")
        detail = error.get("detail") or {}
    except Exception:
        pass
    if not isinstance(detail, dict):
        detail = {}
    return HostedError(message or f"the endpoint refused with HTTP {status}",
                       status=status, code=code, detail=detail)


class HostedError(RuntimeError):
    """The endpoint could not answer. Distinct from answering with nothing.

    The whole point of the class is that `except HostedError` and `if not text` are
    different questions. A caller that cannot tell them apart reports an unreachable store
    as an empty one, which is the failure this file spent thirty minutes at a time
    demonstrating.

    `code` and `detail` carry the server's own account of the refusal when it sent one.
    They were thrown away until a quota-exhausted account spent a day reporting "recall
    failed -- see capture.log": the server had said which allowance, how much of it, and
    when it resets, and every word was discarded one frame below the banner that needed
    it. `code` is the machine token (`quota_exhausted`); `detail` is the object beside it.
    Both are **empty rather than `None`** when the failure was transport-level and there
    was nothing to read -- `code` is `""` and `detail` is `{}`, so a caller tests
    truthiness and never identity. An earlier draft of this docstring said `None`, which
    would have made `if err.code is None` a branch that never runs: the same shape of
    defect this class was added to fix, in the sentence describing it.
    """

    def __init__(self, message: str, *, status: "int | None" = None,
                 code: str = "", detail: "dict | None" = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.detail = detail or {}


def credentials() -> "dict | None":
    """`{'api_key': ..., 'server_url': ...}` or None when not logged in.

    Two sources, in the library's own order: `MEMVARA_API_KEY` / `MEMVARA_SERVER_URL`
    first, then the file `memvara-mcp login` writes. Matching `memvara/remote/creds.py`
    rather than picking an order here is the point -- a machine that sets both should not
    reach a different store depending on which client happened to read it.

    Reading only the file was survivable while the library's client was the write path on
    such a machine, because it resolved the variable itself. With one client serving both
    directions this is the only place left that can, and without it an install configured
    by environment variable is simply "not logged in" to every hook: no recall, and
    `capture.py` logging `failed=no store or login` on every turn.

    Each field resolves independently, as it does there, so a key from the environment and
    a URL from the file compose rather than one shadowing the other wholesale.
    """
    data: dict = {}
    try:
        with open(CREDENTIALS, encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            data = loaded
    except (OSError, ValueError):
        pass
    api_key = (os.environ.get("MEMVARA_API_KEY") or "").strip() or data.get("api_key")
    if not api_key:
        return None
    server_url = ((os.environ.get("MEMVARA_SERVER_URL") or "").strip()
                  or data.get("server_url") or DEFAULT_BASE)
    return {"api_key": api_key, "server_url": server_url}


def _context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


class HostedRecall:
    """One kept-alive connection to the hosted MCP endpoint.

    Constructed cheaply and connected lazily: a client that dials on __init__ would pay
    the handshake even when the daemon it belongs to is never asked anything.
    """

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE) -> None:
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._conn = None
        self._session: "str | None" = None
        self._schemas: "dict[str, set[str]] | None" = None
        self._id = 0
        #: True when the last `recall()` had to drop its `min_score` because this
        #: deployment's tool surface has no such argument. Initialised here so reading it
        #: before the first call is a plain False rather than an AttributeError.
        self.unfiltered = False

    # -- transport -------------------------------------------------------------

    def _connect(self):
        import http.client
        import urllib.parse

        parts = urllib.parse.urlsplit(self._base)
        host = parts.hostname or "app.memvara.dev"
        port = parts.port
        if parts.scheme == "http":
            return http.client.HTTPConnection(host, port, timeout=TIMEOUT_SEC)
        return http.client.HTTPSConnection(host, port, timeout=TIMEOUT_SEC,
                                           context=_context())

    def _rpc(self, method: str, params: "dict | None" = None,
             retry: bool = True) -> "dict | None":
        self._id += 1
        body = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            body["params"] = params

        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            "authorization": f"Bearer {self._key}",
            "user-agent": USER_AGENT,
        }
        if self._session:
            headers["mcp-session-id"] = self._session

        try:
            if self._conn is None:
                self._conn = self._connect()
            self._conn.request("POST", MCP_PATH, json.dumps(body), headers)
            response = self._conn.getresponse()
            raw = response.read()
        except Exception:
            # A kept-alive connection the server has since closed raises on reuse. That is
            # normal and recoverable exactly once: reconnect and try again, so a daemon
            # that has idled does not answer the first prompt after a gap with silence.
            self.close()
            if retry:
                return self._rpc(method, params, retry=False)
            return None

        session = response.getheader("mcp-session-id")
        if session:
            self._session = session
        if response.status != 200:
            # A session id the server has forgotten -- it restarted, or the session aged
            # out -- refuses every subsequent call, while this client goes on sending the
            # same dead id because nothing here ever cleared it. `_ensure_session` then
            # short-circuits on the truthy value and never shakes hands again, so the
            # client stays dead for the rest of its life. Measured: a resident daemon
            # answering every prompt of a session with silence, while a fresh client on
            # the same credentials answered the same query in full.
            #
            # Drop the session and shake hands again, exactly once. The recursion
            # terminates because the retry runs with `retry=False`, and because the
            # `initialize` call inside `_ensure_session` has no session of its own to
            # invalidate.
            #
            # Only for the statuses that mean "this session is not who you think it is".
            # It used to fire on ANY non-200, which made a 402 cost two round trips per
            # prompt -- tear down a healthy session, shake hands, replay, get 402 again --
            # and four on the episode-escalation path. A refusal the server will repeat is
            # not a session problem, and retrying it is only slower.
            if response.status in _STALE_SESSION and retry and self._session:
                self._session = None
                self.close()
                if self._ensure_session():
                    return self._rpc(method, params, retry=False)
            # The body is the whole point of a refusal and this is the only frame that
            # still has it. Hand it back so `_call` can raise something a person can act
            # on rather than "no reply".
            raise _refusal(response.status, raw)
        return _decode(raw)

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    # -- the calls a hook makes ------------------------------------------------

    def _ensure_session(self) -> bool:
        if self._session:
            return True
        reply = self._rpc("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "memvara-hook", "version": "0.1"},
        })
        if reply is None:
            return False
        # Some servers issue no session id and are stateless. Treat a successful
        # initialize as sufficient rather than requiring the header.
        self._session = self._session or "stateless"
        try:
            self._rpc("notifications/initialized")
        except HostedError:
            # A notification has no reply worth having and the session is already open.
            # Failing here would throw away a handshake that succeeded.
            pass
        return True

    def accepts(self, tool: str, argument: str) -> bool:
        """Whether the server's schema for `tool` actually has `argument`.

        Asked rather than assumed, because argument validation on the other end is closed:
        an argument the server has not heard of is a hard rejection, not a silent ignore.
        A client that guesses wrong therefore loses the whole write rather than losing one
        field, which is the wrong way round for a field that only adds provenance.

        One `tools/list` per process answers it for every call afterwards, and a probe that
        fails answers False -- so an older server, or no answer at all, costs the provenance
        and keeps the fact.
        """
        if self._schemas is None:
            self._schemas = {}
            try:
                reply = self._rpc("tools/list", {}) if self._ensure_session() else None
            except HostedError:
                # Stated above: a probe that fails answers False. A refusal here costs the
                # provenance field and keeps the fact, which is the right way round -- and
                # is why this catch cannot be narrowed to the transport case.
                reply = None
            result = reply.get("result") if isinstance(reply, dict) else None
            listed = result.get("tools") if isinstance(result, dict) else None
            for entry in listed if isinstance(listed, list) else []:
                if not isinstance(entry, dict):
                    continue
                schema = entry.get("inputSchema")
                props = schema.get("properties") if isinstance(schema, dict) else None
                name = entry.get("name")
                if isinstance(name, str) and isinstance(props, dict):
                    self._schemas[name] = set(props)
        return argument in self._schemas.get(tool, set())

    def _call(self, tool: str, arguments: dict) -> str:
        """One tool call. Returns its text, or raises `HostedError`.

        The `isError` check is the one that looks redundant and is not: a tool that
        refuses answers HTTP 200 with the flag set, so a refusal arrives looking exactly
        like a success and is only distinguishable here.
        """
        if not self._ensure_session():
            raise HostedError(f"no session on the hosted endpoint for {tool}")
        reply = self._rpc("tools/call", {"name": tool, "arguments": arguments})
        if not isinstance(reply, dict):
            raise HostedError(f"no reply to {tool}")
        if reply.get("error") is not None:
            raise HostedError(str(reply["error"]))
        result = reply.get("result")
        if not isinstance(result, dict):
            raise HostedError(f"malformed reply to {tool}")
        text = _content_text(result)
        if result.get("isError"):
            raise HostedError(text or f"{tool} reported an error")
        return text

    def recall(self, query: str, *, k: int = 6, budget: int = 700,
               header: "str | None" = None,
               include_episodes: bool = False,
               memory_types: "list[str] | None" = None,
               min_score: float = 0.0) -> str:
        """Recall text, or raise `HostedError`. An empty string is a real answer.

        Empty means this store had nothing relevant, which is information; a failure means
        the question was never asked, which is not. They used to be the same value. See the
        module docstring.
        """
        if not query.strip():
            return ""
        args: dict = {"query": query, "k": k, "budget": budget}
        if min_score:
            args["min_score"] = min_score
        if include_episodes:
            args["include_episodes"] = True
        if memory_types:
            # The tool has always taken this and this client never sent it, which is why
            # the standing procedural set could not be asked for separately from everything
            # else -- and a preference that applies to every turn was competing per prompt
            # with facts that apply to one.
            args["memory_types"] = list(memory_types)
        try:
            text = self._call("memory_recall", args)
        except HostedError:
            # Optional arguments are dropped one at a time, cumulatively, in the order
            # that loses least -- the floor before the episodes, because unfiltered
            # memories beat none and a widened brief beats a narrow one.
            #
            # Written as a loop rather than as a chain of branches because the chain is
            # what broke: `min_score` was added as the first branch and returned from
            # inside it, so a call carrying BOTH arguments and rejected because of
            # `include_episodes` retried with the episodes still attached, failed again,
            # and propagated -- leaving the older `include_episodes` fallback below
            # unreachable for the one call site that uses it. Dropping in sequence has no
            # such ordering hazard: whatever the server objected to is gone by the end.
            #
            # `include_episodes` is the only boolean argument anywhere in the tool surface,
            # and the server's own validator has no branch for that type: a boolean falls
            # through to the string check and dies on a `KeyError: 'boolean'` looking up the
            # article for the error message it was about to raise. So that argument has
            # never worked on any deployment, for either value. Both drops self-heal the
            # day the server grows the branch, with no release here.
            optional = [key for key in ("min_score", "include_episodes") if key in args]
            if not optional:
                raise
            for index, key in enumerate(optional):
                del args[key]
                if key == "min_score":
                    # Recorded where a person actually looks. The flag alone was not
                    # enough: nothing read it, so a hosted store that cannot filter
                    # returned unfiltered memories while every visible signal said the
                    # recall had succeeded normally.
                    self.unfiltered = True
                    log_line("recall", "hosted rejected min_score; this recall is "
                                       "UNFILTERED -- the floor was not applied")
                try:
                    text = self._call("memory_recall", args)
                    break
                except HostedError:
                    if index == len(optional) - 1:
                        raise
        if not text:
            return ""
        return _reheader(text, header)

    def stats(self) -> str:
        """The server's own scope/writes/count report, or raise.

        `session_start` wants a line naming the binding, and the server already formats
        exactly that. Deriving a second version of it here would be a second thing to keep
        true.
        """
        return self._call("memory_stats", {})

    def add(self, text: str, *, role: str = "user") -> str:
        """Store one turn as an episode. Returns the server's receipt line, or raises.

        On a `fast-path-only` server this extracts nothing and stores everything: the
        episode is committed before the extraction gate is even consulted, so the prose is
        durable and searchable for zero model calls. That is the whole reason this is worth
        calling on every turn -- see `capture.py`.
        """
        if not text.strip():
            return ""
        return self._call("memory_add", {"text": text, "role": role})

    def remember(self, subject: str, predicate: str, obj: str, *,
                 confidence: float = 1.0,
                 memory_type: "str | None" = None,
                 true_since: "str | None" = None,
                 extractor: "str | None" = None,
                 sources: "list[str] | None" = None) -> str:
        """Write one triple, or raise. Returns the server's receipt line.

        Reads and writes both raise now, but for different reasons, and the write's is the
        older and stronger one: a caller cannot tell a `None` meaning "stored nothing" from
        one meaning "nothing to store", so a silent failure here is counted as a fact that
        landed and the store that gained nothing reports a successful hook.

        `memory_type` matters more than it looks. Nothing on the write path infers one from
        the words, so an omitted type means the predicate's registered default, and an
        unregistered predicate has none -- it becomes `semantic`. A standing instruction
        filed as `semantic` is invisible to a `procedural` filter, which is the filter
        anything about how to do the work should be found by.
        """
        args: dict = {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
        }
        if memory_type:
            args["memory_type"] = memory_type
        if true_since:
            args["true_since"] = true_since
        if extractor and self.accepts("memory_remember", "extractor"):
            # Sent only when the server says it takes it. Left off, the claim reports
            # itself as "Derived by user", which is what let a hook's own inference be
            # read back in a later session as something the user had stated.
            args["extractor"] = extractor
        if sources and self.accepts("memory_remember", "sources"):
            # Episode IDS, never the turn text. `_cite` on the other side STORES anything
            # handed to it as an Episode and merely LINKS a string, so sending the turn
            # would store a second copy of the one `_keep_turn` has just written.
            #
            # Probed rather than assumed, for the same reason as `extractor`: argument
            # validation there is closed, so an argument an older server has not heard of
            # loses the whole write rather than one field. memvara/memvara#76 added this
            # and is unreleased as of 2026-08-25, so on today's endpoint the probe answers
            # False and a fact is written exactly as before -- unexplainable, but written.
            args["sources"] = list(sources)
        return self._call("memory_remember", args)


def _reheader(text: str, header: "str | None") -> str:
    """Apply the caller's header, replacing the server's own rather than stacking on it.

    `memory_recall` renders its own header line, and the local library route *replaces*
    that line when a caller passes `header=`. This route used to prepend, so the hosted
    block carried two stacked headers where the local one carried a single -- the two
    routes are supposed to be byte-identical, and a caller comparing them would have found
    the difference before a reader did.

    The rule is deliberately narrow: drop the first line, and only when a replacement is
    being supplied and that line looks like a header rather than content. Recall renders
    its notes as `- ` bullets, so a leading line ending in a colon is not one of them.
    """
    if header is None:
        return text
    first, _, rest = text.partition("\n")
    stripped = first.strip()
    if rest and stripped.endswith(":") and not stripped.startswith("- "):
        text = rest
    return f"{header}\n{text}"


def _content_text(result: dict) -> str:
    """The text blocks of a tool result, joined. Empty when there are none."""
    return "\n".join(
        block.get("text", "") for block in result.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def _decode(raw: bytes) -> "dict | None":
    """A JSON-RPC reply, whether it arrived as JSON or as one SSE frame."""
    body = raw.decode("utf-8", "replace").strip()
    if not body:
        return None
    if body.startswith("{"):
        try:
            return json.loads(body)
        except ValueError:
            return None
    # text/event-stream: the payload is on `data:` lines.
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            try:
                parsed = json.loads(line[5:].strip())
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


#: One `HostedRecall` per (api key, server) for the life of this process, keyed rather
#: than a bare singleton so a credentials file that legitimately names two different
#: projects mid-process -- unlikely, but cheap to get right -- still gets two clients
#: rather than one wrongly shared between them.
#:
#: `recall.py` can reach `open_hosted()` from up to three places in a single invocation:
#: `fast.recall()`'s main pass, its episode-widening retry, and `_standing_refresh()`'s own
#: `open_writer()`. A fresh `HostedRecall` per call meant a fresh `_ensure_session()`
#: handshake per call -- a full `_rpc()` round trip with its own one retry -- so a hook that
#: reached this three times paid for the handshake three times before any of the three tool
#: calls it actually wanted even started. `close()` clears `_conn` but never `_session` (see
#: `HostedRecall.close`), so a cached instance's session survives a caller closing it after
#: its own use; the only round trip `_ensure_session()` ever needed happens once per process
#: instead of once per call.
#:
#: `daemon.py` calls this exactly once, at startup, and holds the result for the process's
#: whole life -- caching changes nothing there. The win is entirely in the short-lived hook
#: processes that used to rebuild the handshake on every call within their one invocation.
_HOSTED_CACHE: "dict[tuple[str, str], HostedRecall]" = {}


def open_hosted() -> "HostedRecall | None":
    """A hosted client if this machine is logged in, else None.

    Cached per process by `(api_key, server_url)` -- see `_HOSTED_CACHE` above.
    """
    creds = credentials()
    if creds is None:
        return None
    key = (str(creds["api_key"]), str(creds.get("server_url") or DEFAULT_BASE))
    cached = _HOSTED_CACHE.get(key)
    if cached is not None:
        return cached
    client = HostedRecall(key[0], key[1])
    _HOSTED_CACHE[key] = client
    return client
