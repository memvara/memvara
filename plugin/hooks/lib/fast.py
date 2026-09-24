"""The client half: ask the daemon, and never depend on there being one.

The contract this file exists to keep is that the daemon is an optimisation and not a
dependency. Every path through `recall()` returns the same text; only the latency differs.
If that stopped being true — if a missing daemon meant a missing memory block — a
background process would be trading a real risk for 136ms, which is not a trade worth
making on someone's prompt path.

Order of preference, and what each costs when it fails:

1. **Daemon.** ~38ms end to end, most of it this client's own interpreter startup. A
   missing or wedged one costs the connect attempt, which is sub-millisecond against a
   socket that is not there.

   A daemon that answers is no longer automatically believed. It used to be: any reply,
   including the empty string, ended the search. But the empty string was also what a
   *failed* query returned, so one broken backend behind a live socket silently disabled
   recall for an entire session while this exact fallback chain sat unused. The daemon now
   says which of the two happened and only `ok: true` is authoritative.
2. **In-process library.** ~148ms, the pre-daemon behaviour, always correct. Skipped
   entirely when the library is not installed, which is the normal hosted case.
3. **Hosted over stdlib HTTP.** ~390ms cold. Needs no `pip install`, which is the point:
   the hosted install story is "paste a URL", and a hook that waited for a Python package
   would be silently dead on exactly the machines this is aimed at.
4. **Nothing.** No store, no login: empty string, no output, no error.

Spawning is deliberately *after* answering. The first prompt of a session should not wait
on a process that cannot help it yet, so the daemon is started for the benefit of the next
one and this prompt takes the slow path.

**Every read says whether the library may rewrite its query.** A library store whose model
can chat rewrites by default, so a plain read has to ask for one: `query_rewrite=False`.
`recall(query_rewrite=True)` is how the recall hook asks for a rewrite, and it does that
only when `lib.read_model.allowed()` says setup verified a key. A library released before
query rewrite has no such argument and never rewrites, so it is asked exactly as before
(`read_kinds`, decided once per store). The hosted route never asks for a rewrite; see
`lib.hosted`.

A rewrite is one model call with a 10-second deadline, and the hook's own allowance is 10
seconds in all. So a rewritten read gets `rewrite_wait` seconds, `REWRITE_WAIT_SEC` unless
the caller says otherwise, and after that the plain read is served instead. Once a daemon
has been handed a rewrite and did not answer in time, or answered with a failure, the
fallback below it is a plain read: the daemon may still be making that model call, and a
second one here would be billed twice for one prompt.
"""

from __future__ import annotations

import json
import os
import sys
import time

from .ipc import CLIENT_TIMEOUT_SEC, log_line, send, socket_path, store_key

#: Set in a spawned daemon's environment so a daemon can never spawn a daemon.
SENTINEL = "MEMVARA_DAEMON"

#: How long a read that asks for a query rewrite may take before the plain read is served
#: instead, in seconds. The library's own deadline for the model call is 10 seconds, which
#: is the recall hook's whole allowance, so the hook stops waiting sooner. Five seconds is
#: long enough for a small model's reply of up to 300 tokens and leaves the hook time to
#: serve the plain read and print its banner. `recall.py` starts a rewrite only when this
#: much of its own budget is left.
REWRITE_WAIT_SEC = 5.0

#: The clock the daemon wait is measured with. A name here so a test can move it.
_clock = time.monotonic


def read_kinds(store: object, method: str = "recall") -> "tuple[dict, dict]":
    """`(plain, rewrite)`: the keyword arguments for each kind of read of `store`.

    `method` is `recall` for every read a hook makes, and `search` for the one test call
    `lib.read_model.check()` makes.

    Decided once per store, from the signature of its `recall`, by whoever holds the store:
    the daemon when it starts, the in-process route when it opens its handle, the
    session-start hook once. Every read then spreads one of the two dicts, spelled
    `**plain_read` or `**read_kind`, which is how `tests/test_read_stages.py` knows the read
    says which kind it is.

    A library store has taken `query_rewrite` since query rewrite was added, and rewrites
    unless it is told `False`, so its plain read is `{"query_rewrite": False}`. Its
    rewritten read also asks for `with_ids=True`, which returns a `RecallResult` whose
    `rewrite` says how the model call went; `text_of` reads it. Every library released
    before query rewrite has no such argument and raises `TypeError` when handed one, which
    the hook would report as a store it could not ask. Those never rewrite, so both of
    their dicts are empty, and so are the hooks' own hosted client's. A method that
    forwards `**kwargs` is taken to accept both arguments, since the library's wrappers do
    exactly that. An empty `rewrite` means this store cannot rewrite.
    """
    import inspect  # noqa: PLC0415 - only the in-process route and the daemon need it

    try:
        parameters = inspect.signature(getattr(store, method)).parameters
    except (AttributeError, TypeError, ValueError):
        return {}, {}
    forwards = any(p.kind is p.VAR_KEYWORD for p in parameters.values())
    if not forwards and "query_rewrite" not in parameters:
        return {}, {}
    rewrite: dict = {"query_rewrite": True}
    if method == "recall" and (forwards or "with_ids" in parameters):
        rewrite["with_ids"] = True
    return {"query_rewrite": False}, rewrite


def text_of(result: object) -> str:
    """The text of one read, noting a rejected key on the way.

    A rewritten read returns a `RecallResult`; every other read returns text. When the
    provider refused the key (`key_rejected`), the plain read was still served, and the
    verification is marked failed (`lib.read_model.rejected`) so that the next prompt does
    not spend another refused call on a key nobody has checked since.
    """
    rewrite = getattr(result, "rewrite", None)
    if getattr(rewrite, "outcome", None) == "key_rejected":
        from .read_model import rejected  # noqa: PLC0415 - only a rejected key needs it

        rejected()
    return str(getattr(result, "text", result) or "")


def _within(wait: float, call, fallback):
    """`call()` when it returns within `wait` seconds, else `fallback()`.

    `call` runs on a daemon thread, so a model call still in flight when the wait ends does
    not hold the hook's process open: the process exits when the hook is done, and the
    call's reply is never read. An exception from `call` is raised here, as it would have
    been without the thread.

    The fallback reads the same store while the abandoned thread may still be inside it.
    That is not a race: `SQLiteStore` gives every thread its own reader connection, and the
    library runs a rewrite's phrasings on threads of their own for the same reason.
    """
    import threading  # noqa: PLC0415 - only a rewritten read needs it

    box: list = []

    def run() -> None:
        try:
            box.append((True, call()))
        except BaseException as exc:  # noqa: BLE001 - handed back to the caller below
            box.append((False, exc))

    worker = threading.Thread(target=run, name="memvara-rewrite", daemon=True)
    worker.start()
    worker.join(wait)
    if not box:
        log_line("recall", f"query rewrite still running after {wait:g}s; "
                           "served the plain read")
        return fallback()
    finished, value = box[0]
    if not finished:
        raise value
    return value


def _spawn(root: str) -> None:
    """Start a daemon for next time. Best effort, and silent about failing."""
    if os.environ.get(SENTINEL):
        return
    env = dict(os.environ)
    env[SENTINEL] = "1"
    # Imported here, not at module scope: `subprocess` costs 5.8ms and is only ever needed
    # on the fallback path, which has already lost far more than that.
    import subprocess

    try:
        subprocess.Popen(
            [sys.executable, os.path.join(root, "daemon.py")],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Detach from this hook's process group: the daemon must outlive the hook,
            # and must not receive the signals Claude Code sends to its own children.
            start_new_session=True,
            env=env,
            cwd=root,
        )
    except (OSError, ValueError):
        pass


#: What a caller may be told about a failure, as a token rather than a sentence. Words are
#: the banner's business; this file's job is to say which kind of failure it was without
#: importing anything to do it.
QUOTA = "quota"

#: The token for a plan's daily recall allowance being used up: `daily` alone,
#: `daily:<seconds until it resets>`, or `daily:<HH:MM>` for the reset time in UTC when the
#: refusal carried no wait. Paid plans meter recalls per day, and the service answers a
#: spent daily allowance with HTTP 429 and code `rate_limited`, the same status and code as
#: a plain rate limit. The two differ in `detail`: an allowance names its `reason`
#: (`over_period_allowance`) and `resets_at`, and a rate limit names the `rule` that bound
#: instead. Read from memvara-cloud's `rest/limits.py`.
DAILY = "daily"


def _reason(exc: "BaseException") -> str:
    """The short token for a failure, or `""` when there is nothing useful to add.

    Duck-typed on the attribute rather than on the class, so this file keeps its promise
    not to import `lib.hosted` -- which pulls in `ssl` and `http.client` -- on a path that
    runs for every prompt against a ~30ms budget. `getattr` on an exception costs nothing
    and an exception that does not carry a code answers `""`.
    """
    code = getattr(exc, "code", "")
    detail = getattr(exc, "detail", None)
    if not isinstance(detail, dict):
        detail = {}
    if code == "rate_limited" and detail.get("reason") == "over_period_allowance":
        # The wait comes from `Retry-After`, which the service sends with this refusal as
        # the seconds until the allowance resets. Without it, the reset time is read from
        # `detail.resets_at`, and only in the UTC form the service writes, because a
        # misread offset would show a person the wrong time.
        wait = getattr(exc, "retry_after", None)
        if isinstance(wait, int):
            return f"{DAILY}:{wait}"
        when = str(detail.get("resets_at") or "")
        if len(when) >= 16 and when[10] == "T" and when.endswith(("+00:00", "Z")):
            return f"{DAILY}:{when[11:16]}"
        return DAILY
    if code != "quota_exhausted":
        return ""
    when = str(detail.get("resets_at") or "")[:10]
    # The date rides along because it is the half that makes the banner actionable: "spent"
    # tells the reader to stop retrying, and only "resets on the 1st" tells them how long
    # for. Joined into the token rather than given its own slot -- one more slot for one
    # more fact does not generalise, and the caller splitting on a colon does.
    return f"{QUOTA}:{when}" if when else QUOTA


def recall(query: str, *, k: int = 6, budget: int = 700, header: str | None = None,
           include_episodes: bool = False, memory_types: "list[str] | None" = None,
           min_score: float = 0.0, query_rewrite: bool = False,
           rewrite_wait: float = REWRITE_WAIT_SEC,
           spawn: bool = True) -> "tuple[str, bool | None, str]":
    """Recall text for `query`, by whatever route is available.

    Returns `(text, ok, reason)`. `ok` has three states, because there are three things
    that can happen
    and collapsing any two of them hides a real one:

    * `True` -- a store was asked and answered. `text` may still be empty, and that is a
      fact about the store rather than about the plumbing.
    * `False` -- a store was there and could not be reached. This is the state that used to
      be indistinguishable from the one above, and a hosted client with a stale session
      exploited exactly that: "no matching memories", every prompt, for a whole session,
      over a store that was full. Nobody investigates an empty store.
    * `None` -- there is no store to ask. No database, no library, no credentials. Not a
      failure, and reporting it as one sends someone who has simply not logged in to read a
      log that will tell them nothing.

    The third slot is `reason`: `""` when there is nothing to add, else a short token the
    caller can turn into words: `"quota"` for a spent monthly allowance and `"daily"` for a
    spent daily one, each with its reset after a colon when the refusal said. It exists
    because `False` alone sent a user to read a log about a store that was answering
    perfectly and telling him, in the body of a 402, exactly which allowance was spent and
    when it resets.

    A plain tuple rather than a NamedTuple on purpose: `typing` is not imported anywhere on
    this path, and this file runs on every prompt against a ~30ms budget. A third slot
    costs nothing; a class would cost the import.

    `query_rewrite=True` asks a library store to rewrite the query first, and the read is
    abandoned for the plain one after `rewrite_wait` seconds. See the module docstring.
    """
    if not query.strip():
        return "", True, ""

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        path = socket_path(store_key())
    except Exception:
        path = None

    if path is not None:
        request = {"q": query, "k": k, "budget": budget}
        if min_score:
            # Sent whenever it is set, so the daemon and the direct path apply the same
            # floor. A daemon is an optimisation and never a dependency: the two routes
            # returning different text for one query is the failure that rule exists for.
            request["min_score"] = min_score
        if header:
            request["header"] = header
        if include_episodes:
            request["include_episodes"] = True
        if memory_types:
            request["memory_types"] = list(memory_types)
        if query_rewrite:
            # Sent only when asked, like the floor: the daemon reads a missing key as a
            # plain read.
            request["query_rewrite"] = True
        wait = rewrite_wait if query_rewrite else CLIENT_TIMEOUT_SEC
        began = _clock()
        answer = send(path, request, timeout=wait)
        served = _served(answer)
        if served is not None:
            # `""` from a healthy daemon is a real answer -- this store has nothing
            # relevant -- and must not send the slow path off to ask again. A daemon
            # reporting failure is the opposite and falls through.
            return served, True, ""
        if query_rewrite and (answer is not None or _clock() - began >= wait):
            # The daemon took the rewrite and did not serve it. A refused connection
            # returns at once, so a wait this long means a daemon was there. It may still
            # be making the model call, so the read below must not make a second one.
            log_line("recall", "the daemon did not serve the rewritten read in time; "
                               "reading without a rewrite")
            query_rewrite = False

    store, plain_read, rewrite_read = _local_store()
    if store is None:
        # No local library or no local store. Hosted is the remaining route, and on a
        # paste-the-URL install it is the only one there ever was.
        from .hosted import open_hosted

        client = open_hosted()
        if client is None:
            # Nothing is configured at all -- no local database, no library to read one
            # with, and no credentials file. Distinct from a store that would not answer.
            return "", None, ""
        try:
            # No `query_rewrite` here, whatever the caller asked: the hosted client always
            # asks its server for a plain read. See `lib.read_model`.
            text = client.recall(query, k=k, budget=budget, header=header,
                                 include_episodes=include_episodes,
                                 memory_types=memory_types, min_score=min_score)
        except Exception as exc:
            # Including HostedError. Nothing below this to fall through to -- but the
            # caller still has a banner to print, and "could not ask" is not "nothing
            # to say". Nor is "could not ask" the same as "would not": a refusal the
            # server explained is worth repeating rather than flattening to False.
            return "", False, _reason(exc)
        finally:
            client.close()
        if spawn and path is not None:
            _spawn(root)
        return text, True, ""

    try:
        kwargs = {"k": k, "budget": budget}
        if min_score:
            kwargs["min_score"] = min_score
        if header:
            kwargs["header"] = header
        if include_episodes:
            kwargs["include_episodes"] = True
        if memory_types:
            kwargs["memory_types"] = list(memory_types)
        read_kind = rewrite_read if query_rewrite else {}
        if read_kind:
            text = text_of(_within(rewrite_wait,
                                   lambda: store.recall(query, **kwargs, **read_kind),
                                   lambda: store.recall(query, **kwargs, **plain_read)))
        else:
            text = text_of(store.recall(query, **kwargs, **plain_read))
    except Exception as exc:
        if spawn and path is not None:
            _spawn(root)
        return "", False, _reason(exc)

    if spawn and path is not None:
        _spawn(root)
    return text, True, ""


#: The store this process opened for in-process reads: `(opener, store, plain, rewrite)`.
#: The recall hook can read twice on one prompt -- the episode-widening retry follows the
#: first read -- and a second `open_store()` would open a second, independent handle on
#: the same store. The opener is part of the key so that a caller that replaces
#: `lib.open.open_store`, as the tests do, gets the store it asked for.
_OPENED: "tuple[object, object, dict, dict] | None" = None


def _local_store() -> "tuple[object, dict, dict]":
    """`(store, plain, rewrite)` for in-process reads, or `(None, {}, {})` for none.

    Opened once per process, and its kinds of read decided once (`read_kinds`). "No store"
    is not kept: on a hosted install it is the normal answer, and asking again is cheap.
    """
    global _OPENED
    from . import open as opener  # noqa: PLC0415 - imports pathlib; not on the daemon route

    if _OPENED is not None and _OPENED[0] is opener.open_store:
        return _OPENED[1], _OPENED[2], _OPENED[3]
    store = opener.open_store()
    if store is None:
        return None, {}, {}
    plain, rewrite = read_kinds(store)
    _OPENED = (opener.open_store, store, plain, rewrite)
    return store, plain, rewrite


def _served(answer: "str | None") -> "str | None":
    """The daemon's text if it answered successfully, else None meaning "fall through".

    Three cases collapse to None and should: no daemon at all (`answer is None`), a daemon
    reporting a failed query (`ok: false`), and a reply this client cannot parse -- which is
    not expected, since the socket address digests the sources of both ends, but is the same
    situation from here.
    """
    if answer is None:
        return None
    try:
        reply = json.loads(answer)
    except ValueError:
        return None
    if not isinstance(reply, dict) or not reply.get("ok"):
        return None
    return str(reply.get("text") or "")
