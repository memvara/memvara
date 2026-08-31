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
"""

from __future__ import annotations

import json
import os
import sys

from .ipc import send, socket_path, store_key

#: Set in a spawned daemon's environment so a daemon can never spawn a daemon.
SENTINEL = "MEMVARA_DAEMON"


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


def _reason(exc: "BaseException") -> str:
    """The short token for a failure, or `""` when there is nothing useful to add.

    Duck-typed on the attribute rather than on the class, so this file keeps its promise
    not to import `lib.hosted` -- which pulls in `ssl` and `http.client` -- on a path that
    runs for every prompt against a ~30ms budget. `getattr` on an exception costs nothing
    and an exception that does not carry a code answers `""`.
    """
    if getattr(exc, "code", "") != "quota_exhausted":
        return ""
    detail = getattr(exc, "detail", None)
    when = str((detail or {}).get("resets_at") or "")[:10]
    # The date rides along because it is the half that makes the banner actionable: "spent"
    # tells the reader to stop retrying, and only "resets on the 1st" tells them how long
    # for. Joined into the token rather than given its own slot -- one more slot for one
    # more fact does not generalise, and the caller splitting on a colon does.
    return f"{QUOTA}:{when}" if when else QUOTA


def recall(query: str, *, k: int = 6, budget: int = 700, header: str | None = None,
           include_episodes: bool = False, memory_types: "list[str] | None" = None,
           min_score: float = 0.0,
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
    caller can turn into words -- `"quota"` today. It exists because `False` alone sent a
    user to read a log about a store that was answering perfectly and telling him, in the
    body of a 402, exactly which allowance was spent and when it resets.

    A plain tuple rather than a NamedTuple on purpose: `typing` is not imported anywhere on
    this path, and this file runs on every prompt against a ~30ms budget. A third slot
    costs nothing; a class would cost the import.
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
        answer = send(path, request)
        served = _served(answer)
        if served is not None:
            # `""` from a healthy daemon is a real answer -- this store has nothing
            # relevant -- and must not send the slow path off to ask again. A daemon
            # reporting failure is the opposite and falls through.
            return served, True, ""

    from .open import open_store

    store = open_store()
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
        text = str(store.recall(query, **kwargs) or "")
    except Exception as exc:
        if spawn and path is not None:
            _spawn(root)
        return "", False, _reason(exc)

    if spawn and path is not None:
        _spawn(root)
    return text, True, ""


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
