"""Whether the per-prompt recall may ask a model to rewrite its query.

The library can rewrite a query before it searches: one call to the chat model the store
was built with (`MEMVARA_LLM` and `MEMVARA_LLM_MODEL` in the MCP server block) returns other
phrasings and a date range, and the read searches all of them. The recall hook runs on
every prompt, so a rewrite there is one model call per prompt, billed to the user's key.
The hook therefore asks for one only when all three of these hold:

1. the `query_rewrite` switch is on (`lib.settings`);
2. `/memvara:setup verify-key` made a test call through the library and the model answered
   it, which the check records as the outcome `applied`;
3. the model configured now is the one that was checked: the same `MEMVARA_LLM` and the
   same `MEMVARA_LLM_MODEL`. Setup showed the user the cost of that model, and a change of
   model needs a new check.

Otherwise the hook asks for a plain read.

**A key that stops working stops the rewrites.** A key can be rotated or revoked after the
check, and the backend and model would still match. When a rewrite during a recall comes
back `key_rejected`, the plain read is served as always, and `rejected()` marks the record
failed, so the next prompt does not spend another refused call. A replaced key that works
is not detected, and it does not need to be: the user approved one call per prompt to that
model, on their own key.

The check's result lives in its own state file, `~/.memvara/.hooks/read_model.json`,
written atomically under a lock through `lib.state_file`, because a hook process can change
it (`rejected()`) while setup writes it:

    {"outcome": "applied", "reason": "", "backend": "anthropic", "model_setting": "",
     "model": "<the model the backend resolved>", "checked_at": "2026-09-23T10:00:00Z"}

An earlier build kept the record under `read_model` in `~/.memvara/settings.json`, which is
a flat map of switches written without a lock. `recorded()` reads that old place only while
the state file is missing, and moves the record across the first time it finds one.

`outcome` is one of the library's five (`applied`, `fallback`, `key_rejected`, `disabled`,
`unconfigured`) or one of three the check adds: `no_local_store` when there is no local
store to check, which is the normal state of a hosted install; `unsupported` when the
installed library predates query rewrite; and `error` when the check itself raised, with the
exception's class name as `reason`. `status` is present when the provider answered with an
HTTP status.

A hosted install is never checked and never rewrites from this hook. The hosted server
would rewrite with the organisation's own key, which this machine cannot see or test, so the
hooks' hosted client asks it for a plain read (`lib.hosted`).

What `allowed()` costs a prompt: nothing more than the switch lookup when the switch is off,
then one read of a small state file, and the client's server block only when a check was
recorded (`lib.ipc.client_env`, read once per process). `check()` imports the library, and
only `/memvara:setup` calls it.
"""

from __future__ import annotations

import os
import time

from . import settings, state_file
from .fast import read_kinds
from .ipc import client_env

#: The settings key an earlier build recorded the check under. Read only to migrate.
KEY = "read_model"

#: Where the check is recorded. Beside the other hook state, not in the plugin, which is
#: replaced wholesale on every update.
STATE = os.path.join(os.path.expanduser("~"), ".memvara", ".hooks", "read_model.json")

#: The question the check asks the model to rewrite. It names a time, so a working model
#: has something to do with both halves of its answer: other phrasings, and a date range.
PROBE = "what did we decide about the release last week"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _lock() -> str:
    return STATE + ".lock"


def configured() -> "tuple[str, str]":
    """`(backend, model setting)` as the store the hook opens would read them.

    The environment is `lib.ipc.client_env`, the one rule the daemon's address and the
    store the hooks open also use. The backend is normalised the way `ServerConfig`
    normalises it; an unset model setting is `""`, meaning the backend's own default.
    """
    env = client_env()
    return ((env.get("MEMVARA_LLM") or "none").strip().lower(),
            (env.get("MEMVARA_LLM_MODEL") or "").strip())


def recorded() -> "dict | None":
    """The last check's record, or `None` when there is none. Never raises.

    Reads the settings file's old `read_model` entry only when the state file is missing,
    and moves it into the state file the first time, so the old place is read once.
    """
    record = state_file.read_json(STATE)
    if record:
        return record
    old = settings.stored(KEY)
    if not isinstance(old, dict):
        return None
    save(old)
    return old


def save(record: dict) -> bool:
    """Record a check, replacing the last one. `False` when it could not be written."""
    return state_file.update_json(STATE, lambda _was: dict(record), lock_path=_lock(),
                                  prefix=".read-model-")


def rejected() -> None:
    """Mark a verified key as rejected, because a rewrite during a recall was refused.

    Changes nothing unless the record says `applied`: a check that already failed stays as
    it was, and with no record there is nothing to turn off. Never raises.
    """
    record = recorded()
    if not isinstance(record, dict) or record.get("outcome") != "applied":
        return

    def change(was: object) -> dict:
        current = dict(was) if isinstance(was, dict) else dict(record)
        current.update(outcome="key_rejected", reason="refused during a recall",
                       checked_at=_now())
        return current

    state_file.update_json(STATE, change, lock_path=_lock(), prefix=".read-model-")


def allowed() -> bool:
    """Whether the per-prompt recall may ask for a query rewrite. Never raises.

    The cheap tests come first, so a user who switched the feature off pays one dictionary
    lookup on a file the hook has already read.
    """
    return settings.enabled("query_rewrite") and verified_for_current_config()


def verified_for_current_config() -> bool:
    """Whether the last check found a working key for the model configured now. Never raises.

    `allowed()` is this and the `query_rewrite` switch. `/memvara:setup` asks this alone,
    to learn whether turning the switch on would start one model call per prompt. The
    record must say `applied`, which a rejection during a recall undoes (`rejected()`),
    and name the same `MEMVARA_LLM` and `MEMVARA_LLM_MODEL` as `configured()`.
    """
    record = recorded()
    if not isinstance(record, dict) or record.get("outcome") != "applied":
        return False
    return (record.get("backend"), record.get("model_setting")) == configured()


def check() -> dict:
    """Make one test rewrite through the library, and return the record to `save()`.

    Opens the store exactly as the hooks do (`lib.open.open_store`) and runs one search with
    `query_rewrite=True` and `k=1`, so the model call goes through the same code, the same
    backend and the same 10-second deadline as a rewritten recall. The cost is that one
    chat call. Never raises: whatever goes wrong is an outcome. Writes nothing; setup
    decides whether to save the result.
    """
    from .open import open_store  # noqa: PLC0415 - imports the library; setup only

    backend, model_setting = configured()
    record: dict = {"outcome": "", "reason": "", "backend": backend,
                    "model_setting": model_setting, "model": "", "checked_at": _now()}
    store = open_store()
    if store is None:
        record["outcome"] = "no_local_store"
        return record
    read_kind = read_kinds(store, "search")[1]
    try:
        llm = getattr(store, "llm", None)
        if callable(getattr(llm, "chat", None)):
            record["model"] = str(getattr(llm, "model", "") or "")
        # A library released before query rewrite: its `search()` has no such argument.
        rewrite = (getattr(store.search(PROBE, k=1, **read_kind), "rewrite", None)
                   if read_kind else None)
        if rewrite is None:
            record["outcome"] = "unsupported"
        else:
            record["outcome"] = rewrite.outcome
            record["reason"] = rewrite.reason or ""
            if rewrite.status is not None:
                record["status"] = rewrite.status
    except Exception as exc:  # noqa: BLE001 - reported to setup, never raised
        record["outcome"] = "error"
        record["reason"] = type(exc).__name__
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    return record
