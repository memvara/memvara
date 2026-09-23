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

Otherwise the hook asks for a plain read. A key that stops working after the check costs
nothing extra: the library reports `key_rejected` and serves the plain read.

The check's result lives in `~/.memvara/settings.json` under the key `read_model`, beside
the switches, because the hook already reads that file on every prompt:

    {"outcome": "applied", "reason": "", "backend": "anthropic", "model_setting": "",
     "model": "<the model the backend resolved>", "checked_at": "2026-09-23T10:00:00Z"}

`outcome` is one of the library's five (`applied`, `fallback`, `key_rejected`, `disabled`,
`unconfigured`) or one of three the check adds: `no_local_store` when there is no local
store to check, which is the normal state of a hosted install; `unsupported` when the
installed library predates query rewrite; and `error` when the check itself raised, with the
exception's class name as `reason`. `status` is present when the provider answered with an
HTTP status.

A hosted install is never checked and never rewrites from this hook. The hosted server
would rewrite with the organisation's own key, which this machine cannot see or test, so the
hooks' hosted client asks it for a plain read (`lib.hosted`).

Importing this module costs what the hook already pays: `allowed()` reads the settings file
the switches come from, and reads the client's server block only when a check was recorded.
`check()` imports the library, and only `/memvara:setup` calls it.
"""

from __future__ import annotations

import os
import time

from . import settings
from .fast import rewrite_kwargs
from .ipc import server_env

#: The settings key the check is recorded under.
KEY = "read_model"

#: The question the check asks the model to rewrite. It names a time, so a working model
#: has something to do with both halves of its answer: other phrasings, and a date range.
PROBE = "what did we decide about the release last week"


def configured() -> "tuple[str, str]":
    """`(backend, model setting)` as the store the hook opens would read them.

    The same precedence as `lib.open.open_store`: a variable set in this process wins over
    the client's server block. The backend is normalised the way `ServerConfig` normalises
    it; an unset model setting is `""`, meaning the backend's own default.
    """
    env = {**server_env(), **{k: v for k, v in os.environ.items() if k.startswith("MEMVARA_")}}
    return ((env.get("MEMVARA_LLM") or "none").strip().lower(),
            (env.get("MEMVARA_LLM_MODEL") or "").strip())


def allowed() -> bool:
    """Whether the per-prompt recall may ask for a query rewrite. Never raises.

    The cheap tests come first, so a user who never ran the check pays one dictionary
    lookup on a file the hook has already read.
    """
    if not settings.enabled("query_rewrite"):
        return False
    record = settings.stored(KEY)
    if not isinstance(record, dict) or record.get("outcome") != "applied":
        return False
    return (record.get("backend"), record.get("model_setting")) == configured()


def check() -> dict:
    """Make one test rewrite through the library, and return the record to store.

    Opens the store exactly as the hooks do (`lib.open.open_store`) and runs one search with
    `query_rewrite=True` and `k=1`, so the model call goes through the same code, the same
    backend and the same 10-second deadline as a rewritten recall. The cost is that one
    chat call. Never raises: whatever goes wrong is an outcome.
    """
    from .open import open_store  # noqa: PLC0415 - imports the library; setup only

    backend, model_setting = configured()
    record: dict = {"outcome": "", "reason": "", "backend": backend,
                    "model_setting": model_setting, "model": "",
                    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    store = open_store()
    if store is None:
        record["outcome"] = "no_local_store"
        return record
    try:
        llm = getattr(store, "llm", None)
        if callable(getattr(llm, "chat", None)):
            record["model"] = str(getattr(llm, "model", "") or "")
        read_kind = rewrite_kwargs(getattr(store, "search", None), True)
        if not read_kind:
            # A library released before query rewrite: its `search()` has no such argument.
            record["outcome"] = "unsupported"
            return record
        try:
            results = store.search(PROBE, k=1, **read_kind)
        except Exception as exc:  # noqa: BLE001 - reported to setup, never raised
            record["outcome"] = "error"
            record["reason"] = type(exc).__name__
            return record
        rewrite = getattr(results, "rewrite", None)
        if rewrite is None:
            record["outcome"] = "unsupported"
            return record
        record["outcome"] = rewrite.outcome
        record["reason"] = rewrite.reason or ""
        if rewrite.status is not None:
            record["status"] = rewrite.status
        return record
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
