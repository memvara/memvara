"""The per-prompt recall hook and query rewrite.

The recall hook runs on every prompt. A query rewrite there is one model call per prompt,
billed to the user's key, so the hook asks for one only when `/memvara:setup verify-key`
made a test call that the model answered, for the model configured now, and the
`query_rewrite` switch is on. Every other prompt gets a plain read, which the library must
be told about explicitly: a store whose `llm=` can chat rewrites by default.

Whatever happens to the rewrite, the prompt still gets its memories. A rewrite that runs
past the hook's wait is abandoned and the plain read is served, and a library too old to
know the argument is asked exactly as before.

Nothing here sleeps or reaches a network. The model is a fake `Chat`; a slow model is one
that blocks on an event the test never sets until it has checked the result.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
from typing import Any

import pytest

from memvara import HashingEmbedder, Memvara, NullLLM
from memvara.llm.base import Usage

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

from lib import fast, ipc, read_model, settings  # noqa: E402
from lib import open as opener  # noqa: E402
from lib.hosted import HostedRecall  # noqa: E402

#: Taken before any fixture replaces it, for the one test that reads a real config file.
REAL_SERVER_ENV = ipc.server_env

REWRITE_REPLY = json.dumps({"queries": ["release decision"], "date_range": None})


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Keep every test away from the real `~/.memvara`, the client's config and switches."""
    monkeypatch.setattr(settings, "SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(settings, "_LOADED", None)
    monkeypatch.setattr(read_model, "STATE", str(tmp_path / "hooks" / "read_model.json"))
    monkeypatch.setattr(ipc, "server_env", lambda: {})
    monkeypatch.setattr(fast, "log_line", lambda *a, **k: None)
    monkeypatch.setattr(fast, "_OPENED", None)
    for name in list(os.environ):
        if name.startswith("MEMVARA_FEATURE_") or name in ("MEMVARA_LLM", "MEMVARA_LLM_MODEL"):
            monkeypatch.delenv(name)


def write_settings(data: dict) -> None:
    pathlib.Path(settings.SETTINGS).write_text(json.dumps(data), encoding="utf-8")
    settings._LOADED = None


def write_record(record: object) -> None:
    """Put a verification record where `/memvara:setup verify-key --yes` puts it."""
    path = pathlib.Path(read_model.STATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def stored_record() -> dict:
    return json.loads(pathlib.Path(read_model.STATE).read_text(encoding="utf-8"))


def verified(**overrides: Any) -> dict:
    record = {"outcome": "applied", "reason": "", "backend": "anthropic",
              "model_setting": "", "model": "a-model", "checked_at": "2026-09-23T10:00:00Z"}
    record.update(overrides)
    return record


class FakeChat(NullLLM):
    """An extraction backend that can also chat. Records every call; can refuse or block."""

    model = "a-model"

    def __init__(self, reply: str = REWRITE_REPLY, *, raises: Exception | None = None,
                 gate: "threading.Event | None" = None) -> None:
        super().__init__()
        self.reply = reply
        self.raises = raises
        self.gate = gate
        self.calls: list[str] = []

    def chat(self, system: str, prompt: str, *, json_object: bool,
             max_completion_tokens: int, timeout: float,
             usage: Usage | None = None) -> str:
        self.calls.append(prompt)
        if self.gate is not None:
            # Bounded, so a regression that makes a second blocked call fails the test
            # instead of hanging it.
            self.gate.wait(timeout=5)
        if self.raises is not None:
            raise self.raises
        return self.reply


def store(llm: Any = None, **kw: Any) -> Memvara:
    made = Memvara(llm=llm if llm is not None else NullLLM(),
                   embedder=HashingEmbedder(dim=64), user="alice", **kw)
    made.remember("user", "decided", "ship the release on Friday")
    return made


# --- the switch store carries the library's defaults -----------------------------------


def test_extraction_chunks_reads_as_off_when_nothing_sets_it():
    """The library ships `extraction_chunks` off; the hooks' copy used to say on."""
    assert settings.enabled("extraction_chunks") is False
    assert settings.enabled("query_rewrite") is True


def test_reload_reads_a_settings_file_changed_in_this_process():
    """`/memvara:setup` writes the file and then reports what the hooks will read."""
    write_settings({"query_rewrite": True})
    assert settings.enabled("query_rewrite") is True
    pathlib.Path(settings.SETTINGS).write_text('{"query_rewrite": false}', encoding="utf-8")
    assert settings.enabled("query_rewrite") is True, "read once per process"
    settings.reload()
    assert settings.enabled("query_rewrite") is False


def test_verified_for_the_current_config_ignores_the_switch(monkeypatch):
    """Setup asks it to learn whether turning the switch on would start rewrites."""
    monkeypatch.setenv("MEMVARA_LLM", "anthropic")
    assert read_model.verified_for_current_config() is False
    write_record(verified())
    write_settings({"query_rewrite": False})
    assert read_model.verified_for_current_config() is True
    assert read_model.allowed() is False
    monkeypatch.setenv("MEMVARA_LLM_MODEL", "another-model")
    assert read_model.verified_for_current_config() is False


def test_a_setting_that_is_not_a_switch_is_handed_over_as_stored():
    write_settings({"read_model": {"outcome": "applied"}, "recall_mark": False})
    assert settings.stored("read_model") == {"outcome": "applied"}
    assert settings.stored("nothing") is None


# --- when the hook may rewrite -----------------------------------------------------------


def test_nothing_recorded_means_a_plain_read(monkeypatch):
    monkeypatch.setenv("MEMVARA_LLM", "anthropic")
    assert read_model.allowed() is False


def test_a_verified_key_for_the_configured_model_allows_a_rewrite(monkeypatch):
    monkeypatch.setenv("MEMVARA_LLM", "anthropic")
    write_record(verified())
    assert read_model.allowed() is True


def test_the_client_config_names_the_model_when_the_environment_does_not(monkeypatch):
    """The hook finds the model where the MCP server does: the client's server block."""
    monkeypatch.setattr(ipc, "server_env",
                        lambda: {"MEMVARA_LLM": "openai", "MEMVARA_LLM_MODEL": "m-2"})
    write_record(verified(backend="openai", model_setting="m-2"))
    assert read_model.configured() == ("openai", "m-2")
    assert read_model.allowed() is True
    monkeypatch.setenv("MEMVARA_LLM_MODEL", "m-3")
    assert read_model.configured() == ("openai", "m-3"), "the environment wins, as it does "
    "for the store the hook opens"
    assert read_model.allowed() is False


@pytest.mark.parametrize("switch", [{"query_rewrite": False}, {}])
def test_the_switch_off_means_a_plain_read_whatever_was_verified(monkeypatch, switch):
    monkeypatch.setenv("MEMVARA_LLM", "anthropic")
    write_record(verified())
    write_settings(switch)
    if not switch:
        monkeypatch.setenv("MEMVARA_FEATURE_QUERY_REWRITE", "0")
    assert read_model.allowed() is False


@pytest.mark.parametrize("record", [
    verified(outcome="key_rejected"),
    verified(outcome="fallback", reason="timeout"),
    verified(outcome="unconfigured"),
    verified(outcome="no_local_store"),
    "applied",
    ["applied"],
])
def test_a_check_the_model_did_not_answer_means_a_plain_read(monkeypatch, record):
    monkeypatch.setenv("MEMVARA_LLM", "anthropic")
    write_record(record)
    assert read_model.allowed() is False


@pytest.mark.parametrize("backend, model", [("openai", ""), ("anthropic", "another-model")])
def test_a_different_model_from_the_one_checked_means_a_plain_read(monkeypatch, backend,
                                                                   model):
    """Setup showed the cost of one model. A change of model needs a new check."""
    monkeypatch.setenv("MEMVARA_LLM", backend)
    if model:
        monkeypatch.setenv("MEMVARA_LLM_MODEL", model)
    write_record(verified())
    assert read_model.allowed() is False


# --- where the verification lives -------------------------------------------------------


def test_a_saved_verification_is_a_state_file_of_its_own(monkeypatch):
    """Not a key in the switch file, which is a flat map of switches written unlocked."""
    monkeypatch.setenv("MEMVARA_LLM", "anthropic")
    assert read_model.save(verified()) is True
    assert stored_record() == verified()
    assert not pathlib.Path(settings.SETTINGS).exists()
    assert read_model.allowed() is True


def test_a_verification_in_the_old_place_is_read_once_and_moved(monkeypatch):
    """An earlier build of this branch kept it under `read_model` in settings.json."""
    monkeypatch.setenv("MEMVARA_LLM", "anthropic")
    write_settings({"read_model": verified()})
    assert read_model.allowed() is True
    assert stored_record() == verified(), "moved to the state file on first read"
    write_settings({"read_model": verified(outcome="fallback")})
    assert read_model.allowed() is True, "the old place is not read again"


def test_a_state_file_that_cannot_be_written_is_reported(tmp_path, monkeypatch):
    blocker = tmp_path / "a-file"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setattr(read_model, "STATE", str(blocker / "read_model.json"))
    assert read_model.save(verified()) is False


# --- a key that stops working ---------------------------------------------------------


def _rejected_key() -> Exception:
    rejected = RuntimeError("401")
    rejected.status_code = 401  # type: ignore[attr-defined]
    return rejected


def test_a_key_the_provider_rejects_during_a_recall_stops_the_rewrites(monkeypatch,
                                                                       tmp_path):
    """A rotated or revoked key would otherwise cost every prompt a refused call."""
    _no_daemon(monkeypatch, tmp_path)
    monkeypatch.setenv("MEMVARA_LLM", "anthropic")
    write_record(verified())
    monkeypatch.setattr(opener, "open_store",
                        lambda: store(FakeChat(raises=_rejected_key())))
    text, ok, _ = fast.recall("what did we decide", query_rewrite=True, spawn=False)
    assert ok is True and "Friday" in text
    assert stored_record()["outcome"] == "key_rejected"
    assert stored_record()["backend"] == "anthropic", "the rest of the record is kept"
    assert read_model.allowed() is False


def test_the_daemon_stops_the_rewrites_on_a_rejected_key_too(monkeypatch):
    import daemon as daemon_hook

    monkeypatch.setenv("MEMVARA_LLM", "anthropic")
    write_record(verified())
    served = daemon_hook.Daemon("/tmp/unused-reject.sock",
                                store(FakeChat(raises=_rejected_key())))
    reply = served._answer({"q": "what did we decide", "k": 2, "budget": 300,
                            "query_rewrite": True})
    assert reply["ok"] is True and "Friday" in reply["text"]
    assert read_model.allowed() is False


def test_a_rejection_with_nothing_verified_writes_nothing():
    read_model.rejected()
    assert not pathlib.Path(read_model.STATE).exists()


def test_a_rewrite_that_answered_leaves_the_verification_alone(monkeypatch, tmp_path):
    _no_daemon(monkeypatch, tmp_path)
    monkeypatch.setenv("MEMVARA_LLM", "anthropic")
    write_record(verified())
    monkeypatch.setattr(opener, "open_store", lambda: store(FakeChat()))
    text, ok, _ = fast.recall("what did we decide", query_rewrite=True, spawn=False)
    assert ok is True and "Friday" in text
    assert stored_record() == verified()


# --- one rule for the environment -------------------------------------------------------


def test_the_process_environment_wins_over_the_clients_server_block(monkeypatch):
    monkeypatch.setattr(ipc, "server_env",
                        lambda: {"MEMVARA_DB": "/from/block.db", "MEMVARA_LLM": "openai"})
    monkeypatch.setenv("MEMVARA_LLM", "anthropic")
    env = ipc.client_env()
    assert (env["MEMVARA_DB"], env["MEMVARA_LLM"]) == ("/from/block.db", "anthropic")
    assert read_model.configured() == ("anthropic", "")


def test_only_the_shared_helper_merges_the_server_block():
    """`store_key`, `open_store` and the rewrite decision must agree on one rule."""
    sources = {name: (HOOKS / name).read_text(encoding="utf-8")
               for name in ("lib/ipc.py", "lib/open.py", "lib/read_model.py")}
    import re

    for name, source in sources.items():
        calls = re.findall(r"(?<!def )\bserver_env\(\)", source)
        assert len(calls) == (1 if name == "lib/ipc.py" else 0), name
    for name in ("lib/open.py", "lib/read_model.py"):
        assert "client_env()" in sources[name], name
    assert "client_env()" in sources["lib/ipc.py"].split("def store_key")[1]


def test_the_client_config_is_read_once_per_process(monkeypatch, tmp_path):
    monkeypatch.setattr(ipc, "server_env", REAL_SERVER_ENV)  # not the fixture's stand-in
    config = tmp_path / "client.json"
    config.write_text(json.dumps({"mcpServers": {"memvara": {"env": {"MEMVARA_DB": "x"}}}}),
                      encoding="utf-8")
    monkeypatch.setattr(ipc, "_CLIENT_CONFIGS", (str(config),))
    monkeypatch.setattr(ipc, "_SERVER_ENV", None)
    opened: list[str] = []
    real_open = open

    def counting_open(path, *args, **kwargs):
        if str(path) == str(config):
            opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)
    assert ipc.server_env() == {"MEMVARA_DB": "x"}
    ipc.store_key()
    ipc.client_env()
    assert len(opened) == 1
    monkeypatch.setattr(ipc, "_CLIENT_CONFIGS", ())
    assert ipc.server_env() == {}, "a different set of config files is read afresh"


# --- the check setup makes ---------------------------------------------------------------


def test_the_check_makes_one_rewrite_call_and_records_that_it_answered(monkeypatch):
    monkeypatch.setenv("MEMVARA_LLM", "Anthropic ")
    model = FakeChat()
    closed: list[bool] = []
    made = store(model)
    monkeypatch.setattr(made, "close", lambda: closed.append(True))
    monkeypatch.setattr(opener, "open_store", lambda: made)
    record = read_model.check()
    assert not pathlib.Path(read_model.STATE).exists(), "setup decides whether to save it"
    assert len(model.calls) == 1
    assert read_model.PROBE in model.calls[0]
    assert record["outcome"] == "applied"
    assert (record["backend"], record["model_setting"], record["model"]) == (
        "anthropic", "", "a-model")
    assert record["reason"] == "" and "status" not in record
    assert record["checked_at"].endswith("Z")
    assert closed == [True], "the store the check opened is closed again"


def test_a_rejected_key_is_recorded_with_its_status(monkeypatch):
    rejected = RuntimeError("401")
    rejected.status_code = 401  # type: ignore[attr-defined]
    monkeypatch.setattr(opener, "open_store", lambda: store(FakeChat(raises=rejected)))
    record = read_model.check()
    assert (record["outcome"], record["status"]) == ("key_rejected", 401)


def test_a_failed_call_is_recorded_with_its_reason(monkeypatch):
    monkeypatch.setattr(opener, "open_store", lambda: store(FakeChat(reply="not json")))
    record = read_model.check()
    assert (record["outcome"], record["reason"]) == ("fallback", "malformed")


def test_a_store_with_no_chat_model_is_unconfigured(monkeypatch):
    monkeypatch.setattr(opener, "open_store", lambda: store())
    record = read_model.check()
    assert (record["outcome"], record["model"]) == ("unconfigured", "")


def test_a_store_built_with_rewrite_off_says_so(monkeypatch):
    model = FakeChat()
    monkeypatch.setattr(opener, "open_store", lambda: store(model, query_rewrite=False))
    assert read_model.check()["outcome"] == "disabled"
    assert model.calls == []


def test_no_local_store_is_its_own_outcome(monkeypatch):
    """A hosted install: its server would rewrite on a key this machine cannot check."""
    monkeypatch.setattr(opener, "open_store", lambda: None)
    assert read_model.check()["outcome"] == "no_local_store"


def test_a_library_older_than_query_rewrite_is_unsupported(monkeypatch):
    class Older:
        def search(self, query, k=6):
            return []

    monkeypatch.setattr(opener, "open_store", lambda: Older())
    assert read_model.check()["outcome"] == "unsupported"


def test_a_check_that_raises_is_recorded_not_raised(monkeypatch):
    class Broken:
        def search(self, query, k=6, query_rewrite=True):
            raise OSError("disk")

    monkeypatch.setattr(opener, "open_store", lambda: Broken())
    record = read_model.check()
    assert (record["outcome"], record["reason"]) == ("error", "OSError")


# --- which keywords a backend is handed ------------------------------------------------


def test_a_backend_that_takes_query_rewrite_gets_both_kinds_of_read():
    class Takes:
        def recall(self, query, *, k=6, query_rewrite=True):
            return ""

    assert fast.read_kinds(Takes()) == ({"query_rewrite": False}, {"query_rewrite": True})


def test_a_library_store_is_asked_for_the_rewrite_outcome_too():
    """`with_ids=True` returns a `RecallResult`, whose `rewrite` says a key was rejected."""
    class Forwards:
        def recall(self, query, **kwargs):
            return ""

    both = ({"query_rewrite": False}, {"query_rewrite": True, "with_ids": True})
    assert fast.read_kinds(Forwards()) == both
    assert fast.read_kinds(store()) == both


def test_a_backend_that_predates_query_rewrite_is_asked_as_before():
    """Every released library before query rewrite, and the hooks' own hosted client."""
    class Older:
        def recall(self, query, k=6, budget=700):
            return ""

    class NoSignature:
        recall = len

    assert fast.read_kinds(Older()) == ({}, {})
    assert fast.read_kinds(HostedRecall("key")) == ({}, {})
    assert fast.read_kinds(NoSignature()) == ({}, {}), "a builtin with no signature"


def test_the_daemon_decides_once_at_startup(monkeypatch):
    import daemon as daemon_hook

    decided: list[object] = []
    real = daemon_hook.read_kinds

    def counting(backend):
        decided.append(backend)
        return real(backend)

    monkeypatch.setattr(daemon_hook, "read_kinds", counting)
    served = daemon_hook.Daemon("/tmp/unused-once.sock", Recorder())
    for rewrite in (False, True, False):
        served._answer({"q": "a", "k": 1, "budget": 50, "query_rewrite": rewrite})
    assert len(decided) == 1


def test_the_widening_retry_reuses_the_store_the_first_read_opened(monkeypatch, tmp_path):
    """One process, one handle: a second `open_store()` is a second store to open."""
    _no_daemon(monkeypatch, tmp_path)
    opened: list[Recorder] = []
    decided: list[object] = []
    real = fast.read_kinds

    def open_store():
        opened.append(Recorder())
        return opened[-1]

    def counting(backend):
        decided.append(backend)
        return real(backend)

    monkeypatch.setattr(opener, "open_store", open_store)
    monkeypatch.setattr(fast, "read_kinds", counting)
    fast.recall("q", query_rewrite=True, spawn=False)
    fast.recall("q", include_episodes=True, spawn=False)
    assert len(opened) == 1 and len(opened[0].calls) == 2
    assert len(decided) == 1


def test_a_different_opener_is_a_different_store(monkeypatch, tmp_path):
    _no_daemon(monkeypatch, tmp_path)
    first, second = Recorder(), Recorder()
    monkeypatch.setattr(opener, "open_store", lambda: first)
    fast.recall("q", spawn=False)
    monkeypatch.setattr(opener, "open_store", lambda: second)
    fast.recall("q", spawn=False)
    assert (len(first.calls), len(second.calls)) == (1, 1)


# --- the in-process route ----------------------------------------------------------------


def _no_daemon(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fast, "socket_path", lambda *a, **k: str(tmp_path / "absent.sock"))


def test_the_in_process_route_asks_for_a_plain_read_by_default(monkeypatch, tmp_path):
    _no_daemon(monkeypatch, tmp_path)
    model = FakeChat()
    monkeypatch.setattr(opener, "open_store", lambda: store(model))
    text, ok, _ = fast.recall("what did we decide", spawn=False)
    assert ok is True and "Friday" in text
    assert model.calls == [], "a store whose model can chat rewrites unless told not to"


def test_the_in_process_route_rewrites_when_asked(monkeypatch, tmp_path):
    _no_daemon(monkeypatch, tmp_path)
    model = FakeChat()
    monkeypatch.setattr(opener, "open_store", lambda: store(model))
    text, ok, _ = fast.recall("what did we decide", query_rewrite=True, spawn=False)
    assert ok is True and "Friday" in text
    assert len(model.calls) == 1


def test_a_failed_rewrite_still_serves_the_memories(monkeypatch, tmp_path):
    _no_daemon(monkeypatch, tmp_path)
    rejected = RuntimeError("403")
    rejected.status_code = 403  # type: ignore[attr-defined]
    monkeypatch.setattr(opener, "open_store", lambda: store(FakeChat(raises=rejected)))
    text, ok, _ = fast.recall("what did we decide", query_rewrite=True, spawn=False)
    assert ok is True and "Friday" in text


def test_a_rewrite_past_the_wait_is_abandoned_for_the_plain_read(monkeypatch, tmp_path):
    """The hook has 10 seconds in all and the model call may take 10 on its own."""
    _no_daemon(monkeypatch, tmp_path)
    logged: list[str] = []
    monkeypatch.setattr(fast, "log_line", lambda name, line: logged.append(line))
    gate = threading.Event()
    model = FakeChat(gate=gate)
    monkeypatch.setattr(opener, "open_store", lambda: store(model))
    try:
        text, ok, _ = fast.recall("what did we decide", query_rewrite=True,
                                  rewrite_wait=0.01, spawn=False)
    finally:
        gate.set()
    assert ok is True and "Friday" in text
    assert len(model.calls) == 1, "the plain read that was served made no second call"
    assert any("plain read" in line for line in logged), logged


def test_an_error_in_the_rewritten_read_is_reported_as_a_failed_recall(monkeypatch,
                                                                       tmp_path):
    _no_daemon(monkeypatch, tmp_path)

    class Broken:
        def recall(self, query, **kwargs):
            raise OSError("disk")

    monkeypatch.setattr(opener, "open_store", lambda: Broken())
    assert fast.recall("q", query_rewrite=True, spawn=False) == ("", False, "")


def test_an_older_library_is_asked_without_the_argument_even_when_rewrite_is_on(
        monkeypatch, tmp_path):
    _no_daemon(monkeypatch, tmp_path)

    class Older:
        def __init__(self):
            self.calls: list[dict] = []

        def recall(self, query, k=6, budget=700, header=None, min_score=0.0):
            self.calls.append({"k": k, "budget": budget})
            return "- a memory"

    older = Older()
    monkeypatch.setattr(opener, "open_store", lambda: older)
    for asked in (False, True):
        text, ok, _ = fast.recall("q", query_rewrite=asked, spawn=False)
        assert (text, ok) == ("- a memory", True)
    assert len(older.calls) == 2


# --- the daemon route --------------------------------------------------------------------


class Recorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def recall(self, query, **kwargs):
        self.calls.append(dict(kwargs))
        return f"- {query}"


def test_the_daemon_is_told_to_rewrite_only_when_the_hook_asks(monkeypatch, tmp_path):
    sent: list[tuple[dict, float]] = []

    def send(path, request, timeout):
        sent.append((request, timeout))
        return json.dumps({"ok": True, "text": "- served"})

    monkeypatch.setattr(fast, "socket_path", lambda *a, **k: str(tmp_path / "d.sock"))
    monkeypatch.setattr(fast, "send", send)
    assert fast.recall("q", spawn=False)[0] == "- served"
    assert fast.recall("q", query_rewrite=True, rewrite_wait=4.0, spawn=False)[0] == "- served"
    (plain, plain_wait), (rewrite, rewrite_wait) = sent
    assert "query_rewrite" not in plain and plain_wait == ipc.CLIENT_TIMEOUT_SEC
    assert rewrite["query_rewrite"] is True and rewrite_wait == 4.0


def test_the_daemon_hands_the_backend_the_kind_of_read_it_was_asked_for():
    import daemon as daemon_hook

    backend = Recorder()
    served = daemon_hook.Daemon("/tmp/unused-rewrite.sock", backend)
    assert served._answer({"q": "a", "k": 2, "budget": 100})["ok"] is True
    assert served._answer({"q": "a", "k": 2, "budget": 100, "query_rewrite": True})["ok"]
    assert [call["query_rewrite"] for call in backend.calls] == [False, True]


def test_a_daemon_that_timed_out_on_a_rewrite_is_not_asked_twice(monkeypatch, tmp_path):
    """The daemon is still making the model call. The fallback read must not make another."""
    clock = [0.0]

    def send(path, request, timeout):
        clock[0] += timeout
        return None

    backend = Recorder()
    monkeypatch.setattr(fast, "socket_path", lambda *a, **k: str(tmp_path / "d.sock"))
    monkeypatch.setattr(fast, "send", send)
    monkeypatch.setattr(fast, "_clock", lambda: clock[0])
    monkeypatch.setattr(opener, "open_store", lambda: backend)
    text, ok, _ = fast.recall("q", query_rewrite=True, rewrite_wait=4.0, spawn=False)
    assert (text, ok) == ("- q", True)
    assert backend.calls[0]["query_rewrite"] is False


def test_a_daemon_that_failed_a_rewrite_is_not_asked_twice(monkeypatch, tmp_path):
    backend = Recorder()
    monkeypatch.setattr(fast, "socket_path", lambda *a, **k: str(tmp_path / "d.sock"))
    monkeypatch.setattr(fast, "send", lambda path, request, timeout: '{"ok": false}')
    monkeypatch.setattr(opener, "open_store", lambda: backend)
    fast.recall("q", query_rewrite=True, spawn=False)
    assert backend.calls[0]["query_rewrite"] is False


def test_no_daemon_at_all_leaves_the_rewrite_to_the_in_process_route(monkeypatch, tmp_path):
    """A refused connection returns at once; only a daemon that was reached rules it out."""
    backend = Recorder()
    _no_daemon(monkeypatch, tmp_path)
    monkeypatch.setattr(opener, "open_store", lambda: backend)
    fast.recall("q", query_rewrite=True, spawn=False)
    assert backend.calls[0]["query_rewrite"] is True


def test_a_slow_rewrite_does_not_hold_up_another_clients_plain_read():
    """The model call runs outside the daemon's lock, so a second session is not kept
    waiting past its client timeout and sent to the slow route."""
    import time

    import daemon as daemon_hook

    started, release = threading.Event(), threading.Event()

    class Slow(Recorder):
        def recall(self, query, **kwargs):
            if kwargs.get("query_rewrite"):
                started.set()
                release.wait(timeout=5)
            return super().recall(query, **kwargs)

    served = daemon_hook.Daemon("/tmp/unused-lock.sock", Slow())
    rewrite = threading.Thread(target=served._answer, args=(
        {"q": "slow", "k": 1, "budget": 50, "query_rewrite": True},))
    rewrite.start()
    try:
        assert started.wait(timeout=5), "the rewrite never reached the store"
        began = time.monotonic()
        reply = served._answer({"q": "plain", "k": 1, "budget": 50})
        waited = time.monotonic() - began
    finally:
        release.set()
        rewrite.join(timeout=5)
    assert reply == {"ok": True, "text": "- plain"}
    assert waited < ipc.CLIENT_TIMEOUT_SEC, f"the plain read waited {waited:.1f}s"


@pytest.mark.parametrize("asked", [False, True])
def test_both_routes_ask_the_backend_the_same_thing(monkeypatch, tmp_path, asked):
    import daemon as daemon_hook

    direct, resident = Recorder(), Recorder()
    _no_daemon(monkeypatch, tmp_path)
    monkeypatch.setattr(opener, "open_store", lambda: direct)
    direct_text, ok, _ = fast.recall("q", k=3, budget=200, query_rewrite=asked, spawn=False)
    request = {"q": "q", "k": 3, "budget": 200}
    if asked:
        request["query_rewrite"] = True
    reply = daemon_hook.Daemon(str(tmp_path / "unused.sock"), resident)._answer(request)
    assert (reply["text"], reply["ok"]) == (direct_text, ok)
    assert resident.calls == direct.calls


# --- the hosted route --------------------------------------------------------------------


def hosted(monkeypatch, offers: "bool | None") -> "tuple[HostedRecall, list[dict]]":
    """A client whose probe answers `offers`: yes, no, or `None` for a probe that failed."""
    sent: list[dict] = []
    client = HostedRecall("key")

    def call(tool, args):
        sent.append(dict(args))
        return "- a memory"

    monkeypatch.setattr(client, "_call", call)
    monkeypatch.setattr(client, "_ensure_session", lambda: True)
    monkeypatch.setattr(client, "offers",
                        lambda tool, argument: offers if (tool, argument) == (
                            "memory_recall", "query_rewrite") else False)
    return client, sent


def test_the_hosted_client_asks_for_a_plain_read_when_the_server_offers_rewrite(monkeypatch):
    """The hosted server would rewrite on the organisation's key, which setup cannot check."""
    client, sent = hosted(monkeypatch, offers=True)
    assert client.recall("q").strip() == "- a memory"
    assert sent[0]["query_rewrite"] is False


def test_the_hosted_client_sends_nothing_to_a_server_without_rewrite(monkeypatch):
    client, sent = hosted(monkeypatch, offers=False)
    client.recall("q")
    assert "query_rewrite" not in sent[0]


def test_a_hosted_endpoint_with_no_session_is_asked_nothing_more(monkeypatch):
    """The probe for the argument needs a session, and so does the call. A failed
    handshake is reported once, not once per optional argument the retries drop."""
    from lib.hosted import HostedError

    client, sent = hosted(monkeypatch, offers=True)
    handshakes: list[bool] = []
    monkeypatch.setattr(client, "_ensure_session", lambda: handshakes.append(True) or False)
    with pytest.raises(HostedError, match="no session"):
        client.recall("q", min_score=0.29, include_episodes=True)
    assert (handshakes, sent) == ([True], [])


def test_a_failed_probe_still_asks_for_a_plain_read(monkeypatch):
    """A probe that fails says nothing about the server, so the opt-out is still sent."""
    client, sent = hosted(monkeypatch, offers=None)
    client.recall("q")
    assert sent[0]["query_rewrite"] is False


def test_a_server_that_refuses_the_opt_out_is_asked_again_without_it(monkeypatch):
    """Safe to drop: a server that does not know the argument cannot rewrite."""
    from lib.hosted import HostedError

    client, sent = hosted(monkeypatch, offers=None)

    def call(tool, args):
        sent.append(dict(args))
        if "query_rewrite" in args:
            raise HostedError("memory_recall: unknown argument(s) query_rewrite.")
        return "- a memory"

    monkeypatch.setattr(client, "_call", call)
    assert client.recall("q").strip() == "- a memory"
    assert ["query_rewrite" in a for a in sent] == [True, False]


def test_a_failure_that_is_not_about_the_opt_out_keeps_it(monkeypatch):
    """A transient failure must not buy a retry that lets the server rewrite."""
    from lib.hosted import HostedError

    client, sent = hosted(monkeypatch, offers=None)

    def call(tool, args):
        sent.append(dict(args))
        raise HostedError("upstream timed out")

    monkeypatch.setattr(client, "_call", call)
    with pytest.raises(HostedError, match="timed out"):
        client.recall("q", min_score=0.29)
    assert sent and all(a.get("query_rewrite") is False for a in sent)


def test_a_probe_that_fails_is_asked_again_on_the_next_call(monkeypatch):
    """A resident daemon lives for half an hour; one bad moment must not last that long."""
    client = HostedRecall("key")
    monkeypatch.setattr(client, "_ensure_session", lambda: True)
    tools = {"result": {"tools": [{"name": "memory_recall", "inputSchema": {
        "properties": {"query": {}, "query_rewrite": {}}}}]}}
    replies = iter([None, tools])
    probes: list[str] = []

    def rpc(method, params=None, retry=True):
        probes.append(method)
        return next(replies)

    monkeypatch.setattr(client, "_rpc", rpc)
    assert client.offers("memory_recall", "query_rewrite") is None
    assert client.accepts("memory_recall", "query_rewrite") is True
    assert client.offers("memory_recall", "query_rewrite") is True
    assert client.offers("memory_recall", "nothing") is False
    assert probes == ["tools/list", "tools/list"], "a probe that answered is kept"


def test_a_refused_probe_is_not_kept_either(monkeypatch):
    from lib.hosted import HostedError

    client = HostedRecall("key")
    monkeypatch.setattr(client, "_ensure_session", lambda: True)

    def rpc(method, params=None, retry=True):
        raise HostedError("503")

    monkeypatch.setattr(client, "_rpc", rpc)
    assert client.offers("memory_recall", "query_rewrite") is None
    assert client.accepts("memory_recall", "query_rewrite") is False
    assert client._schemas is None


# --- the hook itself ---------------------------------------------------------------------


def run_hook(monkeypatch, tmp_path, *, allowed: bool, block: str = "- a memory",
             decide=None) -> list[dict]:
    import recall as recall_hook

    asked: list[dict] = []

    def fake_recall(query, **kwargs):
        asked.append(kwargs)
        return block, True, ""

    monkeypatch.setattr(recall_hook, "SEEN_DIR", str(tmp_path / "recalled"))
    monkeypatch.setattr(recall_hook, "payload", lambda: {
        "session_id": "s1", "prompt": "what did we decide about the release", "cwd": ""})
    monkeypatch.setattr(recall_hook, "write", lambda host, reply: None)
    monkeypatch.setattr(recall_hook, "log_line", lambda *a, **k: None)
    monkeypatch.setattr(recall_hook, "due_capture_alert", lambda: "")
    monkeypatch.setattr(recall_hook, "due_alert_for_model", lambda: "")
    monkeypatch.setattr(recall_hook, "bind_project", lambda cwd: None)
    monkeypatch.setattr(recall_hook, "fast_recall", fake_recall)
    monkeypatch.setattr(recall_hook, "rewrite_allowed", decide or (lambda: allowed))
    monkeypatch.setattr(recall_hook, "_standing_refresh", lambda *a, **k: ("", None))
    assert recall_hook.main() == 0
    return asked


def test_the_hook_asks_for_a_plain_read_without_a_verified_key(monkeypatch, tmp_path):
    asked = run_hook(monkeypatch, tmp_path, allowed=False)
    assert [kw["query_rewrite"] for kw in asked] == [False]


def test_the_hook_asks_for_a_rewrite_with_a_verified_key(monkeypatch, tmp_path):
    import recall as recall_hook

    asked = run_hook(monkeypatch, tmp_path, allowed=True)
    assert [kw["query_rewrite"] for kw in asked] == [True]
    assert asked[0]["rewrite_wait"] == recall_hook.REWRITE_WAIT_SEC


def test_the_widening_retry_is_always_a_plain_read(monkeypatch, tmp_path):
    """It runs only on a thin prompt, and a second rewrite would be a second model call."""
    asked = run_hook(monkeypatch, tmp_path, allowed=True, block="")
    assert [kw["query_rewrite"] for kw in asked] == [True, False]


def test_no_rewrite_is_started_that_the_hooks_budget_cannot_wait_for(monkeypatch, tmp_path):
    import recall as recall_hook

    late = recall_hook.OVERALL_BUDGET_SEC - recall_hook.REWRITE_WAIT_SEC
    clock = [0.0]

    def monotonic_after_start():
        # The first reading is the hook's start; every later one is `late` seconds on.
        value = clock[0]
        clock[0] = late
        return value

    monkeypatch.setattr(recall_hook.time, "monotonic", monotonic_after_start)
    asked = run_hook(monkeypatch, tmp_path, allowed=True)
    assert [kw["query_rewrite"] for kw in asked] == [False]


def test_the_budget_is_checked_before_the_verification_is_read(monkeypatch, tmp_path):
    """The comparison is free; the decision reads files."""
    import recall as recall_hook

    def never():
        raise AssertionError("read the verification with no budget left to use it")

    late = recall_hook.OVERALL_BUDGET_SEC
    clock = [0.0]

    def monotonic_after_start():
        value = clock[0]
        clock[0] = late
        return value

    monkeypatch.setattr(recall_hook.time, "monotonic", monotonic_after_start)
    asked = run_hook(monkeypatch, tmp_path, allowed=True, decide=never)
    assert [kw["query_rewrite"] for kw in asked] == [False]


def test_the_wait_fits_inside_the_hooks_own_budget():
    import recall as recall_hook

    assert 0 < recall_hook.REWRITE_WAIT_SEC < recall_hook.OVERALL_BUDGET_SEC
    assert recall_hook.REWRITE_WAIT_SEC == fast.REWRITE_WAIT_SEC
