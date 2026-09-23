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

REWRITE_REPLY = json.dumps({"queries": ["release decision"], "date_range": None})


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Keep every test away from the real `~/.memvara`, the client's config and switches."""
    monkeypatch.setattr(settings, "SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(settings, "_LOADED", None)
    monkeypatch.setattr(read_model, "server_env", lambda: {})
    monkeypatch.setattr(fast, "log_line", lambda *a, **k: None)
    for name in list(os.environ):
        if name.startswith("MEMVARA_FEATURE_") or name in ("MEMVARA_LLM", "MEMVARA_LLM_MODEL"):
            monkeypatch.delenv(name)


def write_settings(data: dict) -> None:
    pathlib.Path(settings.SETTINGS).write_text(json.dumps(data), encoding="utf-8")
    settings._LOADED = None


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
    write_settings({"read_model": verified()})
    assert read_model.allowed() is True


def test_the_client_config_names_the_model_when_the_environment_does_not(monkeypatch):
    """The hook finds the model where the MCP server does: the client's server block."""
    monkeypatch.setattr(read_model, "server_env",
                        lambda: {"MEMVARA_LLM": "openai", "MEMVARA_LLM_MODEL": "m-2"})
    write_settings({"read_model": verified(backend="openai", model_setting="m-2")})
    assert read_model.configured() == ("openai", "m-2")
    assert read_model.allowed() is True
    monkeypatch.setenv("MEMVARA_LLM_MODEL", "m-3")
    assert read_model.configured() == ("openai", "m-3"), "the environment wins, as it does "
    "for the store the hook opens"
    assert read_model.allowed() is False


@pytest.mark.parametrize("switch", [{"query_rewrite": False}, {}])
def test_the_switch_off_means_a_plain_read_whatever_was_verified(monkeypatch, switch):
    monkeypatch.setenv("MEMVARA_LLM", "anthropic")
    write_settings({"read_model": verified(), **switch})
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
    write_settings({"read_model": record})
    assert read_model.allowed() is False


@pytest.mark.parametrize("backend, model", [("openai", ""), ("anthropic", "another-model")])
def test_a_different_model_from_the_one_checked_means_a_plain_read(monkeypatch, backend,
                                                                   model):
    """Setup showed the cost of one model. A change of model needs a new check."""
    monkeypatch.setenv("MEMVARA_LLM", backend)
    if model:
        monkeypatch.setenv("MEMVARA_LLM_MODEL", model)
    write_settings({"read_model": verified()})
    assert read_model.allowed() is False


# --- the check setup makes ---------------------------------------------------------------


def test_the_check_makes_one_rewrite_call_and_records_that_it_answered(monkeypatch):
    monkeypatch.setenv("MEMVARA_LLM", "Anthropic ")
    model = FakeChat()
    closed: list[bool] = []
    made = store(model)
    monkeypatch.setattr(made, "close", lambda: closed.append(True))
    monkeypatch.setattr(opener, "open_store", lambda: made)
    record = read_model.check()
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


# --- which keyword a backend is handed ---------------------------------------------------


def test_a_backend_that_takes_query_rewrite_is_told_which_kind_of_read():
    class Takes:
        def recall(self, query, *, k=6, query_rewrite=True):
            return ""

    class Forwards:
        def recall(self, query, **kwargs):
            return ""

    for backend in (Takes(), Forwards()):
        assert fast.rewrite_kwargs(backend.recall, True) == {"query_rewrite": True}
        assert fast.rewrite_kwargs(backend.recall, False) == {"query_rewrite": False}


def test_a_backend_that_predates_query_rewrite_is_asked_as_before():
    """Every released library before query rewrite, and the hooks' own hosted client."""
    class Older:
        def recall(self, query, k=6, budget=700):
            return ""

    assert fast.rewrite_kwargs(Older().recall, True) == {}
    assert fast.rewrite_kwargs(HostedRecall("key").recall, False) == {}
    assert fast.rewrite_kwargs(len, False) == {}, "a builtin with no signature"


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


def hosted(monkeypatch, offers: bool) -> "tuple[HostedRecall, list[dict]]":
    sent: list[dict] = []
    client = HostedRecall("key")

    def call(tool, args):
        sent.append(dict(args))
        return "- a memory"

    monkeypatch.setattr(client, "_call", call)
    monkeypatch.setattr(client, "_ensure_session", lambda: True)
    monkeypatch.setattr(client, "accepts",
                        lambda tool, argument: offers and (tool, argument) == (
                            "memory_recall", "query_rewrite"))
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


# --- the hook itself ---------------------------------------------------------------------


def run_hook(monkeypatch, tmp_path, *, allowed: bool, block: str = "- a memory") -> list[dict]:
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
    monkeypatch.setattr(recall_hook, "rewrite_allowed", lambda: allowed)
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


def test_the_wait_fits_inside_the_hooks_own_budget():
    import recall as recall_hook

    assert 0 < recall_hook.REWRITE_WAIT_SEC < recall_hook.OVERALL_BUDGET_SEC
    assert recall_hook.REWRITE_WAIT_SEC == fast.REWRITE_WAIT_SEC
