"""Framework adapters: LangChain, LlamaIndex, CrewAI.

**None of the three frameworks is installed here, and none is needed.** That is not a
convenience — it is the property under test. `import memvara` must keep working with
numpy alone, so every framework import in `memvara/integrations/**` happens inside a
function, and the way to prove that is a suite that exercises the whole surface with
nothing installed. The fakes below go into `sys.modules` under the real import paths, so
the adapters' own `require()` and `__getattr__` run for real against them rather than
being monkeypatched out of the way.

The fake base classes mirror interfaces read out of the real packages — **langchain-core
1.5.3, llama-index-core 0.14.23, crewai 1.15.13** — rather than remembered from a
tutorial. Two runtime bugs came out of driving the real ones and both are pinned here as
tests, because a fake that agrees with the adapter is worth nothing:

* `langchain_core` 1.x returns a *callable string* from `BaseMessage.text`, so the
  obvious `callable(...)` probe calls it and earns a deprecation warning per turn.
* `crewai.memory.types.embed_text` guards its result with `if not result:`, which is a
  `ValueError` on the ndarray an `Embedder` returns.

What most of these tests assert is not "it works" but **what refuses, and what is lost**.
The three interfaces model memory as a message list or a vector store; memvara is neither,
and the adapters' value is that they say so at the call site rather than months later.
"""

from __future__ import annotations

import asyncio
import sys
import warnings
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from memvara import Memvara, HashingEmbedder, NullLLM, Scope
from memvara.compat import NOTE_PREDICATE, note_subject
from memvara.integrations import IntegrationError
from memvara.integrations import _common
from memvara.integrations import crewai as ca
from memvara.integrations import langchain as lc
from memvara.integrations import llamaindex as li

TZ = timezone.utc
T0 = datetime(2024, 3, 1, tzinfo=TZ)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def mem():
    # NullLLM by name, not by default: the default warns about degraded extraction, and
    # a suite that trips that warning teaches everyone to filter the category.
    m = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice")
    yield m
    m.close()


@pytest.fixture(autouse=True)
def _fresh_module_state():
    """Clear the adapters' per-process caches between tests.

    Every lazily-composed class and every lazily-imported module is memoized, which is
    right in a process and wrong in a suite: one test's fake base would otherwise decide
    what a later test sees, and the once-per-process transcript warning would fire for
    exactly one test in whatever order pytest picked.
    """
    caches = (lc._message_classes, lc._history_class, lc._retriever_class,
              li._block_class, li._retriever_class)
    for cache in caches:
        cache.cache_clear()
    lc._WARNED_TRANSCRIPT = False
    yield
    for cache in caches:
        cache.cache_clear()


# =====================================================================================
# Fake frameworks. Each mirrors the real base class's contract closely enough that a
# subclass which satisfies the fake satisfies the real one — the signatures were read
# out of the installed packages, and the real ones are exercised separately by hand.
# =====================================================================================

# --- langchain_core ------------------------------------------------------------------


class FakeBaseChatMessageHistory:
    """`langchain_core.chat_history.BaseChatMessageHistory`.

    The real one is a plain ABC whose only abstract method is `clear`, with
    `add_message` delegating to `add_messages` when a subclass overrides it. Both are
    reproduced because both are contract: a subclass that broke the delegation would
    make `add_user_message` raise, and nothing else here would notice.
    """

    def add_message(self, message):
        self.add_messages([message])

    def add_user_message(self, content):
        self.add_message(FakeMessage(content, "human"))

    def clear(self):  # pragma: no cover - overridden by the adapter
        raise NotImplementedError


class FakeMessage:
    def __init__(self, content, kind, **extra):
        self.content = content
        self.type = kind
        for key, value in extra.items():
            setattr(self, key, value)

    @property
    def text(self):
        """A *callable string*, exactly as langchain-core 1.x returns.

        `str(...)` with a `__call__` bolted on is what the real back-compat shim is, and
        it is the reason `_text_of` cannot use a bare `callable()` probe.
        """
        return _CallableStr(self.content)


class _CallableStr(str):
    def __call__(self):  # pragma: no cover - calling it is the bug, so nothing should
        return str(self)


def fake_messages_module():
    def maker(kind, *fields):
        def build(content="", **kw):
            return FakeMessage(content, kind, **{f: kw.get(f) for f in fields})
        return build

    return SimpleNamespace(
        HumanMessage=maker("human"),
        AIMessage=maker("ai"),
        SystemMessage=maker("system"),
        ToolMessage=lambda content="", tool_call_id="": FakeMessage(
            content, "tool", tool_call_id=tool_call_id),
        ChatMessage=lambda content="", role="": FakeMessage(content, "chat", role=role),
    )


class FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeBaseRetriever:
    """`langchain_core.retrievers.BaseRetriever`, minus pydantic.

    The real base is a pydantic `RunnableSerializable`, so it takes keyword arguments
    and assigns declared fields. Assigning whatever it is given is the part that matters
    for the adapter; validation is pydantic's job and is not what is under test.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def invoke(self, query, config=None, **kwargs):
        return self._get_relevant_documents(query, run_manager=None)


def install_langchain(monkeypatch):
    """Put the fakes on the real import paths, so `require()` finds them."""
    monkeypatch.setitem(sys.modules, "langchain_core.chat_history",
                        SimpleNamespace(BaseChatMessageHistory=FakeBaseChatMessageHistory))
    monkeypatch.setitem(sys.modules, "langchain_core.messages", fake_messages_module())
    monkeypatch.setitem(sys.modules, "langchain_core.documents",
                        SimpleNamespace(Document=FakeDocument))
    monkeypatch.setitem(sys.modules, "langchain_core.retrievers",
                        SimpleNamespace(BaseRetriever=FakeBaseRetriever))


# --- llama_index.core ----------------------------------------------------------------


class FakeChatMessage:
    def __init__(self, content=None, role="user"):
        self.content = content
        self.role = SimpleNamespace(value=role)


class FakeBaseMemoryBlock:
    """`llama_index.core.memory.BaseMemoryBlock`, minus pydantic.

    `aget`/`aput` are the public wrappers the framework calls; `_aget`/`_aput` are what
    a block implements. `aput`'s short-term-memory gate is reproduced because it is the
    only behaviour of the base that a block can be caught out by.
    """

    accept_short_term_memory = True

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    async def aget(self, messages=None, **block_kwargs):
        return await self._aget(messages, **block_kwargs)

    async def aput(self, messages, from_short_term_memory=False, session_id=None):
        if from_short_term_memory and not self.accept_short_term_memory:
            return
        await self._aput(messages)


class FakeLIBaseRetriever:
    """`llama_index.core.retrievers.BaseRetriever` — a plain ABC with a real `__init__`."""

    def __init__(self, callback_manager=None, verbose=False):
        self.callback_manager = callback_manager
        self.verbose = verbose

    def retrieve(self, str_or_query_bundle):
        return self._retrieve(str_or_query_bundle)


class FakeTextNode:
    def __init__(self, text="", id_=None, metadata=None):
        self.text = text
        self.id_ = id_
        self.metadata = metadata or {}


class FakeNodeWithScore:
    def __init__(self, node, score):
        self.node = node
        self.score = score


def install_llamaindex(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_index.core.memory",
                        SimpleNamespace(BaseMemoryBlock=FakeBaseMemoryBlock))
    monkeypatch.setitem(sys.modules, "llama_index.core.retrievers",
                        SimpleNamespace(BaseRetriever=FakeLIBaseRetriever))
    monkeypatch.setitem(sys.modules, "llama_index.core.schema",
                        SimpleNamespace(NodeWithScore=FakeNodeWithScore,
                                        TextNode=FakeTextNode))


# --- crewai --------------------------------------------------------------------------


class FakeMemoryRecord:
    """`crewai.memory.types.MemoryRecord`. Field names and defaults are the real ones."""

    _n = 0

    def __init__(self, content, id=None, scope="/", categories=None, metadata=None,
                 importance=0.5, created_at=None, last_accessed=None, embedding=None,
                 source=None, private=False):
        FakeMemoryRecord._n += 1
        self.id = id or f"rec{FakeMemoryRecord._n}"
        self.content = content
        self.scope = scope
        self.categories = list(categories or [])
        self.metadata = dict(metadata or {})
        self.importance = importance
        # Naive, as CrewAI's own `datetime.utcnow()` default is — see `_blob`. Stepping
        # rather than fixed, because the real default is a clock read and two records
        # created in sequence do not share an instant.
        self.created_at = created_at or (datetime(2024, 3, 1, 12, 0, 0)
                                         + timedelta(seconds=FakeMemoryRecord._n))
        self.last_accessed = last_accessed or self.created_at
        self.embedding = embedding
        self.source = source
        self.private = private


class FakeScopeInfo:
    def __init__(self, path, record_count=0, categories=None, oldest_record=None,
                 newest_record=None, child_scopes=None):
        self.path = path
        self.record_count = record_count
        self.categories = list(categories or [])
        self.oldest_record = oldest_record
        self.newest_record = newest_record
        self.child_scopes = list(child_scopes or [])


CREWAI_TYPES = SimpleNamespace(MemoryRecord=FakeMemoryRecord, ScopeInfo=FakeScopeInfo)


def crew_embed(storage, text):
    """`crewai.memory.types.embed_text`, reproduced including the guard that bit us."""
    result = storage.embedder([text])
    if not result:                          # a ValueError if `embedder` returns ndarray
        return []
    first = result[0]
    if hasattr(first, "tolist"):
        return list(first.tolist())
    return [float(x) for x in first]


# =====================================================================================
# Shared machinery
# =====================================================================================

def test_require_returns_the_named_attributes_when_the_module_is_there():
    (claim,) = _common.require("memvara.types", "Claim", extra="x", needs="y")
    from memvara.types import Claim
    assert claim is Claim


def test_a_missing_framework_names_the_extra_and_says_memvara_does_not_need_it():
    """The error a user hits first. It has to say what to install *and* that the absence
    is an adapter's problem rather than a broken memvara install."""
    with pytest.raises(ImportError, match=r"memvara\[langchain\]"):
        _common.require("no_such_framework_pkg", "Thing", extra="langchain",
                        needs="langchain-core>=0.3")


def test_a_present_package_missing_the_class_is_reported_as_version_skew(monkeypatch):
    """Not as "not installed". Telling someone to reinstall a package they already have
    is how an afternoon disappears."""
    monkeypatch.setitem(sys.modules, "pretend_fw", SimpleNamespace())
    with pytest.raises(ImportError, match="version skew"):
        _common.require("pretend_fw", "Missing", extra="x", needs="pretend-fw>=1.0")


def test_bind_takes_the_memvaras_own_scope_when_nothing_overrides_it(mem):
    memvara, scope = _common.bind(mem)
    assert memvara is mem
    assert scope == Scope("default", "alice", None, None)


def test_bind_narrows_a_scoped_memvara_without_widening_it(mem):
    """A server layer holds a `ScopedMemvara` per request and has no public way back to
    the `Memvara`. Accepting one here is the difference between usable and not."""
    memvara, scope = _common.bind(mem.scope(user="bob"), session="s1")
    assert memvara is mem
    assert scope == Scope("default", "bob", None, "s1")


def test_scope_kw_is_exactly_what_every_memvara_method_takes(mem):
    assert _common.scope_kw(Scope("t", "u", "a", "s")) == {
        "tenant": "t", "user": "u", "agent": "a", "session": "s"}
    mem.remember("user", "lives_in", "Berlin", **_common.scope_kw(Scope("t", "u")))
    assert mem.count(tenant="t", user="u") == 1


def test_result_metadata_carries_both_time_axes_not_one_timestamp(mem):
    """The whole reason the adapters bother with a metadata dict. A `Document` or a
    `TextNode` is a string plus a mapping, and if the mapping loses valid-time then
    memvara has been reduced to a slower vector store on the way out."""
    mem.remember("user", "lives_in", "Berlin", valid_from=T0, recorded_at=T0)
    meta = _common.result_metadata(mem.search("where do they live?")[0])
    assert meta["kind"] == "claim"
    assert (meta["subject"], meta["predicate"], meta["object"]) == (
        "user", "lives_in", "Berlin")
    assert meta["valid_from"] == T0.isoformat() and meta["valid_to"] is None
    assert meta["recorded_at"] == T0.isoformat() and meta["invalidated_at"] is None
    assert "recency=" in meta["why"]


def test_a_retired_value_still_reports_when_belief_in_it_ended(mem):
    mem.remember("user", "lives_in", "Berlin", valid_from=T0, recorded_at=T0)
    mem.remember("user", "lives_in", "Lisbon", valid_from=T0 + timedelta(days=30),
                 recorded_at=T0 + timedelta(days=30))
    old = [c for c in mem.history("user", "lives_in") if c.object == "Berlin"][0]
    meta = _common.result_metadata(
        [r for r in mem.search("where do they live?", include_invalidated=True)
         if r.claim.id == old.id][0])
    assert meta["invalidated_at"] is not None and meta["valid_to"] is not None


def test_an_episode_result_gets_no_predicate_so_a_turn_cannot_pass_as_a_fact(mem):
    """A filter keyed on `metadata["predicate"]` must never match a verbatim remark."""
    mem.add("I have been thinking about moving to Lisbon")
    episodes = [r for r in mem.search("Lisbon", include_episodes=True)
                if r.kind == "episode"]
    meta = _common.result_metadata(episodes[0])
    assert meta["kind"] == "episode" and "predicate" not in meta
    assert meta["role"] == "user" and meta["memvara_id"].startswith("ep_")


def test_every_adapter_error_is_catchable_with_one_clause():
    """An application wiring two frameworks should not need two except clauses, and
    `except NotImplementedError` around a migration shim should catch all of them."""
    for error in (lc.LangChainCompatError, li.LlamaIndexCompatError,
                  ca.CrewAICompatError):
        assert issubclass(error, IntegrationError)
        assert issubclass(error, NotImplementedError)


# =====================================================================================
# LangChain — chat message history
# =====================================================================================

@pytest.fixture()
def history(mem, monkeypatch):
    install_langchain(monkeypatch)
    return lc.MemvaraChatMessageHistory(mem, session="s1", transcript_warning=False)


def test_the_history_is_a_real_basechatmessagehistory(history):
    """Subclassed rather than duck-typed, so `add_user_message` and the async defaults
    come from the base and cannot drift from it."""
    assert isinstance(history, FakeBaseChatMessageHistory)
    assert type(history).__name__ == "MemvaraChatMessageHistory"


def test_the_composed_class_is_minted_once_so_isinstance_stays_stable(monkeypatch):
    install_langchain(monkeypatch)
    assert lc.MemvaraChatMessageHistory is lc.MemvaraChatMessageHistory
    assert lc.MemvaraRetriever is lc.MemvaraRetriever


def test_messages_round_trip_role_and_text_through_the_store(history):
    messages = fake_messages_module()
    history.add_messages([
        messages.HumanMessage(content="I live in Berlin"),
        messages.AIMessage(content="Noted."),
        messages.SystemMessage(content="be brief"),
        messages.ToolMessage(content="{}", tool_call_id="tc1"),
    ])
    got = history.messages
    assert [(m.type, m.content) for m in got] == [
        ("human", "I live in Berlin"), ("ai", "Noted."),
        ("system", "be brief"), ("tool", "{}")]
    assert got[3].tool_call_id == "tc1"


def test_messages_written_in_one_batch_come_back_in_the_order_they_were_sent(history):
    """The ordering hazard that is invisible on a fast clock: four turns written in one
    call can land on one tick — 15 ms wide on Windows — and a transcript reassembled in
    uuid order is a chain whose prompt changes without an input changing."""
    messages = fake_messages_module()
    history.add_messages([messages.HumanMessage(content=f"turn {i}") for i in range(8)])
    assert [m.content for m in history.messages] == [f"turn {i}" for i in range(8)]


def test_an_exactly_repeated_turn_is_stored_once_so_the_transcript_is_not_a_log(history):
    """Memvara's tier-0 dedupe is keyed on `(scope, role, content)`, so a user who says
    "ok" twice in one session appears once. That is the right design for a memory store
    — re-ingesting a transcript has to be cheap and idempotent — and it means
    `messages` cannot be sold as verbatim replay. Pinned here so the day it changes,
    the warning text and the docstring are forced to change with it."""
    messages = fake_messages_module()
    for content in ("ok", "sure", "ok"):
        history.add_messages([messages.HumanMessage(content=content)])
    assert [m.content for m in history.messages] == ["ok", "sure"]


def test_the_transcript_warning_names_both_ways_the_list_is_smaller_than_it_looks(
        mem, monkeypatch):
    install_langchain(monkeypatch)
    with pytest.warns(lc.MemvaraTranscriptWarning) as caught:
        lc.MemvaraChatMessageHistory(mem, session="s1").messages
    text = str(caught[0].message)
    assert "not the memory" in text and "deduplicates" in text


def test_a_turn_this_adapter_did_not_write_still_appears_in_the_transcript(mem, history):
    """`mem.add()` from a cron job, the MCP server or another adapter is part of what was
    said. A history that only showed its own writes would be a private buffer."""
    mem.add("I work at Acme", session="s1")
    assert [m.type for m in history.messages] == ["human"]


def test_a_chat_message_keeps_its_own_role_rather_than_being_folded_into_a_neighbour(
        history):
    messages = fake_messages_module()
    history.add_messages([messages.ChatMessage(content="hmm", role="critic")])
    got = history.messages[0]
    assert (got.type, got.role) == ("chat", "critic")


def test_an_empty_batch_costs_no_write(history):
    history.add_messages([])
    assert history.last_receipt is None


def test_the_write_receipt_is_kept_because_the_interface_returns_none(history):
    """`llm_calls` is the number this library exists to drive down, and `add_messages`
    is typed to return `None`. Dropping the receipt would make the cost unobservable
    from inside a chain."""
    history.add_messages([fake_messages_module().HumanMessage(content="I live in Berlin")])
    assert history.last_receipt.llm_calls == 0
    assert len(history.last_receipt.added) == 1


def test_the_transcript_is_scoped_to_this_session_and_reaches_nothing_sideways(
        mem, monkeypatch):
    install_langchain(monkeypatch)
    one = lc.MemvaraChatMessageHistory(mem, session="s1", transcript_warning=False)
    two = lc.MemvaraChatMessageHistory(mem, session="s2", transcript_warning=False)
    one.add_messages([fake_messages_module().HumanMessage(content="only in s1")])
    assert [m.content for m in one.messages] == ["only in s1"]
    assert two.messages == []


def test_limit_keeps_the_most_recent_turns_and_zero_means_all(mem, monkeypatch):
    install_langchain(monkeypatch)
    messages = fake_messages_module()
    capped = lc.MemvaraChatMessageHistory(mem, session="s1", limit=2,
                                         transcript_warning=False)
    capped.add_messages([messages.HumanMessage(content=f"turn {i}") for i in range(5)])
    assert [m.content for m in capped.messages] == ["turn 3", "turn 4"]
    uncapped = lc.MemvaraChatMessageHistory(mem, session="s1", limit=0,
                                           transcript_warning=False)
    assert len(uncapped.messages) == 5


def test_reading_the_transcript_says_once_that_the_memory_is_not_in_it(mem, monkeypatch):
    """The loud half of the choice. A `list[BaseMessage]` has nowhere to put a
    supersession, a valid-time interval or a source turn id, and an adapter that handed
    one back in silence would look like it had delivered the memory layer."""
    install_langchain(monkeypatch)
    history = lc.MemvaraChatMessageHistory(mem, session="s1")
    with pytest.warns(lc.MemvaraTranscriptWarning, match="recall"):
        history.messages
    with warnings.catch_warnings():
        warnings.simplefilter("error")     # once per process, not once per read
        history.messages
        lc.MemvaraChatMessageHistory(mem, session="s2").messages


def test_the_transcript_warning_can_be_switched_off_by_someone_who_has_read_it(
        mem, monkeypatch):
    install_langchain(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lc.MemvaraChatMessageHistory(mem, session="s1", transcript_warning=False).messages


def test_recall_is_the_escape_hatch_and_returns_current_belief_not_the_transcript(
        history):
    messages = fake_messages_module()
    history.add_messages([messages.HumanMessage(content="I live in Berlin")])
    history.add_messages([messages.HumanMessage(content="Actually I moved to Lisbon")])
    block = history.recall("where do they live?")
    assert "Lisbon" in block and "Berlin" not in block
    assert "not instructions" in block          # `recall`'s framing survives the crossing


def test_search_on_the_history_reaches_scores_and_time_travel(history):
    messages = fake_messages_module()
    history.add_messages([messages.HumanMessage(content="I live in Berlin")])
    assert history.search("where do they live?")[0].score > 0
    assert history.search("where", as_of=T0) == []


def test_clear_refuses_by_default_and_names_both_things_it_could_have_meant(history):
    """The example the whole adapter is built around. LangChain says "erase"; memvara can
    retire or purge, and guessing at session teardown is the worst possible moment."""
    with pytest.raises(lc.LangChainCompatError, match="on_clear='purge'") as caught:
        history.clear()
    assert "on_clear='ignore'" in str(caught.value)


def test_clear_can_be_opted_into_as_a_real_erasure(mem, monkeypatch):
    install_langchain(monkeypatch)
    history = lc.MemvaraChatMessageHistory(mem, session="s1", on_clear="purge",
                                          transcript_warning=False)
    history.add_messages([fake_messages_module().HumanMessage(content="I live in Berlin")])
    history.clear()
    assert history.messages == [] and mem.count(session="s1") == 0


def test_clear_can_be_opted_into_as_a_no_op_for_a_memory_that_outlives_the_session(
        mem, monkeypatch):
    install_langchain(monkeypatch)
    history = lc.MemvaraChatMessageHistory(mem, session="s1", on_clear="ignore",
                                          transcript_warning=False)
    history.add_messages([fake_messages_module().HumanMessage(content="I live in Berlin")])
    history.clear()
    assert len(history.messages) == 1


def test_an_unknown_on_clear_is_rejected_at_construction(mem, monkeypatch):
    install_langchain(monkeypatch)
    with pytest.raises(ValueError, match="on_clear='wipe'"):
        lc.MemvaraChatMessageHistory(mem, session="s1", on_clear="wipe")


def test_the_history_repr_names_the_scope_and_the_deletion_policy(history):
    assert "default/alice/*/s1" in repr(history) and "on_clear=error" in repr(history)


def test_the_base_classs_add_message_path_still_works(history):
    """`add_user_message` goes through `BaseChatMessageHistory.add_message`, which only
    delegates when the subclass overrides `add_messages`. Nothing else here would catch
    that regression."""
    history.add_user_message("I live in Berlin")
    assert [m.content for m in history.messages] == ["I live in Berlin"]


# --- text and role extraction --------------------------------------------------------

def test_a_callable_string_from_langchain_1x_is_read_as_a_string(recwarn):
    """The measured bug. `BaseMessage.text` is a property returning a `str` subclass with
    `__call__`, so a bare `callable()` probe calls it and earns a
    `LangChainDeprecationWarning` on every single turn."""
    assert lc._text_of(FakeMessage("hello", "human")) == "hello"
    assert not recwarn.list


def test_a_message_that_only_has_text_as_a_method_still_works():
    """langchain-core before 1.0. Supporting both is three lines and stops the adapter
    pinning a minor version."""
    assert lc._text_of(SimpleNamespace(text=lambda: "old style")) == "old style"


@pytest.mark.parametrize("message, expected", [
    (SimpleNamespace(content="plain"), "plain"),
    (SimpleNamespace(content=[{"type": "text", "text": "a"}]),
     "[{'type': 'text', 'text': 'a'}]"),
])
def test_a_message_with_no_text_attribute_falls_back_to_content(message, expected):
    assert lc._text_of(message) == expected


@pytest.mark.parametrize("message, role", [
    (FakeMessage("x", "human"), "user"),
    (FakeMessage("x", "ai"), "assistant"),
    (FakeMessage("x", "function"), "tool"),
    (FakeMessage("x", "chat", role="critic"), "critic"),
    (FakeMessage("x", "something_new"), "something_new"),
    (SimpleNamespace(), "user"),
])
def test_roles_map_across_and_unknown_ones_are_kept_rather_than_folded(message, role):
    assert lc._role_of(message) == role


def test_only_non_content_fields_are_copied_into_meta():
    """Storing the whole serialized message would put the text on disk twice, and a
    redaction or erasure pass only has to miss the second copy once."""
    assert lc._write_meta(FakeMessage("secret", "human")) == {"lc_type": "human"}
    assert lc._write_meta(FakeMessage("{}", "tool", tool_call_id="tc1")) == {
        "lc_type": "tool", "tool_call_id": "tc1"}


def test_a_stored_turn_with_no_adapter_metadata_falls_back_to_its_role(monkeypatch):
    install_langchain(monkeypatch)
    from memvara.types import Episode
    classes = lc._message_classes()
    assert lc._to_message(Episode(content="hi", role="assistant"), classes).type == "ai"
    assert lc._to_message(Episode(content="hi", role="oracle"), classes).type == "chat"


def test_the_message_classes_are_imported_lazily_and_say_so_when_absent():
    with pytest.raises(ImportError, match=r"memvara\[langchain\]"):
        lc._message_classes()


# =====================================================================================
# LangChain — retriever
# =====================================================================================

@pytest.fixture()
def retriever(mem, monkeypatch):
    install_langchain(monkeypatch)
    mem.remember("user", "lives_in", "Berlin", valid_from=T0, recorded_at=T0)
    mem.remember("user", "lives_in", "Lisbon", valid_from=T0 + timedelta(days=30),
                 recorded_at=T0 + timedelta(days=30))
    return lc.MemvaraRetriever(memory=mem, user="alice", k=5)


def test_the_retriever_returns_documents_carrying_the_whole_claim(retriever):
    """The clean fit. Nothing about a claim is lost on the way into LangChain — the
    triple, both time axes, the ranking explanation and the source turn ids all ride in
    `metadata`, which is the only place the interface leaves for them."""
    docs = retriever.invoke("where do they live?")
    assert docs[0].page_content == "user lives in Lisbon"
    meta = docs[0].metadata
    assert meta["predicate"] == "lives_in" and meta["object"] == "Lisbon"
    assert meta["valid_from"] == (T0 + timedelta(days=30)).isoformat()
    assert meta["score"] == docs[0].metadata["score"] > 0
    assert isinstance(meta["sources"], list)


def test_the_retired_value_is_simply_absent_rather_than_ranked_below(retriever):
    """Contradiction resolution survives the adapter because it happened before it: the
    old value is retired in the store, so there is nothing for a chain to filter."""
    assert [d.page_content for d in retriever.invoke("where do they live?")] == [
        "user lives in Lisbon"]


def test_time_travel_survives_an_interface_that_never_imagined_it(mem, monkeypatch):
    """`as_of` is a property of the query, not of the response shape, so a LangChain
    retriever can answer "what did we believe in March" with no extension to
    `BaseRetriever` at all."""
    install_langchain(monkeypatch)
    mem.remember("user", "lives_in", "Berlin", valid_from=T0, recorded_at=T0)
    mem.remember("user", "lives_in", "Lisbon", valid_from=T0 + timedelta(days=30),
                 recorded_at=T0 + timedelta(days=30))
    past = lc.MemvaraRetriever(memory=mem, user="alice",
                              as_of=T0 + timedelta(days=10))
    assert [d.page_content for d in past.invoke("where do they live?")] == [
        "user lives in Berlin"]


def test_episodes_come_back_labelled_so_a_remark_cannot_pass_as_a_fact(mem, monkeypatch):
    install_langchain(monkeypatch)
    mem.add("I have been thinking about moving to Lisbon")
    docs = lc.MemvaraRetriever(memory=mem, user="alice", k=6,
                              include_episodes=True).invoke("Lisbon")
    assert {d.metadata["kind"] for d in docs} == {"episode"}
    assert all("predicate" not in d.metadata for d in docs)


def test_the_retriever_declares_run_manager_so_langchain_uses_the_modern_path(monkeypatch):
    """`BaseRetriever.__init_subclass__` reads the signature to decide. A subclass that
    omitted `run_manager` would silently get the legacy calling convention and a slower
    async path — it would still pass every functional test above."""
    import inspect
    install_langchain(monkeypatch)
    parameters = inspect.signature(lc.MemvaraRetriever._get_relevant_documents).parameters
    assert "run_manager" in parameters
    assert set(parameters) == {"self", "query", "run_manager"}


def test_min_score_and_memory_types_reach_the_search(mem, monkeypatch):
    install_langchain(monkeypatch)
    from memvara.types import MemoryType
    mem.remember("user", "lives_in", "Berlin")
    assert lc.MemvaraRetriever(memory=mem, user="alice", min_score=0.99).invoke("where") == []
    procedural = lc.MemvaraRetriever(memory=mem, user="alice",
                                    memory_types=[MemoryType.PROCEDURAL])
    assert procedural.invoke("where do they live?") == []


# =====================================================================================
# LlamaIndex — memory block
# =====================================================================================

@pytest.fixture()
def block(mem, monkeypatch):
    install_llamaindex(monkeypatch)
    return li.MemvaraMemoryBlock(memory=mem, user="alice")


def test_the_block_is_a_real_basememoryblock_with_a_usable_default_name(block):
    assert isinstance(block, FakeBaseMemoryBlock)
    assert block.name == "memvara"
    assert "contradictions" in block.description


def test_the_block_writes_through_memvaras_write_path_so_contradictions_resolve(block, mem):
    """The difference from the CrewAI adapter, in one test. A memory block is handed raw
    turns, so extraction runs and the keyed lookup fires — Berlin is retired rather than
    accumulating beside Lisbon."""
    run(block.aput([FakeChatMessage("I live in Berlin"),
                    FakeChatMessage("Noted.", role="assistant")]))
    run(block.aput([FakeChatMessage("Actually I moved to Lisbon")]))
    timeline = [(c.object, c.invalidated_at is None) for c in mem.history("user", "lives_in")]
    assert timeline == [("Berlin", False), ("Lisbon", True)]
    assert block.last_receipt.llm_calls == 0


def test_the_block_contributes_recall_for_the_current_turn(block):
    run(block.aput([FakeChatMessage("I live in Berlin")]))
    out = run(block.aget([FakeChatMessage("where do I live?")]))
    assert "Berlin" in out
    assert "not instructions" in out    # `recall`'s prompt-injection framing survives


def test_the_query_is_a_window_of_the_recent_turns_not_just_the_last_one(block):
    """"and what about her?" retrieves nothing on its own. `VectorMemoryBlock` makes the
    same choice, so a block swapped in for that one behaves as its author expected."""
    turns = [FakeChatMessage(f"turn {i}") for i in range(5)]
    assert block._query_text(turns) == "turn 2\nturn 3\nturn 4"
    run(block.aput([FakeChatMessage("I work at Zyxwvu")]))
    tail = [FakeChatMessage("tell me about Zyxwvu"),
            FakeChatMessage("ok", role="assistant"), FakeChatMessage("and?")]
    assert "Zyxwvu" in run(block.aget(tail))


@pytest.mark.parametrize("messages", [None, [], [FakeChatMessage(None)]])
def test_nothing_to_query_on_contributes_nothing_rather_than_an_empty_header(
        block, messages):
    assert run(block.aget(messages)) == ""


def test_messages_with_no_text_are_not_written(block, mem):
    run(block.aput([FakeChatMessage(None), FakeChatMessage("")]))
    assert block.last_receipt is None and mem.count() == 0


def test_the_full_context_window_is_used_when_it_is_switched_off(mem, monkeypatch):
    install_llamaindex(monkeypatch)
    block = li.MemvaraMemoryBlock(memory=mem, user="alice", context_window=0)
    run(block.aput([FakeChatMessage("I work at Acme")]))
    assert "Acme" in run(block.aget([FakeChatMessage("tell me about Acme")] +
                                    [FakeChatMessage("ok") for _ in range(9)]))


def test_search_and_history_are_where_the_structure_lives(block, mem):
    """`_aget` returns a string, so scores and provenance die at that boundary. These
    two are the escape hatch, and without them the block would be a lossy wrapper with
    no way back."""
    run(block.aput([FakeChatMessage("I live in Berlin")]))
    run(block.aput([FakeChatMessage("Actually I moved to Lisbon")]))
    top = block.search("where do they live?")[0]
    assert top.claim.predicate == "lives_in" and top.explain.summary()
    assert [c.object for c in block.history("user", "lives_in")] == ["Berlin", "Lisbon"]
    assert block.search("where", as_of=T0) == []


def test_the_block_honours_the_bases_short_term_memory_gate(mem, monkeypatch):
    install_llamaindex(monkeypatch)
    block = li.MemvaraMemoryBlock(memory=mem, user="alice",
                                 accept_short_term_memory=False)
    run(block.aput([FakeChatMessage("I live in Berlin")], from_short_term_memory=True))
    assert mem.count() == 0


@pytest.mark.parametrize("role, expected", [
    ("assistant", "assistant"), ("tool", "tool"), ("chatbot", "chatbot"),
])
def test_message_roles_pass_through_unmapped_because_they_already_agree(role, expected):
    assert li._role_of(FakeChatMessage("x", role=role)) == expected


def test_a_message_with_no_role_is_a_user_turn():
    assert li._role_of(SimpleNamespace()) == "user"


# =====================================================================================
# LlamaIndex — retriever and refusals
# =====================================================================================

def test_the_llamaindex_retriever_returns_nodes_carrying_the_claim(mem, monkeypatch):
    install_llamaindex(monkeypatch)
    mem.remember("user", "lives_in", "Lisbon")
    nodes = li.MemvaraRetriever(mem, user="alice").retrieve("where do they live?")
    assert isinstance(nodes[0], FakeNodeWithScore)
    assert nodes[0].node.text == "user lives in Lisbon"
    assert nodes[0].node.metadata["predicate"] == "lives_in"
    assert nodes[0].node.id_ == nodes[0].node.metadata["memvara_id"]
    assert nodes[0].score > 0


def test_the_retriever_accepts_a_query_bundle_or_a_bare_string(mem, monkeypatch):
    install_llamaindex(monkeypatch)
    mem.remember("user", "lives_in", "Lisbon")
    retriever = li.MemvaraRetriever(mem, user="alice")
    bundle = SimpleNamespace(query_str="where do they live?", embedding=[0.0] * 64)
    assert [n.node.text for n in retriever.retrieve(bundle)] == ["user lives in Lisbon"]


def test_a_pre_embedded_query_bundle_is_answered_from_its_text_not_its_vector(
        mem, monkeypatch):
    """`QueryBundle.embedding` comes from LlamaIndex's embed_model and is not a point in
    memvara's space. Using it would be exactly the vector-store mistake `as_vector_store`
    refuses, and it would fail silently — a wrong-space cosine still returns a ranking."""
    install_llamaindex(monkeypatch)
    mem.remember("user", "lives_in", "Lisbon")
    nonsense = SimpleNamespace(query_str="where do they live?", embedding=[9.9] * 1536)
    assert li.MemvaraRetriever(mem, user="alice").retrieve(nonsense)[0].node.text == (
        "user lives in Lisbon")


def test_the_retriever_passes_the_frameworks_own_arguments_up_to_the_base(mem, monkeypatch):
    install_llamaindex(monkeypatch)
    retriever = li.MemvaraRetriever(mem, user="alice", verbose=True)
    assert retriever.verbose is True


def test_llamaindex_time_travel_and_episode_labelling(mem, monkeypatch):
    install_llamaindex(monkeypatch)
    mem.remember("user", "lives_in", "Berlin", valid_from=T0, recorded_at=T0)
    mem.remember("user", "lives_in", "Lisbon", valid_from=T0 + timedelta(days=30),
                 recorded_at=T0 + timedelta(days=30))
    past = li.MemvaraRetriever(mem, user="alice", as_of=T0 + timedelta(days=1),
                              k=3, min_score=0.0, memory_types=None)
    assert [n.node.text for n in past.retrieve("where")] == ["user lives in Berlin"]
    mem.add("I have been thinking about Porto")
    with_turns = li.MemvaraRetriever(mem, user="alice", include_episodes=True, k=6)
    assert "episode" in {n.node.metadata["kind"] for n in with_turns.retrieve("Porto")}


def test_standing_in_as_a_vector_store_is_refused_and_names_the_retriever(mem):
    """The most interesting refusal. A vector store is handed an embedding and never the
    query text; memvara retrieves from text. Serving one means degrading to cosine top-k
    in someone else's vector space, which is every reason to use memvara, gone."""
    with pytest.raises(li.LlamaIndexCompatError, match="MemvaraRetriever") as caught:
        li.as_vector_store(mem)
    assert "query_embedding" in str(caught.value)


def test_standing_in_as_the_chat_buffer_is_refused_and_names_the_composable_shape(mem):
    with pytest.raises(li.LlamaIndexCompatError, match="memory_blocks") as caught:
        li.as_chat_memory(mem)
    assert "MemvaraMemoryBlock" in str(caught.value)


def test_the_llamaindex_classes_are_lazy_and_name_the_extra_when_absent():
    for name in ("MemvaraMemoryBlock", "MemvaraRetriever"):
        with pytest.raises(ImportError, match=r"memvara\[llama-index\]"):
            getattr(li, name)


# =====================================================================================
# CrewAI — the storage backend
# =====================================================================================

@pytest.fixture()
def storage(mem):
    return ca.MemvaraStorage(mem, user="alice", types=CREWAI_TYPES)


def record(content, **kw):
    return FakeMemoryRecord(content, **kw)


def saved(storage, *records):
    for item in records:
        item.embedding = crew_embed(storage, item.content)
    storage.save(list(records))
    return records


def test_the_backend_has_every_method_the_protocol_names(storage):
    """`StorageBackend` is a `Protocol`, so a missing method is not a type error — it is
    an `AttributeError` inside CrewAI, at whatever point of the flow first needs it."""
    required = ["save", "search", "delete", "update", "get_record", "list_records",
                "get_scope_info", "list_scopes", "list_categories", "count", "reset",
                "asave", "asearch", "adelete"]
    assert [n for n in required if not callable(getattr(storage, n, None))] == []


def test_a_record_round_trips_with_every_field_crewai_put_on_it(storage):
    original = record("Alice lives in Berlin", scope="/acme/eng", categories=["profile"],
                      metadata={"ticket": "T-1"}, importance=0.9, source="alice",
                      private=True)
    saved(storage, original)
    back = storage.get_record(original.id)
    assert (back.id, back.content, back.scope) == (
        original.id, "Alice lives in Berlin", "/acme/eng")
    assert back.categories == ["profile"] and back.metadata == {"ticket": "T-1"}
    assert (back.importance, back.source, back.private) == (0.9, "alice", True)
    assert back.created_at == original.created_at


def test_created_at_comes_back_naive_because_crewais_own_scorer_does_naive_arithmetic(
        storage):
    """`compute_composite_score` does `datetime.utcnow() - record.created_at`. Handing
    back a helpfully-normalized aware datetime raises `TypeError` inside CrewAI, on
    their line, for our reason."""
    original = saved(storage, record("Alice lives in Berlin",
                                     created_at=datetime(2024, 3, 1, 12, 0, 0)))[0]
    back = storage.get_record(original.id)
    assert back.created_at == datetime(2024, 3, 1, 12, 0, 0)
    assert back.created_at.tzinfo is None
    # The arithmetic CrewAI's scorer does, which an aware datetime would make a TypeError.
    assert (datetime(2024, 6, 1) - back.created_at).days == 91


def test_user_metadata_cannot_collide_with_the_adapters_own_bookkeeping(storage, mem):
    """One nested blob rather than six flat keys: `MemoryRecord.metadata` is the
    caller's dict, and any flat scheme is one unlucky key name away from user data
    overwriting the record's scope."""
    original = saved(storage, record("hi", scope="/real",
                                     metadata={"scope": "/fake", "importance": 99}))[0]
    back = storage.get_record(original.id)
    assert back.scope == "/real" and back.importance == 0.5
    assert back.metadata == {"scope": "/fake", "importance": 99}


def test_records_are_backdated_so_as_of_over_a_crewai_store_means_something(storage, mem):
    old = record("Alice lives in Berlin", created_at=datetime(2024, 1, 1, 12, 0))
    saved(storage, old)
    assert mem.count(as_of=datetime(2023, 12, 1, tzinfo=TZ)) == 0
    assert mem.count(as_of=datetime(2024, 2, 1, tzinfo=TZ)) == 1


def test_an_unknown_record_id_is_none_rather_than_an_error(storage):
    assert storage.get_record("nope") is None


def test_saving_the_same_record_twice_reinforces_instead_of_duplicating(storage):
    first = saved(storage, record("Alice lives in Berlin"))[0]
    saved(storage, record("Alice lives in Berlin", id=first.id))
    assert storage.count() == 1


# --- the embedder seam ---------------------------------------------------------------

def test_the_embedder_returns_plain_lists_because_crewai_truth_tests_the_result(storage):
    """The measured bug. `embed_text` guards with `if not result:`, and that is a
    `ValueError: the truth value of an array with more than one element is ambiguous`
    on the ndarray an `Embedder` returns. Only running the real package finds this."""
    out = storage.embedder(["hello", "world"])
    assert isinstance(out, list) and isinstance(out[0], list)
    assert isinstance(out[0][0], float)
    assert crew_embed(storage, "hello") == out[0]


def test_search_recovers_the_query_text_and_runs_real_hybrid_retrieval(storage):
    """The crux. `StorageBackend.search` is handed a vector and never the query, and
    memvara retrieves from text — so the backend supplies the embedder, remembers what it
    embedded, and gets the question back."""
    saved(storage, record("Alice lives in Berlin"), record("Alice prefers pytest"))
    hits = storage.search(crew_embed(storage, "pytest"), limit=5)
    assert [r.content for r, _ in hits][0] == "Alice prefers pytest"
    assert [r.content for r, _ in storage.search(crew_embed(storage, "Berlin"))][0] == (
        "Alice lives in Berlin")
    assert all(0.0 <= score <= 1.0 for _, score in hits)


def test_a_vector_this_backend_did_not_produce_is_refused_rather_than_searched(storage):
    """A foreign embedding is not a bad query, it is a point in a different space — and
    a cosine against it still returns a confident-looking ranking. Refusing is the only
    honest answer, and the message has to carry the one-line fix."""
    with pytest.raises(ca.CrewAICompatError, match=r"embedder=storage\.embedder") as e:
        storage.search([0.1] * 1536)
    assert "1536" in str(e.value) and "64" in str(e.value)


def test_an_evicted_query_says_which_knob_to_turn(mem):
    """The cache is bounded, so this is reachable. It must not look like a wiring error
    when it is a sizing one."""
    small = ca.MemvaraStorage(mem, user="alice", types=CREWAI_TYPES, query_cache=1)
    stale = crew_embed(small, "first question")
    crew_embed(small, "second question")
    with pytest.raises(ca.CrewAICompatError, match="query_cache"):
        small.search(stale)
    assert small.search(crew_embed(small, "second question")) == []


def test_re_embedding_a_text_keeps_it_warm_rather_than_ageing_it_out(mem):
    small = ca.MemvaraStorage(mem, user="alice", types=CREWAI_TYPES, query_cache=2)
    for text in ("a", "b", "a", "c"):
        crew_embed(small, text)
    assert small.search(crew_embed(small, "a")) == []          # "a" was refreshed
    assert list(small._queries.values()) == ["c", "a"]


# --- filtering -----------------------------------------------------------------------

def test_search_filters_by_crewai_scope_and_category_after_ranking(storage):
    saved(storage,
          record("Alice lives in Berlin", scope="/acme/eng", categories=["profile"]),
          record("Alice prefers pytest", scope="/acme", categories=["prefs"]))
    query = crew_embed(storage, "alice")
    assert [r.content for r, _ in storage.search(query, scope_prefix="/acme/eng")] == [
        "Alice lives in Berlin"]
    assert [r.content for r, _ in storage.search(query, categories=["prefs"])] == [
        "Alice prefers pytest"]
    assert storage.search(query, scope_prefix="/other") == []


def test_a_scope_prefix_matches_whole_segments_not_string_prefixes(storage):
    """`/acme` must not reach into `/acmecorp`. A bare `startswith` would hand one
    tenant's memories to another and look perfectly fine doing it."""
    saved(storage, record("theirs", scope="/acmecorp"), record("ours", scope="/acme/eng"))
    assert [r.content for r in storage.list_records(scope_prefix="/acme")] == ["ours"]
    assert storage.count("/acme") == 1 and storage.count() == 2


def test_search_respects_limit_and_the_min_score_floor(storage):
    saved(storage, *(record(f"Alice fact number {i}") for i in range(5)))
    query = crew_embed(storage, "Alice fact")
    assert len(storage.search(query, limit=2)) == 2
    assert storage.search(query, limit=5, min_score=0.99) == []


def test_a_metadata_filter_is_refused_because_post_filtering_would_lie_about_recall(
        storage):
    """Applied after ranking, it silently returns fewer results than asked for — and the
    caller cannot tell that from "nothing matched"."""
    query = crew_embed(storage, "anything")
    with pytest.raises(ca.CrewAICompatError, match="scope_prefix"):
        storage.search(query, metadata_filter={"ticket": "T-1"})
    with pytest.raises(ca.CrewAICompatError, match="scope_prefix"):
        storage.delete(metadata_filter={"ticket": "T-1"})


def test_oversampling_is_what_stops_a_scoped_search_under_filling(mem):
    """Post-ranking filters thin the list, so the backend asks for more than it needs.
    The knob exists because the honest statement is that the filter is not in the index."""
    thin_storage = ca.MemvaraStorage(mem, user="alice", types=CREWAI_TYPES, oversample=1)
    wide_storage = ca.MemvaraStorage(mem, user="alice", types=CREWAI_TYPES, oversample=20)
    saved(thin_storage, *(record(f"Alice noise {i}", scope="/noise") for i in range(6)))
    saved(thin_storage, record("Alice signal", scope="/wanted"))
    thin = thin_storage.search(crew_embed(thin_storage, "Alice"),
                               scope_prefix="/wanted", limit=1)
    wide = wide_storage.search(crew_embed(wide_storage, "Alice"),
                               scope_prefix="/wanted", limit=1)
    assert thin == [] and [r.content for r, _ in wide] == ["Alice signal"]


def test_a_claim_that_is_not_a_crewai_record_is_never_returned_as_one(storage, mem):
    """One memvara store can hold CrewAI records, imported mem0 notes and ordinary
    extracted facts. A backend that handed a `lives_in` triple back as a `MemoryRecord`
    would be inventing an id and a scope for it."""
    mem.remember("user", "lives_in", "Berlin")
    saved(storage, record("Alice lives in Berlin"))
    assert storage.count() == 1
    assert len(storage.search(crew_embed(storage, "Berlin"), limit=10)) == 1


# --- update, delete, reset -----------------------------------------------------------

def test_update_is_a_supersession_so_the_version_it_replaced_survives(storage, mem):
    """Better than CrewAI's own contract, which overwrites the row. Here the old value
    is retired with a pointer to what replaced it, and the whole timeline is walkable."""
    original = saved(storage, record("Alice lives in Berlin"))[0]
    storage.update(record("Alice lives in Lisbon", id=original.id,
                          created_at=original.created_at))
    assert storage.get_record(original.id).content == "Alice lives in Lisbon"
    assert storage.get_record(original.id).created_at == original.created_at
    timeline = mem.history(note_subject(original.id, prefix=ca.SUBJECT_PREFIX),
                           NOTE_PREDICATE)
    assert [c.object for c in timeline] == ["Alice lives in Berlin",
                                            "Alice lives in Lisbon"]
    assert timeline[0].invalidated_by == timeline[1].id
    assert storage.count() == 1


def test_deleting_retires_by_default_and_says_so_once(storage, mem):
    """CrewAI's own dedup path deletes a superseded record, where retirement is right.
    A data-deletion request is the other reading, and the default must not quietly
    under-serve it."""
    first, second = saved(storage, record("one"), record("two"))
    with pytest.warns(ca.CrewAIDeletionWarning, match="on_delete='erase'"):
        assert storage.delete(record_ids=[first.id]) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # once per instance, not per record
        assert storage.delete(record_ids=[second.id]) == 1
    assert storage.count() == 0
    assert len(mem.history(note_subject(first.id, prefix=ca.SUBJECT_PREFIX),
                           NOTE_PREDICATE)) == 1


def test_deleting_nothing_does_not_warn_about_a_retirement_that_did_not_happen(storage):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert storage.delete(record_ids=["nope"]) == 0


def test_erase_mode_removes_the_text_and_the_turn_behind_it(mem):
    storage = ca.MemvaraStorage(mem, user="alice", types=CREWAI_TYPES, on_delete="erase")
    first = saved(storage, record("Alice lives in Berlin"))[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # an erasure has nothing to disclose
        assert storage.delete(record_ids=[first.id]) == 1
    assert mem.history(note_subject(first.id, prefix=ca.SUBJECT_PREFIX),
                       NOTE_PREDICATE) == []
    assert mem.stats()["episodes"] == 0


def test_retire_mode_is_the_informed_choice_and_stays_silent(mem):
    storage = ca.MemvaraStorage(mem, user="alice", types=CREWAI_TYPES, on_delete="retire")
    first = saved(storage, record("Alice lives in Berlin"))[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert storage.delete(record_ids=[first.id]) == 1


def test_an_unknown_on_delete_is_rejected_at_construction(mem):
    with pytest.raises(ValueError, match="on_delete='wipe'"):
        ca.MemvaraStorage(mem, user="alice", types=CREWAI_TYPES, on_delete="wipe")


def test_delete_can_select_by_scope_by_category_and_by_age(storage):
    saved(storage,
          record("a", scope="/x", categories=["k"], created_at=datetime(2024, 1, 1)),
          record("b", scope="/x", categories=["j"], created_at=datetime(2024, 6, 1)),
          record("c", scope="/y", categories=["k"], created_at=datetime(2024, 1, 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ca.CrewAIDeletionWarning)
        assert storage.delete(scope_prefix="/x", categories=["k"],
                              older_than=datetime(2024, 3, 1)) == 1
        assert sorted(r.content for r in storage.list_records()) == ["b", "c"]
        assert storage.delete(older_than=datetime(2024, 3, 1)) == 1
        assert [r.content for r in storage.list_records()] == ["b"]


def test_reset_is_the_one_deletion_that_maps_exactly_so_it_does_not_warn(storage, mem):
    """CrewAI means "wipe" and `purge()` is a wipe — claims, turns, vectors and the text
    index. There is nothing to disclose, so nothing is disclosed."""
    first = saved(storage, record("Alice lives in Berlin"))[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        storage.reset()
    assert storage.count() == 0
    assert mem.history(note_subject(first.id, prefix=ca.SUBJECT_PREFIX),
                       NOTE_PREDICATE) == []
    assert mem.stats()["episodes"] == 0


@pytest.mark.parametrize("everything", [None, "", "/"])
def test_every_spelling_of_the_root_scope_resets_the_whole_binding(storage, everything):
    saved(storage, record("Alice lives in Berlin", scope="/acme"))
    storage.reset(everything)
    assert storage.count() == 0


def test_resetting_one_crewai_scope_erases_only_that_subtree(storage, mem):
    """Memvara cannot purge by CrewAI's scope tree — the two point in opposite directions
    — so those records are erased one at a time. The same erasure, reached the long way."""
    keep, drop = saved(storage, record("keep", scope="/keep"),
                       record("drop", scope="/drop/deep"))
    storage.reset("/drop")
    assert [r.content for r in storage.list_records()] == ["keep"]
    assert mem.history(note_subject(drop.id, prefix=ca.SUBJECT_PREFIX),
                       NOTE_PREDICATE) == []


# --- listing -------------------------------------------------------------------------

def test_list_records_is_newest_first_and_pages(storage):
    saved(storage, *(record(f"fact {i}") for i in range(5)))
    page = storage.list_records(limit=2, offset=1)
    assert len(page) == 2
    assert [r.content for r in storage.list_records()][0] == "fact 4"


def test_scopes_are_enumerated_one_level_at_a_time(storage):
    saved(storage, record("a", scope="/acme/eng"), record("b", scope="/acme/sales/emea"),
          record("c", scope="/other"), record("d", scope="/"))
    assert storage.list_scopes() == ["/acme", "/other"]
    assert storage.list_scopes("/acme") == ["/acme/eng", "/acme/sales"]
    assert storage.list_scopes("/acme/eng") == []


def test_categories_are_counted_within_a_scope(storage):
    saved(storage, record("a", scope="/x", categories=["k", "j"]),
          record("b", scope="/y", categories=["k"]))
    assert storage.list_categories() == {"k": 2, "j": 1}
    assert storage.list_categories("/x") == {"k": 1, "j": 1}


def test_scope_info_reports_the_subtree_including_its_date_range(storage):
    saved(storage,
          record("a", scope="/acme/eng", categories=["k"],
                 created_at=datetime(2024, 1, 1)),
          record("b", scope="/acme", categories=["j"], created_at=datetime(2024, 6, 1)))
    info = storage.get_scope_info("/acme")
    assert isinstance(info, FakeScopeInfo)
    assert (info.path, info.record_count) == ("/acme", 2)
    assert info.categories == ["j", "k"] and info.child_scopes == ["/acme/eng"]
    assert (info.oldest_record, info.newest_record) == (datetime(2024, 1, 1),
                                                        datetime(2024, 6, 1))


def test_an_empty_scope_reports_no_dates_rather_than_inventing_them(storage):
    info = storage.get_scope_info("/nothing/here")
    assert (info.record_count, info.oldest_record, info.newest_record) == (0, None, None)


def test_two_backends_on_one_store_cannot_see_each_other(mem):
    """The memvara scope binding is the isolation, and it is the half of CrewAI's scope
    story that memvara enforces rather than filters."""
    alice = ca.MemvaraStorage(mem, user="alice", types=CREWAI_TYPES)
    bob = ca.MemvaraStorage(mem, user="bob", types=CREWAI_TYPES)
    saved(alice, record("alice's", scope="/shared"))
    assert bob.count() == 0
    assert bob.search(crew_embed(bob, "alice's")) == []


def test_the_storage_repr_names_the_scope_and_the_deletion_policy(storage):
    assert "default/alice" in repr(storage) and "on_delete=warn" in repr(storage)


# --- async ---------------------------------------------------------------------------

def test_the_async_half_is_the_sync_half_off_the_loop_thread(storage):
    """CrewAI declares `asave`/`asearch`/`adelete`; a backend that omits them gets the
    protocol's own default, which calls the sync method *on the loop thread*."""
    item = record("Alice lives in Berlin")
    item.embedding = crew_embed(storage, item.content)
    run(storage.asave([item]))
    assert storage.count() == 1
    hits = run(storage.asearch(crew_embed(storage, "where does alice live?"), limit=3))
    assert [r.content for r, _ in hits] == ["Alice lives in Berlin"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ca.CrewAIDeletionWarning)
        assert run(storage.adelete(record_ids=[item.id])) == 1


# --- lazy typing -----------------------------------------------------------------------

def test_crewais_types_are_imported_on_first_use_not_at_import_time(mem, monkeypatch):
    """The class body needs no CrewAI at all — `StorageBackend` is a Protocol. Only the
    two return types do, and construction must not reach for them."""
    storage = ca.MemvaraStorage(mem, user="alice")
    monkeypatch.setitem(sys.modules, "crewai.memory.types", CREWAI_TYPES)
    first = saved(storage, record("Alice lives in Berlin"))[0]
    assert isinstance(storage.get_record(first.id), FakeMemoryRecord)


def test_a_missing_crewai_names_the_extra_only_when_a_record_is_needed(mem):
    storage = ca.MemvaraStorage(mem, user="alice")
    saved(storage, record("Alice lives in Berlin"))       # save needs no CrewAI type
    with pytest.raises(ImportError, match=r"memvara\[crewai\]"):
        storage.list_records()


def test_the_package_exposes_each_adapter_lazily_and_rejects_anything_else(monkeypatch):
    """`import memvara.integrations` must import no framework — the numpy-only install is
    a CI job, not a slogan."""
    import memvara.integrations as pkg

    install_langchain(monkeypatch)
    install_llamaindex(monkeypatch)
    assert pkg.MemvaraChatMessageHistory is lc.MemvaraChatMessageHistory
    assert pkg.MemvaraMemoryBlock is li.MemvaraMemoryBlock
    assert pkg.MemvaraStorage is ca.MemvaraStorage
    with pytest.raises(AttributeError, match="MemvaraRetriever"):
        pkg.MemvaraRetriever        # ambiguous: two frameworks have one
    with pytest.raises(AttributeError):
        pkg.NotAThing


def test_naming_a_langchain_class_without_the_sdk_says_what_to_install():
    for name in ("MemvaraChatMessageHistory", "MemvaraRetriever"):
        with pytest.raises(ImportError, match=r"memvara\[langchain\]"):
            getattr(lc, name)
    with pytest.raises(AttributeError):
        lc.NotAThing
    with pytest.raises(AttributeError):
        li.NotAThing


def test_the_adapters_import_with_numpy_alone():
    """The actual CI assertion, run here too so it fails in a second rather than in a
    workflow: no module under `memvara.integrations` may import a framework at module
    scope, however convenient."""
    import importlib
    import pkgutil

    import memvara.integrations as pkg

    for module in pkgutil.walk_packages(pkg.__path__, f"{pkg.__name__}."):
        importlib.import_module(module.name)
    assert np.__name__ == "numpy"          # the only third-party import memvara allows
