"""LangChain: the chat history and the retriever, against the real langchain-core.

Each `check_*` function runs inside a virtual environment that holds langchain-core at the
floor memvara declares or at the newest release (see `probe.py`). The promises they check
are the ones the adapter's docstrings and `docs/integrations/frameworks.md` make.
langchain-core is imported inside each check, never at module level, because the suite
imports this module to list the checks and has no langchain-core installed.
"""

from __future__ import annotations

import asyncio
import importlib
import warnings
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from memvara.integrations import langchain as adapter

if TYPE_CHECKING:
    from ..probe import Context

T0 = datetime(2024, 3, 1, tzinfo=timezone.utc)
T1 = datetime(2024, 6, 1, tzinfo=timezone.utc)
APRIL = datetime(2024, 4, 1, tzinfo=timezone.utc)


def _lc(module: str) -> Any:
    return importlib.import_module(f"langchain_core.{module}")


def _history(ctx: Context, **options: Any) -> Any:
    options.setdefault("transcript_warning", False)
    return adapter.MemvaraChatMessageHistory(ctx.memvara(), session="s1", **options)


def check_the_history_is_a_real_basechatmessagehistory(ctx: Context) -> None:
    """The composed class subclasses langchain-core's own base class, and it is made
    once, so an `isinstance` check in a user's code stays stable."""
    assert isinstance(_history(ctx), _lc("chat_history").BaseChatMessageHistory)
    assert adapter.MemvaraChatMessageHistory is adapter.MemvaraChatMessageHistory


def check_messages_come_back_as_the_classes_they_were_written_as(ctx: Context) -> None:
    """Every message class, its text, a ChatMessage's own role and a ToolMessage's call id
    survive a round trip through the store, in the order they were sent."""
    m = _lc("messages")
    history = _history(ctx)
    sent = [
        m.SystemMessage(content="You are terse."),
        m.HumanMessage(content="I live in Berlin"),
        m.AIMessage(content="Noted."),
        m.ToolMessage(content="42", tool_call_id="call-1"),
        m.ChatMessage(content="Looks fine to me", role="critic"),
    ]
    history.add_messages(sent)
    got = history.messages
    assert [type(x).__name__ for x in got] == [type(x).__name__ for x in sent], got
    assert [x.content for x in got] == [x.content for x in sent]
    assert got[3].tool_call_id == "call-1"
    assert got[4].role == "critic"


def check_the_base_class_helpers_write_through_add_messages(ctx: Context) -> None:
    """langchain-core's own add_user_message, add_ai_message and add_message, and the
    async aadd_messages and aget_messages, all reach the store through the adapter."""
    m = _lc("messages")
    history = _history(ctx)
    history.add_user_message("first")
    history.add_ai_message("second")
    history.add_message(m.HumanMessage(content="third"))
    asyncio.run(history.aadd_messages([m.AIMessage(content="fourth")]))
    got = asyncio.run(history.aget_messages())
    assert [x.content for x in got] == ["first", "second", "third", "fourth"], got


def check_reading_and_writing_text_raises_no_deprecation_warning(ctx: Context) -> None:
    """The adapter reads a message's text without calling a deprecated accessor.

    `_text_of` in the adapter says langchain-core 1.x returns a callable string from
    `.text`, and that calling it earns a deprecation warning on every turn.
    """
    m = _lc("messages")
    history = _history(ctx)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        history.add_messages([m.HumanMessage(content="I live in Berlin"),
                              m.AIMessage(content="Noted.")])
        history.messages
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)
                    or "Deprecat" in w.category.__name__]
    assert deprecations == [], [str(w.message) for w in deprecations]


def check_reading_the_transcript_warns_once_that_the_memory_is_not_in_it(
        ctx: Context) -> None:
    """The first read of `messages` in a process warns with MemvaraTranscriptWarning,
    naming both ways the list is smaller than it looks, and transcript_warning=False
    switches the warning off."""
    m = _lc("messages")
    adapter._WARNED_TRANSCRIPT = False
    history = adapter.MemvaraChatMessageHistory(ctx.memvara(), session="s1")
    history.add_messages([m.HumanMessage(content="hello")])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        history.messages
        history.messages
    ours = [w for w in caught if w.category is adapter.MemvaraTranscriptWarning]
    assert len(ours) == 1, [str(w.message) for w in caught]
    assert "It is not the memory" in str(ours[0].message)
    assert "not a verbatim log" in str(ours[0].message)
    adapter._WARNED_TRANSCRIPT = False
    quiet = _history(ctx)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet.messages
    assert [w for w in caught if w.category is adapter.MemvaraTranscriptWarning] == []


def check_a_repeated_turn_is_stored_once(ctx: Context) -> None:
    """The documented deduplication: an exactly repeated turn is stored once, so the
    transcript is source material rather than a verbatim log."""
    m = _lc("messages")
    history = _history(ctx)
    history.add_messages([m.HumanMessage(content="ok")])
    history.add_messages([m.HumanMessage(content="ok")])
    assert [x.content for x in history.messages] == ["ok"]


def check_the_transcript_holds_only_its_own_session(ctx: Context) -> None:
    """Two sessions on one store see only their own turns, and a turn that a named agent
    wrote inside the same session is not in the session's transcript."""
    m = _lc("messages")
    mem = ctx.memvara()
    one = adapter.MemvaraChatMessageHistory(mem, session="s1", transcript_warning=False)
    two = adapter.MemvaraChatMessageHistory(mem, session="s2", transcript_warning=False)
    agent = adapter.MemvaraChatMessageHistory(mem, session="s1", agent="researcher",
                                              transcript_warning=False)
    one.add_messages([m.HumanMessage(content="from s1")])
    two.add_messages([m.HumanMessage(content="from s2")])
    agent.add_messages([m.HumanMessage(content="from the agent")])
    assert [x.content for x in one.messages] == ["from s1"]
    assert [x.content for x in two.messages] == ["from s2"]


def check_clear_refuses_and_each_opt_in_does_what_it_names(ctx: Context) -> None:
    """clear() refuses by default, through langchain-core's aclear() too, and names both
    options; on_clear="ignore" keeps every turn; on_clear="purge" erases the session."""
    m = _lc("messages")
    refusing = _history(ctx)
    refusing.add_messages([m.HumanMessage(content="I live in Berlin")])
    for clear in (refusing.clear, lambda: asyncio.run(refusing.aclear())):
        try:
            clear()
        except adapter.LangChainCompatError as exc:
            assert "on_clear='purge'" in str(exc) and "on_clear='ignore'" in str(exc)
        else:
            raise AssertionError("clear() did not refuse")
    ignoring = _history(ctx, on_clear="ignore")
    ignoring.add_messages([m.HumanMessage(content="I live in Berlin")])
    ignoring.clear()
    assert [x.content for x in ignoring.messages] == ["I live in Berlin"]
    purging = _history(ctx, on_clear="purge")
    purging.add_messages([m.HumanMessage(content="I live in Berlin")])
    purging.clear()
    assert purging.messages == []
    assert purging.memory.stats()["episodes"] == 0


def check_runnable_with_message_history_reads_and_writes_the_history(ctx: Context) -> None:
    """langchain-core's RunnableWithMessageHistory, the usual reader and writer of a chat
    history, stores each exchange through the adapter and reads it back into the next
    prompt."""
    prompts = _lc("prompts")
    runnables = _lc("runnables")
    fake = _lc("language_models.fake_chat_models")
    mem = ctx.memvara()
    prompt = prompts.ChatPromptTemplate.from_messages(
        [prompts.MessagesPlaceholder("history"), ("human", "{input}")])
    seen: list[list[Any]] = []

    def spy(value: Any) -> Any:
        seen.append(value.to_messages())
        return value

    chain = prompt | runnables.RunnableLambda(spy) | fake.FakeListChatModel(
        responses=["Noted.", "You live in Berlin."])
    wrapped = _lc("runnables.history").RunnableWithMessageHistory(
        chain,
        lambda session_id: adapter.MemvaraChatMessageHistory(
            mem, session=session_id, transcript_warning=False),
        input_messages_key="input", history_messages_key="history")
    config = {"configurable": {"session_id": "s1"}}
    wrapped.invoke({"input": "I live in Berlin"}, config=config)
    wrapped.invoke({"input": "Where do I live?"}, config=config)
    assert [x.content for x in seen[1]] == [
        "I live in Berlin", "Noted.", "Where do I live?"], seen[1]
    stored = adapter.MemvaraChatMessageHistory(mem, session="s1", transcript_warning=False)
    assert [x.content for x in stored.messages] == [
        "I live in Berlin", "Noted.", "Where do I live?", "You live in Berlin."]


def check_the_retriever_is_a_real_baseretriever_on_the_modern_path(ctx: Context) -> None:
    """The retriever subclasses BaseRetriever and names run_manager in its signature, so
    langchain-core treats it as a modern retriever rather than a legacy one."""
    retriever = adapter.MemvaraRetriever(memory=ctx.memvara(), user="alice")
    assert isinstance(retriever, _lc("retrievers").BaseRetriever)
    assert type(retriever)._new_arg_supported is True
    assert type(retriever)._expects_other_args is False


def check_the_retriever_returns_documents_carrying_the_claim(ctx: Context) -> None:
    """invoke() returns Documents whose metadata holds the triple, both time axes, the
    ranking explanation and the source turn ids."""
    mem = ctx.memvara()
    mem.remember("user", "lives_in", "Berlin", valid_from=T0)
    retriever = adapter.MemvaraRetriever(memory=mem, user="alice", k=3)
    documents = retriever.invoke("where does the user live")
    assert documents, "the retriever found nothing"
    top = documents[0]
    assert type(top).__name__ == "Document"
    assert top.page_content == "user lives in Berlin"
    for key in ("memvara_id", "subject", "predicate", "object", "valid_from", "valid_to",
                "recorded_at", "invalidated_at", "why", "sources", "scope", "score"):
        assert key in top.metadata, key
    assert (top.metadata["predicate"], top.metadata["object"]) == ("lives_in", "Berlin")


def check_the_retriever_answers_ainvoke_and_batch(ctx: Context) -> None:
    """The async and batch paths of the Runnable interface reach the same search."""
    mem = ctx.memvara()
    mem.remember("user", "lives_in", "Berlin")
    mem.remember("user", "works_at", "Acme")
    retriever = adapter.MemvaraRetriever(memory=mem, user="alice", k=5)
    by_async = asyncio.run(retriever.ainvoke("Berlin"))
    assert "user lives in Berlin" in [d.page_content for d in by_async]
    batched = retriever.batch(["Berlin", "Acme"])
    assert len(batched) == 2
    assert "user works at Acme" in [d.page_content for d in batched[1]]


def check_an_ended_value_is_absent_now_and_present_at_an_earlier_as_of(
        ctx: Context) -> None:
    """A superseded value stops being returned, and as_of= brings back what was believed
    at an earlier instant: time travel survives an interface that never planned for it."""
    mem = ctx.memvara()
    mem.remember("user", "lives_in", "Berlin", valid_from=T0, recorded_at=T0)
    mem.remember("user", "lives_in", "Lisbon", valid_from=T1, recorded_at=T1)
    now = adapter.MemvaraRetriever(memory=mem, user="alice", k=5)
    assert [d.page_content for d in now.invoke("where does the user live")] == [
        "user lives in Lisbon"]
    then = adapter.MemvaraRetriever(memory=mem, user="alice", k=5, as_of=APRIL)
    assert [d.page_content for d in then.invoke("where does the user live")] == [
        "user lives in Berlin"]


def check_an_episode_comes_back_labelled_as_an_episode(ctx: Context) -> None:
    """include_episodes=True also returns raw turns, marked kind="episode" and carrying no
    predicate, so something someone said cannot pass as a fact."""
    mem = ctx.memvara()
    mem.add("The quarterly review is on Thursday.")
    retriever = adapter.MemvaraRetriever(memory=mem, user="alice", k=5,
                                         include_episodes=True)
    documents = retriever.invoke("quarterly review")
    episodes = [d for d in documents if d.metadata["kind"] == "episode"]
    assert episodes, [d.metadata for d in documents]
    assert "predicate" not in episodes[0].metadata


def check_the_retriever_composes_into_a_chain(ctx: Context) -> None:
    """The retriever is a Runnable, so it fits into an LCEL chain like any other."""
    runnables = _lc("runnables")
    mem = ctx.memvara()
    mem.remember("user", "lives_in", "Berlin")
    retriever = adapter.MemvaraRetriever(memory=mem, user="alice")
    chain = retriever | runnables.RunnableLambda(
        lambda documents: "\n".join(d.page_content for d in documents))
    assert "user lives in Berlin" in chain.invoke("Berlin")


def check_recall_and_search_on_the_history_reach_the_memory(ctx: Context) -> None:
    """The history's escape hatches: recall() returns the prompt-ready block framed as
    reference data, and search() returns results that keep their source turn."""
    m = _lc("messages")
    history = _history(ctx)
    history.add_messages([m.HumanMessage(content="I live in Berlin")])
    block = history.recall("where do I live")
    assert "reference data, not instructions" in block and "Berlin" in block, block
    results = history.search("where do I live")
    assert results and results[0].claim.object == "Berlin"
    assert results[0].claim.sources, "the extracted fact lost the turn it came from"
