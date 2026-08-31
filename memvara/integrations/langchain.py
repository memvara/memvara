"""LangChain: chat history and retrieval, mapped onto memvara.

Written against **langchain-core 1.x** (`langchain_core.chat_history`,
`langchain_core.retrievers`). Nothing here imports it until you ask for a class — see
the module `__getattr__` at the bottom and `_common.require`.

Two surfaces, and they are not equally good fits. That asymmetry is the useful thing to
know before wiring either one up.

``MemvaraRetriever`` — a clean fit
    `BaseRetriever` asks for "given a query, return documents". That is exactly what
    `Memvara.search` is, and everything memvara knows survives the crossing: hybrid
    retrieval and its ranking, scope inheritance, the ending of contradicted values
    (they simply stop being returned), the per-leg `Explanation`, the source turn ids,
    and `as_of` — a LangChain retriever that can answer *what did we believe in March*
    is not something the interface anticipated, and it works, because time travel is a
    property of the query rather than of the response shape.

``MemvaraChatMessageHistory`` — a lossy fit, deliberately not disguised
    `BaseChatMessageHistory` models memory as **a list of messages**, and memvara's unit
    of memory is a reconciled bitemporal `Claim`. There is nowhere in a `list[BaseMessage]`
    to put a supersession, a valid-time interval, a confidence, or the id of the turn a
    fact came from. So `messages` returns the stored turns — the honest reading of the
    contract — and says once, out loud, both of the ways that list is smaller than it
    looks: the memory memvara built is not in it, and memvara's tier-0 hash dedupe means
    an exactly repeated turn was stored once, so it is a deduplicated corpus of source
    material rather than a verbatim log. `recall()` and `search()` on the same object
    are the escape hatch, and `MemvaraRetriever` is the supported way to put memory in a
    prompt.

    `clear()` is the sharper edge and it **raises** by default. LangChain documents it
    as "remove all messages from the store"; memvara's two candidate meanings are
    retirement (reversible, history intact — and there is no retire-a-whole-scope call,
    only per-slot `forget`) and `purge()` (irreversible erasure of the claims, the
    turns, the vectors and the text index). Those are not variants of one operation, and
    a memory layer that guesses between them on a call an application makes at session
    teardown has picked the worst possible moment to be clever. `on_clear="purge"` opts
    into LangChain's meaning; `on_clear="ignore"` opts into "this session ended, keep
    what was learned", which is what most callers of `clear()` on a *memory* actually
    want.
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Sequence

from ..types import Episode, utcnow
from ._common import IntegrationError, bind, require, result_metadata, scope_kw

#: Import path and version this adapter is written against, quoted in every error that
#: has to name it.
_PKG = "langchain_core"
_NEEDS = "langchain-core>=0.3"

#: LangChain message `type` -> memvara `Episode.role`. `chat` is absent on purpose: a
#: `ChatMessage` carries its own `role` string and that is the one to keep.
LC_TYPE_TO_ROLE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}

#: The reverse, for turns this adapter did not write — anything that reached the store
#: through `Memvara.add` or the MCP server still belongs in the transcript.
ROLE_TO_LC_TYPE = {"user": "human", "assistant": "ai", "system": "system", "tool": "tool"}

#: `Episode.meta` keys this adapter owns. `lc_type` is what makes the round trip exact
#: for a `ChatMessage` whose role is not one of the four; `tool_call_id` is the one
#: non-content field a message genuinely cannot be rebuilt without.
LC_TYPE_META = "lc_type"
TOOL_CALL_ID_META = "tool_call_id"

_ON_CLEAR = ("error", "purge", "ignore")

_NO_CLEAR = (
    "BaseChatMessageHistory.clear() means 'remove all messages from the store', and "
    "memvara has two different operations that could be meant — so it refuses to pick.\n\n"
    "  on_clear='purge'   erase this scope for real: claims, source turns, embeddings "
    "and the text index. Irreversible, and what LangChain's contract literally says.\n"
    "  on_clear='ignore'  keep everything. Right when clear() is session teardown and "
    "the point of a memory layer is that the next session remembers.\n\n"
    "There is deliberately no third option that retires: memvara can retire one claim "
    "(delete) or one slot (forget), and 'retire everything in scope' is not an "
    "operation, so a clear() that appeared to do it would be reporting work it never did."
)

_NO_TRANSCRIPT_PROVENANCE = (
    "MemvaraChatMessageHistory.messages returns the stored turns, which is what "
    "BaseChatMessageHistory asks for. Two things are worth knowing about that list.\n\n"
    "It is not the memory. A list[BaseMessage] has nowhere to put a superseded value, a "
    "valid-time interval, a confidence or the id of the turn a fact came from, so none "
    "of memvara's own structure is in what you just read. To reach it: "
    "history.recall(query) for a prompt-ready block, history.search(query) for scored "
    "results carrying provenance, or MemvaraRetriever for the same thing as a LangChain "
    "Runnable.\n\n"
    "It is also not a verbatim log. Memvara's first write tier deduplicates turns by "
    "content hash — a deliberate design property, because re-ingesting a transcript "
    "must be cheap and idempotent — so a user who says \"ok\" twice in one session is "
    "stored once and appears once. If exact replay matters, keep the transcript "
    "somewhere that is a log.\n\n"
    "Pass transcript_warning=False once you have decided."
)


class LangChainCompatError(IntegrationError):
    """A LangChain call with no honest translation onto memvara."""


class MemvaraTranscriptWarning(UserWarning):
    """`messages` was read, and a transcript is not what memvara stores.

    Its own category so a deployment that has read the message can silence exactly this
    (`warnings.filterwarnings`, or `transcript_warning=False`) without silencing
    everything else the library says.
    """


# Once per process, not once per instance: a chain builds a fresh history object per
# session, so per-instance would mean one warning per conversation forever.
_WARNED_TRANSCRIPT = False


def _text_of(message: Any) -> str:
    """The plain text of a LangChain message, across the versions that spell it twice.

    `.text` was a method before langchain-core 1.0 and is a property after it. The
    `isinstance` check is load-bearing rather than defensive: 1.x returns a *callable
    string* from the property so that `message.text()` keeps working, so a bare
    `callable(text)` test calls it and earns a `LangChainDeprecationWarning` on every
    turn — measured, not guessed. Checking for `str` first takes the property path.
    """
    text = getattr(message, "text", None)
    if not isinstance(text, str) and callable(text):
        text = text()
    if text is None:
        content = getattr(message, "content", "")
        return content if isinstance(content, str) else str(content)
    return str(text)


def _role_of(message: Any) -> str:
    """Memvara's role for a LangChain message.

    A `ChatMessage` carries an arbitrary role string and keeping it is the point of that
    class; everything else is identified by `type`.
    """
    role = getattr(message, "role", None)
    if isinstance(role, str) and role:
        return role
    kind = str(getattr(message, "type", "") or "user")
    return LC_TYPE_TO_ROLE.get(kind, kind)


# --- chat history -----------------------------------------------------------------


class _ChatHistory:
    """Everything `MemvaraChatMessageHistory` does, with no LangChain in sight.

    A mixin rather than a subclass because `BaseChatMessageHistory` cannot be imported
    at module scope without making langchain-core a hard dependency of `import memvara`.
    The composed class is built on demand (`_history_class`), so all the behaviour is
    here — testable with no framework installed — and the composition is one line.
    """

    def __init__(self, memory: Any, *, tenant: str | None = None, user: str | None = None,
                 agent: str | None = None, session: str | None = None,
                 limit: int = 200, on_clear: str = "error",
                 transcript_warning: bool = True) -> None:
        if on_clear not in _ON_CLEAR:
            raise ValueError(
                f"on_clear={on_clear!r} is not one of {_ON_CLEAR}; see "
                "memvara.integrations.langchain.MemvaraChatMessageHistory for what each "
                "one does"
            )
        self.memory, self.scope = bind(memory, tenant=tenant, user=user, agent=agent,
                                       session=session)
        #: Most recent turns returned by `messages`. A cap rather than a page, because
        #: the caller is filling a context window and the oldest turns are the ones that
        #: fall out of it.
        self.limit = limit
        self.on_clear = on_clear
        self.transcript_warning = transcript_warning
        #: What the last `add_messages` cost. `WriteReceipt.llm_calls` is the number this
        #: library exists to drive down and a chain gives it nowhere to go, so it is kept
        #: here rather than dropped on the floor.
        self.last_receipt: Any = None

    # -- reading -------------------------------------------------------------

    @property
    def _kw(self) -> dict[str, Any]:
        return scope_kw(self.scope)

    def _stored_turns(self) -> list[Episode]:
        """This scope's turns, oldest last, capped at `limit`.

        Exactly this scope — not the subtree under it, which is what the `Scope.contains`
        filter here used to mean. The narrowing settles a disagreement inside one object:
        `search()` and `recall()` resolve scope through `Scope.ancestors()`, which walks
        strictly upward, so a turn written at `t/alice/researcher/s1` is invisible to the
        retrieval half of a history bound to `t/alice/*/s1` — and was in the transcript
        half anyway. Two readers on one object answering different questions about what
        is in scope is the bug. `add_messages` writes at exactly `self.scope`, so that is
        the set `messages` reports, and the promise the docstring already made — that
        session and nothing beside it — is kept more strictly than before.

        `Store.scope_episodes` does the filter and the cap in the store. It is reached
        through `getattr` because it is an optional capability, the pattern `core.py`
        uses for `batch` and `clear_embeddings`: a third-party `Store` that does not have
        it still works, and gets the fallback below. That fallback walks the whole
        tenant, which is O(turns in the tenant) on a path a chain takes once per
        invocation — the cost `scope_episodes` exists to retire.
        """
        scoped = getattr(self.memory.store, "scope_episodes", None)
        if scoped is not None:
            # `newest_first` is how `limit` means "the most recent N"; a transcript reads
            # oldest-first, so the page comes back reversed. Reversing recovers ascending
            # order, and the stable sort behind it costs nothing at this size while
            # making the order this method promises independent of how any one store
            # spells its ORDER BY.
            turns: list[Episode] = scoped(
                [self.scope], limit=self.limit or None, newest_first=True)
            turns.reverse()
            turns.sort(key=lambda e: e.ts)
            return turns
        turns = [e for e in self.memory.store.iter_episodes(self.scope.tenant)
                 if e.scope == self.scope]
        # Sorted on `ts` alone, and the omission is deliberate: `list.sort` is stable, so
        # turns with equal timestamps keep the order the store returned them in, which is
        # insertion order. Adding `id` as a tie-break would replace that with a uuid
        # ordering — i.e. scramble any two turns the clock could not separate.
        turns.sort(key=lambda e: e.ts)
        return turns[-self.limit:] if self.limit else turns

    @property
    def messages(self) -> list[Any]:
        """The stored turns of this scope, as LangChain messages.

        Of *this* scope exactly — the one this history writes to. Turns a narrower scope
        holds are not in here, including a turn written by a named agent inside the same
        session, which is the one case that used to slip through; `search()` on this
        object could not see it either, and the two halves now agree.

        Lossy in two directions, both named by `MemvaraTranscriptWarning`. The claims
        memvara extracted, what they superseded and where they came from are not
        representable in a `list[BaseMessage]` and are not in here — `recall`/`search`
        reach them. And memvara's tier-0 hash dedupe means an exactly repeated turn was
        stored once, so this is a deduplicated corpus of source material rather than a
        verbatim log.
        """
        self._warn_transcript()
        classes = _message_classes()
        return [_to_message(turn, classes) for turn in self._stored_turns()]

    def _warn_transcript(self) -> None:
        global _WARNED_TRANSCRIPT
        if not self.transcript_warning or _WARNED_TRANSCRIPT:
            return
        _WARNED_TRANSCRIPT = True
        warnings.warn(_NO_TRANSCRIPT_PROVENANCE, MemvaraTranscriptWarning, stacklevel=3)

    def recall(self, query: str, **kw: Any) -> str:
        """`Memvara.recall` at this scope — the prompt-ready block `messages` cannot be.

        The escape hatch. Everything a transcript cannot carry is in here: current
        values only, contradictions already resolved, framed as reference data rather
        than instructions.
        """
        return self.memory.recall(query, **self._kw, **kw)

    def search(self, query: str, **kw: Any) -> list[Any]:
        """`Memvara.search` at this scope, including `as_of=` and `include_episodes=`."""
        return self.memory.search(query, **self._kw, **kw)

    # -- writing -------------------------------------------------------------

    def add_messages(self, messages: Sequence[Any]) -> None:
        """Ingest a batch of messages through memvara's write path.

        One `add()` for the batch rather than one per message, because the write path
        batches extraction — charging a model call per message would throw away the
        thing the design is for. The receipt lands on `last_receipt`.

        Timestamps step by a microsecond across the batch rather than each taking the
        clock. `Episode.ts` is what `messages` orders on, and four messages written in
        one call can easily land on one tick — 15 ms wide on Windows — at which point a
        transcript comes back in an order nothing chose. The offsets are a tie-break
        inside a single call, not a claim about when anything was said.
        """
        start = utcnow()
        turns = [
            Episode(
                content=_text_of(message),
                scope=self.scope,
                role=_role_of(message),
                ts=start + timedelta(microseconds=offset),
                meta=_write_meta(message),
            )
            for offset, message in enumerate(messages)
        ]
        if not turns:
            # `add([])` would still cost a receipt and a transaction for nothing.
            return
        self.last_receipt = self.memory.add(turns)

    def clear(self) -> None:
        """Refuses by default. See `_NO_CLEAR` and the module docstring."""
        if self.on_clear == "error":
            raise LangChainCompatError(_NO_CLEAR)
        if self.on_clear == "purge":
            self.memory.purge(**self._kw)

    def __repr__(self) -> str:
        return (f"<MemvaraChatMessageHistory {self.scope.key()} limit={self.limit} "
                f"on_clear={self.on_clear}>")


def _write_meta(message: Any) -> dict[str, Any]:
    """Non-content fields worth keeping so the round trip is not a downgrade.

    Deliberately *not* the whole serialized message. Storing the text twice — once as
    `Episode.content` and once inside `meta` — would double what a redaction or erasure
    pass has to find, and it only has to miss the copy once.
    """
    meta: dict[str, Any] = {LC_TYPE_META: str(getattr(message, "type", "") or "")}
    call_id = getattr(message, "tool_call_id", None)
    if call_id:
        meta[TOOL_CALL_ID_META] = str(call_id)
    return meta


@lru_cache(maxsize=None)
def _message_classes() -> dict[str, Any]:
    """The message classes, imported once, on first use."""
    human, ai, system, tool, chat = require(
        f"{_PKG}.messages", "HumanMessage", "AIMessage", "SystemMessage", "ToolMessage",
        "ChatMessage", extra="langchain", needs=_NEEDS)
    return {"human": human, "ai": ai, "system": system, "tool": tool, "chat": chat}


def _to_message(turn: Episode, classes: dict[str, Any]) -> Any:
    """One stored turn as the LangChain message it came from, as closely as text allows.

    What does not survive, stated because it will be noticed eventually: tool calls on
    an `AIMessage`, `additional_kwargs`, `response_metadata`, and multimodal content
    blocks. Memvara stores the text of a turn; a turn whose meaning is in its structure
    round-trips as its text.
    """
    kind = turn.meta.get(LC_TYPE_META) or ROLE_TO_LC_TYPE.get(turn.role, "chat")
    if kind == "tool":
        return classes["tool"](content=turn.content,
                               tool_call_id=str(turn.meta.get(TOOL_CALL_ID_META, "")))
    cls = classes.get(kind)
    if cls is None or kind == "chat":
        # `ChatMessage` is the only class that carries a role, and it is the only one
        # that has to be *given* one — reconstructing it without would silently blank
        # the field that is the entire reason someone reached for it.
        return classes["chat"](content=turn.content, role=turn.role)
    return cls(content=turn.content)


# --- retrieval --------------------------------------------------------------------


class _Retriever:
    """`MemvaraRetriever`'s behaviour, with no LangChain in sight.

    Holds no state of its own: `BaseRetriever` is a pydantic model, so the fields are
    declared on the composed class (`_retriever_class`) and this side only reads them.
    That is also why there is no `__init__` here — pydantic owns construction, and a
    mixin `__init__` would win the MRO and break it.
    """

    if TYPE_CHECKING:  # pragma: no cover
        # For the type checker only, and the guard is load-bearing rather than tidy.
        # Pydantic v2 collects a model's fields from the annotations of every class in
        # the MRO, mixins included — measured on 2.12.5: annotating these for real here
        # reorders the composed model's fields, and any name this mixin declared that
        # `_retriever_class` did not would become a *required* field of it, demanded by
        # a class that owns no construction and can give it no default. Under
        # `TYPE_CHECKING` nothing reaches `__annotations__` at runtime, so the composed
        # class stays the single declaration site.
        memory: Any
        tenant: Any
        user: Any
        agent: Any
        session: Any
        k: int
        min_score: float
        as_of: Any
        memory_types: Any
        include_episodes: bool

    def _memvara(self) -> tuple[Any, Any]:
        return bind(self.memory, tenant=self.tenant, user=self.user, agent=self.agent,
                    session=self.session)

    def _results(self, query: str) -> list[Any]:
        memory, scope = self._memvara()
        return memory.search(
            query, k=self.k, min_score=self.min_score, as_of=self.as_of,
            memory_types=self.memory_types, include_episodes=self.include_episodes,
            **scope_kw(scope))

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Any]:
        """The `BaseRetriever` hook. `run_manager` is accepted and unused.

        Named in the signature on purpose: `BaseRetriever.__init_subclass__` reads it to
        decide whether this is a modern retriever, and a subclass that omits it gets the
        legacy calling convention and a slower async path.
        """
        (document,) = require(f"{_PKG}.documents", "Document",
                              extra="langchain", needs=_NEEDS)
        return [
            document(page_content=result.text, metadata=result_metadata(result))
            for result in self._results(query)
        ]


# --- composition ------------------------------------------------------------------


@lru_cache(maxsize=None)
def _history_class(base: type) -> type:
    """`_ChatHistory` composed with `BaseChatMessageHistory`.

    Cached so the class is minted once and `isinstance` is stable — two calls returning
    two structurally identical classes is the kind of thing that passes every test and
    then fails one `isinstance` check in somebody's dispatch table.
    """

    class MemvaraChatMessageHistory(_ChatHistory, base):  # type: ignore[misc, valid-type]
        """Memvara as a LangChain chat message history, bound to one scope.

            MemvaraChatMessageHistory(mem, user="alice", session="s1")

        `messages` is the stored transcript and nothing more; `recall()` and `search()`
        on the same object reach the memory built from it. `clear()` raises unless
        `on_clear` says which of memvara's two deletions you meant.
        """

    return MemvaraChatMessageHistory


@lru_cache(maxsize=None)
def _retriever_class(base: type) -> type:
    """`_Retriever` composed with `BaseRetriever`, with the pydantic fields declared.

    The fields live here rather than on the mixin because `BaseRetriever` is a pydantic
    model and pydantic collects fields from the annotations present when the class is
    created — an annotation on a plain mixin is invisible to it.

    `arbitrary_types_allowed` is what lets `memory` hold an `Memvara`; `Any` alone would
    satisfy validation, but a subclass that annotates it precisely should not have to
    rediscover this.
    """

    class MemvaraRetriever(_Retriever, base):  # type: ignore[misc, valid-type]
        """Memvara as a LangChain retriever.

            MemvaraRetriever(memory=mem, user="alice", k=5)

        Every result is a `Document` whose `page_content` is the memory and whose
        `metadata` carries the triple, both time axes, the ranking explanation and the
        source turn ids — see `_common.result_metadata`. `as_of=` retrieves against
        what was believed at an instant, which is a property of the query and therefore
        survives an interface that never imagined it. `include_episodes=True` also
        returns raw turns, marked `kind="episode"` so nothing downstream can mistake one
        for a fact.
        """

        model_config = {"arbitrary_types_allowed": True}

        memory: Any
        tenant: Any = None
        user: Any = None
        agent: Any = None
        session: Any = None
        k: int = 8
        min_score: float = 0.0
        as_of: Any = None
        memory_types: Any = None
        include_episodes: bool = False

    return MemvaraRetriever


def __getattr__(name: str) -> Any:
    # PEP 562, same as `memvara.llm`: naming a class here must not import langchain-core,
    # or the numpy-only install stops being able to `import memvara`.
    if name == "MemvaraChatMessageHistory":
        (base,) = require(f"{_PKG}.chat_history", "BaseChatMessageHistory",
                          extra="langchain", needs=_NEEDS)
        return _history_class(base)
    if name == "MemvaraRetriever":
        (base,) = require(f"{_PKG}.retrievers", "BaseRetriever",
                          extra="langchain", needs=_NEEDS)
        return _retriever_class(base)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MemvaraChatMessageHistory", "MemvaraRetriever", "LangChainCompatError",
    "MemvaraTranscriptWarning",
]
