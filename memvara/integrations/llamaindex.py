"""LlamaIndex: a long-term memory block and a retriever, mapped onto memvara.

Written against **llama-index-core 0.13+**, where `Memory` composes a short-term chat
buffer with a list of long-term `memory_blocks`. Nothing here imports it until you ask
for a class.

**Which LlamaIndex abstraction is memvara?** There are three candidates and only one of
them is honest.

``BaseMemoryBlock`` — yes, this one
    A memory block is asked "here are the recent messages, produce the content that
    should be in the prompt" (`_aget`) and "here are messages, absorb them" (`_aput`).
    That is memvara's `recall()` and `add()` with the arguments already in the right
    order. It is also the only one of the three where the *write path* is memvara's: the
    hash dedupe, the salience gate, the rule extractor and the batched model call all
    run, and `WriteReceipt.llm_calls` is usually zero.

``BaseMemory`` — no. It is a chat buffer.
    `get`/`put`/`set`/`get_all`/`reset` over `ChatMessage`, i.e. the transcript. Memvara
    stores reconciled facts; standing in for the buffer would mean either handing back a
    transcript it does not maintain, or handing back facts dressed as messages. Use the
    composable `Memory` with an `MemvaraMemoryBlock` in `memory_blocks=` — that is the
    supported shape, and `as_chat_memory()` says so.

``BasePydanticVectorStore`` — no, and this is the interesting refusal.
    `query()` receives a `query_embedding` computed by LlamaIndex's `embed_model`, and
    `add(nodes)` receives nodes carrying vectors from that same model. Memvara's index
    binds to *its own* embedder (`EmbedderMismatchError` exists precisely to stop two
    spaces mixing), and its ranking is BM25 fused with vectors and rescored by a
    per-predicate half-life. Standing in as a vector store means one of two things: drop
    the incoming vectors and re-embed, which is unanswerable because the query *text*
    never arrives at that interface — or store them, at which point memvara is a numpy
    matmul with extra steps and every reason to use it is gone. `as_vector_store()`
    raises and names `MemvaraRetriever`, which gets the query as a string and keeps all
    of it.

What is lost through `BaseMemoryBlock`: `_aget` returns **a string**, so scores,
provenance and `as_of` have nowhere to go. `MemvaraMemoryBlock.search()` and `.history()`
are the escape hatch, and `MemvaraRetriever` is the same retrieval with the structure
intact.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Sequence

from ..types import Episode
from ._common import IntegrationError, bind, require, result_metadata, scope_kw

_PKG = "llama_index.core"
_NEEDS = "llama-index-core>=0.13"

_NO_VECTOR_STORE = (
    "Memvara is not a vector store, and standing in for one would throw away the reason "
    "to use it.\n\n"
    "BasePydanticVectorStore.query() is handed a query_embedding produced by "
    "LlamaIndex's embed_model — the query text never reaches this interface — and "
    "add(nodes) is handed nodes already carrying vectors from that model. Memvara's "
    "index belongs to its own embedder (see memvara.EmbedderMismatchError), and its "
    "ranking fuses BM25 with vectors and rescores by a per-predicate half-life. Serving "
    "a foreign embedding means either re-embedding text this interface does not give "
    "us, or degrading to cosine top-k with none of the above.\n\n"
    "Use memvara.integrations.llamaindex.MemvaraRetriever instead: it implements "
    "BaseRetriever, receives the query as a string, and returns NodeWithScore carrying "
    "the triple, both time axes and the ranking explanation in node metadata."
)

_NO_CHAT_MEMORY = (
    "BaseMemory is LlamaIndex's chat buffer — get/put/set/get_all/reset over "
    "ChatMessage — and memvara's unit of memory is a reconciled bitemporal claim, not a "
    "message. Standing in for the buffer would mean either serving a transcript memvara "
    "does not maintain, or serving facts dressed as messages so the agent cannot tell "
    "what was said from what is believed.\n\n"
    "The supported shape is the composable one, where memvara is long-term memory beside "
    "the buffer rather than instead of it:\n"
    "    from llama_index.core.memory import Memory\n"
    "    from memvara.integrations.llamaindex import MemvaraMemoryBlock\n"
    "    memory = Memory.from_defaults(\n"
    "        session_id='s1',\n"
    "        memory_blocks=[MemvaraMemoryBlock(memory=mem, user='alice')],\n"
    "    )"
)


class LlamaIndexCompatError(IntegrationError):
    """A LlamaIndex call with no honest translation onto memvara."""


def _text_of(message: Any) -> str:
    """The text of a `ChatMessage`. `content` is `None` for a message with no text."""
    return str(getattr(message, "content", None) or "")


def _role_of(message: Any) -> str:
    """Memvara's role for a `ChatMessage`.

    `MessageRole` is a `str` enum whose values are already memvara's vocabulary
    ("user", "assistant", "system", "tool"), so this is an unwrap rather than a mapping
    — and the ones that are not ("chatbot", "model", "developer") are stored as
    themselves rather than folded into a neighbour.
    """
    role = getattr(message, "role", None)
    return str(getattr(role, "value", role) or "user")


# --- the memory block -------------------------------------------------------------


class _MemoryBlock:
    """`MemvaraMemoryBlock`'s behaviour, with no LlamaIndex in sight.

    A mixin, for the same reason as in the LangChain adapter: `BaseMemoryBlock` cannot
    be imported at module scope without making llama-index-core a hard dependency of
    `import memvara`. It defines no `__init__` — `BaseMemoryBlock` is a pydantic model
    and owns construction; the fields are declared on the composed class.

    The two the framework calls (`_aget`, `_aput`) cross to memvara through
    `asyncio.to_thread`. The interface is async and the library is not, and running a
    synchronous `encode()` plus a SQLite write straight from `_aput` would block the
    event loop for exactly as long as the write takes — see `memvara.aio`, which exists
    for this. `search` and `history` stay synchronous because they are escape hatches
    an application calls on its own terms, not hooks the framework awaits.
    """

    def _memvara(self) -> tuple[Any, Any]:
        return bind(self.memory, tenant=self.tenant, user=self.user, agent=self.agent,
                    session=self.session)

    def _query_text(self, messages: Sequence[Any]) -> str:
        """The retrieval query, from the tail of the conversation.

        A window rather than the last message alone, because "and what about her?"
        retrieves nothing on its own. `VectorMemoryBlock` makes the same choice, so a
        block swapped in for that one behaves the way its author expected.
        """
        window = messages[-self.context_window:] if self.context_window > 0 else messages
        return "\n".join(t for t in (_text_of(m) for m in window) if t)

    async def _aget(self, messages: Sequence[Any] | None = None,
                    **block_kwargs: Any) -> str:
        """The block's contribution to the prompt: memvara's `recall()` for this turn.

        Returns `""` when there is nothing to query on, which is the block contract's
        way of saying "contribute nothing" — an empty header would otherwise appear in
        the prompt announcing a memory section with no memories under it.

        The framing survives the crossing and matters: `recall()` labels its output as
        retrieved data rather than instructions and flattens each memory to one line, so
        a stored sentence cannot forge prompt structure around itself. A block that
        joined the claims itself would lose that.
        """
        if not messages:
            return ""
        query = self._query_text(messages)
        if not query:
            return ""
        memory, scope = self._memvara()
        return await asyncio.to_thread(
            memory.recall, query, k=self.k, min_score=self.min_score,
            include_episodes=self.include_episodes, **scope_kw(scope))

    async def _aput(self, messages: Sequence[Any]) -> None:
        """Absorb messages through memvara's write path.

        One `add()` for the batch: extraction is batched, so a call per message would
        pay a model call per message and throw away the property the write path is for.
        The receipt lands on `last_receipt` because the interface returns `None` and
        `llm_calls` is the number this library exists to drive down.
        """
        memory, scope = self._memvara()
        turns = [Episode(content=text, scope=scope, role=_role_of(message))
                 for message in messages if (text := _text_of(message))]
        if not turns:
            return
        self.last_receipt = await asyncio.to_thread(memory.add, turns)

    # -- escape hatches ------------------------------------------------------

    def search(self, query: str, **kw: Any) -> list[Any]:
        """`Memvara.search` at this block's scope — scores, provenance and `as_of=`.

        `_aget` returns a string because that is what a memory block returns; everything
        structured about a result dies at that boundary. This is where it lives.
        """
        memory, scope = self._memvara()
        return memory.search(query, **scope_kw(scope), **kw)

    def history(self, subject: str, predicate: str) -> list[Any]:
        """`Memvara.history` at this block's scope: every value the slot ever held."""
        memory, scope = self._memvara()
        return memory.history(subject, predicate, **scope_kw(scope))


# --- retrieval --------------------------------------------------------------------


class _Retriever:
    """`MemvaraRetriever`'s behaviour, with no LlamaIndex in sight.

    `llama_index.core.retrievers.BaseRetriever` is a plain ABC with a real `__init__`
    (callback manager, object map), so unlike the memory block this mixin *does*
    construct, and hands the framework's own arguments up the MRO.
    """

    def __init__(self, memory: Any, *, tenant: str | None = None, user: str | None = None,
                 agent: str | None = None, session: str | None = None, k: int = 8,
                 min_score: float = 0.0, as_of: Any = None, memory_types: Any = None,
                 include_episodes: bool = False, **retriever_kwargs: Any) -> None:
        self.memory, self.scope = bind(memory, tenant=tenant, user=user, agent=agent,
                                       session=session)
        self.k = k
        self.min_score = min_score
        #: Retrieval against what was believed at an instant. A LlamaIndex retriever has
        #: no way to express this and does not need one — it is a property of the query,
        #: so it rides on the object and survives an interface that never imagined it.
        self.as_of = as_of
        self.memory_types = memory_types
        self.include_episodes = include_episodes
        super().__init__(**retriever_kwargs)

    def _retrieve(self, query_bundle: Any) -> list[Any]:
        """The `BaseRetriever` hook. Takes the query as a *string*, which is the point.

        `query_bundle.embedding` is deliberately ignored: if LlamaIndex has pre-embedded
        the query with its own model, that vector is not in memvara's space, and using it
        would be the vector-store mistake `as_vector_store` refuses.
        """
        node_with_score, text_node = require(
            f"{_PKG}.schema", "NodeWithScore", "TextNode", extra="llama-index",
            needs=_NEEDS)
        query = getattr(query_bundle, "query_str", None)
        query = query if query is not None else str(query_bundle)
        results = self.memory.search(
            query, k=self.k, min_score=self.min_score, as_of=self.as_of,
            memory_types=self.memory_types, include_episodes=self.include_episodes,
            **scope_kw(self.scope))
        return [
            node_with_score(
                node=text_node(text=result.text, id_=metadata["memvara_id"],
                               metadata=metadata),
                score=result.score,
            )
            for result, metadata in ((r, result_metadata(r)) for r in results)
        ]


# --- refusals ---------------------------------------------------------------------


def as_vector_store(memory: Any) -> Any:
    """Always raises. Memvara is not a vector store — see `_NO_VECTOR_STORE`.

    A function that only raises, rather than nothing at all, because somebody porting a
    LlamaIndex app *will* look for this name, and finding a refusal that names
    `MemvaraRetriever` is better than finding nothing and writing a bad one.
    """
    raise LlamaIndexCompatError(_NO_VECTOR_STORE)


def as_chat_memory(memory: Any) -> Any:
    """Always raises. `BaseMemory` is the chat buffer — see `_NO_CHAT_MEMORY`."""
    raise LlamaIndexCompatError(_NO_CHAT_MEMORY)


# --- composition ------------------------------------------------------------------


@lru_cache(maxsize=None)
def _block_class(base: type) -> type:
    """`_MemoryBlock` composed with `BaseMemoryBlock`, with the pydantic fields declared.

    `name` gets a default here and has none on the base: a block is identified by name
    in the prompt template, and making every caller type `name="memvara"` is friction
    with no decision behind it.
    """

    class MemvaraMemoryBlock(_MemoryBlock, base):  # type: ignore[misc, valid-type]
        """Memvara as a LlamaIndex long-term memory block.

            Memory.from_defaults(session_id="s1",
                                 memory_blocks=[MemvaraMemoryBlock(memory=mem,
                                                                  user="alice")])

        `_aget` contributes `Memvara.recall()` for the current turn; `_aput` runs the
        batch through memvara's write path. Scores, provenance and time travel do not fit
        through a block's string return — `search()` and `history()` on this object are
        where they live.
        """

        model_config = {"arbitrary_types_allowed": True}

        name: str = "memvara"
        description: Any = (
            "Long-term memory: reconciled facts about the user, contradictions already "
            "retired."
        )
        memory: Any = None
        tenant: Any = None
        user: Any = None
        agent: Any = None
        session: Any = None
        k: int = 8
        min_score: float = 0.0
        include_episodes: bool = False
        context_window: int = 3
        #: What the last `_aput` cost. Set at runtime, so it is a declared field rather
        #: than an attribute pydantic would reject.
        last_receipt: Any = None

    return MemvaraMemoryBlock


@lru_cache(maxsize=None)
def _retriever_class(base: type) -> type:
    """`_Retriever` composed with `BaseRetriever`. Cached so `isinstance` is stable."""

    class MemvaraRetriever(_Retriever, base):  # type: ignore[misc, valid-type]
        """Memvara as a LlamaIndex retriever.

            index_or_engine.as_query_engine(retriever=MemvaraRetriever(mem, user="alice"))

        Returns `NodeWithScore`, each node carrying the triple, both time axes, the
        ranking explanation and the source turn ids in `metadata` — see
        `_common.result_metadata`. `as_of=` retrieves against what was believed at an
        instant. `include_episodes=True` also returns raw turns, marked
        `kind="episode"` in metadata so nothing downstream mistakes one for a fact.
        """

    return MemvaraRetriever


def __getattr__(name: str) -> Any:
    # PEP 562, same as `memvara.llm`: naming a class here must not import
    # llama-index-core, or the numpy-only install stops being able to `import memvara`.
    if name == "MemvaraMemoryBlock":
        (base,) = require(f"{_PKG}.memory", "BaseMemoryBlock", extra="llama-index",
                          needs=_NEEDS)
        return _block_class(base)
    if name == "MemvaraRetriever":
        (base,) = require(f"{_PKG}.retrievers", "BaseRetriever", extra="llama-index",
                          needs=_NEEDS)
        return _retriever_class(base)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MemvaraMemoryBlock", "MemvaraRetriever", "LlamaIndexCompatError",
    "as_vector_store", "as_chat_memory",
]
