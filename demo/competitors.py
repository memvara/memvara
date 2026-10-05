"""Another memory system, mem0, as an arm of the answer-quality run. Off by default.

    PYTHONPATH=. python3 demo/harness.py --reader stub --arm-mem0

`demo/baselines.py` has five arms and every one of them is either a control or memvara.
The roadmap's "What is still missing" asks for the part that is not here: a comparison
against another system **on answers**, rather than on architecture. `bench/compare.py`
compares against a reimplementation of mem0's documented design and `bench/mem0_real.py`
compares against the real package, but both score stored state — how many slots hold the
current value — and neither puts a reader in the seat. This arm does.

The arm is off unless asked for, because it needs a package that a clean checkout does not
install. That is why it lives here rather than in `ARMS`: an arm that cannot run on a clean
checkout must not be able to break the offline run that CI depends on.

## mem0

Needs the `mem0ai` package, and nothing else — no key, no network. It is driven by the
same oracle `bench/mem0_real.py` uses: a perfect extractor in mem0's own shape, emitting
the ground-truth facts for the turn being added. So mem0 gets the same facts
`memvara_structured` gets, with 100% extraction recall and no hallucinations, which is
strictly better than any real model would do. Anything it gets wrong here, it gets wrong
because of how it is built.

**What it is built like matters most on this corpus.** mem0 2.x's add path emits only
`ADD` — its extraction prompt says "Your sole operation is ADD" — so a new value is linked
to the one it contradicts rather than retiring it, and both stay live. Six of the twenty
questions ask about a value that moved. A store that never supersedes holds the old value
and the new one, and the reader has to choose between them with nothing to choose on. That
is not a defect in this arm; it is the measurement, and it is what the report's trap column
counts.

Two things are deliberately identical to the memvara arms, because a difference in either
would make the comparison a comparison of budgets. The retrieval budget is `DEFAULT_K`
slots capped at `MAX_CONTEXT_CHARS`, and the block is rendered under `recall()`'s own
header, so the prompts differ in which values they carry and in nothing else.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from demo import baselines as bl
from demo.baselines import Arm, Context, Question, Turn, Write

from memvara import HashingEmbedder, Memvara

__all__ = ["CompetitorUnavailable", "Mem0Arm", "build_competitors"]

#: Memories per mem0 search: `DEFAULT_K`, the same slot budget
#: `naive_rag` and both memvara arms are held to.
_K = bl.DEFAULT_K


def stored_notes(arm: str, notes: Sequence[str], *, seen: Sequence[Turn],
                 max_chars: int) -> Context:
    """One arm's retrieved notes as the context a reader sees.

    An arm here returns a list of stored sentences and has to turn it into the shape the
    memvara arms produce: `recall()`'s own header, one bullet per note, the shared character
    cap, and the entry count taken from the rendered text after the cap rather than from the
    list that went in. Written once because the three parts are each load-bearing and each easy to
    get subtly differently. The header carries the framing that tells a model the lines
    below are reference data and not instructions, and it is `recall()`'s so that a block
    of stored notes looks the same whichever system produced it — the harness blinds the
    reader to which arm it is reading, and that blinding is only as good as the prompts
    look alike.
    """
    lines = [f"- {' '.join(str(note).split())}" for note in notes if str(note).strip()]
    text = bl.clip("\n".join([Memvara.RECALL_HEADER, *lines]), max_chars)
    return Context(arm=arm, text=text, turns_visible=len(seen),
                   items_used=bl.count_entries(text))


class CompetitorUnavailable(SystemExit):
    """An optional arm was asked for without what it needs. Raised with the fix in it.

    A `SystemExit`, like every other refusal a `demo/` entry point makes — see
    `demo/hosted.py`'s credential checks. A missing package is something the person running
    the command has to go and fix, so the useful output is the sentence saying what to do,
    not a traceback through the arm that noticed.
    """


# --- mem0 -------------------------------------------------------------------------


def _mem0_providers() -> tuple[Any, Any, Any, Any, Any]:
    """Import mem0 and build the oracle providers against its own base classes.

    Imported here rather than at module scope for two reasons. `demo/harness.py` imports
    this module unconditionally, so a missing optional dependency must not stop `--help`
    from working. And the oracle classes subclass mem0's bases, which cannot be done at a
    module scope that has to import without mem0 installed.

    The providers are built fresh per call and close over nothing global.
    `bench/mem0_real.py` keeps its oracle state in a module-level object because it lets
    mem0's factory construct the providers by import path; here they are constructed
    directly and assigned, so the state can be closed over instead. Two arms in one
    process then cannot overwrite each other's facts, which a module global would allow.

    Three environment defaults are set first, for the same reasons `bench/mem0_real.py`
    sets them, and each was found by running this rather than by reading mem0:

    * mem0 starts a PostHog client at import time. A benchmark should not phone home
      about itself, so both telemetry switches go off before the import.
    * `Memory.from_config` builds the *stock* providers before there is anywhere to hand
      a replacement in, and the stock OpenAI embedder refuses to construct without a key
      in the environment. The placeholder is never used: both providers are replaced on
      the next two lines and nothing calls them in between, so no request is made.

    `setdefault`, never assignment. A run that is also using `--reader openai` has a real
    key in `OPENAI_API_KEY`, and overwriting it would break the reader in the same
    process to satisfy a client this arm is about to throw away.
    """
    import os

    os.environ.setdefault("MEM0_TELEMETRY", "False")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("OPENAI_API_KEY", "sk-not-used-by-this-arm")
    try:
        from mem0 import Memory
        from mem0.configs.embeddings.base import BaseEmbedderConfig
        from mem0.configs.llms.base import BaseLlmConfig
        from mem0.embeddings.base import EmbeddingBase
        from mem0.llms.base import LLMBase
    except ImportError as exc:
        raise CompetitorUnavailable(
            f"--arm-mem0 needs the mem0 package, which is not importable ({exc}). "
            "Install it with:  pip install mem0ai   (33 packages; memvara's install is "
            "2). It is not a dependency of this library and the other arms do not need "
            "it.") from None
    return Memory, BaseLlmConfig, BaseEmbedderConfig, LLMBase, EmbeddingBase


class Mem0Arm:
    """The real mem0 package as an arm, fed the facts the structured memvara arm is fed.

    One store per question *instant*: what a store may hold depends on `asked_at` and
    never on the question's wording, and twenty questions share three instants here.
    Filling one is one `add` per visible turn, each with an embed, so rebuilding per
    question would multiply the work by about seven and change no context.
    """

    #: The value mem0 files every memory under. One customer, as the corpus has.
    USER = "customer"

    def __init__(self, *, facts: Sequence[Write] = bl.SUPPORT_FACTS, k: int = _K,
                 max_chars: int = bl.MAX_CONTEXT_CHARS, dim: int = bl.EMBED_DIM) -> None:
        self.facts = tuple(facts)
        self.k = k
        self.max_chars = max_chars
        self.dim = dim
        #: One mem0 `Memory` per question instant, kept for the run. Public because the
        #: tests assert on what reached the store rather than on what the oracle intended.
        self.stores: dict[Any, Any] = {}

    def __call__(self, question: Question, turns: Sequence[Turn]) -> Context:
        """The `Arm` signature: one question, the whole history, one context."""
        seen = bl.visible_turns(question, turns)
        store = self._store_for(question, turns)
        found = store.search(question.text, filters={"user_id": self.USER},
                             top_k=self.k)
        rows = found["results"] if isinstance(found, dict) else found
        return stored_notes("mem0", [row.get("memory", "") for row in rows],
                            seen=seen, max_chars=self.max_chars)

    def _store_for(self, question: Question, turns: Sequence[Turn]) -> Any:
        """The store for this question's instant, filled the first time it is asked for."""
        instant = question.asked_at
        if instant in self.stores:
            return self.stores[instant]

        Memory, llm_config, embed_config, llm_base, embed_base = _mem0_providers()
        embedder = HashingEmbedder(dim=self.dim)
        # The facts the desk had recorded by this instant, indexed by the instant of the
        # turn that carried each one. Turn instants are unique in this corpus, so a turn
        # identifies at most one group of facts and the oracle cannot emit a fact twice.
        # Keying on anything wider — the prompt text, say — is what produced the firehose
        # bug `bench/mem0_real.py` documents.
        due: dict[Any, list[Write]] = {}
        for fact in bl.visible_facts(question, self.facts):
            due.setdefault(fact.at, []).append(fact)
        #: The facts for the turn being added right now. The loop below fills it and the
        #: oracle empties it — see `OracleLLM.generate_response` for why it is a cell the
        #: caller sets rather than something read out of the prompt.
        current: list[Write] = []

        class OracleLLM(llm_base):  # type: ignore[misc, valid-type]
            """A perfect extractor in mem0's shape, for the turn being added right now.

            Returns the JSON mem0's additive extraction prompt asks for. The sentence it
            emits is the fact's own `text` — the same string `memvara_structured` indexes
            — so both systems hold the same words and a difference between their rows is
            a difference in which values survived, not in how they were phrased.

            **It reads a cell the caller set, never the prompt, and that is load-bearing.**
            mem0's additive prompt embeds the last k messages, so a turn appears in the
            prompt of every later add in its window. An oracle that looked for known turns
            in the prompt text would re-extract each of them and emit every fact many
            times over — `bench/mem0_real.py` hit exactly that, measured mem0 under a
            firehose no real extractor would produce, and got numbers that flattered
            memvara. The first version of this arm keyed on `messages[-1]["content"]`,
            which is not the turn either: it is the whole rendered prompt, so nothing
            matched and mem0 was handed an empty store. Both mistakes are quiet.

            Emptying the cell is what bounds it to one emission per `add`, whatever mem0
            does with the response or however many times it calls back.
            """

            def generate_response(self, messages: Any = None, **_: Any) -> str:
                emitted, current[:] = list(current), []
                return json.dumps({"memory": [{"text": f.text, "event": "ADD"}
                                              for f in emitted]})

        class OracleEmbedder(embed_base):  # type: ignore[misc, valid-type]
            """memvara's `HashingEmbedder`, so neither system wins on vector quality."""

            def embed(self, text: str, memory_action: Any = None) -> list[float]:
                return list(embedder.encode([text])[0].tolist())

            def embed_batch(self, texts: Sequence[str],
                            memory_action: Any = None) -> list[list[float]]:
                return [list(v.tolist()) for v in embedder.encode(list(texts))]

        # `MemoryConfig` validates the provider name against a hardcoded list in a
        # pydantic validator, separate from the factory registry, so registering a
        # provider is not enough to get past construction. Build with stock providers and
        # swap the two components afterwards: the add path, the prompts, the vector store
        # and the history database are then all the real thing.
        store = Memory.from_config({
            "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
            "embedder": {"provider": "openai",
                         "config": {"model": "text-embedding-3-small",
                                    "embedding_dims": self.dim}},
            "vector_store": {"provider": "qdrant",
                             # `path` is what decides whether this is really in memory,
                             # and `on_disk: False` alone is not enough. Left unset,
                             # mem0 hands qdrant a default storage folder, qdrant takes
                             # a lock on it, and the second store this arm builds in the
                             # same process dies with "already accessed by another
                             # instance of Qdrant client". One question hid it and three
                             # instants found it. `:memory:` is the documented spelling
                             # for no folder at all, and two clients built with it are
                             # independent, which is what a per-instant store needs.
                             "config": {"collection_name": self._collection(instant),
                                        "embedding_model_dims": self.dim,
                                        "path": ":memory:",
                                        "on_disk": False}},
        })
        store.llm = OracleLLM(llm_config(model="oracle"))
        store.embedding_model = OracleEmbedder(embed_config(model="oracle"))

        for turn in bl.visible_turns(question, turns):
            # Per turn, which is how an agent loop actually calls mem0, and how it is
            # charged. The cell hands the oracle this turn's facts and only this turn's;
            # it is cleared again afterwards so that a turn carrying none cannot inherit
            # the previous turn's.
            current[:] = due.get(turn.at, [])
            store.add(turn.text, user_id=self.USER)
            current.clear()

        self.stores[instant] = store
        return store

    @staticmethod
    def _collection(instant: Any) -> str:
        """A qdrant collection per instant. In memory, so the name only has to be unique."""
        return f"demo{bl.instant_tag(instant)}"

    def check(self) -> None:
        """Import mem0 now, so a missing package is refused while the arms are built.

        Without this the first `import mem0` would happen on the first context, which is
        inside `plan()` — after the run has started, and after any other arm asked for in
        the same command has already done its work. The cost of finding out early is one
        import that the run needs anyway.
        """
        _mem0_providers()

    def note(self) -> str:
        """The line the report prints, naming what this arm was."""
        from importlib.metadata import PackageNotFoundError, version as installed

        try:
            version = installed("mem0ai")
        except PackageNotFoundError:
            version = "unknown version"
        return (f"  mem0 arm: the mem0ai package {version}, driven by the same "
                f"ground-truth facts the memvara_structured arm gets, with a perfect "
                f"extraction oracle (demo/competitors.py). Its add path never supersedes, "
                f"so a value that moved is held beside the one that replaced it.")


# --- the setup step ---------------------------------------------------------------


def build_competitors(args: Any) -> tuple[dict[str, Arm], list[str]]:
    """The optional arms this run asked for, and the lines the report prints about them.

    Called from `demo/harness.py`'s `build_arms`, which is where every arm and store is
    built. Returns an empty mapping and no notes when `--arm-mem0` was not given, so the
    default run is the five arms it has always been.
    """
    arms: dict[str, Arm] = {}
    notes: list[str] = []
    if getattr(args, "arm_mem0", False):
        mem0 = Mem0Arm()
        mem0.check()
        arms["mem0"] = mem0
        notes.append(mem0.note())
    return arms, notes

