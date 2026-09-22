"""Two other memory systems as arms of the answer-quality run, both off by default.

    PYTHONPATH=. python3 demo/harness.py --reader stub --arm-mem0
    PYTHONPATH=. python3 demo/harness.py --reader stub --arm-supermemory \\
        --supermemory-key-file ~/.config/memvara/supermemory.key \\
        --supermemory-container memvara-demo-2026-09 \\
        --supermemory-ingest-path /v3/documents --supermemory-search-path /v3/search

`demo/baselines.py` has five arms and every one of them is either a control or memvara.
The roadmap's "What is still missing" asks for the part that is not here: a comparison
against another system **on answers**, rather than on architecture. `bench/compare.py`
compares against a reimplementation of mem0's documented design and `bench/mem0_real.py`
compares against the real package, but both score stored state — how many slots hold the
current value — and neither puts a reader in the seat. These two arms do.

Both are off unless asked for, and each needs something this repository cannot supply.
That is why they live here rather than in `ARMS`: an arm that cannot run on a clean
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

## Supermemory

Needs an account, and this repository does not have one. What it knows about Supermemory
is one endpoint: `memvara/compat/supermemory_import.py` reads `POST /v3/documents/list`,
which lists documents that already exist. An arm has to write a corpus and then query it,
and neither of those calls has ever been made from here.

So the arm ships with **no default write path and no default search path**. Both are
settings with no value, and turning the arm on without them is refused with the reason.
That is the honest position: a default that looked plausible would be a guess that reads
like a documented fact, and the next person to quote it would have no way to tell. Anybody
with an account can supply the two paths and run the arm; until somebody does, there is no
Supermemory row in `docs/BENCHMARKS.md` and the README says why.

It also refuses without an explicit container tag. A run writes one document per visible
turn — six hundred at `--corpus-scale 10` — and nothing here can take them back out of
somebody's account afterwards. `demo/hosted.py` refuses this machine's own memvara
credentials for the same reason and at the same point: before anything is sent.

Network access goes through the importer's own injectable `fetch`, `(url, key, body) ->
payload`, rather than a second HTTP client written here. That is how the importer is
tested without a network, and it is how a caller routes through a client of their own.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

from demo import baselines as bl
from demo.baselines import Arm, Context, Question, Turn, Write

from memvara import HashingEmbedder, Memvara
from memvara.compat.supermemory_import import (
    SupermemoryError,
    _http_fetch,
    read_supermemory_key,
)

__all__ = ["COMPETITOR_ARMS", "CompetitorUnavailable", "Mem0Arm", "SupermemoryArm",
           "build_competitors"]

#: The arms this module can add, in reporting order. Neither is in `ARMS`.
COMPETITOR_ARMS = ("mem0", "supermemory")

#: The one Supermemory endpoint anything in this repository has ever called. Named here
#: because the arm's refusal quotes it: it is the whole evidence base for what this
#: repository knows about that API, and it is a read.
SUPERMEMORY_VERIFIED_PATH = "/v3/documents/list"

#: Their documented host, shared with the importer so the two cannot drift.
SUPERMEMORY_BASE_URL = "https://api.supermemory.ai"

#: Documents per search request, and memories per mem0 search: `DEFAULT_K`, the same slot
#: budget `naive_rag` and both memvara arms are held to.
_K = bl.DEFAULT_K


class CompetitorUnavailable(SystemExit):
    """An optional arm was asked for without what it needs. Raised with the fix in it.

    A `SystemExit`, like every other refusal a `demo/` entry point makes — see
    `demo/hosted.py`'s credential checks. A missing package or an unconfigured account is
    something the person running the command has to go and fix, so the useful output is
    the sentence saying what to do, not a traceback through the arm that noticed.
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
        lines = [f"- {' '.join(str(row.get('memory', '')).split())}" for row in rows]
        text = bl.clip("\n".join([Memvara.RECALL_HEADER, *lines]), self.max_chars)
        return Context(arm="mem0", text=text, turns_visible=len(seen),
                       items_used=bl.count_entries(text))

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
        return f"demo{instant:%Y%m%dT%H%M}"

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


# --- Supermemory ------------------------------------------------------------------


class SupermemoryArm:
    """Supermemory as an arm, over the importer's injectable `fetch`.

    Every requirement is checked in `__init__`, which runs while the arms are being built
    — before any arm has produced a context and before a reader has been called. A run
    that discovered a missing setting on the first question would already have filled the
    mem0 stores and spent the reader's first calls.
    """

    #: There is no default for either. See the module docstring: this repository has
    #: called one Supermemory endpoint and it is a read, so it has nothing to default to.
    DEFAULT_INGEST_PATH: str | None = None
    DEFAULT_SEARCH_PATH: str | None = None

    def __init__(self, *, api_key: str, container: str,
                 ingest_path: str | None = DEFAULT_INGEST_PATH,
                 search_path: str | None = DEFAULT_SEARCH_PATH,
                 base_url: str = SUPERMEMORY_BASE_URL, k: int = _K,
                 max_chars: int = bl.MAX_CONTEXT_CHARS,
                 fetch: Callable[[str, str, Mapping[str, Any]],
                                 Mapping[str, Any]] | None = None,
                 key_problem: str = "") -> None:
        missing = []
        if not api_key:
            # `key_problem` carries why the key could not be read, so that a run missing
            # both the key and the endpoint paths names all four things at once instead
            # of finding the next one each time it is run.
            missing.append(key_problem or "an API key (--supermemory-key-file PATH)")
        if not container:
            missing.append("a container tag of its own (--supermemory-container TAG)")
        if not ingest_path:
            missing.append("the path that writes a document "
                           "(--supermemory-ingest-path PATH)")
        if not search_path:
            missing.append("the path that searches (--supermemory-search-path PATH)")
        if missing:
            raise CompetitorUnavailable(
                "--arm-supermemory needs " + "; ".join(missing) + ". The container keeps "
                "a run out of whatever space the account defaults to, because a run "
                "writes one document per visible turn and nothing here can remove them "
                "afterwards. The two paths have no default because this repository has "
                f"never called them: the importer reads {SUPERMEMORY_VERIFIED_PATH} and "
                "that is the whole of what it knows. Supply the paths from Supermemory's "
                "own documentation rather than letting this guess them.")
        self.api_key = api_key
        self.container = container
        self.ingest_url = base_url.rstrip("/") + ingest_path
        self.search_url = base_url.rstrip("/") + search_path
        self.base_url = base_url
        self.k = k
        self.max_chars = max_chars
        self.fetch = fetch or _http_fetch
        #: Containers whose corpus has been written. Same reasoning as `Mem0Arm`, and
        #: here it also bounds what a run costs somebody's account.
        self.filled: set[str] = set()

    def container_for(self, question: Question) -> str:
        """One container per question instant, under the tag this run was given.

        A single container cannot serve two instants, and the reason is not hypothetical.
        `demo/harness.py` asks questions in the order `demo/scenario.py` lists them, which
        is **not** `asked_at` order: the eight August questions come first and the April
        one is ninth. Sharing one container, the April question would search a store that
        already held the whole history, and would read turns from four months after it was
        asked — the cutoff every other arm keeps, silently broken for this one. It would
        also write the overlapping turns a second time, into an account this code cannot
        clean up. Neither failure shows in the report. `demo/hosted.py` splits hosted
        scopes by instant for the same reason.
        """
        return f"{self.container}-{question.asked_at:%Y%m%dT%H%M}"

    def __call__(self, question: Question, turns: Sequence[Turn]) -> Context:
        seen = bl.visible_turns(question, turns)
        container = self.container_for(question)
        self._fill(container, seen)
        found = self._call(self.search_url,
                           {"q": question.text, "limit": self.k,
                            "containerTags": [container]})
        rows = found.get("results") or found.get("memories") or []
        lines = [f"- {' '.join(str(self._text_of(row)).split())}" for row in rows
                 if self._text_of(row)]
        text = bl.clip("\n".join([Memvara.RECALL_HEADER, *lines]), self.max_chars)
        return Context(arm="supermemory", text=text, turns_visible=len(seen),
                       items_used=bl.count_entries(text))

    def _fill(self, container: str, seen: Sequence[Turn]) -> None:
        """Write this instant's visible turns into its own container, once."""
        if container in self.filled:
            return
        for turn in seen:
            self._call(self.ingest_url, {
                "content": f"[{turn.at:%Y-%m-%d}] {turn.role}: "
                           f"{' '.join(turn.text.split())}",
                "containerTags": [container],
            })
        self.filled.add(container)

    def _call(self, url: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        """One request, with the key handed to the transport and never put in the body."""
        try:
            return self.fetch(url, self.api_key, body)
        except SupermemoryError as exc:
            raise CompetitorUnavailable(f"Supermemory refused {url}: {exc}") from None

    @staticmethod
    def _text_of(row: Any) -> str:
        """The text of one returned document, under whichever key carries it."""
        if not isinstance(row, Mapping):
            return ""
        for key in ("content", "memory", "summary", "title"):
            value = row.get(key)
            if value:
                return str(value)
        return ""

    def note(self) -> str:
        return (f"  supermemory arm: {self.base_url}, containers {self.container}-<question "
                f"instant>, one per instant, write {self.ingest_url}, search "
                f"{self.search_url}. The endpoint paths were supplied on the command line; "
                f"this repository has only ever called {SUPERMEMORY_VERIFIED_PATH}.")


# --- the setup step ---------------------------------------------------------------


def build_competitors(args: Any) -> tuple[dict[str, Arm], list[str]]:
    """The optional arms this run asked for, and the lines the report prints about them.

    Called from `demo/harness.py`'s `build_arms`, which is where every arm and store is
    built. Returns an empty mapping and no notes when neither flag was given, so the
    default run is the five arms it has always been.
    """
    arms: dict[str, Arm] = {}
    notes: list[str] = []
    if getattr(args, "arm_mem0", False):
        mem0 = Mem0Arm()
        mem0.check()
        arms["mem0"] = mem0
        notes.append(mem0.note())
    if getattr(args, "arm_supermemory", False):
        key, key_problem = _supermemory_key(args.supermemory_key_file)
        supermemory = SupermemoryArm(
            api_key=key, key_problem=key_problem,
            container=args.supermemory_container or "",
            ingest_path=args.supermemory_ingest_path,
            search_path=args.supermemory_search_path,
            base_url=args.supermemory_base_url or SUPERMEMORY_BASE_URL)
        arms["supermemory"] = supermemory
        notes.append(supermemory.note())
    return arms, notes


def _supermemory_key(path: str | None) -> tuple[str, str]:
    """The key from the file at `path`, or the reason there is none.

    Read at run time and never stored anywhere but in the arm. `read_supermemory_key`
    already reads the plugin's file and raises with the fix in the message, so the
    no-path case is its job; a `--supermemory-key-file` is read the same way, which is
    what lets a demo use a key that is not the one the plugin signed in with.

    Returns the reason rather than raising it, so that the arm can put it in the one
    refusal that lists everything missing. Raising here would report the key on the first
    run and the container and the two paths on the second.
    """
    try:
        return read_supermemory_key(path), ""
    except SupermemoryError as exc:
        return "", f"an API key (--supermemory-key-file PATH): {exc}"
