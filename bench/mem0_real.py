"""Head-to-head against the real mem0 package, not a reimplementation of it.

`bench/compare.py` measures memvara against `bench/baseline.py`, a reimplementation of
mem0's *documented* architecture. That was always the weakest claim in the README — a
benchmark where we wrote both competitors. This runs the actual `mem0ai` package.

    pip install mem0ai        # 33 packages; memvara's install is 2
    PYTHONPATH=. python3 bench/mem0_real.py

**Both systems are driven by the same perfect oracle**, so the comparison isolates
architecture from model quality. The oracle is deliberately generous to mem0: it extracts
the ground-truth fact from every turn with no misses and no hallucinations, which is
strictly better than any real model would do. Anything mem0 gets wrong here, it gets wrong
because of how it is built, not because the model was weak.

Fully offline. Qdrant runs in `:memory:`, and both systems embed with the same
`HashingEmbedder`, so neither gets an advantage from vector quality either.

## What reading the source turned up, before any measurement

mem0 2.x is not the architecture the memvara README was written against, and two of the
differences matter:

* **One LLM call per `add()`, not two.** 2.x uses a single `ADDITIVE_EXTRACTION_PROMPT`
  that receives existing memories alongside the new turn.
  `DEFAULT_UPDATE_MEMORY_PROMPT` still exists in `configs/prompts.py` and is no longer
  reached from the add path.
* **The add path never supersedes.** Every event it emits is `ADD` — the extraction
  prompt states "Your sole operation is ADD". Contradiction is handled by *linking* a new
  memory to the one it conflicts with, not by retiring anything. `update()` and `delete()`
  are explicit API calls the application has to decide to make.

The second is the interesting one, and it is why this file scores "stale values left
live". Under 2.x that number is not a retrieval failure — it is the documented design.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field

# Set before importing mem0: it starts a PostHog client at import time otherwise, and a
# benchmark should not phone home about itself.
os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# Never used — the oracle replaces both clients — but mem0 constructs the stock providers
# before we can swap them, and those constructors want a key present.
os.environ.setdefault("OPENAI_API_KEY", "sk-not-used-by-this-benchmark")

from memvara import Memvara, HashingEmbedder

from compare import NEEDLE_VALUE, ScriptedLLM, build_workload, make_extractor

EMBED_DIM = 256

try:
    from mem0 import Memory as Mem0Memory
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.configs.llms.base import BaseLlmConfig
    from mem0.embeddings.base import EmbeddingBase
    from mem0.llms.base import LLMBase
    from mem0.utils.factory import EmbedderFactory, LlmFactory
except ImportError as exc:  # pragma: no cover - the whole point is that this is optional
    print(f"\n  cannot import mem0: {exc}\n\n  This benchmark needs it:  pip install mem0ai\n")
    sys.exit(1)


# --- the shared oracle, wearing mem0's interfaces -------------------------------


@dataclass
class OracleState:
    """Module-level, because mem0 constructs its providers by import path and gives us
    no hook to pass anything in. The factory takes a string, not an instance."""

    extractor: object = None
    embedder: object = None
    current: str = ""          # the turn being added right now
    llm_calls: int = 0
    embed_calls: int = 0


STATE = OracleState()

#: mem0 embeds the extracted memory *text*, so the oracle has to find the turn that text
#: came from. Extraction is verbatim here, which makes the mapping exact.
_FACT_OF_TEXT: dict[str, list[dict]] = {}


class OracleLLM(LLMBase):
    """A perfect extractor in mem0's shape.

    `generate_response` returns the JSON `ADDITIVE_EXTRACTION_PROMPT` asks for. We do not
    attempt to be a language model: we return the ground-truth facts for the turn being
    added. mem0 therefore gets 100% extraction recall and 0% hallucination — a ceiling no
    real deployment reaches.

    **It keys off `STATE.current`, not the prompt text, and that is load-bearing.** The
    first version of this file scanned the whole prompt for known turns, which looked
    obviously correct and was not: mem0's additive prompt embeds `last_k_messages`, so
    every earlier turn in the window matched and got re-extracted. Each fact was emitted
    eleven times, mem0 was measured under a firehose of repeats no real extractor would
    produce, and the resulting numbers flattered memvara. A benchmark whose bug favours its
    author is the one to distrust most.
    """

    def generate_response(self, messages, tools=None, tool_choice="auto", **kwargs):
        STATE.llm_calls += 1
        out = []
        for f in STATE.extractor.get(STATE.current, []):
            text = f"user {f['predicate'].replace('_', ' ')} is {f['object']}"
            _FACT_OF_TEXT[text] = [f]
            out.append({"text": text, "event": "ADD"})
        return json.dumps({"memory": out})


class OracleEmbedder(EmbeddingBase):
    """memvara's `HashingEmbedder`, so neither system wins on vector quality."""

    def embed(self, text, memory_action=None):
        STATE.embed_calls += 1
        return STATE.embedder.encode([text])[0].tolist()

    def embed_batch(self, texts, memory_action=None):
        STATE.embed_calls += len(texts)
        return [v.tolist() for v in STATE.embedder.encode(list(texts))]


# `MemoryConfig` validates the provider name against a hardcoded list in a pydantic
# validator, separate from the factory registry — so registering a provider is not enough
# to get it past construction. We therefore build the Memory with stock providers and swap
# the two components afterwards. That injects test doubles without touching mem0's
# architecture: the add path, the prompts, the vector store and the history DB are all the
# real thing.


# --- scoring --------------------------------------------------------------------


def _predicate_of(text: str) -> str | None:
    facts = _FACT_OF_TEXT.get(text)
    return facts[0]["predicate"] if facts else None


def score_mem0(api, w) -> dict:
    rows = api.get_all(filters={"user_id": "alice"})
    rows = rows["results"] if isinstance(rows, dict) else rows
    by_pred: dict[str, list[str]] = {}
    for r in rows:
        text = r.get("memory", "")
        pred = _predicate_of(text)
        if pred:
            by_pred.setdefault(pred, []).append(_FACT_OF_TEXT[text][0]["object"])

    stale = correct = 0
    for pred, want in w.truth.items():
        got = by_pred.get(pred, [])
        correct += want in got
        stale += len([g for g in got if g != want])

    found = api.search(NEEDLE_VALUE, filters={"user_id": "alice"}, top_k=3)
    found = found["results"] if isinstance(found, dict) else found
    return {
        "correct": correct,
        "stale": stale,
        "live": len(rows),
        "needle": any(NEEDLE_VALUE in r.get("memory", "") for r in found),
    }


def score_memvara(mem: Memvara, w) -> dict:
    live = mem.get_all()
    by_pred: dict[str, list[str]] = {}
    for c in live:
        by_pred.setdefault(c.predicate, []).append(c.object)

    stale = correct = 0
    for pred, want in w.truth.items():
        got = by_pred.get(pred, [])
        correct += want in got
        stale += len([g for g in got if g != want])

    hits = [r.claim.object for r in mem.search(NEEDLE_VALUE, k=3)]
    return {"correct": correct, "stale": stale, "live": len(live),
            "needle": NEEDLE_VALUE in hits}


# --- the run --------------------------------------------------------------------


def run_mem0(w) -> tuple[dict, float, int]:
    STATE.extractor = w.facts
    STATE.embedder = HashingEmbedder(dim=EMBED_DIM)
    STATE.llm_calls = STATE.embed_calls = 0

    api = Mem0Memory.from_config({
        "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
        "embedder": {"provider": "openai",
                     "config": {"model": "text-embedding-3-small",
                                "embedding_dims": EMBED_DIM}},
        "vector_store": {"provider": "qdrant",
                         "config": {"collection_name": f"b{uuid.uuid4().hex[:8]}",
                                    "embedding_model_dims": EMBED_DIM,
                                    "on_disk": False}},
    })
    # The swap. Nothing downstream of here is a stand-in — this is mem0's own add path.
    api.llm = OracleLLM(BaseLlmConfig(model="oracle"))
    api.embedding_model = OracleEmbedder(BaseEmbedderConfig(model="oracle"))

    start = time.perf_counter()
    for turn in w.turns:
        # Per turn, which is how an agent loop actually calls it.
        STATE.current = turn
        api.add(turn, user_id="alice")
    elapsed = time.perf_counter() - start
    return score_mem0(api, w), elapsed, STATE.llm_calls


def run_memvara(w) -> tuple[dict, float, int]:
    llm = ScriptedLLM(make_extractor(w))
    mem = Memvara(embedder=HashingEmbedder(dim=EMBED_DIM), llm=llm, user="alice")
    start = time.perf_counter()
    mem.add(w.turns)
    elapsed = time.perf_counter() - start
    out = score_memvara(mem, w)
    mem.close()
    return out, elapsed, llm.calls


def _span(values) -> str:
    lo, hi = min(values), max(values)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def run(chitchat_ratio: int = 4, trials: int = 5) -> None:
    w = build_workload(chitchat_ratio=chitchat_ratio)
    slots = len(w.truth)

    print(f"\n  {len(w.turns)} turns, {w.fact_turns} carrying a durable fact, "
          f"{slots} distinct facts revised over time.")
    print("  Both systems: the same perfect extraction oracle, the same embedder,"
          "\n  no network, no API key.\n")

    # Repeated, because a single run of mem0 is not reproducible — see below. Reporting
    # one number would have meant reporting whichever one we happened to draw.
    m = [run_mem0(w) for _ in range(trials)]
    e = [run_memvara(w) for _ in range(trials)]

    def col(runs, key):
        return _span([r[0][key] for r in runs])

    rows = [
        ("LLM calls on the write path",
         _span([r[2] for r in m]), _span([r[2] for r in e])),
        (f"Current value stored correctly (of {slots})",
         f"{col(m, 'correct')}/{slots}", f"{col(e, 'correct')}/{slots}"),
        ("Stale values left live", col(m, "stale"), col(e, "stale")),
        ("Live rows in the store", col(m, "live"), col(e, "live")),
        ("Rare literal found (BM25 case)",
         "yes" if all(r[0]["needle"] for r in m) else "not always",
         "yes" if all(r[0]["needle"] for r in e) else "not always"),
        ("Wall clock, median",
         f"{sorted(r[1] for r in m)[trials // 2] * 1000:.0f} ms",
         f"{sorted(r[1] for r in e)[trials // 2] * 1000:.0f} ms"),
        ("Identical result every run",
         "no" if len({r[0]["correct"] for r in m}) > 1 else "yes",
         "no" if len({r[0]["correct"] for r in e}) > 1 else "yes"),
        ("Install size", "33 packages", "2 packages"),
    ]
    width = max(len(r[0]) for r in rows)
    print(f"  {'metric':<{width}}  {'mem0 2.x':>12}  {'memvara':>12}   ({trials} runs)")
    print(f"  {'-' * width}  {'-' * 12}  {'-' * 12}")
    for label, a, b in rows:
        print(f"  {label:<{width}}  {str(a):>12}  {str(b):>12}")

    print(f"""
  Read before quoting any of this:

  * mem0 is charged per turn and memvara receives the transcript in one `add()`, so
    the call-count row is partly an ingestion-granularity choice, not purely an
    architectural one. `bench/compare.py` reports the equal-granularity number.
  * The oracle gives mem0 perfect extraction. A real model would do worse.
  * "Stale values left live" is not a bug in mem0 2.x. Its add path emits only ADD
    and its extraction prompt says so outright — contradictions are *linked*, not
    retired. Whether that is right depends on whether your application asks "what
    is true now" or "what has this person ever said".
  * mem0's result is a *range* because it is not reproducible. The oracle returns
    byte-identical JSON every run and both systems use the same deterministic
    embedder, so there is no model variance here at all — and mem0 still lands on
    a different final state between runs. We did not isolate the cause inside
    mem0; we only established that it is not coming from the model or the
    embeddings, because in this harness neither varies.

    This is the "keyed lookup has no threshold to get wrong" claim, measured
    against the real package instead of argued against a reimplementation.

  * One workload, n=1, written by us. This shows a mechanism, not a verdict.
""")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
