"""Seed a store, or regenerate the hit corpus.

    python -m benchmarks.plugin_recall.seed --target supermemory --url http://localhost:6767
    python -m benchmarks.plugin_recall.seed --target memvara --db /tmp/bench.db
    python -m benchmarks.plugin_recall.seed --emit-cases benchmarks/plugin_recall/cases/v1_hits.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from . import emit_cases, facts

#: Both stores are namespaced so a benchmark run cannot be confused with real memory, and
#: so the seeded rows are findable afterwards by whoever has to clear them out.
CONTAINER = "plugin-recall-bench"


def seed_supermemory(url: str, container: str) -> int:
    written = 0
    for fact in facts():
        body = json.dumps({
            "content": fact["prose"],
            "containerTag": container,
            "metadata": {"sm_source": "plugin-recall-bench", "fact_id": fact["id"]},
        }).encode()
        request = urllib.request.Request(
            f"{url.rstrip('/')}/v3/documents", data=body,
            # An explicit User-Agent, always. The stdlib default is rejected outright by
            # some edges, and the 403 that comes back says nothing about the client's name
            # being the cause -- a trap this repository has already paid for once.
            headers={"Content-Type": "application/json", "User-Agent": "plugin-recall-bench/1"},
            method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status >= 300:
                raise RuntimeError(f"{fact['id']}: HTTP {response.status}")
        written += 1
    return written


def _ollama_llm(model: str, url: str):
    """memvara's OpenAI backend, pointed at a local Ollama.

    `MEMVARA_LLM` accepts only `none` and `anthropic`, so this cannot be reached from the
    environment -- but `memvara.llm.openai.OpenAILLM` takes a client, and Ollama serves an
    OpenAI-compatible API, so the library supports it even though the server config has no
    name for it. Constructed here rather than proposed as a new backend: that is a change
    to the core's configuration surface and belongs in its own review, not smuggled in
    through a benchmark.
    """
    from openai import OpenAI

    from memvara.llm.openai import OpenAILLM

    # Ollama ignores the key but the client requires one to be present.
    client = OpenAI(base_url=url, api_key="ollama")

    # Qwen3.5 reasons by default and the reasoning is not optional through the
    # OpenAI-compatible surface: asked to extract one sentence it spent 4,606 tokens and
    # 182 seconds thinking before answering, and with any sane `max_tokens` it returns an
    # empty `content` having spent the whole budget on `reasoning`. `reasoning_effort:
    # none` -- measured -- returns the same answer in 8 tokens.
    #
    # Injected by wrapping `create` because `OpenAILLM` builds its own call and the
    # parameter belongs to this deployment rather than to the library. Anything that
    # rejects the field (a real OpenAI endpoint on an older model) is retried without it,
    # so a wrapper added for one local model cannot break the backend it wraps.
    create = client.chat.completions.create

    def create_without_reasoning(**kwargs):
        try:
            return create(**kwargs, extra_body={"reasoning_effort": "none"})
        except Exception:
            return create(**kwargs)

    client.chat.completions.create = create_without_reasoning
    return OpenAILLM(model=model, client=client)


def seed_memvara(db: Path, *, llm_model: str = "", llm_url: str = "") -> int:
    # Built through the *server's* config, not `Memvara(path)`, so the store is created
    # with the same embedder the hook will later open it with. Seeding with the library
    # default instead produced a 384-dimensional store that the hook -- configured for
    # `hashing:512:3-5` -- refused to open, and `open_store()` answers that refusal with
    # `None`, which sends the hook to the hosted store instead. The benchmark then scored
    # a 0% hit rate against a store it had never read: every fact present, every write
    # successful, and the reader looking somewhere else entirely.
    import os

    from memvara.server.config import ServerConfig, build_memvara

    env = {**os.environ, "MEMVARA_DB": str(db), "MEMVARA_MODE": "local"}
    config = ServerConfig.from_env(env)
    if llm_model:
        # Constructed directly rather than through `build_memvara`, which hardcodes
        # `NullLLM() if config.llm == "none" else _anthropic()` -- `MEMVARA_LLM` has no
        # name for an OpenAI-compatible endpoint, so there is no environment that reaches
        # this. Everything else is copied from `build_memvara` verbatim, the embedder
        # above all: it decides whether the hook can open the store at all, and a mismatch
        # sends recall silently to the hosted store instead.
        #
        # Passed to the constructor, not assigned afterwards. `Memvara.__init__` builds
        # its write pipeline with `self.llm` bound at that moment (core.py:791), so
        # `memory.llm = ...` after the fact rewires nothing and every `add()` quietly
        # takes the deterministic path.
        from memvara import Memvara
        from memvara.server.config import _embedder, _registry

        memory = Memvara(
            config.path,
            llm=_ollama_llm(llm_model, llm_url),
            embedder=_embedder(config.embedder),
            registry=_registry(config),
            **config.scope_kwargs,
        )
    else:
        memory = build_memvara(config)
    if False:
        # Extraction from prose, through the same model supermemory is given, so the two
        # systems are compared on the same input and the same reader rather than on one
        # being handed pre-structured triples the other has to derive.
        #
        # `memory.llm`, not `memory._llm`. The first draft assigned the private name, which
        # Python happily created as a new unused attribute: every `add()` then ran the
        # deterministic path, the seeder still reported "seeded 15 facts" because it counts
        # calls, and the store came out with 15 episodes and **zero claims**. Nothing
        # raised. Count the claims, not the calls.
        pass
    try:
        if llm_model:
            for fact in facts():
                # `add`, not `remember`: the point of configuring a model is to make the
                # extractor do the work, which is what supermemory's memory agent is
                # doing with the identical sentence.
                memory.add(fact["prose"], role="user")
            # Report what the store HOLDS, not how many calls were made. A silent
            # extraction failure otherwise looks identical to a successful seed.
            import sqlite3

            with sqlite3.connect(str(db)) as raw:
                claims = raw.execute("select count(*) from claims").fetchone()[0]
            if not claims:
                raise RuntimeError(
                    f"{len(facts())} episodes were stored and the extractor produced no "
                    "claims. The model is reachable but nothing was learned from it -- "
                    "check the llm is attached and answering.")
            return claims
        for fact in facts():
            subject, predicate, obj = fact["triple"]
            # `remember`, not `add`. Extraction from prose needs a model, and with none
            # configured the deterministic fast path matches only a fixed set of sentence
            # forms -- so half these facts would be accepted and silently not stored, and
            # the benchmark would report a retrieval failure for a write that never
            # happened.
            memory.remember(subject, predicate, obj, text=fact["prose"])
        return len(facts())
    finally:
        memory.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", choices=("supermemory", "memvara"))
    parser.add_argument("--url", default="http://localhost:6767",
                        help="supermemory base URL. Point this at a local server; seeding "
                             "a hosted account leaves rows you may not be able to delete.")
    parser.add_argument("--container", default=CONTAINER)
    parser.add_argument("--db", type=Path, help="memvara store path. Use a throwaway file.")
    parser.add_argument("--llm-model", default="",
                        help="Extract from prose with this model instead of writing "
                             "triples directly. Use the same model given to the other "
                             "system, or the comparison is between two readers.")
    parser.add_argument("--llm-url", default="http://localhost:11434/v1",
                        help="OpenAI-compatible base URL for --llm-model.")
    parser.add_argument("--emit-cases", type=Path)
    args = parser.parse_args(argv)

    if args.emit_cases:
        print(f"wrote {emit_cases(args.emit_cases)} hit cases to {args.emit_cases}")
    if args.target == "supermemory":
        print(f"seeded {seed_supermemory(args.url, args.container)} facts into "
              f"{args.url} (containerTag={args.container})")
    elif args.target == "memvara":
        if not args.db:
            print("error: --db is required for memvara", file=sys.stderr)
            return 2
        written = seed_memvara(args.db, llm_model=args.llm_model, llm_url=args.llm_url)
        how = f"via {args.llm_model} extraction" if args.llm_model else "as triples"
        print(f"seeded {written} facts into {args.db} ({how})")
    if not args.target and not args.emit_cases:
        parser.error("nothing to do: pass --target or --emit-cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
