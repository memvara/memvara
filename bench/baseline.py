"""A reimplementation of mem0's documented architecture, for comparison.

IMPORTANT — read this before quoting any number from the benchmark:

This is **not** mem0. It is a faithful reimplementation of the architecture mem0
describes, written so the comparison can run offline and deterministically:

  * every `add()` costs two LLM calls — one to extract facts from the turn, one to
    adjudicate ADD / UPDATE / DELETE / NOOP against the retrieved neighbours;
  * memories are opaque strings with an embedding, carrying no structure;
  * conflict detection is "embed the new fact, pull the top-k most similar existing
    memories, ask the model which to replace";
  * retrieval is pure vector top-k.

Both systems in the benchmark are driven by the *same* scripted extractor, so extraction
quality is held constant and what is measured is the architecture — not one model being
smarter than another. Numbers here describe this reimplementation's behavior; treat them
as a characterization of the design, not as a published benchmark result against the
mem0 package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Memory:
    text: str
    vec: np.ndarray
    subject: str
    predicate: str
    obj: str
    deleted: bool = False


@dataclass
class BaselineStats:
    llm_calls: int = 0
    embeds: int = 0
    conflicts_detected: int = 0
    conflicts_missed: int = 0


class Mem0StyleMemory:
    """Vector-top-k retrieval plus per-write LLM adjudication."""

    def __init__(self, embedder, extractor, *, top_k: int = 5,
                 update_threshold: float = 0.75) -> None:
        self.embedder = embedder
        self.extractor = extractor
        self.top_k = top_k
        self.update_threshold = update_threshold
        self.memories: list[Memory] = []
        self.stats = BaselineStats()

    # -- write ---------------------------------------------------------------

    def add(self, turns: list[str]) -> None:
        for turn in turns:
            # Call 1: extraction. mem0 runs this on every turn, including "ok, thanks".
            self.stats.llm_calls += 1
            facts = self.extractor(turn)
            if not facts:
                continue

            for f in facts:
                text = f"{f['subject']} {f['predicate'].replace('_', ' ')} {f['object']}"
                vec = self._embed(text)

                # Retrieve neighbours to adjudicate against. This is the step whose
                # recall bounds contradiction detection: a conflicting memory outside
                # the top-k is invisible to the adjudicating model.
                neighbours = self._search_vecs(vec, self.top_k)

                # Call 2: the ADD/UPDATE/DELETE decision.
                self.stats.llm_calls += 1
                replaced = False
                for mem, sim in neighbours:
                    same_slot = (mem.subject == f["subject"]
                                 and mem.predicate == f["predicate"])
                    if same_slot and sim >= self.update_threshold:
                        mem.deleted = True
                        replaced = True
                        break

                # Did a real conflict exist that we failed to see? Measured against
                # ground truth, not against what the retrieval returned.
                truly_conflicting = [
                    m for m in self.memories
                    if not m.deleted and m.subject == f["subject"]
                    and m.predicate == f["predicate"] and m.obj != f["object"]
                ]
                if truly_conflicting and not replaced:
                    self.stats.conflicts_missed += 1
                elif replaced:
                    self.stats.conflicts_detected += 1

                self.memories.append(Memory(text=text, vec=vec, subject=f["subject"],
                                            predicate=f["predicate"], obj=f["object"]))

    # -- read ----------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[str]:
        qv = self._embed(query)
        return [m.text for m, _ in self._search_vecs(qv, k)]

    def live(self) -> list[Memory]:
        return [m for m in self.memories if not m.deleted]

    # -- internals -----------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        self.stats.embeds += 1
        return self.embedder.encode([text])[0]

    def _search_vecs(self, qv: np.ndarray, k: int) -> list[tuple[Memory, float]]:
        live = self.live()
        if not live:
            return []
        mat = np.stack([m.vec for m in live])
        sims = mat @ qv
        order = np.argsort(-sims)[:k]
        return [(live[int(i)], float(sims[int(i)])) for i in order]
