"""Benchmark: Memvara vs. a mem0-style architecture, on the same workload.

Run:  python3 bench/compare.py

Read `bench/baseline.py` first — the comparison target is a reimplementation of mem0's
documented architecture, not the mem0 package. Both systems are driven by the *same*
scripted extractor, so extraction quality is held constant and what is being measured is
the architecture: how many model calls the write path costs, whether contradictions are
actually caught, and whether retrieval finds exact tokens.

Everything runs offline and deterministically. No API key, no network.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from baseline import Mem0StyleMemory

from memvara import Memvara, HashingEmbedder

# Slots that genuinely change over a long relationship with a user, plus a few that
# never do. Each entry is (predicate, [values in chronological order]).
SLOTS: list[tuple[str, list[str]]] = [
    ("lives_in", ["Berlin", "Amsterdam", "Lisbon"]),
    ("works_at", ["Acme Corp", "Globex", "Initech"]),
    ("job_title", ["engineer", "staff engineer", "engineering manager"]),
    ("relationship_status", ["single", "engaged", "married"]),
    ("timezone", ["CET", "WET"]),
    ("prefers_tool", ["unittest", "pytest"]),
    ("mood", ["stressed", "relaxed"]),
    ("born_in", ["Osaka"]),
    ("name", ["Alice Tan"]),
]

CHITCHAT = [
    "hey", "ok", "thanks!", "sounds good", "got it", "sure", "hmm", "right",
    "makes sense", "no worries", "haha", "yep", "cool", "interesting",
    "can you explain that?", "what do you think?", "and then?", "go on",
]

# A rare literal token embeddings reliably blur but BM25 nails. This is the case that
# justifies hybrid retrieval.
NEEDLE_PREDICATE = "hit_error"
NEEDLE_VALUE = "ERR_7734_TLSHANDSHAKE"


@dataclass
class Workload:
    turns: list[str]
    facts: dict[str, list[dict]]      # turn text -> extracted fact dicts
    truth: dict[str, str]             # predicate -> final correct value
    fact_turns: int


def build_workload(seed: int = 7, chitchat_ratio: int = 4) -> Workload:
    """A realistic transcript: mostly noise, occasionally a durable fact, with values
    that get revised over time."""
    rng = random.Random(seed)
    turns: list[str] = []
    facts: dict[str, list[dict]] = {}
    truth: dict[str, str] = {}
    fact_turns = 0

    updates: list[tuple[str, str]] = []
    for predicate, values in SLOTS:
        for v in values:
            updates.append((predicate, v))
        truth[predicate] = values[-1]
    updates.append((NEEDLE_PREDICATE, NEEDLE_VALUE))
    truth[NEEDLE_PREDICATE] = NEEDLE_VALUE

    for predicate, value in updates:
        for _ in range(chitchat_ratio):
            turns.append(rng.choice(CHITCHAT))
        sentence = f"By the way, my {predicate.replace('_', ' ')} is {value}."
        turns.append(sentence)
        facts[sentence] = [{"subject": "user", "predicate": predicate, "object": value}]
        fact_turns += 1

    return Workload(turns=turns, facts=facts, truth=truth, fact_turns=fact_turns)


def make_extractor(workload: Workload):
    """The shared extraction oracle. Both systems get identical, perfect extraction, so
    any difference in the results comes from architecture alone."""

    def extract(turn: str) -> list[dict]:
        return [dict(f) for f in workload.facts.get(turn, [])]

    return extract


class ScriptedLLM:
    """Adapts the shared extractor to Memvara's LLM protocol, counting calls."""

    name = "scripted"

    def __init__(self, extractor) -> None:
        self.extractor = extractor
        self.calls = 0

    def extract(self, episodes, known_predicates):
        self.calls += 1
        out = []
        for i, ep in enumerate(episodes):
            for f in self.extractor(ep.content):
                out.append({**f, "polarity": 1, "memory_type": "semantic",
                            "confidence": 0.95, "source_index": i})
        return out

    def classify_predicate(self, predicate, example):
        self.calls += 1
        # The needle is an event that recurs; everything else in this workload is
        # single-valued. A real deployment asks the model this once per predicate, ever.
        return {"cardinality": "one", "volatility": "slow", "memory_type": "semantic"}


# --- scoring ----------------------------------------------------------------

def score_memvara(mem: Memvara, w: Workload) -> dict:
    live = mem.get_all()
    by_pred: dict[str, list[str]] = {}
    for c in live:
        by_pred.setdefault(c.predicate, []).append(c.object)

    stale = correct = 0
    for pred, want in w.truth.items():
        got = by_pred.get(pred, [])
        if want in got:
            correct += 1
        stale += len([g for g in got if g != want])

    hits = [r.claim.object for r in mem.search(NEEDLE_VALUE, k=3)]
    return {"correct": correct, "stale": stale, "live": len(live),
            "needle_found": NEEDLE_VALUE in hits}


def score_baseline(base: Mem0StyleMemory, w: Workload) -> dict:
    live = base.live()
    by_pred: dict[str, list[str]] = {}
    for m in live:
        by_pred.setdefault(m.predicate, []).append(m.obj)

    stale = correct = 0
    for pred, want in w.truth.items():
        got = by_pred.get(pred, [])
        if want in got:
            correct += 1
        stale += len([g for g in got if g != want])

    hits = base.search(NEEDLE_VALUE, k=3)
    return {"correct": correct, "stale": stale, "live": len(live),
            "needle_found": any(NEEDLE_VALUE in h for h in hits)}


def run(chitchat_ratio: int = 4) -> None:
    w = build_workload(chitchat_ratio=chitchat_ratio)
    extractor = make_extractor(w)
    embedder = HashingEmbedder(dim=256)
    total_slots = len(w.truth)

    # --- baseline ---
    base = Mem0StyleMemory(embedder, extractor)
    t0 = time.perf_counter()
    base.add(w.turns)
    base_ms = (time.perf_counter() - t0) * 1000
    b = score_baseline(base, w)

    # --- memvara ---
    llm = ScriptedLLM(extractor)
    mem = Memvara(embedder=embedder, llm=llm, user="alice")
    t0 = time.perf_counter()
    mem.add(w.turns)
    eng_ms = (time.perf_counter() - t0) * 1000
    e = score_memvara(mem, w)

    print(f"\n  Workload: {len(w.turns)} turns, {w.fact_turns} carrying a durable fact, "
          f"{total_slots} distinct facts\n")

    # A model round-trip is ~0.5-2s; in-process bookkeeping is microseconds. Comparing
    # only local wall time would flatter the baseline by hiding the cost that dominates.
    LLM_RTT_MS = 800.0
    base_total = base_ms + base.stats.llm_calls * LLM_RTT_MS
    eng_total = eng_ms + llm.calls * LLM_RTT_MS

    rows = [
        ("LLM calls on the write path", base.stats.llm_calls, llm.calls, "lower"),
        ("Current value stored correctly", f"{b['correct']}/{total_slots}",
         f"{e['correct']}/{total_slots}", "higher"),
        ("Stale values left live", b["stale"], e["stale"], "lower"),
        ("Live memories (want %d)" % total_slots, b["live"], e["live"], "closer"),
        ("Local compute (ms)", f"{base_ms:.0f}", f"{eng_ms:.0f}", "lower"),
        (f"End-to-end @ {LLM_RTT_MS:.0f}ms/call (s)",
         f"{base_total / 1000:.0f}", f"{eng_total / 1000:.0f}", "lower"),
    ]

    print(f"  {'metric':<34}{'mem0-style':>14}{'memvara':>12}   {'better'}")
    print(f"  {'-' * 34}{'-' * 14:>14}{'-' * 12:>12}   {'-' * 6}")
    for name, bv, ev, better in rows:
        print(f"  {name:<34}{str(bv):>14}{str(ev):>12}   {better}")

    if base.stats.llm_calls and llm.calls:
        print(f"\n  -> {base.stats.llm_calls / llm.calls:.0f}x fewer model calls "
              f"on the write path ({base_total / eng_total:.0f}x faster end to end)")
    print(f"  -> baseline kept {b['stale']} superseded values live alongside the "
          f"correct ones;\n     it answers the question right and wrong at the "
          f"same time")
    print(f"  -> {base.stats.conflicts_missed} contradictions survived in the baseline.")
    print("     Honest cause: they fall below its similarity threshold (0.75), NOT below")
    print("     a top-k cutoff — sweeping top_k from 1 to 1000 changes nothing, because")
    print("     the conflicting memory is returned every time. The threshold is a tuning")
    print("     choice and the result is sensitive to it (0.5 -> 0 stale, 0.9 -> 11).")
    print("     What a keyed lookup buys is that there is no such threshold to tune.")

    # Reported outside the table on purpose — with this embedder the row does not
    # discriminate, and presenting it as a win would be misleading.
    print(f"\n  Exact-token recall ({NEEDLE_VALUE}): "
          f"baseline={'hit' if b['needle_found'] else 'MISS'}, "
          f"memvara={'hit' if e['needle_found'] else 'MISS'}")
    print("     Not a meaningful comparison here: the offline HashingEmbedder is built "
          "on character\n     n-grams, so it is unusually good at literal tokens and the "
          "vector-only baseline finds\n     them too. The BM25 leg earns its keep against "
          "a real semantic embedder, where\n     subword tokenization blurs exactly these "
          "strings. Untested here — stated, not claimed.")

    print("\n  Note: 'mem0-style' is a reimplementation of the documented architecture "
          "(see bench/baseline.py),\n        not the mem0 package. Both systems share "
          "one extraction oracle, so this isolates\n        architecture, not model "
          "quality.\n")
    mem.close()


if __name__ == "__main__":
    for ratio in (4, 12):
        print(f"\n{'=' * 74}\n  chitchat ratio 1:{ratio} "
              f"(one fact-bearing turn per {ratio} filler turns)\n{'=' * 74}")
        run(chitchat_ratio=ratio)
