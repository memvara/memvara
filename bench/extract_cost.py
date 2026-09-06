"""What one extraction costs, by claim schema.

Run:  PYTHONPATH=. python3 bench/extract_cost.py --episodes local/spike/episodes.json

Extraction on a CPU-hosted model is dominated by generation, and most of what gets
generated is field names rather than facts. This harness measures that directly. It has
two halves, and the first needs no server at all:

  accounting  serialize one claim under each schema and count the characters and, if
              tiktoken is installed, the tokens. Answers "how much shorter is the short
              shape" without touching the model.

  live        send the same episodes to a real OpenAI-compatible endpoint under each
              schema, and report wall time, reported tokens, claims returned and — if you
              pass `--gold` — how many of the facts you expected came back.

The second half is the one that decides anything. The accounting half predicts a saving;
only the live half says whether the model still finds the same facts once it stops being
told to fill in every field. Those are different questions and a smaller, faster response
that has dropped a fact is not an improvement.

Episodes file: JSON, either a list of turns or a list of lists of turns, where a turn is
`{"role": "user", "content": "..."}`. Each inner list is sent as one `extract()` call, so
a file of three lists reproduces three separate extractions.

Gold file: JSON list of objects you expect to see, as plain strings — `["samply",
"Lisbon"]`. Matching is a case-insensitive substring test against each claim's object, so
treat the count as a screen rather than a score. Read the claims themselves before
believing a number moved.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Any, Sequence

from memvara.llm.base import (
    CLAIM_SCHEMA,
    MAX_CLAIMS,
    Usage,
    bounded_claim_schema,
    self_hosted_claim_schema,
)
from memvara.types import Episode, Scope

#: A claim with every field filled, as the shipped schema obliges the model to emit it.
#: Real values rather than placeholders, because the object and predicate strings are a
#: meaningful part of the length and a row of "x" would flatter the short shape.
SAMPLE_CLAIM = {
    "subject": "user", "predicate": "prefers_profiler", "object": "samply",
    "polarity": 1, "memory_type": "procedural", "confidence": 0.9,
    "source_index": 0, "when": None, "amount": None, "unit": None,
}

#: What `self_hosted_claim_schema` lets the model leave out. Dropping these from the
#: sample is what the short shape looks like on the wire.
OPTIONAL = ("polarity", "confidence", "when", "amount", "unit")

VARIANTS: dict[str, Any] = {
    "full": CLAIM_SCHEMA,
    "capped": bounded_claim_schema(MAX_CLAIMS),
    "terse": self_hosted_claim_schema(MAX_CLAIMS),
}


def _tokens(text: str) -> int | None:
    """Token count under o200k, or None when tiktoken is not installed.

    A proxy: the model you are benchmarking has its own tokenizer, and JSON punctuation is
    where they differ most. The ratio between the variants is what this is for, and that
    holds up better across tokenizers than either absolute number does.
    """
    try:
        import tiktoken
    except ImportError:
        return None
    return len(tiktoken.get_encoding("o200k_base").encode(text))


def accounting(claims: int) -> None:
    """Serialized cost of one response, per schema. No server involved."""
    print(f"\n=== accounting ({claims} claims per response) ===\n")
    short = {k: v for k, v in SAMPLE_CLAIM.items() if k not in OPTIONAL}
    rows = [("full / capped", SAMPLE_CLAIM), ("terse", short)]
    baseline = None
    for label, one in rows:
        body = json.dumps({"claims": [one] * claims}, separators=(",", ":"))
        n = _tokens(body)
        baseline = baseline if baseline is not None else (n or len(body))
        now = n or len(body)
        unit = "tokens" if n is not None else "chars (install tiktoken for tokens)"
        print(f"  {label:<16}{now:>6} {unit:<40}{now / baseline:>6.0%} of full")
    if _tokens("") is None:
        print("\n  tiktoken is not installed, so these are characters. "
              "pip install tiktoken for the number that matters.")


def load_batches(path: str) -> list[list[Episode]]:
    raw = json.load(open(path))
    batches = raw if raw and isinstance(raw[0], list) else [raw]
    return [
        [Episode(content=t["content"], role=t.get("role", "user"),
                 scope=Scope(tenant="bench"))
         for t in batch]
        for batch in batches
    ]


def live(batches: Sequence[Sequence[Episode]], gold: Sequence[str],
         predicates: Sequence[str], model: str, base_url: str, reps: int) -> None:
    """The same episodes through each schema, against a real endpoint."""
    from memvara.llm.openai import OpenAILLM

    print(f"\n=== live: {model} at {base_url} ===\n")
    print(f"  {'schema':<10}{'p50 s':>8}{'max s':>8}{'in':>8}{'out':>8}"
          f"{'claims':>8}{'gold':>8}")
    for name in VARIANTS:
        llm = OpenAILLM(model=model, base_url=base_url,
                        max_claims=MAX_CLAIMS if name != "full" else None,
                        terse=(name == "terse"))
        times: list[float] = []
        usage = Usage()
        found: list[dict[str, Any]] = []
        for _ in range(reps):
            for batch in batches:
                t0 = time.perf_counter()
                found += llm.extract(batch, predicates, usage=usage)
                times.append(time.perf_counter() - t0)
        hits = sum(
            any(g.lower() in str(c["object"]).lower() for c in found) for g in gold)
        print(f"  {name:<10}{statistics.median(times):>8.1f}{max(times):>8.1f}"
              f"{usage.input_tokens // reps:>8}{usage.output_tokens // reps:>8}"
              f"{len(found) // reps:>8}{f'{hits}/{len(gold)}' if gold else '-':>8}")
        # The distinct claims, because the table above cannot show a well-formed claim
        # that is false — `gate / lives_in / "Port 55434"` counts as a claim and, if its
        # object happens to contain a gold string, as a hit. Reading these is the check
        # the numbers cannot do for you. Deduplicated because repeated passes over the
        # same episodes produce the same claims and a wall of repeats hides the one that
        # changed between schemas.
        for triple in dict.fromkeys(
                (c["subject"], c["predicate"], c["object"]) for c in found):
            print("      {} / {} / {}".format(*triple))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", type=int, default=8,
                    help="claims per response for the accounting half (default 8, the "
                         "measured mean on the production box)")
    ap.add_argument("--episodes", help="JSON file of turns; omit to run accounting only")
    ap.add_argument("--gold", help="JSON list of object strings you expect back")
    ap.add_argument("--predicates", help="JSON list of known predicates to send")
    ap.add_argument("--model", default="phi-4-mini-instruct")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the episodes. Latency here is bimodal, so one "
                         "sample proves nothing — the max column is the one to read.")
    args = ap.parse_args(argv)

    accounting(args.claims)
    if args.episodes:
        live(load_batches(args.episodes),
             json.load(open(args.gold)) if args.gold else [],
             json.load(open(args.predicates)) if args.predicates else [],
             args.model, args.base_url, args.reps)
    else:
        print("\n  No --episodes given, so nothing was measured against a model. The "
              "accounting above predicts a saving; only a live run says whether the "
              "same facts still come back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
