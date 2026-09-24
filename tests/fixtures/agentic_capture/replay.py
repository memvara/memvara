"""Replay the turns in `turns.json` through both capture paths and compare what they store.

This runs the headless agent command for real, under the login of whoever runs it, so it
costs tokens and takes a few minutes. It is not a test and pytest never runs it; the
tests in `tests/test_hook_agentic_capture.py` fake the command instead.

    python3 tests/fixtures/agentic_capture/replay.py --out /tmp/p3h-replay [--repeat 2]

For each turn and each path (the single-call extractor, then agentic capture) it seeds a
fresh local store at `<out>/<turn>-<path>.db`, runs the path over the turn exactly as
`capture.py` would, and reads the store back. Every file the hooks would write under
`~/.memvara` is written under `<out>` instead. It prints one row per run and writes
`<out>/replay.json`.

What it measures, per run:

* **found**: expected changes that happened, out of the expected total. A `new`
  predicate counts when a new live claim has it; a `replace` predicate counts when the
  seeded claim is no longer live.
* **unwanted**: new live claims under a predicate the turn did not call for, either as a
  new fact or as the replacement of a stored one. On the
  restated-instruction turn this is a duplicate of a stored fact; on the pasted-prompt
  turn it is the prompt stored as a memory.
* **tokens in / out** and **seconds**, from each run's own usage report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent


def _setup(out: Path) -> dict:
    """Import the hooks with every file they write pointed under `out`."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "plugin" / "hooks"))
    from lib import agentic, extract, ipc, usage, write

    write.LOG = out / "capture.log"
    ipc._HOME = str(out / "home")
    ipc.RUNTIME_DIR = str(out / "home" / "run")
    ipc._CLIENT_CONFIGS = ()
    usage.DEFAULT_PATH = out / "usage.jsonl"
    spent: dict = {}

    def record(usage_: dict, *, model: str, recorder=None) -> None:
        for key in ("input_tokens", "cache_read_input_tokens",
                    "cache_creation_input_tokens", "output_tokens"):
            spent[key] = spent.get(key, 0) + int(usage_.get(key) or 0)

    extract.record_extraction = record
    agentic.record_extraction = record
    return {"agentic": agentic, "extract": extract, "write": write, "spent": spent}


def _seed(path: Path, seed: list):
    from memvara import Memvara
    from memvara.embed import HashingEmbedder
    from memvara.llm import NullLLM
    from memvara.types import MemoryType

    for leftover in path.parent.glob(path.name + "*"):
        leftover.unlink()
    mem = Memvara(str(path), embedder=HashingEmbedder(dim=512), llm=NullLLM())
    for row in seed:
        mem.remember(row["subject"], row["predicate"], row["object"], confidence=0.7,
                     memory_type=MemoryType(row["memory_type"]))
    return mem


def _snapshot(mem, subjects, predicates) -> "list[tuple[str, str, str, bool]]":
    rows = []
    for subject in subjects:
        for predicate in predicates:
            for claim in mem.history(subject, predicate):
                live = claim.valid_to is None and claim.invalidated_at is None
                rows.append((subject, predicate, claim.object, live))
    return rows


def _score(case: dict, before, after) -> dict:
    seeded = {(s, p, o) for s, p, o, _ in before}
    new = [(s, p, o) for s, p, o, live in after if live and (s, p, o) not in seeded]
    live_seed = {(s, p) for s, p, o, live in after if live and (s, p, o) in seeded}
    expect = case["expect"]
    wanted_new = expect.get("new", [])
    wanted_replace = expect.get("replace", [])
    found = sum(1 for p in wanted_new if any(n[1] == p for n in new))
    found += sum(1 for p in wanted_replace
                 if not any(key[1] == p for key in live_seed))
    # A new value under a predicate the turn replaces is the replacement, not noise.
    unwanted = [n for n in new if n[1] not in wanted_new and n[1] not in wanted_replace]
    return {"found": found, "expected": len(wanted_new) + len(wanted_replace),
            "new": [f"{s} {p} {o}"[:120] for s, p, o in new],
            "unwanted": len(unwanted)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--only", default="", help="run only turns whose name has this")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    hooks = _setup(out)
    agentic, extract, write = hooks["agentic"], hooks["extract"], hooks["write"]
    spent = hooks["spent"]
    data = json.loads((HERE / "turns.json").read_text(encoding="utf-8"))
    subjects = ("user", data["project"])
    predicates = tuple(extract.VOCABULARY)
    os.environ["MEMVARA_FEATURE_ENCRYPTION"] = "0"
    os.environ["MEMVARA_EMBEDDER"] = "hashing"
    os.environ["PYTHONPATH"] = str(ROOT)
    cwd = str(ROOT)

    rows = []
    for round_ in range(args.repeat):
        for number, case in enumerate(data["turns"]):
            if args.only and args.only not in case["name"]:
                continue
            for path_name in ("single", "agentic"):
                db = out / f"{number}-{path_name}-{round_}.db"
                mem = _seed(db, data["seed"])
                os.environ["MEMVARA_DB"] = str(db)
                before = _snapshot(mem, subjects, predicates)
                spent.clear()
                started = time.monotonic()
                fell_back = False
                if path_name == "single":
                    facts = extract.triples(case["turn"], cwd, injected=[])
                    write.store_facts(mem, facts, case["turn"])
                else:
                    outcome = agentic.capture(mem, case["turn"], case["context"], cwd, [],
                                              hosted=False)
                    fell_back = outcome is None
                seconds = time.monotonic() - started
                after = _snapshot(mem, subjects, predicates)
                mem.close()
                score = _score(case, before, after)
                row = {"turn": case["name"], "path": path_name, "round": round_,
                       "seconds": round(seconds, 1), "fell_back": fell_back,
                       "tokens_in": spent.get("input_tokens", 0)
                       + spent.get("cache_read_input_tokens", 0)
                       + spent.get("cache_creation_input_tokens", 0),
                       "cache_write": spent.get("cache_creation_input_tokens", 0),
                       "tokens_out": spent.get("output_tokens", 0), **score}
                rows.append(row)
                print(f"{row['turn'][:34]:34} {path_name:8} found {row['found']}/"
                      f"{row['expected']} unwanted {row['unwanted']} "
                      f"in {row['tokens_in']:>6} out {row['tokens_out']:>5} "
                      f"{row['seconds']:>5}s{' FELL BACK' if fell_back else ''}",
                      flush=True)
    (out / "replay.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for path_name in ("single", "agentic"):
        mine = [r for r in rows if r["path"] == path_name]
        if not mine:
            continue
        n = len(mine)
        print(f"{path_name:8} found {sum(r['found'] for r in mine)}/"
              f"{sum(r['expected'] for r in mine)} unwanted "
              f"{sum(r['unwanted'] for r in mine)} mean in "
              f"{sum(r['tokens_in'] for r in mine) // n} mean out "
              f"{sum(r['tokens_out'] for r in mine) // n} mean "
              f"{sum(r['seconds'] for r in mine) / n:.1f}s fell back "
              f"{sum(1 for r in mine if r['fell_back'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
