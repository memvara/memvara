"""Build `hit` cases from your own recall telemetry. Nothing this writes is publishable.

    python3 benchmarks/plugin_recall/cases/build_private.py \
        --log ~/.memvara/.hooks/recall-sample.log --out ~/plugin-recall-private.jsonl

A `hit` case asserts that a particular fact about a particular person is in a particular
store. That claim cannot be shipped -- it is neither true for anyone else nor safe to
publish -- so this script exists instead of a committed corpus, and writes outside the
repository by default.

## Where the gold comes from, and the circularity you are buying

The input is a log of prompts a memory plugin already answered, and the gold pattern is
taken from the answer it gave. Two consequences, both worth stating before a number
built this way is quoted:

* **Against the plugin that wrote the log, this is not an independent test.** It measures
  reproducibility -- does the same prompt still surface the same fact after a change --
  which is a real and useful regression signal, and is not evidence of quality.
* **Against a different plugin it is closer to a fair test**, since the gold was authored
  by neither the prompt nor that plugin, but it still inherits one system's opinion about
  which facts were worth surfacing. A fact the first plugin never found is not in the log
  and so is never asked of the second.

The honest use is regression detection on your own machine. It is not a leaderboard.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: `TIME carried=n  prompt='...'  mem1='...'  mem2='...'`, the shape memvara's recall hook
#: writes. Another plugin can be measured by emitting the same three fields; the format is
#: documented here rather than imported so this script has no dependency on any plugin.
LINE = re.compile(r"prompt='(?P<prompt>.*?)'\s+(?P<mems>mem\d+='.*)$")
MEM = re.compile(r"mem\d+='(?P<text>.*?)'(?=\s+mem\d+='|$)")

#: Prompts shorter than this carry too little to be a fair retrieval target, and prompts
#: that are a task notification or a slash command are not the user talking.
MIN_PROMPT = 12
MACHINE = ("<task-notification>", "/", "!", "#", "<")


def _phrase(memory: str) -> str | None:
    """A regex matching the distinctive opening of a recalled memory.

    The leading `- ` and the subject token are dropped and the next few words are taken
    verbatim. Short and generic words are poor evidence that the right fact came back, so
    a phrase that ends up with fewer than three substantial words is rejected outright --
    a case that matches on `the memvara` would pass for any output at all, which is the
    failure `cases.load` refuses to accept from a hand-written corpus and should refuse
    here too.
    """
    body = memory.lstrip("- ").strip()
    words = [w for w in re.split(r"\s+", body) if len(w) > 3][1:6]
    if len(words) < 3:
        return None
    return r"\s+".join(re.escape(w) for w in words)


def build(log: Path, *, limit: int | None = None) -> list[dict]:
    cases: list[dict] = []
    seen_prompts: set[str] = set()
    for line_no, line in enumerate(log.read_text(errors="replace").splitlines(), start=1):
        found = LINE.search(line)
        if not found:
            continue
        prompt = found.group("prompt").strip()
        if (len(prompt) < MIN_PROMPT or prompt.startswith(MACHINE)
                or prompt in seen_prompts):
            continue
        patterns = [p for p in (_phrase(m.group("text"))
                                for m in MEM.finditer(found.group("mems"))) if p]
        if not patterns:
            continue
        seen_prompts.add(prompt)
        cases.append({
            "id": f"priv-{line_no}",
            "kind": "hit",
            "prompt": prompt,
            "why": f"Recalled at least one of {len(patterns)} facts when this log was "
                   "written; a run that no longer surfaces any of them is a regression.",
            "expect": patterns,
        })
        if limit and len(cases) >= limit:
            break
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, required=True,
                        help="Recall sample log, e.g. ~/.memvara/.hooks/recall-sample.log")
    parser.add_argument("--out", type=Path, required=True,
                        help="Where to write the JSONL. Keep it outside the repository.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    log = args.log.expanduser()
    if not log.is_file():
        print(f"error: no log at {log}", file=sys.stderr)
        return 2
    cases = build(log, limit=args.limit)
    if not cases:
        print(f"error: {log} produced no usable cases. Expected lines shaped "
              "\"prompt='...'  mem1='...'\".", file=sys.stderr)
        return 1

    out = args.out.expanduser()
    if out.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        # Refused rather than warned. These cases quote a person's own prompts and their
        # store's contents; a path inside the repository is one `git add -A` away from
        # publishing them, and this repository is public.
        print(f"error: {out} is inside the repository. These cases are private -- write "
              "them somewhere else.", file=sys.stderr)
        return 2
    out.write_text("".join(json.dumps(c) + "\n" for c in cases))
    print(f"wrote {len(cases)} hit cases to {out}")
    print("These quote your prompts and your store. Do not commit them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
