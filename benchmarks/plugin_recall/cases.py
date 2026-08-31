"""The corpus, and why it is shaped the way it is.

A case is one prompt plus what should happen to it. There are two kinds, and the split is
the whole design:

`silence`
    A prompt with no answer in anybody's store. Injecting anything is filler by
    construction, so these are **store-independent**: they grade the same way against a
    full store, an empty one, or a competitor's. Every case here is synthetic and every one
    is safe to publish. This is the half that is normally missing, and it is the half that
    catches the failure a hit-only benchmark rewards -- a plugin that fills its slot on
    every prompt scores perfectly on hits and zero here.

`hit`
    A prompt whose answer is a fact about a specific person, in a specific store. These
    **cannot be published and cannot be shipped**: the gold answer is a claim about the
    operator's own work. `cases/build_private.py` derives them from the operator's own
    recall telemetry, on their machine, into a file they keep. A run with none loaded
    reports `hit_rate` as unavailable rather than as zero, because "no evidence" and "it
    missed" are different findings and collapsing them would flatter a plugin that was
    never asked.

The corpus is therefore honest about what it can prove without a store, rather than
inventing gold answers it has no standing to assert.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases" / "v1.jsonl"

KINDS = ("silence", "hit")


class CaseError(ValueError):
    """A corpus file that cannot be trusted to score anything."""


@dataclass(frozen=True)
class Case:
    id: str
    kind: str
    prompt: str
    why: str
    expect: tuple[re.Pattern[str], ...] = ()

    @property
    def family(self) -> str:
        """The group an id names, for the per-family breakdown in the report.

        Ids are written `family-thing` and the prefix is the family. It is derived rather
        than stored because a second field that must agree with the id is a second thing
        to get wrong, and the report is the only consumer.
        """
        return self.id.split("-", 1)[0]

    def matched(self, context: str) -> bool:
        """Did the injected context contain the fact this case is about?"""
        return any(pattern.search(context) for pattern in self.expect)


def _one(raw: dict, source: Path, line_no: int) -> Case:
    where = f"{source}:{line_no}"
    for field in ("id", "kind", "prompt", "why"):
        if not str(raw.get(field, "")).strip():
            raise CaseError(f"{where}: case is missing a non-empty {field!r}.")
    kind = raw["kind"]
    if kind not in KINDS:
        raise CaseError(f"{where}: kind {kind!r} is not one of {KINDS}.")

    patterns = raw.get("expect") or []
    if kind == "hit" and not patterns:
        raise CaseError(
            f"{where}: a hit case with no `expect` patterns can never fail, which makes it "
            "worse than absent -- it inflates the denominator with a case that scores a "
            "point for any output at all.")
    if kind == "silence" and patterns:
        raise CaseError(
            f"{where}: a silence case must not carry `expect` patterns. Silence is scored "
            "on whether anything was injected at all; a pattern here would read as though "
            "some injections were acceptable, and none are.")
    try:
        compiled = tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    except re.error as exc:
        raise CaseError(f"{where}: {exc}") from exc
    return Case(raw["id"], kind, raw["prompt"], raw["why"], compiled)


def load(*paths: Path) -> list[Case]:
    """Read one or more JSONL corpora, refusing anything ambiguous.

    Duplicate ids across files are an error rather than a last-one-wins merge: a private
    corpus is built from a log and a public one is hand-written, and the day those collide
    is the day a published silence number quietly starts including somebody's private case.
    """
    cases: list[Case] = []
    seen: dict[str, Path] = {}
    for path in paths:
        if not path.is_file():
            raise CaseError(f"No corpus at {path}.")
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except ValueError as exc:
                raise CaseError(f"{path}:{line_no}: {exc}") from exc
            case = _one(raw, path, line_no)
            if case.id in seen:
                raise CaseError(
                    f"{path}:{line_no}: id {case.id!r} was already defined in {seen[case.id]}.")
            seen[case.id] = path
            cases.append(case)
    if not cases:
        raise CaseError(f"No cases in {', '.join(str(p) for p in paths)}.")
    return cases
