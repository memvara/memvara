"""The release-bar measurement for agentic extraction: two runs that differ only in the switch.

    PYTHONPATH=. python3 bench/extraction_bar.py claims --dry-run            # offline rehearsal
    PYTHONPATH=. python3 bench/extraction_bar.py claims \\
        --llm openai --llm-model gpt-5.4
    PYTHONPATH=. python3 bench/extraction_bar.py claims \\
        --llm openai --llm-model gpt-5.4 --write-path worker
    PYTHONPATH=. python3 bench/extraction_bar.py longmemeval --dry-run --out runs/rehearsal.jsonl
    PYTHONPATH=. python3 bench/extraction_bar.py longmemeval \\
        --llm openai --llm-model gpt-5.4 --write-path worker \\
        --reader openai --model gpt-5.4 --judge llm \\
        --concurrency 8 --out runs/bar-longmemeval.jsonl

Agentic extraction (`agentic_extraction`, `memvara/write/agentic.py`) ships switched off.
The phase 3 parity design (`docs/superpowers/specs/2026-09-23-parity-phase-3-extraction-
and-cloud-design.md`, section 3.1) lets it default on only when two things hold, and this
script measures both. Issue #239 asked for it.

## The two halves of the bar

**`claims`: no fewer claims and no more duplicates.** The demo's support history
(`demo/scenario.py`, the corpus `demo/harness.py` runs on) is written into two stores with
the same extraction model: one with the switch off, which is today's single-call
extraction, and one with it on. The report counts, for each arm, the claims written, the
claims the store already held (reinforced), the claims ended, the live claims left at the
end, and the duplicates among them. The bar is met when the agentic arm wrote at least as
many claims as the single-call arm and left no more duplicates.

A duplicate is a live claim that repeats another live claim's value. It is counted two
ways, and the bar uses the sum. **Same slot**: the same subject, predicate and value,
after ignoring case, spacing and surrounding punctuation. The reconciler merges exact
repeats, so a copy here differs from its twin in spelling alone. **Other predicate**: the
same subject and value filed under a second predicate, such as `lives_in Porto` beside
`home_city Porto`. The pollution guard catches that inside one extraction but not across
writes, and it is the duplicate a model that cannot see the store is most likely to make.
A subject that genuinely has one value under two predicates would also count here, so read
the claims before trusting a difference of one or two.

**`longmemeval`: judged accuracy within the reader noise floor.** The 199-question
LongMemEval-S sample behind the judged numbers in `docs/BENCHMARKS.md`
(`bench/samples/longmemeval_s_199.txt`) is run twice, once per arm. Every question gets a
fresh store, and its haystack is written through the arm's extraction, then retrieved,
answered and judged exactly as `bench/longmemeval.py` does it. The bar is met when the
agentic arm answers no more than 7 fewer questions correctly than the single-call arm.
`docs/BENCHMARKS.md` records the floor: the reader disagrees with itself on 7.8% of
identical prompts, so a single-run difference under about 8 questions is noise.

**What the accuracy half compares.** The published 177 of 199 was produced in the
MemoryBench harness through the hosted service, with `gpt-5.4` as reader and judge. This
script runs in this repository's harness instead: `bench/longmemeval.py`'s reader prompt,
retrieval budget and judge prompts. Its single-call arm stands in for the production
number, and the measurement is the difference between the two arms on the same questions,
reader and judge. Its absolute accuracy is not comparable with 177 of 199. Pass
`--reader openai --model gpt-5.4 --judge llm` to use the published reader and judge model.

## The two write paths

`--write-path add` writes each batch with `add()`, which is what a library caller and the
local MCP server do. A caller is waiting, so the tool loop gets 25 seconds
(`AGENTIC_SYNC_TIMEOUT`) before it falls back to one extraction call.

`--write-path worker` is the hosted service's arrangement. A handle with no model stores
every turn with extraction deferred, as the hosted API does. A second handle on the same
store, with the model, then reads the stored turns through `reextract()`, `--batch` turns
at a time and oldest first, as the hosted extraction worker does, and the loop gets 180
seconds (`AGENTIC_TIMEOUT`). The hosted worker reads one turn at a time by default, and
so does this.

## Fallbacks and refusals

A batch the switch sent to the tool loop can still end up extracted by one call: when the
backend cannot run tools (`unsupported`), when the loop times out, when the model's answer
cannot be used twice (`malformed`), when it is still calling tools after 12 answers
(`step_limit`), or when a request fails twice (`error`). The report counts each reason. An
agentic arm that fell back on every batch measured single-call extraction twice, so the
bar is then reported as not measured. Proposals the write refused are counted by reason
too (`not_read`, `broader_scope`, `invalid`, `instruction_echo`, `not_applied`).

## What it costs, and how to run it cheaply first

Both halves need a model key: `ANTHROPIC_API_KEY` for `--llm anthropic`, `OPENAI_API_KEY`
for `--llm openai` (with `--llm-base-url` for an OpenAI-compatible server). The key is read
from the environment and never printed. `--dry-run` runs everything with a scripted
stand-in model and the string-match judge; its numbers mean nothing and it never reports
the bar as met.

The `claims` half is 64 turns and costs little. The `longmemeval` half writes each
question's whole haystack through the extraction model, twice. `bench/longmemeval.py`
records about 57 million input tokens of single-call extraction for all 500 questions;
the 199-question sample is about two fifths of that per arm, and the agentic arm adds its
tool steps on top. `--limit N` runs the first N questions of the sorted sample as a pilot.
A prefix under-represents temporal-reasoning, because the `gpt4_` ids sort last and are
mostly that type, so a pilot says whether the run works and what it costs, not whether the
bar is met. `--out` is written one row at a time, and a re-run with the same file skips
every (arm, question) pair already in it, so a run that dies resumes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import string
import sys
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "bench")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evalkit as ek  # noqa: E402
import longmemeval as lme  # noqa: E402

from memvara import DegradedExtractionWarning, Memvara, NullLLM  # noqa: E402
from memvara.embed import HashingEmbedder  # noqa: E402
from memvara.llm.base import Message, ToolRun, ToolSpec, Usage  # noqa: E402
from memvara.telemetry import WRITE_AGENTIC, MemoryRecorder  # noqa: E402
from memvara.types import Claim, WriteReceipt  # noqa: E402

#: The two arms, in report order. `single_call` is today's extraction with the switch off.
ARMS = ("single_call", "agentic")

#: The 199 question ids behind the judged numbers in `docs/BENCHMARKS.md`.
SAMPLE_PATH = _ROOT / "bench" / "samples" / "longmemeval_s_199.txt"

#: A single-run difference smaller than this many questions is noise: `docs/BENCHMARKS.md`
#: records 7.8% reader self-disagreement on identical prompts over the 199-question sample.
NOISE_FLOOR_QUESTIONS = 8

MET, NOT_MET, NOT_MEASURED = "met", "not met", "not measured"


# -- the sample -------------------------------------------------------------------------


def load_sample(path: str | Path = SAMPLE_PATH) -> list[str]:
    """The sample's question ids, in file order. Lines starting with `#` are comments."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def select_sample(items: Sequence[lme.Instance], ids: Sequence[str]) -> list[lme.Instance]:
    """The instances `ids` names, in the order of `ids`.

    A dataset file that lacks one of them is refused by name rather than scored on the
    rest, because a run over 198 questions would be reported beside a number over 199.
    """
    by_id = {item.qid: item for item in items}
    missing = [qid for qid in ids if qid not in by_id]
    if missing:
        raise SystemExit(f"The dataset file has no question {', '.join(missing)}. The "
                         "sample is fixed; use the LongMemEval-S file it was drawn from "
                         "(bench/longmemeval.py --download --dataset s).")
    return [by_id[qid] for qid in ids]


# -- counting ---------------------------------------------------------------------------

_EDGE_PUNCTUATION = string.punctuation + string.whitespace


def _norm(value: str) -> str:
    return " ".join(value.casefold().split()).strip(_EDGE_PUNCTUATION)


def duplicates(claims: Iterable[Claim]) -> tuple[int, int]:
    """How many live claims repeat another's value: `(same slot, other predicate)`.

    Same slot: the same subject, predicate and value after ignoring case, spacing and
    surrounding punctuation, counted once per extra copy. Other predicate: a value the
    subject already holds under a different predicate, counted once per extra predicate.
    A claim counted as a same-slot copy is not counted again as another predicate.
    """
    slots: Counter[tuple[str, str, str]] = Counter()
    for c in claims:
        slots[(_norm(c.subject), _norm(c.predicate), _norm(c.object))] += 1
    same_slot = sum(n - 1 for n in slots.values())
    predicates: dict[tuple[str, str], set[str]] = {}
    for subject, predicate, value in slots:
        predicates.setdefault((subject, value), set()).add(predicate)
    other = sum(len(p) - 1 for p in predicates.values())
    return same_slot, other


@dataclass
class Tally:
    """What one arm's extraction did, summed over every write it made."""

    #: `add()` calls on the `add` path, `reextract()` calls on the `worker` path.
    writes: int = 0
    turns: int = 0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    claims_written: int = 0
    reinforced: int = 0
    ended: int = 0
    retired: int = 0
    unextracted: int = 0
    #: Writes whose model call failed, so nothing was read. See `WriteReceipt.deferred`.
    failed: int = 0
    live_claims: int = 0
    duplicate_same_slot: int = 0
    duplicate_other_predicate: int = 0
    #: Batches the tool loop finished. Counted from the `write.agentic` series.
    agentic_runs: int = 0
    #: Batches the switch sent to the tool loop that went to single-call extraction, by
    #: `WriteReceipt.agentic_fallback` reason.
    fallbacks: dict[str, int] = field(default_factory=dict)
    #: Proposals the write did not make, by `RefusedProposal.reason`.
    refused: dict[str, int] = field(default_factory=dict)

    @property
    def duplicates(self) -> int:
        return self.duplicate_same_slot + self.duplicate_other_predicate

    @property
    def agentic_batches(self) -> int:
        return self.agentic_runs + sum(self.fallbacks.values())

    def record(self, receipt: WriteReceipt, *, write: bool = True) -> None:
        """Add one receipt. `write=False` for a store-only `add()` on the worker path,
        whose `deferred` flag is the plan rather than a failure."""
        if write:
            self.writes += 1
            self.failed += bool(receipt.deferred)
        self.llm_calls += receipt.llm_calls
        self.tokens_in += receipt.tokens_in
        self.tokens_out += receipt.tokens_out
        self.claims_written += len(receipt.added)
        self.reinforced += len(receipt.reinforced)
        self.ended += len(receipt.ended)
        self.retired += len(receipt.retired)
        self.unextracted += receipt.unextracted
        if receipt.agentic_fallback:
            _bump(self.fallbacks, receipt.agentic_fallback)
        for refusal in receipt.proposals_refused:
            _bump(self.refused, refusal.reason)

    def merge(self, other: "Tally") -> "Tally":
        for f in fields(self):
            mine, theirs = getattr(self, f.name), getattr(other, f.name)
            if isinstance(mine, dict):
                for key, n in theirs.items():
                    _bump(mine, key, n)
            else:
                setattr(self, f.name, mine + theirs)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Tally":
        return cls(**{f.name: raw[f.name] for f in fields(cls) if f.name in raw})


def _bump(counts: dict[str, int], key: str, n: int = 1) -> None:
    counts[key] = counts.get(key, 0) + n


# -- the stand-in model for a dry run ---------------------------------------------------

_TURN_LINE = re.compile(r"^\[(\d+)\] \w+: (.*)$", re.MULTILINE)
_WORD = re.compile(r"[a-z0-9]+")


def _rehearsal_value(text: str) -> str:
    """The first three words of a turn's first line: a value the grounding check accepts."""
    first = text.splitlines()[0] if text else ""
    return " ".join(_WORD.findall(first.lower())[:3]) or "nothing"


class RehearsalChat:
    """A scripted model for `--dry-run` and the tests. Never a measurement.

    Both paths propose the same memory for every turn it is shown, `user mentioned <the
    turn's first three words>`, so the two arms can be compared line by line. The tool loop
    searches before each proposal, as the real instructions ask. `end_unread=True` also
    proposes ending a memory it never read, which the write must refuse as `not_read`.
    """

    name = "rehearsal/scripted"
    is_noop = False
    reports_usage = False
    accepts_guidance = True

    def __init__(self, *, end_unread: bool = False) -> None:
        self.end_unread = end_unread
        self.extract_calls = 0
        self.tool_runs = 0
        self.timeouts: list[float] = []

    def extract(self, episodes: Sequence[Any], known_predicates: Sequence[str],
                guidance: Any = None) -> list[dict[str, Any]]:
        self.extract_calls += 1
        return [self._fact(i, ep.content) for i, ep in enumerate(episodes)]

    def run_tools(self, system: str, messages: Sequence[Message],
                  tools: Sequence[ToolSpec], *, max_steps: int, timeout: float,
                  usage: Usage | None = None) -> ToolRun:
        self.tool_runs += 1
        self.timeouts.append(timeout)
        by_name = {t.name: t for t in tools}
        steps = 1
        for match in _TURN_LINE.finditer(messages[-1].content):
            index, text = int(match.group(1)), match.group(2)
            by_name["search_memories"].handler({"query": text, "k": 5})
            by_name["propose_claim"].handler({**self._fact(index, text),
                                              "expires_at": None})
            if self.end_unread:
                by_name["propose_end"].handler({"claim_id": "cl_never_read",
                                                "reason": "rehearsal",
                                                "source_index": index})
            steps += 1
        return ToolRun(steps=steps, requests=steps, finished=True, text="done")

    def classify_predicate(self, predicate: str, example: Any) -> dict[str, str]:
        return {"cardinality": "many", "volatility": "slow", "memory_type": "semantic"}

    @staticmethod
    def _fact(index: int, text: str) -> dict[str, Any]:
        return {"subject": "user", "predicate": "mentioned",
                "object": _rehearsal_value(text), "source_index": index,
                "confidence": 0.9, "memory_type": "semantic", "valid_from": None,
                "amount": None, "unit": None}


# -- writing one arm --------------------------------------------------------------------

Batch = Sequence[Mapping[str, Any]]

#: Builds one store handle: `build(llm, **options)`, where `options` are further `Memvara`
#: keyword arguments. Each mode passes its own, so the two arms differ only in `options`.
Build = Callable[..., Memvara]


def write_arm(build: Build, llm: Any, batches: Sequence[Batch], *, arm: str,
              path: str = "add", batch: int = 1) -> tuple[Tally, Memvara]:
    """Write `batches` through one arm on one write path, and count what happened.

    Returns the tally and the handle that extracted, which the caller reads from and then
    closes. On the `add` path every batch is one `add()` on a handle with the model. The
    `worker` path is the hosted service's arrangement: a handle with no model and
    `write_extraction_deferred=True` stores every batch, as the API does, and a second
    handle on the same store, with the model, reads the stored turns through
    `reextract()`, `batch` at a time and oldest first, as the extraction worker does. A
    turn a read produced nothing from has no claim citing it and would be offered again,
    so every turn a read was handed is excluded from the next read, as the worker does.
    """
    telemetry = MemoryRecorder()
    options = {"telemetry": telemetry, "write_agentic_extraction": arm == "agentic"}
    if path == "add":
        mem = writer = build(llm, **options)
    else:
        with warnings.catch_warnings():
            # The storing handle has no model on purpose; the warning says so to a
            # caller who did not mean it.
            warnings.simplefilter("ignore", DegradedExtractionWarning)
            writer = build(NullLLM(), write_extraction_deferred=True)
        mem = build(llm, store=writer.store, **options)
    tally = Tally()
    try:
        for turns in batches:
            turns = list(turns)
            if not turns:
                continue
            tally.turns += len(turns)
            tally.record(writer.add(turns), write=path == "add")
        if path == "worker":
            seen: set[str] = set()
            while True:
                pending = mem.pending_extraction(limit=batch, exclude=seen)
                if not pending:
                    break
                seen.update(ep.id for ep in pending)
                tally.record(mem.reextract(pending))
        live = mem.get_all()
    except BaseException:
        mem.close()
        raise
    tally.live_claims = len(live)
    tally.duplicate_same_slot, tally.duplicate_other_predicate = duplicates(live)
    tally.agentic_runs = telemetry.total(WRITE_AGENTIC, outcome="agentic")
    return tally, mem


def compare_claims(batches: Sequence[Batch], llm: Any, *, path: str = "add",
                   batch: int = 1) -> dict[str, Tally]:
    """Both arms over the same batches with the same model, in a fresh store each."""
    def build(model: Any, **options: Any) -> Memvara:
        return Memvara(embedder=HashingEmbedder(dim=ek.BASELINE_EMBED_DIM), llm=model,
                       user="customer", **options)

    out: dict[str, Tally] = {}
    for arm in ARMS:
        out[arm], mem = write_arm(build, llm, batches, arm=arm, path=path, batch=batch)
        mem.close()
    return out


def demo_batches() -> list[list[dict[str, Any]]]:
    """The demo's support history, one turn per write, as `demo/baselines.py` writes it."""
    from demo import scenario

    return [[{"role": t.role, "content": t.text, "ts": t.at}]
            for t in scenario.conversation()]


# -- the verdicts -----------------------------------------------------------------------


def _fallback_note(agentic: Tally) -> str:
    fell = sum(agentic.fallbacks.values())
    if not fell:
        return ""
    reasons = ", ".join(f"{n} {why}" for why, n in sorted(agentic.fallbacks.items()))
    return (f"; {fell} of {agentic.agentic_batches} agentic batches fell back to "
            f"single-call extraction ({reasons})")


def claims_verdict(tallies: Mapping[str, Tally]) -> tuple[str, str]:
    """Whether the claims half of the bar is met, and why, in one sentence."""
    single, agentic = tallies["single_call"], tallies["agentic"]
    if agentic.agentic_runs == 0:
        reasons = ", ".join(sorted(agentic.fallbacks)) or "no batch reached the model"
        return NOT_MEASURED, (f"the tool loop never finished a batch ({reasons}), so both "
                              "arms measured single-call extraction")
    if agentic.claims_written < single.claims_written:
        return NOT_MET, (f"agentic wrote {agentic.claims_written} claims, single-call "
                         f"{single.claims_written}")
    if agentic.duplicates > single.duplicates:
        return NOT_MET, (f"agentic left {agentic.duplicates} duplicates, single-call "
                         f"{single.duplicates}")
    return MET, (f"agentic wrote {agentic.claims_written} claims against "
                 f"{single.claims_written} and left {agentic.duplicates} duplicates against "
                 f"{single.duplicates}" + _fallback_note(agentic))


def accuracy_verdict(single: Mapping[str, bool | None], agentic: Mapping[str, bool | None],
                     *, expected: Sequence[str]) -> tuple[str, str]:
    """Whether the accuracy half of the bar is met over the questions in `expected`.

    Decided only when both arms judged every expected question. The agentic arm meets it
    when it answers fewer than `NOISE_FLOOR_QUESTIONS` fewer questions correctly than the
    single-call arm; answering more is never a failure.
    """
    for name, rows in (("single-call", single), ("agentic", agentic)):
        judged = [q for q in expected if rows.get(q) is not None]
        if len(judged) < len(expected):
            return NOT_MEASURED, (f"the {name} arm has a judged answer for {len(judged)} "
                                  f"of {len(expected)} questions")
    right_single = sum(bool(single[q]) for q in expected)
    right_agentic = sum(bool(agentic[q]) for q in expected)
    loss = right_single - right_agentic
    said = (f"agentic answered {right_agentic} of {len(expected)} correctly, single-call "
            f"{right_single}")
    if loss >= NOISE_FLOOR_QUESTIONS:
        return NOT_MET, (f"{said}: {loss} fewer, and the noise floor is "
                         f"{NOISE_FLOOR_QUESTIONS}")
    return MET, f"{said}: within the noise floor of {NOISE_FLOOR_QUESTIONS} questions"


# -- reporting --------------------------------------------------------------------------


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{why} {n}" for why, n in sorted(counts.items())) or "none"


def extraction_table(tallies: Mapping[str, Tally]) -> str:
    rows = []
    for arm in ARMS:
        t = tallies[arm]
        rows.append([arm, t.writes, t.turns, t.llm_calls,
                     f"{t.tokens_in:,} / {t.tokens_out:,}", t.claims_written,
                     t.reinforced, t.ended, t.live_claims,
                     f"{t.duplicates} ({t.duplicate_same_slot} + "
                     f"{t.duplicate_other_predicate})",
                     t.agentic_runs])
    table = ek.render_table(
        ["arm", "writes", "turns", "model calls", "tokens in / out", "claims written",
         "reinforced", "ended", "live", "duplicates (slot + predicate)", "agentic runs"],
        rows)
    notes = [f"  {arm}: fallbacks: {_counts(tallies[arm].fallbacks)}; proposals refused: "
             f"{_counts(tallies[arm].refused)}; model calls that failed: "
             f"{tallies[arm].failed}" for arm in ARMS]
    return "\n".join([table, "", *notes])


REHEARSAL = ("  REHEARSAL: a scripted stand-in model wrote these claims and the string-match "
             "judge\n  graded any answers. The numbers say the script runs, not how either "
             "extractor does.")


def config_lines(llm: Any, args: Any) -> list[str]:
    lines = [f"  extraction model: {getattr(llm, 'name', type(llm).__name__)}, the same "
             "object in both arms",
             f"  write path: {args.write_path}"
             + (f", {args.batch} turn(s) per reextract()" if args.write_path == "worker"
                else ", one add() per batch")]
    return lines


def verdict_line(verdict: tuple[str, str], *, rehearsal: bool) -> str:
    if rehearsal:
        return "  bar: not measured (a dry run is a rehearsal, never a measurement)"
    return f"  bar: {verdict[0]} ({verdict[1]})"


# -- the command line -------------------------------------------------------------------


def build_extraction_model(args: Any) -> Any:
    """The extraction model both arms share. Refuses a missing key before anything runs."""
    if args.dry_run:
        return RehearsalChat()
    if not args.llm:
        raise SystemExit("Pass --llm anthropic or --llm openai: the bar compares two "
                         "extractors on one model, and there is no default model. Use "
                         "--dry-run to rehearse with no key.")
    variable = "ANTHROPIC_API_KEY" if args.llm == "anthropic" else "OPENAI_API_KEY"
    ek.require_key(variable, f"--llm {args.llm}")
    if args.llm == "anthropic":
        if args.llm_base_url:
            raise SystemExit("--llm-base-url applies to --llm openai.")
        from memvara.llm.anthropic import AnthropicLLM

        return AnthropicLLM(**({"model": args.llm_model} if args.llm_model else {}))
    from memvara.llm.openai import OpenAILLM

    kwargs: dict[str, Any] = {}
    if args.llm_model:
        kwargs["model"] = args.llm_model
    if args.llm_base_url:
        kwargs["base_url"] = args.llm_base_url
    return OpenAILLM(**kwargs)


def run_claims(args: Any, llm: Any, out: Callable[[str], None]) -> int:
    tallies = compare_claims(demo_batches(), llm, path=args.write_path, batch=args.batch)
    out("")
    out("  agentic extraction against single-call extraction: claims and duplicates")
    out("  corpus: demo/scenario.py, the support history demo/harness.py runs on")
    for line in config_lines(llm, args):
        out(line)
    out("")
    out(extraction_table(tallies))
    out("")
    out(verdict_line(claims_verdict(tallies), rehearsal=args.dry_run))
    if args.dry_run:
        out("")
        out(REHEARSAL)
    out("")
    return 0


def _read_rows(path: str | None) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def score_question(item: lme.Instance, arm: str, *, llm: Any, args: Any,
                   budget: ek.RetrievalBudget, embedder: Any, reader: ek.Reader,
                   judge: ek.Judge | None) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
    """One question in one arm: a fresh store, the haystack written through the arm's
    extraction, then retrieval, the reader and the judge as `bench/longmemeval.py` does."""
    def build(model: Any, **options: Any) -> Memvara:
        return lme.build_memory(item.qid, budget, model, embedder=embedder,
                                options=options)

    batches = [[{"role": t.role, "content": t.text, "ts": t.ts} for t in session]
               for session in item.sessions]
    tally, mem = write_arm(build, llm, batches, arm=arm, path=args.write_path,
                           batch=args.batch)
    try:
        pending = lme.prepare_one(mem, item, budget=budget, source=ek.ContextSource.MEMORY,
                                  read_stats=ek.RetrievalStats())
    finally:
        mem.close()
    result, calls = lme.finish_one(pending, reader=reader, judge=judge, stem=None)
    row = {"arm": arm, "qid": item.qid, "category": item.category,
           "judged": result.judged, "prediction": result.prediction, "gold": item.answer,
           "extraction": tally.to_dict()}
    return row, calls


def accuracy_table(rows: Sequence[Mapping[str, Any]], ids: Sequence[str]) -> str:
    by_arm = {arm: {r["qid"]: r for r in rows if r["arm"] == arm} for arm in ARMS}
    categories = sorted({r["category"] for r in rows})
    table = []
    for arm in ARMS:
        mine = [by_arm[arm][q] for q in ids if q in by_arm[arm]]
        cells = [arm, len(mine), sum(bool(r["judged"]) for r in mine)]
        for cat in categories:
            sub = [r for r in mine if r["category"] == cat]
            cells.append(f"{sum(bool(r['judged']) for r in sub)}/{len(sub)}")
        table.append(cells)
    return ek.render_table(["arm", "answered", "correct", *categories], table)


def paired_line(single: Mapping[str, Any], agentic: Mapping[str, Any],
                ids: Sequence[str]) -> str:
    both = [q for q in ids if q in single and q in agentic]
    wins = sum(1 for q in both if agentic[q] and not single[q])
    losses = sum(1 for q in both if single[q] and not agentic[q])
    return (f"  paired over {len(both)} questions: agentic right and single-call wrong "
            f"{wins}, single-call right and agentic wrong {losses}")


def run_longmemeval(args: Any, llm: Any, out: Callable[[str], None]) -> int:
    if args.dry_run:
        items = lme.fixture()
    else:
        path = args.data or ek.require(ek.LME_S, args.cache)
        items = select_sample(lme.load(path), load_sample())
    if args.limit:
        items = items[: args.limit]
    ids = [item.qid for item in items]

    reader = ek.build_reader(args)
    judge = ek.build_judge(args, reader)
    budget = ek.RetrievalBudget(k=args.k, max_chars=args.max_chars)
    embedder = ek.build_embedder("hashing")

    rows = [r for r in _read_rows(args.out) if r["qid"] in set(ids)]
    done = {(r["arm"], r["qid"]) for r in rows}
    jobs = [(item, arm) for item in items for arm in ARMS if (arm, item.qid) not in done]
    if done:
        out(f"\n  {len(done)} already scored in {args.out}; running the other {len(jobs)}")

    ledger = ek.TokenLedger()
    sink = Path(args.out).open("a", encoding="utf-8") if args.out else None
    try:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, args.concurrency)) as pool:
            futures = [pool.submit(score_question, item, arm, llm=llm, args=args,
                                   budget=budget, embedder=embedder, reader=reader,
                                   judge=judge) for item, arm in jobs]
            for future in concurrent.futures.as_completed(futures):
                row, calls = future.result()
                for role, answer in calls:
                    ledger.record(role, answer)
                rows.append(row)
                if sink is not None:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
    finally:
        if sink is not None:
            sink.close()

    tallies = {arm: Tally() for arm in ARMS}
    judged: dict[str, dict[str, bool | None]] = {arm: {} for arm in ARMS}
    for r in rows:
        tallies[r["arm"]].merge(Tally.from_dict(r["extraction"]))
        judged[r["arm"]][r["qid"]] = r["judged"]

    out("")
    out("  agentic extraction against single-call extraction: judged LongMemEval accuracy")
    out(f"  sample: {'the dry-run fixture' if args.dry_run else SAMPLE_PATH.name}, "
        f"{len(ids)} question(s), a fresh store per question and arm")
    for line in config_lines(llm, args):
        out(line)
    out(ek.run_settings_block(reader) or f"  reader {reader.name}")
    out(f"  judge {judge.name if judge is not None else 'none'}")
    out("")
    out(accuracy_table(rows, ids))
    out("")
    out(paired_line(judged["single_call"], judged["agentic"], ids))
    out(f"  noise floor: {NOISE_FLOOR_QUESTIONS} questions (7.8% reader self-disagreement, "
        "docs/BENCHMARKS.md)")
    out("")
    out(extraction_table(tallies))
    out("")
    verdict = accuracy_verdict(judged["single_call"], judged["agentic"], expected=ids)
    if not args.dry_run and len(ids) < len(load_sample()):
        verdict = (NOT_MEASURED, f"--limit ran {len(ids)} of the sample's "
                                 f"{len(load_sample())} questions")
    out(verdict_line(verdict, rehearsal=args.dry_run))
    out("  This compares the two arms in this repository's harness. The published 177 of "
        "199\n  came from the MemoryBench harness through the hosted service; only the "
        "difference\n  between the two arms here is the measurement.")
    if not getattr(reader, "is_stub", False) and ledger.rows():
        # The reader's and the judge's cost. The extraction model's tokens are in the
        # table above, from the receipts.
        out("")
        out(ek.cost_block(ledger))
    if args.dry_run:
        out("")
        out(REHEARSAL)
    out("")
    return 0


def main(argv: Sequence[str] | None = None,
         out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("half", choices=["claims", "longmemeval"],
                        help="claims: claims written and duplicates on the demo history. "
                             "longmemeval: judged accuracy on the 199-question sample")
    parser.add_argument("--dry-run", action="store_true",
                        help="a scripted stand-in model and, for longmemeval, the "
                             "three-question fixture and the string-match judge. No key")
    parser.add_argument("--llm", choices=["anthropic", "openai"], default=None,
                        help="the extraction backend both arms use")
    parser.add_argument("--llm-model", default=None,
                        help="the extraction model id; the backend's default when omitted")
    parser.add_argument("--llm-base-url", default=None, metavar="URL",
                        help="--llm openai: an OpenAI-compatible server for extraction")
    parser.add_argument("--write-path", choices=["add", "worker"], default="add",
                        help="add: one add() per batch, 25 s for the tool loop. worker: "
                             "store first, then reextract(), 180 s, as the hosted "
                             "extraction worker does")
    parser.add_argument("--batch", type=int, default=1, metavar="N",
                        help="--write-path worker: turns per reextract() call (default 1, "
                             "as the hosted worker)")
    # longmemeval only.
    parser.add_argument("--data", default=None, help="the LongMemEval-S JSON file")
    parser.add_argument("--cache", default=None, help="dataset cache directory")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="the first N questions of the sorted sample: a pilot, never "
                             "the bar")
    parser.add_argument("--reader", default="stub", choices=["stub", "anthropic", "openai"])
    parser.add_argument("--judge", default="none", choices=["none", "containment", "llm"])
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--k", type=int, default=12, help="retrieval budget, results")
    parser.add_argument("--max-chars", type=int, default=4000,
                        help="retrieval budget, characters of context")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="longmemeval: one JSON row per (arm, question), appended as "
                             "each finishes. A re-run with the same path resumes")
    ek.add_reader_arguments(parser)
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be at least 1")

    llm = build_extraction_model(args)
    if args.half == "claims":
        return run_claims(args, llm, out)
    return run_longmemeval(args, llm, out)


if __name__ == "__main__":  # pragma: no cover - entry point
    try:
        raise SystemExit(main())
    except ek.DatasetMissing as missing:
        print(f"\n{missing}")
        raise SystemExit(1)
