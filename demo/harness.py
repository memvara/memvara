"""Blinded five-arm answer-quality run over the support-history demo.

    PYTHONPATH=. python3 demo/harness.py --reader stub          # offline, one command
    PYTHONPATH=. python3 demo/harness.py --reader stub --corpus-scale 10   # the second size
    PYTHONPATH=. python3 demo/harness.py --reader anthropic --judge llm \\
        --model claude-opus-5 --effort low --max-tokens 4096 --thinking adaptive \\
        --checkpoint runs/hosted.checkpoint.jsonl --concurrency 4 --out runs/hosted.jsonl
    PYTHONPATH=. python3 demo/harness.py --dump   runs/demo.jsonl
    # ...answer them into runs/answers.jsonl as {"id": ..., "answer": ...}
    PYTHONPATH=. python3 demo/harness.py --dump runs/demo.jsonl --answers runs/answers.jsonl

Three kinds of reader, and the differences between them are the differences between a
smoke test, a measurement and a sanity check.

`--reader stub` runs both phases in one process against `evalkit.StubReader` and
`ContainmentJudge`, so it needs no key, no dump file and no answerer, and it produces the
same report twice. That is what makes it the *guarded* path: it is the form a test can
run, and a repeated run is a diff rather than a new experiment. What it cannot do is
measure answer quality — the stub picks the retrieved line with the most words in common
with the question, so its `correct` column is a property of the corpus and the arms and
nothing else. `evalkit.stub_caveat` prints that on every run of it, and the number must
never be quoted beside the ones in `docs/BENCHMARKS.md`.

`--reader anthropic` and `--reader openai` are the measurement: a model behind an API
answers every arm's questions in one process. Everything that decides its answers is
pinned by a flag and printed under the report's title exactly as it was sent — the model
id, the effort, the output budget and the thinking setting on Anthropic; the model id,
the output budget, the temperature and the seed on OpenAI — so the run can be repeated by
somebody else. The Anthropic models reject sampling parameters, so none is sent and the
header says so. `--judge llm` grades with a second model on the same provider: the
reader's twin, with `--judge-model` swapped in. `--checkpoint PATH` keeps every completed
call so a run that dies resumes rather than pays again, and `--concurrency N` issues N
calls at once; both live in `evalkit` and are shared with the `bench/` runners. The
report always carries the floor (`none`) and the ceiling (`full_transcript`) beside the
three memory arms and names which is which, because a memory score without both beside
it is uninterpretable. `--corpus-scale N` runs any reader over the authored history padded
to N times its length with generated tickets that move no fact (`demo/distractors.py`),
which is how the token argument gets its second point.

`--reader file` is the two-phase round trip for an answerer outside this process, a
person or an agent. Phase one writes one file containing every question under every arm
in `demo/baselines`, merged and shuffled together. Phase two reads the answers back,
re-derives which item belonged to which arm, and scores. Its number is a sanity check
rather than a measurement: there is no model id, seed or temperature to quote beside it,
and the same contexts answered again will not give the same answers.

## The two numbers, and why the second one is the headline

**correct** — judged against `Question.gold`.

**trapped** — the answer gave `Question.trap`, the superseded value. "Wrong" is a diffuse
category that mixes hallucination, abstention, misreading and bad luck; "said the value
that stopped being true in April" is one specific failure with one specific cause, and it
is the only failure a bitemporal memory layer claims to prevent. A marketing before/after
claim can rest on a trapped rate. It cannot rest on an accuracy delta, because an accuracy
delta at this sample size is noise wearing a percentage sign.

The two are not complements and are reported separately for that reason. An answer of
"they moved from Portland to Seattle in April" is both correct *and* contains the trap
string; `trapped only` — trapped and not correct — is the unambiguous failure, and it is
in the table beside the other two.

## Blinding

`evalkit.FileReader` does the blinding and its docstring is the specification; this module
does not invent a second dump format. What is arranged here on top of it:

* **One dump for all five arms.** Each arm is dumped into the same path in turn, and
  `FileReader.finish()` merges what is already there and re-shuffles the union. Five
  separate files, or one file written arm-by-arm, would be readable by position.
* **Order carries nothing.** `finish()` sorts by prompt digest and then shuffles with the
  seed, so the final order is independent of the order the arms ran in — which means
  there is nothing to leak by running them in a fixed sequence, and nothing to fix by
  running them in a random one.
* **One system prompt for every arm**, `SYSTEM`. A per-arm instruction would be a label.
* **The dump carries `{"id", "system_prompt", "prompt"}` and nothing else.** No arm name,
  no question id, no `kind`, no `trap`, no `gold`. The arm attribution lives in the key
  file, which `FileReader` writes separately and which is not opened until the answers
  are in.
* **`Today's date` is on every prompt**, from `Question.asked_at`, identically for all
  five arms. Without it "what plan are they on *now*" is unanswerable by construction and
  the floor arm is handicapped differently from the others; with it, it is a control.

**What is not blinded, and cannot be.** memvara's `recall()` writes its own headers
("Known about the user (stored notes…"), and an answerer who has read this repository
will know them on sight. Stripping them would change what is being measured — the
rendering is part of the product. Beyond that: `full_transcript` is the longest context in
the file by a wide margin and is in date order, and `none` is empty, so both are
identifiable by shape alone. `naive_rag` is the one arm with real cover, since it shares
`render_turns`'s exact line format with `full_transcript` and differs only in selection
and ordering. This is procedural blinding, worth exactly the discipline of whoever runs
it, and it is why `evalkit.stub_caveat` prints on every report.

## Judging

`ContainmentJudge` needs no key and works today. `LLMJudge` is used the moment a model is
available and is the better instrument. Note the judge is deliberately *not* built by
`evalkit.build_judge`: that function refuses `LLMJudge` next to a `FileReader`, on the
grounds that it would be the answerer marking their own work. Here the reader is a
`FileReader` by design and the judge is a separate model, which is the configuration that
function has no way to express.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# `bench/` is a directory of scripts rather than a package, so `evalkit` is imported by
# bare name exactly as its own runners import it. `demo/` *is* a package, so the sibling
# is imported by its package path: a bare `import baselines` from inside a package loads
# the same file a second time under a second name, and two module identities for one file
# means a monkeypatch, an `isinstance`, or a module-level constant silently applies to
# only one of them. The repository root goes on the path too, so `python3 demo/harness.py`
# works without `PYTHONPATH=.`.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "bench")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evalkit as ek  # noqa: E402

from demo.baselines import (  # noqa: E402
    ARMS,
    CHARS_PER_TOKEN,
    Arm,
    Context,
    Question,
    Turn,
)
from demo.distractors import scale_conversation  # noqa: E402

#: The reader's instruction, identical for every arm. Deliberately says nothing about
#: memory, retrieval or systems: naming any of that would tell the answerer what kind of
#: context to expect and what the experiment is about.
SYSTEM = (
    "You are a customer support assistant answering a question about one customer's "
    "account. Answer from the material below the question and from nothing else. Be "
    "brief — a value, a name, a date. If the material does not contain the answer, say "
    "you do not know rather than guessing."
)

#: `Question.kind` -> the `question_type` the judge is asked under. `ContainmentJudge`
#: only distinguishes abstention; `LLMJudge` uses the type to pick its prompt, and these
#: mappings are the ones that matter:
#:
#: * `current` and `correction` -> `knowledge-update`, whose prompt says in as many words
#:   that an answer reporting a superseded earlier value is wrong. That is the grading
#:   rule this whole run exists to apply.
#: * `historical` -> `default`. A question about a past state is *supposed* to be answered
#:   with the old value; grading it under `knowledge-update` would mark the correct answer
#:   wrong, which would flatter every arm that cannot do history.
#: * `unanswerable` -> `abstention`, the one type `ContainmentJudge` also honours.
JUDGE_TYPES: dict[str, str] = {
    "current": "knowledge-update",
    "correction": "knowledge-update",
    "historical": "default",
    "unanswerable": ek.ABSTENTION_TYPE,
}

#: The type the *trap* check is made under, for every `kind`. `default`'s prompt is
#: "answer yes if the response contains the correct answer" — asked with `gold=trap`, that
#: is exactly the question "did the response give the superseded value", which is what is
#: being counted. It must not be `knowledge-update` (whose prompt would grade the trap
#: *down* for being superseded, inverting the count) and it must not be `abstention` (a
#: trapped answer on an unanswerable question is the opposite of an abstention).
TRAP_JUDGE_TYPE = "default"


# --- planning the run -----------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One (arm, question) pair and the exact prompt it produces."""

    id: str
    arm: str
    qid: str
    prompt: str
    context: Context


def resolve_arms(arms: Mapping[str, Arm] | None) -> Mapping[str, Arm]:
    """`ARMS` when nothing was asked for, and never when an empty mapping was.

    `arms or ARMS` reads correctly and is wrong: an empty dict is falsy, so a caller
    narrowing the run to a subset that turned out to be empty would silently get all five
    arms back and a report that looked fine. Explicit `is None`, then a refusal.
    """
    if arms is None:
        return ARMS
    if not arms:
        raise ValueError("no arms: a comparison needs at least one")
    return arms


def plan(questions: Sequence[Question], turns: Sequence[Turn], *,
         arms: Mapping[str, Arm] | None = None) -> list[Item]:
    """Every (arm, question) pair, with its prompt and its blinded id.

    Deterministic, and used by both phases: the dump writes these, and scoring re-derives
    them to map an answer id back to the arm that produced it. Re-deriving rather than
    reading the arm out of the key file is what makes a stale dump detectable — see
    `stale_ids`.
    """
    arms = resolve_arms(arms)
    out: list[Item] = []
    for name, arm in arms.items():
        for q in questions:
            ctx = arm(q, turns)
            prompt = ek.build_prompt(q.text, ctx.text,
                                     asked_on=q.asked_at.date().isoformat())
            out.append(Item(id=ek.item_id(prompt), arm=name, qid=q.id, prompt=prompt,
                            context=ctx))
    return out


def key_path_for(dump_path: str | Path) -> Path:
    """Where `FileReader` puts the key for this dump.

    Asked of `FileReader` rather than reconstructed from the suffix rule, so the two
    cannot drift. Constructing one is free — nothing is written until `finish()`.
    """
    return ek.FileReader(dump=str(dump_path)).key_path


# --- phase one: the dump --------------------------------------------------------


@dataclass(frozen=True)
class DumpResult:
    path: Path
    key_path: Path
    items: int
    #: Ids produced by more than one (arm, question) pair. `FileReader.finish()` merges
    #: those into a single item and keeps the first arm's attribution, silently — so a
    #: five-arm dump would quietly become a four-and-a-bit-arm dump while the report
    #: still said five. Surfaced rather than repaired: two arms that build byte-identical
    #: context for a question genuinely are the same arm for that question, and the honest
    #: response is to say so rather than to perturb one of them apart.
    collisions: dict[str, list[str]] = field(default_factory=dict)
    note: str = ""


def dump(questions: Sequence[Question], turns: Sequence[Turn],
         path: str | Path, *, arms: Mapping[str, Arm] | None = None,
         seed: int = 20260813, now: Any = None) -> DumpResult:
    """Write every arm's questions into one blinded file, plus its key.

    One `FileReader` per arm, all pointed at the same path, each merging what the last one
    wrote. That is the documented head-to-head flow and the only one that produces a dump
    an answerer cannot read by position.
    """
    items = plan(questions, turns, arms=arms)
    owners: dict[str, list[str]] = {}
    for item in items:
        owners.setdefault(item.id, []).append(f"{item.arm}/{item.qid}")

    for name in resolve_arms(arms):
        reader = ek.FileReader(dump=str(path), seed=seed, system_label=name, now=now)
        for item in items:
            if item.arm == name:
                reader.answer(SYSTEM, item.prompt)
        note = reader.finish()

    return DumpResult(path=Path(path), key_path=key_path_for(path), items=len(owners),
                      collisions={k: v for k, v in owners.items() if len(v) > 1},
                      note=note)


# --- phase two: scoring ---------------------------------------------------------


@dataclass
class Tally:
    """One cell of the results table: an arm, or an arm crossed with a `kind`."""

    n: int = 0
    #: Items an answer came back for. An unanswered item is counted in `n` and never in
    #: `correct`, so a partial run reads as a low score *and* prints a missing count,
    #: rather than reading as a high score over whatever happened to be answered.
    answered: int = 0
    correct: int = 0
    #: Items with a `trap` defined at all. The denominator for `trapped`, and never `n`: a
    #: trapped rate over questions with no superseded value to give is a rate diluted by
    #: questions that could not have produced it.
    trap_defined: int = 0
    trapped: int = 0
    #: Trapped and not correct — gave the superseded value and nothing else. The
    #: unambiguous failure, separated from answers that recite the whole history.
    trapped_only: int = 0


def rate(num: int, den: int) -> str:
    """A percentage, or `-` when the denominator is empty. Never 0% for 0/0."""
    return "-" if den == 0 else f"{100.0 * num / den:.0f}%"


@dataclass
class Scored:
    """One judged answer, so a run can be audited rather than trusted."""

    id: str
    arm: str
    qid: str
    kind: str
    question: str
    gold: str
    trap: str | None
    answer: str
    correct: bool
    trapped: bool
    context_chars: int
    #: Why the reader stopped, from the provider: `end_turn` for a finished answer,
    #: `max_tokens` for one the budget cut off, `refusal` for one a classifier declined.
    #: `""` for the stub and the file reader, which consult no model. Kept per row so an
    #: audit can see *which* items were never answered rather than only how many; the
    #: report's cost block prints the counts.
    stop_reason: str = ""


def stale_ids(key_path: str | Path, items: Sequence[Item]) -> list[str]:
    """Ids the key file records that the current scenario no longer produces.

    The key file is the record of what was actually dumped. Scoring re-derives the mapping
    from `demo/scenario.py` as it stands *now*, and in a repository where another
    workstream owns that file the two can drift between the dump going out and the answers
    coming back. A silent drift scores a subset of the run and reports it as the whole run,
    so the mismatch is returned and the caller refuses.
    """
    known = {item.id for item in items}
    rows = json.loads(Path(key_path).read_text(encoding="utf-8"))["items"]
    return [str(row["id"]) for row in rows if str(row["id"]) not in known]


def hosted(reader: ek.Reader) -> bool:
    """True for a model behind an API: neither the stub nor a person."""
    return not getattr(reader, "is_stub", False) and not getattr(reader, "is_human", False)


def score(items: Sequence[Item], questions: Sequence[Question], *,
          reader: ek.Reader, judge: ek.Judge, ledger: ek.TokenLedger | None = None,
          concurrency: int = 1) -> list[Scored]:
    """Judge every answer for correctness and for the trap, separately.

    The answers come back through a `Reader` rather than a dict so that the same path
    serves a `FileReader` holding a completed round trip and, unchanged, an API reader —
    and so the unanswered count is `FileReader.missing`'s rather than a second
    reimplementation of it.

    One job per item — the reader's call and up to two judge calls — issued
    `concurrency` at a time through `ek.score_in_order`, which bills every call into
    `ledger` in item order on the calling thread. At the default of 1 nothing runs on
    another thread, and the rows come back in item order whatever the concurrency was.
    """
    by_id = {q.id: q for q in questions}

    def one(item: Item) -> tuple[Scored, list[tuple[str, ek.Answer]]]:
        q = by_id[item.qid]
        out = reader.answer(SYSTEM, item.prompt)
        hypothesis = out.text
        calls = [("reader", out)]
        correct = trapped = False
        if hypothesis:
            correct, verdict = judge.judge(q.text, q.gold, hypothesis,
                                           JUDGE_TYPES.get(q.kind, "default"))
            calls.append(("judge", verdict))
            if q.trap is not None:
                trapped, verdict = judge.judge(q.text, q.trap, hypothesis, TRAP_JUDGE_TYPE)
                calls.append(("judge", verdict))
        return Scored(id=item.id, arm=item.arm, qid=q.id, kind=q.kind,
                      question=q.text, gold=q.gold, trap=q.trap,
                      answer=hypothesis, correct=correct, trapped=trapped,
                      context_chars=item.context.chars,
                      stop_reason=out.stop_reason), calls

    return ek.score_in_order(one, items,
                             ledger=ek.TokenLedger() if ledger is None else ledger,
                             concurrency=concurrency)


def tally(scored: Iterable[Scored], key: Callable[[Scored], Any]) -> dict[Any, Tally]:
    """Group scored answers by `key(row)` and count them. Insertion order is preserved,
    so a caller that hands rows over in arm order gets a table in arm order."""
    out: dict[Any, Tally] = {}
    for row in scored:
        cell = out.setdefault(key(row), Tally())
        cell.n += 1
        cell.answered += bool(row.answer)
        cell.correct += row.correct
        if row.trap is not None:
            cell.trap_defined += 1
            cell.trapped += row.trapped
            cell.trapped_only += row.trapped and not row.correct
    return out


# --- reporting ------------------------------------------------------------------


def size_table(items: Sequence[Item], arms: Mapping[str, Arm] | None = None) -> str:
    """Context size per arm: the argument for a memory layer, or against one.

    `full_transcript`'s row is the point of the table. At a small corpus size it is small
    enough to be a serious competitor, and the honest reading of a run where it wins is
    that the memory layer is not earning its place *yet*. The row that decides whether it
    ever does is not the accuracy — it is this one, multiplied by however many months of
    history the deployment actually has.
    """
    rows = []
    for name in resolve_arms(arms):
        mine = [i for i in items if i.arm == name]
        chars = [i.context.chars for i in mine]
        used = [i.context.items_used for i in mine]
        seen = [i.context.turns_visible for i in mine]
        rows.append([
            name,
            f"{sum(chars) / len(chars):.0f}",
            max(chars),
            f"{sum(chars) / len(chars) / CHARS_PER_TOKEN:.0f}",
            f"{sum(used) / len(used):.1f} / {sum(seen) / len(seen):.1f}",
        ])
    return ek.render_table(
        ["arm", "mean chars", "max chars", "mean ~tokens", "items used / turns seen"],
        rows)


def results_table(cells: Mapping[Any, Tally], label: str) -> str:
    rows = []
    for key, t in cells.items():
        rows.append([
            key if isinstance(key, str) else " / ".join(str(k) for k in key),
            t.n, t.answered,
            rate(t.correct, t.n),
            f"{rate(t.trapped, t.trap_defined)} ({t.trapped}/{t.trap_defined})",
            rate(t.trapped_only, t.trap_defined),
        ])
    return ek.render_table([label, "n", "answered", "correct", "trapped", "trapped only"],
                           rows)


#: Everything the containment judge gets wrong here, printed on every run that uses it.
#: Longer than `evalkit.stub_caveat`'s one line because this harness asks the judge a
#: second question it was never designed for — "did the answer give the trap" — and the
#: failure modes of that use are not the failure modes of grading against gold.
CONTAINMENT_CAVEAT = """\
  THE JUDGE IS A SUBSTRING RULE, not a model. Specifically, on this run:

  * It marks a correct paraphrase WRONG. gold "Starter" against an answer of "the
    entry-level tier" scores zero. Every `correct` column is therefore a floor, and the
    floor is lower for whichever arm produces answers in its own words.
  * It marks an answer CORRECT for containing the gold string in any role, including
    "they are NOT on Starter any more" and "Starter, Pro or Growth — unclear".
  * `trapped` is the same rule run with the trap as the reference, so an answer that
    recites the history ("upgraded to Pro in February, back to Starter in June") counts
    as BOTH correct and trapped. That is what `trapped only` is for.
  * `ContainmentJudge` also passes on token-F1 >= 0.6, which for a one-word gold or trap
    fires on any terse answer sharing most of its words. Short answers are graded more
    generously than long ones, in both directions.
  * Abstention is detected by marker phrases, so an `unanswerable` question answered
    "there is nothing in the account about that" scores by whether that wording happens
    to be on the list.

  Re-run with --judge llm once a key exists. Nothing about the arms changes; only the
  instrument does."""


def run_header(reader: ek.Reader, judge: ek.Judge, arms: Mapping[str, Arm]) -> str:
    """What a run is quoted with, printed under the title.

    Nothing for the stub, which has no parameters and whose report is pinned byte for
    byte by `test_the_offline_run_is_identical_twice`. For every other reader: the
    reader and its pinned parameters (`ek.run_settings_block`), the judge, which arm is
    the floor and which the ceiling — a memory score without both beside it is
    uninterpretable, and a narrowed run has to say what it lacks rather than be read as
    if it had them — and what the checkpoint replayed.
    """
    if getattr(reader, "is_stub", False):
        return ""
    lines = [ek.run_settings_block(reader) or f"  reader {reader.name}"]
    # A model judge prints its own parameters: it is the reader's twin by construction
    # on the hosted path, but the header should say so rather than have it inferred.
    judge_lines = ek.reader_settings_lines(getattr(judge, "reader", judge),
                                           sampling_note=False)
    lines.append(f"  judge {judge.name}" + (":" if judge_lines else ""))
    lines += judge_lines
    memory = [name for name in arms if name not in ("none", "full_transcript")]
    lines.append("  " + "   ".join([
        "floor: none" if "none" in arms else "NO FLOOR ARM in this run",
        ("ceiling: full_transcript (a reader ceiling, not a memory result)"
         if "full_transcript" in arms else "NO CEILING ARM in this run"),
    ]))
    lines.append("  " + ("measurement: " + ", ".join(memory) if memory
                         else "NO MEMORY ARM in this run"))
    if "none" not in arms or "full_transcript" not in arms:
        lines.append("  A memory score with no floor and no ceiling beside it is "
                     "uninterpretable. Run all five arms.")
    note = ek.checkpoint_note(reader)
    if note:
        lines.append(note)
    return "\n".join(lines)


def corpus_note(scale: int, authored: int, total: int) -> str:
    """One line naming the haystack when it is not the authored one, `""` when it is.

    Empty at scale 1 rather than "corpus: the authored 64 turns", so the offline report
    stays byte-identical to what it was before the flag existed. At any other scale the
    line is what stops a table of "turns seen" being read as the authored corpus.
    """
    if scale == 1:
        return ""
    return (f"  corpus: scale {scale} — {total} turns, the {authored} authored plus "
            f"{total - authored} generated distractor-ticket turns that name no value a "
            "question is about (demo/distractors.py)")


def report(items: Sequence[Item], scored: Sequence[Scored], *, reader: ek.Reader,
           judge: ek.Judge, arms: Mapping[str, Arm] | None = None,
           ledger: ek.TokenLedger | None = None, corpus: str = "") -> str:
    """The whole result, including everything that makes it less than it looks."""
    arms = resolve_arms(arms)
    order = list(arms)
    out = ["", "  five-arm answer quality, blinded", ""]
    if corpus:
        out += [corpus, ""]
    header = run_header(reader, judge, arms)
    if header:
        out += [header, ""]
    out.append(size_table(items, arms))

    degraded = sorted({i.arm for i in items if i.context.degraded})
    if degraded:
        out += ["", "  EXTRACTION RAN DEGRADED for: " + ", ".join(degraded) + ".",
                "  No `llm=`, so only the deterministic rule extractor contributed claims and",
                "  everything it did not recognise reached the prompt as a raw turn rather",
                "  than a resolved fact. That is a real configuration and it is the one",
                "  measured here, but it is the memory layer's weaker half: supersession only",
                "  fires on facts that became claims."]
    if "memvara_structured" in degraded:
        # Without this the line above reads as a caveat on the structured arm's claim
        # tier, which would be false and would understate the only arm that exercises the
        # thing being tested. Its claims came from the desk's own fields; no model was
        # asked to read them back out of prose, which is the entire point of it.
        out += ["", "  It reads differently for `memvara_structured`, and the difference is",
                "  the finding: its claims come from declared, structured writes, so the",
                "  missing model costs it only the episode tail. On this corpus the",
                "  `memvara` arm's claim tier is EMPTY — the rule extractor's vocabulary is",
                "  first-person declaratives and a support history is not written that way —",
                "  so that arm is lexical episode retrieval with a different ranker, and its",
                "  row is not a measurement of bitemporal reasoning. The structured arm's is."]

    naive = [i for i in items if i.arm == "naive_rag"]
    saturated = [i for i in naive
                 if i.context.turns_visible > 0
                 and i.context.items_used >= i.context.turns_visible]
    if saturated:
        out += ["", f"  naive_rag RETRIEVED EVERY VISIBLE TURN on {len(saturated)} of "
                    f"{len(naive)} questions.",
                "  On those it is `full_transcript` in a different order, not a retrieval",
                "  arm, and any difference between the two rows is the ordering alone. The",
                "  corpus needs more turns than k for that comparison to mean anything."]

    missing = sum(1 for r in scored if not r.answer)
    if missing:
        out += ["", f"  {missing} of {len(scored)} items have no answer. They are counted",
                "  as incorrect, which is right for a scoring run and wrong for a",
                "  conclusion: finish the dump before quoting any of these numbers."]

    out += ["", "  per arm", ""]
    out.append(results_table(tally(scored, lambda r: r.arm), "arm"))
    out += ["", "  per arm and question kind", ""]
    by_kind = sorted(scored, key=lambda r: (order.index(r.arm), r.kind))
    out.append(results_table(tally(by_kind, lambda r: (r.arm, r.kind)), "arm / kind"))

    if any(r.arm == "none" and r.kind == "unanswerable" for r in scored):
        out += ["", "  The `none / unanswerable` row is an artefact, not a finding. An arm",
                "  with no context abstains because it has nothing to abstain from, so it",
                "  scores that kind by construction and would do so on any corpus. Read the",
                "  floor on the other three kinds."]

    out += ["", "  `trapped` is the headline. `correct` says an arm produced an answer;",
            "  `trapped` says it gave the specific superseded value the product claims to",
            "  prevent, and that is the only column a before/after claim can rest on.", ""]

    if ledger is not None and hosted(reader) and ledger.rows():
        # Tokens, dollars and the answers that never finished, from the usage the
        # provider reported. Only for a model: the stub and the file reader spend nothing
        # and a cost block for them would be a table of zeros under a heading that reads
        # as a finding.
        out += [ek.cost_block(ledger)]

    caveat = ek.stub_caveat(reader, judge)
    if caveat:
        out += [caveat, ""]
    if isinstance(judge, ek.ContainmentJudge):
        out += [CONTAINMENT_CAVEAT, ""]
    return "\n".join(out)


def write_jsonl(path: str | Path, scored: Sequence[Scored]) -> None:
    """Per-question output, so a run can be audited rather than trusted."""
    with Path(path).open("w", encoding="utf-8") as out:
        for row in scored:
            out.write(json.dumps(vars(row), ensure_ascii=False) + "\n")


# --- the offline run ------------------------------------------------------------


@dataclass(frozen=True)
class Offline:
    """One end-to-end run with the answerer inside this process.

    Carries the reader and the judge as well as the rows because `report` needs both to
    print the right caveats, and a caller that had to rebuild them could rebuild them
    differently from the run they are describing. The ledger is what the run cost, from
    the usage the provider reported; `None` only when a caller scored without one.
    """

    items: tuple[Item, ...]
    scored: tuple[Scored, ...]
    reader: ek.Reader
    judge: ek.Judge
    ledger: ek.TokenLedger | None = None
    #: `corpus_note`'s line, or `""` for the authored corpus.
    corpus: str = ""

    def report(self, *, arms: Mapping[str, Arm] | None = None) -> str:
        return report(list(self.items), list(self.scored), reader=self.reader,
                      judge=self.judge, arms=arms, ledger=self.ledger, corpus=self.corpus)


def in_process(questions: Sequence[Question], turns: Sequence[Turn], *,
               reader: ek.Reader, judge: ek.Judge,
               arms: Mapping[str, Arm] | None = None,
               concurrency: int = 1, corpus: str = "") -> Offline:
    """Plan, answer and score in one process: the stub, or a model behind an API.

    The dump/answers round trip exists because a person or an agent cannot be in this
    process. A stub can, and so can a model behind an API, which reads one prompt at a
    time and has no memory between them — so blinding has nothing to protect against
    here and skipping it is not a shortcut. `FileReader`'s shuffle defends against an
    answerer who can read the file; neither of these can.
    """
    items = plan(questions, turns, arms=arms)
    ledger = ek.TokenLedger()
    scored = score(items, questions, reader=reader, judge=judge, ledger=ledger,
                   concurrency=concurrency)
    return Offline(items=tuple(items), scored=tuple(scored), reader=reader, judge=judge,
                   ledger=ledger, corpus=corpus)


def offline(questions: Sequence[Question], turns: Sequence[Turn], *,
            arms: Mapping[str, Arm] | None = None,
            reader: ek.Reader | None = None,
            judge: ek.Judge | None = None) -> Offline:
    """`in_process` with the stub reader and the containment judge: no key, no file.

    What this path is *for* is repeatability: it is deterministic end to end, so two runs
    of it differ only where the library does, which is the property a test can assert and
    a `git bisect` can use. Its accuracy column is not a measurement of anything — see the
    module docstring and `evalkit.StubReader`.
    """
    return in_process(questions, turns, arms=arms,
                      reader=ek.StubReader() if reader is None else reader,
                      judge=ek.ContainmentJudge() if judge is None else judge)


# --- CLI ------------------------------------------------------------------------


def build_reader(args: Any) -> ek.Reader:
    """The in-process reader `--reader` names: the stub, or a model with its parameters
    pinned by the `ek.add_reader_arguments` flags, checkpointed when `--checkpoint` was
    given. The file reader is built where its two phases are, in `main`."""
    reader = ek.StubReader() if args.reader == "stub" else ek.hosted_reader(args.reader, args)
    if args.checkpoint:
        return ek.CheckpointedReader(reader, ek.Checkpoint(args.checkpoint))
    return reader


def build_judge(name: str, *, model: str | None = None,
                like: ek.Reader | None = None) -> ek.Judge:
    """The judge, by name. Containment works today; llm works the moment a key exists.

    `like` is the reader to build the model judge from: its twin on the same provider
    with the same pinned parameters and `model` swapped in, so a run states one
    configuration. Without one — a direct call with nothing to mirror — the judge is an
    `AnthropicReader` on its defaults.

    Not `evalkit.build_judge`: that one refuses an `LLMJudge` beside a `FileReader`,
    because in its runners the reader and the judge would be the same party. Here they
    never are — on the file round trip the reader is a person and the judge a model —
    so the refusal does not apply, and the configuration it forbids is the correct one.
    """
    if name == "containment":
        return ek.ContainmentJudge()
    if like is not None:
        return ek.LLMJudge(like.spawn(model))
    return ek.LLMJudge(ek.AnthropicReader(model=model or "claude-opus-5"))


def load_scenario() -> tuple[list[Question], list[Turn]]:
    """`demo/scenario.py`, or a message naming who owns it.

    Imported here rather than at module scope so the arms and the scoring machinery stay
    importable, and testable, without it — this is the one line of the demo that depends
    on the corpus, and everything else is written against the contract.
    """
    try:
        from demo import scenario
    except ImportError as exc:  # pragma: no cover - a missing sibling module
        raise SystemExit(
            "demo/scenario.py is missing. It supplies conversation() and questions(); "
            "this module supplies the arms, the blinding and the scoring."
        ) from exc
    return list(scenario.questions()), list(scenario.conversation())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blinded five-arm answer-quality run.")
    parser.add_argument("--reader", default="file",
                        choices=["file", "stub", "anthropic", "openai"],
                        help="file: the two-phase blinded round trip for a person or an "
                             "agent. stub: one offline, deterministic process, for "
                             "checking the pipeline runs. anthropic | openai: a model "
                             "behind an API, in one process — the measurement; pin it "
                             "with --model and the flags beside it")
    parser.add_argument("--dump", metavar="PATH", default=None,
                        help="the blinded dump: written in phase one, and read for its "
                             "key file in phase two. Required by --reader file")
    parser.add_argument("--answers", metavar="PATH", default=None,
                        help='phase two: {"id": ..., "answer": ...} per line')
    parser.add_argument("--seed", type=int, default=20260813,
                        help="shuffle seed, recorded in the key file")
    parser.add_argument("--corpus-scale", type=int, default=1, metavar="N",
                        help="pad the history to N times its authored length with "
                             "generated distractor tickets that name no value a question "
                             "is about (demo/distractors.py). 1, the default, is the "
                             "authored corpus. Both phases of --reader file must use the "
                             "same N; the stale-dump check catches a mismatch")
    parser.add_argument("--judge", default="containment", choices=["containment", "llm"])
    parser.add_argument("--judge-model", default=None,
                        help="model for --judge llm: the reader's twin with this model "
                             "swapped in. Defaults to the reader's own model on a hosted "
                             "reader, and to claude-opus-5 on the file round trip")
    parser.add_argument("--out", default=None, help="write per-question JSONL here")
    # --model, --effort, --max-tokens, --thinking, --temperature, --sampling-seed,
    # --concurrency and --checkpoint: one definition, shared with the bench/ runners.
    ek.add_reader_arguments(parser)
    args = parser.parse_args(argv)
    if args.corpus_scale < 1:
        parser.error("--corpus-scale must be at least 1")

    questions, turns = load_scenario()
    authored = len(turns)
    turns = scale_conversation(turns, args.corpus_scale)
    corpus = corpus_note(args.corpus_scale, authored, len(turns))

    if args.reader != "file":
        # The stub and the hosted readers share one path, because both are inside this
        # process. It deliberately ignores --dump and --answers rather than refusing
        # them: those belong to the round trip, and a run that is told to blind itself
        # against a reader that cannot read the file has nothing to blind. --judge llm
        # grades with the reader's twin on the hosted path; over stub answers it is an
        # Anthropic judge on the same flags, a real thing to want when debugging the
        # judge itself.
        reader = build_reader(args)
        like = None
        if args.judge == "llm":
            like = reader if hosted(reader) else ek.hosted_reader("anthropic", args)
        judge = build_judge(args.judge, model=args.judge_model, like=like)
        run = in_process(questions, turns, reader=reader, judge=judge,
                         concurrency=args.concurrency, corpus=corpus)
        print(run.report())
        if args.out:
            write_jsonl(args.out, run.scored)
        return 0

    if args.checkpoint or args.concurrency > 1:
        parser.error("--checkpoint and --concurrency apply to --reader anthropic, openai "
                     "or stub. The file round trip is its own resume mechanism, and its "
                     "answerer is outside this process.")
    if args.dump is None:
        parser.error("--dump is required by --reader file")

    if args.answers is None:
        result = dump(questions, turns, args.dump, seed=args.seed)
        print(result.note)
        if result.collisions:
            print(f"  {len(result.collisions)} PROMPT COLLISIONS: two arms built "
                  "byte-identical context, so those merged into one item and the dump "
                  "holds fewer items than arms x questions.")
            for cid, owners in sorted(result.collisions.items()):
                print(f"    {cid}  {', '.join(owners)}")
        return 0

    items = plan(questions, turns)
    stale = stale_ids(key_path_for(args.dump), items)
    if stale:
        print(f"  {len(stale)} items in the key file are not produced by the scenario as "
              "it stands now.\n  The dump is stale: re-dump and re-answer, or score the "
              "run the dump belongs to.")
        return 1

    reader = ek.FileReader(answers=args.answers)
    judge = build_judge(args.judge, model=args.judge_model,
                        like=ek.hosted_reader("anthropic", args) if args.judge == "llm"
                        else None)
    scored = score(items, questions, reader=reader, judge=judge)
    print(report(items, scored, reader=reader, judge=judge, corpus=corpus))
    if args.out:
        write_jsonl(args.out, scored)
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
