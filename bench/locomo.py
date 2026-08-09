"""LOCOMO — long-term conversational memory, run against memvara.

    PYTHONPATH=. python3 bench/locomo.py --dry-run                    # offline, no key
    PYTHONPATH=. python3 bench/locomo.py --download                   # 2.8 MB, once
    PYTHONPATH=. python3 bench/locomo.py --score retrieval            # no model at all
    PYTHONPATH=. python3 bench/locomo.py --reader anthropic --judge llm

## Scoring retrieval on its own

`--score retrieval` drops the reader entirely and asks only whether retrieval surfaced
the evidence — see `evalkit.score_retrieval`. LOCOMO supports the strong form of that
question: every turn carries a `dia_id` and every QA item lists the `dia_id`s an
annotator marked as its evidence, so "did the memory return the turn the answer is in"
is a fact about the run rather than a string comparison. 2,815 evidence fields across
the file, nine of which are malformed: six pack several ids into one string (`"D9:1 D4:4
D4:6"`, `"D8:6; D9:17"`) and three are simply wrong (a bare `"D"`, a `"D:11:26"`, a
`"D30:05"`). Splitting on separators recovers the first six and turns the file's 2,815
fields into 2,824 ids, of which exactly 5 still name no turn; those 5 questions leave
the evidence table and are counted in the report rather than repaired by guessing. Four
more questions carry an empty `evidence` list and leave it for that reason.

Category 5 is excluded from this mode in both measures. It has no gold answer to look
for, and retrieving the turns its bait was built from is neither success nor failure.
Exactly one answerable gold reduces to no content tokens under the presence rule and is
reported as unmeasurable rather than as a miss, which is why the string table reads
n=1,539 against 1,540 questions. It is the TV show titled `"That"`, eaten by
`evalkit.PRESENCE_STOPWORDS` — a real cost of that list, named here rather than
absorbed, and one question in 1,540 is the size of it.

## The dataset, and where it actually comes from

`locomo10.json` lives in the `snap-research/locomo` GitHub repository and is **2.8 MB,
public, ungated and needs no token** — verified against the host, not assumed. It is
not vendored here; `--download` puts it in `$MEMVARA_BENCH_DATA` (default
`~/.cache/memvara-bench`) and a run without it fails with the URL and a `curl` command.

Measured from the file itself: 10 conversations, 272 sessions, 5,882 turns, 726,756
characters of dialogue, and 1,986 QA items — 1,540 in categories 1–4 and 446 in category
5. The "1,540 questions" figure papers quote is the answerable subset, and this harness
reports it that way.

Two things the file will do to a careless loader. It holds **288**
`session_N_date_time` keys against those 272 session lists — 16 timestamps name a
session that is not in the file — so the loader walks the session *lists* and lets the
orphan dates fall away rather than materialising sixteen empty sessions. And 1,226 turns
carry an image with a BLIP caption; those captions are ingested as text, because
dropping them silently would remove the evidence for some questions while leaving the
questions in the denominator. This is still a text-only run of a multimodal benchmark
and that is a real caveat, not a footnote. All 272 session timestamps parse; the
undated-turn counter in the report exists for the file changing under us, not for a
defect in it today.

| category | meaning       | n     | scored by                        |
|----------|---------------|-------|----------------------------------|
| 1        | multi-hop     | 282   | F1, BLEU-1, judge                |
| 2        | temporal      | 321   | F1, BLEU-1, judge                |
| 3        | open-domain   | 96    | F1, BLEU-1, judge                |
| 4        | single-hop    | 841   | F1, BLEU-1, judge                |
| 5        | adversarial   | 446   | abstention rate — **not** F1     |

Category 5 asks about things that never happened. Its `answer` field is absent and an
`adversarial_answer` holds the plausible wrong answer the question baits; that bait is
deliberately never used as a gold, and the reference rule — output contains "no
information available" or "not mentioned" — is what scores it. The reader is instructed
to use the first phrase, so the rule is a check on whether it abstained rather than a
vocabulary lottery.

## How memvara ingests two humans talking

Every turn is written with `role` set to the speaker's name and the speaker's name
prefixed into the text, one `add()` per session, timestamped from
`session_N_date_time`. Two consequences, both worth knowing before reading a score:

* **The deterministic write path is inert on this data.** `SalienceGate` drops any turn
  whose role is not `user`, and `FastExtractor` skips them too — both are built around
  a single first-person user, which two named humans are not. So with no `llm=`
  configured, nothing is extracted, every turn is stored as an episode, and retrieval is
  episode retrieval. That is the honest behaviour of the shipped defaults on this
  dataset, and the run reports `claims added: 0` rather than hiding it.
* **`read_max_episodes` is raised to `k`.** The library caps raw turns at 3 per read
  because they are meant to be a tail on a list of facts. Here they are the whole
  answer, so the cap would be a harness bug rather than a property of the system.

## What a real run costs

Per conversation the reader sees a question plus at most `--max-chars` of context, so
roughly 1.2k input tokens and under 100 output tokens per question. Across all 1,986
questions that is on the order of 2.5M input and 0.2M output tokens: about **$17 on
`claude-opus-5`** at $5/$25 per MTok, or about $10 on `claude-sonnet-5`. Adding
`--judge llm` roughly doubles the call count at much smaller prompts — budget another
$3–5. The 1,540-question answerable subset is proportionally cheaper. Ingestion costs
nothing unless an extraction model is configured.

## What this does not establish

It compares memvara against itself under three context sources (see
`evalkit.ContextSource`), not against another memory layer. A published LOCOMO score
was produced with a different reader, a different retrieval budget and a different
judge; quoting this number beside one of those compares harnesses. And the answer format
is squeezed short by the system prompt because F1 punishes verbosity — which is a
property of the metric, not of the memory, and is exactly why the judge column exists.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import evalkit as ek

from memvara import Memvara, NullLLM

CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
ANSWERABLE = (1, 2, 3, 4)
ADVERSARIAL = 5

SYSTEM = (
    "You answer questions about a conversation between two people, using only the "
    "retrieved excerpts below the question.\n"
    "Answer with the shortest phrase that answers it — a name, a date, a place, a short "
    "list. No sentence, no preamble, no explanation.\n"
    "For a question about when something happened, give the date as the excerpts give it.\n"
    "If the excerpts do not contain the answer, reply exactly: No information available."
)

#: `"1:56 pm on 8 May, 2023"` — the only shape in the file, checked across all 288
#: session timestamps. Parsed rather than ignored because 321 of the questions are
#: temporal and memvara's whole proposition is that time is a first-class axis.
WHEN_FORMAT = "%I:%M %p on %d %B, %Y"


# --- the dataset ----------------------------------------------------------------


@dataclass(slots=True)
class Session:
    index: int
    when: datetime | None
    turns: list[ek.Turn]
    raw_when: str = ""


@dataclass(slots=True)
class LocomoQA:
    question: str
    answer: str
    category: int
    #: Position in the conversation's `qa` list *as the file has it*. The question id is
    #: built from this rather than from the loop counter so a row in the JSONL points at
    #: the same dataset item whether or not `--shuffle` reordered the run.
    index: int = 0
    evidence: list[str] = field(default_factory=list)
    adversarial_answer: str = ""

    @property
    def is_adversarial(self) -> bool:
        return self.category == ADVERSARIAL

    @property
    def evidence_ids(self) -> frozenset[str]:
        """The `dia_id`s this question's answer lives in, as the annotators recorded them.

        Split on separators because nine of the file's 2,815 references pack more than
        one id into a single string (`"D9:1 D4:4 D4:6"`, `"D8:6; D9:17"`). Splitting
        recovers those exactly; the leftovers — a bare `"D"`, a `"D:11:26"` — come back
        as ids that resolve to no turn, and the caller drops the question from the
        evidence measure rather than guessing what was meant.
        """
        return frozenset(part for raw in self.evidence
                         for part in _EVIDENCE_SPLIT.split(raw) if part)


#: `;`, `,` or whitespace inside one evidence field. See `LocomoQA.evidence_ids`.
_EVIDENCE_SPLIT = re.compile(r"[;,\s]+")


@dataclass(slots=True)
class Sample:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session]
    qa: list[LocomoQA]
    undated: int = 0

    @property
    def haystack(self) -> str:
        return "\n".join(t.text for s in self.sessions for t in s.turns)

    @property
    def dia_ids(self) -> frozenset[str]:
        """Every turn id actually ingested, which is what an evidence id must match."""
        return frozenset(t.label for s in self.sessions for t in s.turns if t.label)


def parse_when(raw: str) -> datetime | None:
    """`"1:56 pm on 8 May, 2023"` to an aware datetime, or None if it will not parse.

    None rather than an exception: one unparseable string in a 2.8 MB file should cost
    that session its timestamp and show up in the report as a counted defect, not kill
    a run that is otherwise fine.
    """
    try:
        return datetime.strptime(str(raw).strip(), WHEN_FORMAT).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _turn_text(turn: dict[str, Any], speaker: str) -> str:
    """One turn as memvara will store it: speaker-attributed, caption included.

    The speaker prefix is load-bearing. `role` carries the name too, but `recall()`
    renders an episode's *content* into the prompt and nothing else, so without the
    prefix the reader would get a wall of unattributed sentences and every "what did
    Caroline do" question would be unanswerable for a reason that has nothing to do
    with retrieval.
    """
    text = str(turn.get("text") or "").strip()
    caption = str(turn.get("blip_caption") or "").strip()
    if caption:
        text = f"{text} [shared an image: {caption}]".strip()
    return f"{speaker}: {text}"


def _session_indices(conversation: dict[str, Any]) -> list[int]:
    out = []
    for key, value in conversation.items():
        if key.startswith("session_") and isinstance(value, list):
            suffix = key[len("session_"):]
            if suffix.isdigit():
                out.append(int(suffix))
    return sorted(out)


def parse_sample(raw: dict[str, Any], *, base: datetime | None = None) -> Sample:
    conversation = raw["conversation"]
    speaker_a = str(conversation.get("speaker_a") or "A")
    speaker_b = str(conversation.get("speaker_b") or "B")
    fallback = base or datetime(2023, 1, 1, tzinfo=timezone.utc)
    sessions: list[Session] = []
    undated = 0
    for i in _session_indices(conversation):
        raw_when = str(conversation.get(f"session_{i}_date_time") or "")
        when = parse_when(raw_when)
        if when is None:
            undated += 1
            # A synthetic, monotonic stand-in keeps session order intact so ordering
            # questions are not corrupted by one bad string; the count is reported.
            when = fallback.replace(year=fallback.year + i // 12, month=i % 12 + 1)
        turns = []
        for turn in conversation[f"session_{i}"]:
            speaker = str(turn.get("speaker") or speaker_a)
            turns.append(ek.Turn(role=speaker.lower().replace(" ", "_"),
                                 text=_turn_text(turn, speaker), ts=when,
                                 # The annotators' own address for this turn, carried
                                 # through ingestion so `--score retrieval` can check
                                 # what came back against what they marked.
                                 label=str(turn.get("dia_id") or "")))
        sessions.append(Session(index=i, when=when, turns=turns, raw_when=raw_when))

    qa = []
    for index, item in enumerate(raw.get("qa", [])):
        answer = item.get("answer")
        qa.append(LocomoQA(
            question=str(item.get("question") or ""),
            # Six gold answers in the file are integers; `str` here rather than in the
            # scorer so everything downstream sees one type.
            answer="" if answer is None else str(answer),
            category=int(item.get("category", 0)),
            index=index,
            evidence=[str(e) for e in item.get("evidence") or []],
            adversarial_answer=str(item.get("adversarial_answer") or ""),
        ))
    return Sample(sample_id=str(raw.get("sample_id") or "?"), speaker_a=speaker_a,
                  speaker_b=speaker_b, sessions=sessions, qa=qa, undated=undated)


def load(path: str | Path) -> list[Sample]:
    with Path(path).open(encoding="utf-8") as fh:
        return [parse_sample(raw) for raw in json.load(fh)]


# --- the offline fixture --------------------------------------------------------
#
# In the dataset's own JSON shape, so `--dry-run` exercises the real parser — including
# the integer answer, the missing category-5 `answer`, and the image caption — rather
# than a convenient in-memory object that the real loader would never produce.

FIXTURE: list[dict[str, Any]] = [{
    "sample_id": "fixture-1",
    "conversation": {
        "speaker_a": "Ada",
        "speaker_b": "Bo",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Ada", "dia_id": "D1:1",
             "text": "I finally moved to Lisbon last month, the flat is tiny but sunny."},
            {"speaker": "Bo", "dia_id": "D1:2",
             "text": "Congratulations! Weren't you in Berlin for years?"},
            {"speaker": "Ada", "dia_id": "D1:3",
             "text": "Eight years in Berlin, yes. I start at Globex on Monday."},
            {"speaker": "Bo", "dia_id": "D1:4",
             "text": "I adopted a greyhound called Pepper.",
             "blip_caption": "a grey dog asleep on a blue sofa"},
        ],
        "session_2_date_time": "10:37 am on 27 June, 2023",
        "session_2": [
            {"speaker": "Ada", "dia_id": "D2:1",
             "text": "Globex did not work out. I joined Initech two weeks ago."},
            {"speaker": "Bo", "dia_id": "D2:2",
             "text": "Pepper has learned to open the fridge."},
            {"speaker": "Ada", "dia_id": "D2:3",
             "text": "I ran the Lisbon half marathon in 2022, before the move."},
        ],
    },
    "qa": [
        {"question": "Where does Ada live?", "answer": "Lisbon",
         "evidence": ["D1:1"], "category": 4},
        {"question": "Where did Ada work after leaving Globex?", "answer": "Initech",
         "evidence": ["D2:1"], "category": 1},
        {"question": "When did Ada run a half marathon?", "answer": 2022,
         "evidence": ["D2:3"], "category": 2},
        {"question": "What kind of pet would suit Bo's flat?",
         "answer": "A calm dog like a greyhound",
         "evidence": ["D1:4"], "category": 3},
        {"question": "What did Ada say about her sailing trip?",
         "adversarial_answer": "she enjoyed it", "evidence": ["D1:1"], "category": 5},
    ],
}]


def fixture() -> list[Sample]:
    return [parse_sample(raw) for raw in FIXTURE]


# --- the run --------------------------------------------------------------------


def build_memory(sample: Sample, budget: ek.RetrievalBudget, llm: Any = None,
                 read_k: int | None = None) -> Memvara:
    """A store per conversation, which is the unit a LOCOMO question is about.

    `read_max_episodes=k` because the library's default of 3 assumes raw turns are a
    tail on a list of extracted facts. On this dataset, with the shipped extractor
    inert (see the module docstring), they are the entire answer.

    `read_k` raises that cap for `--score retrieval`, whose curve reads deeper than the
    budget. It has to be set on the constructor rather than per call — the cap is a
    property of the retriever — and a cap left at the budget would truncate the curve
    at the budget and make every deeper column a lie.
    """
    return Memvara(
        user=sample.sample_id,
        llm=llm if llm is not None else NullLLM(),
        read_max_episodes=read_k or budget.k,
    )


def answer_one(
    mem: Memvara,
    qa: LocomoQA,
    haystack: str,
    *,
    reader: ek.Reader,
    judge: ek.Judge | None,
    ledger: ek.TokenLedger,
    budget: ek.RetrievalBudget,
    source: ek.ContextSource,
    read_stats: ek.RetrievalStats,
    stem: Callable[[str], str] | None,
    qid: str,
) -> ek.QuestionResult:
    context, ms, hits = ek.retrieve(mem, qa.question, budget, source, haystack)
    read_stats.record(ms, len(context), hits, len(haystack))
    out = reader.answer(SYSTEM, ek.build_prompt(qa.question, context))
    ledger.record("reader", out)

    result = ek.QuestionResult(
        qid=qid,
        category=CATEGORIES.get(qa.category, f"category-{qa.category}"),
        question=qa.question,
        # An adversarial item has no gold and its `adversarial_answer` is the bait, not
        # an answer — scoring against it would reward hallucinating exactly what the
        # question was designed to elicit.
        gold="" if qa.is_adversarial else qa.answer,
        prediction=out.text,
        is_abstention=qa.is_adversarial,
        did_abstain=ek.abstained(out.text),
        context_chars=len(context),
        retrieval_ms=ms,
    )
    if qa.is_adversarial:
        result.judged = result.did_abstain
        return result

    result.f1 = ek.token_f1(out.text, qa.answer, stem)
    result.bleu1 = ek.bleu1(out.text, qa.answer, stem)
    result.exact = ek.exact_match(out.text, qa.answer, stem)
    if judge is not None:
        ok, verdict = judge.judge(qa.question, qa.answer, out.text, result.category)
        ledger.record("judge", verdict)
        result.judged = ok
    return result


def run(
    samples: Sequence[Sample],
    *,
    reader: ek.Reader,
    judge: ek.Judge | None = None,
    budget: ek.RetrievalBudget | None = None,
    source: ek.ContextSource = ek.ContextSource.MEMORY,
    limit: int = 0,
    llm: Any = None,
    ledger: ek.TokenLedger | None = None,
    stem: Callable[[str], str] | None = None,
) -> tuple[list[ek.QuestionResult], ek.IngestStats, ek.RetrievalStats, ek.TokenLedger]:
    budget = budget or ek.RetrievalBudget()
    ledger = ledger or ek.TokenLedger()
    totals, read_stats, results = ek.IngestStats(), ek.RetrievalStats(), []

    for sample in samples:
        mem = build_memory(sample, budget, llm)
        # Once per conversation, not once per question: joining 590 turns for each of
        # ~200 questions is the kind of accidental O(n²) that turns a ten-minute run
        # into an hour and gets blamed on the memory layer.
        haystack = sample.haystack
        try:
            stats = ek.ingest(mem, [s.turns for s in sample.sessions])
            stats.undated_turns = sum(len(s.turns) for s in sample.sessions
                                      if not s.raw_when or parse_when(s.raw_when) is None)
            totals.merge(stats)
            for qa in sample.qa:
                if limit and len(results) >= limit:
                    break
                results.append(answer_one(
                    mem, qa, haystack, reader=reader, judge=judge, ledger=ledger,
                    budget=budget, source=source, read_stats=read_stats, stem=stem,
                    qid=f"{sample.sample_id}:{qa.index}",
                ))
        finally:
            mem.close()
        if limit and len(results) >= limit:
            break
    return results, totals, read_stats, ledger


def run_retrieval(
    samples: Sequence[Sample],
    *,
    budget: ek.RetrievalBudget | None = None,
    plan: ek.RetrievalPlan | None = None,
    limit: int = 0,
    llm: Any = None,
) -> tuple[list[ek.RetrievalScore], ek.IngestStats, ek.RetrievalStats, Counter]:
    """The same ingest and the same retrieval as `run()`, scored with no reader.

    The read path is exercised twice per question and that is deliberate:
    `ek.retrieve()` at the budget, so the context size and the latency reported are the
    ones a real integration would see, and then a deeper `search()` for the ranks. Only
    the first is charged to `RetrievalStats`.
    """
    budget = budget or ek.RetrievalBudget()
    plan = plan or ek.RetrievalPlan()
    totals, read_stats = ek.IngestStats(), ek.RetrievalStats()
    scores: list[ek.RetrievalScore] = []
    excluded: Counter = Counter()

    for sample in samples:
        mem = build_memory(sample, budget, llm, read_k=plan.depth(budget))
        haystack = sample.haystack
        labels: dict[str, str] = {}
        try:
            stats = ek.ingest(mem, [s.turns for s in sample.sessions], labels)
            stats.undated_turns = sum(len(s.turns) for s in sample.sessions
                                      if not s.raw_when or parse_when(s.raw_when) is None)
            totals.merge(stats)
            known = sample.dia_ids
            for qa in sample.qa:
                if limit and len(scores) >= limit:
                    break
                if qa.is_adversarial:
                    # No gold to look for, and its evidence turns are what the bait was
                    # built from — retrieving them is neither right nor wrong.
                    excluded["are not scored at all: category 5 has no gold answer, "
                             "and its\n    evidence turns are what the bait was built "
                             "from"] += 1
                    continue
                context, ms, hits = ek.retrieve(
                    mem, qa.question, budget, ek.ContextSource.MEMORY, haystack)
                read_stats.record(ms, len(context), hits, len(haystack))
                items, _ = ek.retrieval_pass(mem, qa.question, plan, budget, labels)
                wanted = qa.evidence_ids
                usable = bool(wanted) and wanted <= known
                if wanted and not usable:
                    excluded["are missing from the evidence table only: their evidence "
                             "ids name\n    no turn in the file, and were not repaired "
                             "by guessing"] += 1
                elif not wanted:
                    excluded["are missing from the evidence table only: the annotators "
                             "recorded\n    no evidence for them"] += 1
                scores.append(ek.score_retrieval(
                    f"{sample.sample_id}:{qa.index}",
                    CATEGORIES.get(qa.category, f"category-{qa.category}"),
                    items,
                    ek.EvidenceGold(answer=qa.answer,
                                    labels=wanted if usable else frozenset(),
                                    has_labels=usable,
                                    # Distinct turns actually in the store, not turns in
                                    # the file: two byte-identical turns dedupe to one
                                    # episode, and the denominator has to be what
                                    # retrieval could have returned.
                                    pool=len(set(labels.values()))),
                    context=context, haystack_chars=len(haystack), retrieval_ms=ms,
                    ks=plan.ks, threshold=plan.threshold, stem=plan.stem,
                    stopwords=plan.stopwords,
                ))
        finally:
            mem.close()
        if limit and len(scores) >= limit:
            break
    return scores, totals, read_stats, excluded


# --- reporting ------------------------------------------------------------------


def _pct(values: Sequence[bool | None]) -> str:
    known = [v for v in values if v is not None]
    return f"{100 * sum(known) / len(known):.1f}%" if known else "-"


def report(
    results: Sequence[ek.QuestionResult],
    ingest_stats: ek.IngestStats,
    read_stats: ek.RetrievalStats,
    ledger: ek.TokenLedger,
    *,
    reader: ek.Reader,
    judge: ek.Judge | None,
    budget: ek.RetrievalBudget,
    source: ek.ContextSource,
    stemmed: bool,
) -> str:
    grouped = ek.group_by_category(results)
    rows = []
    for category in (CATEGORIES[c] for c in ANSWERABLE):
        items = grouped.get(category, [])
        if not items:
            continue
        rows.append((
            category, len(items),
            f"{100 * ek.mean([r.f1 for r in items]):.1f}",
            f"{100 * ek.mean([r.bleu1 for r in items]):.1f}",
            _pct([r.judged for r in items]),
        ))
    answerable = [r for r in results if not r.is_abstention]
    if answerable:
        rows.append((
            "all answerable", len(answerable),
            f"{100 * ek.mean([r.f1 for r in answerable]):.1f}",
            f"{100 * ek.mean([r.bleu1 for r in answerable]):.1f}",
            _pct([r.judged for r in answerable]),
        ))

    conversations = len({r.qid.split(":")[0] for r in results})
    out = [
        "",
        f"  LOCOMO — {len(results)} questions over {conversations} "
        f"conversation{'' if conversations == 1 else 's'}",
        f"  reader={reader.name}  judge={judge.name if judge else 'none'}  "
        f"context={source.value}  k={budget.k}  max_chars={budget.max_chars}  "
        f"stem={'porter' if stemmed else 'none'}",
        "",
        ek.render_table(["category", "n", "F1", "BLEU-1", "judge"], rows) if rows
        else "  no answerable questions in this slice",
        "",
    ]

    adversarial = grouped.get(CATEGORIES[ADVERSARIAL], [])
    if adversarial:
        abstain_rate = 100 * sum(r.did_abstain for r in adversarial) / len(adversarial)
        out += [
            ek.render_table(
                ["adversarial (category 5)", "n", "abstained", "hallucinated"],
                [("questions with no answer", len(adversarial),
                  f"{abstain_rate:.1f}%", f"{100 - abstain_rate:.1f}%")],
            ),
            "",
        ]

    out += [ek.retrieval_block(ingest_stats, read_stats), "", ek.cost_block(ledger), ""]
    out.append(ek.source_caveat(source))
    banner = ek.stub_caveat(reader, judge)
    if banner:
        out += ["", banner]
    out += [
        "",
        "  Read before quoting any of this:",
        "",
        "  * One reader, one retrieval budget, one judge. A published LOCOMO score used",
        "    different ones, so putting the two side by side compares harnesses as much",
        "    as it compares memory layers.",
        "  * F1 and BLEU-1 punish a verbose but correct answer. The system prompt squeezes",
        "    the reader short to protect them, which is a property of the metric rather",
        "    than of the memory — the judge column is the one that survives paraphrase.",
        "  * Text only. 1,226 of the 5,882 turns carry an image; their BLIP captions are",
        "    ingested as text and the pixels are not.",
        "  * Category 5 is scored by abstention with the reference string rule, never",
        "    against its `adversarial_answer`, which is the bait rather than a gold.",
        "",
    ]
    return "\n".join(out)


# --- CLI ------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None,
         out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ek.add_common_arguments(parser)
    parser.add_argument("--dataset", default=ek.LOCOMO10.key, help=argparse.SUPPRESS)
    parser.add_argument("--samples", type=int, default=0,
                        help="stop after N conversations (0 = all 10)")
    args = parser.parse_args(argv)

    if args.download:
        ek.fetch(ek.LOCOMO10, args.cache, log=out)
        return 0

    if args.dry_run:
        samples = fixture()
        out("\n  --dry-run: five built-in questions, one per category, "
            + ek.dry_run_reader_note(args))
    else:
        path = args.data or ek.require(ek.LOCOMO10, args.cache)
        samples = load(path)
    if args.shuffle:
        # Questions arrive grouped by conversation and clustered by category, so
        # `--limit 40` unshuffled is forty questions from one conversation and two
        # categories. Seeded, so the slice is reproducible.
        rng = random.Random(args.shuffle)
        rng.shuffle(samples)
        for sample in samples:
            rng.shuffle(sample.qa)
    if args.samples:
        samples = samples[: args.samples]
    if (args.limit or args.samples) and not args.shuffle and not args.dry_run:
        out("\n  NOTE: this is a slice in file order, which is grouped by conversation "
            "and\n  clusters categories. Pass --shuffle SEED for a representative one.")

    budget = ek.RetrievalBudget(k=args.k, max_chars=args.max_chars,
                                include_episodes=not args.no_episodes)

    if args.score == "retrieval":
        plan = ek.build_plan(args)
        scores, ingest_stats, read_stats, excluded = run_retrieval(
            samples, budget=budget, plan=plan, limit=args.limit)
        out(ek.retrieval_report(
            scores, ingest_stats, read_stats, title="LOCOMO", plan=plan, budget=budget,
            categories=[CATEGORIES[c] for c in ANSWERABLE],
            unmeasurable=[f"{count:,} questions {reason}"
                          for reason, count in sorted(excluded.items())],
        ))
        if args.out:
            ek.write_retrieval_jsonl(args.out, scores)
            out(f"  per-question results: {args.out}\n")
        return 0

    reader = ek.build_reader(args)
    judge = ek.build_judge(args, reader)
    results, ingest_stats, read_stats, ledger = run(
        samples, reader=reader, judge=judge, budget=budget,
        source=ek.ContextSource(args.context), limit=args.limit,
        ledger=ek.build_ledger(args, reader), stem=ek.build_stemmer(args),
    )
    if getattr(reader, "dumping", False):
        # The dump phase has no answers yet, so every result is empty. Printing the
        # table would print a run that scored 0.0 on everything and looked like a
        # finding.
        out(reader.finish())
        return 0
    out(report(results, ingest_stats, read_stats, ledger, reader=reader, judge=judge,
               budget=budget, source=ek.ContextSource(args.context), stemmed=args.stem))
    if getattr(reader, "missing", 0):
        out(f"  {reader.missing:,} of {reader.calls:,} questions had no row in the "
            f"answers file; each was scored as an empty answer.\n")
    if args.out:
        ek.write_jsonl(args.out, results)
        out(f"  per-question results: {args.out}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    try:
        sys.exit(main())
    except ek.DatasetMissing as missing:
        print(f"\n{missing}")
        sys.exit(1)
