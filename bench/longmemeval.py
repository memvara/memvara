"""LongMemEval — chat-assistant long-term memory, run against memvara.

    PYTHONPATH=. python3 bench/longmemeval.py --dry-run                  # offline, no key
    PYTHONPATH=. python3 bench/longmemeval.py --download --dataset oracle
    PYTHONPATH=. python3 bench/longmemeval.py --score retrieval          # no model at all
    PYTHONPATH=. python3 bench/longmemeval.py --dataset s --reader anthropic --judge llm

## The dataset, and where it actually comes from

Three files on HuggingFace under `xiaowu0162/longmemeval-cleaned` — **public, ungated,
no token** — with sizes read from the datasets API rather than guessed:

| `--dataset` | file                          | size    | what it is                        |
|-------------|-------------------------------|---------|-----------------------------------|
| `oracle`    | `longmemeval_oracle.json`     | 15 MB   | evidence sessions only. Easy.     |
| `s`         | `longmemeval_s_cleaned.json`  | 277 MB  | ~115K tokens of haystack each     |
| `m`         | `longmemeval_m_cleaned.json`  | 2.7 GB  | ~500 sessions each                |

Nothing is vendored and nothing downloads implicitly: `--download` writes into
`$MEMVARA_BENCH_DATA` (default `~/.cache/memvara-bench`), and a run without the file fails
with the URL and a `curl` command. The oracle file is a smoke test, not a headline — it
hands the system only the sessions that contain the answer, so a score on it says
nothing about retrieval under distraction. `s` is what "LongMemEval" means in a paper.

Counted from the oracle file: 500 instances across six question types — 133
temporal-reasoning, 133 multi-session, 78 knowledge-update, 70 single-session-user, 56
single-session-assistant, 30 single-session-preference — of which 30 carry a
`question_id` ending in `_abs` and are unanswerable. Abstention is reported as its own
category, never folded into the type it was drawn from.

## Why this is not the LOCOMO pipeline

Three differences that do not abstract away, which is why there are two files:

* **The haystack is per question.** Each instance ships its own `haystack_sessions`, so
  the faithful setting is a *fresh store per question* — the default here. `--share-store`
  ingests every session once into one store, keyed by `haystack_session_ids`; it is far
  cheaper and it changes the task, because retrieval can then reach another question's
  haystack. It prints a warning saying so, and its numbers are not LongMemEval numbers.
* **The official metric is a judge, not overlap.** Gold answers are free-form phrases,
  and correctness turns on paraphrase, on "the *updated* value", on off-by-one-day
  tolerance. F1 and BLEU-1 are still computed and printed, clearly marked secondary.
* **The turns are `user` / `assistant`.** Unlike LOCOMO's two named humans, this shape
  is what memvara's `SalienceGate` and `FastExtractor` were built for, so the
  deterministic write path actually fires here and the run reports how much it extracted.

The `has_answer: true` flag on evidence turns is **deliberately ignored** — using it
would be an oracle leak dressed up as retrieval, and a test asserts the loader never
reads it.

## Scoring retrieval on its own

`--score retrieval` drops the reader and asks only whether retrieval surfaced the
evidence — see `evalkit.score_retrieval`. This file supports the strong form of that
question directly: all 500 instances carry `answer_session_ids`, 948 references in
total, every one of which names a session that is in the same instance's
`haystack_session_ids`. So "did we retrieve the session the annotators marked" is a
fact about the run, and it is reported separately from, and trusted over, the
string-based measure that runs alongside it.

`answer_session_ids` is used **only after retrieval, to grade it**. Nothing on the
ingest or the query path can see it, for the same reason `has_answer` is untouched.

The 30 `_abs` instances keep their evidence measure and lose their string measure:
their gold answer is a refusal sentence ("The information provided is not enough…"),
and looking for its words in the retrieved text would measure nothing, while the
sessions an annotator marked as establishing the absence are still a real target.

One more category defeats the string rule and is not excluded, because the number is
worth seeing. `single-session-preference`'s 30 golds are not answers, they are
meta-descriptions of what a good answer would be — "The user would prefer responses
that suggest resources specifically tailored to Adobe Premiere Pro, especially those
that delve into its advanced settings…", thirty-odd content tokens of it. No single
retrieved turn can carry 60% of that, and the measured ceiling across all 30 is a best
coverage of 0.6 on exactly one of them. The row therefore reads 0.0% on every string
column and that is the metric failing, not retrieval. The `best cov` column is in the
table so this is visible rather than deduced, and the evidence table is where that
category's actual retrieval quality is (poorly) reported.

**The file is grouped by question type.** The first 60 instances of the oracle file are
all `temporal-reasoning`, so `--limit 60` unshuffled is not a 12% sample of the
benchmark, it is the whole of its hardest category. `--shuffle SEED` fixes that, and a
slice taken without it prints a warning rather than quietly reporting a biased number.

## What a real run costs

**Measured: ~$3 to ~$9 on `claude-opus-5` for all 500 questions**, plus $1–2 for a
`--judge llm` pass. See `evalkit`'s "What a run actually costs" section, which is the
one place the procedure and the numbers live.

The reader only ever sees the retrieval budget: 3,639 characters ≈ 985 input tokens
per question, measured from a dumped slice rather than assumed. Unlike LOCOMO's, that
figure is close to the 1.2k this file used to claim — these sessions are long enough to
approach the `--max-chars` cap. The output side is the same correction as there: the
old "under 100 output tokens" predates `claude-opus-5` thinking by default and billing
thinking at the output rate, and thinking is what the bill turns on.

Ingestion is the number that will surprise you. With no `llm=` configured it is free.
With an extraction model on `--dataset s` it is 500 × ~115K tokens ≈ **57M input tokens
of extraction**, which is several hundred dollars before a single question is asked.
That is a property of the benchmark's shape rather than of memvara — every memory layer
that extracts on write pays it — but it should be a decision, not a surprise, so
`--llm` is not wired to a default here at all.

## What this does not establish

The same caveat as `bench/locomo.py`: this compares memvara against itself under three
context sources, with one reader and one judge, using judge prompts written from the
reference protocol's description rather than copied from it. It is not comparable
token-for-token with a published autograder score.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import evalkit as ek

from memvara import Memvara, NullLLM
from memvara.embed import embedder_name

QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)

SYSTEM = (
    "You answer a question about a user's earlier conversations with an assistant, "
    "using only the retrieved excerpts below the question.\n"
    "Answer concisely — a phrase or one short sentence. No preamble.\n"
    "Where a fact was later revised, answer with the most recent value.\n"
    "Use the question's date to resolve relative time expressions.\n"
    "If the excerpts do not contain the answer, say so plainly rather than guessing."
)

#: `"2023/04/10 (Mon) 23:07"`, the shape used by both `question_date` and
#: `haystack_dates`.
WHEN_FORMAT = "%Y/%m/%d (%a) %H:%M"

DATASET_ALIASES = {
    "oracle": ek.LME_ORACLE,
    "s": ek.LME_S,
    "m": ek.LME_M,
}


# --- the dataset ----------------------------------------------------------------


@dataclass(slots=True)
class Instance:
    qid: str
    question_type: str
    question: str
    answer: str
    asked_on: datetime | None
    asked_on_raw: str
    sessions: list[list[ek.Turn]]
    session_ids: list[str]
    #: The sessions an annotator marked as containing the evidence. Read only by
    #: `--score retrieval`, and only to grade a retrieval that has already happened —
    #: nothing on the ingest or query path can reach it.
    answer_session_ids: list[str] = field(default_factory=list)
    undated: int = 0

    @property
    def is_abstention(self) -> bool:
        """`_abs` on the id is the reference protocol's marker for unanswerable."""
        return self.qid.endswith("_abs")

    @property
    def category(self) -> str:
        """Abstention is its own row, never folded into the type it was drawn from."""
        return ek.ABSTENTION_TYPE if self.is_abstention else self.question_type

    @property
    def haystack(self) -> str:
        return "\n".join(t.text for s in self.sessions for t in s)


def parse_when(raw: str) -> datetime | None:
    """`"2023/04/10 (Mon) 23:07"` to an aware datetime, or None if it will not parse."""
    try:
        return datetime.strptime(str(raw).strip(), WHEN_FORMAT).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def session_label(qid: str, session_ids: Sequence[str], index: int) -> str:
    """This session's address, matching what `--share-store` deduplicates on.

    The synthesised form has to agree with `run()`'s, or a run with no
    `haystack_session_ids` would key its store on one id and grade its retrieval
    against another.
    """
    return session_ids[index] if session_ids else f"{qid}:{index}"


def parse_instance(raw: dict[str, Any], *, base: datetime | None = None) -> Instance:
    fallback = base or datetime(2023, 1, 1, tzinfo=timezone.utc)
    dates = list(raw.get("haystack_dates") or [])
    session_ids = [str(s) for s in raw.get("haystack_session_ids") or []]
    qid = str(raw.get("question_id") or "?")
    sessions: list[list[ek.Turn]] = []
    undated = 0
    for i, session in enumerate(raw.get("haystack_sessions") or []):
        when = parse_when(dates[i]) if i < len(dates) else None
        if when is None:
            undated += 1
            when = fallback
        label = session_label(qid, session_ids, i)
        turns = []
        for turn in session:
            # `has_answer` is present on evidence turns and is never read: retrieving
            # by it would be an oracle, not a memory.
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            turns.append(ek.Turn(role=str(turn.get("role") or "user").strip().lower(),
                                 text=content, ts=when, label=label))
        sessions.append(turns)
    asked_raw = str(raw.get("question_date") or "")
    return Instance(
        qid=qid,
        question_type=str(raw.get("question_type") or "unknown"),
        question=str(raw.get("question") or ""),
        answer=str(raw.get("answer") if raw.get("answer") is not None else ""),
        asked_on=parse_when(asked_raw),
        asked_on_raw=asked_raw,
        sessions=sessions,
        session_ids=session_ids,
        answer_session_ids=[str(s) for s in raw.get("answer_session_ids") or []],
        undated=undated,
    )


def load(path: str | Path, *, limit: int = 0) -> list[Instance]:
    """Read the instance file.

    Loaded whole rather than streamed. That is fine for `oracle` (15 MB) and for `s`
    (277 MB on a machine with a few gigabytes free); `m` is 2.7 GB and will need a
    streaming parser before it is usable, which is stated rather than pretended.
    """
    with Path(path).open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if limit:
        raw = raw[:limit]
    return [parse_instance(item) for item in raw]


# --- the offline fixture --------------------------------------------------------
#
# In the dataset's own JSON shape, so `--dry-run` exercises the real loader: the
# `_abs` id, the `has_answer` flag the loader must ignore, and a knowledge-update
# instance whose earlier value is still in the haystack.

FIXTURE: list[dict[str, Any]] = [
    {
        "question_id": "fx_single_user",
        "question_type": "single-session-user",
        "question": "What breed is my dog?",
        "answer": "a greyhound",
        "question_date": "2023/06/01 (Thu) 09:00",
        "haystack_session_ids": ["fx_a_1", "fx_a_2"],
        "haystack_dates": ["2023/04/10 (Mon) 17:50", "2023/05/02 (Tue) 11:20"],
        "haystack_sessions": [
            [{"role": "user", "content": "I adopted a greyhound called Pepper.",
              "has_answer": True},
             {"role": "assistant", "content": "Greyhounds are wonderfully lazy indoors."}],
            [{"role": "user", "content": "Pepper worked out how to open the fridge."},
             {"role": "assistant", "content": "Time for a childproof latch."}],
        ],
        "answer_session_ids": ["fx_a_1"],
    },
    {
        "question_id": "fx_knowledge_update",
        "question_type": "knowledge-update",
        "question": "Where do I work now?",
        "answer": "Initech",
        "question_date": "2023/07/01 (Sat) 10:00",
        "haystack_session_ids": ["fx_b_1", "fx_b_2"],
        "haystack_dates": ["2023/04/12 (Wed) 08:05", "2023/06/20 (Tue) 14:30"],
        "haystack_sessions": [
            [{"role": "user", "content": "I work at Globex as a staff engineer."},
             {"role": "assistant", "content": "Noted."}],
            [{"role": "user", "content": "I left Globex. I work at Initech now.",
              "has_answer": True},
             {"role": "assistant", "content": "Congratulations on the move."}],
        ],
        "answer_session_ids": ["fx_b_2"],
    },
    {
        "question_id": "fx_temporal_abs",
        "question_type": "temporal-reasoning",
        "question": "How long after my scuba certification did I dive in Egypt?",
        "answer": "There is no information about a scuba certification.",
        "question_date": "2023/07/02 (Sun) 12:00",
        "haystack_session_ids": ["fx_c_1"],
        "haystack_dates": ["2023/05/05 (Fri) 19:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "I booked a walking holiday in the Peak District."},
             {"role": "assistant", "content": "Pack waterproofs."}],
        ],
        "answer_session_ids": [],
    },
]


def fixture() -> list[Instance]:
    return [parse_instance(raw) for raw in FIXTURE]


# --- the run --------------------------------------------------------------------


def build_memory(user: str, budget: ek.RetrievalBudget, llm: Any = None,
                 read_k: int | None = None, embedder: Any = None,
                 w_graph: float = 0.0, w_temporal: float = 0.0,
                 reranker: Any = None, rerank_top_n: int = 0) -> Memvara:
    """One store, scoped to a user.

    `read_max_episodes=k` for the same reason as in `bench/locomo.py`: raw turns are
    capped at 3 by default because they are meant to be a tail on a fact list, and a
    conversation benchmark needs them as a first-class result. `read_k` raises the cap
    for `--score retrieval`, whose recall curve reads deeper than the budget.

    **`embedder` must be passed and must be recorded**, for the reason set out on
    `locomo.build_memory`. This file was the last one in `bench/` that did not: it had
    no `--embedder` flag and no pin, so its vector leg was whatever `default_embedder()`
    happened to return — `LocalEmbedder` on any machine with sentence-transformers
    installed, `HashingEmbedder` everywhere else. The pairing is what makes it bad
    rather than merely unpinned: LOCOMO pinned hashing and this did not, so the two
    figures quoted side by side in the README were produced by different embedders.
    `None` is still accepted and still means `default_embedder()`, because
    `build_memory(qid, budget)` is a documented two-argument call; `main()` always
    passes one.
    """
    # The episode cap rises with the reranker's window, exactly as in
    # `locomo.build_memory`: the reranker can only reorder what it was handed, so a cap
    # below `--rerank N` would silently make `--rerank 50` mean `--rerank 20` and the row
    # would be attributed to the larger window.
    #
    # `window` is resolved once and both settings read it, so the two cannot disagree.
    # Computing the cap from the caller's `rerank_top_n` while passing `rerank_top_n or
    # 20` to the reader reintroduced that gap by another route: a caller handing over a
    # reranker with `rerank_top_n=0` got a 20-candidate window over a pool capped at
    # `budget.k`, which is the same silent narrowing in the other direction. The window
    # is 0 when no reranker was given, because a stage that does not exist has no window.
    window = (rerank_top_n or 20) if reranker is not None else 0
    episodes = max(read_k or budget.k, window)
    return Memvara(user=user, llm=llm if llm is not None else NullLLM(),
                  embedder=embedder, read_max_episodes=episodes,
                  read_reranker=reranker, read_rerank_top_n=window or 20,
                  read_w_graph=w_graph, read_w_temporal=w_temporal)


def answer_one(
    mem: Any,
    item: Instance,
    *,
    reader: ek.Reader,
    judge: ek.Judge | None,
    ledger: ek.TokenLedger,
    budget: ek.RetrievalBudget,
    source: ek.ContextSource,
    read_stats: ek.RetrievalStats,
    stem: Callable[[str], str] | None,
) -> ek.QuestionResult:
    haystack = item.haystack
    context, ms, hits = ek.retrieve(mem, item.question, budget, source, haystack)
    read_stats.record(ms, len(context), hits, len(haystack))
    prompt = ek.build_prompt(item.question, context, asked_on=item.asked_on_raw or None)
    out = reader.answer(SYSTEM, prompt)
    ledger.record("reader", out)

    result = ek.QuestionResult(
        qid=item.qid,
        category=item.category,
        question=item.question,
        gold=item.answer,
        prediction=out.text,
        f1=ek.token_f1(out.text, item.answer, stem),
        bleu1=ek.bleu1(out.text, item.answer, stem),
        exact=ek.exact_match(out.text, item.answer, stem),
        is_abstention=item.is_abstention,
        did_abstain=ek.abstained(out.text, ek.ABSTENTION_MARKERS),
        context_chars=len(context),
        retrieval_ms=ms,
    )
    if judge is not None:
        ok, verdict = judge.judge(item.question, item.answer, out.text, result.category)
        ledger.record("judge", verdict)
        result.judged = ok
    elif item.is_abstention:
        # With no judge there is still one thing worth scoring without a model: whether
        # an unanswerable question was declined. Answerable accuracy stays unscored
        # rather than being faked from overlap.
        result.judged = result.did_abstain
    return result


def run(
    items: Sequence[Instance],
    *,
    reader: ek.Reader,
    judge: ek.Judge | None = None,
    budget: ek.RetrievalBudget | None = None,
    source: ek.ContextSource = ek.ContextSource.MEMORY,
    llm: Any = None,
    ledger: ek.TokenLedger | None = None,
    stem: Callable[[str], str] | None = None,
    share_store: bool = False,
    embedder: Any = None,
    w_graph: float = 0.0,
    reranker: Any = None,
    rerank_top_n: int = 0,
) -> tuple[list[ek.QuestionResult], ek.IngestStats, ek.RetrievalStats, ek.TokenLedger]:
    budget = budget or ek.RetrievalBudget()
    ledger = ledger or ek.TokenLedger()
    totals, read_stats, results = ek.IngestStats(), ek.RetrievalStats(), []

    if share_store:
        shared = build_memory("shared", budget, llm, embedder=embedder,
                              w_graph=w_graph, reranker=reranker,
                              rerank_top_n=rerank_top_n)
        # Sessions are deduplicated by their dataset id, so a session that appears in
        # several questions' haystacks is written once. Whether that actually saves
        # anything depends on how much the haystacks overlap in `longmemeval_s`, which
        # could not be checked without downloading 277 MB — the saving is a hypothesis,
        # the change to the task is not.
        seen: set[str] = set()
        try:
            for item in items:
                ids = item.session_ids or [
                    f"{item.qid}:{i}" for i in range(len(item.sessions))]
                fresh = []
                for sid, turns in zip(ids, item.sessions):
                    if sid in seen:
                        continue
                    seen.add(sid)
                    fresh.append(turns)
                totals.merge(ek.ingest(shared, fresh))
            for item in items:
                results.append(answer_one(
                    shared, item, reader=reader, judge=judge, ledger=ledger,
                    budget=budget, source=source, read_stats=read_stats, stem=stem))
        finally:
            shared.close()
        return results, totals, read_stats, ledger

    for item in items:
        mem = build_memory(item.qid, budget, llm, embedder=embedder,
                           w_graph=w_graph, reranker=reranker,
                           rerank_top_n=rerank_top_n)
        try:
            stats = ek.ingest(mem, item.sessions)
            stats.undated_turns = item.undated
            totals.merge(stats)
            results.append(answer_one(
                mem, item, reader=reader, judge=judge, ledger=ledger, budget=budget,
                source=source, read_stats=read_stats, stem=stem))
        finally:
            mem.close()
    return results, totals, read_stats, ledger


def score_one(
    mem: Any,
    item: Instance,
    *,
    budget: ek.RetrievalBudget,
    plan: ek.RetrievalPlan,
    labels: dict[str, str],
    read_stats: ek.RetrievalStats,
    excluded: Counter,
) -> ek.RetrievalScore:
    """One question's retrieval, scored with no reader. See `ek.score_retrieval`."""
    haystack = item.haystack
    context, ms, hits = ek.retrieve(
        mem, item.question, budget, ek.ContextSource.MEMORY, haystack)
    read_stats.record(ms, len(context), hits, len(haystack))
    items, _ = ek.retrieval_pass(mem, item.question, plan, budget, labels)

    wanted = frozenset(item.answer_session_ids)
    ingested = frozenset(t.label for s in item.sessions for t in s if t.label)
    usable = bool(wanted) and wanted <= ingested
    if wanted and not usable:
        excluded["are missing from the evidence table only: an answer_session_id names "
                 "no\n    session that was ingested"] += 1
    elif not wanted:
        excluded["are missing from the evidence table only: no answer_session_ids "
                 "were\n    recorded for them"] += 1
    if item.is_abstention:
        excluded["are missing from the string table only: unanswerable (_abs), whose "
                 "gold\n    is a refusal sentence rather than an answer"] += 1
    return ek.score_retrieval(
        item.qid, item.category, items,
        ek.EvidenceGold(answer=item.answer,
                        labels=wanted if usable else frozenset(),
                        has_labels=usable,
                        score_answer=not item.is_abstention,
                        # Sessions retrieval could actually have returned, which under
                        # `--share-store` is every question's sessions rather than this
                        # one's — the chance column has to follow the store it ran on.
                        pool=len(set(labels.values()))),
        context=context, haystack_chars=len(haystack), retrieval_ms=ms,
        ks=plan.ks, threshold=plan.threshold, stem=plan.stem, stopwords=plan.stopwords,
    )


def run_retrieval(
    items: Sequence[Instance],
    *,
    budget: ek.RetrievalBudget | None = None,
    plan: ek.RetrievalPlan | None = None,
    llm: Any = None,
    share_store: bool = False,
    embedder: Any = None,
    w_graph: float = 0.0,
    w_temporal: float = 0.0,
    reranker: Any = None,
    rerank_top_n: int = 0,
) -> tuple[list[ek.RetrievalScore], ek.IngestStats, ek.RetrievalStats, Counter]:
    """`run()`'s ingest and retrieval, scored with no reader and no judge."""
    budget = budget or ek.RetrievalBudget()
    plan = plan or ek.RetrievalPlan()
    totals, read_stats = ek.IngestStats(), ek.RetrievalStats()
    scores: list[ek.RetrievalScore] = []
    excluded: Counter = Counter()

    if share_store:
        shared = build_memory("shared", budget, llm, read_k=plan.depth(budget),
                              embedder=embedder, w_graph=w_graph,
                              w_temporal=w_temporal, reranker=reranker,
                              rerank_top_n=rerank_top_n)
        labels: dict[str, str] = {}
        seen: set[str] = set()
        try:
            for item in items:
                fresh = []
                for i, turns in enumerate(item.sessions):
                    sid = session_label(item.qid, item.session_ids, i)
                    if sid in seen:
                        continue
                    seen.add(sid)
                    fresh.append(turns)
                totals.merge(ek.ingest(shared, fresh, labels))
            for item in items:
                scores.append(score_one(shared, item, budget=budget, plan=plan,
                                        labels=labels, read_stats=read_stats,
                                        excluded=excluded))
        finally:
            shared.close()
        return scores, totals, read_stats, excluded

    for item in items:
        mem = build_memory(item.qid, budget, llm, read_k=plan.depth(budget),
                           w_graph=w_graph, w_temporal=w_temporal,
                           embedder=embedder, reranker=reranker,
                           rerank_top_n=rerank_top_n)
        per_item: dict[str, str] = {}
        try:
            stats = ek.ingest(mem, item.sessions, per_item)
            stats.undated_turns = item.undated
            totals.merge(stats)
            scores.append(score_one(mem, item, budget=budget, plan=plan,
                                    labels=per_item, read_stats=read_stats,
                                    excluded=excluded))
        finally:
            mem.close()
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
    dataset: str,
    share_store: bool,
) -> str:
    grouped = ek.group_by_category(results)
    rows = []
    for category in (*QUESTION_TYPES, ek.ABSTENTION_TYPE):
        items = grouped.get(category, [])
        if not items:
            continue
        rows.append((
            category, len(items), _pct([r.judged for r in items]),
            f"{100 * ek.mean([r.f1 for r in items]):.1f}",
            f"{100 * ek.mean([r.bleu1 for r in items]):.1f}",
        ))
    answerable = [r for r in results if not r.is_abstention]
    if answerable:
        rows.append((
            "all answerable", len(answerable), _pct([r.judged for r in answerable]),
            f"{100 * ek.mean([r.f1 for r in answerable]):.1f}",
            f"{100 * ek.mean([r.bleu1 for r in answerable]):.1f}",
        ))

    out = [
        "",
        f"  LongMemEval ({dataset}) — {len(results)} questions",
        f"  reader={reader.name}  judge={judge.name if judge else 'none'}  "
        f"context={source.value}  k={budget.k}  max_chars={budget.max_chars}  "
        f"store={'shared' if share_store else 'per-question'}",
        "",
        ek.render_table(["question type", "n", "judged correct", "F1", "BLEU-1"], rows)
        if rows else "  no questions in this slice",
        "",
        "  Judged accuracy is the benchmark's metric. F1 and BLEU-1 are printed because",
        "  they are free, and are secondary: gold answers are free-form phrases, so a",
        "  correct paraphrase scores badly on both.",
        "",
        ek.retrieval_block(ingest_stats, read_stats),
        "",
        ek.cost_block(ledger),
        "",
        ek.source_caveat(source),
    ]
    banner = ek.stub_caveat(reader, judge)
    if banner:
        out += ["", banner]
    if share_store:
        out += [
            "",
            "  --share-store WAS SET. Every question's haystack is in one store, so",
            "  retrieval can reach sessions belonging to other questions. That is a",
            "  different and easier-to-get-wrong task; these are not LongMemEval numbers.",
        ]
    if dataset == "oracle":
        out += [
            "",
            "  --dataset oracle hands the memory only the sessions containing the answer.",
            "  It exercises the pipeline cheaply and says nothing about retrieval under",
            "  distraction, which is the thing LongMemEval was built to measure.",
        ]
    out += [
        "",
        "  The judge prompts are written from the reference protocol's description, not",
        "  copied from it, so this is not byte-comparable with the published autograder.",
        "  Pass the official strings via LLMJudge(prompts=...) when that matters.",
        "",
    ]
    return "\n".join(out)


# --- CLI ------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None,
         out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ek.add_common_arguments(parser)
    ek.add_rerank_arguments(parser)
    parser.add_argument("--dataset", default="oracle", choices=sorted(DATASET_ALIASES),
                        help="oracle (15 MB, easy) | s (277 MB, standard) | m (2.7 GB)")
    parser.add_argument("--share-store", action="store_true",
                        help="one store for every question — cheaper, and a different "
                             "task. Not a LongMemEval result.")
    args = parser.parse_args(argv)
    spec = DATASET_ALIASES[args.dataset]

    if args.download:
        ek.fetch(spec, args.cache, log=out)
        return 0

    if args.dry_run:
        items = fixture()
        out("\n  --dry-run: three built-in instances, one of them unanswerable, "
            + ek.dry_run_reader_note(args))
    else:
        path = args.data or ek.require(spec, args.cache)
        # Only pre-slice when the order is being kept: shuffling a slice of the file
        # order would still be a slice of one question type.
        items = load(path, limit=0 if args.shuffle else args.limit)
    if args.shuffle:
        random.Random(args.shuffle).shuffle(items)
    elif args.limit and not args.dry_run:
        out("\n  NOTE: the file is grouped by question type — the first 60 instances of "
            "the\n  oracle file are all temporal-reasoning. Pass --shuffle SEED for a "
            "representative slice.")
    if args.limit:
        items = items[: args.limit]

    budget = ek.RetrievalBudget(k=args.k, max_chars=args.max_chars,
                                include_episodes=not args.no_episodes)
    # Printed unconditionally, in both scoring modes, including for the default. This
    # file had no `--embedder` at all and took `default_embedder()`, so its vector leg
    # was a property of what happened to be pip-installed and no report said which.
    embedder = ek.build_embedder(args.embedder)
    out(f"\n  --embedder {args.embedder}: {embedder_name(embedder)}")

    # Printed unconditionally too, and for the same reason: a reranker row that does not
    # say which reranker ran, over how deep a window, is not comparable to the row above
    # it. `--rerank 0` is the shipped default and prints "off" rather than nothing, so a
    # baseline row states its configuration as explicitly as a reranked one.
    #
    # The name comes from the reranker itself rather than from the flags, which is the
    # difference between recording a configuration and recording a request. Every
    # reranker sets `name` to something that identifies it exactly —
    # `cross-encoder:cross-encoder/ms-marco-MiniLM-L-6-v2`, `coverage:1` — so a run that
    # took `--rerank-model`'s default still writes down which model that was. Printing
    # the flag instead would log "default model", and the default is free to change in a
    # later release, leaving an archived log that cannot be attributed to anything.
    # `bench/locomo.py` reads `name` for this reason and the two runners must agree.
    reranker = ek.build_reranker(args)
    out(f"  --rerank {args.rerank}: "
        + (f"{args.reranker} reranker "
           f"({getattr(reranker, 'name', type(reranker).__name__)}) over the top "
           f"{args.rerank} fused candidates, cut to k afterwards"
           if reranker is not None else "off (the shipped default)"))

    if args.score == "retrieval":
        plan = ek.build_plan(args)
        scores, ingest_stats, read_stats, excluded = run_retrieval(
            items, budget=budget, plan=plan, share_store=args.share_store,
            embedder=embedder, w_graph=args.w_graph, w_temporal=args.w_temporal,
            reranker=reranker, rerank_top_n=args.rerank)
        out(ek.retrieval_report(
            scores, ingest_stats, read_stats,
            title=f"LongMemEval ({args.dataset})", plan=plan, budget=budget,
            categories=[*QUESTION_TYPES, ek.ABSTENTION_TYPE],
            unmeasurable=[f"{count:,} questions {reason}"
                          for reason, count in sorted(excluded.items())],
        ))
        if args.share_store:
            out("  --share-store WAS SET: one store for every question, so retrieval "
                "can reach\n  another question's sessions. Not a LongMemEval number.\n")
        if args.out:
            ek.write_retrieval_jsonl(args.out, scores)
            out(f"  per-question results: {args.out}\n")
        return 0

    reader = ek.build_reader(args)
    judge = ek.build_judge(args, reader)
    results, ingest_stats, read_stats, ledger = run(
        items, reader=reader, judge=judge, budget=budget,
        source=ek.ContextSource(args.context),
        ledger=ek.build_ledger(args, reader), stem=ek.build_stemmer(args),
        share_store=args.share_store, embedder=embedder, w_graph=args.w_graph,
        reranker=reranker, rerank_top_n=args.rerank,
    )
    if getattr(reader, "dumping", False):
        # No answers exist yet, so every result is empty. Printing the table would
        # print a run that scored zero on everything and looked like a finding.
        out(reader.finish())
        return 0
    note = ek.answers_note(reader)
    if note:
        out("\n" + note)
    out(report(results, ingest_stats, read_stats, ledger, reader=reader, judge=judge,
               budget=budget, source=ek.ContextSource(args.context),
               dataset=args.dataset, share_store=args.share_store))
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
