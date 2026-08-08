"""Shared machinery for the LOCOMO and LongMemEval harnesses.

    PYTHONPATH=. python3 bench/locomo.py --dry-run
    PYTHONPATH=. python3 bench/longmemeval.py --dry-run

Both benchmarks ask the same shape of question — ingest a long conversation into a
memory, retrieve against a question, hand the retrieval to a reader model, score the
answer — so dataset caching, metrics, readers, judges, cost accounting and the results
table live here. The two *pipelines* do not merge, and the runners say why in their own
docstrings: LOCOMO is two humans talking with string-overlap gold answers and one
category scored by abstention; LongMemEval is user/assistant sessions with a
per-question haystack and an LLM judge as the official metric. Forcing one loop over
both would have meant lying about one of them.

## The thing a memory benchmark gets wrong

Any of these numbers can be manufactured by giving the reader the whole transcript.
Then you have measured a language model's long-context recall and reported it as a
memory result. This harness makes that structurally impossible on the measured path:
`RetrievalBudget` requires a positive `k` *and* a positive character cap, there is no
unbounded setting, and the only way to put a full transcript in front of the reader is
`ContextSource.FULL`, which every report labels a **reader ceiling** rather than a
score. Three context sources give the triple that makes a number interpretable:

    NONE    the reader answering from its own priors, with no memory at all — the floor
    MEMORY  what engram retrieved under the stated budget — the measurement
    FULL    the whole haystack in the prompt — the ceiling, and not a memory result

A memory layer is worth something to the extent MEMORY sits above NONE while costing a
fraction of FULL. Every run reports the mean context size and its share of the haystack
so a reader can check the budget was real.

## What this controls for, and what it does not

Controlled: both context sources see the same reader, the same prompt, the same
retrieval budget and the same scorer. Retrieval cost — wall clock, characters, and the
write path's model calls — is reported next to accuracy, because the architectural claim
is about the write path and an eval that prints only F1 hides it.

**Not** controlled: reader nondeterminism (the current models reject `temperature`, so
there is no seed to pin — re-runs will differ, and the runners print how many questions
were asked so a difference can be judged against sample size); ingestion granularity
(engram receives a whole session per `add()`); and, most importantly, this is engram
measured against *itself* under different context sources. It is not a head-to-head
against another memory layer. Comparing to a published LOCOMO or LongMemEval number
from a paper compares two harnesses as much as two systems — the reader model, the
retrieval budget, the judge and the prompt all differ. Report the configuration or the
number means nothing.

## Metrics

LOCOMO's reference scorer normalises (lowercase, strip punctuation, drop `a|an|the|and`,
collapse whitespace), applies a Porter stemmer, and computes token-level F1. The
stemmer is `nltk`, which this repository does not depend on, so `token_f1` takes an
optional `stem` callable and `porter_stemmer()` loads nltk lazily for a caller who wants
byte-comparability with published numbers. Unstemmed F1 runs a little lower on
morphological variants; the runners print which mode was used.

BLEU-1 here is clipped unigram precision times the standard brevity penalty, computed
directly rather than via `sacrebleu`, so it needs no dependency and no tokenizer
version to match.

## Cost

`TokenLedger` bills reader and judge calls separately, from the `usage` the provider
returns rather than from an estimate. Prices are per million tokens as published on
2026-06-24 and will drift; a model with no entry is counted and reported as *unpriced*
rather than silently valued at zero.
"""

from __future__ import annotations

import json
import os
import re
import string
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

# --- dataset acquisition --------------------------------------------------------
#
# Nothing large is vendored into the repository and nothing downloads implicitly. A
# runner that cannot find its dataset fails with the exact command that fetches it, and
# `--download` is the only code path that writes one to disk.


@dataclass(frozen=True)
class DatasetSpec:
    """Where a benchmark's data actually lives, and how big it actually is.

    `size_bytes` was read from the host on 2026-08-09 (a GitHub raw `Content-Length`
    and the HuggingFace datasets API), not estimated — the point of the field is that a
    user knows what they are about to download before they start.
    """

    key: str
    filename: str
    url: str
    size_bytes: int
    licence: str
    note: str = ""

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1_000_000


LOCOMO10 = DatasetSpec(
    key="locomo10",
    filename="locomo10.json",
    url="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    size_bytes=2_805_274,
    licence="see LICENSE.txt in snap-research/locomo — no click-through, no auth",
    note="10 conversations, 5,882 turns, 1,986 QA items (1,540 answerable + 446 adversarial)",
)

#: The three LongMemEval configurations. `oracle` carries only the evidence sessions,
#: which makes it an easy setting and a cheap smoke test rather than a headline result;
#: `s` is the one papers mean by "LongMemEval_S". `m` is a 2.7 GB file and is here for
#: completeness — ingesting it is a multi-day job at this harness's granularity.
LME_ORACLE = DatasetSpec(
    key="longmemeval_oracle",
    filename="longmemeval_oracle.json",
    url=("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
         "resolve/main/longmemeval_oracle.json"),
    size_bytes=15_388_478,
    licence="public HuggingFace dataset — no gate, no token required",
    note="500 instances, evidence sessions only (1-6 per question). Easy setting.",
)
LME_S = DatasetSpec(
    key="longmemeval_s",
    filename="longmemeval_s_cleaned.json",
    url=("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
         "resolve/main/longmemeval_s_cleaned.json"),
    size_bytes=277_383_467,
    licence="public HuggingFace dataset — no gate, no token required",
    note="500 instances, ~115K tokens of haystack each. The standard setting.",
)
LME_M = DatasetSpec(
    key="longmemeval_m",
    filename="longmemeval_m_cleaned.json",
    url=("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
         "resolve/main/longmemeval_m_cleaned.json"),
    size_bytes=2_737_100_077,
    licence="public HuggingFace dataset — no gate, no token required",
    note="500 instances, ~500 sessions each. 2.7 GB.",
)

DATASETS: dict[str, DatasetSpec] = {
    d.key: d for d in (LOCOMO10, LME_ORACLE, LME_S, LME_M)
}


class DatasetMissing(FileNotFoundError):
    """The dataset is not on disk and this harness will not fetch it behind your back."""


def cache_root(root: str | os.PathLike[str] | None = None) -> Path:
    """Where datasets are cached. `ENGRAM_BENCH_DATA` overrides the default."""
    if root is not None:
        return Path(root)
    env = os.environ.get("ENGRAM_BENCH_DATA")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "engram-bench"


def local_path(spec: DatasetSpec, root: str | os.PathLike[str] | None = None) -> Path:
    return cache_root(root) / spec.filename


def require(spec: DatasetSpec, root: str | os.PathLike[str] | None = None) -> Path:
    """The dataset's path, or a `DatasetMissing` that says exactly how to get it."""
    path = local_path(spec, root)
    if path.exists():
        return path
    raise DatasetMissing(
        f"{spec.key} is not at {path}.\n\n"
        f"  {spec.note}\n"
        f"  {spec.size_mb:.1f} MB — {spec.licence}\n\n"
        "Fetch it with either of:\n"
        f"    python3 bench/{'locomo' if spec is LOCOMO10 else 'longmemeval'}.py "
        f"--download --dataset {spec.key}\n"
        f"    mkdir -p {path.parent} && curl -L -o {path} \\\n        {spec.url}\n"
    )


def fetch(
    spec: DatasetSpec,
    root: str | os.PathLike[str] | None = None,
    *,
    #: Resolved at call time rather than bound as a default, so a caller (or a test)
    #: that replaces `urllib.request.urlopen` is actually honoured.
    opener: Callable[[str], Any] | None = None,
    chunk: int = 1 << 20,
    log: Callable[[str], None] = print,
) -> Path:
    """Download `spec` into the cache. Only ever called from an explicit `--download`.

    Writes to a `.part` file and renames on success, because a benchmark that reads a
    truncated 277 MB JSON and reports whatever parsed is worse than one that fails.

    `urllib` verifies TLS against the interpreter's CA store, and a Python installed
    from python.org on macOS ships without one until `Install Certificates.command` has
    been run — this environment is one of those, so the failure is re-raised pointing at
    `curl`, which uses the system store and works.
    """
    open_url = opener if opener is not None else urllib.request.urlopen
    path = local_path(spec, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    log(f"  fetching {spec.key} ({spec.size_mb:.1f} MB) from {spec.url}")
    got = 0
    try:
        with open_url(spec.url) as response, partial.open("wb") as out:
            while True:
                block = response.read(chunk)
                if not block:
                    break
                out.write(block)
                got += len(block)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download {spec.key}: {exc}\n\n"
            "If that is a TLS certificate error, this interpreter has no CA bundle. "
            "Use curl, which does:\n"
            f"    mkdir -p {path.parent} && curl -L -o {path} \\\n        {spec.url}\n"
        ) from exc
    partial.replace(path)
    log(f"  wrote {got:,} bytes to {path}")
    return path


# --- metrics --------------------------------------------------------------------

_PUNCT = set(string.punctuation)
#: The reference scorer drops `and` alongside the articles. Kept exactly, including the
#: oddity, because changing it would make these numbers incomparable to published ones
#: for no gain.
_ARTICLES = re.compile(r"\b(a|an|the|and)\b")


def normalize_answer(text: Any) -> str:
    """Lowercase, strip punctuation, drop `a|an|the|and`, collapse whitespace.

    A port of the LOCOMO reference `normalize_answer`. Accepts non-strings because six
    LOCOMO gold answers are integers (`2022`), and a scorer that crashed on those would
    silently be scoring 1,980 questions while claiming 1,986.
    """
    lowered = str(text).lower()
    stripped = "".join(ch for ch in lowered if ch not in _PUNCT)
    return " ".join(_ARTICLES.sub(" ", stripped).split())


def tokenize(text: Any, stem: Callable[[str], str] | None = None) -> list[str]:
    words = normalize_answer(text).split()
    return [stem(w) for w in words] if stem is not None else words


def porter_stemmer() -> Callable[[str], str]:
    """The reference scorer's stemmer, loaded lazily.

    nltk is not a dependency of this repository and is not needed to run anything here.
    It exists as an option because published LOCOMO F1 is stemmed, and an unstemmed
    number is close but not the same number.
    """
    try:
        from nltk.stem.porter import PorterStemmer
    except ImportError as exc:  # pragma: no cover - exercised with an injected fake
        raise ImportError(
            "--stem needs nltk, which engram does not depend on: pip install 'nltk>=3.8'. "
            "Without it F1 is computed unstemmed, which is slightly lower on "
            "morphological variants and is what the harness reports by default."
        ) from exc
    return PorterStemmer().stem


def token_f1(prediction: Any, gold: Any, stem: Callable[[str], str] | None = None) -> float:
    """Token-level F1 over normalized tokens, counting multiplicity.

    `Counter` intersection rather than set intersection, so a prediction that repeats a
    gold token five times gets credit once — the reference behaviour, and the difference
    between scoring an answer and scoring a stutter.
    """
    pred, want = tokenize(prediction, stem), tokenize(gold, stem)
    if not pred or not want:
        # Two empties are a match; one empty is a miss. Both are reachable: an
        # abstaining reader produces the first, a gold of `None` the second.
        return float(not pred and not want)
    overlap = sum((Counter(pred) & Counter(want)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(want)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: Any, gold: Any, stem: Callable[[str], str] | None = None) -> bool:
    """Order-insensitive token-set equality, as the reference computes it."""
    return set(tokenize(prediction, stem)) == set(tokenize(gold, stem))


def bleu1(prediction: Any, gold: Any, stem: Callable[[str], str] | None = None) -> float:
    """Clipped unigram precision times the brevity penalty.

    The brevity penalty is what stops the degenerate strategy: without it, answering
    with the single most likely gold word scores 1.0 on every question.
    """
    pred, want = tokenize(prediction, stem), tokenize(gold, stem)
    if not pred or not want:
        return float(not pred and not want)
    clipped = sum((Counter(pred) & Counter(want)).values())
    precision = clipped / len(pred)
    if precision == 0.0:
        return 0.0
    if len(pred) >= len(want):
        return precision
    # exp(1 - r/c), computed without importing math for one call.
    return precision * (2.718281828459045 ** (1 - len(want) / len(pred)))


#: The LOCOMO reference scores an adversarial (category 5) answer correct when the
#: output contains one of exactly these two phrases. Kept verbatim so the number is
#: comparable; the runners instruct the reader to use the first one, which makes the
#: rule fair rather than a vocabulary lottery.
LOCOMO_ABSTENTION = ("no information available", "not mentioned")

#: A wider net, for LongMemEval, whose official abstention metric is a judge rather
#: than a string match. Only used when no judge is configured.
ABSTENTION_MARKERS = LOCOMO_ABSTENTION + (
    "i don't know", "i do not know", "cannot be determined", "no information",
    "isn't mentioned", "was not mentioned", "never mentioned", "not stated",
)


def abstained(text: str, markers: Sequence[str] = LOCOMO_ABSTENTION) -> bool:
    lowered = str(text).lower()
    return any(m in lowered for m in markers)


# --- readers --------------------------------------------------------------------


@dataclass(slots=True)
class Answer:
    """One reader or judge response, with the tokens it actually cost.

    Token counts come from the provider's `usage`, never from an estimate — an eval
    that reports a guessed cost is reporting a guess.
    """

    text: str
    model: str = "stub"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class Reader(Protocol):
    name: str
    #: True for a reader that consults no model. The report refuses to present a stub
    #: run as a measurement, so this has to be declared rather than inferred.
    is_stub: bool

    def answer(self, system: str, prompt: str) -> Answer: ...


#: Separates the question from the retrieved block inside a reader prompt. A constant
#: rather than a convention because `StubReader` has to find the boundary again, and a
#: stub that mistook the question for its own context would answer by quoting it.
CONTEXT_MARKER = "\n--- retrieved memory ---\n"


class StubReader:
    """A deterministic, offline stand-in for a language model.

    It picks the line of the retrieved context with the highest unigram overlap with
    the question and returns it, and abstains when there is no context. That is enough
    to drive ingest → retrieve → answer → score → report end to end with no key, which
    is what the tests need and what a user should run before spending money.

    **Its scores are not a measurement of anything.** It cannot reason, cannot combine
    two sessions, and cannot read a date. `is_stub` is how the report knows to say so.
    """

    name = "stub"
    is_stub = True

    def __init__(self, abstain_with: str = "No information available.",
                 min_overlap: int = 1) -> None:
        self.abstain_with = abstain_with
        #: How many words a line must share with the question before the stub will
        #: repeat it. One is the honest floor for a bag-of-words matcher; raising it
        #: makes the stub abstain more, which is a knob for exercising that path, not
        #: an improvement in judgement.
        self.min_overlap = min_overlap
        self.calls = 0

    def answer(self, system: str, prompt: str) -> Answer:
        self.calls += 1
        question, context = _split_prompt(prompt)
        wanted = set(tokenize(question))
        best, best_score = "", 0
        for line in context.splitlines():
            body = line[2:] if line.startswith("- ") else line
            score = len(wanted & set(tokenize(body)))
            # Strictly greater, so the first line wins ties and the output is a
            # function of the input rather than of dict ordering.
            if score > best_score:
                best, best_score = body.strip(), score
        if best_score < self.min_overlap:
            return Answer(text=self.abstain_with, model="stub")
        return Answer(text=best, model="stub")


def _split_prompt(prompt: str) -> tuple[str, str]:
    """Recover (question, context) from a prompt built by `build_prompt`.

    Split on the marker being *present*, not on the remainder being non-empty: an empty
    context is the floor configuration, and treating it as "no marker" would hand the
    stub the question as its own context and let it answer by quoting the question.
    """
    if CONTEXT_MARKER not in prompt:
        return prompt, prompt
    head, _, rest = prompt.partition(CONTEXT_MARKER)
    return head, rest


def build_prompt(question: str, context: str, *, asked_on: str | None = None) -> str:
    """The reader's user turn: the question first, then whatever retrieval produced.

    Question first is deliberate. It is the one part of the prompt that is never
    attacker-controlled, and putting it ahead of stored text means a memory that tries
    to restate the task is arguing with an instruction the model has already read.
    """
    when = f"\nToday's date: {asked_on}" if asked_on else ""
    return f"Question: {question}{when}{CONTEXT_MARKER}{context}"


def _text_of_anthropic(response: Any) -> str:
    for block in getattr(response, "content", None) or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                return str(block.get("text") or "")
        elif getattr(block, "type", None) == "text":
            return str(getattr(block, "text", "") or "")
    return ""


def _attr(obj: Any, name: str) -> Any:
    """Attribute or key, whichever this object has. SDK objects are the first, test
    doubles and raw JSON the second."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _usage(response: Any, model: str, text: str) -> Answer:
    """Token counts from whatever shape the provider reports them in.

    The two providers spell the same four numbers differently — Anthropic's
    `input_tokens` / `output_tokens` versus OpenAI's `prompt_tokens` /
    `completion_tokens` — and reading only the first spelling is how a cost report
    silently comes back as $0.00 for a run that spent real money.
    """
    usage = _attr(response, "usage")

    def field(*names: str) -> int:
        for name in names:
            value = _attr(usage, name) if usage is not None else None
            if value:
                return int(value)
        return 0

    details = _attr(usage, "prompt_tokens_details") if usage is not None else None
    return Answer(
        text=text,
        model=model,
        input_tokens=field("input_tokens", "prompt_tokens"),
        output_tokens=field("output_tokens", "completion_tokens"),
        cache_read_tokens=(field("cache_read_input_tokens")
                           or int(_attr(details, "cached_tokens") or 0)),
        cache_write_tokens=field("cache_creation_input_tokens"),
    )


class AnthropicReader:
    """Reader backed by the Messages API.

    No sampling parameters are sent. `temperature`, `top_p` and `top_k` are rejected
    outright by the current models, so there is no seed to pin here and no pretending
    otherwise: two runs of this harness against the same data will differ, and the
    report prints the sample size so the difference can be judged against it.

    `effort` defaults to `low` because the task is extractive question answering over a
    short retrieved block, and paying for deep reasoning would measure the reader rather
    than the memory. Raise it deliberately, and say so when you quote a number.
    """

    is_stub = False

    def __init__(
        self,
        model: str = "claude-opus-5",
        client: Any = None,
        effort: str = "low",
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.name = f"anthropic/{model}"
        self._client = client if client is not None else self._default_client()

    @staticmethod
    def _default_client() -> Any:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicReader needs the `anthropic` package: "
                "pip install 'engram[anthropic]'. Pass client= to inject one, or run "
                "with --reader stub to exercise the harness offline."
            ) from exc
        return anthropic.Anthropic()

    def answer(self, system: str, prompt: str) -> Answer:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort},
        )
        return _usage(response, self.model, _text_of_anthropic(response).strip())


class OpenAIReader:
    """Reader backed by Chat Completions, mirroring `engram/llm/openai.py`'s transport."""

    is_stub = False

    def __init__(
        self,
        model: str = "gpt-4.1",
        client: Any = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = f"openai/{model}"
        self._client = client if client is not None else self._default_client()

    @staticmethod
    def _default_client() -> Any:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAIReader needs the `openai` package: pip install 'engram[openai]'. "
                "Pass client= to inject one, or run with --reader stub to exercise the "
                "harness offline."
            ) from exc
        return openai.OpenAI()

    def answer(self, system: str, prompt: str) -> Answer:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        choices = _attr(response, "choices") or []
        message = _attr(choices[0], "message") if choices else None
        text = str(_attr(message, "content") or "") if message is not None else ""
        return _usage(response, self.model, text.strip())


# --- judges ---------------------------------------------------------------------


class Judge(Protocol):
    name: str

    def judge(self, question: str, gold: str, hypothesis: str,
              question_type: str) -> tuple[bool, Answer]: ...


class ContainmentJudge:
    """An offline judge: the gold tokens appear in the hypothesis, or F1 clears a bar.

    Deterministic and free, and materially stricter than a model on paraphrase — a
    correct answer worded differently scores wrong. It exists so `--dry-run` produces a
    judged accuracy at all, and so the tests can pin the loop. Do not quote its number
    as a LongMemEval result; the official protocol is an LLM judge.
    """

    name = "containment"

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def judge(self, question: str, gold: str, hypothesis: str,
              question_type: str) -> tuple[bool, Answer]:
        if question_type == ABSTENTION_TYPE:
            return abstained(hypothesis, ABSTENTION_MARKERS), Answer("", model="stub")
        want, got = tokenize(gold), set(tokenize(hypothesis))
        contained = bool(want) and all(w in got for w in want)
        ok = contained or token_f1(hypothesis, gold) >= self.threshold
        return ok, Answer("", model="stub")


#: The pseudo question type used for an unanswerable question, in both benchmarks.
ABSTENTION_TYPE = "abstention"

#: What the judge is asked, per LongMemEval question type. These are **our** prompts,
#: written from the reference protocol's description rather than copied from it, so a
#: run here is not byte-identical to the published autograder. Pass `prompts=` with the
#: official strings when a number has to be directly comparable.
JUDGE_PROMPTS: dict[str, str] = {
    "default": (
        "Answer yes if the response contains the correct answer, and no otherwise. "
        "Wording may differ from the reference answer; only the content matters."
    ),
    "temporal-reasoning": (
        "Answer yes if the response contains the correct answer, and no otherwise. "
        "Do not penalise an off-by-one difference in a number of days."
    ),
    "knowledge-update": (
        "Answer yes if the response gives the updated, most recent value. An answer "
        "that reports a superseded earlier value is no."
    ),
    "single-session-preference": (
        "Answer yes if the response recalls and applies the user's stated preference "
        "correctly, and no otherwise."
    ),
    ABSTENTION_TYPE: (
        "The question is unanswerable from the conversation. Answer yes if the "
        "response says so rather than inventing an answer, and no otherwise."
    ),
}


class LLMJudge:
    """The official-shaped judge: a model answers yes/no, per question type.

    Parsing is `"yes" in response.lower()`, which is what the reference does. It is
    crude — a judge that begins "yes and no" scores correct — and it is kept because
    changing it would silently shift every number away from the published protocol.
    """

    def __init__(self, reader: Reader, prompts: dict[str, str] | None = None) -> None:
        self.reader = reader
        self.prompts = dict(JUDGE_PROMPTS if prompts is None else prompts)
        self.name = f"llm-judge/{reader.name}"

    def judge(self, question: str, gold: str, hypothesis: str,
              question_type: str) -> tuple[bool, Answer]:
        instruction = self.prompts.get(question_type, self.prompts["default"])
        system = (
            "You grade another model's answer to a question about a conversation. "
            "Reply with exactly one word, yes or no.\n" + instruction
        )
        prompt = (
            f"Question: {question}\n"
            f"Reference answer: {gold}\n"
            f"Response to grade: {hypothesis}"
        )
        out = self.reader.answer(system, prompt)
        return "yes" in out.text.lower(), out


# --- cost -----------------------------------------------------------------------


@dataclass(frozen=True)
class Price:
    """Dollars per million tokens, plus the cache multipliers the API bills at."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25


#: Published list prices, per million tokens, as of 2026-06-24. These drift; a run that
#: matters should re-check them. A model missing from this table is *not* free — it is
#: reported as unpriced, and its tokens still appear in the token totals.
PRICES: dict[str, Price] = {
    "claude-fable-5": Price(10.00, 50.00),
    "claude-opus-5": Price(5.00, 25.00),
    "claude-opus-4-8": Price(5.00, 25.00),
    "claude-opus-4-7": Price(5.00, 25.00),
    "claude-sonnet-5": Price(3.00, 15.00),
    "claude-sonnet-4-6": Price(3.00, 15.00),
    "claude-haiku-4-5": Price(1.00, 5.00),
    "stub": Price(0.0, 0.0),
}
PRICES_AS_OF = "2026-06-24"


@dataclass
class _Tally:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, answer: Answer) -> None:
        self.calls += 1
        self.input_tokens += answer.input_tokens
        self.output_tokens += answer.output_tokens
        self.cache_read_tokens += answer.cache_read_tokens
        self.cache_write_tokens += answer.cache_write_tokens

    @property
    def tokens(self) -> int:
        return (self.input_tokens + self.output_tokens
                + self.cache_read_tokens + self.cache_write_tokens)


class TokenLedger:
    """What a run cost, split by role and by model.

    Roles are separated because "the reader cost $4 and the judge cost $9" is a
    different finding from "the run cost $13", and the second one hides that the
    grading was the expensive part.
    """

    def __init__(self, prices: dict[str, Price] | None = None) -> None:
        self.prices = dict(PRICES if prices is None else prices)
        self.by_role: dict[str, dict[str, _Tally]] = {}

    def record(self, role: str, answer: Answer) -> None:
        self.by_role.setdefault(role, {}).setdefault(answer.model, _Tally()).add(answer)

    def override(self, model: str, price: Price) -> None:
        self.prices[model] = price

    def cost(self) -> tuple[float, list[str]]:
        """Total dollars for everything priced, and the models that are not priced."""
        total, unpriced = 0.0, []
        for models in self.by_role.values():
            for model, tally in models.items():
                price = self.prices.get(model)
                if price is None:
                    if model not in unpriced:
                        unpriced.append(model)
                    continue
                total += (
                    tally.input_tokens * price.input_per_mtok
                    + tally.output_tokens * price.output_per_mtok
                    + tally.cache_read_tokens
                    * price.input_per_mtok * price.cache_read_multiplier
                    + tally.cache_write_tokens
                    * price.input_per_mtok * price.cache_write_multiplier
                ) / 1_000_000
        return total, unpriced

    def rows(self) -> list[tuple[str, str, str, str, str]]:
        out = []
        for role in sorted(self.by_role):
            for model in sorted(self.by_role[role]):
                tally = self.by_role[role][model]
                price = self.prices.get(model)
                if price is None:
                    dollars = "unpriced"
                else:
                    sub = TokenLedger(self.prices)
                    sub.by_role = {role: {model: tally}}
                    dollars = f"${sub.cost()[0]:.4f}"
                out.append((role, model, f"{tally.calls:,}",
                            f"{tally.input_tokens:,} / {tally.output_tokens:,}", dollars))
        return out


# --- retrieval budget -----------------------------------------------------------


class ContextSource(str, Enum):
    """Where the reader's context comes from. See the module docstring."""

    NONE = "none"
    MEMORY = "memory"
    FULL = "full"


@dataclass(frozen=True)
class RetrievalBudget:
    """The cap on what retrieval may put in front of the reader.

    Both fields are required to be positive and there is deliberately no unbounded
    setting: that absence is the harness's structural guarantee that a MEMORY run
    cannot become a long-context run by accident. `FULL` exists for the ceiling and is
    labelled as such wherever it is reported.
    """

    k: int = 12
    max_chars: int = 4000
    include_episodes: bool = True
    #: The FULL ceiling still needs a stop, or a 2.7 GB haystack becomes one request.
    full_max_chars: int = 200_000

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be at least 1, got {self.k}: an unbounded "
                             "retrieval budget turns this into a long-context benchmark")
        if self.max_chars < 1:
            raise ValueError(f"max_chars must be at least 1, got {self.max_chars}: an "
                             "unbounded context turns this into a long-context benchmark")


def clip(text: str, limit: int) -> str:
    """Truncate to `limit` characters, visibly."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


# --- ingestion and retrieval ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class Turn:
    """One conversation turn, with the timestamp it happened at.

    The timestamp is load-bearing rather than cosmetic: both benchmarks ask temporal
    questions, and engram's whole proposition is two time axes. Ingesting a dated
    transcript with `utcnow()` on every turn would throw away the axis under test.
    """

    role: str
    text: str
    ts: datetime


@dataclass
class IngestStats:
    """What writing the haystack cost, which is half of what this eval exists to show."""

    turns: int = 0
    sessions: int = 0
    episodes: int = 0
    added: int = 0
    reinforced: int = 0
    retired: int = 0
    skipped: int = 0
    unextracted: int = 0
    llm_calls: int = 0
    wall_ms: float = 0.0
    haystack_chars: int = 0
    undated_turns: int = 0

    def merge(self, other: "IngestStats") -> "IngestStats":
        for name in vars(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self


def ingest(mem: Any, sessions: Iterable[Sequence[Turn]]) -> IngestStats:
    """Write a conversation into a memory, one `add()` per session.

    Per session rather than per turn because that is how an agent loop with a session
    boundary actually calls it, and because engram batches extraction — charging it
    per turn would inflate the model-call count this benchmark is meant to report
    honestly. `bench/compare.py` reports the equal-granularity figure for the same
    reason.
    """
    stats = IngestStats()
    start = time.perf_counter()
    for turns in sessions:
        turns = list(turns)
        if not turns:
            continue
        stats.sessions += 1
        stats.turns += len(turns)
        stats.haystack_chars += sum(len(t.text) for t in turns)
        receipt = mem.add([
            {"role": t.role, "content": t.text, "ts": t.ts} for t in turns
        ])
        stats.episodes += len(receipt.episode_ids)
        stats.added += len(receipt.added)
        stats.reinforced += len(receipt.reinforced)
        stats.retired += len(receipt.invalidated)
        stats.skipped += receipt.skipped
        stats.unextracted += receipt.unextracted
        stats.llm_calls += receipt.llm_calls
    stats.wall_ms = (time.perf_counter() - start) * 1000
    return stats


@dataclass
class RetrievalStats:
    """Latency and size of the read path, per question."""

    ms: list[float] = field(default_factory=list)
    chars: list[int] = field(default_factory=list)
    results: list[int] = field(default_factory=list)
    #: The haystack this question could have been answered from. Kept per question,
    #: because the ratio of the two is the anti-stuffing evidence and a ratio of two
    #: run-wide totals would be the wrong number on any dataset where questions do not
    #: all share one haystack — which is exactly LongMemEval's shape.
    haystack: list[int] = field(default_factory=list)

    def record(self, ms: float, chars: int, results: int, haystack: int = 0) -> None:
        self.ms.append(ms)
        self.chars.append(chars)
        self.results.append(results)
        self.haystack.append(haystack)

    def share(self) -> float:
        """Mean of (context chars / haystack chars), per question.

        Above 1.0 means the reader saw more characters than the haystack holds, which
        happens only on a fixture small enough that `recall()`'s framing dominates. On
        real data it is the number that says the budget was real.
        """
        ratios = [c / h for c, h in zip(self.chars, self.haystack) if h]
        return mean(ratios)

    def __len__(self) -> int:
        return len(self.ms)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. Explicit because `statistics.quantiles` needs n >= 2."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def retrieve(
    mem: Any,
    question: str,
    budget: RetrievalBudget,
    source: ContextSource,
    haystack: str,
) -> tuple[str, float, int]:
    """Build the reader's context. Returns (context, milliseconds, result count).

    The MEMORY path goes through `recall()` rather than `search()` on purpose: it is the
    call a real integration makes, so the measured latency is the real end-to-end read
    cost including rendering, and the context carries engram's own prompt framing —
    which is part of what is being evaluated.
    """
    if source is ContextSource.NONE:
        return "", 0.0, 0
    if source is ContextSource.FULL:
        return clip(haystack, budget.full_max_chars), 0.0, 0
    start = time.perf_counter()
    context = mem.recall(question, k=budget.k, include_episodes=budget.include_episodes)
    elapsed = (time.perf_counter() - start) * 1000
    context = clip(context, budget.max_chars)
    # `recall()` renders one "- " line per result under a header, so counting them is
    # how this reports a result count without paying for a second retrieval. The suite
    # pins that format against `search()` so a change to it fails there rather than
    # quietly corrupting every count in every report.
    return context, elapsed, sum(1 for line in context.splitlines() if line.startswith("- "))


# --- results and aggregation ----------------------------------------------------


@dataclass(slots=True)
class QuestionResult:
    """One scored question. Everything a report or a post-hoc audit needs."""

    qid: str
    category: str
    question: str
    gold: str
    prediction: str
    f1: float = 0.0
    bleu1: float = 0.0
    exact: bool = False
    judged: bool | None = None
    is_abstention: bool = False
    did_abstain: bool = False
    context_chars: int = 0
    retrieval_ms: float = 0.0


def group_by_category(results: Sequence[QuestionResult]) -> dict[str, list[QuestionResult]]:
    grouped: dict[str, list[QuestionResult]] = {}
    for r in results:
        grouped.setdefault(r.category, []).append(r)
    return grouped


# --- reporting ------------------------------------------------------------------


def render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """A left-first, right-rest table matching `bench/mem0_real.py`'s output."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(row: Sequence[str]) -> str:
        first = f"  {row[0]:<{widths[0]}}"
        rest = "".join(f"  {c:>{widths[i + 1]}}" for i, c in enumerate(row[1:]))
        return first + rest

    out = [line([str(h) for h in headers]), line(["-" * w for w in widths])]
    out += [line(row) for row in cells]
    return "\n".join(out)


def cost_block(ledger: TokenLedger) -> str:
    """The dollars-and-tokens section, or a note that nothing was spent."""
    rows = ledger.rows()
    if not rows:
        return "  no model calls\n"
    total, unpriced = ledger.cost()
    out = [render_table(["role", "model", "calls", "in / out tokens", "cost"], rows)]
    out.append(f"\n  total: ${total:.4f}   (list prices as of {PRICES_AS_OF})")
    if unpriced:
        out.append(f"  UNPRICED, and therefore missing from that total: "
                   f"{', '.join(unpriced)}")
    return "\n".join(out) + "\n"


def retrieval_block(ingest_stats: IngestStats, read: RetrievalStats) -> str:
    """Engram's own cost. Reported next to accuracy, not instead of it.

    The write-path model-call count is the number this architecture exists to drive to
    zero, so an eval that printed only F1 would hide the thing engram is actually good
    at — and would also hide it getting worse.
    """
    rows = [
        ("turns ingested", f"{ingest_stats.turns:,}"),
        ("sessions (one add() each)", f"{ingest_stats.sessions:,}"),
        ("haystack characters", f"{ingest_stats.haystack_chars:,}"),
        ("claims added / reinforced / retired",
         f"{ingest_stats.added:,} / {ingest_stats.reinforced:,} / {ingest_stats.retired:,}"),
        ("turns carrying no durable fact", f"{ingest_stats.skipped:,}"),
        ("turns that reached extraction and yielded nothing",
         f"{ingest_stats.unextracted:,}"),
        ("LLM calls on the write path", f"{ingest_stats.llm_calls:,}"),
        ("ingest wall clock", f"{ingest_stats.wall_ms / 1000:.1f} s"),
        ("retrieval p50 / p95",
         f"{percentile(read.ms, 0.5):.1f} / {percentile(read.ms, 0.95):.1f} ms"),
        ("context chars, mean / p95",
         f"{mean(read.chars):.0f} / {percentile(read.chars, 0.95):.0f}"),
        ("results per question, mean", f"{mean(read.results):.1f}"),
        ("context as a share of the question's haystack", f"{read.share():.1%}"),
    ]
    if ingest_stats.undated_turns:
        rows.append(("turns whose timestamp would not parse",
                     f"{ingest_stats.undated_turns:,}"))
    return render_table(["engram cost", "measured"], rows)


def source_caveat(source: ContextSource) -> str:
    """One line saying what the run under this context source does and does not mean."""
    return {
        ContextSource.NONE:
            "  CONTEXT=none — the reader's own priors with no memory at all. This is the "
            "floor,\n  not a result. Anything MEMORY scores below this is worse than "
            "nothing.",
        ContextSource.MEMORY:
            "  CONTEXT=memory — the measured configuration. The reader saw only what "
            "engram\n  retrieved, under the budget printed above.",
        ContextSource.FULL:
            "  CONTEXT=full — the whole haystack in the prompt. This is a READER "
            "CEILING and\n  is NOT a memory result: it measures long-context recall, "
            "which is the failure\n  mode this harness exists to avoid reporting as "
            "memory quality.",
    }[source]


def stub_caveat(reader: Reader, judge: Judge | None) -> str:
    """The banner a run with no model must carry, wherever it is quoted."""
    lines = []
    if getattr(reader, "is_stub", False):
        lines.append(
            "  THE READER IS A STUB. It picks the retrieved line with the most words in\n"
            "  common with the question. It cannot reason, cannot combine sessions and\n"
            "  cannot read a date, and it never abstains while anything was retrieved —\n"
            "  so an abstention row reading 0% is the stub, not a finding. Every accuracy\n"
            "  number here is a pipeline smoke test. Re-run with --reader anthropic."
        )
    if isinstance(judge, ContainmentJudge):
        lines.append(
            "  THE JUDGE IS A STRING MATCH, not the LLM autograder the published "
            "protocol\n  uses. It marks a correctly-paraphrased answer wrong."
        )
    return "\n".join(lines)


def write_jsonl(path: str | os.PathLike[str], results: Sequence[QuestionResult]) -> None:
    """Per-question output, so a run can be audited rather than trusted."""
    with Path(path).open("w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps({
                "question_id": r.qid, "category": r.category, "question": r.question,
                "gold": r.gold, "hypothesis": r.prediction, "f1": round(r.f1, 4),
                "bleu1": round(r.bleu1, 4), "exact_match": r.exact, "judged": r.judged,
                "is_abstention": r.is_abstention, "abstained": r.did_abstain,
                "context_chars": r.context_chars,
                "retrieval_ms": round(r.retrieval_ms, 3),
            }, ensure_ascii=False) + "\n")


# --- shared CLI pieces ----------------------------------------------------------


def add_common_arguments(parser: Any) -> None:
    """The flags both runners share, so they cannot drift apart."""
    parser.add_argument("--dry-run", action="store_true",
                        help="run the built-in fixture with a stub reader; no key, no network")
    parser.add_argument("--data", default=None, help="path to the dataset JSON")
    parser.add_argument("--cache", default=None,
                        help="dataset cache directory (default $ENGRAM_BENCH_DATA or "
                             "~/.cache/engram-bench)")
    parser.add_argument("--download", action="store_true",
                        help="fetch the dataset into the cache and exit")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N questions (0 = all). File order, which is "
                             "grouped — see --shuffle")
    parser.add_argument("--shuffle", type=int, default=0, metavar="SEED",
                        help="shuffle questions with this seed before --limit. Both "
                             "files are grouped (LongMemEval by question type, LOCOMO "
                             "by conversation), so an unshuffled slice is a biased "
                             "sample of one category")
    parser.add_argument("--reader", default="stub",
                        choices=["stub", "anthropic", "openai"])
    parser.add_argument("--model", default=None, help="reader model id")
    parser.add_argument("--effort", default="low",
                        help="Anthropic reader effort (low|medium|high|xhigh|max)")
    parser.add_argument("--judge", default="none",
                        choices=["none", "containment", "llm"])
    parser.add_argument("--judge-model", default=None,
                        help="model for --judge llm (defaults to the reader's)")
    parser.add_argument("--context", default="memory",
                        choices=[s.value for s in ContextSource],
                        help="memory (measured) | none (floor) | full (reader ceiling)")
    parser.add_argument("--k", type=int, default=12, help="retrieval budget, results")
    parser.add_argument("--max-chars", type=int, default=4000,
                        help="retrieval budget, characters of context")
    parser.add_argument("--no-episodes", action="store_true",
                        help="retrieve extracted claims only, no raw turns")
    parser.add_argument("--stem", action="store_true",
                        help="Porter-stem before scoring, matching the reference F1 "
                             "(needs nltk)")
    parser.add_argument("--price-in", type=float, default=None,
                        help="override input price, dollars per million tokens")
    parser.add_argument("--price-out", type=float, default=None,
                        help="override output price, dollars per million tokens")
    parser.add_argument("--out", default=None, help="write per-question JSONL here")


def build_reader(args: Any, *, default_model: str = "claude-opus-5") -> Reader:
    if args.dry_run or args.reader == "stub":
        return StubReader()
    if args.reader == "anthropic":
        return AnthropicReader(model=args.model or default_model, effort=args.effort)
    return OpenAIReader(model=args.model or "gpt-4.1")


def build_judge(args: Any, reader: Reader) -> Judge | None:
    if args.judge == "none":
        # A dry run gets the offline judge whether or not it was asked for: the point of
        # `--dry-run` is to exercise every stage before money is spent, and leaving the
        # judged path dark would defeat it.
        return ContainmentJudge() if args.dry_run else None
    if args.judge == "containment" or args.dry_run:
        return ContainmentJudge()
    if getattr(reader, "is_stub", False):
        # A stub grading a stub produces a number with no relationship to correctness,
        # and it would be printed in the same table as a real one.
        raise SystemExit(
            "--judge llm needs a real reader: pass --reader anthropic or --reader "
            "openai, or use --judge containment for the offline string-match judge."
        )
    if args.judge_model and args.reader == "anthropic":
        return LLMJudge(AnthropicReader(model=args.judge_model, effort=args.effort))
    return LLMJudge(reader)


def build_ledger(args: Any, reader: Reader) -> TokenLedger:
    ledger = TokenLedger()
    if args.price_in is not None and args.price_out is not None:
        model = getattr(reader, "model", "stub")
        ledger.override(model, Price(args.price_in, args.price_out))
    return ledger


def build_stemmer(args: Any) -> Callable[[str], str] | None:
    return porter_stemmer() if args.stem else None
