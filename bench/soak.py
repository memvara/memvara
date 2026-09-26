"""A long, seeded run of memvara, watched for the failures that raise no error.

Run:  PYTHONPATH=. python3 bench/soak.py [--turns 10000] [--seed 0] [--out FILE] [--history DIR]

`memvara/telemetry.py` lists the ways this library can get worse without raising: a
vocabulary of predicates that keeps growing, restated facts that stop ranking higher,
single-valued slots that fill with rows, salience outranking relevance, a gate that drops
one script's turns, a retraction that closes nothing, and a redaction policy that stops
matching. None of them shows up in one call. They show up over thousands of calls, which
is what this script makes: a seeded workload of user turns, direct writes and reads,
driven through the real library with the hashing embedder and no model, and a detector
for each failure mode that reads what the run recorded.

The design of the adversarial suite fixes when each detector fails, in its section "Phase 4"
of `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`.
`docs/claude/testing.md` explains how the run works and how each detector is shown to fire.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evalkit  # noqa: E402
import perf_budget  # noqa: E402

from memvara import Memvara, NullLLM  # noqa: E402
from memvara.redact import EPISODE, PatternRedactor  # noqa: E402
from memvara.schema import BUILTIN_PREDICATES  # noqa: E402
from memvara.select import PLAIN_READ  # noqa: E402
from memvara.telemetry import (  # noqa: E402
    CONSOLIDATE_MERGED,
    GATE_DROP,
    GATE_PASS,
    REDACT_CHANGED,
    REDACT_INSPECTED,
    RETRIEVAL_OBSERVATION_RANK_CORR,
    RETRIEVAL_QUALITY_FACTOR,
    WRITE_RECONCILE,
    WRITE_RETRACTION,
    WRITE_TURNS,
    MemoryRecorder,
)

# --- the workload -------------------------------------------------------------------------

#: Every predicate the workload writes, and whether it holds one value at a time. This is
#: the workload's own intent, read by the detectors, rather than the registry's opinion,
#: so a registry that loses a predicate's cardinality is caught instead of believed.
VOCABULARY: dict[str, bool] = {
    "name": True, "lives_in": True, "works_at": True, "working_on": True,
    "likes": False, "prefers": False,
}

#: A soak shorter than this spends most of itself introducing its cast.
MIN_TURNS = 100

#: How much simulated time the turns are spread over. Three half-lives of a fast-moving
#: predicate (seven days each), so a fast-moving fact still carries a recency signal at
#: the start of the span.
SPAN = timedelta(days=21)

#: The syllables every made-up Latin word is built from.
SYLLABLES = ("ta", "lo", "vi", "ren", "mar", "zo", "quin", "el", "dri", "kan", "sol",
             "bex", "nu", "or", "phi", "gal", "tor", "wen", "ix", "ul", "vas", "pem",
             "rud", "yo")

#: Sentences in other scripts, each naming a made-up place built from that script's
#: syllables, so that turns rarely repeat word for word. All of them state where the
#: speaker lives, which the gate must treat as a fact in every script.
SCRIPTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "cyrillic": ("Я живу в городе {name}.", ("ка", "ло", "ви", "рен", "мар", "зо", "ни", "та")),
    "greek": ("Μένω στην πόλη {name}.", ("κα", "λο", "βι", "ρεν", "μαρ", "ζο", "νι", "τα")),
    "han": ("我住在{name}市。", ("山", "川", "林", "海", "石", "竹", "松", "云")),
    "kana": ("私は{name}に住んでいます。", ("か", "ろ", "み", "れ", "な", "そ", "ひ", "た")),
    "hangul": ("나는 {name}에 살고 있습니다.", ("가", "로", "미", "레", "나", "소", "히", "타")),
    "arabic": ("أعيش في مدينة {name}.", ("كا", "لو", "مي", "ري", "نا", "سو", "هي", "تا")),
    "devanagari": ("मैं {name} शहर में रहता हूँ।", ("का", "लो", "मी", "रे", "ना", "सो", "ही", "ता")),
}

#: Turns that carry no fact. The gate should drop most of them.
FILLER = ("ok thanks", "sounds good", "great, thanks", "cool", "hmm",
          "what time is it?", "can you check that again?")

#: What an agent's recall hook would search with on a prompt about the user.
RECALL_PROMPTS = ("what do I like?", "remind me what I told you about my work",
                  "any notes on my preferences?")

#: The query of a panel read. Every panel claim renders as "<subject> working on <value>",
#: so all of them match it equally and only freshness and salience can order them.
PANEL_QUERY = "who is working on what"

#: Standing preferences in the form an agent writes them: whole sentences. Each contains
#: "with a ", which one of its near-duplicates drops. They are long because the hashing
#: embedder's merge threshold is 0.97: with it, a one-word change to a shorter sentence
#: lands below the threshold and is never merged. Each near-duplicate of these three
#: measured 0.978 or more.
PREFERENCES = (
    "tests written before the code they cover, with a failing run shown first and the "
    "passing run shown after it",
    "reviews that quote the exact line they are about, with a suggested change and the "
    "reason for it written out in full",
    "meetings kept to thirty minutes or less, with a written agenda sent the day before "
    "and notes shared the same afternoon",
)

#: Spellings of each predicate an agent might write: the canonical name and the aliases
#: memvara's built-in vocabulary declares for it. A healthy registry folds them all.
ALIASES: dict[str, tuple[str, ...]] = {
    spec.name: (spec.name, *spec.aliases)
    for spec in BUILTIN_PREDICATES if spec.name in VOCABULARY
}

#: The share of each kind of turn once the introductions are over.
MIX: tuple[tuple[str, float], ...] = (
    ("latin", 0.22), ("filler", 0.05), ("non_latin", 0.06), ("personal_data", 0.03),
    ("person", 0.22), ("panel_write", 0.12), ("preference", 0.03), ("probe", 0.14),
    ("panel_read", 0.10), ("recall", 0.03),
)

#: What a Latin fact turn says, as shares of those turns.
LATIN: tuple[tuple[str, float], ...] = (
    ("move", 0.12), ("job", 0.06), ("favourite", 0.40), ("one_off", 0.27),
    ("retract", 0.10), ("name", 0.05),
)

PEOPLE, CITIES, COMPANIES, PANEL, FAVOURITES = 16, 12, 8, 8, 4


def variant(sentence: str, which: int) -> str:
    """A near-duplicate of `sentence` that the hashing embedder places within the merge
    threshold of it: one word added at the end, or the article in "with a " dropped."""
    if which == 0:
        return sentence + " always"
    return sentence.replace("with a ", "with ", 1)


@dataclass(frozen=True)
class SoakConfig:
    """How long a soak is and which seed it draws from."""

    turns: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.turns < MIN_TURNS:
            raise ValueError(f"a soak needs at least {MIN_TURNS} turns, not {self.turns}: "
                             "its first turns introduce everything it later measures")

    @property
    def span(self) -> timedelta:
        return SPAN

    @property
    def per_day(self) -> int:
        """Turns per simulated day, which is how often consolidation runs."""
        return max(1, round(self.turns / SPAN.days))


@dataclass(frozen=True)
class Turn:
    """One operation of the workload, at position `index` of the run.

    `kind` is `say` (a user turn for `add()`), `remember` (a direct write), `probe` (a
    search whose right first answer is `gold`), `panel` (a search over the panel) or
    `recall`. `fact` marks a user turn that states something, `planted` one that carries
    personal data the redactor must remove, and `script` the script it is written in.
    """

    index: int
    kind: str
    text: str = ""
    subject: str = ""
    predicate: str = ""
    obj: str = ""
    polarity: int = 1
    script: str = "latin"
    fact: bool = False
    planted: bool = False
    gold: tuple[str, str, str] | None = None


class _Words:
    """Made-up words, never the same one twice in a run."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used: set[str] = set()

    def new(self, syllables: int = 3) -> str:
        while True:
            for _ in range(20):
                word = "".join(self.rng.choice(SYLLABLES) for _ in range(syllables))
                if word not in self.used:
                    self.used.add(word)
                    return word
            syllables += 1


class Workload:
    """The turns of one soak, drawn from `config.seed`. Iterating twice gives the same turns.

    The first turns introduce everything the detectors later measure, in a fixed order:
    every person's city, every panel value, the user's name, city, employer and favourite
    likes, one like and its retraction, one turn carrying personal data, and a preference
    followed by its near-duplicate. After that each turn draws its kind from `MIX`.
    """

    def __init__(self, config: SoakConfig) -> None:
        self.config = config

    # The two seams a test replaces to inject a fault into the data rather than the code.

    def pii_text(self, index: int) -> str:
        """A turn carrying a made-up email address and phone number, both in formats
        `PatternRedactor`'s docstring says it removes."""
        return (f"You can reach me at user{index}@example.test or on "
                f"(555) 555-01{index % 100:02d}.")

    def retraction_text(self, item: str) -> str:
        """A user taking back a like, in the form the fast path reads as a retraction."""
        return f"I no longer like {item}."

    def __iter__(self) -> Iterator[Turn]:
        return _World(self).turns()


class _World:
    """What the workload has said so far, which is what decides its next turn."""

    def __init__(self, workload: Workload) -> None:
        self.workload = workload
        self.rng = random.Random(workload.config.seed)
        self.words = _Words(self.rng)
        cap = str.capitalize
        self.people = [cap(self.words.new()) for _ in range(PEOPLE)]
        self.cities = [cap(self.words.new()) for _ in range(CITIES)]
        self.companies = [cap(self.words.new()) for _ in range(COMPANIES)]
        self.panel = [cap(self.words.new()) for _ in range(PANEL)]
        #: The first half of the panel restates one value; the second half changes it.
        self.steady, self.churning = self.panel[:PANEL // 2], self.panel[PANEL // 2:]
        self.working_on = {subject: self.words.new() for subject in self.panel}
        self.city = {person: self.rng.choice(self.cities) for person in self.people}
        self.employer = {person: self.rng.choice(self.companies) for person in self.people}
        self.heavy = {person: self.words.new() for person in self.people}
        self.name = cap(self.words.new())
        self.favourites = [self.words.new() for _ in range(FAVOURITES)]
        #: The user's one-off likes not yet taken back, oldest first.
        self.one_offs: list[str] = []
        self.index = 0

    def _draw(self, table: tuple[tuple[str, float], ...]) -> str:
        roll, total = self.rng.random(), 0.0
        for name, share in table:
            total += share
            if roll < total:
                return name
        return table[-1][0]

    def _spell(self, predicate: str) -> str:
        return self.rng.choice(ALIASES[predicate])

    def _turn(self, kind: str, **fields: Any) -> Turn:
        turn = Turn(self.index, kind, **fields)
        self.index += 1
        return turn

    def _say(self, text: str, *, fact: bool = True, planted: bool = False,
             script: str = "latin") -> Turn:
        return self._turn("say", text=text, fact=fact, planted=planted, script=script)

    def _remember(self, subject: str, predicate: str, obj: str,
                  polarity: int = 1) -> Turn:
        return self._turn("remember", subject=subject, predicate=self._spell(predicate),
                          obj=obj, polarity=polarity)

    def introductions(self) -> Iterator[Turn]:
        for person in self.people:
            yield self._remember(person, "lives_in", self.city[person])
        for subject in self.panel:
            yield self._remember(subject, "working_on", self.working_on[subject])
        yield self._say(f"My name is {self.name}.")
        yield self._say(f"I live in {self.rng.choice(self.cities)}.")
        yield self._say(f"I work at {self.rng.choice(self.companies)}.")
        for favourite in self.favourites:
            yield self._say(f"I like {favourite}.")
        once = self.words.new()
        yield self._say(f"I like {once}.")
        yield self._say(self.workload.retraction_text(once))
        yield self._say(self.workload.pii_text(self.index), planted=True)
        yield self._remember("user", "prefers", PREFERENCES[0])
        yield self._remember("user", "prefers", variant(PREFERENCES[0], 0))

    def latin(self) -> Turn:
        kind = self._draw(LATIN)
        if kind == "move":
            city = self.rng.choice(self.cities)
            return self._say(self.rng.choice((f"I moved to {city}.", f"I live in {city}.",
                                              f"I just moved to {city}.")))
        if kind == "job":
            return self._say(f"I work at {self.rng.choice(self.companies)}.")
        if kind == "one_off":
            item = self.words.new()
            self.one_offs.append(item)
            return self._say(f"I like {item}.")
        if kind == "retract" and self.one_offs:
            item = self.one_offs.pop(self.rng.randrange(len(self.one_offs)))
            return self._say(self.workload.retraction_text(item))
        if kind == "name":
            return self._say(f"My name is {self.name}.")
        return self._say(f"I like {self.rng.choice(self.favourites)}.")

    def non_latin(self) -> Turn:
        script = self.rng.choice(sorted(SCRIPTS))
        template, syllables = SCRIPTS[script]
        name = "".join(self.rng.choice(syllables) for _ in range(3))
        return self._say(template.format(name=name), script=script)

    def person(self) -> Turn:
        person, roll = self.rng.choice(self.people), self.rng.random()
        if roll < 0.25:
            self.city[person] = self.rng.choice(
                [c for c in self.cities if c != self.city[person]])
            return self._remember(person, "lives_in", self.city[person])
        if roll < 0.35:
            self.employer[person] = self.rng.choice(
                [c for c in self.companies if c != self.employer[person]])
            return self._remember(person, "works_at", self.employer[person])
        if roll < 0.85:
            return self._remember(person, "likes", self.heavy[person])
        return self._remember(person, "likes", self.words.new())

    def panel_write(self) -> Turn:
        if self.rng.random() < 0.6:
            subject = self.rng.choice(self.steady)
        else:
            subject = self.rng.choice(self.churning)
            self.working_on[subject] = self.words.new()
        return self._remember(subject, "working_on", self.working_on[subject])

    def preference(self) -> Turn:
        sentence = self.rng.choice(PREFERENCES)
        if self.rng.random() < 0.4:
            sentence = variant(sentence, self.rng.randrange(2))
        return self._remember("user", "prefers", sentence)

    def probe(self) -> Turn:
        person = self.rng.choice(self.people)
        return self._turn("probe", text=f"tell me where {person} lives",
                          gold=(person, "lives_in", self.city[person]))

    def turns(self) -> Iterator[Turn]:
        # The introductions are far shorter than MIN_TURNS, so they always fit.
        yield from self.introductions()
        while self.index < self.workload.config.turns:
            kind = self._draw(MIX)
            if kind == "latin":
                yield self.latin()
            elif kind == "filler":
                yield self._say(self.rng.choice(FILLER), fact=False)
            elif kind == "non_latin":
                yield self.non_latin()
            elif kind == "personal_data":
                yield self._say(self.workload.pii_text(self.index), planted=True)
            elif kind == "person":
                yield self.person()
            elif kind == "panel_write":
                yield self.panel_write()
            elif kind == "preference":
                yield self.preference()
            elif kind == "probe":
                yield self.probe()
            elif kind == "panel_read":
                yield self._turn("panel", text=PANEL_QUERY)
            else:
                yield self._turn("recall", text=self.rng.choice(RECALL_PROMPTS))


# --- running a soak -----------------------------------------------------------------------

#: The user every write and read of a soak belongs to. One user, at the user level, keeps
#: the soak clear of the open bugs about narrower scopes (see tests/harness/known_bugs.py).
USER = "soak"

#: Bytes in one SQLite page: the smallest growth a store file can show, and so the floor
#: an increase in store growth must pass, as 2 ms is for a timing.
PAGE_BYTES = 4096


class SoakRecorder(MemoryRecorder):
    """`MemoryRecorder`, plus the rank correlations emitted while `capture` is a list.

    The soak needs the correlations of its panel reads apart from those of every other
    search, and a recorder cannot tell which search emitted a value. The run points
    `capture` at a list around each panel read, which costs nothing per value.
    """

    def __init__(self) -> None:
        super().__init__()
        self.capture: list[float] | None = None

    def gauge(self, name: str, value: float, /, **tags: str) -> None:
        super().gauge(name, value, **tags)
        if self.capture is not None and name == RETRIEVAL_OBSERVATION_RANK_CORR:
            self.capture.append(float(value))


@dataclass
class Observations:
    """What one soak recorded, which is everything the detectors read.

    `crowded` lists the single-valued slots seen holding more than one live claim at any
    check, each with the most it held. `gate` maps a script to `[reached, passed]` over
    the fact-carrying turns that reached the gate. `planted` holds, for each turn carrying
    planted personal data, its simulated day and whether the redactor changed it.
    `store_bytes` is None for a store kept in memory.
    """

    config: SoakConfig
    recorder: SoakRecorder
    predicates: set[str]
    crowded: list[tuple[str, str, int]]
    panel_correlations: list[float]
    probes: int
    probe_hits: int
    gate: dict[str, list[int]]
    planted: list[tuple[int, bool]]
    unplanted_changed: int
    store_bytes: int | None
    elapsed_s: float
    #: When the run started, in ISO 8601 and UTC; the order history is read in.
    started: str = ""


def run(config: SoakConfig, path: Path | None, *,
        options: Mapping[str, Any] | None = None) -> Observations:
    """Drive a new store through `config.turns` turns of the workload, and observe it.

    `path` is where the store is created, and must not exist yet; None keeps the store in
    memory, which observes everything except store growth. `options` replace the keyword
    arguments `Memvara` is built with, which is how a test gives a run a configuration
    fault: another `registry`, a ranking weight such as `read_w_salience`, or
    `redactor=None`.

    The turns are spread over `config.span` of simulated time ending when the run starts,
    so each turn's valid time is in the past. Transaction time is the wall clock, because
    that is what `add()` records, and for that reason consolidation also runs at the wall
    clock: a pass only sees claims recorded at or before its own instant.
    """
    if path is not None and Path(path).exists():
        raise FileExistsError(f"{path} already exists; a soak starts from an empty store, "
                              "so that its growth and its slots hold only what it wrote")
    rec = SoakRecorder()
    settings: dict[str, Any] = {
        "embedder": evalkit.build_embedder("hashing"), "llm": NullLLM(), "user": USER,
        "telemetry": rec, "redactor": PatternRedactor(), **PLAIN_READ}
    settings.update(options or {})
    mem = Memvara(str(path) if path is not None else ":memory:", **settings)
    observed = Observations(config=config, recorder=rec, predicates=set(), crowded=[],
                            panel_correlations=[], probes=0, probe_hits=0, gate={},
                            planted=[], unplanted_changed=0, store_bytes=None,
                            elapsed_s=0.0)
    crowded: dict[str, tuple[str, str, int]] = {}
    started = datetime.now(timezone.utc).replace(microsecond=0)
    observed.started = started.isoformat()
    origin = started - config.span
    step = config.span / config.turns
    began = time.perf_counter()
    try:
        for turn in Workload(config):
            _execute(mem, rec, turn, origin + step * turn.index, observed)
            if (turn.index + 1) % config.per_day == 0:
                mem.consolidate()
                _count_slots(mem, crowded)
        _count_slots(mem, crowded)
        observed.predicates = {claim.predicate for claim in
                               mem.get_all(states=("live", "ended", "retired"))}
    finally:
        mem.close()
    observed.elapsed_s = time.perf_counter() - began
    observed.crowded = sorted(crowded.values())
    if path is not None:
        observed.store_bytes = store_bytes(Path(path))
    return observed


def _counts(rec: MemoryRecorder) -> tuple[int, int, int]:
    """Gate passes, gate drops and redacted turns so far."""
    return (rec.total(GATE_PASS), rec.total(GATE_DROP),
            rec.total(REDACT_CHANGED, field=EPISODE))


def _execute(mem: Memvara, rec: SoakRecorder, turn: Turn, at: datetime,
             observed: Observations) -> None:
    """Perform one turn at simulated time `at`, and note what the detectors need."""
    if turn.kind == "say":
        before = _counts(rec)
        mem.add(turn.text, ts=at)
        passed, dropped, changed = (a - b for a, b in zip(_counts(rec), before))
        if turn.fact and passed + dropped:
            seen = observed.gate.setdefault(turn.script, [0, 0])
            seen[0] += 1
            seen[1] += passed
        if turn.planted:
            observed.planted.append((turn.index // observed.config.per_day, changed > 0))
        elif changed:
            observed.unplanted_changed += 1
    elif turn.kind == "remember":
        mem.remember(turn.subject, turn.predicate, turn.obj, polarity=turn.polarity,
                     valid_from=at)
    elif turn.kind == "probe":
        results = mem.search(turn.text, k=10, **PLAIN_READ)
        observed.probes += 1
        if results:
            top = results[0].claim
            observed.probe_hits += (top.subject, top.predicate, top.object) == turn.gold
    elif turn.kind == "panel":
        rec.capture = observed.panel_correlations
        try:
            mem.search(turn.text, k=PANEL, **PLAIN_READ)
        finally:
            rec.capture = None
    else:
        mem.recall(turn.text, **PLAIN_READ)


def _count_slots(mem: Memvara, crowded: dict[str, tuple[str, str, int]]) -> None:
    """Note every single-valued slot now holding more than one live claim.

    Single-valued by the workload's `VOCABULARY`, not by the registry, so a registry that
    has lost a predicate's cardinality is caught rather than consulted.
    """
    live: dict[str, list[Any]] = {}
    for claim in mem.get_all():
        if VOCABULARY.get(claim.predicate):
            live.setdefault(claim.fact_key, []).append(claim)
    for key, claims in live.items():
        if len(claims) > 1 and len(claims) > crowded.get(key, ("", "", 0))[2]:
            crowded[key] = (claims[0].subject, claims[0].predicate, len(claims))


def store_bytes(path: Path) -> int:
    """The size on disk of the store at `path`: the database and every file beside it
    whose name starts with the database's, such as its write-ahead log and vector file."""
    return sum(p.stat().st_size for p in path.parent.iterdir()
               if p.name.startswith(path.name) and p.is_file())


# --- the detectors ------------------------------------------------------------------------

OK, FAIL, TRACKED = "ok", "fail", "tracked"

#: The design's thresholds, one per detector.
PREDICATE_HEADROOM = 1.1
RELEVANT_FIRST = 0.95
SCRIPT_RATIO = 0.8
REDACTION_RATIO = 0.99


@dataclass(frozen=True)
class Finding:
    """One detector's verdict on one soak.

    `status` is `ok`, `fail`, or `tracked` for a detector that reports without failing.
    `value` is what was measured and `threshold` what it was held against. `flagged` names
    what the finding singles out: predicates outside the vocabulary, crowded slots, or
    scripts the gate treats worse than Latin.
    """

    detector: str
    status: str
    value: float | None
    threshold: float | None
    detail: str
    flagged: tuple[str, ...] = ()


def predicate_explosion(obs: Observations) -> Finding:
    """More distinct predicates than 1.1 times the vocabulary the workload writes."""
    name, limit = "predicate explosion", PREDICATE_HEADROOM * len(VOCABULARY)
    count = len(obs.predicates)
    if not count:
        return Finding(name, FAIL, None, limit, "not measured: the store holds no claim, "
                       "so no predicate was counted")
    extra = tuple(sorted(obs.predicates - set(VOCABULARY)))
    detail = f"{count} distinct predicates against a vocabulary of {len(VOCABULARY)}"
    if extra:
        detail += f"; outside the vocabulary: {', '.join(extra)}"
    return Finding(name, FAIL if count > limit else OK, float(count), limit, detail, extra)


def recency_refresh(obs: Observations) -> Finding:
    """The median rank correlation of the panel reads, which must stay above zero."""
    name, values = "recency refresh", obs.panel_correlations
    if not values:
        return Finding(name, FAIL, None, 0.0, "not measured: no panel read produced a "
                       "rank correlation, so whether restated facts rank higher was not "
                       "checked")
    median = statistics.median(values)
    detail = (f"median rank correlation {median:+.3f} over {len(values)} panel reads: "
              f"facts restated more often {'rank' if median > 0 else 'do not rank'} "
              "higher")
    return Finding(name, FAIL if median <= 0 else OK, median, 0.0, detail)


def flip_flop(obs: Observations) -> Finding:
    """A single-valued slot holding more than one live claim, or no merge at all."""
    name, rec = "flip-flop row growth", obs.recorder
    if CONSOLIDATE_MERGED not in rec.names():
        return Finding(name, FAIL, None, 1.0, "not measured: no consolidation pass ran, "
                       "so no slot was counted")
    merged = rec.total(CONSOLIDATE_MERGED)
    worst = max((count for _, _, count in obs.crowded), default=1)
    flagged = tuple(f"{subject} {predicate}" for subject, predicate, _ in obs.crowded)
    problems = []
    if obs.crowded:
        problems.append(f"{len(obs.crowded)} single-valued slots held more than one live "
                        f"claim, at most {worst}")
    if not merged:
        problems.append("consolidation merged nothing, though the workload plants "
                        "near-duplicates for it")
    detail = "; ".join(problems) or (
        f"every single-valued slot held at most one live claim, and consolidation merged "
        f"{merged} near-duplicates")
    return Finding(name, FAIL if problems else OK, float(worst), 1.0, detail, flagged)


def salience_over_relevance(obs: Observations) -> Finding:
    """How often a probe's relevant claim ranked first, which must be 95% or more."""
    name = "salience over relevance"
    if not obs.probes:
        return Finding(name, FAIL, None, RELEVANT_FIRST, "not measured: no probe ran")
    rate = obs.probe_hits / obs.probes
    quality = obs.recorder.values(RETRIEVAL_QUALITY_FACTOR)
    above = sum(1 for value in quality if value > 1.0)
    detail = (f"the relevant claim ranked first in {obs.probe_hits} of {obs.probes} "
              f"probes; {above} of {len(quality)} results had a quality factor above 1.0")
    return Finding(name, FAIL if rate < RELEVANT_FIRST else OK, rate, RELEVANT_FIRST,
                   detail)


def script_bias(obs: Observations) -> Finding:
    """Each script's gate pass rate against the Latin rate. Tracked, never failed: the
    gate's vocabulary is English by design (docs/LIMITATIONS.md)."""
    name = "script bias in the gate"
    rates = {script: passed / reached for script, (reached, passed) in obs.gate.items()
             if reached}
    latin = rates.get("latin")
    if not latin:
        return Finding(name, TRACKED, None, SCRIPT_RATIO, "not measured: no Latin fact "
                       "reached the gate, so there is no rate to compare with")
    ratios = {script: rate / latin for script, rate in rates.items() if script != "latin"}
    flagged = tuple(sorted(s for s, ratio in ratios.items() if ratio < SCRIPT_RATIO))
    detail = "gate pass rate of fact-carrying turns: " + ", ".join(
        f"{script} {rate:.0%}" for script, rate in sorted(rates.items()))
    detail += (f"; below {SCRIPT_RATIO} times the Latin rate: {', '.join(flagged)}"
               if flagged else f"; no script below {SCRIPT_RATIO} times the Latin rate")
    return Finding(name, TRACKED, min(ratios.values(), default=None), SCRIPT_RATIO,
                   detail, flagged)


def retraction_noop(obs: Observations) -> Finding:
    """A retraction that closed nothing, which must not happen at all."""
    name, rec = "retraction that retires nothing", obs.recorder
    noop = rec.total(WRITE_RETRACTION, outcome="noop")
    closed = rec.total(WRITE_RETRACTION, outcome="retired")
    if not noop + closed:
        return Finding(name, FAIL, None, 0.0, "not measured: no retraction was counted")
    detail = f"{closed} retractions closed a value and {noop} closed nothing"
    return Finding(name, FAIL if noop else OK, float(noop), 0.0, detail)


def redaction_drift(obs: Observations) -> Finding:
    """The share of turns with planted personal data that the redactor changed, per
    simulated day. Every day must reach 0.99."""
    name, rec = "redaction drift", obs.recorder
    turns = rec.total(WRITE_TURNS)
    if turns and REDACT_INSPECTED not in rec.names():
        return Finding(name, FAIL, None, REDACTION_RATIO, f"not running: {turns} turns "
                       "were written and no string was offered to the redactor, so the "
                       "deployment has lost its policy")
    if not obs.planted:
        return Finding(name, FAIL, None, REDACTION_RATIO, "not measured: no turn carried "
                       "planted personal data")
    days: dict[int, list[bool]] = {}
    for day, changed in obs.planted:
        days.setdefault(day, []).append(changed)
    ratios = {day: sum(changed) / len(changed) for day, changed in days.items()}
    worst = min(sorted(ratios), key=lambda day: ratios[day])
    caught = sum(changed for _, changed in obs.planted)
    detail = (f"the redactor changed {caught} of {len(obs.planted)} turns carrying "
              f"planted personal data; the lowest day was day {worst + 1}, at "
              f"{ratios[worst]:.0%}")
    if obs.unplanted_changed:
        detail += f"; it also changed {obs.unplanted_changed} turns with nothing planted"
    return Finding(name, FAIL if ratios[worst] < REDACTION_RATIO else OK, ratios[worst],
                   REDACTION_RATIO, detail)


def store_growth(obs: Observations, history: Sequence[float],
                 remeasure: Callable[[], float]) -> Finding:
    """Bytes on disk per turn, judged against earlier soaks by the regression rule.

    `history` holds the bytes per turn of earlier soaks with the same turns and seed,
    oldest first. `remeasure` runs the soak again and returns its bytes per turn; it is
    called only when the first two conditions of the rule hold.
    """
    name = "store growth"
    if obs.store_bytes is None:
        return Finding(name, TRACKED, None, None, "not measured: the store was in memory, "
                       "so there are no files to measure")
    turns = obs.config.turns
    per_turn = obs.store_bytes / turns
    verdict = perf_budget.judge(per_turn, history, remeasure, floor=PAGE_BYTES / turns)
    if verdict.outcome == "no history":
        return Finding(name, TRACKED, per_turn, None,
                       f"{per_turn:,.0f} bytes a turn; not judged yet: {verdict.detail}")
    limit = None if verdict.median is None else perf_budget.REGRESSION_RATIO * verdict.median
    return Finding(name, FAIL if verdict.outcome == "regression" else OK, per_turn, limit,
                   f"{per_turn:,.0f} bytes a turn; {verdict.outcome}: {verdict.detail}")


#: Every detector that reads only the run itself, in the order the design lists them.
DETECTORS: tuple[Callable[[Observations], Finding], ...] = (
    predicate_explosion, recency_refresh, flip_flop, salience_over_relevance, script_bias,
    retraction_noop, redaction_drift,
)


def _cannot_remeasure() -> float:
    raise AssertionError("unreachable: judge_soak refuses history without remeasure")


def judge_soak(obs: Observations, *, history: Sequence[float] = (),
               remeasure: Callable[[], float] | None = None) -> list[Finding]:
    """Every detector's finding on `obs`, store growth last.

    Store growth is compared with `history` only when a way to measure again is given,
    because the rule's third condition is a re-measure.
    """
    if history and remeasure is None:
        raise ValueError("comparing store growth with earlier soaks needs a way to measure "
                         "it again: pass remeasure")
    findings = [detector(obs) for detector in DETECTORS]
    findings.append(store_growth(obs, history, remeasure or _cannot_remeasure))
    return findings


# --- the record, the history and the command line ------------------------------------------

#: What every soak record says it is, so a folder of mixed records can be read safely.
RECORD_KIND = "memvara-soak"
RECORD_VERSION = 1


def record(obs: Observations, findings: Sequence[Finding]) -> dict[str, Any]:
    """The run as JSON: what ran, on which machine, what it counted, and every finding."""
    rec = obs.recorder
    return {
        "kind": RECORD_KIND, "version": RECORD_VERSION, "started": obs.started,
        "turns": obs.config.turns, "seed": obs.config.seed,
        "fingerprint": perf_budget.machine_fingerprint(),
        "bytes_per_turn": (None if obs.store_bytes is None
                           else obs.store_bytes / obs.config.turns),
        "elapsed_s": round(obs.elapsed_s, 3),
        "counts": {
            "reconcile": {action: rec.total(WRITE_RECONCILE, action=action)
                          for action in ("add", "reinforce", "supersede", "retract",
                                         "noop")},
            "retraction": {outcome: rec.total(WRITE_RETRACTION, outcome=outcome)
                           for outcome in ("retired", "noop")},
            "merged": rec.total(CONSOLIDATE_MERGED),
        },
        "findings": [dataclasses.asdict(finding) for finding in findings],
    }


def load_history(directory: Path, *, turns: int, seed: int) -> list[float]:
    """Bytes per turn of every earlier soak in `directory` with these turns and this
    seed, oldest first.

    Only a soak of the same length and seed runs the same operations, so only its growth
    can be compared. Files that are not soak records, and soaks kept in memory, are
    skipped. A folder that does not exist yet is an empty history.
    """
    if not directory.is_dir():
        return []
    found: list[tuple[str, float]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(data, dict) and data.get("kind") == RECORD_KIND
                and data.get("turns") == turns and data.get("seed") == seed
                and isinstance(data.get("bytes_per_turn"), (int, float))):
            found.append((str(data.get("started", "")), float(data["bytes_per_turn"])))
    return [value for _, value in sorted(found)]


def main(argv: Sequence[str] | None = None) -> int:
    """Run one soak, print its findings, and return 1 if any failed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--turns", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--store", type=Path, default=None,
                        help="the folder to create the store in, which keeps it; by "
                             "default a temporary folder that is removed afterwards")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the run's record to this JSON file")
    parser.add_argument("--history", type=Path, default=None,
                        help="a folder of earlier records to judge store growth against")
    args = parser.parse_args(argv)
    config = SoakConfig(args.turns, args.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    with tempfile.TemporaryDirectory(prefix="memvara-soak-") as scratch:
        folder = args.store if args.store is not None else Path(scratch)
        folder.mkdir(parents=True, exist_ok=True)
        name = f"soak-{config.turns}-{config.seed}-{stamp}"
        observed = run(config, folder / f"{name}.db")
        history = (load_history(args.history, turns=config.turns, seed=config.seed)
                   if args.history is not None else [])
        again = iter(range(1, 1_000))

        def remeasure() -> float:
            repeat = run(config, folder / f"{name}-again-{next(again)}.db")
            assert repeat.store_bytes is not None
            return repeat.store_bytes / config.turns

        findings = judge_soak(observed, history=history,
                              remeasure=remeasure if history else None)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(record(observed, findings), indent=2) + "\n",
                                encoding="utf-8")
    per_turn = ("" if observed.store_bytes is None
                else f", {observed.store_bytes / config.turns:,.0f} bytes a turn")
    print(f"soak: {config.turns} turns, seed {config.seed}, "
          f"{observed.elapsed_s:.1f} s{per_turn}")
    for finding in findings:
        print(f"{finding.status:7} {finding.detector}: {finding.detail}")
    return 1 if any(finding.status == FAIL for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
