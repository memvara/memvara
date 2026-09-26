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

The design of the adversarial suite (`docs/superpowers/specs/2026-09-25-adversarial-test-
suite-design.md`, "Phase 4") fixes when each detector fails. `docs/claude/testing.md`
explains how the run works and how each detector is shown to fire.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memvara.schema import BUILTIN_PREDICATES  # noqa: E402

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
