"""Tier 1b: deterministic extraction for statement forms that are unambiguous.

A small number of surface forms account for a large share of the durable facts a
personal assistant ever learns: "my name is X", "I live in X", "I work at X", "I'm
allergic to X", "I no longer work at X". Those need no model. Pattern-matching them
turns the most common writes into a zero-token, zero-latency, perfectly reproducible
path, and leaves the LLM for the genuinely messy remainder.

The tuning target is precision, not recall. A wrong triple is a lie the store will
happily repeat for months; a missed triple falls through to the LLM tier one function
call later. So every ambiguity here resolves to "emit nothing": conjunctions, hedges,
negations attached to positive patterns, and pronoun-headed objects are all dropped
rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from ..schema import PredicateRegistry
from ..types import SELF_SUBJECT, Claim, Derivation, Episode

EXTRACTOR = "fast/v1"

# High but not certain: the pattern is exact, yet nobody confirmed the user meant it
# literally. Leaving headroom below 1.0 keeps LLM- and user-asserted claims rankable
# above rule output when they disagree.
CONFIDENCE = 0.95

_OBJ = r"(?P<obj>[^,.;:!?\n]+)"
_OBJ_LAZY = r"(?P<obj>[^,.;:!?\n]+?)"
_OBJ2 = r"(?P<obj2>[^,.;:!?\n]+)"
_I = r"\bi\s+"
_IAM = r"\bi(?:\s+am|\s*['’]m)\s+"
_END = r"\s*$"

_SENTENCE = re.compile(r"[^.!?\n]+[.!?\n]?")
# Commas and semicolons always separate clauses; "and" only does when a new first-person
# subject follows it, so "my name is X and I work at Y" splits into two facts while
# "I like coffee and tea" stays one clause (and is then rejected as a coordinated object).
_CLAUSE_SPLIT = re.compile(r"[,;]|\s+and\s+(?=(?:i|my)\b)", re.IGNORECASE)

# English routinely elides a repeated subject across a conjunction: "I live in Berlin and
# work at Acme" is two facts, but the second clause has no subject of its own, so the
# split above misses it and the whole sentence falls through to the LLM. Restoring the
# elided "I" before a first-person verb turns it back into a case the split already
# handles. Gated on an explicit verb list rather than "any word", because the coordinated
# *object* reading must keep working - in "I like coffee and tea", "tea" is not a verb, so
# nothing is inserted and the clause stays whole (then gets rejected as coordinated).
_ELIDED_SUBJECT = re.compile(
    r"\s+and\s+(?="
    r"(?:live|lives|living|work|works|working|like|likes|love|loves|prefer|prefers|"
    r"hate|hates|dislike|dislikes|speak|speaks|own|owns|use|uses|drive|drives|"
    r"study|studies|teach|teaches|run|runs|manage|manages|am|was)\b)",
    re.IGNORECASE,
)

# Cheap pre-filter: every rule below is anchored on a first-person subject, so a clause
# without one cannot match and does not deserve 20 regex evaluations.
_HAS_SUBJECT = re.compile(r"\b(?:i|my|you)\b", re.IGNORECASE)

# A negation anywhere in the clause disqualifies the *positive* rules. "I'm not allergic
# to peanuts" and "I'm allergic to peanuts" differ by one token and mean opposite things,
# and getting that backwards is worse than extracting nothing.
_NEGATION = re.compile(r"(?:\bnot\b|\bnever\b|\bno\b|n['’]t)", re.IGNORECASE)

# Hedged, hypothetical, or reported speech: the sentence does not assert anything about
# the world, whatever its grammatical shape.
_HYPOTHETICAL = re.compile(
    r"\b(?:if|would|could|should|might|maybe|perhaps|probably|possibly|suppose|"
    r"imagine|wish|unless|whether|assume|pretend|said|says|asked|think|thought|guess)\b",
    re.IGNORECASE,
)

_ARTICLE = re.compile(r"^(?:an?|the)\s+", re.IGNORECASE)
# Adverbial tails that ride along with the value but are not part of it. Left in place
# they fragment the slot: "Lisbon" and "Lisbon last month" would be two different facts.
_FILLER = re.compile(
    r"\s+(?:now|today|these\s+days|currently|at\s+the\s+moment|again|too|"
    r"as\s+well|though|actually|anymore|any\s+more|recently|lately|full\s+time|"
    r"part\s+time|for\s+good|permanently|yesterday|ago|"
    r"(?:last|this|next)\s+(?:week|month|year|summer|winter|spring|fall|autumn)|"
    r"\d+\s+(?:years?|months?|weeks?|days?)|in\s+\d{4})$",
    re.IGNORECASE,
)

_BAD_OBJECT_PREFIX = (
    "to ", "that ", "it ", "this ", "there ", "here ", "you ", "your ", "he ", "she ",
    "they ", "them ", "we ", "us ", "how ", "when ", "what ", "where ", "why ", "who ",
    "being ", "having ", "doing ", "going ", "getting ", "with ", "about ", "because ",
)
_BAD_OBJECT_EXACT = frozenset(
    {"it", "this", "that", "there", "here", "you", "me", "them", "us", "one", "some",
     "any", "something", "anything", "stuff", "things", "thing"}
)

_REFERENTIAL = re.compile(
    r"\b(?:we|us|you|they|them|i|me|my|our|your|their|this|that|those|these|"
    r"whose|which|who)\b",
    re.IGNORECASE,
)
_MAX_OBJECT_CHARS = 80
_MAX_OBJECT_WORDS = 8


@dataclass(frozen=True, slots=True)
class _Rule:
    pattern: re.Pattern[str]
    # (regex group name, canonical predicate, polarity). A rule may emit more than one
    # triple: "I work as a designer at Acme" is genuinely two facts.
    outputs: tuple[tuple[str, str, int], ...]

    @property
    def asserts(self) -> bool:
        return any(pol > 0 for _, _, pol in self.outputs)


def _rule(pattern: str, *outputs: tuple[str, str, int]) -> _Rule:
    return _Rule(re.compile(pattern, re.IGNORECASE), outputs)


# Order is load-bearing twice over: retractions come first so "I no longer work at X"
# is never read as an assertion, and the more specific member of an overlapping pair
# comes first so it wins the single-match-per-clause race below.
_RULES: tuple[_Rule, ...] = (
    # --- retractions (polarity -1) ---
    _rule(_I + r"no\s+longer\s+work(?:s|ed)?\s+(?:at|for)\s+" + _OBJ + _END,
          ("obj", "works_at", -1)),
    _rule(_I + r"no\s+longer\s+live\s+in\s+" + _OBJ + _END, ("obj", "lives_in", -1)),
    _rule(_I + r"no\s+longer\s+(?:like|love|enjoy)\s+" + _OBJ + _END, ("obj", "likes", -1)),
    _rule(_I + r"used\s+to\s+work\s+(?:at|for)\s+" + _OBJ + _END, ("obj", "works_at", -1)),
    _rule(_I + r"used\s+to\s+live\s+in\s+" + _OBJ + _END, ("obj", "lives_in", -1)),
    _rule(_I + r"(?:do\s+not|don['’]t)\s+work\s+(?:at|for)\s+" + _OBJ_LAZY
          + r"\s+any\s?more" + _END, ("obj", "works_at", -1)),
    _rule(_I + r"(?:do\s+not|don['’]t)\s+live\s+in\s+" + _OBJ_LAZY
          + r"\s+any\s?more" + _END, ("obj", "lives_in", -1)),
    _rule(_IAM + r"no\s+longer\s+allergic\s+to\s+" + _OBJ + _END, ("obj", "allergic_to", -1)),

    # --- identity ---
    _rule(r"\bmy\s+name\s+is\s+" + _OBJ + _END, ("obj", "name", 1)),
    _rule(r"\byou\s+can\s+call\s+me\s+" + _OBJ + _END, ("obj", "name", 1)),
    _rule(r"\bmy\s+pronouns\s+are\s+" + _OBJ + _END, ("obj", "pronouns", 1)),
    _rule(r"\bmy\s+time\s?zone\s+is\s+" + _OBJ + _END, ("obj", "timezone", 1)),
    _rule(r"\bmy\s+(?:job\s+)?title\s+is\s+" + _OBJ + _END, ("obj", "job_title", 1)),

    # --- situation ---
    _rule(_I + r"(?:live|reside)\s+in\s+" + _OBJ + _END, ("obj", "lives_in", 1)),
    _rule(_IAM + r"based\s+in\s+" + _OBJ + _END, ("obj", "lives_in", 1)),
    _rule(_I + r"(?:just\s+)?(?:moved|relocated)\s+to\s+" + _OBJ + _END,
          ("obj", "lives_in", 1)),
    _rule(_I + r"work\s+as\s+(?:an?\s+)?" + _OBJ_LAZY + r"\s+at\s+" + _OBJ2 + _END,
          ("obj", "job_title", 1), ("obj2", "works_at", 1)),
    _rule(_I + r"work\s+as\s+(?:an?\s+)?" + _OBJ + _END, ("obj", "job_title", 1)),
    _rule(_I + r"work\s+(?:at|for)\s+" + _OBJ + _END, ("obj", "works_at", 1)),

    # --- preferences and constraints ---
    _rule(_IAM + r"allergic\s+to\s+" + _OBJ + _END, ("obj", "allergic_to", 1)),
    _rule(_IAM + r"(?P<obj>vegan|vegetarian|pescatarian)\b",
          ("obj", "dietary_restriction", 1)),
    _rule(_I + r"(?:really\s+|also\s+)?prefer\s+" + _OBJ + _END, ("obj", "prefers", 1)),
    _rule(_I + r"(?:really\s+|also\s+)?(?:like|love|enjoy)\s+" + _OBJ + _END,
          ("obj", "likes", 1)),
    _rule(_I + r"(?:really\s+|also\s+)?(?:dislike|hate)\s+" + _OBJ + _END,
          ("obj", "dislikes", 1)),
    _rule(_I + r"speak\s+" + _OBJ + _END, ("obj", "speaks", 1)),
)


def _clauses(content: str) -> Iterator[str]:
    """Sentences that are not questions, split further on commas and semicolons."""
    for sentence in _SENTENCE.findall(content):
        s = sentence.strip()
        if not s or s.endswith("?"):
            continue
        s = _ELIDED_SUBJECT.sub(" and I ", s)
        for part in _CLAUSE_SPLIT.split(s):
            p = part.strip().strip("\"'“”‘’()[]").lstrip("-*•– ")
            p = p.rstrip(".,;:!? ").strip()
            if p:
                yield p


def _clean_object(raw: str) -> str | None:
    """Normalize a captured object, or reject it outright.

    Every rejection here is a deliberate hand-off to the LLM tier rather than a guess.
    """
    o = raw.strip().strip("\"'“”‘’ ").rstrip(".,;:!? ").strip()
    o = _ARTICLE.sub("", o).strip()
    prev = ""
    while prev != o:  # filler can stack: "in Berlin now too"
        prev = o
        o = _FILLER.sub("", o).strip()

    if not o or len(o) > _MAX_OBJECT_CHARS:
        return None
    low = o.lower()
    if low in _BAD_OBJECT_EXACT or low.startswith(_BAD_OBJECT_PREFIX):
        return None
    # A coordinated object is two facts wearing one hat ("coffee and tea"). Splitting it
    # correctly needs real parsing, so hand the whole clause to the LLM instead.
    if " and " in low or " or " in low or " but " in low:
        return None
    if len(o.split()) > _MAX_OBJECT_WORDS:
        return None
    # A pronoun inside the value means this is a referential phrase, not a value:
    # "the place we discussed", "the company you know about". Resolving those needs
    # the conversation, which is exactly what the LLM tier has and these rules do not.
    # Real values — "Acme Corp", "San Francisco", "pytest" — never contain one.
    if _REFERENTIAL.search(o):
        return None
    if not any(ch.isalnum() for ch in o):
        return None
    return o


class FastExtractor:
    """Rule-based extraction. Emits nothing rather than something wrong."""

    def __init__(self, registry: PredicateRegistry) -> None:
        self.registry = registry

    def extract(self, ep: Episode) -> list[Claim]:
        content = (ep.content or "").strip()
        if not content:
            return []
        if (ep.role or "user").strip().lower() != "user":
            # Every rule reads "I"/"my" as the user. On an assistant turn that subject
            # binding is simply false.
            return []

        out: list[Claim] = []
        seen: set[tuple[str, str, int]] = set()

        for clause in _clauses(content):
            if not _HAS_SUBJECT.search(clause):
                continue
            if _HYPOTHETICAL.search(clause):
                continue
            negated = _NEGATION.search(clause) is not None

            for rule in _RULES:
                if negated and rule.asserts:
                    continue
                m = rule.pattern.search(clause)
                if m is None:
                    continue
                triples: list[tuple[str, str, int]] = []
                for group, predicate, polarity in rule.outputs:
                    value = _clean_object(m.group(group) or "")
                    if value is None:
                        triples = []
                        break
                    triples.append((predicate, value, polarity))
                if not triples:
                    continue  # cleaning failed; a later, looser rule may still fit
                for predicate, value, polarity in triples:
                    pred = self.registry.normalize(predicate)
                    key = (pred, value.casefold(), polarity)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(self._claim(ep, pred, value, polarity))
                # One rule per clause: a clause asserts one thing, and letting a second,
                # looser rule fire on the same words produces duplicate garbage.
                break

        return out

    def _claim(self, ep: Episode, predicate: str, value: str, polarity: int) -> Claim:
        spec = self.registry.spec(predicate)
        return Claim(
            subject=SELF_SUBJECT,
            predicate=predicate,
            object=value,
            scope=ep.scope,
            polarity=polarity,
            memory_type=spec.memory_type,
            # Valid time is when the user said it; transaction time is set by the caller
            # so a whole batch shares one "when we came to believe this" instant.
            valid_from=ep.ts,
            confidence=CONFIDENCE,
            sources=[ep.id],
            derivation=Derivation.FAST_PATH,
            extractor=EXTRACTOR,
        )
