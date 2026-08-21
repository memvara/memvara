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

**Two things here are not first-person declaratives**, and both are here because the
vocabulary being *only* that was measured and found to extract literally nothing from
sixty-four turns of ordinary support prose (`docs/BENCHMARKS.md`).

*Contact directives* — "ring me", "email from now on", "stop ringing me" — are second
person and imperative, they carry no subject for `_HAS_SUBJECT` to find, and their value
is not in the text at all: "ring me" means the phone and does not contain the word. They
are also the one family that writes `MemoryType.PROCEDURAL`, because a standing
instruction about how to be reached is not a fact about the world.

*Addresses and bare phone numbers* are matched on the **whole sentence**, before the
clause splitter runs. A postal address is one fact with commas in it, and split on those
commas it becomes three fragments and a postcode; a phone number given as an answer is an
entire utterance with no grammar around it for a clause rule to key on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from ..schema import PredicateRegistry
from ..types import Claim, Derivation, Episode

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
# Commas and semicolons always separate clauses; "and" only does when a new clause
# *subject* follows it, so "my name is X and I work at Y" splits into two facts while
# "I like coffee and tea" stays one clause (and is then rejected as a coordinated object).
#
# The subject list is closed-class pronouns and determiners, and it is wider than the
# first person on purpose: "I'm redoing the insurance schedule and it wants the serial"
# is one fact and one aside, not a coordinated object, and read as the latter the whole
# clause was rejected and the fact was lost. Nothing here can appear as the head of a
# coordinated object — "coffee and tea" keeps working because "tea" is not a pronoun.
_CLAUSE_SPLIT = re.compile(
    r"[,;]|\s+and\s+(?=(?:i|my|it|its|that|there|they|we|he|she|you|your)\b)",
    re.IGNORECASE)

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

# Cheap pre-filter over *clause* rules: a clause containing none of these words cannot
# match any of them, and does not deserve twenty regex evaluations.
#
# It used to be "has a first-person subject", which stopped being true when the contact
# directives landed: "email me from now on" is an imperative and has no subject at all,
# and a filter written as `i|my|you` deleted the rule silently rather than slowing it
# down. So the anchor set is the union of what the rules key on — first and second
# person, and the channel verbs — and widening it is part of adding a rule that does not
# need a subject. `tests/test_fast.py` is what catches a rule this filter has swallowed,
# because nothing here can.
_HAS_SUBJECT = re.compile(
    r"\b(?:i|my|me|you|your|ring|ringing|call|calling|phone|phoning|text|texting|"
    r"email|emails|emailing)\b",
    re.IGNORECASE)

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
    #: Objects the rule *is* rather than objects it captures, keyed by the pseudo-group
    #: name used in `outputs`. "ring me" and "give me a call" both mean the phone and
    #: neither contains the word: capturing an object there would store the verb, and
    #: "ringing" and "calling" would be two different values of one slot. A constant is
    #: already canonical, so it skips `_clean_object` — there is nothing to clean and an
    #: article stripper has no business rewriting a vocabulary this module owns.
    constants: tuple[tuple[str, str], ...] = ()
    #: True for a rule evaluated on the whole sentence, before it is split into clauses.
    #: For values that legitimately contain commas: a postal address is one fact with
    #: commas in it, not four clauses, and the clause splitter turns
    #: "41 Coldharbour Road, Lewes, BN7 2GT" into three fragments and a postcode.
    whole_sentence: bool = False

    @property
    def asserts(self) -> bool:
        return any(pol > 0 for _, _, pol in self.outputs)

    def constant(self, group: str) -> str | None:
        return next((v for k, v in self.constants if k == group), None)


def _rule(pattern: str, *outputs: tuple[str, str, int]) -> _Rule:
    return _Rule(re.compile(pattern, re.IGNORECASE), outputs)


#: The pseudo-group name a constant-object rule uses. Never a real regex group.
_FIXED = "fixed"


def _channel(pattern: str, channel: str, polarity: int) -> _Rule:
    """A contact directive: which channel to use, from a verb that does not name it.

    `contact_preference` is `ONE` in the shipped schema, so these supersede each other
    rather than accumulating — which is what makes "email from now on" close "please do
    ring" instead of leaving a store that believes both.
    """
    return _Rule(re.compile(pattern, re.IGNORECASE),
                 ((_FIXED, "contact_preference", polarity),),
                 constants=((_FIXED, channel),))


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

    # --- contact directives, negative (polarity -1) ---
    # First, with the retractions, for the reason the retractions are first: "stop
    # ringing me" contains no negation token, so a positive `ring me` rule placed above
    # it would match and record the opposite of what was said.
    _channel(r"\bstop\s+(?:ring|call|phon)ing\s+me\b", "phone", -1),
    _channel(r"\bstop\s+e?-?mailing\s+me\b", "email", -1),
    _channel(r"\bstop\s+texting\s+me\b", "text", -1),
    _channel(r"\b(?:do\s+not|don['’]t)\s+(?:ring|call|phone)\s+me\b", "phone", -1),
    _channel(r"\b(?:do\s+not|don['’]t)\s+e?-?mail\s+me\b", "email", -1),
    _channel(r"\b(?:do\s+not|don['’]t)\s+text\s+me\b", "text", -1),

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

    # --- contact directives, positive ---
    # Second person and imperative, which is why `_HAS_SUBJECT` is no longer a
    # first-person filter. "Ring me" is not a fact about the world in the way "I live in
    # Berlin" is; it is a standing instruction, which is what `MemoryType.PROCEDURAL`
    # means and why these are the one rule family here that writes one.
    _channel(r"\b(?:please\s+)?(?:do\s+)?(?:ring|call|phone)\s+me\b", "phone", 1),
    _channel(r"\b(?:please\s+)?(?:do\s+)?e?-?mail\s+me\b", "email", 1),
    _channel(r"\b(?:please\s+)?(?:do\s+)?text\s+me\b", "text", 1),
    # The emphatic imperative with no object: "and please do ring". `do` is what makes
    # it safe to read as a standing instruction rather than a one-off request — "please
    # call the office" is an errand and does not match, because the verb has an object.
    _channel(r"\b(?:please\s+)?do\s+(?:ring|call|phone)" + _END, "phone", 1),
    _channel(r"\b(?:please\s+)?do\s+e?-?mail" + _END, "email", 1),
    _channel(r"\b(?:please\s+)?do\s+text" + _END, "text", 1),
    _channel(r"\b(?:ring|call|phone)\s+(?:me\s+)?from\s+now\s+on\b", "phone", 1),
    _channel(r"\be?-?mail\s+(?:me\s+)?from\s+now\s+on\b", "email", 1),
    _channel(r"\btext\s+(?:me\s+)?from\s+now\s+on\b", "text", 1),
)

#: A value that may legitimately contain commas. Only whole-sentence rules use it.
_ADDR = r"(?P<obj>[^.;:!?\n]+)"

#: Rules run on the whole sentence, before the clause splitter sees it.
#:
#: Two families, and they are here for the same reason: their value is not a clause. A
#: postal address has commas in it by convention, and a phone number given as an answer
#: is an entire utterance with no grammar around it at all.
#:
#: They are *not* filtered by `_HAS_SUBJECT` — "everything comes to 41 Coldharbour Road"
#: has no subject in the first or second person and is exactly the sentence worth
#: catching — and a sentence that matches one of them is not then split into clauses, on
#: the same principle as one-rule-per-clause: it asserted one thing.
_SENTENCE_RULES: tuple[_Rule, ...] = tuple(
    _Rule(re.compile(pattern, re.IGNORECASE), outputs, whole_sentence=True)
    for pattern, outputs in (
        # "Everything comes to 41 Coldharbour Road, Lewes, BN7 2GT"
        (r"\b(?:everything|orders?|invoices?|deliveries|parcels?|post|mail)\s+"
         r"(?:comes?|go(?:es)?|should\s+(?:come|go))\s+to\s+" + _ADDR + _END,
         (("obj", "address", 1),)),
        # "my (delivery|postal|billing|home) address is X"
        (r"\bmy\s+(?:delivery\s+|postal\s+|billing\s+|home\s+|new\s+)?address\s+is\s+"
         + _ADDR + _END, (("obj", "address", 1),)),
        # "send/ship/post/deliver (it|them|everything) to X". A negation anywhere in the
        # sentence disqualifies it, which is why "ship them to Coldharbour Road, not the
        # Yard" extracts nothing rather than an address with ", not the Yard" on the end.
        (r"\b(?:send|ship|post|deliver)\s+"
         r"(?:it|them|everything|anything|orders?|invoices?|the\s+invoices?)?\s*to\s+"
         + _ADDR + _END, (("obj", "address", 1),)),
        # A phone number as the entire utterance — how anyone answers "what's your
        # number". The lookahead counts 9 to 15 digits across the whole string, which is
        # what separates a phone number from a year, a price or the serial off a power
        # brick: `HX2-4419-B` has letters, `2026` has four digits, `£79` has two.
        (r"^(?=(?:\D*\d){9,15}\D*$)(?P<obj>\+?\d[\d\s.+-]{7,18})$",
         (("obj", "phone", 1),)),
    )
)


def _sentences(content: str) -> Iterator[str]:
    """Sentences that are not questions, with their terminator and quoting stripped.

    Yielded whole rather than split, because a sentence is the unit the address and
    phone rules read: their values contain the punctuation the clause splitter cuts on.
    """
    for sentence in _SENTENCE.findall(content):
        s = sentence.strip()
        if not s or s.endswith("?"):
            continue
        s = s.strip("\"'“”‘’()[]").lstrip("-*•– ").rstrip(".,;:!? ").strip()
        if s:
            yield s


def _clauses_of(sentence: str) -> Iterator[str]:
    """One sentence, split on commas, semicolons and clause-introducing conjunctions."""
    sentence = _ELIDED_SUBJECT.sub(" and I ", sentence)
    for part in _CLAUSE_SPLIT.split(sentence):
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

        for sentence in _sentences(content):
            # Hedges scope over the whole sentence, not over the clause they sit in.
            # "If it breaks, call me" is a conditional instruction and its second clause,
            # read alone, is a standing preference — which is the reading a clause-scoped
            # guard produced, and it is wrong. The cost is a sentence that hedges one
            # thing and asserts another losing both, which is this module's stated trade:
            # a missed triple is one LLM call away, a wrong one is repeated for months.
            if _HYPOTHETICAL.search(sentence):
                continue
            # The whole-sentence tier first, and exclusively: a sentence that names an
            # address or a phone number has said its one thing, and running the clause
            # rules over the fragments afterwards can only produce pieces of it.
            if self._apply(_SENTENCE_RULES, sentence, ep, out, seen):
                continue
            for clause in _clauses_of(sentence):
                if not _HAS_SUBJECT.search(clause):
                    continue
                self._apply(_RULES, clause, ep, out, seen)

        return out

    def _apply(self, rules: tuple[_Rule, ...], text: str, ep: Episode,
               out: list[Claim], seen: set[tuple[str, str, int]]) -> bool:
        """Try `rules` against `text` in order, emit the first that fits, and stop.

        One rule per unit of text — a clause asserts one thing, and letting a second,
        looser rule fire on the same words produces duplicate garbage. Returns whether
        anything matched, which is what lets the sentence tier claim a sentence outright.
        """
        negated = _NEGATION.search(text) is not None
        for rule in rules:
            if negated and rule.asserts:
                continue
            m = rule.pattern.search(text)
            if m is None:
                continue
            triples: list[tuple[str, str, int]] = []
            for group, predicate, polarity in rule.outputs:
                fixed = rule.constant(group)
                value = fixed if fixed is not None else _clean_object(m.group(group) or "")
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
            return True
        return False

    def _claim(self, ep: Episode, predicate: str, value: str, polarity: int) -> Claim:
        spec = self.registry.spec(predicate)
        return Claim(
            subject="user",
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
