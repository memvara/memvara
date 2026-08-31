"""Turning an answer into something comparable, without a model and without charity.

A benchmark that marks `"London"` wrong because the system said `"london."` measures
punctuation. One that marks `"not London — she moved to Paris"` right because the string
`London` is in there measures nothing at all. Both mistakes are easy, and the second is
the one that flatters.

So there are exactly two ways to be right about a value:

1. **Normalized equality** with the gold value or one of its published aliases. Case,
   punctuation, articles and surrounding whitespace are removed first.
2. **Unambiguous containment**: the gold value appears as a whole phrase inside a short
   answer, *and no competing value for the same slot does*. The competitors come from the
   dataset — every other object ever asserted for that fact slot — so an answer that
   names two of them is ambiguous and scores wrong, however confident it sounds.

Rule 2 is what lets `"She lived in London."` count. It is also the rule that could be
abused, so it is bounded twice over: the answer has a length ceiling, and naming a second
candidate is fatal rather than ignored. `--match strict` turns it off entirely and scores
on rule 1 alone; `docs/benchmarks/agent-memory-benchmark.md` reports both where they
differ.

Dates are their own thing and are not normalized as prose: a date answer must be
ISO-8601. Parsing `"the fifteenth of March"` would need a locale, a calendar and a
tie-break, and every one of those is a place for the benchmark to be quietly wrong.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timezone
from typing import Iterable, Sequence

#: Words dropped before comparison. Small and closed on purpose — a stopword list long
#: enough to be interesting is long enough to make two different answers equal.
_ARTICLES = frozenset({"a", "an", "the"})

#: The longest answer rule 2 will look inside, in tokens. An answer longer than this is
#: a paragraph, and finding a value inside a paragraph is not evidence the system knows
#: it — it is evidence the system said a lot of things.
CONTAINMENT_TOKEN_LIMIT = 12

_PUNCT = re.compile(r"[^\w\s-]", re.UNICODE)
_SPACE = re.compile(r"\s+")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def normalize(text: str | None) -> str:
    """Fold an answer to its comparable form.

    >>> normalize("  The London Office. ")
    'london office'
    >>> normalize("us-east-1") == normalize("US-EAST-1")
    True
    >>> normalize(None)
    ''
    """
    if text is None:
        return ""
    folded = unicodedata.normalize("NFKD", text).casefold()
    folded = _PUNCT.sub(" ", folded)
    words = [w for w in _SPACE.split(folded) if w and w not in _ARTICLES]
    return " ".join(words)


def tokens(text: str | None) -> list[str]:
    """The normalized tokens of an answer.

    >>> tokens("Alice lives in New York")
    ['alice', 'lives', 'in', 'new', 'york']
    """
    normalized = normalize(text)
    return normalized.split() if normalized else []


def _phrase_in(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    """Whole-token-subsequence containment, so `york` does not match `yorkshire`."""
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    span = len(needle)
    return any(list(haystack[i:i + span]) == list(needle)
               for i, token in enumerate(haystack) if token == first)


def matches_value(answer: str | None, gold: str, aliases: Iterable[str] = (),
                  competitors: Iterable[str] = (), *, lenient: bool = True) -> bool:
    """Is `answer` the gold value?

    `competitors` are the other values the same fact slot has ever held. They are what
    makes lenient matching safe: an answer naming two of them is not an answer.

    >>> matches_value("London", "London")
    True
    >>> matches_value("She lived in London.", "London", competitors=["Berlin"])
    True
    >>> matches_value("Berlin, then London", "London", competitors=["Berlin"])
    False
    >>> matches_value("She lived in London.", "London", lenient=False)
    False
    """
    accepted = [gold, *aliases]
    normalized = normalize(answer)
    if not normalized:
        return False
    if any(normalized == normalize(candidate) for candidate in accepted):
        return True
    if not lenient:
        return False

    answer_tokens = tokens(answer)
    if len(answer_tokens) > CONTAINMENT_TOKEN_LIMIT:
        return False
    if not any(_phrase_in(answer_tokens, tokens(candidate)) for candidate in accepted):
        return False
    # Named the gold. Did it also name something it is meant to have ruled out?
    accepted_norms = {normalize(candidate) for candidate in accepted}
    for other in competitors:
        if normalize(other) in accepted_norms:
            continue
        if _phrase_in(answer_tokens, tokens(other)):
            return False
    return True


def matches_set(values: Sequence[str], gold: Sequence[str]) -> bool:
    """Is `values` exactly the gold set, ignoring order and formatting?

    Exact, not overlapping. A system that returned every value it had ever seen would
    score full marks on every set question under an overlap rule, and `change_detection`
    is precisely the category where "name them all" must not be a winning strategy.

    Aliases do not apply to set answers, and `dataset.validate` refuses a set-valued gold
    that carries any — a flat alias list cannot say which member it is an alias *for*, so
    honouring it would mean guessing, and guessing in the scorer is worse than a stricter
    rule stated up front.

    >>> matches_set(["London", "Berlin"], ["Berlin", "London"])
    True
    >>> matches_set(["london.", " BERLIN "], ["Berlin", "London"])
    True
    >>> matches_set(["London", "Berlin", "Paris"], ["Berlin", "London"])
    False
    >>> matches_set(["London"], ["Berlin", "London"])
    False
    """
    got = {normalize(v) for v in values if normalize(v)}
    want = {normalize(v) for v in gold if normalize(v)}
    return bool(want) and got == want


def parse_date(text: str | None) -> date | None:
    """Read a calendar day out of an answer, ISO-8601 only.

    Accepts a bare `YYYY-MM-DD`, a full timestamp, or an ISO date embedded in a sentence.
    Anything else is `None`, which scores as wrong rather than as an abstention — the
    system answered, it just did not answer in the one format the benchmark reads.

    >>> parse_date("2026-03-15")
    datetime.date(2026, 3, 15)
    >>> parse_date("We learned it on 2026-03-15, a week late.")
    datetime.date(2026, 3, 15)
    >>> parse_date("last March") is None
    True
    """
    if not text:
        return None
    raw = text.strip()
    candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        when = datetime.fromisoformat(candidate)
    except ValueError:
        found = _ISO_DATE.search(raw)
        if not found:
            return None
        return date(int(found.group(1)), int(found.group(2)), int(found.group(3)))
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc)
    return when.date()


def matches_date(answer: str | None, gold: str) -> bool:
    """Same calendar day, in UTC.

    >>> matches_date("2026-03-15T09:00:00Z", "2026-03-15")
    True
    >>> matches_date("2026-03-16", "2026-03-15")
    False
    """
    got, want = parse_date(answer), parse_date(gold)
    return got is not None and want is not None and got == want
