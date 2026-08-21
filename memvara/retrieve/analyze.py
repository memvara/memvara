"""Query analysis for the lexical leg.

BM25 weights a term by how *rare* it is, which is normally the property that makes it
robust. At personal-memory scale it inverts. A store holding a few hundred claims about
one person contains "lisbon" and "shellfish" many times and the word "do" almost never,
so `"what do you know about me?"` becomes six full-weight terms with no content in them
and the one claim that happens to contain "do" is returned as the confident answer.
The pathology gets *worse* as the corpus shrinks, which is exactly backwards from the
usual intuition that small corpora are easy - and small is the normal state of a
personal memory, not a transient one.

The fix has to live here rather than in the store. `_fts_query` ORs every alphanumeric
token it is given, so the only lever the retriever holds is *which tokens it hands
over*. Two things follow:

  * a query is reduced to its content terms before it reaches the store, and
  * a query with no content terms at all makes the lexical leg abstain entirely,
    mirroring the zero-norm guard the vector leg already has.

Abstaining is the important half. A ranking assembled from stopword IDF is not a weak
ranking, it is a fabricated one, and fusion cannot tell the difference - it reads
positions, and positions are exactly what a fabricated ranking supplies.
"""

from __future__ import annotations

from dataclasses import dataclass

# Closed-class English function words: determiners, pronouns, prepositions,
# conjunctions, auxiliaries and wh-words. Membership of these classes is fixed by the
# grammar, not by the domain, which is what makes a hand-written list defensible here -
# no open-class word can be added later and no corpus can make "of" informative.
#
# Deliberately absent: anything that could be the content of a memory. "name",
# "never", "like", "work", "live", "know" and "call" all carry meaning in a personal
# store ("never_do" is a whole predicate), and dropping them would trade a
# false-positive problem for a silent false-negative one.
STOPWORDS: frozenset[str] = frozenset("""
    a an the this that these those such other another
    all any both each either every few many most neither no none some
    i me my mine myself we us our ours ourselves you your yours yourself
    he him his she her hers it its they them their theirs one ones
    who whom whose what which whatever whichever
    when where why how whether
    anyone anything everyone everything someone something nothing nobody
    is am are was were be been being
    do does did doing done
    have has had having
    will would shall should can could may might must
    of to in into on onto at for from by with without within about above below
    over under between through during after before again further up down out off
    and or but if then than so as because while until unless though although
    there here also just only very too much more less least same
""".split())

# `_fts_query` in the SQLite store drops single-character tokens, so the analyzer drops
# them too: the two must agree about what a "term" is, or the per-term normalization in
# `scoring.lexical_relevance` divides by a count the store never used.
MIN_TERM_CHARS = 2


def tokenize(raw: str) -> list[str]:
    """Split on runs of alphanumerics, lowercased - the store's tokenization, mirrored.

    Unicode-aware via `str.isalnum`, so CJK text survives rather than being discarded the
    way an ASCII word regex would discard it. **Surviving is not the same as being
    segmented, and the difference is the whole of this limitation.** Chinese, Japanese and
    Korean are written without spaces, so a contiguous run of them is alphanumeric from
    end to end and comes out as one enormous token — the entire phrase, indexed under
    itself. A search for a word inside that phrase matches nothing at all, because the
    index holds no such term.

    Both stores behave this way, for the same reason and independently: SQLite's
    `unicode61` and Postgres's `to_tsvector` neither of them segment. It is a lexical-leg
    limitation rather than a bug in this function, which is faithfully mirroring what the
    store does; fixing it means segmenting on both sides at once. See the CJK segmentation
    issue for the options and why it is not simply switched on.

    The doctest below is Latin because that is the case this function gets right; the
    behaviour above has no honest one-line example, which is part of why the comment used
    to read as though nothing were wrong.

    >>> tokenize("What is my mother's maiden name?")
    ['what', 'is', 'my', 'mother', 'maiden', 'name']
    """
    tokens: list[str] = []
    current: list[str] = []
    for ch in raw.lower():
        if ch.isalnum():
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return [t for t in tokens if len(t) >= MIN_TERM_CHARS]


@dataclass(frozen=True, slots=True)
class LexicalQuery:
    """What the lexical leg should actually ask the store, and what it costs.

    `terms` is empty exactly when the leg must abstain. `text` is the reduced query
    string; re-tokenizing it yields `terms` unchanged, so handing it to a store that
    builds its own MATCH expression is idempotent.
    """

    text: str
    terms: tuple[str, ...]

    @property
    def abstains(self) -> bool:
        return not self.terms


def analyze(raw: str) -> LexicalQuery:
    """Reduce a user query to the terms worth matching on.

    Repeats collapse. A term appearing twice does not make a document twice as
    relevant, but it does double that term's weight inside BM25's sum and inflate the
    denominator this module hands to the per-term normalization - and on a pasted
    document-sized query it is the difference between a 6-term MATCH expression and a
    2,000-term one.

    >>> analyze("what do you know about me?").terms
    ('know',)
    >>> analyze("what is it about?").abstains
    True
    >>> analyze("Where does the user live?").text
    'user live'
    """
    seen: set[str] = set()
    terms: list[str] = []
    for token in tokenize(raw):
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return LexicalQuery(text=" ".join(terms), terms=tuple(terms))
