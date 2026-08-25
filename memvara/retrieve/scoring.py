"""Post-fusion rescoring: recency, confidence, salience, and the [0,1] normalization.

The gap this closes: a pure vector store ranks "works at Initech" (recorded 2023,
superseded 2026) exactly as highly as "works at Acme" today, because cosine
similarity has no opinion about time. Ranking by recency alone is equally wrong -
it would bury a birthplace under this morning's mood.

The fix is to key the decay to the *predicate*, not to the memory. `born_in` has a
100-year half-life and never meaningfully decays; `working_on` has a one-week
half-life and falls out of the ranking within days. Volatility is a schema property
we already know, so this stays a pure function - no LLM on the read path.

The second half of this module answers a different question: *how relevant is this,
in absolute terms?* See `normalized_score` for why that cannot be derived from the
fused rank, however it is rescaled.
"""

from __future__ import annotations

from datetime import datetime

from ..schema import PredicateRegistry
from ..types import Claim, as_utc


def recency_factor(claim: Claim, registry: PredicateRegistry, now: datetime) -> float:
    """Exponential decay on the predicate's half-life, in [0, 1].

    Age is measured from `Claim.trace_from` and never from `recorded_at`. That is
    `valid_from` - when the fact became true in the world - except when the fact has
    since been restated, in which case it is the last time we heard it. Both halves of
    that matter: backfilling a 2019 fact today must not make it look fresh, and a fact
    the user mentioned this morning must not be scored as stale as one nobody has
    brought up since it was first recorded. Keyed off `valid_from` alone, a claim
    observed 91 times over 90 days scored a recency factor of 1.35e-04.

    `now` is supplied rather than read from the clock so that a time-travel query
    decays relative to the moment being asked about. Asking "what did we believe in
    2024" and getting 2024's facts scored as two years stale would defeat the point.
    """
    half_life = registry.half_life_days(claim.predicate)
    if half_life <= 0.0:
        # No sane spec produces this, but a learned predicate could carry a bad value
        # and a ZeroDivisionError inside ranking is a terrible way to find out.
        return 0.0

    age_days = (as_utc(now) - claim.trace_from).total_seconds() / 86400.0
    if age_days <= 0.0:
        # Not yet true, or true as of exactly now. Clamp instead of letting the
        # exponent go positive: a fact scheduled for next month must not outscore a
        # current one by virtue of being in the future.
        return 1.0
    return 0.5 ** (age_days / half_life)


def final_score(
    fusion: float,
    *,
    recency: float,
    confidence: float,
    salience: float,
    w_recency: float,
    w_confidence: float,
    w_salience: float,
) -> float:
    """Scale the fusion score by a quality multiplier. The retriever's *raw* score.

    This is what `Explanation.raw_score` carries and what the ranking used to be sorted
    on. `normalized_score` is now the ranking and the caller-facing number; this stays
    because it is the right lens for debugging fusion itself, and because the two
    disagreeing is a finding rather than a bug.

    Multiplicative, not additive, and that is the load-bearing decision. RRF scores
    live around `1 / (k + 1)` - roughly 0.016 at the default k - while recency,
    confidence and salience all live around 1.0. Adding them would swamp the fusion
    term by two orders of magnitude and turn "search" into "sort every claim in scope
    by salience", which is precisely the ranking pathology this module exists to avoid.

    The arithmetic, stated exactly, because the ratio is the whole argument: at the
    retriever's default weights the multiplier is 1.25 for a fully decayed claim and
    1.5 for a fresh one at equal confidence and salience - a 1.2x span, widening to
    1.5x against a claim with every signal at zero and to 1.66x once salience is
    reinforced past 2.6. Adjacent RRF ranks differ by 1.016x at `rrf_k=60`. Scaling one
    by the other therefore buys freshness about 41 rank positions, not the dozen this
    docstring used to claim, which is why the normalized score bounds quality's
    authority instead of multiplying it in unchecked.

    Every factor is monotone increasing and each weight is independent, so setting a
    weight to 0 removes that signal exactly rather than re-baselining the others.
    Salience above 1.0 (heavily reinforced facts) is intentionally not clamped - it is
    headroom the write path earns by observing something repeatedly.
    """
    return fusion * quality_boost(
        recency=recency,
        confidence=confidence,
        salience=salience,
        w_recency=w_recency,
        w_confidence=w_confidence,
        w_salience=w_salience,
    )


def quality_boost(
    *,
    recency: float,
    confidence: float,
    salience: float,
    w_recency: float,
    w_confidence: float,
    w_salience: float,
) -> float:
    """The quality multiplier on its own: `1 + Σ wᵢ·signalᵢ`.

    At the retriever's default weights (0.25 / 0.15 / 0.10) and nominal signals this
    spans 1.0 (every signal at zero) to 1.5 (every signal at 1.0). Holding confidence
    and salience at 1.0 and varying only decay gives 1.25 → 1.5, a 1.2x span. Salience
    is not capped, so a claim reinforced to 2.6 - the top of the range reported from a
    production store - takes it to 1.66x. That is the number worth holding against the
    fusion term's rank-to-rank gap of 1.016x at `rrf_k=60`.

    That comparison is why the ranking no longer multiplies this into a fused rank:
    41 rank positions of relevance for one step of freshness is not a trade anyone
    chose. `normalized_score` divides it back out into a bounded factor instead.
    """
    return 1.0 + w_recency * recency + w_confidence * confidence + w_salience * salience


# --- absolute relevance -----------------------------------------------------

#: Per-term BM25 at which the lexical leg is judged half-convinced.
#:
#: Measured on a 36-claim personal-memory corpus with the shipped porter/unicode61
#: index: a term matching the claim that actually answers the query contributes 1.5-3.6
#: per query term, while an incidental match - a stemmer collision like
#: "production"/"products", or a common word shared by half the store - contributes
#: 0.3-1.0. Half-saturation at 1.5 puts the first group above 0.5 and the second below
#: 0.4.
#:
#: It is the one fitted constant in this module, and fitting it is cheap because it is
#: a soft knee rather than a threshold: `lexical_relevance` is monotone for every value
#: of it, so it moves calibration, not ordering. Swept from 0.5 to 5.0 on the eval set,
#: P@1 did not move at all and MRR moved by 0.01, while the floor that best separates
#: answerable from unanswerable slid from 0.33 to 0.21. Get it wrong and `min_score`
#: means something slightly different; the ranking is unaffected.
LEXICAL_HALF_SATURATION = 1.5


def lexical_relevance(bm25: float, terms: int) -> float:
    """Map a BM25 score onto [0, 1) as an absolute, query-length-independent signal.

    Two properties have to be fixed before BM25 can mean anything across queries:

    1. **Length.** BM25 sums over query terms, so a seven-word question scores higher
       than a two-word one against the same claim. Dividing by the number of content
       terms turns the sum into an average, in which an unmatched term contributes
       zero - so partial coverage is priced in rather than hidden. A query whose only
       matching term is one of four ("what did the CFO say on the earnings call"
       matching a dog *called* Biscuit) lands well below one that matched both of two.
    2. **Unboundedness.** IDF grows without limit as a term gets rarer. The saturating
       map `x / (x + h)` is monotone, fixes 0 at 0, and approaches 1, so "twice the
       BM25" becomes "somewhat more convinced" instead of "twice as relevant".

    >>> round(lexical_relevance(3.0, terms=1), 3)
    0.667
    >>> lexical_relevance(0.0, terms=3)
    0.0
    >>> lexical_relevance(9.9, terms=0)
    0.0
    """
    if terms <= 0 or bm25 <= 0.0:
        # No terms means the leg abstained; a non-positive BM25 means the row matched
        # nothing worth counting. Neither is evidence, and neither should divide.
        return 0.0
    per_term = bm25 / terms
    return per_term / (per_term + LEXICAL_HALF_SATURATION)


def vector_relevance(cosine: float) -> float:
    """Cosine similarity as absolute evidence, in [0, 1].

    Already absolute - both sides are unit vectors - so the only work is clamping the
    negative half. A vector pointing away from the query is not weak evidence for the
    claim, it is no evidence, and letting it go negative would make it subtract from
    the other retriever's finding.

    >>> vector_relevance(-0.4), vector_relevance(0.75)
    (0.0, 0.75)
    """
    return max(0.0, min(1.0, cosine))


def relevance(
    *,
    vector: float | None,
    lexical: float | None,
    w_vector: float,
    w_lexical: float,
    graph: float | None = None,
    w_graph: float = 0.0,
) -> float:
    """Blend the legs' absolute signals into one [0, 1] relevance.

    `None` means the leg did not run at all - the query embedded to a zero vector, or
    it carried no content terms, or the leg is weighted off. An abstaining leg is
    dropped from the average rather than scored as zero: it has no opinion, and
    averaging in an opinion it never held would halve the score of every result on a
    CJK query or an exact-identifier lookup, both of which are answered correctly by
    one leg alone.

    A leg that *did* run and simply did not return this claim contributes 0.0. That is
    the difference the parameter encodes, and it is what makes corroboration visible:
    two legs agreeing at 0.5 beat one leg alone at 0.5.

    `graph` is a path score from `GraphTraverser`, already an absolute [0, 1] relevance
    by construction (see `retrieve/spread.py`), and it abstains far more often than the
    other two: it does not run at all on a query the intent classifier routes away from
    it, on a store with no `adjacent`, or when the seeds reached nothing. Its default is
    the abstaining one, so a caller that predates the leg gets exactly the two-leg average
    it got before.

    >>> relevance(vector=0.5, lexical=0.5, w_vector=1.0, w_lexical=1.0)
    0.5
    >>> relevance(vector=0.6, lexical=None, w_vector=1.0, w_lexical=1.0)
    0.6
    >>> relevance(vector=0.6, lexical=0.0, w_vector=1.0, w_lexical=1.0)
    0.3

    The graph leg, in each of the three states it can be in. Ran and ranked this claim;
    ran and did not; did not run:

    >>> relevance(vector=0.6, lexical=0.6, graph=0.6,
    ...           w_vector=1.0, w_lexical=1.0, w_graph=1.0)
    0.6
    >>> round(relevance(vector=0.6, lexical=0.6, graph=0.0,
    ...                 w_vector=1.0, w_lexical=1.0, w_graph=1.0), 3)
    0.4
    >>> relevance(vector=0.6, lexical=0.6, graph=None,
    ...           w_vector=1.0, w_lexical=1.0, w_graph=1.0)
    0.6

    A weight of zero is the same as abstaining, and has to be: a leg nobody is counting
    must not divide the ones that are.

    >>> relevance(vector=0.6, lexical=0.6, graph=0.0,
    ...           w_vector=1.0, w_lexical=1.0, w_graph=0.0)
    0.6
    """
    total = 0.0
    weight = 0.0
    for signal, w in ((vector, w_vector), (lexical, w_lexical), (graph, w_graph)):
        if signal is None or w <= 0.0:
            continue
        total += w * signal
        weight += w
    if weight <= 0.0:
        return 0.0
    return total / weight


def normalized_score(
    relevance: float,
    *,
    recency: float,
    confidence: float,
    salience: float,
    w_recency: float,
    w_confidence: float,
    w_salience: float,
) -> float:
    """The number callers threshold on: absolute relevance, in [0, 1].

    `relevance` (the retriever-evidence blend above) is scaled by the quality
    multiplier divided by its own maximum, so quality can only ever pull a result
    *down* from its evidence, by at most `1 / (1 + w_recency + w_confidence +
    w_salience)` - a third at the default weights. Ordering within a query is
    therefore evidence first, freshness as the tiebreak, which is the intended
    priority and the one the shipped arithmetic had backwards.

    **Why this is not the fused score rescaled.** RRF is rank-relative by
    construction: whatever is best gets `1 / (k + 1)`, whether it answers the question
    or is merely the least bad thing in the store. Measured on the eval corpus, the
    shipped score gave `"what is my mother's maiden name?"` 0.0483 against 0.0474 for
    the best answerable query - the unanswerable one scored *higher*. No monotone
    rescale of that number can separate the two, because the information was destroyed
    when the ranks replaced the scores. So normalization is computed from the raw
    retriever outputs - cosine, and BM25 per content term - and the fused rank is kept
    for the candidate set and the explanation, where it is still exactly right.

    The raw fusion product remains available as `Explanation.raw_score`; it is the
    right number for debugging a ranking change and the wrong one for a threshold.

    The result is clamped at 1.0. Only a claim with perfect evidence on both legs and
    salience above 1.0 can reach the clamp, and there ties fall through to the id
    tiebreak like any other.

    >>> round(normalized_score(0.8, recency=1.0, confidence=1.0, salience=1.0,
    ...                        w_recency=0.25, w_confidence=0.15, w_salience=0.10), 3)
    0.8
    >>> round(normalized_score(0.8, recency=0.0, confidence=1.0, salience=1.0,
    ...                        w_recency=0.25, w_confidence=0.15, w_salience=0.10), 3)
    0.667
    """
    span = 1.0 + w_recency + w_confidence + w_salience
    boost = quality_boost(
        recency=recency,
        confidence=confidence,
        salience=salience,
        w_recency=w_recency,
        w_confidence=w_confidence,
        w_salience=w_salience,
    )
    return min(1.0, relevance * boost / span)
