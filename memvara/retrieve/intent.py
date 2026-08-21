"""What kind of question is this, decided without a model.

A retriever with three legs has a cost problem the two-leg version did not: the graph leg
walks the store, and most queries have nothing for it to walk to. "What is my name" is
answered by one row; paying a two-hop expansion out of five entities to confirm that is
latency spent on a question that was already answered, and — worse than the latency — it
puts a real zero on the graph leg for every candidate the walk did not reach, which drags
down the row that *was* the answer. Gating is therefore not an optimization. It is what
keeps the third leg from making the easy questions worse.

**Model-free, and it has to be.** A classifier on the read path that calls an LLM makes
every search cost a model call, which is the property this library exists to avoid, and
makes retrieval non-reproducible, which is the property `hybrid.py`'s docstring promises.
So this is a marker vocabulary over the query's own tokens: closed-class words, a handful
of open-class relation nouns, and two shapes (a possessive chain, `between X and Y`).

**It reads the raw tokens, not `analyze()`'s terms**, and that is the one thing worth
being careful about here. `analyze` drops the stopwords, which is exactly right for BM25
and exactly wrong for this: `when`, `before`, `who`, `whose`, `between` are all closed-class
and all in `STOPWORDS`, and they are precisely the words that say what kind of question is
being asked. The two functions read the same tokenizer and want opposite halves of it.

**What it is not.** It is not an entity extractor and does not try to be — the graph leg's
seeds come from the head of the fused list, for the reasons in `spread.py`. It does not
parse dates. It does not know what the store contains. Given a query it cannot place, it
returns `Intent.OPEN`, whose multipliers are the neutral ones, so the failure mode of not
recognising something is the retriever's previous behaviour rather than a wrong route.
"""

from __future__ import annotations

import re
from enum import Enum

from .analyze import tokenize


class Intent(str, Enum):
    """The four question shapes retrieval weighs differently.

    Named to match the categories LOCOMO reports separately, which is deliberate: a
    per-category benchmark table is the only way to set a per-intent multiplier from a
    measurement rather than from taste, and a classification scheme whose classes do not
    line up with any instrument cannot be tuned at all.

    A `str` enum, so an `Explanation` carrying one serialises into a prompt, a JSON
    payload or a tool result as the word rather than as `<Intent.LOOKUP: 'lookup'>`.
    """

    #: One fact answers it. "What is my name", "where do I live", "what plan am I on".
    LOOKUP = "lookup"
    #: The answer depends on *when*. "What did I do before June", "how long have I…".
    TEMPORAL = "temporal"
    #: The answer is a relation, or is reached through one. "Who does Alice report to",
    #: "where is my manager's employer based", "how are Acme and Tallinn connected".
    RELATIONAL = "relational"
    #: Everything else, including everything this classifier cannot place.
    OPEN = "open"


#: Words that make a question about time. Closed-class wherever possible: `before`,
#: `after`, `since`, `until`, `when` are prepositions and conjunctions whose membership
#: the grammar fixes, so no corpus can make them mean something else.
#:
#: The open-class members are the calendar — month names, weekday names, and the units —
#: and they are here because a query naming one is asking about time whatever else it is
#: doing. `now`, `still`, `currently` and `already` are the aspectual markers that
#: separate "what plan am I on now" from "what plan am I on", which is the distinction
#: `valid_at` exists for.
#:
#: Deliberately absent: `then`, which is far more often a discourse connective ("so then
#: what") than a time reference, and `first`/`last`, which are ordinals over anything.
TEMPORAL_MARKERS: frozenset[str] = frozenset("""
    when whenever while during before after since until till ago
    yesterday today tomorrow tonight
    now currently still already recently lately earlier later previously originally
    latest newest oldest recent previous former
    date dates time times year years month months week weeks day days hour hours
    decade decades century
    january february march april may june july august september october november december
    jan feb mar apr jun jul aug sep sept oct nov dec
    monday tuesday wednesday thursday friday saturday sunday
    morning afternoon evening night weekend
    used
""".split())

#: A four-digit token in the range a stored date could plausibly fall in. Narrow on
#: purpose: `1999` and `2026` are years, and `4419` is a serial number off a power brick.
_YEAR = re.compile(r"^(?:19|20|21)\d\d$")

#: Words that make a question relational. Two kinds, and both are needed.
#:
#: The closed-class half is `whose`, `between`, `via` and `through`. `who` is
#: deliberately **not** among them: "who did I meet on Tuesday" names a person and wants
#: one row, and `who` is far too common in a personal store to be the thing that switches
#: a walk on. It is a lookup marker, and a `who` question becomes relational the moment
#: anything below appears beside it — "who is my *manager*".
#:
#: The open-class half is a small vocabulary of relation nouns and the kinship/org terms
#: that only ever appear in a question about a relation. It is short and it stays short:
#: every word added here is a query routed into a walk, and the cost of being wrong is
#: paid on every query that contains it.
#:
#: **It is measurably too narrow, and widening it by hand is the wrong fix.** On
#: `bench/multihop.py` this list routes two of the three question families past the walk —
#: "which city is the company X works at based in" and "who founded the company that X
#: works at" contain no word in it — so the shipped configuration scores what plain
#: `search` scores and the leg's whole gain is gated away (`docs/BENCHMARKS.md`). Both
#: `works at` and `founded` are relations by any reading, and both are *predicates in the
#: store's own registry*, which is the shape of the answer: derive the markers from
#: `PredicateRegistry` rather than from a list here. That is not done, because adding
#: those two words because a benchmark this repository wrote needs them is how a
#: classifier gets fitted to its own corpus, and the next person would have no way to
#: tell which entries were reasoned and which were retrofitted.
RELATIONAL_MARKERS: frozenset[str] = frozenset("""
    whose between via through connect connects connected connection connections
    relate relates related relation relations relationship relationships
    link links linked linking
    manager managers boss bosses report reports reporting supervisor
    colleague colleagues coworker coworkers employer employers employee employees
    founder founders owner owners parent parents partner partners
    both mutual shared common same
""".split())

#: The interrogatives whose answer is a *value*: a name, a place, a thing, a person.
#: Their presence is what separates a pointed question from an open-ended request, and
#: that is the whole of the `lookup` / `open` distinction.
#:
#: `why` and `how` are deliberately absent. "Why did I switch" and "how did that go" are
#: not answered by one row, and routing them as lookups would gate away the leg that fans
#: out from what the lookup legs found — which, on a question with no single answer, is
#: the only part of the search doing anything useful.
#: `when` and `whose` are absent because they are decided earlier — the first is
#: temporal and the second relational — so listing them here would suggest a race that
#: the ordering in `classify` has already settled.
LOOKUP_MARKERS: frozenset[str] = frozenset("""
    what whats which where who whom
""".split())

#: A possessive chain: two `'s` in one query, as in "my manager's employer's office".
#: One possessive is ordinary ("what is my manager's name"); two is a join, spelled out.
_POSSESSIVE = re.compile(r"['’]s\b")

#: `between X and Y` — the one surface form that names two endpoints and asks for what
#: lies between them, which is `paths_between` written in English.
_BETWEEN_AND = re.compile(r"\bbetween\b.+\band\b", re.IGNORECASE)


def classify(query: str) -> Intent:
    """The query's shape, as one of four intents. Deterministic and model-free.

    Order matters and encodes a priority rather than a taxonomy. A question can be both
    temporal and relational — "who was my manager in 2023" is — and only one route can be
    taken, so the classes are checked in the order of what the retriever can most usefully
    do about them. Time comes first because a wrong answer from the wrong instant is
    wrong in a way no amount of extra recall repairs, while a missed hop is an answer that
    is merely incomplete.

    >>> classify("what is my name?")
    <Intent.LOOKUP: 'lookup'>
    >>> classify("where did I live before I moved to Lisbon?")
    <Intent.TEMPORAL: 'temporal'>
    >>> classify("who does my manager report to?")
    <Intent.RELATIONAL: 'relational'>
    >>> classify("how are Acme and Tallinn connected?")
    <Intent.RELATIONAL: 'relational'>

    An open-ended request names no value to look up, and is the one shape where fanning
    out from whatever the lookup legs found can add something:

    >>> classify("tell me about the project")
    <Intent.OPEN: 'open'>
    >>> classify("why did I switch?")
    <Intent.OPEN: 'open'>

    `who` on its own is a lookup, because most of the time it is one — "who did I meet on
    Tuesday" names a person and wants a row. It becomes relational beside anything in the
    relational vocabulary:

    >>> classify("who founded Acme?")
    <Intent.LOOKUP: 'lookup'>
    >>> classify("who is my manager?")
    <Intent.RELATIONAL: 'relational'>

    A single possessive is ordinary English; two in one query is a join spelled out.
    Neither example names a relation, so the possessive count is what decides them:

    >>> classify("what is my sister's name?")
    <Intent.LOOKUP: 'lookup'>
    >>> classify("what is my sister's school's address?")
    <Intent.RELATIONAL: 'relational'>

    A query with no content at all is `OPEN` rather than an error. The lexical leg
    abstains on it and the vector leg abstains on it, and this must not be the one stage
    that raises:

    >>> classify("")
    <Intent.OPEN: 'open'>
    >>> classify("???")
    <Intent.OPEN: 'open'>
    """
    tokens = set(tokenize(query))
    if tokens & TEMPORAL_MARKERS or any(_YEAR.match(t) for t in tokens):
        return Intent.TEMPORAL
    if (tokens & RELATIONAL_MARKERS or len(_POSSESSIVE.findall(query)) > 1
            or _BETWEEN_AND.search(query)):
        return Intent.RELATIONAL
    if tokens & LOOKUP_MARKERS:
        # A pointed question with no time and no relation in it. One row answers it, and
        # the two lookup legs are what find that row.
        return Intent.LOOKUP
    return Intent.OPEN


#: Per-intent multipliers on the retriever's configured leg weights.
#:
#: **Every entry is 1.0 until a per-category sweep moves it**, which is the rule this
#: table exists to enforce rather than a placeholder. A multiplier picked because it
#: sounds right is a ranking change with no evidence behind it and no way to argue about
#: it afterwards; `docs/BENCHMARKS.md` carries the sweep each non-neutral number came
#: from, and a category that did not improve keeps its 1.0 and says so there.
#:
#: The `graph` and `temporal` columns are the ones that are not neutral, and their zeroes
#: are gates rather than weights: they are checked before the walk or the time query runs,
#: so a query pays nothing rather than paying for work whose result is multiplied away.
#: See `HybridRetriever._graph_search` and `._episode_time_search`.
#:
#: The two are never on together, and the table is where that is decided. `relational` is
#: a question about a join and `temporal` a question about an instant; `lookup` is neither
#: and skips both; `open` is the one shape that could be either, so it runs both and lets
#: fusion pick. Turning both on for a `temporal` query would have the graph leg zeroing
#: every claim it did not reach on a question that was never about a chain.
MULTIPLIERS: dict[Intent, dict[str, float]] = {
    Intent.LOOKUP: {"vector": 1.0, "lexical": 1.0, "graph": 0.0, "temporal": 0.0},
    Intent.TEMPORAL: {"vector": 1.0, "lexical": 1.0, "graph": 0.0, "temporal": 1.0},
    Intent.RELATIONAL: {"vector": 1.0, "lexical": 1.0, "graph": 1.0, "temporal": 0.0},
    Intent.OPEN: {"vector": 1.0, "lexical": 1.0, "graph": 1.0, "temporal": 1.0},
}


def weights(intent: Intent, *, vector: float, lexical: float, graph: float,
            temporal: float) -> tuple[float, float, float, float]:
    """The configured weights, scaled for this intent.

    Multipliers rather than replacements, so a deployment that has tuned `w_vector` keeps
    its tuning and gets the intent's *relative* shift applied on top of it. A table of
    absolute weights would silently discard every configured value, which is the failure
    mode of every "smart routing" layer that has to be turned off in production.

    >>> weights(Intent.LOOKUP, vector=1.0, lexical=1.0, graph=0.6, temporal=0.4)
    (1.0, 1.0, 0.0, 0.0)
    >>> weights(Intent.RELATIONAL, vector=1.0, lexical=1.0, graph=0.6, temporal=0.4)
    (1.0, 1.0, 0.6, 0.0)
    >>> weights(Intent.TEMPORAL, vector=1.0, lexical=1.0, graph=0.6, temporal=0.4)
    (1.0, 1.0, 0.0, 0.4)
    """
    m = MULTIPLIERS[intent]
    return (vector * m["vector"], lexical * m["lexical"], graph * m["graph"],
            temporal * m["temporal"])
