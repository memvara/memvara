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

from typing import TYPE_CHECKING, Callable, Iterable, Mapping

from ..entities import POSSESSIVE
from ..schema import word_stem
from .analyze import STOPWORDS, tokenize

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, annotation only
    from ..schema import PredicateRegistry


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
    spring summer autumn fall winter season seasons
    quarter quarters q1 q2 q3 q4 h1 h2
    used
""".split())

#: On the two ambiguous ones. `spring` is also a coil and `fall` is also a verb, which is
#: the objection to putting either in a list of time words. It is outweighed here: this
#: set only *weights* legs, so the cost of a false positive is a temporal leg voting on a
#: question that was not about time — and it abstains when nothing in scope is near the
#: anchor, which is exactly the case a misread `fall` produces. The cost of the false
#: negative is a plainly temporal question ("what was I working on last summer") routed
#: as a lookup and answered with the leg that ranks on *when* switched off. Months are
#: already here on the same reasoning, and `may` is at least as ambiguous as either.

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
#: store's own registry*, which is the shape of the answer: derive the signal from
#: `PredicateRegistry` rather than from a list here.
#:
#: That is now done, and this list is unchanged by it. `predicate_refs` counts how many
#: *distinct* predicates a question names, and two of them is a chain — one predicate is
#: a question about one slot. Deriving rather than extending is what keeps the classifier
#: off its own corpus: no word was added because a benchmark needed it, and the rule
#: applies to every predicate a store ever learns, including ones this repository has
#: never seen.
#:
#: The list stays because it catches what the registry cannot. "Who is Alice's manager"
#: names a relation in English and no predicate at all; `manager` is a role noun, not a
#: slot name. The two signals fail in opposite directions, which is why both run.
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
#: The pattern is `entities.POSSESSIVE`, so every apostrophe the entity fold recognises
#: counts here too.
_POSSESSIVE = POSSESSIVE

#: `between X and Y` — the one surface form that names two endpoints and asks for what
#: lies between them, which is `paths_between` written in English.
_BETWEEN_AND = re.compile(r"\bbetween\b.+\band\b", re.IGNORECASE)


#: Non-word characters collapse to spaces before a predicate phrase is looked for, so
#: "works at?" and "works-at" both match `works at`.
_WORDS = re.compile(r"[^a-z0-9]+")

#: Cache of predicate phrases per registry, keyed on how many predicates it has learned.
#: A registry grows at runtime — `learn()` adds a predicate the moment a store sees one —
#: so a plain memo would go stale silently. `learned_count` moves whenever it does.
_PHRASES: dict[tuple[int, int], frozenset[str]] = {}


def _phrases(registry: "PredicateRegistry") -> frozenset[str]:
    """Every predicate name and alias the registry knows, spelled the way a person would.

    `works_at` becomes `works at`, because a question says the second and the registry
    stores the first. Matched as phrases and never as tokens: `lives_in` splits into
    `lives` and `in`, and `in` appears in most English questions, so a token index would
    make almost every query look relational.
    """
    key = (id(registry), registry.learned_count)
    cached = _PHRASES.get(key)
    if cached is None:
        cached = frozenset(
            name.replace("_", " ")
            for spec in registry.all_specs()
            for name in (spec.name, *spec.aliases)
        )
        _PHRASES.clear()        # one registry per process in practice; bound the dict
        _PHRASES[key] = cached
    return cached


def predicate_refs(query: str, registry: "PredicateRegistry") -> set[str]:
    """The distinct predicates a question names, folded onto their canonical names.

    Two of them is the signal this exists for: one predicate is a question about one
    slot, which is a lookup, and two is a question that has to pass through one fact to
    reach another, which is what the graph leg is for. "Which city is the company Ada
    works at based in?" names `works_at` and `lives_in`, shares no word with
    `RELATIONAL_MARKERS`, and is exactly the shape the hand-written vocabulary misses.

    Folded before counting, so `based_in` and `located_in` are one predicate rather than
    two — otherwise a question that spelled the same relation twice would look like a
    chain.
    """
    return _named_in(query, _phrases(registry), registry.normalize)


def observed_refs(query: str, predicates: "Iterable[str]",
                  normalize: "Callable[[str], str] | None" = None) -> set[str]:
    """The same count, over a vocabulary that was observed rather than declared.

    `predicate_refs` reads `PredicateRegistry.all_specs()`, which lists what somebody
    *declared*. A predicate written straight through `remember()` is never declared —
    `registry.spec()` synthesizes an answer for it and the registry does not remember —
    so on a store whose vocabulary arrived that way the declared list is the 23 builtins
    and the count is always one or zero.

    Teaching the registry instead was tried and reverted. Recording an observed predicate
    means recording a cardinality, and the only cardinality available is the default; a
    store would then hold `MANY` chosen by nobody, and `memory_remember`'s note — "this
    store has no cardinality recorded for that predicate" — would stop firing because the
    sentence had been made false rather than because the question had been answered. That
    note is the only warning that two live values might be a contradiction. Buying a
    read-path signal by silencing it is the trade this function exists to avoid.

    So the vocabulary comes from the rows instead. Predicates in hand are observed facts
    about the store and commit it to nothing.
    """
    fold = normalize or (lambda name: name)
    spoken = {name.replace("_", " "): name for name in predicates}
    return _named_in(query, spoken, fold)


def _content(predicate: str) -> frozenset[str]:
    """The tokens of a predicate name that carry its meaning, without the joinery.

    `born_on` means "born"; the `on` is the preposition that attaches it to a date, and no
    question says it. Matching the whole phrase therefore missed "when was X born", which
    is how most people ask — 79% of the compositional questions in `bench/twowiki.py`, all
    of them naming a predicate the store held.

    Every token has to be present, which is what keeps this from collapsing into a token
    index: `country_of_citizenship` needs both `country` and `citizenship`, so a question
    mentioning a country does not thereby name the predicate.

    Stemmed, because a question inflects what a predicate name does not. `team_lead` is
    asked as "who *leads* the team", `deploy_region` as "where is it *deployed*", `owned_by`
    as "the team that *owns* it" — and on the Agent Memory Benchmark four of the six
    chained questions named their second predicate only in a form the exact match could
    not see, so the gate read each of them as a question about one slot. The fold is
    `schema.word_stem`, the same one the registry uses to decide that `employer` and
    `employed_by` are one predicate, applied to both sides so it only has to agree with
    itself.
    """
    return frozenset(word_stem(t) for t in tokenize(predicate.replace("_", " "))
                     if t not in STOPWORDS)


def _named_in(query: str, spoken: "Mapping[str, str] | Iterable[str]",
              fold: "Callable[[str], str]") -> set[str]:
    """Which of these predicates the question says, folded and deduplicated.

    Content tokens rather than the whole phrase, and *all* of them rather than any. The
    first is what lets "when was X born" name `born_on`; the second is what stops
    `lives_in` being named by every question containing `in`. A bare token index would
    read almost every query as naming several predicates — the opposite failure to the one
    this fixes, and visible only as latency.
    """
    tokens = {word_stem(t) for t in tokenize(query)}
    pairs = (spoken.items() if isinstance(spoken, Mapping)
             else ((phrase, phrase.replace(" ", "_")) for phrase in spoken))
    # One entry per *thing the question said*, not per predicate that answers to it.
    # `born_in` and `born_out` both reduce to `born`, so "when was Alice born" matched two
    # predicates and read as a chain — from one word. Which of them the caller sees is
    # settled on the name so two stores holding the same schema agree.
    best: dict[frozenset[str], str] = {}
    for _phrase, name in pairs:
        parts = _content(name)
        if parts and parts <= tokens:
            folded = fold(name)
            if folded < best.get(parts, folded + "\uffff"):
                best[parts] = folded
    return set(best.values())


def is_comparison(query: str) -> bool:
    """Is this about which of two named things, rather than about a chain?

    A disjunction between alternatives is a comparison frame. "Which film came out
    earlier, A or B" is answered by looking both up and ranking them; there is no join
    between the halves to walk, and the evidence for one says nothing about the other.

    It matters because a predicate count cannot tell the two apart. "Which film has the
    director died later, A or B" names `director` and `died_on` — two predicates, a chain
    by that measure — while being two independent lookups whose answers are compared.
    Measured on 2WikiMultihopQA, running the walk on that family costs 13.7 points,
    because it spends `k` on a hub's neighbours instead of on the second half.

    The disjunction, and not a list of comparative words. "Earlier", "first" and "younger"
    are what that corpus happens to use, and a rule built from them would be fitted to it.
    The disjunction is the structure underneath and holds for phrasings nobody wrote down.

    >>> is_comparison("Which film came out earlier, The Wrong Box or Soft Shoes?")
    True
    >>> is_comparison("When was Britannicus's father born?")
    False
    """
    return " or " in f" {query.lower().strip()} "


def classify(query: str, registry: "PredicateRegistry | None" = None) -> Intent:
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
    if _about_time(tokens):
        return Intent.TEMPORAL
    if _about_a_relation(query, tokens, registry):
        return Intent.RELATIONAL
    if tokens & LOOKUP_MARKERS:
        # A pointed question with no time and no relation in it. One row answers it, and
        # the two lookup legs are what find that row.
        return Intent.LOOKUP
    return Intent.OPEN


def is_relational(query: str, registry: "PredicateRegistry | None" = None) -> bool:
    """Would this query be `RELATIONAL`, if time were not being asked about as well?

    `classify` returns one label and time outranks relation, so a question that is about
    both — "who *currently* leads the team that owns the checkout service" — reads as
    `TEMPORAL`, and that row's multipliers switch the graph leg off. This is the second
    reading, kept available so the retriever can honour both: the temporal row still
    decides the other legs, and the walk runs because the question is also a chain. It
    is the same repair `HybridRetriever._weights` already made for a caller who named an
    instant as an argument rather than in words.

    >>> is_relational("who currently leads the team that owns the checkout service?")
    False
    >>> is_relational("who is my manager's manager?")
    True

    The first is `False` without a registry because nothing in it is in the relational
    vocabulary: "leads" and "owns" name predicates, and only a registry can say so.
    """
    return _about_a_relation(query, set(tokenize(query)), registry)


def _about_time(tokens: set[str]) -> bool:
    return bool(tokens & TEMPORAL_MARKERS) or any(_YEAR.match(t) for t in tokens)


def _about_a_relation(query: str, tokens: set[str],
                      registry: "PredicateRegistry | None") -> bool:
    return bool(
        tokens & RELATIONAL_MARKERS or len(_POSSESSIVE.findall(query)) > 1
        or _BETWEEN_AND.search(query)
        or (registry is not None and not is_comparison(query)
            and len(predicate_refs(query, registry)) > 1))


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


#: The fourteen phrases that mark a question as being about what the assistant said. The
#: list came out of the LongMemEval work (docs/superpowers/plans/2026-09-03-intent-routed-
#: selection-offline.md on the benchmark branch): on that data the gold turn is a user turn
#: for 94% of questions, and an assistant turn almost only when the question asks what the
#: assistant suggested, recommended, wrote or said. Whole words, any case.
ASSISTANT_PHRASES: tuple[str, ...] = (
    "you suggested", "you recommended", "you mentioned", "you told me", "you provided",
    "you wrote", "you created", "did you say", "can you remind me", "remind me what",
    "remind me which", "remind me who", "remind me how", "remind me of",
)
_ASSISTANT = re.compile("|".join(r"\b" + re.escape(p) + r"\b" for p in ASSISTANT_PHRASES),
                        re.IGNORECASE)


def routed_role(query: str) -> str:
    """Which role's turns a ranked read hands the selector: `"assistant"` when the
    question asks what the assistant said, else `"user"`.

    Model-free, like everything in this module, and for the same reason: it runs on
    every ranked read, in front of the one model call that read makes. It assumes the
    `assistant` role holds a model's turns; a retriever whose two roles are two people
    turns it off with `route_roles=False` (`Memvara(read_route_roles=False)`). It exists
    because the selector's candidate list is short — `rerank_top_n` turns of which it
    sees `top_n` — and assistant turns are long and numerous, so with both roles in the
    list they take the slots the answer-bearing user turns needed. Measured on the
    shipped path without this rule, gold-turn recall was 0.808 against 0.912 with it
    (design spec §6, check 1).

    >>> routed_role("What did you recommend for the trip?")
    'user'
    >>> routed_role("You recommended a hotel — which one?")
    'assistant'
    >>> routed_role("Remind me what you said about the deadline")
    'assistant'
    >>> routed_role("youtold me nothing")
    'user'
    """
    return "assistant" if _ASSISTANT.search(query) else "user"
