"""What tied a result to the question: the entity it names, or nothing.

Retrieval always returns something. Asked "where does Oscar live" of a store that has
never heard of Oscar, the vector leg finds the nearest `lives_in` row and the lexical leg
finds the rows containing "live", and fusion ranks the best of them first at a score that
looks like every other answer. `min_score` is the published lever against that and it
needs a number nobody has on day one — `calibrate.py` records why no constant is right,
and the Agent Memory Benchmark records the case no constant can reach at all: "which
region is Project Chronos deployed to" scores 0.450 against a store that holds Project
Atlas, above two genuine answers at 0.410 and 0.417. It looks exactly like a question
the store can answer, differing only in an entity that does not exist.

This module reads that difference off the rows instead of off the score. A claim is
**anchored** when the question names one of its ends — the entity it is about, or the
value it asserts — and **derived** when the graph leg reached it by walking out of an
anchored claim. A result that is neither surfaced on vocabulary alone, and that is the
whole of what "the store knows nothing about this" looks like from inside a ranker: not
a low score, but a top hit about somebody else.

The match is on the folded entity keys the write path already stamped (`Claim.subject_key`,
`Claim.object_key`), spelled the way `entity_key` spells them, against the question folded
the same way. No entity extraction runs over the query: the candidates supply the
entities, exactly as the graph leg's seeds do (`spread.py`), and the question is only
asked whether it contains them. Every content token of a key has to be present —
"Project Chronos" contains `project` and not `atlas` — which is what keeps a shared first
word from anchoring a sibling.

Two things a key comparison cannot see are handled here and nowhere else. A first-person
statement is stored under the subject `user` and asked about with a pronoun, so the self
subject is anchored by `I`, `my` and the rest of the first person. And a possessive is a
mention: "Bob's employer" names Bob, and the fold would otherwise read it as an entity
called `bobs`.
"""

from __future__ import annotations

from typing import Callable, Iterable

from ..entities import POSSESSIVE, entity_key, key_words
from ..types import SELF_SUBJECT, Claim

#: The three ways a result can be tied to the question, as `Explanation.anchor` reports
#: them. `None` is the fourth state and the finding: nothing in the question named
#: either end of this claim, and no walk led to it from a claim that was named.
SUBJECT = "subject"
OBJECT = "object"
PATH = "path"

#: How a question refers to the self subject (`types.SELF_SUBJECT`, the row a
#: first-person statement is filed under). First person only. "Do you know where Oscar
#: lives" addresses the agent, not the user, and `us` is also a country — both were in
#: this set once and anchored every `user` row on exactly the questions the filter exists
#: to return nothing for. An agent asking "what does the user prefer" names the row by
#: its key, which needs no pronoun.
SELF_PRONOUNS: frozenset[str] = frozenset("""
    i me my mine myself we our ours ourselves
""".split())

#: Every spelling a stored entity key answers to. The registry supplies learned aliases;
#: a retriever built without one gets the key alone.
Spellings = Callable[[str], Iterable[str]]


def query_tokens(query: str) -> frozenset[str]:
    """The question, folded the way an entity key is folded, as a set of tokens.

    >>> sorted(query_tokens("In which city is Bob's employer headquartered?"))
    ['bob', 'city', 'employer', 'headquartered', 'in', 'is', 'which']
    >>> query_tokens("???")
    frozenset()
    """
    return frozenset(entity_key(POSSESSIVE.sub("", query)).split())


def anchor_of(claim: Claim, tokens: frozenset[str],
              spellings: Spellings = lambda key: (key,)) -> str | None:
    """Which end of `claim` the question named, or `None` for neither.

    The subject is checked first, so a claim named at both ends reports `subject`: that
    is the end a question is *about*, and the object is the value it is asking for.

    >>> tokens = query_tokens("Which region is Project Atlas deployed to?")
    >>> anchor_of(Claim(subject="Project Atlas", predicate="deploy_region",
    ...                 object="eu-west-1"), tokens)
    'subject'
    >>> anchor_of(Claim(subject="alice", predicate="works_on",
    ...                 object="Project Atlas"), tokens)
    'object'
    >>> anchor_of(Claim(subject="Project Chronos", predicate="deploy_region",
    ...                 object="us-east-1"), tokens) is None
    True

    Every token of a key has to be there. `Project Atlas` is not named by a question
    that says only "project":

    >>> anchor_of(Claim(subject="Project Atlas", predicate="deploy_region",
    ...                 object="eu-west-1"),
    ...           query_tokens("Which region is Project Chronos deployed to?")) is None
    True

    The self subject is named by a pronoun, and a possessive is a mention:

    >>> anchor_of(Claim(subject="user", predicate="lives_in", object="Lisbon"),
    ...           query_tokens("where do I live?"))
    'subject'
    >>> anchor_of(Claim(subject="Bob", predicate="works_at", object="Globex"),
    ...           query_tokens("In which city is Bob's employer headquartered?"))
    'subject'

    `spellings` widens a key onto the other names the owner has learned for it, so a
    claim filed under `ibm` is anchored by a question that says "Big Blue":

    >>> anchor_of(Claim(subject="IBM", predicate="based_in", object="Armonk"),
    ...           query_tokens("where is Big Blue based?"),
    ...           spellings=lambda key: (key, "big blue") if key == "ibm" else (key,))
    'subject'
    """
    for end, key in ((SUBJECT, claim.subject_key), (OBJECT, claim.object_key)):
        if not key:
            # An empty object is how a retraction clears a slot; it names nothing.
            continue
        if end == SUBJECT and key == SELF_SUBJECT and tokens & SELF_PRONOUNS:
            return end
        for spelling in spellings(key):
            # `key_words` leaves out the digest that finishes a bounded key; a question
            # that repeats a long value names it by its words, never by the digest.
            parts = key_words(spelling)
            if parts and tokens.issuperset(parts):
                return end
    return None
