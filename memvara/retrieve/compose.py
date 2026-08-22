"""Relation terms that name a chain rather than a stored predicate.

"Who is the maternal grandfather of X" is a two-hop question whose evidence is
`(X, mother, Y)` and `(Y, father, Z)`. It names no predicate the store holds. Every rule
in `intent.py` counts predicates the question says out loud, so this shape was invisible
to all of them: measured on 2WikiMultihopQA's `inference` family, the gate ran the walk on
none of the 1,549 and the leg was worth nothing there.

`grandfather` is not a synonym for `father`. It is `father` composed with `father`, and no
amount of string matching gets from one to the other. What is needed is the fact that the
term *is* a composition — not which predicates it composes from, because the gate's
question is only ever "is this a chain".

**The model is asked once, about a vocabulary, and never about a query.** `intent.py`
promises to be model-free and `hybrid.py` promises reproducible retrieval, and both would
be false if a search could block on an API call. So `acquire()` takes the predicates a
store uses and returns the derived terms over them; the read path does a set membership
test against the result. Same shape as `resolve_predicate` on the write side — pay once
per vocabulary, keep the answer, never pay again.

Measured with the terms supplied, `inference` at k=12 goes **52.6% → 86.4%** answer and
**49.0% → 83.8%** chain. That is the whole of the feature's value and it is why the
acquisition is worth a model call.

**Not persisted yet.** The terms live for the life of a `Memvara`, so a long-running
server pays once at startup and a short script pays once per run. Persisting them needs a
place to put them, and the two candidates are both wrong: `put_spec` stores predicates and
a derived term is not one, and a new `Store` method is a protocol change that breaks every
backend that has not grown it. `docs/ROADMAP.md` carries it.
"""

from __future__ import annotations

from typing import Any, Collection, Mapping, Protocol, Sequence, runtime_checkable

from .analyze import tokenize

#: Longest a derived term may be, in tokens. "Paternal grandfather" is two and
#: "great-great-grandmother" is one; a model that answers with a sentence is answering a
#: different question, and a long "term" would match a whole clause of an ordinary query.
MAX_TERM_TOKENS = 3


@runtime_checkable
class RelationComposer(Protocol):
    """An LLM that can name the derived relations over a set of predicates.

    Its own protocol rather than a third method on `LLM`, and the reason is a lesson from
    `Store`: adding a member to a `runtime_checkable` protocol makes every implementation
    that predates it fail `isinstance`, and a downstream backend finds out when its type
    checker does. `LLM` has two methods because the architecture calls a model as rarely
    as possible, and this is not one of the two.

    A backend that does not have this is not broken. `acquire()` returns nothing, the gate
    keeps the behaviour it had, and the questions this would have caught stay missed —
    which is the state every release before this one shipped in.
    """

    def compose_relations(self, predicates: Sequence[str]) -> Mapping[str, int]: ...


def acquire(llm: Any, predicates: Collection[str]) -> frozenset[str]:
    """Derived relation terms over `predicates`, or nothing if the backend cannot say.

    Returns only the terms that compose from **two or more** predicates, because that is
    the gate's question. A model that reports `stepmother: 1` is saying the store has a
    single predicate for it, in which case the ordinary predicate match already finds it
    and treating it as a chain would run the walk on a lookup.

    Everything the model returns is filtered rather than trusted:

    * a term of more than `MAX_TERM_TOKENS` tokens is a phrase, not a relation name, and
      would match a clause of an ordinary question;
    * a term that is itself one of the predicates is a contradiction — the model has said
      a slot name is a composition of slot names — and is dropped rather than reconciled;
    * an arity that is not an integer above one is not an answer to what was asked.

    Filtered rather than validated-and-raised because this is an optional enrichment on a
    path that works without it. A model having a bad day should cost the questions it
    would have helped, never the search that was going to succeed anyway.
    """
    compose = getattr(llm, "compose_relations", None)
    if compose is None:
        return frozenset()
    try:
        answer = compose(sorted(predicates))
    except Exception:
        # Including the network. See the docstring: an enrichment that raises into
        # `Memvara.__init__` would make an optional feature a startup dependency.
        return frozenset()
    if not isinstance(answer, Mapping):
        return frozenset()

    known = {p.lower() for p in predicates}
    out: set[str] = set()
    for term, arity in answer.items():
        if not isinstance(term, str) or isinstance(arity, bool):
            continue
        if not isinstance(arity, int) or arity < 2:
            continue
        folded = term.strip().lower()
        if not folded or folded in known:
            continue
        if len(tokenize(folded)) > MAX_TERM_TOKENS:
            continue
        out.add(folded)
    return frozenset(out)


def names_derived(query: str, terms: Collection[str]) -> bool:
    """Does this question name a relation that is a chain rather than a slot?

    Substring rather than token matching, deliberately and unusually for this package.
    A derived term is often hyphenated or possessive in the question — "father-in-law",
    "X's paternal grandmother" — and `tokenize` splits both, so a token test would miss
    the shapes this exists for. The terms are long and specific enough that a substring
    match does not collide the way a bare token index would: `uncle` is the shortest at
    five characters, and `MAX_TERM_TOKENS` keeps a term from being a clause.

    >>> names_derived("Who is the paternal grandfather of Reginald?", {"grandfather"})
    True
    >>> names_derived("Who is Marie's father-in-law?", {"father-in-law"})
    True
    >>> names_derived("Where was the director born?", {"grandfather", "uncle"})
    False
    """
    if not terms:
        return False
    low = query.lower()
    return any(term in low for term in terms)
