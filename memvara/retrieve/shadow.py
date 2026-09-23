"""A repository's own value hides the user-wide one, for single-valued facts.

A fact written without a project is user-wide: it is read from every repository. A fact
written inside a repository occupies that repository's slot, whose key includes the
project, so writing it does not end the user-wide value. Ending it would be wrong, because
the user-wide value is still the answer in every other repository.

So both stay stored, and the choice is made when reading. A present-tense read bound to a
project leaves out a live user-wide claim when the same owner, subject and single-valued
predicate also has a live claim in that project's slot. Reads with no project, reads bound
to another project, reads at another instant, and many-valued predicates are unaffected:
for a many-valued predicate both values hold at once, and hiding one would lose a fact.

Nothing is written. Once the repository's own value ends, the user-wide one answers again.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from ..schema import PredicateRegistry
from ..store import Store
from ..types import Claim, Scope, fact_key_for

__all__ = ["shadowed"]


def shadowed(store: Store, registry: PredicateRegistry, claims: Iterable[Claim],
             scope: Scope) -> frozenset[str]:
    """The ids among `claims` that a present-tense read at `scope` must leave out.

    Costs one indexed slot count per distinct slot among the candidates that could be
    shadowed, which are the live, user-wide claims of single-valued, project-relative
    predicates. A read with no project costs nothing.
    """
    project = scope.project
    if project is None:
        return frozenset()
    occupied: dict[str, bool] = {}
    hidden: set[str] = set()
    for claim in claims:
        if claim.scope.project is not None or claim.state != "live":
            continue
        spec = registry.spec(claim.predicate)
        # A predicate declared global is always written without a project, so the
        # project slot can never hold a value for it and there is nothing to ask.
        if not spec.functional or not spec.project_scoped:
            continue
        key = fact_key_for(replace(claim.scope, project=project), claim.subject_key,
                           claim.predicate)
        if key not in occupied:
            occupied[key] = store.count_competing(claim.scope.tenant, key) > 0
        if occupied[key]:
            hidden.add(claim.id)
    return frozenset(hidden)
