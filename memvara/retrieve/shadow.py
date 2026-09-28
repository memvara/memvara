"""A narrower scope's own value hides a broader one, for single-valued facts.

A reader's scope has a chain of levels it reads from, narrowest first
(`Scope.ancestors`): its session, its agent, its project, the same session and agent
outside the project, and the user. Each level can hold its own value for one question.
A value written inside a repository, in a session or by an agent is a local value: it
does not end the broader value, because that value is still the answer everywhere else.
The reconciler therefore ends only values at exactly the writer's scope
(`Reconciler._at_scope`).

So both stay stored, and the choice is made when reading. A present-tense read leaves out
a live claim when a narrower level of the reader's own chain holds a live claim for the
same owner, subject and single-valued predicate. The first level in the chain that holds
a value answers. The comparison stays inside one owner, tenant plus user, as the slots
do, so the tenant's own level is never hidden by a user's value.

Reads at another instant, and many-valued predicates, are unaffected: at another instant
the narrower level may not have had its value yet, and for a many-valued predicate both
values hold at once, so hiding one would lose a fact. A value a sibling session, agent or
project holds is not in the reader's chain, so it hides nothing from this reader.

Nothing is written. Once the narrower value ends, the broader one answers again.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ..schema import PredicateRegistry
from ..store import Store
from ..types import Claim, Scope, fact_key_for, owner_key

__all__ = ["shadowed"]


def shadowed(store: Store, registry: PredicateRegistry, claims: Iterable[Claim],
             scope: Scope) -> frozenset[str]:
    """The ids among `claims` that a present-tense read at `scope` must leave out.

    Pass the claims the read would otherwise return, after its cheap filters, because
    each distinct candidate slot is part of a lookup. A candidate at level i of the
    reader's chain is hidden when a level before i with the same owner holds a live claim
    in the candidate's slot, as that level spells the slot: a level with a project uses
    the project's key, one without uses the key with no project.

    The lookups are one `Store.occupied_slots` call for each level of the chain at which
    a candidate sits, asking only about the levels before it; a candidate at the reader's
    own level, or with no narrower level of its owner, costs nothing. A store without
    `occupied_slots` is asked `competing_claims` once per slot, and a store that raises
    `NotImplementedError` from the lookup it has leaves the read unshadowed rather than
    failing it.

    The narrower levels come from `Scope.ancestors`, which builds them with
    `stored_scope`, so a scope read back from a store with '*' or '' at some level gets
    its chain like any other rather than raising.
    """
    owner = owner_key(scope)
    # The reader's chain without the levels of another owner: the tenant's own level,
    # which has no user, is compared with nothing.
    mine = [level for level in scope.ancestors() if owner_key(level) == owner]
    if len(mine) < 2:
        # A reader bound to its user alone has no narrower level, and pays nothing.
        return frozenset()
    rank = {level: i for i, level in enumerate(mine)}
    # For each rank at which a candidate sits: the slot keys to ask about, by candidate.
    asked: dict[int, dict[str, set[str]]] = {}
    for claim in claims:
        if claim.state != "live":
            continue
        # None for a claim of another owner, and 0 for one at the reader's own level:
        # neither has a narrower level of its owner to be hidden by.
        at = rank.get(claim.scope)
        if not at:
            continue
        spec = registry.spec(claim.predicate)
        if not spec.functional:
            continue
        # A predicate declared global is never written with a project, so a level with a
        # project cannot hold a value for it.
        keys = {fact_key_for(level, claim.subject_key, claim.predicate)
                for level in mine[:at] if spec.project_scoped or level.project is None}
        if keys:
            asked.setdefault(at, {})[claim.id] = keys
    hidden: set[str] = set()
    for at, by_claim in asked.items():
        wanted = set().union(*by_claim.values())
        occupied = _occupied(store, scope.tenant, wanted, mine[:at])
        if occupied is None:
            return frozenset()
        hidden.update(cid for cid, keys in by_claim.items() if keys & occupied)
    return frozenset(hidden)


def _occupied(store: Store, tenant: str, keys: set[str],
              scopes: Sequence[Scope]) -> set[str] | None:
    """Which of `keys` hold a live claim stored at one of `scopes`, by whichever lookup
    this store offers, or `None` when it cannot say."""
    batched = getattr(store, "occupied_slots", None)
    try:
        if batched is not None:
            return set(batched(tenant, keys, scopes=scopes))
        at = {s.key() for s in scopes}
        return {key for key in keys
                if any(c.scope.key() in at for c in store.competing_claims(tenant, key))}
    except NotImplementedError:
        return None
