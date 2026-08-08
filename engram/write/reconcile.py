"""The contradiction engine.

mem0 resolves conflicts by embedding the new fact, pulling the top-k nearest existing
memories, and asking an LLM for ADD / UPDATE / DELETE / NOOP. Three consequences follow
from that, all bad: the conflicting memory has to make the top-k or the contradiction is
never seen, every write pays for a model call, and the same pair of facts can resolve
differently on two runs.

This module replaces all of it with one indexed lookup. `store.competing_claims` is keyed
on (tenant, fact_key) where `fact_key = hash(tenant, subject, predicate)`, so it returns
*every* live claim occupying the slot the candidate wants — not the ones that happened to
score well. Whether occupying the same slot is a conflict is then a schema question the
`PredicateRegistry` already answers. No embeddings, no top-k cliff, no model, and the
same inputs produce the same result on every run.

Two invariants the implementation is built around:

* Nothing is deleted. Superseding sets `invalidated_at`, `invalidated_by` and `valid_to`,
  so `as_of` still returns what we believed at any past instant.
* Unknown predicates are `Cardinality.MANY`. Keeping two competing facts degrades
  ranking; retiring a true one destroys information. Only errors of the first kind are
  recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from ..schema import PredicateRegistry
from ..store.base import Store
from ..types import Claim, fact_key_for, owner_key, utcnow


@dataclass(slots=True)
class ReconcileResult:
    action: str                  # "add" | "reinforce" | "supersede" | "retract" | "noop"
    claim: Claim | None          # the stored/updated claim
    invalidated: list[Claim] = field(default_factory=list)  # claims this one retired


class Reconciler:
    """Decides what a candidate claim does to the claims already on record."""

    def __init__(self, store: Store, registry: PredicateRegistry) -> None:
        self.store = store
        self.registry = registry
        # Set by `WritePipeline` from its own `reinforce_bump`; kept as an attribute
        # rather than a constructor argument so the documented signature stays exact.
        self.reinforce_bump = 0.25
        # Salience is unbounded upward otherwise, and a fact repeated in a loop would
        # eventually outrank everything else forever.
        self.max_salience = 5.0

    # -- public ---------------------------------------------------------------

    def apply(self, claim: Claim, *, now: datetime | None = None) -> ReconcileResult:
        t = now or utcnow()
        self._canonicalize(claim)
        if claim.recorded_at > t:
            # Transaction time is when *we* commit to believing it, which is `t` by
            # definition. A clock read taken when the Claim was constructed can land
            # microseconds ahead of the batch instant, and a claim recorded "in the
            # future" fails its own `is_live(t)` check — so it would silently neither
            # reinforce nor supersede anything. Backdating (t in the past) stays legal
            # for replays and imports.
            claim.recorded_at = t

        # A positive assertion needs all three parts to mean anything. "user likes
        # <nothing>" would occupy a slot, surface in `recall()`, and answer no question.
        # Empty objects stay meaningful for retraction, where they mean "clear the whole
        # slot", so the object guard is deliberately positive-only.
        if not claim.subject or not claim.predicate:
            return ReconcileResult("noop", None, [])
        if claim.polarity > 0 and not claim.object:
            return ReconcileResult("noop", None, [])

        tenant = claim.scope.tenant
        owner = owner_key(claim.scope)

        # 1. Exact duplicate: the same assertion is already live. Re-observation is
        #    evidence, not a new fact.
        if claim.polarity > 0:
            live_same = self._live(self.store.find_by_value(tenant, claim.value_key), t, owner)
            if live_same:
                keep = self._canonical_of(live_same)
                return ReconcileResult("reinforce", self.reinforce(keep, claim.sources, t), [])

        # 2. Retraction: the user is taking something back.
        if claim.polarity < 0:
            return self._retract(claim, t, owner)

        # 3. Conflict, then 4. accumulate.
        superseded, newer = self._victims(claim, t, owner)
        if newer:
            # This claim is history: something already on record was true *later*. Close
            # its valid interval where the next value begins, so it is retrievable via
            # `as_of` and `history` but never answers a present-tense question. It is not
            # invalidated — we still believe it, it simply stopped being true.
            boundary = min(c.valid_from for c in newer)
            if claim.valid_to is None or claim.valid_to > boundary:
                claim.valid_to = boundary
        self.store.put_claim(claim)
        if superseded:
            self._retire(superseded, t, claim.id)
            return ReconcileResult("supersede", claim, superseded)
        return ReconcileResult("add", claim, [])

    def reinforce(self, claim: Claim, sources: Sequence[str],
                  now: datetime | None = None) -> Claim:
        """Record an independent re-observation of a claim we already hold.

        `sources` are merged rather than replaced: provenance is cumulative, and a claim
        that three separate turns support is a different thing from one that has a single
        source that keeps getting overwritten.
        """
        obs = claim.observation_count + 1
        salience = min(self.max_salience, claim.salience + self.reinforce_bump)
        self.store.reinforce(claim.id, salience, obs, list(sources))
        return self.store.get_claim(claim.id) or claim

    # -- internals ------------------------------------------------------------

    def _canonicalize(self, claim: Claim) -> None:
        """Fold the predicate onto its canonical name before any key is derived.

        `fact_key` and `value_key` hash the predicate, so `resides_in` and `lives_in`
        would otherwise land in different slots and their contradiction would be
        invisible — the exact failure the registry exists to prevent.
        """
        auto_rendered = claim.render()
        claim.subject = claim.subject.strip()
        claim.object = claim.object.strip()
        canonical = self.registry.normalize(claim.predicate)
        if canonical:
            claim.predicate = canonical
        # Only re-render text the Claim generated for itself; a caller-supplied
        # natural-language rendering is theirs to keep.
        if not claim.text.strip() or claim.text.strip() == auto_rendered.strip():
            claim.text = claim.render()

    @staticmethod
    def _live(claims: Sequence[Claim], t: datetime, owner: str) -> list[Claim]:
        """Live claims belonging to the same person.

        The owner check is redundant with the keys — `fact_key` and `value_key` already
        hash tenant+user — and is kept anyway so the "one person's memory never touches
        another's" invariant is enforced here in readable code, not implied by a hash.
        Note it is deliberately *not* a full scope match: agent and session are excluded,
        so "I moved to Lisbon" learned in a new session still retires the old city.
        """
        return [c for c in claims if owner_key(c.scope) == owner and c.is_live(t)]

    @staticmethod
    def _canonical_of(claims: Sequence[Claim]) -> Claim:
        # Earliest recording wins, id breaks ties: the choice must not depend on row
        # order coming back from the store.
        return min(claims, key=lambda c: (c.recorded_at, c.id))

    def _victims(self, claim: Claim, t: datetime, owner: str) -> list[Claim]:
        spec = self.registry.spec(claim.predicate)
        victims: dict[str, Claim] = {}

        if spec.functional:
            for c in self.store.competing_claims(claim.scope.tenant, claim.fact_key, t):
                if owner_key(c.scope) == owner and c.value_key != claim.value_key:
                    victims[c.id] = c
        # else: Cardinality.MANY, including every predicate we have no spec for. Values
        # accumulate and nothing is retired.

        for other in spec.supersedes:
            # A different predicate covering the same slot ("unemployed" ends
            # "works_at"). `fact_key_for` is the only supported way to build a key for a
            # predicate other than the claim's own; hashing it by hand here is how this
            # lookup silently drifts out of sync with what the store indexed and starts
            # matching nothing.
            fk = fact_key_for(claim.scope, claim.subject, self.registry.normalize(other))
            for c in self.store.competing_claims(claim.scope.tenant, fk, t):
                if owner_key(c.scope) == owner:
                    victims[c.id] = c

        # Supersession runs along *valid* time, not arrival order. A fact backfilled
        # today but true from 2019 must not retire the 2026 fact that replaced it — the
        # older statement is history, not news. Splitting here keeps `remember(
        # valid_from=...)`'s documented promise of honest historical import; without it,
        # importing a user's past silently rewrites their present.
        older, newer = [], []
        for c in sorted(victims.values(), key=lambda c: (c.recorded_at, c.id)):
            (newer if c.valid_from > claim.valid_from else older).append(c)
        return older, newer

    def _retire(self, victims: Sequence[Claim], t: datetime, by: str | None) -> None:
        for v in victims:
            v.invalidated_at = t          # transaction time: we stopped believing it now
            v.invalidated_by = by
            if v.valid_to is None or v.valid_to > t:
                v.valid_to = t            # valid time: it stopped being true now
            # `put_claim` rather than `store.invalidate`, because the Store protocol has
            # no way to set `valid_to` and both axes must move together or an `as_of`
            # query lands between them and sees an inconsistent world.
            self.store.put_claim(v)

    def _retract(self, claim: Claim, t: datetime, owner: str) -> ReconcileResult:
        tenant = claim.scope.tenant
        slot = [c for c in self.store.competing_claims(tenant, claim.fact_key, t)
                if owner_key(c.scope) == owner]

        target = claim.object.strip().casefold()
        if target:
            # "I no longer work at Acme" says nothing about an employer we recorded as
            # Globex, so only the named value is retired.
            matches = [c for c in slot if c.object.strip().casefold() == target]
        else:
            matches = list(slot)
        matches.sort(key=lambda c: (c.recorded_at, c.id))

        if not matches:
            prior = [c for c in self.store.find_by_value(tenant, claim.value_key)
                     if owner_key(c.scope) == owner]
            if prior:
                # We have already processed this exact retraction; re-running it must not
                # accumulate tombstones. Provenance still merges.
                keep = self._canonical_of(prior)
                return ReconcileResult("noop", self.reinforce(keep, claim.sources, t), [])

        # The retraction is stored as a tombstone: born already invalidated, so it can
        # never be live and never answers a query, but "why did you stop believing that?"
        # still has an answer with source episodes attached. Discarding it would be the
        # only place in the system where evidence is thrown away.
        claim.invalidated_at = t
        claim.valid_to = t
        self.store.put_claim(claim)

        if matches:
            self._retire(matches, t, claim.id)
        return ReconcileResult("retract", claim, matches)
