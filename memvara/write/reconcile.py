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

* Nothing is deleted. Superseding closes the old claim's *valid* time and points
  `invalidated_by` at its successor, so `as_of` still returns what we believed at any
  past instant and `valid_at` still returns what was true at any past instant.
* Unknown predicates are `Cardinality.MANY`. Keeping two competing facts degrades
  ranking; retiring a true one destroys information. Only errors of the first kind are
  recoverable.

**Supersession ends a claim; it does not retire it.** The reconciler is only ever told
"here is the new value" — never "the old one was a mistake" — and those are different
events on different clocks. Berlin stopped being true when Lisbon began, so valid time
closes; we were never wrong about Berlin, so transaction time does not. Closing both, as
this module used to, marks every superseded claim as an error and empties out the one
question the two axes exist to answer: *what do we now believe was true in June*. A
caller who really is correcting the record says so with `close="retired"`; see
`memvara.types.Closure`.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Sequence

from ..entities import EntityRegistry
from ..schema import PredicateRegistry
from ..store.base import Store
from ..types import (
    ENTITY_REKEY,
    MAX_SALIENCE,
    OBJECT_ENTITY,
    SUBJECT_ENTITY,
    Claim,
    Closure,
    as_utc,
    close_out,
    default_entity,
    fact_key_for,
    owner_key,
    utcnow,
)

#: The share of a reinforcement bump that a *massed* repetition earns — one that arrives
#: while the trace is still fully available, e.g. the same fact stated twice in one
#: conversation. The rest is earned only in proportion to how much the trace had faded.
#:
#: Bjork & Bjork's new theory of disuse, applied literally: the gain in storage strength
#: from a successful retrieval is inversely related to current retrieval strength. That
#: is the mechanism behind the spacing effect (Ebbinghaus; Cepeda et al. 2006), and the
#: shipped rule had it exactly backwards — a flat `+0.25` regardless of spacing, which
#: made repeating something sixteen times inside one `add()` the cheapest way to pin a
#: claim at the ceiling forever.
#:
#: Not zero, because a restatement is still evidence and `observation_count` is not the
#: only place that should show it; small, because it is the number a poisoning loop gets
#: to multiply.
#:
#: Measured on a FAST predicate, ten observations either way: massed went from 3.25 to
#: 1.23 and distributed over ten months from 0.05 to 3.15 — a 65x inversion turned into
#: a 2.6x ordering that points the right way.
MASSED_SHARE = 0.1


@dataclass(slots=True)
class ReconcileResult:
    action: str                  # "add" | "reinforce" | "supersede" | "retract" | "noop"
    claim: Claim | None          # the stored/updated claim
    invalidated: list[Claim] = field(default_factory=list)  # claims this one retired


class Reconciler:
    """Decides what a candidate claim does to the claims already on record."""

    def __init__(self, store: Store, registry: PredicateRegistry,
                 entities: EntityRegistry | None = None) -> None:
        self.store = store
        self.registry = registry
        #: Entity identity for subjects and objects — the right-hand-side twin of
        #: `registry`. Owned here rather than passed down from `WritePipeline` because
        #: `_canonicalize` is the single chokepoint where a stored string is normalized
        #: before any key is derived from it, and identity has to be decided there or
        #: not at all.
        self.entities = entities if entities is not None else EntityRegistry(store)
        # Set by `WritePipeline` from its own `reinforce_bump`; kept as an attribute
        # rather than a constructor argument so the documented signature stays exact.
        self.reinforce_bump = 0.25
        self.max_salience = MAX_SALIENCE
        #: Optional model hook for merging two spellings the deterministic fold cannot
        #: see are one thing ("Big Blue" / "IBM"). `None` means the write path is
        #: exactly as free as it was — which is the default, because the fold already
        #: decides identity for everything and a merge only refines it. Called at most
        #: once per (owner, novel fold), ever; see `EntityRegistry.acquire`.
        self.resolve_entity: Callable[[str, Sequence[str]], str | None] | None = None

    # -- public ---------------------------------------------------------------

    def apply(self, claim: Claim, *, now: datetime | None = None,
              close: Closure = "ended") -> ReconcileResult:
        """Reconcile one candidate against the claims already on record.

        `close` decides which clock stops on whatever this candidate displaces, and
        `"ended"` is the only answer the reconciler could reach on its own: a candidate
        is a new value, and a new value is news about the world, not an accusation
        against the record. `close="retired"` is the caller saying the thing they are
        replacing was never true — a correction rather than a change — and only a caller
        can know that. See `memvara.types.Closure`.
        """
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
                return ReconcileResult(
                    "reinforce",
                    self.reinforce(keep, claim.sources, self._observed_at(claim, t)),
                    [])

        # 2. Retraction: the user is taking something back.
        if claim.polarity < 0:
            return self._retract(claim, t, owner, close)

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
            # The new value's `valid_from` is when the old one stopped being true — not
            # `t`, which is merely when we found out.
            self._retire(superseded, t, claim.id, claim.valid_from, close=close)
            return ReconcileResult("supersede", claim, superseded)
        return ReconcileResult("add", claim, [])

    def reinforce(self, claim: Claim, sources: Sequence[str],
                  observed_at: datetime | None = None) -> Claim:
        """Record an independent re-observation of a claim we already hold.

        `observed_at` is when the *evidence was uttered*, not when this process is
        running. They coincide during a live conversation and diverge in exactly the
        case that matters: a replay of someone's existing memory store, whose whole
        purpose is rebuilding real history. Stamped with wall-clock now, every replayed
        observation claims to have happened today and the recency signal the import
        exists to reconstruct is destroyed as it is written.

        `sources` are merged rather than replaced: provenance is cumulative, and a claim
        that three separate turns support is a different thing from one that has a single
        source that keeps getting overwritten.

        Three things move, and the first two are what make repetition actually work.
        The *storage* strength (`Claim.salience_base`) goes up — writing the bump onto
        `salience` instead put it on the one variable the nightly pass recomputes from
        scratch, so it was erased as soon as the claim was older than `0.415 * half_life`
        (2.9 days for a FAST predicate) and, since age only grows, erased permanently.
        The observation instant is stamped, so freshness is measured from the last time
        anyone mentioned the fact rather than from the first. Together they are the
        difference between a fact mentioned daily for 90 days settling at the salience
        floor and settling at the ceiling.
        """
        t = observed_at or utcnow()
        base = self._reinforced_base(claim)
        claim.record_observation(t, base)
        claim.salience = min(self.max_salience, base)
        claim.observation_count += 1
        # Merged in Python rather than left to `store.reinforce`, because the base and
        # the observation instant live in `meta` and the protocol has no way to set it —
        # so one `put_claim` writes all three coherently instead of a partial update
        # followed by a second write that could be interleaved.
        claim.sources = list(dict.fromkeys(list(claim.sources) + list(sources)))
        self.store.put_claim(claim)
        return self.store.get_claim(claim.id) or claim

    def _reinforced_base(self, claim: Claim) -> float:
        """Storage strength after this observation. Bigger when the trace was weaker.

        `retrievability` is how available the claim was *before* we saw it again: its
        decayed salience over its undecayed base. At 1.0 nothing had faded, so the
        repetition is massed and earns `MASSED_SHARE` of the bump; near 0 the trace was
        nearly gone and a successful re-encounter is worth the whole of it.

        Read off the claim rather than recomputed from the half-life on purpose: the
        salience floor makes the ratio a slight *over*-estimate of true retrievability
        for a very cold claim, which errs toward the smaller gain, and it keeps
        reinforcement working for a store whose decay pass has never run.
        """
        base = claim.salience_base
        retrievability = min(1.0, claim.salience / base) if base > 0.0 else 1.0
        gain = self.reinforce_bump * (
            MASSED_SHARE + (1.0 - MASSED_SHARE) * (1.0 - retrievability))
        return min(self.max_salience, base + gain)

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _observed_at(candidate: Claim, t: datetime) -> datetime:
        """When the incoming assertion was made, as far as we can tell.

        `valid_from` and not `t`, because every write path sets it to exactly that: the
        episode's timestamp for extracted claims, and the caller's own `valid_from` for
        `remember()`, which is documented as the way to backfill history honestly. Using
        the reconciliation instant instead would stamp a replayed 2024 observation as
        having happened today, which is the one thing an import must not do.

        Clamped to `t` for the same reason `recorded_at` is: a claim asserted as true
        from next month is not evidence about the freshness of anything today, and the
        default `valid_from` is a clock read that can land microseconds past the batch
        instant.
        """
        began = as_utc(candidate.valid_from)
        return began if began < t else t

    def _canonicalize(self, claim: Claim) -> None:
        """Fold all three parts onto their canonical identities before any key exists.

        `fact_key` and `value_key` hash the predicate, so `resides_in` and `lives_in`
        would otherwise land in different slots and their contradiction would be
        invisible — the exact failure the registry exists to prevent. They hash the
        subject and object too, and until this method resolved those as well the same
        hole was wide open on the other side: "Acme", "Acme Corp", "acme inc" and "ACME"
        were four employers, so a single-valued predicate manufactured three job changes
        out of one job.

        This is the only place it can happen. Every write path lands here, and it runs
        before the first key is derived.
        """
        auto_rendered = claim.render()
        claim.subject = claim.subject.strip()
        claim.object = claim.object.strip()
        canonical = self.registry.normalize(claim.predicate)
        if canonical:
            claim.predicate = canonical
        self._stamp(claim)
        # Only re-render text the Claim generated for itself; a caller-supplied
        # natural-language rendering is theirs to keep.
        if not claim.text.strip() or claim.text.strip() == auto_rendered.strip():
            claim.text = claim.render()

    def _stamp(self, claim: Claim, *, learn: bool = True) -> None:
        """Pin each end of this claim to the entity it resolved to, where that matters.

        A stamp is written only when resolution landed somewhere the deterministic fold
        would not have — that is, only for an alias. Two reasons, and both are load
        bearing:

        * `entity_key` is a pure function, so a fold needs no stamp to be stable; a
          stamp for it would be a copy of a value that cannot change, written into every
          claim in the store and surfaced to anyone reading `Claim.meta` as if it were
          their own metadata.
        * The alias table *can* change, and a stamp is exactly what stops it changing
          the past. A claim written before "Big Blue" was known to be IBM keeps the
          identity it was written with, so `history()` does not silently restructure
          itself the day the alias is learned. Applying it to existing rows is a
          separate, dated, dry-run-first operation — see `backfill_entities`.

        `learn=False` re-reads today's answer without teaching the registry or asking a
        model anything. That is what a migration pass wants: it is applying aliases that
        already exist, and a scan of an entire store must not import every value ever
        written into the entity table, nor turn a dry run into a spending decision.
        """
        owner = owner_key(claim.scope)
        for meta_key, surface in ((SUBJECT_ENTITY, claim.subject),
                                  (OBJECT_ENTITY, claim.object)):
            resolution = self.entities.resolve(owner, surface, register=learn)
            novel = learn and bool(resolution.key) and not resolution.resolved
            if novel and self.resolve_entity is not None:
                # A fold nobody has seen here before. The identity we already have is
                # correct and free; a model is asked only whether it is a *second* name
                # for something we hold, and only once per form, ever.
                if self.entities.acquire(owner, surface, self.resolve_entity):
                    resolution = self.entities.resolve(owner, surface)
            if resolution.key and resolution.key != default_entity(surface):
                claim.meta[meta_key] = resolution.key
            else:
                claim.meta.pop(meta_key, None)

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

    def _victims(self, claim: Claim, t: datetime,
                 owner: str) -> tuple[list[Claim], list[Claim]]:
        spec = self.registry.spec(claim.predicate)
        victims: dict[str, Claim] = {}

        if spec.functional:
            for c in self.store.competing_claims(claim.scope.tenant, claim.fact_key,
                                                 valid_at=t, known_at=t):
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
            fk = fact_key_for(claim.scope, claim.subject_key,
                              self.registry.normalize(other))
            for c in self.store.competing_claims(claim.scope.tenant, fk,
                                                 valid_at=t, known_at=t):
                if owner_key(c.scope) == owner:
                    victims[c.id] = c

        # Supersession runs along *valid* time, not arrival order. A fact backfilled
        # today but true from 2019 must not retire the 2026 fact that replaced it — the
        # older statement is history, not news. Splitting here keeps `remember(
        # valid_from=...)`'s documented promise of honest historical import; without it,
        # importing a user's past silently rewrites their present.
        older: list[Claim] = []
        newer: list[Claim] = []
        for c in sorted(victims.values(), key=lambda c: (c.recorded_at, c.id)):
            (newer if c.valid_from > claim.valid_from else older).append(c)
        return older, newer

    def _retire(self, victims: Sequence[Claim], t: datetime, by: str | None,
                valid_to: datetime | None = None, *,
                close: Closure = "ended") -> None:
        """Close out displaced claims on **one** axis: the one that says why.

        `close="ended"` stops the world clock at `valid_to`: the claim was true and is
        not any more. `close="retired"` stops the belief clock at `t`: the claim was
        never true and we have stopped holding it. Both are end-of-life and they are not
        the same end, which is what this signature exists to keep apart.

        Neither closure touches the other axis, and that restraint is the fix. Stamping
        `invalidated_at` on a superseded claim says *we were mistaken* about a value that
        was simply overtaken — so "what do we now believe was true in June" returned
        nothing on any history this library wrote itself. Stamping `valid_to` on a
        corrected claim is the same error mirrored: it asserts a world event ("and then
        it stopped being true") that a correction never witnessed. One write, one
        assertion, one clock.

        `valid_to` is when the world moved, which is the *successor's* `valid_from` and
        not `t` — a fact learned today about a move that happened in July closes in July.
        Getting that wrong leaves the old value overlapping its own replacement: two live
        answers to a single-valued question. It is invisible unless a write is backdated,
        because `valid_from` otherwise defaults to now and the two instants coincide.
        Defaults to `t` for callers where the distinction is genuinely absent.

        `invalidated_by` is written under either closure. It is the *pointer*, not the
        retirement — "this is what displaced me" is true whichever clock stopped, and
        `why()` reads it to report the supersession chain.

        The stamping itself is `types.close_out`, shared with `Memvara.forget`,
        `Memvara.delete` and `Memvara.supersede`, so the two words cannot come to mean
        one thing here and another at the facade.
        """
        boundary = t if valid_to is None else valid_to
        for v in victims:
            close_out(v, t if close == "retired" else boundary, by, close)
            # `put_claim` rather than `store.invalidate`, because the Store protocol has
            # no way to set `valid_to`, and no way to write `invalidated_by` without also
            # writing `invalidated_at` — which is exactly the conflation this method
            # exists to stop making.
            self.store.put_claim(v)

    def _retract(self, claim: Claim, t: datetime, owner: str,
                 close: Closure = "ended") -> ReconcileResult:
        tenant = claim.scope.tenant
        slot = [c for c in self.store.competing_claims(tenant, claim.fact_key,
                                                       valid_at=t, known_at=t)
                if owner_key(c.scope) == owner]

        # Entity identity, the same notion `value_key` uses. It used to be a plain
        # casefold here and a case-*sensitive* hash there, so retraction matched
        # "ACME" against "Acme" while deduplication did not — one operation folding
        # case and its inverse not folding it is a bug in whichever direction you read
        # it from.
        target = claim.object_key
        if target:
            # "I no longer work at Acme" says nothing about an employer we recorded as
            # Globex, so only the named value is retired.
            matches = [c for c in slot if c.object_key == target]
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
                return ReconcileResult(
                    "noop",
                    self.reinforce(keep, claim.sources, self._observed_at(claim, t)),
                    [])
            if target and slot:
                # A named retraction that hit nothing. Object matching is exact (modulo
                # entity identity), so "peanut" does not retract "Peanuts" — and writing
                # a tombstone here would leave a record that looks exactly like one that
                # worked. For a safety-critical retraction ("I'm not allergic to X") that
                # is the worst possible outcome: the claim stays live and the audit trail
                # says it was withdrawn. Record nothing and report a no-op, so the empty
                # `receipt.invalidated` is the caller's unambiguous signal to re-check
                # the value they meant.
                return ReconcileResult("noop", None, [])

        # The retraction is stored as a tombstone: born already invalidated, so it can
        # never be live and never answers a query, but "why did you stop believing that?"
        # still has an answer with source episodes attached. Discarding it would be the
        # only place in the system where evidence is thrown away.
        #
        # Both axes here, and this is the one row where that is right. A tombstone is not
        # an assertion about the world at all — it is bookkeeping — so there is no true
        # interval to preserve and nothing an audit loses by it being unreachable from
        # either clock. Everything the retraction *says* lives on the claims below.
        claim.invalidated_at = t
        claim.valid_to = t
        self.store.put_claim(claim)

        if matches:
            # **A retraction is a world event, not a correction**, so it ends its targets
            # rather than retiring them. Every negative the write path can produce says
            # the same kind of thing — "I no longer work at X", "I used to live in X",
            # "I don't work at X any more" (see `write/fast.py`, where all eight negative
            # rules are of that shape, and `Claim.render`, which words a negative claim as
            # "no longer") — and each of those is the user reporting that the world moved,
            # not that we misheard them. The employment was real; it finished. A caller
            # who means "you recorded that wrongly, it was never true" says
            # `close="retired"` and gets the other axis.
            #
            # A retraction dated in the past ("I stopped working there in March") closes
            # the interval in March, not today — same distinction as a supersession.
            self._retire(matches, t, claim.id, claim.valid_from, close=close)
        return ReconcileResult("retract", claim, matches)


# --- late-alias backfill --------------------------------------------------------


@dataclass(slots=True)
class RekeyReport:
    """What a `backfill_entities` pass did, or would do."""

    scanned: int = 0
    written: int = 0     # claims whose stored key columns were rewritten
    merged: int = 0      # claims folded into an earlier claim of the same value
    retired: int = 0     # claims superseded by the rebuilt chain
    dry_run: bool = True

    def __str__(self) -> str:
        return (f"<RekeyReport scanned={self.scanned} written={self.written} "
                f"merged={self.merged} retired={self.retired}"
                f"{' dry-run' if self.dry_run else ''}>")

    __repr__ = __str__


def backfill_entities(reconciler: Reconciler, tenant: str, *, dry_run: bool = True,
                      now: datetime | None = None) -> RekeyReport:
    """Apply the current entity resolution to claims that were written before it.

    Two situations need this and there is no third. A store written by a build whose
    keys hashed raw strings holds rows whose `fact_key` and `value_key` columns no longer
    match what the code derives, so the old claims are invisible to reconciliation. And
    an alias learned in month six (`EntityRegistry.learn_alias`) applies from month six
    onward, because claims carry a stamp of the identity they were written with.

    **This rewrites history, and that is the entire reason it is a separate function
    with `dry_run=True` as its default.** Claims that coexisted start retiring each
    other, and `slot_history()` returns a differently-shaped past afterwards. Nothing is
    deleted — every id survives, every source episode still resolves — but the shape
    changes, so it happens when an operator asks for it and never as a side effect of a
    write. Run it dry, read the report, then run it for real.

    The procedure, in the order it has to happen:

    1. Re-stamp every claim, retired ones included, and rewrite its key columns. Retired
       claims must move too or `history()` loses the past it is there to show.
    2. Replay each slot's *live* claims in `(recorded_at, id)` order — the same total
       order `Reconciler` uses — so the supersession chain rebuilds identically on every
       run and on every replica. Exact duplicates under the new identity fold into the
       earliest of them; for a single-valued predicate the rest form a chain.
    3. Stamp each touched claim with a dated `ENTITY_REKEY` record, so `why()` can say
       why history changed and not merely that it did.

    Claim ids are preserved throughout. Replaying through `Reconciler.apply` would have
    been less code and would have minted new ids for claims that receipts, logs and
    `invalidated_by` pointers already reference.
    """
    t = now or utcnow()
    report = RekeyReport(dry_run=dry_run)
    claims = sorted(
        reconciler.store.iter_claims(tenant, include_invalidated=True),
        key=lambda c: (as_utc(c.recorded_at), c.id),
    )

    slots: dict[str, list[Claim]] = {}
    for claim in claims:
        report.scanned += 1
        # `learn=False`: this pass applies the aliases that already exist. Registering
        # every value it walks past would import the store into the entity table, and a
        # dry run would stop being dry.
        reconciler._stamp(claim, learn=False)                       # noqa: SLF001
        if claim.is_live(t):
            slots.setdefault(claim.fact_key, []).append(claim)

    for group in slots.values():
        functional = reconciler.registry.spec(group[0].predicate).functional
        by_value: dict[str, Claim] = {}
        current: Claim | None = None
        for claim in group:
            keeper = by_value.get(claim.value_key)
            if keeper is not None:
                _fold_into(keeper, claim, t)
                report.merged += 1
                continue
            if functional and current is not None:
                # Valid time moves to the newer claim's recording instant, which is when
                # the slot would have changed hands had the identities been right at the
                # time. Dating it `now` instead would claim we believed two employers
                # simultaneously for however long the store is old. Transaction time does
                # not move at all: this is a supersession being reconstructed, and the
                # rebuilt chain has to have the shape `Reconciler` would have written.
                _supersede(current, claim, as_utc(claim.recorded_at))
                report.retired += 1
                # No longer a live occupant, so a *later* claim of that same value is a
                # return to a previous employer and must supersede in its turn — not
                # fold into a claim we stopped believing two steps ago.
                by_value.pop(current.value_key, None)
            by_value[claim.value_key] = claim
            current = claim

    if not dry_run:
        batch = getattr(reconciler.store, "batch", None)
        with (batch() if batch is not None else nullcontext()):
            for claim in claims:
                reconciler.store.put_claim(claim)
                report.written += 1
    return report


def _note(claim: Claim, at: datetime, reason: str, other: str) -> None:
    """Record that a backfill moved this claim, and what it was moved against."""
    claim.meta.setdefault(ENTITY_REKEY, []).append(
        {"at": at.timestamp(), "reason": reason, "claim": other})


def _fold_into(keeper: Claim, loser: Claim, at: datetime) -> None:
    """Two claims that turn out to assert the same thing become one.

    Invalidated in *transaction* time only, exactly as consolidation's merge does:
    we stopped believing it separately, but it was never false, so `valid_to` stays
    unset and an `as_of` query from before the backfill still returns the old world.
    """
    keeper.sources = list(dict.fromkeys(keeper.sources + loser.sources))
    keeper.observation_count += loser.observation_count
    loser.invalidated_at = at
    loser.invalidated_by = keeper.id
    _note(loser, at, "merged", keeper.id)
    _note(keeper, at, "absorbed", loser.id)


def _supersede(older: Claim, newer: Claim, at: datetime) -> None:
    """One link of a rebuilt chain: the older value ends where the newer one begins.

    The exact mirror of `_fold_into`, and the pair is worth reading together — they are
    the two ways a backfill can displace a claim, and each closes the one axis that says
    which. A duplicate that folds was never false, so only belief moves; a value that was
    replaced was true until it was replaced, so only the world moves.
    """
    older.invalidated_by = newer.id
    # Clamped exactly as `Reconciler._retire` clamps, so the rebuilt chain cannot produce
    # a row `Reconciler` never could: an interval that ends before it starts is not a
    # shorter fact, it is a row no `as_of` window can return consistently.
    edge = max(at, as_utc(older.valid_from))
    if older.valid_to is None or older.valid_to > edge:
        older.valid_to = edge
    _note(older, at, "superseded", newer.id)
