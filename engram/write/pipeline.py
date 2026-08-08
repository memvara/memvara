"""The write path, as a cost ladder.

mem0 spends one LLM call per `add()`: extraction plus an ADD/UPDATE/DELETE decision, on
every turn, forever. This pipeline is organised so that call is the last resort rather
than the first step, and `WriteReceipt.llm_calls` reports honestly how often we still
needed it.

    Tier 0   store the episode, drop content-hash repeats, catch near-duplicate
             restatements against existing claim embeddings          -- 0 calls
    Tier 1   SalienceGate drops factless turns, FastExtractor handles
             the unambiguous statement forms                          -- 0 calls
    Tier 2   whatever survived both, batched into one extract()       -- 1 call per add()
             plus one resolve_predicate() per *novel surface form*, ever

Reconciliation, deduplication and contradiction resolution sit below all of this and
never call a model at all.

The word doing the work in tier 2 is "novel". A model does not spell a predicate the
same way twice, and `fact_key` hashes the predicate string, so `works_at`,
`employed_by_company` and `employer_name` were three slots that could not contradict
each other - which is how a measured 2,058-extraction run over six concepts ended up
holding 31 live claims and answering "where do you work?" with four employers at once.
`PredicateRegistry.resolve` now folds surface forms deterministically before anything is
billed, and the model is asked only about what falls out of that, once, after which the
answer is recorded as an alias and persisted.
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from time import perf_counter
from typing import Any, Mapping, Sequence

from ..embed.base import Embedder
from ..llm.base import LLM
from ..schema import Cardinality, PredicateRegistry, PredicateSpec, Volatility
from ..store.base import Store
from ..types import Claim, Derivation, Episode, MemoryType, WriteReceipt, utcnow
from .fast import FastExtractor
from .gate import SalienceGate
from .reconcile import ReconcileResult, Reconciler

_EXAMPLE_CHARS = 400


class WritePipeline:
    """Runs the tiers in order and reports what each one cost."""

    def __init__(self, store: Store, embedder: Embedder, registry: PredicateRegistry,
                 llm: LLM, *, near_dup_threshold: float = 0.97,
                 reinforce_bump: float = 0.25) -> None:
        self.store = store
        self.embedder = embedder
        self.registry = registry
        self.llm = llm
        self.near_dup_threshold = near_dup_threshold
        self.gate = SalienceGate()
        self.fast = FastExtractor(registry)
        self.reconciler = Reconciler(store, registry)
        self.reconciler.reinforce_bump = reinforce_bump
        # Surface forms we have already paid to resolve. The registry (and behind it the
        # store) is the durable cache; this set additionally covers the cases the
        # registry cannot record — a resolution that came back unusable, or one refused
        # because the learned cap was reached — so a pathological surface form cannot
        # bill us twice.
        self._resolved: set[str] = set()
        # A rejected embedding is warned about once per pipeline, not once per claim —
        # a misconfigured embedder would otherwise emit one warning per write forever.
        self._warned_embedding = False

    # -- public ---------------------------------------------------------------

    def add(self, episodes: Sequence[Episode]) -> WriteReceipt:
        t0 = perf_counter()
        # One instant for the whole batch. Reconciliation compares timestamps, so a clock
        # read per claim would make the outcome depend on how long the batch took.
        now = utcnow()
        receipt = WriteReceipt()
        if not episodes:
            receipt.latency_ms = (perf_counter() - t0) * 1000.0
            return receipt

        # One transaction for the whole batch. Ingesting a transcript writes an episode
        # row, a claim row and an FTS row per turn, plus reconciliation updates; a
        # durability round-trip on each of those costs far more than the work itself,
        # and a half-applied transcript is not a state any caller wants to recover from.
        with self._transaction():
            fresh = self._tier0_store(episodes, receipt, now)
            kept = self._tier0_near_dupes(fresh, receipt, now)
            gated, fast_claims = self._tier1(kept, receipt)
            llm_claims = self._tier2(gated, receipt, now)

            # Reconcile in input order so a batch containing two claims for the same slot
            # resolves the same way every run.
            candidates: list[Claim] = []
            for ep in kept:
                candidates.extend(fast_claims.get(ep.id, ()))
                candidates.extend(llm_claims.get(ep.id, ()))

            to_embed: list[Claim] = []
            for claim in candidates:
                claim.recorded_at = now
                self._absorb(self.reconciler.apply(claim, now=now), receipt, to_embed)
            self._write_embeddings(to_embed)

        receipt.latency_ms = (perf_counter() - t0) * 1000.0
        return receipt

    def _transaction(self):
        """Batch commits when the store supports it; a no-op otherwise.

        Kept behind `getattr` so third-party `Store` implementations that never heard of
        batching keep working — they just commit per statement as before.
        """
        batch = getattr(self.store, "batch", None)
        return batch() if batch is not None else nullcontext()

    def assert_claim(self, claim: Claim) -> WriteReceipt:
        """Write a caller-supplied claim. Never consults a model, by construction."""
        t0 = perf_counter()
        now = utcnow()
        receipt = WriteReceipt()
        if claim.derivation is Derivation.LLM_EXTRACT and not claim.extractor:
            # Still at the dataclass default, so nobody claimed authorship: this came in
            # through the API and the provenance should say so.
            claim.derivation = Derivation.USER
            claim.extractor = "api/assert"

        to_embed: list[Claim] = []
        self._absorb(self.reconciler.apply(claim, now=now), receipt, to_embed)
        self._write_embeddings(to_embed)
        receipt.latency_ms = (perf_counter() - t0) * 1000.0
        return receipt

    # -- tier 0 ---------------------------------------------------------------

    def _tier0_store(self, episodes: Sequence[Episode], receipt: WriteReceipt,
                     now) -> list[Episode]:
        fresh: list[Episode] = []
        for ep in episodes:
            existing = self.store.find_episode_by_hash(ep.scope.tenant, ep.hash)
            if existing is not None:
                # Byte-identical text we have already extracted from. Re-running any
                # extractor on it can only reproduce what it produced the first time, so
                # we reinforce those claims directly and pay nothing.
                receipt.episode_ids.append(existing.id)
                receipt.skipped += 1
                receipt.reinforced.extend(self._reinforce_from_source(existing, now))
                continue
            self.store.add_episode(ep)
            receipt.episode_ids.append(ep.id)
            fresh.append(ep)
        return fresh

    def _reinforce_from_source(self, ep: Episode, now) -> list[Claim]:
        """Bump every claim that already cites this episode.

        Linear in the tenant's claim count: the Store protocol carries no reverse
        provenance index and inventing one is outside this subsystem's ownership. It runs
        only on exact repeats, which is the cheap case to begin with.
        """
        out: list[Claim] = []
        for c in self.store.iter_claims(ep.scope.tenant):
            if ep.id in c.sources:
                out.append(self.reconciler.reinforce(c, [ep.id], now))
        return out

    def _tier0_near_dupes(self, episodes: Sequence[Episode], receipt: WriteReceipt,
                          now) -> list[Episode]:
        if not episodes:
            return []
        vecs = self.embedder.encode([ep.content for ep in episodes])
        kept: list[Episode] = []
        for ep, vec in zip(episodes, vecs):
            hit = self._near_duplicate(vec, ep, now)
            if hit is None:
                kept.append(ep)
                continue
            # A restatement of something we already believe. Reinforce and move on
            # rather than extracting a claim that would immediately dedupe anyway.
            receipt.reinforced.append(hit)
            receipt.skipped += 1
        return kept

    def _near_duplicate(self, vec, ep: Episode, now) -> Claim | None:
        try:
            hits = self.store.vector_search(vec, ep.scope.ancestors(), 1, now, False)
        except ValueError:
            # The index was built by a different embedder. That is a real
            # misconfiguration, but it should surface where it matters (retrieval and
            # `set_embedding`), not by taking down an otherwise valid write through an
            # optimization the write does not depend on.
            return None
        if not hits:
            return None
        claim_id, score = hits[0]
        if score < self.near_dup_threshold:
            return None
        claim = self.store.get_claim(claim_id)
        if claim is None:
            return None
        return self.reconciler.reinforce(claim, [ep.id], now)

    # -- tier 1 ---------------------------------------------------------------

    def _tier1(self, episodes: Sequence[Episode],
               receipt: WriteReceipt) -> tuple[list[Episode], dict[str, list[Claim]]]:
        gated: list[Episode] = []
        fast_claims: dict[str, list[Claim]] = {}
        for ep in episodes:
            ok, _reason = self.gate.carries_fact(ep)
            if not ok:
                receipt.skipped += 1
                continue
            claims = self.fast.extract(ep)
            if claims:
                fast_claims[ep.id] = claims
            else:
                gated.append(ep)
        return gated, fast_claims

    # -- tier 2 ---------------------------------------------------------------

    def _tier2(self, episodes: Sequence[Episode], receipt: WriteReceipt,
               now) -> dict[str, list[Claim]]:
        out: dict[str, list[Claim]] = {}
        if not episodes:
            return out

        if getattr(self.llm, "is_noop", False):
            # A backend that consults no model must not be billed for one. These turns
            # genuinely reached tier 2 and yielded nothing, which is the honest thing to
            # report — silently returning an empty receipt is how the default
            # configuration reads as "your library is broken" instead of "no extractor".
            receipt.unextracted = len(episodes)
            return out

        # One call for the whole batch, not one per turn. Turns share context, and the
        # per-request overhead dominates at this size.
        try:
            raw = self.llm.extract(episodes, self.registry.prompt_vocabulary())
        except Exception:
            # Episodes are the source of truth that every provenance guarantee rests on,
            # and they are already written. Letting an extraction failure propagate rolls
            # back the enclosing transaction and destroys them — so a provider 429 during
            # a long transcript silently loses the raw text it was derived from. The same
            # guard already wraps predicate acquisition; it was missing on the expensive
            # call, which is the one that actually fails.
            receipt.llm_calls += 1
            receipt.deferred = True
            receipt.unextracted = len(episodes)
            return out
        receipt.llm_calls += 1
        receipt.llm_calls += self._acquire_predicates(raw, episodes)

        for item in raw:
            claim = self._claim_from_dict(item, episodes, now)
            if claim is not None:
                out.setdefault(claim.sources[0], []).append(claim)
        receipt.unextracted = sum(1 for ep in episodes if ep.id not in out)
        return out

    # -- predicate identity ---------------------------------------------------

    def _acquire_predicates(self, raw: Sequence[Mapping[str, Any]],
                            episodes: Sequence[Episode]) -> int:
        """Give every novel surface form a canonical home. Once per form, ever.

        The deterministic pre-pass runs first and answers for the large majority for
        free; only what it declines to guess at reaches a model. That ordering is what
        makes the call affordable — it is paid once per *spelling*, not once per write,
        and never again in this process or any later one.
        """
        calls = 0
        for item in raw:
            resolution = self.registry.resolve(str(item.get("predicate", "") or ""))
            if resolution.resolved or not resolution.name:
                continue
            surface = resolution.name
            if surface in self._resolved:
                continue
            # Marked before the call, not after: if the model raises or answers with
            # nonsense we still must not ask again.
            self._resolved.add(surface)
            calls += self._acquire(surface, item, episodes)
        return calls

    def _acquire(self, surface: str, item: Mapping[str, Any],
                 episodes: Sequence[Episode]) -> int:
        tenant = self._tenant_of(item, episodes)
        if self.registry.at_capacity:
            # Unbounded schema growth is the root cause; this is the backstop for when
            # resolution is wrong. Past the cap a novel form folds onto its nearest
            # existing predicate instead of claiming a slot of its own — and folding is
            # a deterministic token-overlap decision, so it costs nothing and stays
            # reproducible. `nearest` returning None means not one content word is
            # shared, and on that evidence the form is left unregistered (multi-valued,
            # retiring nothing) rather than attached to a stranger.
            near = self.registry.nearest(surface)
            if near is not None:
                self._register_alias(near, surface, tenant)
            return 0

        resolve = getattr(self.llm, "resolve_predicate", None)
        try:
            if resolve is not None:
                answer = resolve(surface, self.registry.candidates(surface))
            else:
                # A backend predating contract D. It cannot merge, so the best it can do
                # is describe the new predicate — the pre-existing behaviour, kept so a
                # third-party LLM implementation does not silently stop working.
                answer = {**self.llm.classify_predicate(
                    surface, self._example(item, episodes)), "canonical": None}
        except Exception:
            # Acquisition is an enrichment, not a precondition. A rate limit or a network
            # blip must not cost the caller the whole batch of facts — the predicate
            # simply stays unresolved (and therefore multi-valued, the safe default), and
            # the claim's own memory_type carries the decision. Marked resolved above, so
            # we do not retry in a hot loop.
            return 1

        canonical = str(answer.get("canonical") or "")
        if canonical and self.registry.known(canonical):
            # A merge. Anything else the model said about cardinality describes a
            # predicate we are not creating, so it is discarded rather than applied to
            # the target — the canonical predicate's own spec is authoritative.
            self._register_alias(canonical, surface, tenant)
            return 1
        # Either genuinely new, or the model named something that does not exist. Both
        # land here: a canonical we cannot look up is indistinguishable from a
        # hallucination, and inventing the slot it names would be worse than a duplicate.
        learned = self.registry.learn(
            surface,
            _coerce(Cardinality, answer.get("cardinality"), Cardinality.MANY),
            _coerce(Volatility, answer.get("volatility"), Volatility.SLOW),
            _coerce(MemoryType, answer.get("memory_type"), MemoryType.SEMANTIC),
        )
        if learned.learned:
            # `learn` refuses past the cap and hands back a synthesized spec instead.
            # Persisting that one would resurrect it at the next open and raise the
            # ceiling a process at a time, which is the failure the cap exists to stop.
            self._persist(learned, tenant)
        return 1

    def _register_alias(self, canonical: str, surface: str, tenant: str) -> None:
        self._persist(self.registry.learn_alias(canonical, surface), tenant)

    def _persist(self, spec: PredicateSpec, tenant: str) -> None:
        """Durably record what we just paid for.

        The in-memory registry is a cache; the store is the thing that makes "asked
        once, ever" true across processes rather than only within one. Specs are
        tenant-scoped (contract A) because a global table lets one tenant's resolution
        silently set another tenant's contradiction behaviour and decay half-life.
        """
        put_spec = getattr(self.store, "put_spec", None)
        if put_spec is None:
            return
        try:
            put_spec(spec, tenant)
        except TypeError:
            # A `Store` predating contract A. Its predicates table is global, which is
            # the bug W2 is fixing; keeping the call working is still better than losing
            # the acquisition entirely and re-paying for it every restart.
            put_spec(spec)

    @staticmethod
    def _tenant_of(item: Mapping[str, Any], episodes: Sequence[Episode]) -> str:
        idx = item.get("source_index")
        if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(episodes):
            return episodes[idx].scope.tenant
        return episodes[0].scope.tenant

    @staticmethod
    def _example(item: Mapping[str, Any], episodes: Sequence[Episode]) -> str:
        idx = item.get("source_index")
        if isinstance(idx, int) and 0 <= idx < len(episodes):
            return episodes[idx].content[:_EXAMPLE_CHARS]
        return str(item.get("object", ""))[:_EXAMPLE_CHARS]

    def _claim_from_dict(self, item: Mapping[str, Any], episodes: Sequence[Episode],
                         now) -> Claim | None:
        """Trust boundary for model output: anything malformed is dropped, not repaired."""
        idx = item.get("source_index")
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(episodes):
            # Without a source we cannot attach provenance, and a claim with no
            # provenance is exactly what this library exists not to store.
            return None
        ep = episodes[idx]

        subject = str(item.get("subject", "") or "").strip() or "user"
        predicate = self.registry.normalize(str(item.get("predicate", "") or ""))
        obj = str(item.get("object", "") or "").strip()
        if not predicate or not obj:
            return None

        try:
            polarity = -1 if int(item.get("polarity", 1)) < 0 else 1
        except (TypeError, ValueError):
            polarity = 1
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.7))))
        except (TypeError, ValueError):
            confidence = 0.7

        if self.registry.known(predicate):
            # The schema is authoritative over a per-call opinion; otherwise the same
            # predicate would decay differently depending on which turn produced it.
            memory_type = self.registry.spec(predicate).memory_type
        else:
            memory_type = _coerce(MemoryType, item.get("memory_type"), MemoryType.SEMANTIC)

        return Claim(
            subject=subject,
            predicate=predicate,
            object=obj,
            scope=ep.scope,
            polarity=polarity,
            memory_type=memory_type,
            valid_from=ep.ts,
            recorded_at=now,
            confidence=confidence,
            sources=[ep.id],
            derivation=Derivation.LLM_EXTRACT,
            extractor=getattr(self.llm, "name", "llm"),
        )

    # -- shared ---------------------------------------------------------------

    @staticmethod
    def _absorb(res: ReconcileResult, receipt: WriteReceipt, to_embed: list[Claim]) -> None:
        if res.action in ("add", "supersede") and res.claim is not None:
            receipt.added.append(res.claim)
            to_embed.append(res.claim)
        elif res.action == "reinforce" and res.claim is not None:
            receipt.reinforced.append(res.claim)
        elif res.action == "retract" and res.claim is not None:
            # The retraction tombstone is stored but is not an added fact; only the
            # claims it retired belong in the receipt's visible outcome.
            to_embed.append(res.claim)
        receipt.invalidated.extend(res.invalidated)

    def _write_embeddings(self, claims: Sequence[Claim]) -> None:
        if not claims:
            return
        vecs = self.embedder.encode([c.text for c in claims])
        for claim, vec in zip(claims, vecs):
            try:
                self.store.set_embedding(claim.id, vec)
            except ValueError as e:
                # The store is the source of truth; the vector index is derived from it.
                # Losing a derived entry must never lose the claim — and since `add()`
                # now runs inside one transaction, raising here would roll back an
                # entire transcript over a single bad vector. Warn once, because a
                # misconfigured embedder that silently produced an unsearchable store
                # would be far worse than a noisy one.
                if not self._warned_embedding:
                    self._warned_embedding = True
                    warnings.warn(
                        f"embedding rejected ({e}); claims are still stored, but will "
                        "not be reachable by vector search until re-embedded",
                        RuntimeWarning, stacklevel=2,
                    )


def _coerce(enum_cls, raw: Any, fallback):
    """Map model-supplied strings onto an enum, falling back on anything unexpected."""
    try:
        return enum_cls(str(raw).strip().lower())
    except (ValueError, AttributeError):
        return fallback
