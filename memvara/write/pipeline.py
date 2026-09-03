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

**Where the transaction starts.** The tiers used to run inside one `store.batch()`,
which on `SQLiteStore` holds a process-wide lock for the length of the block - so tier
2's `llm.extract()` put an Anthropic round trip inside the lock and one slow extraction
stalled every read and every write for every tenant in the process. Under a server that
is not slowness, it is an outage. Candidate production now runs outside every
transaction and only reconciliation and the embedding writes are inside one.

Measured with a 1.0 s fake extraction and a reader thread searching a 200-claim store
throughout: **1 completed search in the whole window at p50 1,006 ms, against 516
completed searches at p50 1.91 ms and p95 2.23 ms.** The write itself takes the same
1.01 s either way - the point is not that the writer got faster, it is that it stopped
being everyone else's problem.

The trade that buys, accepted deliberately: episodes commit before claims do, so a crash
in between leaves an episode with no claims extracted from it. That is the recoverable
direction. Episodes are the source of truth every provenance guarantee rests on, the
content hash makes a retry converge on the same rows rather than duplicating them, and
an unextracted episode is retrievable in its own right (`include_episodes`). The other
ordering loses raw text to a provider timeout, and nothing can reconstruct that.
"""

from __future__ import annotations

import re
import warnings
from contextlib import nullcontext
from datetime import datetime
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from ..embed.base import Embedder
from ..llm.base import LLM, Usage
from ..redact import Redactor, redact_claim, redact_episode
from ..schema import Cardinality, PredicateRegistry, PredicateSpec, Volatility
from ..store.base import Store
from ..telemetry import (
    FAST_HIT,
    FAST_MISS,
    GATE_DROP,
    GATE_PASS,
    PREDICATE_ACCUMULATED,
    PREDICATE_ALIAS,
    PREDICATE_CAPPED,
    PREDICATE_LEARNED,
    WRITE_CLAIMS,
    WRITE_COLLAPSED,
    WRITE_DISPUTED,
    WRITE_EMBEDDING_REJECTED,
    WRITE_EMBEDDING_UNUSABLE,
    WRITE_EXTRACT_MS,
    WRITE_LATENCY_MS,
    WRITE_MEMORY_CLAIMS,
    WRITE_MEMORY_EPISODES,
    WRITE_LLM_CALLS,
    WRITE_LOCK_HELD_MS,
    WRITE_RECONCILE,
    WRITE_RETRACTION,
    WRITE_TOKENS_IN,
    WRITE_TOKENS_OUT,
    WRITE_TURNS,
    Recorder,
    script_of,
)
from ..types import (
    SELF_SUBJECT, Claim, Closure, Derivation, Episode, MemoryType, WriteReceipt, utcnow,
)
from .fast import FastExtractor
from .when import normalize_unit, resolve
from .gate import SalienceGate
from .reconcile import ReconcileResult, Reconciler

_EXAMPLE_CHARS = 400

#: English function words, excluded from the grounding check below so a claim is not
#: credited for sharing "the" or "with" with its source -- every sentence does.
_GROUNDING_STOPWORDS = frozenset("""
a an the of to in on for and or is are was were be been being with as by
at from this that these those it its it's their his her they he she you
your i we our not no do does did has have had can will would should may
might must if then than so but into over under about after before
""".split())

_GROUNDING_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_./\-]*")


def _content_words(text: str) -> list[str]:
    return [w for w in _GROUNDING_WORD_RE.findall(text.lower())
            if w not in _GROUNDING_STOPWORDS and len(w) > 1]


def _wholly_ungrounded(obj: str, source: str) -> bool:
    """True when none of `obj`'s content words appear anywhere in `source`.

    Deliberately blunt. It catches total fabrication -- an object sharing not one word
    with the turn it is supposedly drawn from -- and nothing subtler. A claim that
    reuses real vocabulary but inverts or misattributes it (wrong polarity, the right
    words on the wrong subject) passes this check clean; it is a precision filter for
    wholesale invention, not a semantic-correctness checker.

    Substring containment, not exact-token membership, because this store's own content
    is full of paths and hyphenated identifiers: an object of "expense-tracker" has to
    match inside a source token like "expense-tracker-bb03f971", and a set of exact
    tokens would call that ungrounded.

    Validated against 144 claims from two 4B-class local models run over 20 real
    conversational episodes: zero false positives at this exact rule -- every claim it
    flagged turned out, on manual re-reading of the source, to be fabricated. It does
    not generalise past what was measured: a claim that correctly paraphrases its
    source with no vocabulary in common would also trip it, which is why this is only
    the *trigger* under the default `"auto"` mode -- the embedding rescue below gets a
    veto before anything is rejected.
    """
    words = _content_words(obj)
    if not words:
        # Nothing to check is not evidence of fabrication -- an object this check
        # cannot parse into words must not be rejected on the strength of that alone.
        return False
    src = source.lower()
    if not _GROUNDING_WORD_RE.search(src):
        # The same rule, mirrored onto the source. A turn the tokenizer cannot read at
        # all -- 我住在北京, from which the model correctly extracts `lives_in: Beijing`
        # -- would trip this check on every claim it yields, because no Latin-script
        # object can share a token with it. That is the check being out of its depth,
        # not the claim being ungrounded, so it abstains. Cross-script grounding is the
        # semantic rescue's job, and only a multilingual embedder can actually do it;
        # a *mixed*-script source with any tokenizable content still gets the ordinary
        # check, which is a known limit of this shape rather than an oversight -- see
        # the English-centrism entry in docs/ROADMAP.md for the umbrella.
        return False
    return not any(w in src for w in words)


#: Cosine floor for the embedding rescue under `reject_ungrounded="auto"`. A lexically
#: ungrounded object whose best chunk-cosine against its source episode reaches this is
#: kept -- read as a paraphrase rather than an invention.
#:
#: Measured, not guessed, on the 33 fabricated claims from the eval behind this feature
#: plus 8 hand-built genuine paraphrases sharing zero vocabulary with their sources,
#: under the MiniLM `LocalEmbedder`. Every wholesale invention -- the "Acme" employer,
#: the fictional pet-and-pollen persona, the `"unknown"` template stubs -- scored 0.33
#: or below; the paraphrases that rescue exists for scored 0.45 and up; the separating
#: region on that data is [0.34, 0.42] and this sits inside it with margin on the side
#: that matters. The only two fabrications above it (0.43, 0.46) were typo-variants of
#: text genuinely in the source -- misreadings, not inventions, and the least dangerous
#: thing this filter can miss.
#:
#: Under the default `HashingEmbedder` the same pairs score 0.0-0.11 -- character
#: n-grams have nothing to say about meaning -- so nothing is ever rescued and "auto"
#: degrades to the strict lexical check. That is graceful rather than accidental: the
#: rescue's quality follows the embedder's, and a deployment that cares about paraphrase
#: rescue is one that has configured a semantic embedder.
_GROUNDING_RESCUE_COSINE = 0.40

#: The rescue compares the object against the source in chunks this wide, taking the
#: best score. MiniLM-class embedders truncate around 512 tokens, and episodes here run
#: to several thousand characters -- a fact grounded late in a long turn would otherwise
#: be invisible to a single whole-episode embedding.
_GROUNDING_CHUNK_CHARS = 1200

#: A re-observation identified outside the transaction and applied inside it: the claim
#: to bump, the episodes that evidence it, and when the evidence was uttered. Deferred
#: rather than applied on the spot because the identification is a read and the bump is
#: a write, and only the write belongs in the transaction.
_Reinforcement = tuple[Claim, list[str], datetime]


class UnembeddableTextWarning(UserWarning):
    """A claim was stored with an all-zero vector and can never be found by meaning.

    Its own category for the reason `DegradedExtractionWarning` has one: a deployment
    that knowingly stores text its embedder cannot read should be able to silence this by
    policy without silencing everything else the library says. It is a `UserWarning`
    rather than a `RuntimeWarning` because the fix is a configuration choice — which
    embedder to run — and not a fault in the running code.
    """


class WritePipeline:
    """Runs the tiers in order and reports what each one cost."""

    def __init__(self, store: Store, embedder: Embedder, registry: PredicateRegistry,
                 llm: LLM, *, near_dup_threshold: float = 0.97,
                 reinforce_bump: float = 0.25,
                 evidence_roles: Iterable[str] | None = SalienceGate.DEFAULT_EVIDENCE_ROLES,
                 telemetry: Recorder | None = None,
                 redactor: Redactor | None = None,
                 reject_ungrounded: bool | str = "auto") -> None:
        self.store = store
        self.embedder = embedder
        self.registry = registry
        self.llm = llm
        self.near_dup_threshold = near_dup_threshold
        if not (reject_ungrounded is True or reject_ungrounded is False
                or reject_ungrounded == "auto"):
            raise TypeError(
                f"reject_ungrounded={reject_ungrounded!r} is not a mode. Use \"auto\" "
                "(the default: reject a model-proposed claim only when it shares no "
                "vocabulary with its cited turn AND the embedder finds no semantic tie "
                "either), True (the lexical check alone, no rescue), or False (off).")
        #: `"auto"` by default, and defaulting *on* is a considered reversal of how this
        #: shipped. Three facts make it defensible. The check only ever runs on claims a
        #: model proposed -- `remember()` and the fast path never reach it, so nothing a
        #: caller asserts is ever filtered. The destructive direction is *storing*, not
        #: rejecting: a fabricated value in a ONE-cardinality slot supersedes and ends
        #: the true fact that was there (`works_at: "Acme"` retires the user's real
        #: employer), so letting fabrication through destroys information the way this
        #: library exists not to. And the residual false-positive class -- a genuine
        #: paraphrase sharing zero vocabulary with its source that the embedder *also*
        #: cannot connect -- was observed zero times in 144 real claims, survives only
        #: as 3 of 8 hand-built adversarial cases, and costs one claim from one turn
        #: rather than anything already stored.
        self.reject_ungrounded = reject_ungrounded
        #: Rewrites text before anything durable happens to it, or `None` — the default,
        #: and a fast path rather than a no-op object on the same terms as `telemetry`:
        #: one `is not None` test per call, not per turn. See `memvara.redact` for why
        #: this has to run ahead of the content hash and not merely ahead of the disk.
        self.redactor = redactor
        #: Aggregate metrics sink, or `None`. `None` is the fast path and the default:
        #: every emission below is guarded by an `is not None` test and everything a
        #: metric needs computed - a script classification, a tag dict - is computed
        #: inside that guard. See `memvara.telemetry`.
        self.telemetry = telemetry
        self.gate = SalienceGate(evidence_roles=evidence_roles)
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
        # Same rule, different failure: see `UnembeddableTextWarning`. Separate flags so
        # a store hitting both does not have one silence the other, which would leave the
        # quieter of the two indistinguishable from not happening.
        self._warned_unembeddable = False
        # An embedder that fails during the grounding rescue is warned about once per
        # pipeline, same rule as the two flags above: the rescue fails open (the claim
        # is kept), so the only lasting harm of a broken embedder here is silence about
        # the mode quietly running lexical-only.
        self._warned_grounding = False

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
        rec = self.telemetry
        pre_redaction_script: dict[str, str] = {}
        if self.redactor is not None:
            # First, ahead of everything, because everything else in this method is
            # downstream of the text: `ep.hash` is a stored digest of it, `add_episode`
            # writes it and indexes it for BM25, `encode` may post it to a hosted
            # embedder and `extract` may post it to a model provider. Redacting after
            # any one of those is not redacting.
            if rec is not None:
                # Classify *before* redacting. A replacement token is Latin —
                # "[redacted:phone]" is thirteen Latin letters — so on a short non-Latin
                # turn it outvotes the real script, and `_tier1` would report a Han turn
                # as `latin`. That silently corrupts the gate/fast-path script slices,
                # which exist precisely to measure how English-centric tier 1 is, and it
                # does so only for deployments careful enough to enable redaction.
                pre_redaction_script = {ep.id: script_of(ep.content) for ep in episodes}
            for ep in episodes:
                redact_episode(self.redactor, ep, telemetry=rec)

        # -- candidate production, with no transaction open ----------------------
        # Everything slow lives here: `_tier2` makes a network round trip to a model and
        # `_tier0_near_dupes` may make one to a hosted embedder. Holding the store's
        # write lock across either is the difference between a memory layer and an
        # outage, and nothing in this stretch writes a claim.
        fresh, pending = self._tier0_partition(episodes, receipt, now)

        # Episodes commit on their own, first. See the module docstring for the trade.
        if fresh:
            with self._transaction():
                for ep in fresh:
                    self.store.add_episode(ep)

        kept = self._tier0_near_dupes(fresh, receipt, now, pending)
        gated, fast_claims = self._tier1(kept, receipt, pre_redaction_script)
        llm_claims = self._tier2(gated, receipt, now)

        # Reconcile in input order so a batch containing two claims for the same slot
        # resolves the same way every run.
        candidates: list[Claim] = []
        for ep in kept:
            candidates.extend(fast_claims.get(ep.id, ()))
            candidates.extend(llm_claims.get(ep.id, ()))
        if self.redactor is not None:
            # Belt and braces, and cheap: these were extracted from turns the hook
            # already cleaned, so it should find nothing. It runs anyway so that "every
            # claim in the store passed the redactor exactly once" holds for all four
            # write paths rather than for three of them plus an argument about
            # transitivity — and so a rule tightened for claim objects but not for prose
            # still applies where the value actually lands.
            for claim in candidates:
                redact_claim(self.redactor, claim, telemetry=rec)

        # -- the one transaction that spans claim state --------------------------
        # Still one transaction rather than one per claim: a transcript writes a claim
        # row, an FTS row and a vector per turn plus reconciliation updates, and a
        # durability round trip on each costs far more than the work. What changed is
        # that nothing inside it can block on a network.
        lock_t0 = perf_counter() if rec is not None else 0.0
        with self._transaction():
            for claim, sources, observed_at in pending:
                receipt.reinforced.append(
                    self.reconciler.reinforce(claim, sources, observed_at))
            to_embed: list[Claim] = []
            for claim in candidates:
                claim.recorded_at = now
                self._absorb(claim, self.reconciler.apply(claim, now=now),
                             receipt, to_embed)
            self._write_embeddings(to_embed)

        receipt.latency_ms = (perf_counter() - t0) * 1000.0
        if rec is not None:
            # `lock_held_ms` against `latency_ms` is the measurement this restructure
            # exists to move: the gap between them is work the rest of the process was
            # not blocked by, and it used to be zero.
            rec.timing(WRITE_LOCK_HELD_MS, (perf_counter() - lock_t0) * 1000.0)
            rec.timing(WRITE_LATENCY_MS, receipt.latency_ms)
            rec.counter(WRITE_TURNS, len(episodes))
            rec.counter(WRITE_LLM_CALLS, receipt.llm_calls)
            # What this write actually added, which is what a bill is computed from.
            # `added` is the `add` and `supersede` outcomes and nothing else, so a batch
            # that only reinforced what was already known moves neither of these — the
            # dedup promise, stated as an emission rather than as prose.
            #
            # `len(fresh)` rather than `len(episodes)`: `_tier0_partition` sent exact
            # repeats to `pending` without storing them, and charging for a row nobody
            # wrote is the failure `write.turns` cannot avoid by construction.
            rec.counter(WRITE_MEMORY_CLAIMS, len(receipt.added))
            rec.counter(WRITE_MEMORY_EPISODES, len(fresh))
        return receipt

    def reextract(self, episodes: Sequence[Episode]) -> WriteReceipt:
        """Extract from turns that are already stored. `add()` minus tier 0.

        The case this exists for: a turn reached the store and no claim was ever derived
        from it. That happens two ways and both are ordinary. A deployment running with
        no model keeps every turn and extracts from almost none of them — 96 of 129
        episodes on one measured store. And a provider failure mid-write sets
        `receipt.deferred`, keeps the episodes and returns; nothing retried them until
        this existed, so the facts in that batch were lost while the text sat on disk.

        Tier 0 is skipped and must be: these episodes are stored and embedded already, so
        re-running it would find each one as an exact repeat of itself. Tier 1 runs, and
        runs *first*, because the salience gate is free and the model is not — episodes
        commit before the gate in `add()`, so every acknowledgement and one-word reply in
        the store looks exactly like an unextracted fact from the outside. Without the
        gate here a scheduled sweep would spend a model call on "thanks" on every pass,
        forever.

        **An episode that already has claims is skipped, and that is the idempotency
        guarantee.** Not a convenience: re-reading a stored turn is not new evidence about
        anything, but the reconciler cannot tell that from a genuine repeat, so an
        identical claim arriving twice reconciles to `reinforce` and bumps salience.
        Measured, not assumed — a second extraction of one stored turn returns
        `action="reinforce"`. A sweep that ran twice would therefore quietly promote
        whatever it had already extracted, which is a ranking change nobody asked for and
        nothing would report. Counted on `receipt.already_extracted`.

        What this cannot tell is "the model read this and found nothing" from "no model
        has read this yet": neither leaves a mark on the episode. That population is real
        and `reject_ungrounded` feeds it — a turn whose only extracted claim was refused
        as ungrounded ends with no claims citing it and looks untouched. So every turn
        read here is reported on `receipt.episode_ids`, and a scheduler hands those back
        as `Memvara.pending_extraction(exclude=...)`. Core does not record attempts on the
        episode, because the durable set belongs to whatever is doing the scheduling.
        """
        receipt = WriteReceipt()
        t0, rec = perf_counter(), self.telemetry
        now = utcnow()

        fresh: list[Episode] = []
        for ep in episodes:
            if self.store.claims_citing(ep.scope.tenant, ep.id):
                receipt.already_extracted += 1
                continue
            fresh.append(ep)
        receipt.episode_ids = [ep.id for ep in fresh]

        gated, fast_claims = self._tier1(fresh, receipt)
        llm_claims = self._tier2(gated, receipt, now)

        candidates: list[Claim] = []
        for ep in fresh:
            candidates.extend(fast_claims.get(ep.id, ()))
            candidates.extend(llm_claims.get(ep.id, ()))
        if self.redactor is not None:
            # The same belt-and-braces `add()` applies, for the same reason: "every claim
            # in the store passed the redactor exactly once" has to hold for every write
            # path or it is an argument about transitivity rather than a property.
            for claim in candidates:
                redact_claim(self.redactor, claim, telemetry=rec)

        lock_t0 = perf_counter() if rec is not None else 0.0
        with self._transaction():
            to_embed: list[Claim] = []
            for claim in candidates:
                claim.recorded_at = now
                self._absorb(claim, self.reconciler.apply(claim, now=now),
                             receipt, to_embed)
            self._write_embeddings(to_embed)

        receipt.latency_ms = (perf_counter() - t0) * 1000.0
        if rec is not None:
            rec.timing(WRITE_LOCK_HELD_MS, (perf_counter() - lock_t0) * 1000.0)
            rec.timing(WRITE_LATENCY_MS, receipt.latency_ms)
            rec.counter(WRITE_TURNS, len(fresh))
            rec.counter(WRITE_LLM_CALLS, receipt.llm_calls)
            rec.counter(WRITE_MEMORY_CLAIMS, len(receipt.added))
            # No `write.memory_episodes`: this path stores no episode. Emitting a zero
            # would be indistinguishable from a write that stored none and was charged
            # for it, which is the confusion that series was added to end.
        return receipt

    def _transaction(self):
        """Batch commits when the store supports it; a no-op otherwise.

        Kept behind `getattr` so third-party `Store` implementations that never heard of
        batching keep working — they just commit per statement as before.
        """
        batch = getattr(self.store, "batch", None)
        return batch() if batch is not None else nullcontext()

    def assert_claim(self, claim: Claim, *, close: Closure = "ended",
                     asserted_type: MemoryType | None = None) -> WriteReceipt:
        """Write a caller-supplied claim. Never consults a model, by construction.

        `close` is forwarded to `Reconciler.apply` and decides which clock stops on
        anything this claim displaces: `"ended"` (the world changed — the default) or
        `"retired"` (the record was wrong). `add()` has no such parameter on purpose:
        extraction only ever produces reports about the world, and a model is not
        allowed to decide that something we already stored was a mistake.
        """
        t0 = perf_counter()
        now = utcnow()
        receipt = WriteReceipt()
        if self.redactor is not None:
            # The door `remember()`, `supersede()` and the importer come through, where
            # the value arrives as a structured field and never was a conversation turn.
            # Before `reconciler.apply`, which derives both keys from these strings.
            redact_claim(self.redactor, claim, telemetry=self.telemetry)
        if claim.derivation is Derivation.LLM_EXTRACT and not claim.extractor:
            # Still at the dataclass default, so nobody claimed authorship: this came in
            # through the API and the provenance should say so.
            claim.derivation = Derivation.USER
            claim.extractor = "api/assert"

        to_embed: list[Claim] = []
        self._absorb(claim, self.reconciler.apply(claim, now=now, close=close,
                                                  asserted_type=asserted_type),
                     receipt, to_embed)
        self._write_embeddings(to_embed)
        receipt.latency_ms = (perf_counter() - t0) * 1000.0
        if self.telemetry is not None:
            # No `lock_held_ms` counterpart: this path opens no transaction of its own,
            # so there is no window to report and inventing one would make the write
            # path's two entry points look comparable when they are not.
            self.telemetry.timing(WRITE_LATENCY_MS, receipt.latency_ms)
            # The counter `add()` has and this path did not, which left every write that
            # skips extraction invisible to anything counting writes. One per call rather
            # than per row absorbed, matching `write.turns`: both count what was handed
            # in, so a fact that reinforced an existing claim still says a write happened.
            self.telemetry.counter(WRITE_CLAIMS)
            # And the billable half, on the same terms `add()` emits it. `write.claims`
            # above says a call happened; this says what the call left behind, which are
            # different numbers whenever the assertion reinforced or retracted rather
            # than landing. No episode counterpart: this path stores none.
            self.telemetry.counter(WRITE_MEMORY_CLAIMS, len(receipt.added))
        return receipt

    # -- tier 0 ---------------------------------------------------------------

    def _tier0_partition(self, episodes: Sequence[Episode], receipt: WriteReceipt,
                         now) -> tuple[list[Episode], list[_Reinforcement]]:
        """Split the batch into genuinely new turns and exact repeats. Reads only.

        Nothing is written here. The episode rows are inserted by the caller, in a
        transaction of their own, and the reinforcements this identifies are applied in
        the claim transaction at the end - so a lookup that used to sit behind the write
        lock now costs nothing but a read.
        """
        fresh: list[Episode] = []
        pending: list[_Reinforcement] = []
        # Hash-identical turns *within one batch* used to be caught by the lookup seeing
        # an insert made earlier in the same transaction. Every lookup now happens before
        # every insert, so the batch has to remember its own turns or a transcript that
        # repeats a line would store it twice and hand back two different episode ids.
        seen: dict[tuple[str, str], Episode] = {}
        for ep in episodes:
            key = (ep.scope.tenant, ep.hash)
            existing = seen.get(key)
            if existing is None:
                existing = self.store.find_episode_by_hash(ep.scope.tenant, ep.hash)
            if existing is not None:
                # Byte-identical text we have already extracted from. Re-running any
                # extractor on it can only reproduce what it produced the first time, so
                # we reinforce those claims directly and pay nothing.
                receipt.episode_ids.append(existing.id)
                receipt.skipped += 1
                pending.extend(self._reinforcements_from_source(existing, now))
                continue
            seen[key] = ep
            receipt.episode_ids.append(ep.id)
            fresh.append(ep)
        return fresh, pending

    def _reinforcements_from_source(self, ep: Episode, now) -> list[_Reinforcement]:
        """Every claim that already cites this episode, queued for a bump.

        One indexed lookup where this used to scan the tenant's claims. The scan was
        defensible while exact repeats were rare; the redaction seam is what stopped them
        being rare, because two turns differing only inside a redacted span hash
        identically once the redactor has run, and that is this branch. Its cost rose
        with the store, so a redacting workload's total cost was quadratic — 1.57 / 2.85 /
        5.52 ms per round at 100 / 200 / 400 rounds, against 0.26 / 0.28 / 0.28 now.

        Behind `getattr` because `claims_citing` is new to the `Store` protocol and a
        third-party store predating it must keep working rather than raise on the write
        path. Same pattern as `_transaction`.
        """
        # `ep.ts`, not `now`: the observation happened when the turn was uttered.
        # Stamping wall-clock time here would mark every turn of a replayed historical
        # transcript as observed today, which is exactly the recency signal an import
        # exists to reconstruct. Clamped so a turn dated in the future cannot push the
        # trace clock ahead.
        at = min(ep.ts, now)
        citing = getattr(self.store, "claims_citing", None)
        if citing is None:
            found: Iterable[Claim] = (c for c in self.store.iter_claims(ep.scope.tenant)
                                      if ep.id in c.sources)
        else:
            found = citing(ep.scope.tenant, ep.id)
        # Claims that are no longer in force are skipped: reinforcement raises a claim's
        # storage strength so retrieval ranks it higher, and neither a claim we have
        # stopped believing nor one that has finished being true has a present-tense
        # ranking to raise.
        #
        # `is_live` and not `invalidated_at is None`, which is what this read used to be.
        # Once supersession stopped closing transaction time, that test started returning
        # *every superseded version of the slot* — so a repeated turn would have walked
        # back up the chain restating the storage strength and the observation stamp of
        # every city the user had ever lived in. Same set as before the change; the
        # predicate is now the one that actually means it.
        return [(c, [ep.id], at) for c in found if c.is_live(now)]

    def _tier0_near_dupes(self, episodes: Sequence[Episode], receipt: WriteReceipt,
                          now, pending: list[_Reinforcement]) -> list[Episode]:
        if not episodes:
            return []
        # Outside the transaction on purpose: `encode` is a local hash for the shipped
        # embedder and a network round trip for a hosted one, and the write lock must
        # not depend on which was configured.
        vecs = self.embedder.encode([ep.content for ep in episodes])
        kept: list[Episode] = []
        for ep, vec in zip(episodes, vecs):
            hit = self._near_duplicate(vec, ep, now)
            if hit is None:
                kept.append(ep)
                continue
            # A restatement of something we already believe. Queue a reinforcement and
            # move on rather than extracting a claim that would immediately dedupe.
            claim, at = hit
            pending.append((claim, [ep.id], at))
            receipt.skipped += 1
        return kept

    def _near_duplicate(self, vec, ep: Episode, now) -> tuple[Claim, datetime] | None:
        """The claim this turn merely restates, and when the restatement was uttered."""
        try:
            hits = self.store.vector_search(vec, ep.scope.ancestors(), 1,
                                            valid_at=now, known_at=now)
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
        # Same reasoning as `_reinforcements_from_source`: a near-duplicate restatement
        # is evidence dated to the turn that carried it, not to when we processed it.
        return claim, min(ep.ts, now)

    # -- tier 1 ---------------------------------------------------------------

    def _tier1(self, episodes: Sequence[Episode], receipt: WriteReceipt,
               scripts: Mapping[str, str] | None = None,
               ) -> tuple[list[Episode], dict[str, list[Claim]]]:
        rec = self.telemetry
        scripts = scripts or {}
        gated: list[Episode] = []
        fast_claims: dict[str, list[Claim]] = {}
        for ep in episodes:
            ok, reason = self.gate.carries_fact(ep)
            # Both tier-1 stages are English by construction - a filler vocabulary and a
            # set of English sentence patterns - and both fail *quietly* on text they
            # were not built for: the gate drops the turn, the fast path misses and the
            # turn costs a model call. Neither is visible without the script slice, so
            # "the write path is cheap" stays an unqualified claim when it may only be
            # true for Latin-script users.
            # The pre-redaction reading when there is one; see `add`.
            script = "" if rec is None else scripts.get(ep.id) or script_of(ep.content)
            if not ok:
                receipt.skipped += 1
                if rec is not None:
                    rec.counter(GATE_DROP, reason=reason, script=script)
                continue
            if rec is not None:
                rec.counter(GATE_PASS, reason=reason, script=script)
            claims = self.fast.extract(ep)
            if claims:
                fast_claims[ep.id] = claims
                if rec is not None:
                    rec.counter(FAST_HIT, script=script)
            else:
                gated.append(ep)
                if rec is not None:
                    rec.counter(FAST_MISS, script=script)
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
        rec, extract_t0 = self.telemetry, perf_counter()
        # One accumulator for the whole batch, allocated only for a backend that says it
        # will fill it — an older three-argument implementation must not be handed a
        # keyword it does not accept. Not gated on `rec`: tokens land on the receipt too,
        # and a caller who never configured telemetry still gets to see what they spent.
        usage = Usage() if getattr(self.llm, "reports_usage", False) else None
        try:
            if usage is None:
                raw = self.llm.extract(episodes, self.registry.prompt_vocabulary())
            else:
                raw = self.llm.extract(episodes, self.registry.prompt_vocabulary(),
                                       usage=usage)
        except Exception:
            if rec is not None:
                rec.timing(WRITE_EXTRACT_MS, (perf_counter() - extract_t0) * 1000.0)
            # A call that raised may still have burned tokens — a provider that timed out
            # after generating, or a response rejected while being parsed. Publishing
            # what it reported is the difference between an outage that shows up on the
            # bill and one that shows up in the metrics too.
            self._report_usage(receipt, usage)
            # A provider 429 is not a reason to lose a transcript. Episodes are already
            # committed and are the source of truth every provenance guarantee rests on,
            # and the same batch's fast-path claims are still in hand and still correct —
            # letting the failure propagate would discard those too and leave the caller
            # with an exception instead of a receipt saying what was lost. The same guard
            # already wraps predicate acquisition; it was missing on the expensive call,
            # which is the one that actually fails.
            receipt.llm_calls += 1
            receipt.deferred = True
            receipt.unextracted = len(episodes)
            return out
        if rec is not None:
            rec.timing(WRITE_EXTRACT_MS, (perf_counter() - extract_t0) * 1000.0)
        receipt.llm_calls += 1
        # Acquisition shares the accumulator: the caller is billed for a write, not for a
        # round trip, and a novel surface form costing a second call is part of the same
        # write. Reported after it, so those tokens are inside the total.
        receipt.llm_calls += self._acquire_predicates(raw, episodes, usage)
        self._report_usage(receipt, usage)

        for item in raw:
            claim = self._claim_from_dict(item, episodes, now, receipt)
            if claim is not None:
                out.setdefault(claim.sources[0], []).append(claim)
        receipt.unextracted = sum(1 for ep in episodes if ep.id not in out)
        return out

    # -- predicate identity ---------------------------------------------------

    def _report_usage(self, receipt: WriteReceipt, usage: Usage | None) -> None:
        """Put what a write consumed on its receipt, and on the two token series.

        `reported == 0` is the case that must stay silent: the backend either does not
        measure or came back without a usage block, and 0 is not the answer — a call that
        reached a provider consumed something. Emitting a zero would understate a bill and
        would drag the fleet-wide average toward it, which is the failure direction that
        favours us and therefore the one to refuse.
        """
        if usage is None or usage.reported == 0:
            return
        receipt.tokens_in += usage.input_tokens
        receipt.tokens_out += usage.output_tokens
        if self.telemetry is not None:
            self.telemetry.counter(WRITE_TOKENS_IN, usage.input_tokens)
            self.telemetry.counter(WRITE_TOKENS_OUT, usage.output_tokens)

    def _acquire_predicates(self, raw: Sequence[Mapping[str, Any]],
                            episodes: Sequence[Episode],
                            usage: Usage | None = None) -> int:
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
            calls += self._acquire(surface, item, episodes, usage)
        return calls

    def _acquire(self, surface: str, item: Mapping[str, Any],
                 episodes: Sequence[Episode], usage: Usage | None = None) -> int:
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
            if self.telemetry is not None:
                # Any of this at all means the ceiling is now load-bearing rather than a
                # backstop, and `folded=no` means a surface form is live and unregistered
                # - multi-valued, retiring nothing, which is where duplicate slots come
                # from.
                self.telemetry.counter(
                    PREDICATE_CAPPED, folded="yes" if near is not None else "no")
            return 0

        resolve = getattr(self.llm, "resolve_predicate", None)
        # `usage` is None for any backend that did not advertise `reports_usage`, and the
        # keyword is then never sent — same courtesy the `classify_predicate` fallback
        # below extends to a backend predating contract D.
        kw: dict[str, Any] = {} if usage is None else {"usage": usage}
        try:
            if resolve is not None:
                answer = resolve(surface, self.registry.candidates(surface), **kw)
            else:
                # A backend predating contract D. It cannot merge, so the best it can do
                # is describe the new predicate — the pre-existing behaviour, kept so a
                # third-party LLM implementation does not silently stop working.
                answer = {**self.llm.classify_predicate(
                    surface, self._example(item, episodes), **kw), "canonical": None}
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
            if self.telemetry is not None:
                self.telemetry.counter(PREDICATE_ALIAS)
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
            if self.telemetry is not None:
                # Novel registrations per day. A simulation of 10,000 extractions over
                # six concepts produced 41 predicates; this is the rate that did it, and
                # it should fall toward zero as a tenant's vocabulary settles. Rising
                # steadily is predicate explosion, and it has no other symptom until
                # `recall()` starts answering with four employers at once.
                self.telemetry.counter(PREDICATE_LEARNED)
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

    def _grounding_rescued(self, obj: str, source: str) -> bool:
        """The embedder's veto over the lexical trigger, under `"auto"`.

        A lexically ungrounded object is kept anyway when its best chunk-cosine against
        the source reaches `_GROUNDING_RESCUE_COSINE` -- that is what a genuine
        paraphrase looks like and what a wholesale invention does not (the constant's
        docstring carries the measurements). Chunked because MiniLM-class embedders
        truncate long input, and a fact grounded late in a 7,000-character turn must
        not be invisible to the comparison.

        Fails open. A broken embedder here must cost the rescue's *precision*, never a
        claim: the mode degrades to keeping what it cannot check, warns once, and the
        lexical trigger alone is not grounds for rejection when the second opinion the
        mode promises cannot be obtained.
        """
        try:
            chunks = [source[i:i + _GROUNDING_CHUNK_CHARS]
                      for i in range(0, max(len(source), 1), _GROUNDING_CHUNK_CHARS)]
            vectors = self.embedder.encode([obj] + chunks)
            norms = (vectors ** 2).sum(axis=1) ** 0.5
            norms[norms == 0.0] = 1.0
            unit = vectors / norms[:, None]
            best = float(max(unit[1:] @ unit[0]))
        except Exception as e:
            if not self._warned_grounding:
                self._warned_grounding = True
                warnings.warn(
                    f"embedding failed during the grounding rescue ({e}); ungrounded "
                    "claims are being kept rather than rejected until it recovers",
                    RuntimeWarning, stacklevel=2,
                )
            return True
        return best >= _GROUNDING_RESCUE_COSINE

    def _claim_from_dict(self, item: Mapping[str, Any], episodes: Sequence[Episode],
                         now, receipt: WriteReceipt) -> Claim | None:
        """Trust boundary for model output: anything malformed is dropped, not repaired.

        `receipt` is threaded through only so a rejection for lack of grounding can be
        counted where the decision is made -- see `reject_ungrounded`. A structurally
        malformed item (missing predicate, out-of-range source) is dropped the same way
        it always was, uncounted; only the new reason has a number attached to it.
        """
        idx = item.get("source_index")
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(episodes):
            # Without a source we cannot attach provenance, and a claim with no
            # provenance is exactly what this library exists not to store.
            return None
        ep = episodes[idx]

        subject = str(item.get("subject", "") or "").strip() or SELF_SUBJECT
        predicate = self.registry.normalize(str(item.get("predicate", "") or ""))
        obj = str(item.get("object", "") or "").strip()
        if not predicate or not obj:
            return None

        if self.reject_ungrounded and _wholly_ungrounded(obj, ep.content):
            if not (self.reject_ungrounded == "auto"
                    and self._grounding_rescued(obj, ep.content)):
                receipt.ungrounded += 1
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

        # The model reports the expression it saw; `write.when` decides what it means, and
        # falls back to the episode's timestamp when nothing was stated or nothing
        # resolved. A model that computed its own date would be doing arithmetic it is
        # measurably bad at, in a field nothing downstream can check.
        # Only for predicates that accumulate — see `fast.py:_claim` for why a stated
        # boundary on a superseding predicate changes which value reads as current.
        mention = item.get("when")
        resolved = (resolve(mention, ep.ts)
                    if isinstance(mention, str)
                    and not self.registry.functional(predicate) else None)
        valid_from, precision = resolved if resolved else (ep.ts, None)
        raw_amount = item.get("amount")
        amount = float(raw_amount) if isinstance(raw_amount, (int, float)) \
            and not isinstance(raw_amount, bool) else None
        unit = normalize_unit(item.get("unit")) if amount is not None else None

        return Claim(
            subject=subject,
            predicate=predicate,
            object=obj,
            scope=ep.scope,
            polarity=polarity,
            memory_type=memory_type,
            valid_from=valid_from,
            temporal_precision=precision,
            amount=amount,
            unit=unit,
            recorded_at=now,
            confidence=confidence,
            sources=[ep.id],
            derivation=Derivation.LLM_EXTRACT,
            extractor=getattr(self.llm, "name", "llm"),
        )

    # -- shared ---------------------------------------------------------------

    def _absorb(self, candidate: Claim, res: ReconcileResult, receipt: WriteReceipt,
                to_embed: list[Claim]) -> None:
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
        if res.accumulated is not None:
            # Above the telemetry guard, not inside it: the receipt is the account of
            # what this write did and must not depend on whether anyone wired a metrics
            # backend. Same rule `unextracted` follows.
            receipt.accumulated.append(res.accumulated)
        receipt.disputed.extend(res.disputed)
        receipt.collapsed.extend(res.collapsed)
        if res.retyped is not None:
            receipt.retyped.append(res.retyped)

        rec = self.telemetry
        if rec is None:
            return
        rec.counter(WRITE_RECONCILE, action=res.action)
        if res.accumulated is not None:
            # **A value that piled up under an undecided predicate**, which the receipt
            # names one write at a time and this counts across a deployment. It is the
            # aggregate half of the same signal `write.retraction{outcome="noop"}` is:
            # an outcome the API reports as an ordinary success, because it is one — the
            # row is fine and the schema is the thing that was never decided.
            rec.counter(PREDICATE_ACCUMULATED)
        if res.disputed:
            # **A write that resolved nothing**, which every other series reports as an
            # ordinary success: the claim is stored, so `write.reconcile{action="add"}`
            # moves exactly as it does for a first write into an empty slot. One per
            # candidate rather than per victim, so the series counts writes that hit a
            # more confident incumbent rather than the size of the slot they hit.
            rec.counter(WRITE_DISPUTED)
        for _ in res.collapsed:
            # **A closed claim that answers nothing**, counted per claim because that is
            # what is unreachable. `write.reconcile{action="supersede"}` looks identical
            # whether the displaced value kept an interval or lost one.
            rec.counter(WRITE_COLLAPSED)
        if candidate.polarity < 0:
            # **A retraction that retires nothing is an anomaly**, and the API cannot
            # tell you so: `forget()` returns an ordinary receipt whether it cleared the
            # slot or matched no claim at all. Two things produce it and both matter -
            # a user trying and failing to take back something poisoned into their
            # memory, and a `forget()` whose predicate or object does not match what is
            # actually on record. The second is the ordinary bug; the first is the one
            # that has to be caught from outside, because the adversary's whole aim is
            # that it leave no trace.
            rec.counter(WRITE_RETRACTION,
                        outcome="retired" if res.invalidated else "noop")

    def _unembeddable(self, claim: Claim) -> None:
        """Report a claim whose vector carries nothing, which nothing else would.

        A rejected embedding raises and is handled below. This one succeeds: an all-zero
        row is stored like any other, retrieval correctly abstains on it rather than
        inventing a rank, and the claim answers by predicate and by lexical match. Every
        layer behaves well and the result is a fact that vector search can never return,
        with no exception anywhere to say so. Detection is one norm the embedder has
        already computed and thrown away.

        The usual cause is a script the configured embedder cannot tokenise — the shipped
        `HashingEmbedder` matches `[a-z0-9']+`, so Han, Kana, Hangul, Arabic and Hebrew
        all reduce to no tokens and the character n-grams are built over the *rejoined
        word list*, which is empty too. The script goes in the message because "this text
        embedded to nothing" is not actionable and "your Han text embedded to nothing" is.

        **What this does not catch**, and it is the larger half: a *mixed* text embeds to
        a perfectly good vector built from the Latin part alone. `remember("user",
        "lives_in", "里斯本")` renders as `user lives in 里斯本`, whose norm is 1.0 — so
        every `lives_in` claim looks the same in vector space no matter which city it
        names. A norm is a whole-string measure and cannot see a component contributing
        nothing. This guard is the floor, not the fix; the fix is an embedder that
        tokenises the script.
        """
        if self.telemetry is not None:
            # Per-claim, because the warning fires once per process and a count is the
            # only thing that answers "how much of this store is unsearchable".
            self.telemetry.counter(WRITE_EMBEDDING_UNUSABLE, script=script_of(claim.text))
        if self._warned_unembeddable:
            return
        self._warned_unembeddable = True
        warnings.warn(
            f"{self.embedder!r} produced an all-zero embedding for {script_of(claim.text)} "
            "text, so those claims are stored but can never be returned by meaning — only "
            "by predicate or by exact lexical match. The shipped HashingEmbedder only "
            "tokenises [a-z0-9'], so any non-Latin script embeds to nothing. Configure an "
            "embedder that covers the scripts you store. Mixed-script text does not reach "
            "this warning and is affected too: the vector is built from the Latin part "
            "alone.",
            UnembeddableTextWarning, stacklevel=2,
        )

    def _write_embeddings(self, claims: Sequence[Claim]) -> None:
        if not claims:
            return
        vecs = self.embedder.encode([c.text for c in claims])
        for claim, vec in zip(claims, vecs):
            if not vec.any():
                self._unembeddable(claim)
            try:
                self.store.set_embedding(claim.id, vec)
            except ValueError as e:
                # The store is the source of truth; the vector index is derived from it.
                # Losing a derived entry must never lose the claim — and since the claim
                # write runs inside one transaction, raising here would roll back an
                # entire transcript over a single bad vector. Warn once, because a
                # misconfigured embedder that silently produced an unsearchable store
                # would be far worse than a noisy one.
                if self.telemetry is not None:
                    # Warn-*once* is right for a human reading stderr and useless for
                    # anyone asking six months later how big the hole is. The counter is
                    # per-claim and is the only thing that answers that.
                    self.telemetry.counter(WRITE_EMBEDDING_REJECTED)
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
