"""Entity identity: the predicate registry's resolution model, applied to values.

`schema.py` fixed the left-hand side of a claim. `works_at`, `employed_by_company` and
`workplace` were three slots that could not contradict each other, and folding them onto
one canonical predicate took 31 live claims over six concepts down to 6. The right-hand
side had exactly the same hole and nobody had closed it:

    Acme       -> +1 -0
    Acme Corp  -> +1 -1
    acme inc   -> +1 -1
    ACME       -> +1 -1

Four spellings of one employer, three job changes that never happened, each one
well-provenanced and confidently explained by `why()`. `history()` — the feature the
README leads with — was reporting a plausible lie. For a MANY-cardinality predicate the
same gap accumulates rather than churns: `likes` ends up holding `COFFEE`, `Coffee` and
`coffee` as three independent preferences.

The fix mirrors `PredicateRegistry` deliberately, including its refusals:

    empty -> known alias -> known entity -> "novel, and it is its own entity"

Three properties that are not negotiable, and the reasons they are not:

* **Resolution is a pure function plus a dict lookup.** No embedding threshold, no
  clock, no model on the write path. `entity_key` alone decides identity for anything
  the deterministic fold catches, which is the overwhelming majority — see
  `EntityRegistry.methods`. A model may be asked *once per novel fold*, exactly like
  `resolve_predicate`, and only to merge two folds that are genuinely spelled
  differently ("Big Blue" / "IBM"). Nothing in this module asks one by itself.
* **Identity is scoped to the owner, never to the tenant.** Alice's "Acme" and Bob's
  "Acme" are different entities; one tenant deciding two names are one thing must not
  decide it for another, and one *user* deciding it must not decide it for a sibling.
* **When uncertain, a surface form is a new entity.** Wrongly merging two entities
  destroys a distinction permanently; leaving them apart costs a duplicate slot that a
  later alias can still fix. The registry never folds on similarity, only on an exact
  key match or an explicitly recorded alias.

## Merging changes history, and that is handled explicitly

Folding two entities together retroactively changes what contradicts what. Claims that
coexisted peacefully start retiring each other, so `slot_history()` can return a
differently-shaped past after a merge than before it. Nothing is deleted, but
"append-only *and* stable" would weaken to "append-only".

So it does not happen implicitly. `Reconciler` **stamps** the resolved identity into
`Claim.meta` at write time (`SUBJECT_ENTITY` / `OBJECT_ENTITY` in `types.py`), and that
stamp is what every later key derivation reads. A claim keeps the identity it was
written with; an alias learned in month six does not silently re-key month one.

Applying a late alias to existing claims is a separate, named, dry-run-by-default
operation: `memvara.write.reconcile.backfill_entities`. It re-stamps every claim, rewrites
the stored key columns, and replays the newly-colliding live claims in
`(recorded_at, id)` order so the supersession chain rebuilds deterministically — and it
writes a timestamped record into each touched claim's `meta` so `why()` can explain why
history changed. That is the whole trade: history is stable unless an operator asks for
it not to be, and when they do, the change is dated and attributable.

## The amortization curve is worse here than for predicates, and it is worth saying so

Predicates saturate: a deployment meets a few dozen and then stops. Entity surface forms
never saturate — there is always another company, another drink, another person. What
saves this is that entity resolution does not *need* acquisition the way predicates did:
the fold is a total function, so a never-before-seen entity still gets a correct, stable
identity for free. Acquisition buys only the cases the fold cannot see ("Big Blue" is
IBM), and those are rare. Measured over the six-employer simulation in
`tests/test_entities.py`, 31 surface forms fold to exactly 6 identities with zero model
calls.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace
from typing import Callable, Sequence

#: Separator between the tenant and the rest of an owner key, and between an owner and
#: an entity key inside a persisted entity id. Defined here rather than in `types.py`
#: because this module is the one that has to take the two apart again — `types.owner_key`
#: imports it back, so there is exactly one definition of the format.
OWNER_SEP = "\x1f"

#: Corporate form suffixes. They are punctuation, not name: a model writes "Acme",
#: "Acme Corp" and "Acme, Inc." for the same employer within one conversation.
_LEGAL_FORMS = frozenset({
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "lp", "plc",
    "gmbh", "ag", "sa", "sas", "srl", "spa", "bv", "nv", "oy", "ab", "as", "aps",
    "pty", "pte", "kk", "kg", "sarl",
})

#: Stripped only in leading position, and only when something follows.
_ARTICLES = frozenset({"the"})

#: Removed rather than treated as a separator, so "O'Reilly" is one token and not two.
_APOSTROPHES = frozenset("'‘’ʼ´`")

#: How many known entities to offer a model asked to merge a surface form. Bounded for
#: the same reason the predicate shortlist is: an owner's entity set has no ceiling worth
#: relying on, and shipping all of it would be an unbounded per-call token tax.
_CANDIDATE_LIMIT = 16

#: Ceiling on *registered* entities per owner. Unlike the predicate cap this one costs
#: almost nothing when hit: the fold is a pure function, so an unregistered entity still
#: resolves to a correct, stable identity. All the cap bounds is how many entities can
#: carry aliases and occupy a row.
DEFAULT_ENTITY_CAP = 5000


def entity_key(surface: str) -> str:
    """Deterministic identity of an entity surface form. A pure function.

    Case, whitespace, punctuation, accents and corporate form are all ways of spelling
    the same name; none of them is information. What survives is the content words, in
    order — order is kept because "Sun Microsystems" and "Microsystems Sun" are not
    obviously the same thing and this function is not in the business of guessing.

    >>> entity_key("Acme, Inc.")
    'acme'
    >>> entity_key("The Acme Corporation") == entity_key("ACME")
    True
    >>> entity_key("Acme Labs") == entity_key("Acme")
    False

    Returns "" for a surface form with no content at all, which callers read as "no
    entity here" — an empty object is meaningful for retraction, where it means "clear
    the whole slot".
    """
    tokens = _tokens(surface)
    if not tokens:
        return ""
    # Skipped entirely when it would leave nothing, so a company genuinely called "Inc"
    # keeps an identity of its own instead of colliding with every other unfoldable name.
    kept = [t for t in tokens if t not in _LEGAL_FORMS]
    if kept:
        tokens = kept
    if len(tokens) > 1 and tokens[0] in _ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


def _tokens(surface: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", surface).casefold()
    out: list[str] = []
    word: list[str] = []
    for ch in decomposed:
        if unicodedata.combining(ch) or ch in _APOSTROPHES:
            continue          # "Zoë" -> "zoe", "O'Reilly" -> "oreilly"
        if ch.isalnum():
            word.append(ch)
        elif word:
            out.append("".join(word))
            word = []
    if word:
        out.append("".join(word))
    return out


def entity_id(owner: str, key: str) -> str:
    """Persisted id for one owner's entity. See `E-1` in `docs/WORKPLAN.md`.

    The owner is inside the id rather than in a column of its own because the store's
    entity table is tenant-scoped while entity *identity* is owner-scoped: two users of
    one tenant must be able to hold different entities under the same name.
    """
    return f"{owner}{OWNER_SEP}{key}"


def split_entity_id(eid: str) -> tuple[str, str]:
    """Inverse of `entity_id`. `("", eid)` if it carries no owner at all."""
    owner, _, key = eid.rpartition(OWNER_SEP)
    return owner, key


def tenant_of(owner: str) -> str:
    """The tenant an owner key belongs to. See `types.owner_key`."""
    return owner.partition(OWNER_SEP)[0]


@dataclass(frozen=True, slots=True)
class EntitySpec:
    key: str                        # identity that goes into fact_key / value_key
    canonical: str                  # display form: the first spelling we ever saw
    aliases: tuple[str, ...] = ()   # *folded* surface forms merged onto this entity


@dataclass(frozen=True, slots=True)
class EntityResolution:
    """The outcome of the deterministic pass.

    `resolved is False` is not a failure. It means "this fold is new here", which is a
    perfectly good identity — it is only a *question* if the caller has a model handy and
    suspects the entity already exists under a different spelling.
    """

    key: str
    canonical: str
    method: str      # "empty" | "alias" | "known" | "novel"
    resolved: bool


class EntityRegistry:
    """Owner-scoped entity identity, loaded lazily and persisted as it is learned.

    One instance serves every tenant a store holds; rows are read and written per tenant
    (contract E-1) and identity is keyed per owner, which is the pair of scopes the two
    guarantees need. See the module docstring for why merging is never implicit.
    """

    def __init__(self, store: object | None = None, *,
                 max_entities: int = DEFAULT_ENTITY_CAP) -> None:
        self._store = store
        self._max = max_entities
        self._specs: dict[str, dict[str, EntitySpec]] = {}   # owner -> key -> spec
        self._alias: dict[str, dict[str, str]] = {}          # owner -> fold -> key
        self._asked: set[tuple[str, str]] = set()
        self._loaded: set[str] = set()
        #: Resolutions by method, so a deployment can see what fraction of its entity
        #: identity was decided for free. This is the number the design argument rests
        #: on and it should not require a profiler to read.
        self.methods: dict[str, int] = {}

    # -- resolution ------------------------------------------------------------

    def resolve(self, owner: str, surface: str, *,
                register: bool = True) -> EntityResolution:
        """Fold a surface form onto an entity identity. Deterministic, and free.

        Given (registry state, surface) the answer is fixed: two processes holding the
        same rows resolve the same string the same way, which is what makes `value_key`
        a stable value identity rather than a function of whichever spelling the model
        picked that morning.

        A novel fold is registered on the spot, which is what makes the *next* sighting
        free and is the only side effect this method has. `register=False` suppresses it
        for a reader that wants today's answer without teaching the registry anything —
        a migration pass over an existing store, where registering every value ever
        written would import the whole store into the entity table as a side effect of
        reading it.
        """
        self._load(tenant_of(owner))
        key = entity_key(surface)
        if not key:
            return self._counted(EntityResolution("", "", "empty", False))
        # Aliases are consulted before entities on purpose. A merged-away entity may
        # still have a row of its own — contract E-1 has no delete — and reading that
        # row as an entity would silently undo the merge on the next process start.
        target = self._alias.get(owner, {}).get(key)
        if target is not None:
            spec = self._specs[owner][target]
            return self._counted(EntityResolution(spec.key, spec.canonical, "alias", True))
        # A separate name from the alias branch's `spec`: that one is a direct index and
        # cannot be None, this one is a lookup that can be. Sharing the name made the
        # variable's type the union of the two, which is neither of the things it holds.
        known = self._specs.get(owner, {}).get(key)
        if known is not None:
            return self._counted(EntityResolution(known.key, known.canonical, "known", True))
        display = " ".join(surface.split())
        if register:
            self._register(owner, EntitySpec(key, display))
        return self._counted(EntityResolution(key, display, "novel", False))

    def _counted(self, res: EntityResolution) -> EntityResolution:
        self.methods[res.method] = self.methods.get(res.method, 0) + 1
        return res

    def known(self, owner: str, surface: str) -> bool:
        """Whether this owner already has an identity for `surface`. No side effects."""
        self._load(tenant_of(owner))
        key = entity_key(surface)
        return bool(key) and (key in self._specs.get(owner, {})
                              or key in self._alias.get(owner, {}))

    def all(self, owner: str) -> list[EntitySpec]:
        self._load(tenant_of(owner))
        return sorted(self._specs.get(owner, {}).values(), key=lambda s: s.key)

    # -- acquisition -----------------------------------------------------------

    def candidates(self, owner: str, surface: str,
                   limit: int = _CANDIDATE_LIMIT) -> list[str]:
        """A bounded, stably-ordered shortlist to offer a model. Most words shared first."""
        self._load(tenant_of(owner))
        tokens = frozenset(entity_key(surface).split())
        return sorted(
            self._specs.get(owner, {}),
            key=lambda k: (-len(tokens & frozenset(k.split())), k),
        )[:limit]

    def acquire(self, owner: str, surface: str,
                ask: Callable[[str, Sequence[str]], str | None]) -> bool:
        """Ask a model whether `surface` is an existing entity under another name.

        The only place this module can cost anything, and it is bounded the same way
        predicate acquisition is: once per (owner, fold), ever, whatever the answer. The
        deterministic pass has already decided identity by the time we get here — a merge
        only ever *improves* it, so a refusal, a hallucination or a 429 all cost the
        caller nothing but the call.

        Returns whether a merge was recorded.
        """
        self._load(tenant_of(owner))
        key = entity_key(surface)
        if not key:
            return False
        marker = (owner, key)
        if marker in self._asked:
            return False
        pool = [c for c in self.candidates(owner, surface) if c != key]
        if not pool:
            # Nothing exists to merge onto, so the only possible answer is "no". Spending
            # a call to hear it is the failure mode this whole design is about.
            return False
        # Marked before the call, not after: a model that raises or answers with nonsense
        # must not be asked the same question again.
        self._asked.add(marker)
        try:
            answer = ask(surface, pool)
        except Exception:
            # Enrichment, not a precondition. The fold already gave us an identity.
            return False
        target = entity_key(str(answer or ""))
        if not target or target == key or target not in self._specs.get(owner, {}):
            # A canonical we cannot look up is indistinguishable from a hallucination,
            # and inventing the entity it names is worse than holding two.
            return False
        self.learn_alias(owner, target, surface)
        return True

    def learn_alias(self, owner: str, canonical: str, surface: str) -> EntitySpec:
        """Record that `surface` is another spelling of an entity we already hold.

        This is what a model call buys: not a description, a *merge*. From here on the
        surface form stops being an identity of its own — for this process, and once the
        row lands, for every process after it.
        """
        self._load(tenant_of(owner))
        target = entity_key(canonical)
        bucket = self._specs.get(owner, {})
        spec = bucket.get(target)
        if spec is None:
            raise KeyError(
                f"cannot alias {surface!r} onto unknown entity {canonical!r}; "
                "resolve it first"
            )
        key = entity_key(surface)
        if not key or key == target or key in spec.aliases:
            return spec
        # The absorbed fold stops being an entity in its own right. Its row may survive
        # in the store; `resolve` checks aliases first so that row can never win.
        bucket.pop(key, None)
        merged = replace(spec, aliases=spec.aliases + (key,))
        bucket[target] = merged
        self._alias.setdefault(owner, {})[key] = target
        self._persist(owner, merged)
        return merged

    # -- persistence -----------------------------------------------------------

    def _register(self, owner: str, spec: EntitySpec) -> None:
        bucket = self._specs.setdefault(owner, {})
        if len(bucket) >= self._max:
            # Past the cap the fold still resolves `spec.key` correctly and for free —
            # all that is lost is the ability to attach an alias to it, which is exactly
            # the capability the cap exists to bound.
            return
        bucket[spec.key] = spec
        self._persist(owner, spec)

    def _persist(self, owner: str, spec: EntitySpec) -> None:
        put = getattr(self._store, "put_entity", None)
        if put is None:
            return      # a Store predating contract E-1; resolution is unaffected
        put(entity_id(owner, spec.key), spec.canonical, spec.aliases, tenant_of(owner))

    def _load(self, tenant: str) -> None:
        if tenant in self._loaded:
            return
        self._loaded.add(tenant)
        all_entities = getattr(self._store, "all_entities", None)
        if all_entities is None:
            return
        for eid, canonical, aliases in all_entities(tenant):
            owner, key = split_entity_id(eid)
            if not owner or not key:
                # Not something this module wrote. Trusting it would give some other
                # tenant's — or nobody's — entity an owner, and a wrong merge is the one
                # error here that cannot be undone.
                continue
            self._specs.setdefault(owner, {})[key] = EntitySpec(
                key, canonical, tuple(aliases))
            for alias in aliases:
                self._alias.setdefault(owner, {})[alias] = key
