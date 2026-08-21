"""Predicate schema: the contradiction engine.

mem0 detects contradictions by embedding the new fact, pulling the top-k most similar
existing memories, and asking an LLM to emit ADD / UPDATE / DELETE / NOOP. That fails
in three ways that matter in production:

  1. If the contradicting memory is not in the top-k, the contradiction is never seen
     and both facts survive. Recall of the conflict is bounded by recall of the search.
  2. It costs an LLM call on the write path, every time.
  3. It is non-deterministic. The same pair of facts can resolve differently on two runs.

The observation that fixes all three: contradiction is mostly a *schema* property, not a
semantic one. "Lives in" takes one value at a time. "Likes" takes many. If you know the
predicate's cardinality, a conflict is an index lookup on (subject, predicate) - exact,
free, and total. No embedding search, no LLM, no top-k cliff.

The LLM's job moves off the write path and onto schema acquisition: when we meet an
unfamiliar predicate we ask about it *once* and cache the answer forever.

That leaves one hole, and it is the one that matters most in production: a model does
not phrase a predicate the same way twice. `works_at`, `employed_by_company`,
`job_employer`, `workplace` and `employer_name` are five spellings of one question, and
because `fact_key` hashes the predicate *string*, five spellings are five slots.
Cardinality never applies across them, so the contradiction engine above silently stops
working — not by returning a wrong answer, but by never being asked. A measured run of
2,058 extractions over six concepts ended with 31 live claims where six were true.

So `normalize()` is not a dictionary lookup; it is a resolution pre-pass, tried in
strictly increasing order of confidence-cost:

    exact canonical -> known alias -> morphology -> derivation -> "genuinely new"

Everything up to the last step is free, deterministic, and total. Only what falls out
the bottom is worth a model call, and that call happens once per surface form, ever,
after which the surface form is recorded as an alias and never asked about again.

Two rules keep the pre-pass from doing damage. It never folds when a key is claimed by
two different predicates (`works_at` and `working_on` both stem to "work"; merging an
employer with a current task is worse than leaving them apart), and the number of
*learned* predicates is capped, because unbounded schema growth is the actual failure
mode and the cap is the backstop for the times resolution guesses wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from .types import MemoryType


class Cardinality(str, Enum):
    """How many values a predicate can hold at once for one subject."""

    ONE = "one"    # single-valued: a new value invalidates the old one
    MANY = "many"  # multi-valued: values accumulate


class Volatility(str, Enum):
    """How fast a fact goes stale, which sets its recency half-life in ranking."""

    STATIC = "static"  # birthplace, date of birth - effectively never decays
    SLOW = "slow"      # employer, city, marital status - changes over years
    FAST = "fast"      # current task, mood, what they're working on today


HALF_LIFE_DAYS: dict[Volatility, float] = {
    Volatility.STATIC: 36500.0,  # 100y - decay term stays ~1.0
    Volatility.SLOW: 730.0,      # 2y
    Volatility.FAST: 7.0,        # 1w
}


@dataclass(frozen=True, slots=True)
class PredicateSpec:
    name: str
    cardinality: Cardinality = Cardinality.MANY
    volatility: Volatility = Volatility.SLOW
    memory_type: MemoryType = MemoryType.SEMANTIC
    aliases: tuple[str, ...] = ()
    # Asserting this predicate also retires these others. Handles the case where two
    # different predicates cover the same slot, e.g. asserting `unemployed` should
    # retire `works_at` even though the predicate names differ.
    supersedes: tuple[str, ...] = ()
    learned: bool = False  # True if acquired at runtime rather than declared up front

    @property
    def half_life_days(self) -> float:
        return HALF_LIFE_DAYS[self.volatility]

    @property
    def functional(self) -> bool:
        return self.cardinality is Cardinality.ONE


def _p(
    name: str,
    card: Cardinality,
    vol: Volatility,
    *aliases: str,
    mtype: MemoryType = MemoryType.SEMANTIC,
    supersedes: tuple[str, ...] = (),
) -> PredicateSpec:
    return PredicateSpec(name, card, vol, mtype, tuple(aliases), supersedes)


# A starter schema for the personal-assistant domain. Deliberately small: it is meant to
# cover the common cases out of the box and then grow by learning, not to be exhaustive.
BUILTIN_PREDICATES: tuple[PredicateSpec, ...] = (
    # --- identity: single-valued and effectively permanent ---
    _p("name", Cardinality.ONE, Volatility.STATIC, "is_named", "called", "full_name"),
    _p("born_on", Cardinality.ONE, Volatility.STATIC, "birthday", "date_of_birth", "dob"),
    _p("born_in", Cardinality.ONE, Volatility.STATIC, "birthplace", "place_of_birth"),
    _p("pronouns", Cardinality.ONE, Volatility.STATIC, "uses_pronouns"),
    _p("speaks", Cardinality.MANY, Volatility.STATIC, "speaks_language", "language"),

    # --- situation: single-valued but genuinely changes ---
    _p("lives_in", Cardinality.ONE, Volatility.SLOW,
       "resides_in", "located_in", "based_in", "lives_at", "home_is", "moved_to",
       "city"),
    _p("works_at", Cardinality.ONE, Volatility.SLOW,
       "employed_at", "employer", "employed_by", "company", "works_for", "joined",
       "workplace"),
    _p("job_title", Cardinality.ONE, Volatility.SLOW, "role", "title", "position", "works_as"),
    _p("timezone", Cardinality.ONE, Volatility.SLOW, "tz", "in_timezone"),
    _p("relationship_status", Cardinality.ONE, Volatility.SLOW, "marital_status"),
    _p("owns_pet", Cardinality.MANY, Volatility.SLOW, "has_pet", "pet"),

    # --- preferences: mostly multi-valued, and procedural rather than factual ---
    _p("likes", Cardinality.MANY, Volatility.SLOW, "enjoys", "loves", "is_a_fan_of",
       mtype=MemoryType.SEMANTIC),
    _p("dislikes", Cardinality.MANY, Volatility.SLOW, "hates", "avoids",
       mtype=MemoryType.SEMANTIC),
    _p("allergic_to", Cardinality.MANY, Volatility.STATIC, "allergy", "is_allergic_to"),
    _p("dietary_restriction", Cardinality.MANY, Volatility.SLOW, "diet", "eats"),

    # --- contact details: single-valued, and the reason they are here ---
    # Not on the original vocabulary, and added with the deterministic extractor that
    # produces them (`write/fast.py`). Cardinality is the whole point: an unknown
    # predicate defaults to `MANY`, so without these specs a second address would sit
    # beside the first, both live, with nothing saying which one to post to — and nothing
    # would warn, because accumulating is exactly what `MANY` is supposed to do.
    _p("address", Cardinality.ONE, Volatility.SLOW,
       "postal_address", "delivery_address", "mailing_address", "ships_to",
       "street_address"),
    _p("phone", Cardinality.ONE, Volatility.SLOW,
       "phone_number", "mobile", "telephone", "contact_number", "cell"),
    #: Which channel to reach this user on. `ONE` because "how should I contact you" has
    #: one standing answer: a reversal is a supersession, not a second preference sitting
    #: alongside the first. A deployment that genuinely accepts several channels at once
    #: declares its own spec at `MANY` and closes the slot explicitly, which is what
    #: `demo/baselines.py` does and why it has to.
    _p("contact_preference", Cardinality.ONE, Volatility.SLOW,
       "contact_via", "contact_method", "preferred_contact", "reach_me_by",
       mtype=MemoryType.PROCEDURAL),

    # --- procedural: how this user wants the agent to behave ---
    _p("prefers", Cardinality.MANY, Volatility.SLOW, "wants", "would_rather",
       mtype=MemoryType.PROCEDURAL),
    _p("prefers_tool", Cardinality.ONE, Volatility.SLOW, "uses_tool", "tool_of_choice",
       mtype=MemoryType.PROCEDURAL),
    _p("communication_style", Cardinality.ONE, Volatility.SLOW, "tone", "style",
       mtype=MemoryType.PROCEDURAL),
    _p("never_do", Cardinality.MANY, Volatility.SLOW, "avoid_doing", "do_not",
       mtype=MemoryType.PROCEDURAL),

    # --- fast-moving state: decays out of the ranking within days ---
    _p("working_on", Cardinality.ONE, Volatility.FAST, "current_task", "current_project",
       mtype=MemoryType.EPISODIC),
    _p("goal", Cardinality.MANY, Volatility.FAST, "wants_to", "plans_to",
       mtype=MemoryType.EPISODIC),
    _p("mood", Cardinality.ONE, Volatility.FAST, "feeling", "feels"),
    _p("located_now", Cardinality.ONE, Volatility.FAST, "currently_in", "travelling_to"),
)


# --- morphology --------------------------------------------------------------
# Three tiers of tokens that modify how a predicate is *spelled* without changing which
# question it answers. They are stripped in this order, and a tier is skipped entirely
# when applying it would leave nothing behind - which is what keeps `name`, `company`
# and `job` usable as predicates in their own right while still being strippable as
# modifiers. `company_name` reaches ("company",) only because tier 1 runs before tier 3.

#: Pure slot metadata. "the name of the employer" is still the employer.
_SLOT_NOISE = frozenset({"name", "names", "value", "values", "field", "label", "info"})
#: Grammatical glue and deixis. "current_city" and "city" are the same slot; a memory
#: store has no use for a predicate that means "the non-current city".
_PARTICLES = frozenset({
    "is", "are", "was", "were", "be", "the", "a", "an",
    "of", "by", "at", "in", "on", "to", "for", "with",
    "my", "our", "your", "their", "user", "users",
    "current", "currently", "present",
})
#: Domain heads a model reaches for when it wants to sound specific.
_DOMAIN_NOISE = frozenset({"company", "job"})
_NOISE_TIERS: tuple[frozenset[str], ...] = (_SLOT_NOISE, _PARTICLES, _DOMAIN_NOISE)

#: Longest-first, so "occupation" reduces via "ation" rather than "ion".
_SUFFIXES: tuple[str, ...] = ("ation", "ment", "ing", "ion", "ed", "er", "or", "al")
#: Below this a "stem" is noise: stripping "al" off "goal" leaves "go", which would
#: happily match half the vocabulary.
_MIN_STEM = 3

#: How many predicates to offer a model when asking it to resolve a surface form. The
#: list is a hint, not an enumeration - sending an unbounded vocabulary is the token tax
#: this workstream exists to remove.
_CANDIDATE_LIMIT = 32

#: Ceiling on predicates acquired at runtime, per registry (and therefore per tenant,
#: since specs are loaded per tenant). 200 is far above what any real deployment needs
#: and far below the point where the schema stops being a schema.
DEFAULT_LEARNED_CAP = 200


def _slugify(raw: str) -> str:
    out: list[str] = []
    prev_us = False
    for ch in raw.strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        elif not prev_us:
            out.append("_")
            prev_us = True
    return "".join(out).strip("_")


def _content_tokens(slug: str) -> list[str]:
    tokens = [t for t in slug.split("_") if t]
    for tier in _NOISE_TIERS:
        kept = [t for t in tokens if t not in tier]
        if kept:
            tokens = kept
    return tokens


def _singular(token: str) -> str:
    # Plural only. Stripping verb inflections here would collapse `works_at` into
    # `working_on`, which is a different question with a different half-life.
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            return token[: -len(suffix)]
    return token


def _strict_key(slug: str) -> tuple[str, ...]:
    """Order-insensitive identity of a predicate's content words, singularized."""
    return tuple(sorted(_singular(t) for t in _content_tokens(slug)))


def _loose_key(slug: str) -> tuple[str, ...]:
    """`_strict_key` with derivational suffixes stripped: employer ~ employed ~ employ."""
    return tuple(sorted(_stem(t) for t in _strict_key(slug)))


@dataclass(frozen=True, slots=True)
class Resolution:
    """The outcome of the deterministic pre-pass.

    `resolved is False` is not a failure - it is the pre-pass declining to guess, which
    is the only safe answer when nothing matched. The caller decides whether that is
    worth a model call.
    """

    name: str      # the predicate to use: a canonical name, or the slug if unresolved
    method: str    # "empty" | "canonical" | "alias" | "morphological" | "derivational" | "novel"
    resolved: bool


class PredicateRegistry:
    """Normalizes predicate names and answers the cardinality question.

    Unknown predicates default to MANY. That is the conservative direction: keeping two
    facts that turn out to conflict degrades ranking, while dropping one that turns out
    not to conflict destroys information. Errors should fall on the recoverable side.

    The same asymmetry governs resolution: an uncertain surface form becomes its own
    predicate rather than being folded onto a neighbour, because a wrong fold destroys a
    distinction permanently while a missed fold only leaves two slots where one would
    do - and the learned cap bounds how much of that we tolerate.
    """

    def __init__(self, specs: tuple[PredicateSpec, ...] = BUILTIN_PREDICATES, *,
                 max_learned: int = DEFAULT_LEARNED_CAP) -> None:
        self._specs: dict[str, PredicateSpec] = {}
        self._alias: dict[str, str] = {}
        # (key -> (tier, canonical)) where tier 0 is a canonical name and tier 1 an
        # alias, and `canonical is None` marks the key as claimed by two predicates.
        self._strict: dict[tuple[str, ...], tuple[int, str | None]] = {}
        self._loose: dict[tuple[str, ...], tuple[int, str | None]] = {}
        self._cache: dict[str, Resolution] = {}
        self._stale = True
        self.max_learned = max_learned
        for s in specs:
            self.register(s)

    # -- normalization -------------------------------------------------------

    def _ensure_index(self) -> None:
        if self._stale:
            self._reindex()

    def _reindex(self) -> None:
        """Rebuild every derived index from `_specs`.

        Rebuilt wholesale rather than patched incrementally because specs are replaced,
        not just added - `learn_alias` swaps a spec for a copy carrying one more alias,
        and an incrementally-maintained index would keep serving the old one's entries
        forever. The registry is a few hundred rows and this runs on registration, not
        on lookup.
        """
        self._alias = {}
        self._strict = {}
        self._loose = {}
        for spec in self._specs.values():
            self._add_keys(spec.name, spec.name, tier=0)
            for alias in spec.aliases:
                slug = _slugify(alias)
                if slug and slug not in self._specs:
                    self._alias[slug] = spec.name
                self._add_keys(slug, spec.name, tier=1)
        self._stale = False

    def _add_keys(self, slug: str, canonical: str, *, tier: int) -> None:
        if not slug:
            return
        # A non-empty slug always yields a non-empty key: `_content_tokens` only drops a
        # noise tier when something survives it, so there is no empty-key case to guard.
        for index, key in ((self._strict, _strict_key(slug)),
                           (self._loose, _loose_key(slug))):
            previous = index.get(key)
            if previous is None or tier < previous[0]:
                index[key] = (tier, canonical)
            elif tier == previous[0] and previous[1] != canonical:
                # Two predicates want the same key. Refuse to serve it at all: picking
                # either one silently merges two distinct slots, and the registration
                # order that decided it is not something a caller can see or control.
                index[key] = (tier, None)

    def _lookup(self, index: dict[tuple[str, ...], tuple[int, str | None]],
                key: tuple[str, ...]) -> str | None:
        found = index.get(key)
        return None if found is None else found[1]

    def resolve(self, raw: str) -> Resolution:
        """Fold a surface predicate onto a canonical one, deterministically.

        Deterministic given (registry state) alone - no embeddings, no thresholds, no
        clock. Two processes holding the same specs resolve the same surface form the
        same way, which is what makes `fact_key` a stable slot identity rather than a
        function of whichever phrasing the model happened to pick that morning.
        """
        self._ensure_index()
        hit = self._cache.get(raw)
        if hit is not None:
            return hit
        self._cache[raw] = out = self._resolve(_slugify(raw))
        return out

    def _resolve(self, slug: str) -> Resolution:
        if not slug:
            return Resolution("", "empty", False)
        if slug in self._specs:
            return Resolution(slug, "canonical", True)
        alias = self._alias.get(slug)
        if alias is not None:
            return Resolution(alias, "alias", True)
        morphological = self._lookup(self._strict, _strict_key(slug))
        if morphological is not None:
            return Resolution(morphological, "morphological", True)
        derivational = self._lookup(self._loose, _loose_key(slug))
        if derivational is not None:
            return Resolution(derivational, "derivational", True)
        return Resolution(slug, "novel", False)

    def normalize(self, raw: str) -> str:
        """Map a surface predicate onto its canonical name.

        Without this, `lives_in` and `resides_in` are different slots and the
        contradiction between them is invisible - which is exactly how free-text memory
        stores end up holding two cities for one person.
        """
        return self.resolve(raw).name

    # -- resolution support --------------------------------------------------

    def _overlap(self, tokens: frozenset[str], name: str) -> int:
        spec = self._specs[name]
        best = 0
        for form in (spec.name, *spec.aliases):
            best = max(best, len(tokens & frozenset(_strict_key(_slugify(form)))))
        return best

    def _affinity(self, tokens: frozenset[str], name: str) -> tuple[int, bool, str]:
        """Sort key: most shared content words, then declared over learned, then name.

        The middle term is the one worth explaining. A declared predicate was written
        down by a person; a learned one was a model's guess that happened to arrive
        first. On a tie those are not equal evidence, and preferring the guess would let
        the first novel phrasing a deployment ever saw become the attractor every later
        phrasing collapses into.
        """
        return (-self._overlap(tokens, name), self._specs[name].learned, name)

    def candidates(self, surface: str, limit: int = _CANDIDATE_LIMIT) -> list[str]:
        """A bounded, deterministic shortlist to offer a model resolving `surface`.

        Bounded because the alternative - shipping the whole vocabulary on every call -
        is an unbounded token tax that grows exactly when the vocabulary is growing
        fastest, and it invalidates the cached prompt prefix at the same moment.
        """
        self._ensure_index()
        tokens = frozenset(_strict_key(_slugify(surface)))
        return sorted(self._specs, key=lambda n: self._affinity(tokens, n))[:limit]

    def nearest(self, surface: str) -> str | None:
        """The existing predicate a surface form is closest to, or None if nothing is.

        Used only once the learned cap is reached, where the choice is "fold onto
        something" or "grow the schema forever". `None` means not even one content word
        is shared, and inventing a relationship on that evidence is worse than leaving
        the predicate unregistered (and therefore multi-valued, which retires nothing).
        """
        self._ensure_index()
        tokens = frozenset(_strict_key(_slugify(surface)))
        if not tokens:
            return None
        best = min(self._specs, key=lambda n: self._affinity(tokens, n), default=None)
        if best is None or self._overlap(tokens, best) == 0:
            return None
        return best

    def prompt_vocabulary(self, limit: int = 64) -> list[str]:
        """Predicate names to show an extractor, builtins first.

        Declared predicates come first and in a fixed order, so the cacheable head of
        the prompt does not move when a predicate is learned; learned ones fill whatever
        budget is left. Sorting the whole thing instead would interleave each new
        predicate into the middle and invalidate the cached prefix on every acquisition.
        """
        self._ensure_index()
        declared = sorted(n for n, s in self._specs.items() if not s.learned)
        learned = sorted(n for n, s in self._specs.items() if s.learned)
        return (declared + learned)[:limit]

    # -- registry ------------------------------------------------------------

    def register(self, spec: PredicateSpec) -> PredicateSpec:
        self._specs[spec.name] = spec
        self._cache.clear()
        self._stale = True
        return spec

    @property
    def learned_count(self) -> int:
        return sum(1 for s in self._specs.values() if s.learned)

    @property
    def at_capacity(self) -> bool:
        return self.learned_count >= self.max_learned

    def spec(self, predicate: str) -> PredicateSpec:
        name = self.normalize(predicate)
        found = self._specs.get(name)
        if found is not None:
            return found
        # Unknown: synthesize a conservative default. Not registered, so a later
        # `learn()` can still install a real spec for it.
        return PredicateSpec(name=name, cardinality=Cardinality.MANY, volatility=Volatility.SLOW)

    def known(self, predicate: str) -> bool:
        return self.normalize(predicate) in self._specs

    def spec_is_declared(self, predicate: str) -> bool:
        """True when this predicate holds a declaration rather than a guess.

        Builtins and loaded packs are declarations; anything the write path classified is
        a guess. `Memvara` uses this to stop rehydration overwriting the former with the
        latter.
        """
        found = self._specs.get(self.normalize(predicate))
        return found is not None and not found.learned

    def learn(
        self,
        predicate: str,
        cardinality: Cardinality,
        volatility: Volatility = Volatility.SLOW,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        aliases: tuple[str, ...] = (),
    ) -> PredicateSpec:
        """Install a spec discovered at runtime (typically proposed by an LLM, once).

        This is the amortization that keeps the write path cheap: the Nth occurrence of
        a predicate costs nothing, because the 1st one paid for the schema.

        Refused past `max_learned`, and refused silently: the caller is a write path
        with a claim in hand, and failing the write over a schema-growth ceiling would
        trade a ranking problem for data loss. The predicate stays unregistered, which
        means multi-valued, which retires nothing.
        """
        name = self.normalize(predicate)
        if name not in self._specs and self.at_capacity:
            return self.spec(name)
        return self.register(
            PredicateSpec(
                name=name,
                cardinality=cardinality,
                volatility=volatility,
                memory_type=memory_type,
                aliases=aliases,
                learned=True,
            )
        )

    def learn_alias(self, canonical: str, surface: str) -> PredicateSpec:
        """Record that `surface` is another spelling of an existing predicate.

        This is what a model call on the write path buys: not a classification, a
        *merge*. The surface form stops being a slot of its own from here on, for this
        process and - once the caller persists the returned spec - for every process
        after it. Aliases do not count against `max_learned`, because folding a form
        onto an existing predicate is the behaviour the cap exists to encourage.
        """
        target = self.normalize(canonical)
        spec = self._specs.get(target)
        if spec is None:
            raise KeyError(
                f"cannot alias {surface!r} onto unknown predicate {canonical!r}; "
                "register it first"
            )
        slug = _slugify(surface)
        if not slug or slug in spec.aliases or slug == spec.name:
            return spec
        return self.register(replace(spec, aliases=spec.aliases + (slug,)))

    def functional(self, predicate: str) -> bool:
        return self.spec(predicate).functional

    def half_life_days(self, predicate: str) -> float:
        return self.spec(predicate).half_life_days

    def superseded_by(self, predicate: str) -> tuple[str, ...]:
        return self.spec(predicate).supersedes

    def all_specs(self) -> list[PredicateSpec]:
        return sorted(self._specs.values(), key=lambda s: s.name)

    def __len__(self) -> int:
        return len(self._specs)


# -- declared vocabularies -----------------------------------------------------
#
# The registry has always accepted `specs=`, so a Python caller could declare a vocabulary
# from the first release. An MCP client cannot: it launches a process and sets environment
# variables, and there was no variable for this. So every server-backed store — which is
# every plugin install — was pinned to the 23 builtins with no way to say otherwise, and
# every predicate outside them fell to the unregistered default. That is what these load.

#: Shipped vocabularies, by the name `MEMVARA_PREDICATES` accepts.
PACKS_DIR = Path(__file__).resolve().parent / "packs"


class PredicatePackError(ValueError):
    """A declared vocabulary could not be read. Raised with the fix in the message."""


def available_packs() -> list[str]:
    """Names of the vocabularies that ship with the package."""
    try:
        return sorted(p.stem for p in PACKS_DIR.glob("*.toml"))
    except OSError:
        return []


def _toml_reader() -> Any:
    """`tomllib`, or a refusal that names the fix.

    Imported here rather than at module scope because `tomllib` arrived in 3.11 and this
    package supports 3.10: a top-level import fails at *collection*, taking every caller
    of this module down over a feature almost none of them use. Lazy, an older
    interpreter costs only the person who actually declares a vocabulary.

    No `tomli` fallback, deliberately. The backport would make this work on 3.10 at the
    price of a runtime dependency the package does not declare, and "numpy and nothing
    else" is a claim a test pins rather than a slogan. Refusing one optional feature on
    one interpreter is the smaller loss.
    """
    try:
        import tomllib

        return tomllib
    except ModuleNotFoundError:  # pragma: no cover - 3.10 only
        raise PredicatePackError(
            "Declared predicate vocabularies need Python 3.11 or later, which is where "
            "`tomllib` arrives. Everything else in memvara works on 3.10.") from None


def _coerce_enum(enum: "type[Enum]", value: object, field: str, name: str) -> Any:
    if value is None:
        raise KeyError(field)
    try:
        return enum(str(value).strip().lower())
    except ValueError:
        allowed = ", ".join(repr(m.value) for m in enum)
        raise PredicatePackError(
            f"predicate {name!r} has {field}={value!r}, which is not one of {allowed}."
        ) from None


def load_specs(source: str) -> tuple[PredicateSpec, ...]:
    """Read one declared vocabulary: a shipped pack name, or a path to a TOML file.

    Every failure is raised rather than skipped. A vocabulary that half-loads is worse
    than one that does not load at all: the predicates that made it through supersede and
    the ones that did not accumulate, and nothing in the store says which is which.
    """
    raw = source.strip()
    if not raw:
        raise PredicatePackError("MEMVARA_PREDICATES contains an empty entry.")

    path = PACKS_DIR / f"{raw}.toml" if raw.isidentifier() else Path(raw).expanduser()
    if not path.is_file():
        if raw.isidentifier():
            known = ", ".join(repr(n) for n in available_packs()) or "none are installed"
            raise PredicatePackError(
                f"{raw!r} is not a predicate pack that ships with memvara. "
                f"Available: {known}. To load your own file, give a path instead.")
        raise PredicatePackError(f"No predicate file at {path}.")

    tomllib = _toml_reader()
    try:
        with path.open("rb") as handle:
            body = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise PredicatePackError(f"{path} is not valid TOML: {exc}") from None
    except OSError as exc:
        raise PredicatePackError(f"{path} could not be read: {exc}") from None

    entries = body.get("predicate")
    if not isinstance(entries, list) or not entries:
        raise PredicatePackError(
            f"{path} declares no predicates. Each one is a [[predicate]] table with at "
            "least `name`, `cardinality` and `volatility`.")

    specs: list[PredicateSpec] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise PredicatePackError(f"{path} has a [[predicate]] entry that is not a table.")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise PredicatePackError(f"{path} has a [[predicate]] entry with no `name`.")
        if name in seen:
            # Last-wins would be a silent, order-dependent choice between two
            # declarations of the same fact, which is the ambiguity declaring exists
            # to remove.
            raise PredicatePackError(f"{path} declares {name!r} more than once.")
        seen.add(name)

        try:
            cardinality = _coerce_enum(Cardinality, entry.get("cardinality"),
                                       "cardinality", name)
            volatility = _coerce_enum(Volatility, entry.get("volatility"),
                                      "volatility", name)
        except KeyError as exc:
            raise PredicatePackError(
                f"predicate {name!r} in {path} has no {exc.args[0]}. Declaring it is the "
                "point of the file — an omitted field would silently take the "
                "unregistered default this pack exists to replace.") from None

        memory_type = (_coerce_enum(MemoryType, entry["memory_type"], "memory_type", name)
                       if "memory_type" in entry else MemoryType.SEMANTIC)
        specs.append(PredicateSpec(
            name=name,
            cardinality=cardinality,
            volatility=volatility,
            memory_type=memory_type,
            aliases=tuple(str(a) for a in entry.get("aliases", ())),
            supersedes=tuple(str(s) for s in entry.get("supersedes", ())),
            # Declared, not learned. The distinction is load-bearing: `Memvara` refuses to
            # let a persisted *learned* spec overwrite a declared one, which is what makes
            # a pack able to correct a store that already guessed wrong.
            learned=False,
        ))
    return tuple(specs)


def load_all_specs(sources: str) -> tuple[PredicateSpec, ...]:
    """Load a comma-separated list of packs and paths, later entries winning.

    Order is the whole interface for combining them: `engineering,./ours.toml` means "the
    shipped vocabulary, then our corrections to it".
    """
    specs: dict[str, PredicateSpec] = {}
    for entry in sources.split(","):
        if entry.strip():
            for spec in load_specs(entry):
                specs[spec.name] = spec
    return tuple(specs.values())

