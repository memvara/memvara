"""Machinery all three adapters need, in one place so three copies cannot drift.

Two of the three things here exist because of the same constraint: **no framework may
become a hard dependency.** `import memvara` has to keep working with numpy alone, and CI
asserts it by walking every module in the package with nothing else installed. So an
adapter module may not import its framework at module scope, and `require()` is the one
place that import happens — late, and with an error that names the extra.

The third, `bind`, is about not making the caller repeat themselves. Every adapter is
bound to a scope for its whole life (a chat history *is* a session; a CrewAI storage
backend *is* a user's memory), so the four scope keywords are taken once at construction
rather than on every call.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..core import Memvara
from ..retrieve import EpisodeResult
from ..types import Claim, Episode, Scope


class IntegrationError(NotImplementedError):
    """A framework call with no honest translation onto memvara.

    `NotImplementedError` so `except NotImplementedError` around an adapter catches it
    alongside the `Store` protocol's own gaps, and a shared base so an application
    wiring up two frameworks can catch both with one clause. Each adapter subclasses it,
    because "LangChain will never do this" and "CrewAI will never do this" are different
    facts about different interfaces.

    Every message raised through this type must name what to do instead. A refusal that
    only says no is a worse outcome than the plausible-looking wrong answer it replaced,
    because it stops the caller without telling them where to go.
    """


def require(module: str, *names: str, extra: str, needs: str) -> tuple[Any, ...]:
    """Import `names` from `module` at call time, or raise naming the extra.

    Called from inside a function, never at module scope — that is the whole point. The
    two failure modes are told apart on purpose: a missing package is a `pip install`
    away and says so, while a package that is present but missing the name is a version
    skew, and reporting *that* as "not installed" sends people to reinstall something
    they already have.
    """
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"this adapter needs {module}: pip install 'memvara[{extra}]' "
            f"(or {needs} directly). Memvara itself needs numpy and nothing else; the "
            "framework is only required by the adapter you just imported."
        ) from exc
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        raise ImportError(
            f"{module} is installed but has no {', '.join(missing)}. That is a version "
            f"skew rather than a missing package — this adapter is written against "
            f"{needs}."
        )
    return tuple(getattr(mod, n) for n in names)


def bind(memory: Any, *, tenant: str | None = None, user: str | None = None,
         agent: str | None = None, session: str | None = None) -> tuple[Memvara, Scope]:
    """Resolve `(memvara, scope)` from an `Memvara` or a `ScopedMemvara`, plus overrides.

    Adapters keep both halves rather than a `ScopedMemvara`, because two of them need the
    parts a scoped view deliberately hides — the predicate registry and the store — and
    a view that exposed those would not be a narrowing any more.

    A `ScopedMemvara` is accepted because a server layer holds one per request and has no
    public way to get the underlying `Memvara` back out; reaching for `_mem` here is the
    price of that, and is noted in the workstream report as a missing accessor rather
    than papered over.
    """
    inner = getattr(memory, "_mem", None)
    if inner is not None:                     # a ScopedMemvara: start from its scope
        base: Scope = memory.scope
        memvara: Memvara = inner
    else:
        memvara = memory
        base = memory.default_scope
    return memvara, Scope(
        tenant if tenant is not None else base.tenant,
        user if user is not None else base.user,
        agent if agent is not None else base.agent,
        session if session is not None else base.session,
    )


def scope_kw(scope: Scope) -> dict[str, Any]:
    """A `Scope` as the four keyword arguments every `Memvara` method takes."""
    return {"tenant": scope.tenant, "user": scope.user, "agent": scope.agent,
            "session": scope.session}


def result_metadata(result: Any) -> dict[str, Any]:
    """Everything memvara knows about one search result, as a flat dict.

    This is where the guarantees survive the crossing. Both LangChain's `Document` and
    LlamaIndex's `TextNode` carry the memory as text plus a free-form metadata mapping,
    and that mapping is the only place left for the triple, the two time axes, the
    ranking explanation and the source turn ids — the difference between a retrieved
    string and a memory you can audit. One function for both frameworks, because two
    copies of this dict is two chances for a downstream filter to key on a field that
    only one of them emits.

    An episode gets a deliberately *smaller* dict, with no subject/predicate/object and
    no confidence. A verbatim thing someone said once is not a fact, and a filter keyed
    on `metadata["predicate"]` must not quietly match one.
    """
    common = {
        "kind": result.kind,
        "score": result.score,
        "why": result.explain.summary(),
    }
    if isinstance(result, EpisodeResult):
        turn: Episode = result.episode
        return {**common, "memvara_id": turn.id, "role": turn.role,
                "ts": turn.ts.isoformat(), "scope": turn.scope.key()}
    claim: Claim = result.claim
    return {
        **common,
        "memvara_id": claim.id,
        "subject": claim.subject,
        "predicate": claim.predicate,
        "object": claim.object,
        "memory_type": claim.memory_type.value,
        "confidence": claim.confidence,
        "salience": claim.salience,
        # Both axes, because collapsing them into one timestamp is the mistake this
        # library exists to refuse — and an adapter is exactly where it would happen.
        "valid_from": claim.valid_from.isoformat(),
        "valid_to": None if claim.valid_to is None else claim.valid_to.isoformat(),
        "recorded_at": claim.recorded_at.isoformat(),
        "invalidated_at": (None if claim.invalidated_at is None
                           else claim.invalidated_at.isoformat()),
        "derivation": claim.derivation.value,
        "extractor": claim.extractor,
        # `why(id)` resolves these to the text. Carrying the ids means a chain can cite
        # its sources without a second round trip it has to decide to make.
        "sources": list(claim.sources),
        "scope": claim.scope.key(),
    }
