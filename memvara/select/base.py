"""Selector protocol: the model call a ranked read makes, and nothing about how.

A read stays "no model on the read path" by default (`docs/INTERNALS.md`, invariant 1);
`ranked=True` is the one opt-in that breaks it, and only when the caller configured a
`read_selector`. `HybridRetriever` calls this protocol at one point in its read order —
after the reranker has ordered a tenant's turns, before the first few of them are
returned — and asks it to say which of those turns actually bear on the question.

Two members, deliberately no more: `admit()` bounds concurrent model calls before the
expensive stage runs, `select()` names the kept turns after it has. `Reranker` cannot
express this — it returns exactly one score per document, never drops one, and must be
deterministic — which is why this is its own protocol rather than a third case
`Reranker` grows: a model call can time out, be refused a key, or answer nothing, and
those outcomes are the caller's to render (`hybrid.py`), not this protocol's to hide by
always looking the same.

`ModelSelector`, in `model.py`, is the one real implementation this package ships.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence, runtime_checkable

from ..llm.base import Usage


@dataclass(slots=True, frozen=True)
class Candidate:
    """One turn offered to the selector, in the reranker's own order.

    `id` is the episode id, `when` the turn's own timestamp (not the question's), and
    `text` the whole turn — the selector sees full turns, never the 280-character cut
    an unranked `recall()` applies to a turn it did not keep.
    """

    id: str
    when: datetime
    text: str


@dataclass(slots=True, frozen=True)
class Selected:
    """One turn the model named. `span` is a courtesy, not the ranking.

    `span` is the verbatim excerpt the model copied out, kept only when it is actually a
    substring of the turn's own text — after stripping a leading timestamp the model
    copied in along with it, when stripping is what makes it one — and `None`
    otherwise, with the turn still counted as kept: the ranking is what was measured,
    the span is not. Nothing renders it yet; it travels on the wire for the adaptive
    rendering a later phase adds, so the field exists from the first commit rather than
    being appended after callers already depend on its absence.
    """

    id: str
    span: str | None


@dataclass(slots=True, frozen=True)
class Selection:
    """How a ranked read's model consultation went. Carried on the result, not raised.

    `outcome` is one of `applied`, `fallback`, `unconfigured`, `disabled`,
    `key_rejected` — see the design spec's "The outcomes" for what puts a read in each.
    `reason` only accompanies `fallback` (`timeout`, `error`, `provider`, `malformed`);
    `status` is the provider's HTTP status when there was one. `candidates` is how many
    turns the selector was handed — a tenant whose store yields fewer than `top_n` turns
    is visible here rather than silent — and `kept` is how many it named.
    """

    outcome: str
    reason: str | None = None
    status: int | None = None
    candidates: int = 0
    kept: int = 0


@runtime_checkable
class Selector(Protocol):
    """Admits a ranked read, then names which of its candidate turns answer it.

    `admit()` runs first and has to: a cap taken inside `select()` would leave the
    reranker call in front of it unbounded, which is the thread the cap exists to
    bound. It is a context manager held from admission until the model has answered,
    raising `SelectorRefused` when the read should be served unranked instead (the
    operator's own switch, or a rejected key) and `SelectorBusy` when it should not be
    served at all — the one refusal this protocol has.

    `select()` sees the candidates already in reranked order and returns only the ones
    it kept, in that same order — it does not reorder, and does not filter by `k`; that
    is the caller's job. `ModelSelector.select()`, the implementation this package
    ships, raises rather than returns for everything short of a clean answer — see its
    docstring for exactly what and why.

    `top_n` is a data member, not a method, because the caller — `hybrid.py`'s ranked
    stage — has to slice the reranked turn list down to it *before* calling `select()`,
    not inside this protocol: "The selector carries the model call and `top_n`; the
    retriever carries the reranker and its depth" (design spec, "Where it sits"). A
    `runtime_checkable` protocol checks a data member's presence, not its type, so a
    third-party implementation that gets this wrong fails at the type checker rather
    than at `isinstance`.
    """

    def admit(self) -> AbstractContextManager[None]: ...

    top_n: int

    def select(self, question: str, candidates: Sequence[Candidate], *,
               asked_on: datetime | None = None,
               usage: Usage | None = None) -> Sequence[Selected]: ...


class SelectorRefused(Exception):
    """The read is served unranked. `reason` is `disabled` or `key_rejected`.

    `disabled` comes from `admit()` — an operator's own switch, checked before the
    reranker runs so nothing is spent on a read that will be served unranked anyway.
    `key_rejected` comes from `select()` — the provider answered 401 or 403, which is
    not a fallback: a revoked key that served every ranked read unranked for a month
    with nothing saying so is the failure this exists to surface rather than retry past.
    """

    def __init__(self, reason: str, status: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


class SelectorBusy(Exception):
    """The process-wide admission cap is full. The read is not served at all.

    The one outcome among a ranked read's six that is not served — the caller renders
    the refusal (429 with `Retry-After` over `/v1`, a retry-worded tool error over MCP).
    `ModelSelector.admit()` never raises this; it exists for a wrapper that holds a cap,
    such as the hosted service's.
    """
