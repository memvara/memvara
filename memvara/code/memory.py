"""Memvara-backed code context storage and invalidation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from ..core import Memvara
from ..schema import Cardinality, PredicateSpec, Volatility
from ..types import Claim, MemoryType, WriteReceipt
from .index import CodeIndex, CodeSnapshot, Symbol, SymbolChange, SymbolKind

CODE_PREDICATES = (
    PredicateSpec("code_context", Cardinality.ONE, Volatility.FAST, MemoryType.SEMANTIC),
    PredicateSpec("code_path", Cardinality.ONE, Volatility.SLOW, MemoryType.SEMANTIC),
    PredicateSpec("code_signature", Cardinality.ONE, Volatility.FAST, MemoryType.SEMANTIC),
    PredicateSpec("code_kind", Cardinality.ONE, Volatility.STATIC, MemoryType.SEMANTIC),
    PredicateSpec("code_source_hash", Cardinality.ONE, Volatility.FAST, MemoryType.SEMANTIC),
    PredicateSpec("code_fingerprint", Cardinality.ONE, Volatility.FAST, MemoryType.SEMANTIC),
    PredicateSpec("code_parent", Cardinality.ONE, Volatility.SLOW, MemoryType.SEMANTIC),
)


@dataclass(frozen=True, slots=True)
class ContextRecord:
    symbol: Symbol
    context: str
    claim: Claim


ContextBuilder = Callable[[Symbol, CodeSnapshot], str]


class CodeMemory:
    """Keep code understanding in Memvara, with deterministic invalidation.

    The code index decides *whether* something changed. The LLM, supplied as a
    ``context_builder``, decides *what the change means*. Memvara owns the resulting
    context, provenance, history, and retirement. This keeps semantic generation off the
    identity/invalidation path.
    """

    def __init__(self, memory: Memvara) -> None:
        self.memory = memory
        for spec in CODE_PREDICATES:
            self.memory.registry.register(spec)

    def sync(
        self,
        index: CodeIndex,
        *,
        previous: CodeSnapshot | None = None,
        context_builder: ContextBuilder | None = None,
        contexts: Mapping[str, str] | None = None,
        commit: str | None = None,
    ) -> tuple[WriteReceipt, ...]:
        """Reconcile one checkout against the previous snapshot.

        ``contexts`` is useful when the LLM already produced a batch. ``context_builder``
        is the lazy form. A moved/renamed symbol with an identical implementation does
        not cause an LLM call: its semantic context remains valid and only its path is
        updated. A changed symbol retires the previous context and records the new one.
        """
        changes = index.diff(previous)
        receipts: list[WriteReceipt] = []
        contexts = contexts or {}

        for change, before, after in changes:
            if change is SymbolChange.UNCHANGED:
                continue

            target = after or before
            if target is None:
                continue

            if change is SymbolChange.REMOVED:
                self._forget_symbol(target.id)
                continue
            if after is None:
                continue

            receipts.extend(self._remember_structure(after, commit=commit))

            # A move or rename with an identical implementation does not invalidate
            # semantic understanding. Only its structural address needs to change.
            if change in (SymbolChange.MOVED, SymbolChange.RENAMED):
                continue

            context = contexts.get(after.id)
            if context is None and context_builder is not None:
                context = context_builder(after, index.snapshot)
            if context:
                receipts.append(self._remember_context(after, context, commit=commit))

        return tuple(receipts)

    def remember_context(
        self,
        symbol: Symbol,
        context: str,
        *,
        commit: str | None = None,
        confidence: float = 1.0,
    ) -> WriteReceipt:
        """Write one semantic context through the normal Memvara write path."""
        return self._remember_context(symbol, context, commit=commit, confidence=confidence)

    def current_context(self, symbol: Symbol | str) -> ContextRecord | None:
        symbol_id = symbol.id if isinstance(symbol, Symbol) else symbol
        claims = self.memory.history(symbol_id, "code_context")
        live = [claim for claim in claims if claim.is_live]
        if not live:
            return None
        claim = live[-1]
        return ContextRecord(
            symbol=symbol if isinstance(symbol, Symbol) else self._symbol_from_claim(claim),
            context=claim.object,
            claim=claim,
        )

    def recall(self, query: str, *, k: int = 10, **scope: str | None) -> list:
        """Retrieve current code context with Memvara's hybrid retrieval."""
        return self.memory.search(query, k=k, memory_types=[MemoryType.SEMANTIC], **scope)

    def _remember_structure(self, symbol: Symbol, *, commit: str | None) -> list[WriteReceipt]:
        values = (
            ("code_path", symbol.path),
            ("code_signature", symbol.signature),
            ("code_kind", symbol.kind.value),
            ("code_source_hash", symbol.source_hash),
            ("code_fingerprint", symbol.fingerprint),
        )
        if symbol.parent_id:
            values += (("code_parent", symbol.parent_id),)
        receipts: list[WriteReceipt] = []
        for predicate, value in values:
            meta = {"code_symbol": symbol.id}
            if commit is not None:
                meta["commit"] = commit
            receipts.append(self.memory.remember(
                symbol.id,
                predicate,
                value,
                text=f"{symbol.qualified_name} {predicate}: {value}",
                extractor="code-indexer",
                **meta,
            ))
        return receipts

    def _remember_context(
        self,
        symbol: Symbol,
        context: str,
        *,
        commit: str | None,
        confidence: float = 1.0,
    ) -> WriteReceipt:
        meta = {"code_symbol": symbol.id, "code_path": symbol.path}
        if commit is not None:
            meta["commit"] = commit
        return self.memory.remember(
            symbol.id,
            "code_context",
            context,
            text=context,
            confidence=confidence,
            memory_type=MemoryType.SEMANTIC,
            extractor="code-context",
            **meta,
        )

    def _forget_symbol(self, symbol_id: str) -> None:
        for predicate in ("code_context", "code_path", "code_signature", "code_kind",
                          "code_source_hash", "code_fingerprint", "code_parent"):
            self.memory.forget(symbol_id, predicate)

    def _symbol_from_claim(self, claim: Claim) -> Symbol:
        """Best-effort symbol shell for callers asking by id after a restart."""
        path_claims = self.memory.history(claim.subject, "code_path")
        path = path_claims[-1].object if path_claims else ""
        kind_claims = self.memory.history(claim.subject, "code_kind")
        raw_kind = kind_claims[-1].object if kind_claims else SymbolKind.FUNCTION.value
        try:
            kind = SymbolKind(raw_kind)
        except ValueError:
            kind = SymbolKind.FUNCTION
        return Symbol(
            id=claim.subject,
            kind=kind,
            name=claim.subject.rsplit(":", 1)[-1],
            qualified_name=claim.subject,
            path=path,
            signature=claim.object,
            source="",
            source_hash="",
            fingerprint="",
            line_start=0,
            line_end=0,
        )
