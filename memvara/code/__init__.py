"""Change-aware code memory built on Memvara.

P0 deliberately keeps the code model deterministic: Python AST extraction identifies
files and symbols, stable structural identities survive path changes, and snapshots can
report exactly which symbols were added, changed, removed, or moved. Semantic context is
stored through the normal Memvara claim API by :class:`CodeMemory`.
"""

from .index import CodeIndex, CodeSnapshot, Symbol, SymbolChange, SymbolKind
from .memory import CodeMemory, ContextRecord

__all__ = [
    "CodeIndex",
    "CodeMemory",
    "CodeSnapshot",
    "ContextRecord",
    "Symbol",
    "SymbolChange",
    "SymbolKind",
]
