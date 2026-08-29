"""Deterministic source index for code memory.

P0 intentionally starts with Python's standard-library ``ast`` module. The index is not
an LLM summary and not a vector index. It gives Memvara stable-ish semantic anchors,
source fingerprints, and explicit change events. A later parser backend can implement the
same small model for other languages.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping


class SymbolKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    METHOD = "method"
    ASYNC_METHOD = "async_method"
    CLASS_VARIABLE = "class_variable"
    VARIABLE = "variable"


class SymbolChange(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    MOVED = "moved"
    RENAMED = "renamed"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class Symbol:
    """One code symbol and the deterministic facts we can know without an LLM."""

    id: str
    kind: SymbolKind
    name: str
    qualified_name: str
    path: str
    signature: str
    source: str
    source_hash: str
    fingerprint: str
    line_start: int
    line_end: int
    parent_id: str | None = None

    @property
    def address(self) -> str:
        """Human-readable address; deliberately not used as the durable identity."""
        return f"{self.path}:{self.qualified_name}"


@dataclass(frozen=True, slots=True)
class CodeSnapshot:
    root: str
    symbols: Mapping[str, Symbol]
    files: Mapping[str, str]

    @classmethod
    def empty(cls, root: str = ".") -> "CodeSnapshot":
        return cls(str(Path(root).resolve()), {}, {})

    def symbol(self, symbol_id: str) -> Symbol | None:
        return self.symbols.get(symbol_id)

    def by_path(self, path: str) -> tuple[Symbol, ...]:
        normalized = _relative(path, Path(self.root))
        return tuple(s for s in self.symbols.values() if s.path == normalized)


class CodeIndex:
    """Build a deterministic symbol index from a Python repository."""

    def __init__(self, snapshot: CodeSnapshot) -> None:
        self.snapshot = snapshot

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        previous: CodeSnapshot | None = None,
        exclude: Iterable[str] = (".git", ".venv", "venv", "node_modules", "__pycache__"),
    ) -> "CodeIndex":
        root_path = Path(root).resolve()
        excluded = set(exclude)
        symbols: dict[str, Symbol] = {}
        files: dict[str, str] = {}

        for path in sorted(root_path.rglob("*.py")):
            if any(part in excluded for part in path.parts):
                continue
            relative = path.relative_to(root_path).as_posix()
            source = path.read_text(encoding="utf-8")
            file_hash = _sha(source)
            files[relative] = file_hash
            try:
                tree = ast.parse(source, filename=relative, type_comments=True)
            except SyntaxError:
                # A code memory index should not prevent an agent from working on a
                # broken checkout. The file stays visible in ``files``; its symbols are
                # simply unavailable until the syntax is valid again.
                continue
            symbols.update(_extract_symbols(relative, source, tree))

        snapshot = CodeSnapshot(str(root_path), symbols, files)
        if previous is not None:
            snapshot = _carry_forward_moves(previous, snapshot)
        return cls(snapshot)

    @classmethod
    def from_file(cls, path: str | Path, *, root: str | Path | None = None) -> "CodeIndex":
        file_path = Path(path).resolve()
        root_path = Path(root).resolve() if root is not None else file_path.parent
        source = file_path.read_text(encoding="utf-8")
        relative = file_path.relative_to(root_path).as_posix()
        try:
            tree = ast.parse(source, filename=relative, type_comments=True)
        except SyntaxError:
            tree = ast.Module(body=[], type_ignores=[])
        symbols = _extract_symbols(relative, source, tree)
        return cls(CodeSnapshot(str(root_path), symbols, {relative: _sha(source)}))

    def diff(self, previous: CodeSnapshot | None) -> tuple[tuple[SymbolChange, Symbol | None, Symbol | None], ...]:
        """Compare snapshots using symbol ids first, then unique source fingerprints."""
        if previous is None:
            return tuple((SymbolChange.ADDED, None, s) for s in self.snapshot.symbols.values())

        old = dict(previous.symbols)
        new = dict(self.snapshot.symbols)
        changes: list[tuple[SymbolChange, Symbol | None, Symbol | None]] = []

        for symbol_id in sorted(set(old) | set(new)):
            before, after = old.get(symbol_id), new.get(symbol_id)
            if before is None:
                changes.append((SymbolChange.ADDED, None, after))
                continue
            if after is None:
                changes.append((SymbolChange.REMOVED, before, None))
                continue
            if before.path != after.path:
                changes.append((SymbolChange.MOVED, before, after))
            elif before.name != after.name:
                changes.append((SymbolChange.RENAMED, before, after))
            elif before.source_hash != after.source_hash:
                changes.append((SymbolChange.CHANGED, before, after))
            else:
                changes.append((SymbolChange.UNCHANGED, before, after))

        # A moved/renamed symbol can have a new qualified-name id. Collapse the matching
        # remove+add pair when the implementation fingerprint is unique. This is the
        # important P0 rule: path is metadata, not identity.
        matched_old: set[str] = set()
        matched_new: set[str] = set()
        fingerprints: dict[str, list[tuple[str, Symbol]]] = {}
        for symbol_id, symbol in new.items():
            fingerprints.setdefault(symbol.fingerprint, []).append((symbol_id, symbol))
        repaired: list[tuple[SymbolChange, Symbol | None, Symbol | None]] = []
        for change, before, after in changes:
            if change is SymbolChange.REMOVED and before is not None:
                candidates = fingerprints.get(before.fingerprint, [])
                candidates = [(sid, s) for sid, s in candidates if sid not in old]
                if len(candidates) == 1:
                    _, candidate = candidates[0]
                    matched_old.add(before.id)
                    matched_new.add(candidate.id)
                    repaired.append((
                        SymbolChange.RENAMED if before.name != candidate.name else SymbolChange.MOVED,
                        before,
                        candidate,
                    ))
                    continue
            if change is SymbolChange.ADDED and after is not None and after.id in matched_new:
                continue
            repaired.append((change, before, after))
        return tuple(repaired)


def _extract_symbols(path: str, source: str, tree: ast.AST) -> dict[str, Symbol]:
    result: dict[str, Symbol] = {}
    module_name = path[:-3].replace("/", ".") if path.endswith(".py") else path.replace("/", ".")
    module = Symbol(
        id=_symbol_id(SymbolKind.MODULE, module_name),
        kind=SymbolKind.MODULE,
        name=Path(path).stem,
        qualified_name=module_name,
        path=path,
        signature=module_name,
        source=source,
        source_hash=_sha(source),
        fingerprint=_fingerprint(tree),
        line_start=1,
        line_end=max(1, source.count("\n") + 1),
    )
    result[module.id] = module

    def walk(body: list[ast.stmt], prefix: str, parent_id: str | None, in_class: bool = False) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = (
                    SymbolKind.ASYNC_METHOD if isinstance(node, ast.AsyncFunctionDef) and in_class
                    else SymbolKind.METHOD if in_class
                    else SymbolKind.ASYNC_FUNCTION if isinstance(node, ast.AsyncFunctionDef)
                    else SymbolKind.FUNCTION
                )
                qualified = f"{prefix}.{node.name}"
                symbol = _symbol_from_node(kind, node, path, source, qualified, parent_id)
                result[symbol.id] = symbol
                walk(node.body, qualified, symbol.id, False)
            elif isinstance(node, ast.ClassDef):
                qualified = f"{prefix}.{node.name}"
                symbol = _symbol_from_node(SymbolKind.CLASS, node, path, source, qualified, parent_id)
                result[symbol.id] = symbol
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        walk([child], qualified, symbol.id, True)
                    elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                        for target in _assignment_names(child):
                            variable = _variable_from_node(
                                SymbolKind.CLASS_VARIABLE, target, child, path, source, qualified, symbol.id
                            )
                            result[variable.id] = variable
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for target in _assignment_names(node):
                    variable = _variable_from_node(
                        SymbolKind.VARIABLE, target, node, path, source, prefix, parent_id
                    )
                    result[variable.id] = variable

    walk(getattr(tree, "body", []), module_name, module.id)
    return result


def _symbol_from_node(kind: SymbolKind, node: ast.AST, path: str, source: str,
                      qualified: str, parent_id: str | None) -> Symbol:
    segment = ast.get_source_segment(source, node) or ""
    signature = _signature(node)
    return Symbol(
        id=_symbol_id(kind, qualified),
        kind=kind,
        name=getattr(node, "name", qualified.rsplit(".", 1)[-1]),
        qualified_name=qualified,
        path=path,
        signature=signature,
        source=segment,
        source_hash=_sha(segment),
        fingerprint=_fingerprint(node),
        line_start=getattr(node, "lineno", 1),
        line_end=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
        parent_id=parent_id,
    )


def _variable_from_node(kind: SymbolKind, name: str, node: ast.AST, path: str, source: str,
                        prefix: str, parent_id: str | None) -> Symbol:
    qualified = f"{prefix}.{name}"
    segment = ast.get_source_segment(source, node) or ""
    return Symbol(
        id=_symbol_id(kind, qualified), kind=kind, name=name, qualified_name=qualified,
        path=path, signature=name, source=segment, source_hash=_sha(segment),
        fingerprint=_fingerprint(node), line_start=getattr(node, "lineno", 1),
        line_end=getattr(node, "end_lineno", getattr(node, "lineno", 1)), parent_id=parent_id,
    )


def _assignment_names(node: ast.AST) -> tuple[str, ...]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
    return tuple(names)


def _signature(node: ast.AST) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if isinstance(node, ast.ClassDef):
            return f"class {node.name}"
        return type(node).__name__
    args = ast.unparse(node.args)
    return_type = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{prefix}def {node.name}({args}){return_type}"


def _fingerprint(node: ast.AST) -> str:
    """Hash structure while ignoring source locations and declaration names."""
    clone = ast.parse(ast.unparse(node))
    for item in ast.walk(clone):
        for field in ("lineno", "col_offset", "end_lineno", "end_col_offset", "type_comment"):
            if hasattr(item, field):
                setattr(item, field, None)
    return _sha(ast.dump(clone, annotate_fields=True, include_attributes=False))


def _symbol_id(kind: SymbolKind, qualified_name: str) -> str:
    return "code:symbol:" + _sha(f"{kind.value}\0{qualified_name}")[:32]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relative(path: str | Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root).as_posix()


def _carry_forward_moves(previous: CodeSnapshot, current: CodeSnapshot) -> CodeSnapshot:
    """Reuse the old id for unique exact-fingerprint moves/renames."""
    old_by_fp: dict[str, list[Symbol]] = {}
    for symbol in previous.symbols.values():
        old_by_fp.setdefault(symbol.fingerprint, []).append(symbol)
    replacements: dict[str, Symbol] = {}
    used: set[str] = set()
    for new_id, symbol in current.symbols.items():
        if new_id in previous.symbols:
            continue
        candidates = [s for s in old_by_fp.get(symbol.fingerprint, []) if s.id not in used]
        if len(candidates) == 1:
            old = candidates[0]
            used.add(old.id)
            replacements[new_id] = Symbol(
                id=old.id, kind=symbol.kind, name=symbol.name,
                qualified_name=symbol.qualified_name, path=symbol.path,
                signature=symbol.signature, source=symbol.source,
                source_hash=symbol.source_hash, fingerprint=symbol.fingerprint,
                line_start=symbol.line_start, line_end=symbol.line_end,
                parent_id=symbol.parent_id,
            )
    if not replacements:
        return current
    merged = dict(current.symbols)
    for new_id, replacement in replacements.items():
        del merged[new_id]
        merged[replacement.id] = replacement
    return CodeSnapshot(current.root, merged, current.files)
