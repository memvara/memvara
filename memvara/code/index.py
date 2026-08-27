"""Deterministic source index for code memory."""

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
    def __init__(self, snapshot: CodeSnapshot) -> None:
        self.snapshot = snapshot

    @classmethod
    def from_directory(cls, root: str | Path, *, previous: CodeSnapshot | None = None,
                       exclude: Iterable[str] = (".git", ".venv", "venv", "node_modules", "__pycache__")) -> "CodeIndex":
        root_path = Path(root).resolve()
        excluded = set(exclude)
        symbols: dict[str, Symbol] = {}
        files: dict[str, str] = {}
        for path in sorted(root_path.rglob("*.py")):
            if any(part in excluded for part in path.parts):
                continue
            relative = path.relative_to(root_path).as_posix()
            source = path.read_text(encoding="utf-8")
            files[relative] = _sha(source)
            try:
                tree = ast.parse(source, filename=relative, type_comments=True)
            except SyntaxError:
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
        return cls(CodeSnapshot(str(root_path), _extract_symbols(relative, source, tree), {relative: _sha(source)}))

    def diff(self, previous: CodeSnapshot | None) -> tuple[tuple[SymbolChange, Symbol | None, Symbol | None], ...]:
        if previous is None:
            return tuple((SymbolChange.ADDED, None, s) for s in self.snapshot.symbols.values())
        old, new = dict(previous.symbols), dict(self.snapshot.symbols)
        changes: list[tuple[SymbolChange, Symbol | None, Symbol | None]] = []
        for symbol_id in sorted(set(old) | set(new)):
            before, after = old.get(symbol_id), new.get(symbol_id)
            if before is None:
                changes.append((SymbolChange.ADDED, None, after))
            elif after is None:
                changes.append((SymbolChange.REMOVED, before, None))
            elif before.path != after.path:
                changes.append((SymbolChange.MOVED, before, after))
            elif before.name != after.name:
                changes.append((SymbolChange.RENAMED, before, after))
            elif before.source_hash != after.source_hash:
                changes.append((SymbolChange.CHANGED, before, after))
            else:
                changes.append((SymbolChange.UNCHANGED, before, after))
        fingerprints: dict[tuple[SymbolKind, str, str], list[tuple[str, Symbol]]] = {}
        for symbol_id, symbol in new.items():
            fingerprints.setdefault((symbol.kind, symbol.name, symbol.fingerprint), []).append((symbol_id, symbol))
        matched_new: set[str] = set()
        repaired: list[tuple[SymbolChange, Symbol | None, Symbol | None]] = []
        for change, before, after in changes:
            if change is SymbolChange.REMOVED and before is not None:
                candidates = fingerprints.get((before.kind, before.name, before.fingerprint), [])
                candidates = [(sid, s) for sid, s in candidates if sid not in old]
                if len(candidates) == 1:
                    sid, candidate = candidates[0]
                    matched_new.add(sid)
                    repaired.append((SymbolChange.MOVED, before, candidate))
                    continue
            if change is SymbolChange.ADDED and after is not None and after.id in matched_new:
                continue
            repaired.append((change, before, after))
        return tuple(repaired)

def _extract_symbols(path: str, source: str, tree: ast.AST) -> dict[str, Symbol]:
    result: dict[str, Symbol] = {}
    module_name = path[:-3].replace("/", ".") if path.endswith(".py") else path.replace("/", ".")
    module = Symbol(_symbol_id(SymbolKind.MODULE, module_name), SymbolKind.MODULE, Path(path).stem,
                    module_name, path, module_name, source, _sha(source), _fingerprint(tree), 1,
                    max(1, source.count("\n") + 1))
    result[module.id] = module

    def walk(body: list[ast.stmt], prefix: str, parent_id: str | None, in_class: bool = False) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = (SymbolKind.ASYNC_METHOD if isinstance(node, ast.AsyncFunctionDef) and in_class
                        else SymbolKind.METHOD if in_class
                        else SymbolKind.ASYNC_FUNCTION if isinstance(node, ast.AsyncFunctionDef)
                        else SymbolKind.FUNCTION)
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
                            variable = _variable_from_node(SymbolKind.CLASS_VARIABLE, target, child, path, source, qualified, symbol.id)
                            result[variable.id] = variable
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for target in _assignment_names(node):
                    variable = _variable_from_node(SymbolKind.VARIABLE, target, node, path, source, prefix, parent_id)
                    result[variable.id] = variable
    walk(getattr(tree, "body", []), module_name, module.id)
    return result

def _symbol_from_node(kind: SymbolKind, node: ast.AST, path: str, source: str, qualified: str, parent_id: str | None) -> Symbol:
    segment = ast.get_source_segment(source, node) or ""
    name = node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else qualified.rsplit(".", 1)[-1]
    return Symbol(_symbol_id(kind, qualified), kind, name, qualified, path, _signature(node), segment,
                  _sha(segment), _fingerprint(node), getattr(node, "lineno", 1),
                  getattr(node, "end_lineno", getattr(node, "lineno", 1)), parent_id)

def _variable_from_node(kind: SymbolKind, name: str, node: ast.AST, path: str, source: str,
                        prefix: str, parent_id: str | None) -> Symbol:
    qualified = f"{prefix}.{name}"
    segment = ast.get_source_segment(source, node) or ""
    return Symbol(_symbol_id(kind, qualified), kind, name, qualified, path, name, segment,
                  _sha(segment), _fingerprint(node), getattr(node, "lineno", 1),
                  getattr(node, "end_lineno", getattr(node, "lineno", 1)), parent_id)

def _assignment_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return ()
    names: list[str] = []
    def collect(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                collect(item)
    for target in targets:
        collect(target)
    return tuple(names)

def _signature(node: ast.AST) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"class {node.name}" if isinstance(node, ast.ClassDef) else type(node).__name__
    args = ast.unparse(node.args)
    return_type = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{prefix}def {node.name}({args}){return_type}"

def _fingerprint(node: ast.AST) -> str:
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
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    return candidate.resolve().relative_to(root.resolve()).as_posix()

def _carry_forward_moves(previous: CodeSnapshot, current: CodeSnapshot) -> CodeSnapshot:
    old_by_key: dict[tuple[SymbolKind, str, str], list[Symbol]] = {}
    for symbol in previous.symbols.values():
        old_by_key.setdefault((symbol.kind, symbol.name, symbol.fingerprint), []).append(symbol)
    replacements: dict[str, Symbol] = {}
    used: set[str] = set()
    for new_id, symbol in current.symbols.items():
        if new_id in previous.symbols:
            continue
        candidates = [s for s in old_by_key.get((symbol.kind, symbol.name, symbol.fingerprint), []) if s.id not in used]
        if len(candidates) == 1:
            old = candidates[0]
            used.add(old.id)
            replacements[new_id] = Symbol(old.id, symbol.kind, symbol.name, symbol.qualified_name,
                                           symbol.path, symbol.signature, symbol.source, symbol.source_hash,
                                           symbol.fingerprint, symbol.line_start, symbol.line_end, symbol.parent_id)
    if not replacements:
        return current
    parent_map = {new_id: replacement.id for new_id, replacement in replacements.items()}
    merged = dict(current.symbols)
    for new_id, replacement in replacements.items():
        del merged[new_id]
        parent_id = parent_map.get(replacement.parent_id, replacement.parent_id)
        merged[replacement.id] = Symbol(replacement.id, replacement.kind, replacement.name,
                                         replacement.qualified_name, replacement.path, replacement.signature,
                                         replacement.source, replacement.source_hash, replacement.fingerprint,
                                         replacement.line_start, replacement.line_end, parent_id)
    return CodeSnapshot(current.root, merged, current.files)
