"""Turning `--system <name>` into a memory system.

Three shipped adapters are named here. Anything else is resolved as a dotted import path
— `mypackage.adapters:build` or `mypackage.adapters.MyMemory` — so a memory system in
another repository is benchmarked without this file knowing it exists, and without a fork.

    python -m benchmarks.agent_memory --system mypackage.adapters:build

A name in the table wins over an import path, which is why the shipped names are short
and contain no dots: a system that adds itself to the table cannot shadow somebody's
module, and a module path cannot silently replace a published result's meaning.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

from .adapters.base import MemorySystem

#: `name -> "module:attribute"`, imported on demand. Lazy because the memvara adapter
#: imports the library and the two baselines do not: `--system naive` must work in an
#: environment where memvara is not installed at all.
BUILTIN: dict[str, str] = {
    "memvara": "benchmarks.agent_memory.adapters.memvara_adapter:build",
    "memvara-graph": "benchmarks.agent_memory.adapters.memvara_adapter:build_graph",
    "naive": "benchmarks.agent_memory.adapters.naive:build",
    "vector-rag": "benchmarks.agent_memory.adapters.vector_rag:build",
}


def _import(target: str) -> Callable[..., Any]:
    module_name, _, attribute = target.partition(":")
    if not attribute:
        module_name, _, attribute = target.rpartition(".")
    if not module_name or not attribute:
        raise ValueError(
            f"{target!r} is neither a known system {sorted(BUILTIN)} nor an importable "
            "path. Use 'package.module:factory' or 'package.module.ClassName'.")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(f"{module_name} has no attribute {attribute!r}") from exc


def build(name: str, **kwargs: Any) -> MemorySystem:
    """Construct the named system.

    The result is checked against `MemorySystem` before it is returned, so an adapter
    missing a method fails here with a clear message rather than three hundred questions
    later with an `AttributeError` inside the run loop.
    """
    factory = _import(BUILTIN.get(name, name))
    system = factory(**kwargs)
    missing = [m for m in ("reset", "remember", "query", "usage", "close")
               if not callable(getattr(system, m, None))]
    if missing:
        raise TypeError(
            f"{name} produced {type(system).__name__}, which is missing "
            f"{', '.join(missing)}. See benchmarks/agent_memory/CONTRIBUTING.md.")
    for attribute in ("name", "version"):
        if not isinstance(getattr(system, attribute, None), str):
            raise TypeError(
                f"{type(system).__name__} needs a string `{attribute}`; it is recorded "
                "in the result file and a run without it cannot be traced to what "
                "produced it.")
    return system  # type: ignore[return-value]


def available() -> list[str]:
    return sorted(BUILTIN)
