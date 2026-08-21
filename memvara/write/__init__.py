"""Write path: a cost ladder that keeps the LLM off it wherever possible.

`SalienceGate` -> `FastExtractor` -> batched `llm.extract` -> `Reconciler`, wired
together by `WritePipeline`. Every tier exists to make `WriteReceipt.llm_calls` smaller
without losing facts.
"""

from .fast import FastExtractor
from .gate import SalienceGate
from .pipeline import UnembeddableTextWarning, WritePipeline
from .reconcile import Reconciler, ReconcileResult

__all__ = [
    "SalienceGate",
    "FastExtractor",
    "Reconciler",
    "ReconcileResult",
    "WritePipeline",
    "UnembeddableTextWarning",
]
