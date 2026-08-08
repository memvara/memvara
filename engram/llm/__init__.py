"""LLM backends. `NullLLM` is the default; everything else is opt-in."""

from __future__ import annotations

from typing import Any

from .base import LLM, NullLLM

__all__ = ["LLM", "NullLLM", "AnthropicLLM"]


def __getattr__(name: str) -> Any:
    # PEP 562 lazy attribute: `AnthropicLLM` is reachable from this package without the
    # `anthropic` SDK being importable, so the no-API-key default configuration keeps
    # working and paying the SDK's import cost stays opt-in.
    if name == "AnthropicLLM":
        from .anthropic import AnthropicLLM

        return AnthropicLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
