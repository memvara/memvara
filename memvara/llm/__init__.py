"""LLM backends. `NullLLM` is the default; everything else is opt-in."""

from __future__ import annotations

from typing import Any

from .base import LLM, Chat, NullLLM, Usage

# `Usage` is exported because implementing `LLM` outside this package requires it:
# a backend that sets `reports_usage` is handed one and has to type against it. `Chat`
# is exported for the same reason `memvara.select.model` needs it: a third-party backend
# that wants to work with `ModelSelector` implements this, not `LLM`.
__all__ = ["LLM", "Chat", "NullLLM", "Usage", "AnthropicLLM", "OpenAILLM"]


def __getattr__(name: str) -> Any:
    # PEP 562 lazy attribute: the hosted backends are reachable from this package
    # without their SDKs being importable, so the no-API-key default configuration keeps
    # working and paying an SDK's import cost stays opt-in.
    if name == "AnthropicLLM":
        from .anthropic import AnthropicLLM

        return AnthropicLLM
    if name == "OpenAILLM":
        from .openai import OpenAILLM

        return OpenAILLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
