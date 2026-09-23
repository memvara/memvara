"""Images, audio and video, turned into text by a model backend.

This module holds no model code. It checks that the backend the caller passed implements
`memvara.llm.base.Multimodal`, and hands the bytes to `describe_image` (for images) or
`transcribe` (for audio and video). The backend decides which media types it accepts and
raises `MediaUnsupported` with the reason for the rest.
"""

from __future__ import annotations

from ..llm.base import Multimodal
from .errors import MediaUnsupported

__all__ = ["media_to_text"]


def media_to_text(data: bytes, mime: str, llm: object | None) -> str:
    """A description of an image, or a transcript of audio or video, from `llm`."""
    if not isinstance(llm, Multimodal):
        given = ("no model backend was given" if llm is None
                 else f"{type(llm).__name__} cannot read media")
        raise MediaUnsupported(
            f"{mime} needs a model backend that reads images, audio or video, and "
            f"{given}. Configure AnthropicLLM for images, or OpenAILLM for images, "
            "audio and video.")
    if mime.startswith("image/"):
        return llm.describe_image(data, mime)
    return llm.transcribe(data, mime)
