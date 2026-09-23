"""The `Multimodal` protocol on `AnthropicLLM` and `OpenAILLM`: request shapes and refusals.

Offline against injected fake clients, with no key and no SDK. Every refusal test also
checks that the fake client was never called, because a refusal is only useful if it
comes before the request that would have been billed or rejected.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from memvara.ingest import MediaUnsupported
from memvara.llm import Multimodal
from memvara.llm.anthropic import AnthropicLLM
from memvara.llm.base import (DESCRIBE_IMAGE_MAX_TOKENS, DESCRIBE_IMAGE_PROMPT,
                              DESCRIBE_IMAGE_SYSTEM)
from memvara.llm.openai import OpenAILLM

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


class Recorder:
    """A `create` method that records its keyword arguments and returns `reply`."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def anthropic(reply_text="A screenshot of a login form."):
    messages = Recorder(SimpleNamespace(
        content=[SimpleNamespace(type="text", text=f"  {reply_text}\n")]))
    return AnthropicLLM(client=SimpleNamespace(messages=messages)), messages


def openai(reply=None, transcript=None, **kw):
    completions = Recorder(reply if reply is not None else {
        "choices": [{"message": {"content": " A diagram of three services. "}}]})
    transcriptions = Recorder(transcript if transcript is not None
                              else SimpleNamespace(text=" hello there "))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions),
                             audio=SimpleNamespace(transcriptions=transcriptions))
    return OpenAILLM(client=client, **kw), completions, transcriptions


def test_both_backends_are_multimodal():
    assert isinstance(anthropic()[0], Multimodal)
    assert isinstance(openai()[0], Multimodal)


# -- AnthropicLLM -------------------------------------------------------------------


class TestAnthropic:
    def test_an_image_is_sent_as_base64_with_the_description_instructions(self):
        llm, messages = anthropic()
        assert llm.describe_image(PNG, "image/png") == "A screenshot of a login form."
        assert len(messages.calls) == 1
        call = messages.calls[0]
        assert call["model"] == llm.model
        assert call["system"] == DESCRIBE_IMAGE_SYSTEM
        assert call["max_tokens"] == DESCRIBE_IMAGE_MAX_TOKENS
        image, prompt = call["messages"][0]["content"]
        assert image == {"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.b64encode(PNG).decode("ascii")}}
        assert prompt == {"type": "text", "text": DESCRIBE_IMAGE_PROMPT}

    def test_an_image_type_the_api_does_not_accept_is_refused(self):
        llm, messages = anthropic()
        with pytest.raises(MediaUnsupported, match="not image/tiff"):
            llm.describe_image(b"II*\x00", "image/tiff")
        assert messages.calls == []

    def test_an_image_over_five_megabytes_is_refused(self):
        llm, messages = anthropic()
        with pytest.raises(MediaUnsupported, match="at most 5242880 bytes"):
            llm.describe_image(b"\x00" * (5 * 1024 * 1024 + 1), "image/jpeg")
        assert messages.calls == []

    @pytest.mark.parametrize("mime", ["audio/mpeg", "video/mp4"])
    def test_audio_and_video_are_refused_with_the_reason(self, mime):
        llm, messages = anthropic()
        with pytest.raises(MediaUnsupported) as caught:
            llm.transcribe(b"\x00", mime)
        assert caught.value.code == "media_unsupported"
        assert "does not accept audio or video" in caught.value.reason
        assert "OpenAILLM" in caught.value.reason
        assert messages.calls == []


# -- OpenAILLM ----------------------------------------------------------------------


class TestOpenAIImages:
    def test_an_image_is_sent_as_a_data_url_with_the_description_instructions(self):
        llm, completions, transcriptions = openai()
        assert llm.describe_image(PNG, "image/png") == "A diagram of three services."
        assert len(completions.calls) == 1 and transcriptions.calls == []
        call = completions.calls[0]
        assert call["model"] == llm.model
        assert call["max_completion_tokens"] == DESCRIBE_IMAGE_MAX_TOKENS
        system, user = call["messages"]
        assert system == {"role": "system", "content": DESCRIBE_IMAGE_SYSTEM}
        text, image = user["content"]
        assert text == {"type": "text", "text": DESCRIBE_IMAGE_PROMPT}
        encoded = base64.b64encode(PNG).decode("ascii")
        assert image == {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{encoded}"}}
        assert "extra_body" not in call and "timeout" not in call

    def test_extra_body_and_timeout_are_forwarded(self):
        llm, completions, _ = openai(extra_body={"k": 1}, timeout=30.0)
        llm.describe_image(PNG, "image/webp")
        assert completions.calls[0]["extra_body"] == {"k": 1}
        assert completions.calls[0]["timeout"] == 30.0

    def test_a_refusal_comes_back_empty(self):
        llm, _, _ = openai(reply={"choices": [{"message": {
            "content": None, "refusal": "I can't help with that."}}]})
        assert llm.describe_image(PNG, "image/png") == ""

    def test_an_image_type_the_api_does_not_accept_is_refused(self):
        llm, completions, _ = openai()
        with pytest.raises(MediaUnsupported, match="not image/svg\\+xml"):
            llm.describe_image(b"<svg/>", "image/svg+xml")
        assert completions.calls == []

    def test_an_image_over_twenty_megabytes_is_refused(self):
        llm, completions, _ = openai()
        with pytest.raises(MediaUnsupported, match="at most 20971520 bytes"):
            llm.describe_image(b"\x00" * (20 * 1024 * 1024 + 1), "image/png")
        assert completions.calls == []


class TestOpenAITranscription:
    @pytest.mark.parametrize("mime, filename", [
        ("audio/mpeg", "upload.mp3"),
        ("audio/wav", "upload.wav"),
        ("audio/m4a", "upload.m4a"),
        ("video/mp4", "upload.mp4"),
        ("video/webm", "upload.webm"),
    ])
    def test_the_file_is_named_with_the_extension_the_endpoint_reads(self, mime, filename):
        llm, completions, transcriptions = openai()
        assert llm.transcribe(b"\x01\x02", mime) == "hello there"
        assert completions.calls == []
        assert transcriptions.calls == [{"model": "whisper-1",
                                         "file": (filename, b"\x01\x02", mime)}]

    def test_the_transcription_model_and_timeout_are_the_callers(self):
        llm, _, transcriptions = openai(transcription_model="gpt-4o-transcribe",
                                        timeout=12.0)
        llm.transcribe(b"\x00", "audio/flac")
        assert transcriptions.calls[0]["model"] == "gpt-4o-transcribe"
        assert transcriptions.calls[0]["timeout"] == 12.0

    @pytest.mark.parametrize("reply", ["  plain string  ", {"text": " from a dict "}])
    def test_string_and_dict_replies_are_read(self, reply):
        llm, _, _ = openai(transcript=reply)
        assert llm.transcribe(b"\x00", "audio/ogg") in ("plain string", "from a dict")

    def test_a_reply_with_no_text_is_empty(self):
        llm, _, _ = openai(transcript=SimpleNamespace(text=None))
        assert llm.transcribe(b"\x00", "audio/webm") == ""

    @pytest.mark.parametrize("mime", ["video/quicktime", "video/x-matroska", "audio/aiff"])
    def test_a_container_the_endpoint_does_not_accept_is_refused(self, mime):
        llm, _, transcriptions = openai()
        with pytest.raises(MediaUnsupported) as caught:
            llm.transcribe(b"\x00", mime)
        assert mime in caught.value.reason and "convert the file" in caught.value.reason
        assert transcriptions.calls == []

    def test_a_recording_over_twenty_five_megabytes_is_refused(self):
        llm, _, transcriptions = openai()
        with pytest.raises(MediaUnsupported, match="at most 26214400 bytes"):
            llm.transcribe(b"\x00" * (25 * 1024 * 1024 + 1), "audio/mpeg")
        assert transcriptions.calls == []
