"""
Unit tests for schemas.py — Pydantic request validation.
"""

import pytest
from pydantic import ValidationError

from schemas import ChatRequest, MultimodalMessage, MultimodalChatRequest


class TestChatRequest:
    def test_minimal_valid_request(self):
        req = ChatRequest(model="gpt2", messages=[{"role": "user", "content": "hi"}])
        assert req.model == "gpt2"
        assert req.max_tokens == 512  # default
        assert req.temperature == 0.7  # default
        assert req.stream is False
        assert req.timeout == 120  # default

    def test_full_request(self):
        req = ChatRequest(
            model="meta-models/Muse-Glimmer-30B",
            messages=[{"role": "system", "content": "be terse"},
                      {"role": "user", "content": "hello"}],
            max_tokens=128,
            temperature=0.3,
            stream=True,
            timeout=60,
        )
        assert req.stream is True
        assert req.timeout == 60

    def test_missing_model_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(messages=[{"role": "user", "content": "hi"}])

    def test_missing_messages_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(model="gpt2")

    def test_empty_messages_accepted_at_schema_level(self):
        """Schema accepts empty messages — app-layer validation handles the rest."""
        req = ChatRequest(model="gpt2", messages=[])
        assert req.messages == []


class TestMultimodalMessage:
    def test_plain_string_content(self):
        m = MultimodalMessage(role="user", content="hello")
        assert m.content == "hello"
        assert m.images is None
        assert m.audio is None
        assert m.video is None

    def test_structured_content_blocks(self):
        m = MultimodalMessage(role="user", content=[
            {"type": "image", "url": "https://example.com/x.jpg"},
            {"type": "text", "text": "describe this"},
        ])
        assert isinstance(m.content, list)
        assert len(m.content) == 2

    def test_with_media_fields(self):
        m = MultimodalMessage(role="user", content="describe", images=["b64..."])
        assert m.images == ["b64..."]

    def test_missing_role_raises(self):
        with pytest.raises(ValidationError):
            MultimodalMessage(content="hi")


class TestMultimodalChatRequest:
    def test_valid_request(self):
        req = MultimodalChatRequest(
            model="meta-models/Muse-Glimmer-30B",
            messages=[MultimodalMessage(role="user", content="hi")],
        )
        assert req.timeout == 180  # default
        assert req.max_tokens == 512

    def test_streaming_flag(self):
        req = MultimodalChatRequest(
            model="x",
            messages=[MultimodalMessage(role="user", content="hi")],
            stream=True,
        )
        assert req.stream is True
