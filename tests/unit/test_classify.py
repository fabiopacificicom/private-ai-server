"""
Unit tests for the model modality classification helper.
"""

from unittest import mock

from classify import classify_modality, check_services


class TestClassifyModality:
    def test_chat_model(self):
        assert classify_modality("gemma2:latest") == "chat"
        assert classify_modality("deepseek-r1:1.5b") == "chat"
        assert classify_modality("gpt2") == "chat"

    def test_vision_model(self):
        assert classify_modality("Qwen/Qwen3-VL-2B-Instruct") == "vision"
        assert classify_modality("google/gemma-4-E4B-it") == "vision"
        assert classify_modality("qwen2.5vl:3b") == "vision"

    def test_imagegen_model(self):
        assert classify_modality("black-forest-labs/FLUX.1-dev") == "imagegen"
        assert classify_modality("stabilityai/sd-turbo") == "imagegen"
        assert classify_modality("google/diffusiongemma-26B-A4B-it") == "imagegen"

    def test_voice_model(self):
        assert classify_modality("FunAudioLLM/CosyVoice2-0.5B") == "voice"
        assert classify_modality("Qwen/Qwen3-TTS-12Hz-1.7B-Base") == "voice"
        assert classify_modality("Systran/faster-whisper-base") == "voice"

    def test_embedding_model(self):
        assert classify_modality("google/embeddinggemma-300m") == "embeddings"
        assert classify_modality("nomic-embed-text:latest") == "embeddings"

    def test_empty_and_unknown(self):
        assert classify_modality("") == "unknown"
        assert classify_modality(None) == "unknown"


class TestCheckServices:
    def test_both_available(self):
        with mock.patch("classify._probe", return_value=True):
            result = check_services()
        assert result["imagegen"]["available"] is True
        assert result["imagegen"]["service"] == "Open Fantasia"
        assert result["voice"]["available"] is True
        assert result["voice"]["service"] == "Olly Voice"

    def test_both_unavailable(self):
        with mock.patch("classify._probe", return_value=False):
            result = check_services()
        assert result["imagegen"]["available"] is False
        assert result["voice"]["available"] is False

    def test_mixed_availability(self):
        # imagegen reachable, voice not
        def fake_probe(url):
            return "8765" in url

        with mock.patch("classify._probe", side_effect=fake_probe):
            result = check_services()
        assert result["imagegen"]["available"] is True
        assert result["voice"]["available"] is False

    def test_urls_are_reported(self):
        with mock.patch("classify._probe", return_value=True):
            result = check_services()
        assert result["imagegen"]["url"]
        assert result["voice"]["url"]
        assert "8765" in result["imagegen"]["url"]
        assert "8766" in result["voice"]["url"]
