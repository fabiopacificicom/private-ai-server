"""
Unit tests for the Ollama-compatible route helpers.
"""

import pytest

from routes.ollama import _ollama_response, _extract_options


class TestOllamaResponse:
    def test_basic_response_shape(self):
        resp = _ollama_response("gemma2:latest", "hello")
        assert resp["model"] == "gemma2:latest"
        assert resp["message"] == {"role": "assistant", "content": "hello"}
        assert resp["done"] is True
        assert resp["done_reason"] == "stop"
        assert "created_at" in resp

    def test_carries_through_stats(self):
        stats = {
            "total_duration": 1000,
            "load_duration": 200,
            "prompt_eval_count": 5,
            "eval_count": 20,
            "eval_duration": 800,
        }
        resp = _ollama_response("deepseek-r1:1.5b", "hi", stats=stats)
        assert resp["eval_count"] == 20
        assert resp["eval_duration"] == 800
        assert resp["total_duration"] == 1000
        assert resp["prompt_eval_count"] == 5

    def test_no_stats_when_none(self):
        resp = _ollama_response("gemma2:latest", "hi")
        assert "eval_count" not in resp


class TestExtractOptions:
    def test_empty_options(self):
        assert _extract_options(None) == {}
        assert _extract_options({}) == {}

    def test_maps_temperature(self):
        assert _extract_options({"temperature": 0.3}) == {"temperature": 0.3}

    def test_maps_num_predict_to_max_tokens(self):
        assert _extract_options({"num_predict": 256}) == {"max_tokens": 256}

    def test_maps_both(self):
        opts = _extract_options({"temperature": 0.5, "num_predict": 128})
        assert opts == {"temperature": 0.5, "max_tokens": 128}
