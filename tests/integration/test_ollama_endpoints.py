"""
Integration tests for the Ollama-compatible endpoints (/api/chat, /api/tags).
These hit a running server and skip if it's unreachable.
"""

import json
import urllib.error
import urllib.request

import pytest

BASE_URL = "http://127.0.0.1:11434"


def _get(path, timeout=10.0):
    try:
        return urllib.request.urlopen(BASE_URL + path, timeout=timeout)
    except urllib.error.URLError:
        pytest.skip("Server not running at " + BASE_URL)


def _post(path, body, timeout=10.0):
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError:
        pytest.skip("Server not running at " + BASE_URL)


class TestApiTags:
    def test_tags_returns_200(self):
        resp = _get("/api/tags")
        assert resp.status == 200

    def test_tags_has_models_list(self):
        resp = _get("/api/tags")
        data = json.loads(resp.read())
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_tags_models_have_name_and_model(self):
        resp = _get("/api/tags")
        data = json.loads(resp.read())
        if data["models"]:
            m = data["models"][0]
            assert "name" in m
            assert "model" in m

    def test_tags_models_have_modality(self):
        resp = _get("/api/tags")
        data = json.loads(resp.read())
        if data["models"]:
            m = data["models"][0]
            assert "modality" in m
            assert m["modality"] in ("chat", "vision", "imagegen", "voice", "embeddings", "unknown")

    def test_tags_has_services(self):
        resp = _get("/api/tags")
        data = json.loads(resp.read())
        assert "services" in data


class TestApiChatValidation:
    def test_chat_missing_model_returns_4xx(self):
        status, _ = _post("/api/chat", {"messages": [{"role": "user", "content": "hi"}]})
        assert status >= 400

    def test_chat_nonexistent_model_returns_error(self):
        status, body = _post("/api/chat", {
            "model": "nonexistent/model-xyz",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }, timeout=60)
        # Should fail at load_model with 4xx/5xx (not a hang)
        assert status >= 400


class TestApiChatNonStreaming:
    def test_chat_returns_ollama_shape(self):
        status, body = _post("/api/chat", {
            "model": "deepseek-r1:1.5b",
            "messages": [{"role": "user", "content": "Say hi"}],
            "stream": False,
        }, timeout=120)
        assert status == 200
        data = json.loads(body)
        assert data.get("done") is True
        assert "message" in data
        assert data["message"]["role"] == "assistant"
        assert "content" in data["message"]


class TestApiChatStreaming:
    def test_streaming_returns_ndjson(self):
        req = urllib.request.Request(
            BASE_URL + "/api/chat",
            data=json.dumps({
                "model": "deepseek-r1:1.5b",
                "messages": [{"role": "user", "content": "Say hi"}],
                "stream": True,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=120)
        except urllib.error.URLError:
            pytest.skip("Server not running at " + BASE_URL)

        raw = resp.read().decode("utf-8")
        lines = [l for l in raw.strip().split("\n") if l.strip()]
        assert len(lines) > 0

        # First chunk should be an Ollama message chunk
        first = json.loads(lines[0])
        assert "message" in first
        assert first["message"]["role"] == "assistant"
        assert first.get("done") is False

        # Last chunk should signal done
        last = json.loads(lines[-1])
        assert last.get("done") is True
