"""
Integration tests that hit a running uvicorn instance on http://localhost:8005.
These tests verify the full request/response cycle without loading any models.

Skip automatically if the server isn't running.
"""

import json
import urllib.error
import urllib.request

import pytest

BASE_URL = "http://localhost:8005"


def _get(path: str, timeout: float = 5.0):
    try:
        return urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout)
    except urllib.error.URLError:
        pytest.skip(f"Server not running at {BASE_URL}")


def _post(path: str, body: dict, timeout: float = 5.0):
    """POST and return (status_code, body_bytes). Skips if server unreachable."""
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        # Server reachable but returned an error — that's the test signal
        return e.code, e.read()
    except urllib.error.URLError:
        pytest.skip(f"Server not running at {BASE_URL}")


class TestHealthEndpoint:
    def test_health_ok(self):
        resp = _get("/health")
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "status" in data
        assert "uptime_seconds" in data

    def test_diag_ok(self):
        resp = _get("/diag")
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "cuda_available" in data

    def test_status_ok(self):
        resp = _get("/status")
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "vllm_available" in data
        assert "transformers_available" in data
        assert "llama_cpp_available" in data


class TestModelsEndpoint:
    def test_models_ok(self):
        resp = _get("/models")
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_models_have_required_fields(self):
        resp = _get("/models")
        data = json.loads(resp.read())
        if data["models"]:
            m = data["models"][0]
            assert "model" in m
            assert "backend" in m


class TestJobsEndpoint:
    def test_jobs_ok(self):
        resp = _get("/jobs")
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "jobs" in data

    def test_jobs_404_for_unknown_id(self):
        try:
            urllib.request.urlopen(
                f"{BASE_URL}/jobs/00000000-0000-0000-0000-000000000000",
                timeout=5,
            )
            pytest.fail("Expected 404 for unknown job_id")
        except urllib.error.HTTPError as e:
            assert e.code == 404


class TestChatValidation:
    def test_chat_missing_model_returns_4xx(self):
        status, _ = _post("/chat", {"messages": [{"role": "user", "content": "hi"}]})
        assert status >= 400

    def test_chat_invalid_timeout_returns_400(self):
        status, _ = _post("/chat", {
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "timeout": 9999,
        })
        assert status == 400

    def test_chat_nonexistent_model_returns_4xx(self):
        status, _ = _post("/chat", {
            "model": "nonexistent/model-xyz",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        assert status >= 400


class TestMultimodalValidation:
    def test_multimodal_gguf_rejected(self):
        status, _ = _post("/chat/multimodal", {
            "model": r"E:\fake\path\model.gguf",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert status == 400


class TestDebugEndpoints:
    def test_clear_cooldown_missing_model_returns_400(self):
        status, _ = _post("/debug/clear-cooldown", {})
        assert status == 400

    def test_clear_cooldown_unknown_model_ok(self):
        status, _ = _post("/debug/clear-cooldown", {"model": "never-existed"})
        assert status == 200
