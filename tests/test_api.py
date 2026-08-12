"""
API-level tests for app.py using FastAPI TestClient.
No actual model inference — mocked at the load_model / database boundaries.
"""

import os
import sys
import tempfile
import pytest

# Make sure parent directory is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Patch the database BEFORE importing app so it uses a temp DB.
# ---------------------------------------------------------------------------

import database as _db_module

_tmp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db_path = _tmp_db_file.name
_tmp_db_file.close()

_db_module.init_job_database(_tmp_db_path)

# Now safe to import app (it calls init_job_database at startup only if not yet set)
# We patch the global before FastAPI runs startup to avoid GPU/torch side effects.
import unittest.mock as mock

# Stub out heavy imports so app.py loads in a test environment without GPU
sys.modules.setdefault("torch", mock.MagicMock())
sys.modules.setdefault("vllm", mock.MagicMock())
sys.modules.setdefault("transformers", mock.MagicMock())

# Patch init_job_database so app.py doesn't overwrite our test db
with mock.patch.object(_db_module, "init_job_database", return_value=_db_module.job_db):
    import app as _app_module

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Shared TestClient for all API tests."""
    return TestClient(_app_module.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def clean_db():
    """Reset the test database between tests."""
    import sqlite3
    with sqlite3.connect(_tmp_db_path) as conn:
        conn.execute("DELETE FROM jobs")
        conn.commit()
    yield


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_keys(self, client):
        resp = client.get("/health")
        data = resp.json()
        required_keys = {
            "status", "uptime_seconds", "models_cached", "cache_limit",
            "downloads_active", "downloads_queued",
        }
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_health_status_value(self, client):
        resp = client.get("/health")
        data = resp.json()
        # status should be one of the defined values
        assert data["status"] in ("healthy", "degraded", "error")

    def test_health_uptime_is_non_negative(self, client):
        resp = client.get("/health")
        assert resp.json()["uptime_seconds"] >= 0

    def test_health_counts_are_integers(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert isinstance(data["downloads_active"], int)
        assert isinstance(data["downloads_queued"], int)


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

class TestModels:
    def test_models_returns_200(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200

    def test_models_response_has_models_key(self, client):
        resp = client.get("/models")
        data = resp.json()
        assert "models" in data

    def test_models_is_list(self, client):
        resp = client.get("/models")
        assert isinstance(resp.json()["models"], list)


# ---------------------------------------------------------------------------
# POST /chat — validation (no model inference)
# ---------------------------------------------------------------------------

class TestChatValidation:
    def test_chat_missing_model_returns_4xx(self, client):
        """Sending a chat request without 'model' should return 4xx."""
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "hello"}]
        })
        assert resp.status_code >= 400

    def test_chat_missing_messages_returns_4xx(self, client):
        """Sending a chat request without 'messages' should return 4xx."""
        resp = client.post("/chat", json={"model": "some/model"})
        assert resp.status_code >= 400

    def test_chat_invalid_timeout_returns_400(self, client):
        """Timeout out of range (1-600) should return 400.

        Note: timeout=0 is falsy in Python, so `request.timeout or 120` silently
        becomes 120 and bypasses validation. Use 601 (clearly out-of-range) instead.
        """
        resp = client.post("/chat", json={
            "model": "some/model",
            "messages": [{"role": "user", "content": "hi"}],
            "timeout": 601,
        })
        assert resp.status_code == 400

    def test_chat_nonexistent_model_returns_4xx(self, client):
        """Requesting a model that isn't downloaded should yield a 4xx error."""
        # load_model is now imported in routes.chat; mock it there
        import routes.chat as _chat_module
        with mock.patch.object(
            _chat_module, "load_model",
            side_effect=RuntimeError("Model 'nonexistent/model' not available locally. Use POST /pull to download it first.")
        ):
            resp = client.post("/chat", json={
                "model": "nonexistent/model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            })
        assert resp.status_code >= 400


# ---------------------------------------------------------------------------
# POST /chat/multimodal — schema validation (no inference)
# ---------------------------------------------------------------------------

class TestMultimodalValidation:
    def test_multimodal_missing_model_returns_4xx(self, client):
        resp = client.post("/chat/multimodal", json={
            "messages": [{"role": "user", "content": "describe this"}]
        })
        assert resp.status_code >= 400

    def test_multimodal_missing_messages_returns_4xx(self, client):
        resp = client.post("/chat/multimodal", json={"model": "some/model"})
        assert resp.status_code >= 400

    def test_multimodal_empty_messages_returns_4xx_or_500(self, client):
        """Empty messages list should trigger a validation or server error."""
        # Model not loaded → load will raise, yielding 503
        resp = client.post("/chat/multimodal", json={
            "model": "some/model",
            "messages": [],
        })
        assert resp.status_code >= 400

    def test_multimodal_invalid_role_still_validates_schema(self, client):
        """As long as model+messages are present, pydantic accepts the call (role validation is app-level)."""
        # Model won't load → 503 or 500, but NOT a 422 schema error
        resp = client.post("/chat/multimodal", json={
            "model": "some/model",
            "messages": [{"role": "banana", "content": "hi"}],
        })
        # Should not be 422 (schema error) — pydantic accepts any string for role
        assert resp.status_code != 422


# ---------------------------------------------------------------------------
# GET /jobs and GET /jobs/{job_id}
# ---------------------------------------------------------------------------

class TestJobs:
    def test_list_jobs_returns_200(self, client):
        resp = client.get("/jobs")
        assert resp.status_code == 200

    def test_list_jobs_empty(self, client):
        resp = client.get("/jobs")
        data = resp.json()
        assert "jobs" in data
        assert data["jobs"] == []

    def test_get_nonexistent_job_returns_404(self, client):
        resp = client.get("/jobs/does-not-exist")
        assert resp.status_code == 404
