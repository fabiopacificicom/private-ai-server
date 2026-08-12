"""
Integration tests for the sibling-server orchestration endpoints
(/services, /generate, /tts). These hit a running server and skip if unreachable.
"""

import json
import urllib.error
import urllib.request

import pytest

BASE_URL = "http://127.0.0.1:8005"


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


class TestServices:
    def test_services_returns_200(self):
        resp = _get("/services")
        assert resp.status == 200

    def test_services_has_imagegen_and_voice(self):
        resp = _get("/services")
        data = json.loads(resp.read())
        assert "imagegen" in data
        assert "voice" in data

    def test_services_have_install_commands(self):
        resp = _get("/services")
        data = json.loads(resp.read())
        for key in ("imagegen", "voice"):
            assert "install" in data[key]
            assert "git clone" in data[key]["install"]


class TestGenerate:
    def test_generate_returns_503_when_sibling_down(self):
        # Open Fantasia is not expected to be running in CI
        status, body = _post("/generate", {"model": "FLUX.1", "prompt": "a cat"})
        assert status == 503
        data = json.loads(body)
        assert "Open Fantasia" in data["detail"]


class TestTTS:
    def test_tts_returns_503_when_sibling_down(self):
        # Olly Voice is not expected to be running in CI
        status, body = _post("/tts", {"model": "CosyVoice2", "text": "hello"})
        assert status == 503
        data = json.loads(body)
        assert "Olly Voice" in data["detail"]
