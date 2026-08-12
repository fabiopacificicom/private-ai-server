"""
Unit tests for the sibling-server proxy helpers.
"""

import json
from unittest import mock

import pytest

from siblings import (
    SIBLING_INSTALL,
    SiblingUnavailable,
    get_sibling_config,
    proxy_binary,
    proxy_multipart,
    sibling_unavailable_detail,
)


class _FakeResponse:
    def __init__(self, status, body, headers):
        self.status = status
        self._body = body
        self._headers = headers

    def read(self):
        return self._body

    def getheaders(self):
        return list(self._headers.items())


class _FakeConn:
    def __init__(self, response):
        self._response = response
        self.closed = False
        self.sock = mock.Mock()
        self.request_args = None

    def request(self, method, path, body=None, headers=None):
        self.request_args = (method, path, body, headers)

    def getresponse(self):
        return self._response

    def close(self):
        self.closed = True


def _patch_conn(response):
    """Patch http.client.HTTPConnection to return a fake conn with the given response."""
    conn = _FakeConn(response)
    patcher = mock.patch(
        "siblings.http.client.HTTPConnection",
        return_value=conn,
    )
    return patcher, conn


class TestProxyBinary:
    def test_success_returns_raw_bytes_and_headers(self):
        png = b"\x89PNG\r\n\x1a\n" + b"fakepngdata"
        resp = _FakeResponse(200, png, {"Content-Type": "image/png", "X-Saved-Paths": "/tmp/a.png"})
        patcher, conn = _patch_conn(resp)
        with patcher:
            body, headers = proxy_binary("http://127.0.0.1:8765/generate", {"prompt": "hi"})

        assert body == png
        assert headers["content-type"] == "image/png"
        assert headers["x-saved-paths"] == "/tmp/a.png"
        method, path, req_body, req_headers = conn.request_args
        assert method == "POST"
        assert path == "/generate"
        assert req_headers["Content-Type"] == "application/json"
        assert json.loads(req_body) == {"prompt": "hi"}
        assert conn.closed is True

    def test_connection_error_raises_sibling_unavailable(self):
        with mock.patch(
            "siblings.http.client.HTTPConnection",
            side_effect=ConnectionRefusedError("refused"),
        ):
            with pytest.raises(SiblingUnavailable):
                proxy_binary("http://127.0.0.1:8765/generate", {})

    def test_non_2xx_raises_sibling_unavailable_with_detail(self):
        resp = _FakeResponse(503, json.dumps({"detail": "generation in progress"}).encode(),
                             {"Content-Type": "application/json"})
        patcher, _ = _patch_conn(resp)
        with patcher:
            with pytest.raises(SiblingUnavailable) as exc:
                proxy_binary("http://x/generate", {})
        assert "503" in str(exc.value)
        assert "generation in progress" in str(exc.value)


class TestProxyMultipart:
    def test_sends_text_field_and_returns_wav(self):
        wav = b"RIFF....WAVE"
        resp = _FakeResponse(200, wav, {"Content-Type": "audio/wav"})
        patcher, conn = _patch_conn(resp)
        with patcher:
            body, headers = proxy_multipart("http://127.0.0.1:8766/tts", {"text": "hello"})

        assert body == wav
        assert headers["content-type"] == "audio/wav"
        method, path, req_body, req_headers = conn.request_args
        assert method == "POST"
        assert path == "/tts"
        ctype = req_headers["Content-Type"]
        assert ctype.startswith("multipart/form-data; boundary=")
        assert b'name="text"' in req_body
        assert b"hello" in req_body

    def test_connection_error_raises_sibling_unavailable(self):
        with mock.patch(
            "siblings.http.client.HTTPConnection",
            side_effect=ConnectionRefusedError("refused"),
        ):
            with pytest.raises(SiblingUnavailable):
                proxy_multipart("http://127.0.0.1:8766/tts", {"text": "hi"})


class TestSiblingConfig:
    def test_imagegen_config(self):
        cfg = get_sibling_config("imagegen")
        assert cfg["service"] == "Open Fantasia"
        assert cfg["install"] == SIBLING_INSTALL["imagegen"]
        assert cfg["url"]

    def test_voice_config(self):
        cfg = get_sibling_config("voice")
        assert cfg["service"] == "Olly Voice"
        assert cfg["install"] == SIBLING_INSTALL["voice"]
        assert cfg["url"]

    def test_unknown_modality_returns_empty(self):
        assert get_sibling_config("chat") == {}


class TestSiblingUnavailableDetail:
    def test_includes_install_command(self):
        detail = sibling_unavailable_detail("Open Fantasia", "http://127.0.0.1:8765", "git clone x")
        assert "Open Fantasia" in detail
        assert "git clone x" in detail
