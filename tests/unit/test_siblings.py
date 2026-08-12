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
    proxy_json,
    sibling_unavailable_detail,
)


class TestProxyJson:
    def test_success_returns_parsed_json(self):
        resp = mock.Mock()
        resp.read.return_value = json.dumps({"images": ["data:image/png;base64,abc"]}).encode()
        resp.__enter__ = mock.Mock(return_value=resp)
        resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=resp) as m:
            result = proxy_json("http://127.0.0.1:8765/generate", {"prompt": "hi"})

        assert result == {"images": ["data:image/png;base64,abc"]}
        # Verify the request was built correctly
        req = m.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:8765/generate"
        assert req.get_method() == "POST"
        assert req.get_header("Content-type") == "application/json"

    def test_empty_response_returns_empty_dict(self):
        resp = mock.Mock()
        resp.read.return_value = b""
        resp.__enter__ = mock.Mock(return_value=resp)
        resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=resp):
            assert proxy_json("http://x/generate", {}) == {}

    def test_connection_error_raises_sibling_unavailable(self):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=ConnectionRefusedError("refused"),
        ):
            with pytest.raises(SiblingUnavailable):
                proxy_json("http://127.0.0.1:8765/generate", {})

    def test_http_error_raises_sibling_unavailable_with_detail(self):
        import urllib.error
        http_err = urllib.error.HTTPError(
            "http://x/generate", 500, "Internal Server Error", {}, None
        )
        http_err.read = lambda: json.dumps({"detail": "boom"}).encode()
        with mock.patch("urllib.request.urlopen", side_effect=http_err):
            with pytest.raises(SiblingUnavailable) as exc:
                proxy_json("http://x/generate", {})
        assert "500" in str(exc.value)


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
