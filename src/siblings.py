"""
Sibling-server HTTP proxy helpers.

This server is the text + vision-to-text brain of a local inference stack.
Image generation (Open Fantasia) and voice (Olly Voice) are handled by dedicated
sibling servers. These helpers forward image/TTS requests to the right sibling
and surface a clear error (with install instructions) when one is unreachable.

Sibling contracts (confirmed against the running services):
- Open Fantasia  POST /generate — JSON body -> raw image bytes (e.g. image/png)
- Olly Voice    POST /tts      — multipart/form-data (text=...) -> wav bytes
"""

import http.client
import json
import urllib.parse
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import config


class SiblingUnavailable(Exception):
    """Raised when a sibling server cannot be reached."""


# Install commands shown to the user when a sibling is not running.
SIBLING_INSTALL: Dict[str, str] = {
    "imagegen": "git clone https://github.com/fabiopacifici-bot/open-fantasia-imagegen",
    "voice": "git clone https://github.com/fabiopacificicom/olly-voice-server",
}

# Short connect timeout so unreachable siblings fail fast (503) instead of
# hanging; the read timeout is longer to allow real generation work.
CONNECT_TIMEOUT: float = 5.0
READ_TIMEOUT: float = 180.0


def _http_post(url: str, body: bytes, content_type: str,
               connect_timeout: float = CONNECT_TIMEOUT,
               read_timeout: float = READ_TIMEOUT) -> Tuple[bytes, Dict[str, str]]:
    """POST a body to a sibling and return (body_bytes, headers).

    Uses a short connect timeout and a separate longer read timeout so an
    unreachable sibling fails fast while a real request can take its time.
    Raises SiblingUnavailable on connection failure or non-2xx.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    try:
        conn = conn_cls(host, port, timeout=connect_timeout)
        try:
            conn.request("POST", path, body=body,
                         headers={"Content-Type": content_type,
                                  "Content-Length": str(len(body))})
            # Extend the socket read timeout now that we're connected
            conn.sock.settimeout(read_timeout)
            resp = conn.getresponse()
            data = resp.read()
            headers = {k.lower(): v for k, v in resp.getheaders()}
            if resp.status < 200 or resp.status >= 300:
                raise SiblingUnavailable(
                    f"Sibling returned {resp.status}: {_maybe_detail(data)}"
                )
            return data, headers
        finally:
            conn.close()
    except SiblingUnavailable:
        raise
    except Exception as e:
        raise SiblingUnavailable(str(e)) from e


def _maybe_detail(data: bytes) -> str:
    """Best-effort extract of a sibling's JSON error detail from a body."""
    try:
        raw = data.decode("utf-8", errors="replace")
        try:
            return json.loads(raw).get("detail", raw[:200])
        except Exception:
            return raw[:200]
    except Exception:
        return ""


def proxy_binary(url: str, payload: Dict[str, Any],
                 connect_timeout: float = CONNECT_TIMEOUT,
                 read_timeout: float = READ_TIMEOUT) -> Tuple[bytes, Dict[str, str]]:
    """POST a JSON payload to a sibling and return (raw_bytes, response_headers).

    Used for image generation (Open Fantasia), which returns raw image bytes
    (not JSON). Raises SiblingUnavailable on connection failure or non-2xx.
    """
    body = json.dumps(payload).encode("utf-8")
    return _http_post(url, body, "application/json",
                      connect_timeout=connect_timeout, read_timeout=read_timeout)


def proxy_multipart(url: str, fields: Dict[str, Any],
                    connect_timeout: float = CONNECT_TIMEOUT,
                    read_timeout: float = READ_TIMEOUT) -> Tuple[bytes, Dict[str, str]]:
    """POST multipart/form-data to a sibling and return (raw_bytes, headers).

    Used for text-to-speech (Olly Voice), which expects a `text` form field and
    returns a wav stream. Raises SiblingUnavailable on connection failure.
    """
    boundary = "----PrivateAIFormBoundary" + urllib.parse.quote(str(id(url)))
    parts = []
    for key, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        parts.append(str(value).encode("utf-8"))
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return _http_post(url, body, content_type,
                      connect_timeout=connect_timeout, read_timeout=read_timeout)


def sibling_unavailable_detail(service: str, url: str, install: str) -> str:
    """Build a human-readable 503 detail for a sibling that isn't running."""
    return (
        f"{service} ({url}) is not running.\n"
        f"Install it with:\n  {install}\n"
        f"Then start it and retry."
    )


def get_sibling_config(modality: str) -> Dict[str, Any]:
    """Return (url, service_name, install_cmd) for a modality, or None if unknown."""
    if modality == "imagegen":
        return {
            "url": config.IMAGE_SERVER_URL,
            "service": "Open Fantasia",
            "install": SIBLING_INSTALL["imagegen"],
        }
    if modality == "voice":
        return {
            "url": config.VOICE_SERVER_URL,
            "service": "Olly Voice",
            "install": SIBLING_INSTALL["voice"],
        }
    return {}
