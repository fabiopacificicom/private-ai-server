"""
Sibling-server HTTP proxy helpers.

This server is the text + vision-to-text brain of a local inference stack.
Image generation (Open Fantasia) and voice (Olly Voice) are handled by dedicated
sibling servers. These helpers forward image/TTS requests to the right sibling
and surface a clear error (with install instructions) when one is unreachable.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict

import config


class SiblingUnavailable(Exception):
    """Raised when a sibling server cannot be reached."""


# Install commands shown to the user when a sibling is not running.
SIBLING_INSTALL: Dict[str, str] = {
    "imagegen": "git clone https://github.com/fabiopacifici-bot/open-fantasia-imagegen",
    "voice": "git clone https://github.com/fabiopacificicom/olly-voice-server",
}


def proxy_json(url: str, payload: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
    """POST a JSON payload to a sibling server and return the parsed response.

    Raises SiblingUnavailable if the server can't be reached or returns a non-2xx.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        # A reachable sibling that returned an error — surface its detail if any.
        detail = "HTTP error"
        try:
            detail = json.loads(e.read().decode("utf-8")).get("detail", str(e))
        except Exception:
            detail = str(e)
        raise SiblingUnavailable(f"Sibling returned {e.code}: {detail}") from e
    except Exception as e:
        raise SiblingUnavailable(str(e)) from e


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
