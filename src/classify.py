"""
Model modality classification and sibling-server availability.

This server serves text + vision-to-text inference. Image generation and
voice/TTS are handled by dedicated sibling servers (Open Fantasia, Olly Voice).
We classify discovered models by modality so the UI can group them and only
present chat/vision models as chat-usable.
"""

import urllib.request
from typing import Dict, Set

# Modality keyword sets (matched case-insensitively on the lowercased model id)
_IMAGEGEN_KEYWORDS: Set[str] = {
    "flux", "stable-diffusion", "sd-turbo", "sd15", "diffusion",
    "imagegen", "image-edit", "image_edit", "z-image", "z-img", "flux2",
}
_VOICE_KEYWORDS: Set[str] = {
    "tts", "cosyvoice", "whisper", "voice", "speech", "audio", "stt",
}
_EMBEDDING_KEYWORDS: Set[str] = {
    "embedding", "embed", "retrieval",
}
_VISION_KEYWORDS: Set[str] = {
    "vl", "vision", "omni", "gemma-4", "qwen2.5vl", "qwen3-vl",
    "nano-omni", "granite3.2-vision",
}

# Map modalitity -> sibling service name (for UI hints)
MODALITY_SERVICE: Dict[str, str] = {
    "imagegen": "Open Fantasia",
    "voice": "Olly Voice",
}


def classify_modality(model_id: str) -> str:
    """Classify a model id into a modality.

    Order matters: imagegen -> voice -> embeddings -> vision -> chat.
    """
    if not model_id:
        return "unknown"
    low = model_id.lower()

    if any(k in low for k in _IMAGEGEN_KEYWORDS):
        return "imagegen"
    if any(k in low for k in _VOICE_KEYWORDS):
        return "voice"
    if any(k in low for k in _EMBEDDING_KEYWORDS):
        return "embeddings"
    if any(k in low for k in _VISION_KEYWORDS):
        return "vision"
    return "chat"


def _probe(url: str, timeout: float = 2.0) -> bool:
    """Return True if the given base URL responds to /health."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout):
            return True
    except Exception:
        return False


def check_services() -> Dict[str, Dict[str, object]]:
    """Probe sibling servers and report availability."""
    import config
    result: Dict[str, Dict[str, object]] = {}
    if config.IMAGE_SERVER_URL:
        result["imagegen"] = {
            "available": _probe(config.IMAGE_SERVER_URL),
            "url": config.IMAGE_SERVER_URL,
            "service": "Open Fantasia",
        }
    if config.VOICE_SERVER_URL:
        result["voice"] = {
            "available": _probe(config.VOICE_SERVER_URL),
            "url": config.VOICE_SERVER_URL,
            "service": "Olly Voice",
        }
    return result
