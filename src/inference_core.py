"""Guarded integration point for the Rust PyO3 inference core (Issues #3/#4).

This module wraps the compiled ``inference_core_rb`` extension so that the
FastAPI server can *optionally* use the Rust continuous-batching /
speculative-decoding core when the platform ships a compatible ``.pyd``
(Windows-hosted Python), and otherwise transparently falls back to the existing
Python sequential pipeline.

Guarantees (matching the issue requirements):
- The import is wrapped in ``try/except``; a missing or broken binary never
  crashes the running server.
- :data:`CORE_AVAILABLE` exposes whether the Rust core loaded. When ``False``,
  :func:`make_coordinator` returns ``None`` and callers keep using the Python
  sequential path.
- Construction of the async :class:`BatchCoordinator` is deferred until first
  use so an import-time failure of the tokio runtime is also contained.

The final Windows artifact (``inference_core_rb.pyd``) must be built on the
Windows host with ``maturin build --release`` (see
``inference_core_rb/README_WINDOWS_BUILD.md``); until that exists, this module
reports the core as unavailable and the server behaviour is unchanged.
"""

import logging

log = logging.getLogger("ai-server")

# ---------------------------------------------------------------------------
# Optional import — mirrors config.py's backend-availability pattern.
# ---------------------------------------------------------------------------

try:
    import inference_core_rb as _rb

    # Force loading of the extension so a broken binary is caught here rather
    # than at first request. The healthcheck symbol is part of the crate ABI.
    _HEALTH = _rb.healthcheck()
    CORE_AVAILABLE: bool = True
    _CORE_LOAD_ERROR: str = ""
except Exception as _e:  # noqa: BLE001 - import must never break the server
    _rb = None
    CORE_AVAILABLE = False
    _CORE_LOAD_ERROR = repr(_e)
    log.warning("Rust inference core unavailable; using Python sequential path. %s", _e)

# Environment override to force-disable even when the binary is present
# (useful for testing the fallback path in CI / on non-Windows boxes).
import os as _os  # noqa: E402

if _os.getenv("PRIVATE_AI_DISABLE_RUST_CORE", "").strip().lower() in {"1", "true", "yes"}:
    CORE_AVAILABLE = False
    _rb = None


def core_load_error() -> str:
    """Human-readable reason the Rust core is unavailable ('' if available)."""
    if CORE_AVAILABLE:
        return ""
    return _CORE_LOAD_ERROR


# ---------------------------------------------------------------------------
# Coordinator factory
# ---------------------------------------------------------------------------


def make_coordinator(max_batch_size: int = 4, window_ms: int = 100,
                     max_context_window: int = 4096):
    """Return a Rust :class:`BatchCoordinator`, or ``None`` if the core is
    unavailable. Callers must treat ``None`` as "use the Python sequential
    path". Construction is lazy so a runtime-init failure is contained here.
    """
    if not CORE_AVAILABLE or _rb is None:
        log.info("Rust batch coordinator disabled (core not loaded); Python fallback active.")
        return None
    try:
        coord = _rb.BatchCoordinator(max_batch_size, window_ms, max_context_window)
        log.info(
            "Rust batch coordinator active (batch=%s window=%sms ctx=%s)",
            max_batch_size, window_ms, max_context_window,
        )
        return coord
    except Exception as _e:  # noqa: BLE001 - containment per issue requirement
        log.warning("Rust batch coordinator failed to init; Python fallback active. %s", _e)
        return None


def make_speculative_stream(request_id: str):
    """Return a Rust :class:`SpeculativeStream`, or ``None`` when unavailable."""
    if not CORE_AVAILABLE or _rb is None:
        return None
    try:
        return _rb.SpeculativeStream(request_id)
    except Exception as _e:  # noqa: BLE001
        log.warning("Rust speculative stream init failed. %s", _e)
        return None


def make_kv_manager(capacity: int = 16):
    """Return a Rust :class:`KVCacheBlockManager`, or ``None`` when unavailable."""
    if not CORE_AVAILABLE or _rb is None:
        return None
    try:
        return _rb.KVCacheBlockManager(capacity)
    except Exception as _e:  # noqa: BLE001
        log.warning("Rust KV manager init failed. %s", _e)
        return None


__all__ = [
    "CORE_AVAILABLE",
    "core_load_error",
    "make_coordinator",
    "make_speculative_stream",
    "make_kv_manager",
]
