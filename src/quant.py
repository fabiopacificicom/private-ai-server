"""Quantization / layer-split helpers for the transformers fallback.

Issue #5 groundwork: a single, testable surface for quantization-mode selection
(TRANSFORMERS_QUANT) and quantized KV-cache plumbing (KV_CACHE_DTYPE), so EXL2 /
AWQ / INT8-KV integration points are explicit without wiring untested engines.

All helpers are guarded and non-breaking; vLLM remains the default engine and is
not touched here (see loader._try_vllm).
"""

from typing import Optional

import config


def should_quantize(forced_quant: str, preferred_quant: Optional[bool],
                    bitsandbytes_available: bool, device: int,
                    size_bytes: Optional[int]) -> bool:
    """Decide whether to 4-bit quantize the transformers fallback load.

    `forced_quant` comes from config.TRANSFORMERS_QUANT:
      auto -> size-based NF4 (>= Q4_THRESHOLD_BYTES) else fp16
      nf4  -> force 4-bit (requires bitsandbytes)
      fp16 / fp32 / none -> force no quantization
      awq / exl2 -> engines NOT bundled; resolved as integration points, so we
                    fall back to the auto NF4 decision (non-breaking).
    """
    if forced_quant == "nf4":
        return True
    if forced_quant in ("fp16", "fp32", "none"):
        return False
    if forced_quant in ("awq", "exl2"):
        # EXL2/AWQ engines (exllamav2 / autoawq) are not bundled; log + fall back
        # to the auto NF4 decision so loading stays non-breaking. See
        # .specs/plans/phase1-implementation.md for the integration point.
        config.log.warning(
            "Quant mode '%s' requested but engine is not bundled; falling back to "
            "auto (NF4 if large). Integration point: .specs/plans/phase1-implementation.md",
            forced_quant,
        )
    if preferred_quant is not None:
        return bool(preferred_quant)
    return (
        bitsandbytes_available and device >= 0
        and size_bytes is not None and size_bytes >= config.Q4_THRESHOLD_BYTES
    )


def kv_cache_dtype_kwargs() -> dict:
    """Generation kwargs for quantized KV-cache when KV_CACHE_DTYPE is set.

    Opt-in and non-breaking: returns {} when unset. Passed into transformers
    generate() calls; models that reject the dtype surface a cache error which
    the existing generate wrappers already degrade gracefully on.
    """
    dtype = getattr(config, "KV_CACHE_DTYPE", "") or ""
    if not dtype:
        return {}
    return {"kv_cache_dtype": dtype}


def build_max_memory() -> dict:
    """Build the max_memory dict for layer-splitting GPU/CPU offload."""
    mm: dict = {}
    if config.torch is not None and config.torch.cuda.is_available():
        try:
            mm[0] = config.MAX_GPU_MEMORY
        except Exception:
            pass
    mm["cpu"] = config.MAX_CPU_MEMORY
    return mm
