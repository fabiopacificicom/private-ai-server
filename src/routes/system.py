import time

from fastapi import APIRouter, HTTPException

import config
import state
from database import get_job_db

router = APIRouter()


@router.get(
    "/health",
    summary="Liveness probe",
    description="Returns overall health, uptime, cache stats, and GPU memory metrics.",
    tags=["system"],
)
async def health():
    uptime = time.time() - config.server_start_time
    gpu_status = "unavailable"
    gpu_alloc_mb = gpu_reserved_mb = None

    if config.torch is not None and config.torch.cuda.is_available():
        try:
            gpu_status = "available"
            idx = config.torch.cuda.current_device() if hasattr(config.torch.cuda, "current_device") else 0
            gpu_alloc_mb = config.torch.cuda.memory_allocated(idx) / (1024**2)
            gpu_reserved_mb = config.torch.cuda.memory_reserved(idx) / (1024**2)
        except Exception:
            gpu_status = "error"
            config.log.exception("Error collecting GPU stats")

    overall = "healthy" if (config.transformers_available or config.LLM is not None) else "degraded"
    active = len(get_job_db().list_jobs(status_filter="running"))
    queued = len(get_job_db().list_jobs(status_filter="queued"))

    return {
        "status": overall,
        "uptime_seconds": int(uptime),
        "models_cached": len(state.model_cache),
        "cache_limit": config.MAX_CACHE_MODELS,
        "downloads_active": active,
        "downloads_queued": queued,
        "torch_version": getattr(config.torch, "__version__", None) if config.torch else None,
        "cuda_available": config.torch.cuda.is_available() if config.torch else False,
        "gpu_status": gpu_status,
        "gpu_memory_allocated_mb": gpu_alloc_mb,
        "gpu_memory_reserved_mb": gpu_reserved_mb,
    }


@router.get(
    "/status",
    summary="Backend availability + cache state",
    description=(
        "Returns which inference backends are importable, the current cache contents, "
        "and any models in cooldown after failed loads."
    ),
    tags=["system"],
)
async def status():
    return {
        "vllm_available": config.LLM is not None,
        "vllm_import_error": config.vllm_import_error or None,
        "transformers_available": config.transformers_available,
        "llama_cpp_available": config.llama_cpp_available,
        "cache_size": len(state.model_cache),
        "cached_models": list(state.model_meta.items()),
        "failed_loads": state.failed_loads,
        "max_cache_models": config.MAX_CACHE_MODELS,
        "cooldown_seconds": config.COOLDOWN_SECONDS,
    }


@router.get(
    "/diag",
    summary="Torch / CUDA diagnostics",
    description="Returns torch version, CUDA availability, and per-device info.",
    tags=["system"],
)
async def diag():
    info = {"torch_installed": config.torch is not None, "torch_version": None,
            "cuda_available": False, "cuda_count": 0, "cuda_devices": []}
    if config.torch is not None:
        try:
            info["torch_version"] = getattr(config.torch, "__version__", None)
            info["cuda_available"] = config.torch.cuda.is_available()
            info["cuda_count"] = config.torch.cuda.device_count() if info["cuda_available"] else 0
            info["cuda_devices"] = [
                {"index": i, "name": config.torch.cuda.get_device_name(i)}
                for i in range(info["cuda_count"])
            ]
        except Exception:
            config.log.exception("Error collecting diag info")
    return info


@router.post(
    "/debug/clear-cooldown",
    summary="Clear cooldown for a model",
    description=(
        "Removes a model from the failed-load cooldown list so it can be retried "
        "immediately. Useful when fixing transient load errors."
    ),
    tags=["system"],
    responses={
        200: {"description": "Cooldown cleared (or no cooldown existed)."},
        400: {"description": "Missing 'model' field."},
    },
)
async def clear_cooldown(payload: dict):
    model_name = payload.get("model")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing 'model'")
    if model_name in state.failed_loads:
        del state.failed_loads[model_name]
        return {"message": f"Cooldown cleared for '{model_name}'"}
    return {"message": f"No cooldown found for '{model_name}'"}
