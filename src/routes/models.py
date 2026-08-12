import asyncio
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

import config
import state
import model_cache as mcache
from loader import load_model, resolve_model_cache_path, calculate_downloaded_bytes, save_manifest
from database import get_job_db

router = APIRouter()


def get_model_scan_dirs(base_path: str) -> List[str]:
    candidates: List[str] = []

    def _add(path: str) -> None:
        norm = os.path.normpath(path)
        if os.path.isdir(norm) and norm not in candidates:
            candidates.append(norm)

    _add(base_path)
    for sub in ("hub", "huggingface", os.path.join("huggingface", "hub")):
        _add(os.path.join(base_path, sub))
    try:
        for entry in os.listdir(base_path):
            child = os.path.join(base_path, entry)
            if os.path.isdir(child) and entry.lower() in ("hub", "huggingface", "hf", "cache"):
                _add(child)
                _add(os.path.join(child, "hub"))
    except Exception:
        pass
    return candidates


def _resolve_snapshot_path(cache_root: str) -> Optional[str]:
    """Return the most recent snapshot dir inside an HF cache root, or the root itself."""
    snapshots_dir = os.path.join(cache_root, "snapshots")
    if os.path.isdir(snapshots_dir):
        try:
            entries = [
                e for e in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, e))
            ]
            if entries:
                # pick the most recently modified snapshot
                entries.sort(key=lambda e: os.path.getmtime(os.path.join(snapshots_dir, e)), reverse=True)
                return os.path.join(snapshots_dir, entries[0])
        except Exception:
            pass
    return cache_root


def _discover_ollama_models() -> List[Dict[str, Any]]:
    """Discover models from an existing Ollama install.

    Ollama stores models as Docker-registry-style manifests under
    `<store>/manifests/<registry>/<namespace>/<model>/<tag>`, with the actual
    GGUF weights as `sha256-...` blobs under `<store>/blobs/`.

    Returns a list of model entries suitable for /models and /api/tags.
    """
    store = config.OLLAMA_MODELS_DIR
    if not store or not os.path.isdir(store):
        return []

    manifests_dir = os.path.join(store, "manifests")
    blobs_dir = os.path.join(store, "blobs")
    if not os.path.isdir(manifests_dir):
        return []

    import json as _json

    found: List[Dict[str, Any]] = []
    # Walk <store>/manifests/<registry>/<namespace>/<model>/<tag>
    for root, dirs, files in os.walk(manifests_dir):
        for fname in files:
            manifest_path = os.path.join(root, fname)
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = _json.load(f)
            except Exception:
                continue

            # Build model name from the manifest path relative to manifests/
            rel = os.path.relpath(manifest_path, manifests_dir)
            parts = rel.replace("\\", "/").split("/")
            # parts = [registry, namespace, model, tag]
            if len(parts) < 3:
                continue
            model_name_parts = parts[:3]
            # Skip the "library" namespace in the display name (Ollama shows gemma2, not registry/library/gemma2)
            if len(parts) == 4:
                registry, namespace, model, tag = parts
                if namespace in ("library", "hf.co"):
                    name = f"{model}:{tag}"
                else:
                    name = f"{namespace}/{model}:{tag}"
            else:
                name = "/".join(model_name_parts)

            # Find the model layer (GGUF blob)
            model_layer = None
            for layer in (manifest.get("layers") or []):
                if layer.get("mediaType") == "application/vnd.ollama.image.model":
                    model_layer = layer
                    break
            if model_layer is None:
                continue

            digest = model_layer.get("digest", "")  # e.g. sha256:abc...
            blob_name = digest.replace(":", "-")    # e.g. sha256-abc...
            blob_path = os.path.join(blobs_dir, blob_name)
            if not os.path.isfile(blob_path):
                continue

            size = model_layer.get("size") or 0
            found.append({
                "model": name,
                "description": f"Ollama model (GGUF)",
                "loaded": False,
                "backend": "gguf_llama_cpp",
                "size_bytes": size,
                "local_path": blob_path,
                "load_duration": None,
            })
    return found


@router.get(
    "/models",
    summary="List available models",
    description=(
        "Returns an Ollama-compatible list of models. Includes:\n"
        "- Models tracked in the manifest (previously pulled or loaded)\n"
        "- HF snapshot directories discovered under `HF_HOME`\n"
        "- GGUF files discovered under `HF_HOME` and `MODELS_EXTRA_SCAN_DIRS`\n\n"
        "Each entry has: `model`, `backend`, `size_bytes`, `local_path`, `loaded`."
    ),
    tags=["models"],
)
async def list_models():
    models = []
    for name, meta in state.model_meta.items():
        models.append({
            "model": name,
            "description": meta.get("description"),
            "loaded": bool(meta.get("backend")),
            "backend": meta.get("backend"),
            "size_bytes": meta.get("size_bytes"),
            "local_path": meta.get("local_path"),
            "load_duration": meta.get("load_duration_ns"),
        })

    try:
        hf_home = os.getenv("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        seen_repos: set = set()
        seen_guuf: set = set()

        all_scan_dirs = get_model_scan_dirs(hf_home) + config.MODELS_EXTRA_SCAN_DIRS
        for scan_dir in all_scan_dirs:
            for root, dirs, files in os.walk(scan_dir):
                for entry in dirs:
                    if not entry.startswith("models--"):
                        continue
                    repo_id = entry[len("models--"):].replace("--", "/")
                    if repo_id in state.model_meta or repo_id in seen_repos:
                        continue
                    full_path = os.path.join(root, entry)
                    snapshot_path = _resolve_snapshot_path(full_path)
                    # register so /chat can load without re-scanning
                    state.model_meta[repo_id] = {"local_path": snapshot_path}
                    size = 0
                    for r2, _, fs2 in os.walk(full_path):
                        for f2 in fs2:
                            try:
                                size += os.path.getsize(os.path.join(r2, f2))
                            except Exception:
                                pass
                    models.append({
                        "model": repo_id, "description": None, "loaded": False,
                        "backend": None, "size_bytes": size, "local_path": snapshot_path,
                        "load_duration": None,
                    })
                    seen_repos.add(repo_id)

                for fname in files:
                    if not fname.lower().endswith(".gguf"):
                        continue
                    gguf_path = os.path.abspath(os.path.join(root, fname))
                    if gguf_path in seen_guuf:
                        continue
                    try:
                        sz = os.path.getsize(gguf_path)
                    except Exception:
                        sz = None
                    # Surface GGUF files for visibility but mark them as un-loadable
                    # so /chat won't try them. Transformers can't read GGUF, and the
                    # llama-cpp-python backend may not support the model's architecture.
                    models.append({
                        "model": gguf_path, "description": "GGUF (not loadable - requires llama.cpp backend)",
                        "loaded": False, "backend": None, "size_bytes": sz,
                        "local_path": gguf_path, "load_duration": None,
                    })
                    seen_guuf.add(gguf_path)
    except Exception:
        config.log.exception("Error scanning local model cache")

    # Discover models from an existing Ollama install and register them
    # so the Ollama-compatible endpoints (/api/chat, /api/tags) can serve them.
    try:
        ollama_models = _discover_ollama_models()
        for m in ollama_models:
            name = m["model"]
            if name in state.model_meta:
                continue
            state.model_meta[name] = {
                **m,
                "backend": "gguf_llama_cpp",
                "local_path": m["local_path"],
                "size_bytes": m.get("size_bytes"),
            }
            models.append(m)
    except Exception:
        config.log.exception("Error discovering Ollama models")

    save_manifest()
    return {"models": models}


@router.post(
    "/pull",
    summary="Pull a model",
    description=(
        "Download a model from HuggingFace Hub (or register a local GGUF file).\n\n"
        "**Request body:**\n"
        "- `model` (required): HF repo id (e.g. `meta-models/Muse-Glimmer-30B`) "
        "or local path to a `.gguf` file\n"
        "- `quantize` (optional): `auto` | `q4` | `fp16` | `fp32` | `no` (default: `auto`)\n"
        "- `init` (optional): if true, load the model into the cache after download (default: false)\n\n"
        "Returns immediately with `{status, job_id}`. Poll `GET /jobs/{job_id}` for progress."
    ),
    tags=["models"],
    responses={
        200: {"description": "Job accepted (or GGUF registered synchronously)."},
        400: {"description": "Missing 'model' field."},
        500: {"description": "huggingface_hub not available."},
    },
)
async def pull_model(payload: Dict[str, Any]):
    model_name = payload.get("model")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing 'model'")

    if str(model_name).lower().endswith(".gguf"):
        return _register_gguf(model_name)

    quant = payload.get("quantize", "auto")
    raw_init = payload.get("init", False)
    init = raw_init.lower() in ("1", "true", "yes", "y") if isinstance(raw_init, str) else bool(raw_init)

    if not config.hf_hub_available or config.snapshot_download is None:
        raise HTTPException(status_code=500, detail="huggingface_hub not available. pip install huggingface_hub")

    job_id = str(uuid.uuid4())
    get_job_db().create_job({
        "id": job_id, "model": model_name, "quantize": quant, "init": init,
        "status": "queued", "created_at": datetime.utcnow().isoformat() + "Z",
        "started_at": None, "finished_at": None, "error": None,
        "local_path": None, "size_bytes": None, "preferred_quantized": None,
    })
    asyncio.create_task(_background_pull(job_id, model_name, quant, init))
    return {"status": "accepted", "job_id": job_id}


def _register_gguf(model_name: str) -> dict:
    if not config.llama_cpp_available:
        raise HTTPException(status_code=400, detail="llama-cpp-python not installed.")
    gguf_path = os.path.abspath(model_name)
    if not os.path.isfile(gguf_path):
        raise HTTPException(status_code=400, detail=f"GGUF file not found: {gguf_path}")
    state.model_meta[model_name] = {
        **state.model_meta.get(model_name, {}),
        "backend": "gguf_llama_cpp", "local_path": gguf_path,
        "size_bytes": os.path.getsize(gguf_path),
    }
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    get_job_db().create_job({
        "id": job_id, "model": model_name, "quantize": "gguf", "init": False,
        "status": "succeeded", "created_at": now, "started_at": now, "finished_at": now,
        "error": None, "local_path": gguf_path,
        "size_bytes": os.path.getsize(gguf_path), "preferred_quantized": None,
    })
    config.log.info("Registered GGUF: %s -> %s", model_name, gguf_path)
    return {"status": "accepted", "job_id": job_id}


async def _background_pull(job_id: str, model_name: str, quant: str, init: bool) -> None:
    db = get_job_db()
    if not db.get_job(job_id):
        return

    db.update_job(job_id, {
        "status": "running", "started_at": datetime.utcnow().isoformat() + "Z",
        "progress": 0.0, "downloaded_bytes": 0, "total_bytes": None,
    })

    try:
        config.log.info("Pull start: %s (job=%s quant=%s)", model_name, job_id, quant)
        cache_path = resolve_model_cache_path(model_name)

        size_bytes: Optional[int] = None
        try:
            if config.model_info is not None:
                info = await asyncio.to_thread(config.model_info, model_name)
                size_bytes = sum(getattr(s, "size", 0) or 0 for s in (getattr(info, "siblings", []) or []))
        except Exception:
            config.log.exception("Could not probe size for %s", model_name)
        db.update_job(job_id, {"total_bytes": size_bytes})

        await config.download_semaphore.acquire()
        try:
            # Resume any incomplete blobs (huggingface_hub continues .incomplete files).
            # local_dir_use_symlinks=False forces file copies instead of symlinks,
            # avoiding WinError 1314 on Windows (no admin/Developer Mode needed).
            dl_task = asyncio.create_task(
                asyncio.to_thread(
                    config.snapshot_download,
                    repo_id=model_name,
                    local_dir_use_symlinks=False,
                )
            )
            while not dl_task.done():
                await asyncio.sleep(2)
                downloaded = calculate_downloaded_bytes(cache_path)
                progress = min(0.99, downloaded / size_bytes) if size_bytes else None
                db.update_job(job_id, {"downloaded_bytes": downloaded, "progress": progress})
            local_path: str = await dl_task
        finally:
            config.download_semaphore.release()

        db.update_job(job_id, {"local_path": local_path})

        if size_bytes is None:
            try:
                if config.model_info is not None:
                    info = await asyncio.to_thread(config.model_info, model_name)
                    size_bytes = sum(getattr(s, "size", 0) or 0 for s in (getattr(info, "siblings", []) or []))
            except Exception:
                pass

        final_dl = calculate_downloaded_bytes(cache_path)
        db.update_job(job_id, {"total_bytes": size_bytes, "downloaded_bytes": final_dl or 0, "progress": 1.0})

        if quant == "q4":
            preferred_quant: Optional[bool] = True
        elif quant in ("fp32", "fp16", "no"):
            preferred_quant = False
        else:
            preferred_quant = bool(size_bytes and size_bytes >= config.Q4_THRESHOLD_BYTES)

        state.model_meta[model_name] = {
            **state.model_meta.get(model_name, {}),
            "local_path": local_path, "preferred_quantized": preferred_quant, "size_bytes": size_bytes,
        }

        if init:
            try:
                await asyncio.to_thread(load_model, model_name)
            except Exception:
                config.log.exception("Init failed after pull for %s (pull still succeeded)", model_name)

        db.update_job(job_id, {
            "status": "succeeded", "size_bytes": size_bytes, "preferred_quantized": preferred_quant,
            "finished_at": datetime.utcnow().isoformat() + "Z",
        })
        config.log.info("Pull succeeded: %s (job=%s)", model_name, job_id)

    except Exception as e:
        import traceback
        db.update_job(job_id, {
            "status": "failed", "error": str(e), "traceback": traceback.format_exc(),
            "finished_at": datetime.utcnow().isoformat() + "Z",
        })
        config.log.exception("Pull failed: %s (job=%s)", model_name, job_id)
