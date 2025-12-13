import gc
import os
import logging
import time
import uuid
import asyncio
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import OrderedDict

try:
    import torch
except Exception:
    torch = None

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from vllm import LLM, SamplingParams
except Exception:
    # vLLM may not be installed in some environments; keep names for typing
    LLM = None
    SamplingParams = None
    vllm_import_error = str()
    try:
        # capture exception text for later diagnostics
        raise
    except Exception as _e:
        vllm_import_error = repr(_e)

try:
    # transformers will be used as a fallback when vLLM isn't suitable
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    transformers_available = True
except Exception:
    transformers_available = False
    transformers_import_error = repr(Exception())

try:
    # Optional: probe model size via HF Hub and support snapshot download
    from huggingface_hub import model_info, snapshot_download
    hf_hub_available = True
except Exception:
    hf_hub_available = False
    model_info = None
    snapshot_download = None

try:
    # BitsAndBytesConfig for 4-bit quantized loading (transformers >= 4.33+)
    from transformers import BitsAndBytesConfig
    bitsandbytes_available = True
except Exception:
    BitsAndBytesConfig = None
    bitsandbytes_available = False

app = FastAPI()

# Global state
current_model: Optional[object] = None
current_model_name: Optional[str] = None

# Model cache: OrderedDict for LRU (model_name -> model_instance)
model_cache: "OrderedDict[str, Any]" = OrderedDict()
# Track recent failed loads: model_name -> last_failed_timestamp
failed_loads: Dict[str, float] = {}
# Keep track of cached model metadata: backend used, etc.
model_meta: Dict[str, Dict[str, Any]] = {}

# Configuration: max number of models to keep cached, cooldown seconds after failure
MAX_CACHE_MODELS = int(os.getenv("MAX_CACHE_MODELS", "2"))
COOLDOWN_SECONDS = int(os.getenv("MODEL_LOAD_COOLDOWN", "300"))
MAX_CONCURRENT_PULLS = int(os.getenv("MAX_CONCURRENT_PULLS", "2"))

# Job registry for background pulls (job_id -> job metadata)
jobs: Dict[str, Dict[str, Any]] = {}
# Semaphore to limit concurrent background downloads
download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PULLS)

log = logging.getLogger("ai-server")
logging.basicConfig(level=logging.INFO)


class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7


def _evict_lru_if_needed():
    """Evict least-recently-used model if cache exceeds MAX_CACHE_MODELS."""
    while len(model_cache) > MAX_CACHE_MODELS:
        name, mdl = model_cache.popitem(last=False)
        try:
            del mdl
            if torch is not None and hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        except Exception:
            log.exception("Error while evicting model %s", name)
        log.info("Evicted LRU model: %s", name)


def _record_failed_load(model_name: str):
    failed_loads[model_name] = time.time()


def _in_cooldown(model_name: str) -> bool:
    t = failed_loads.get(model_name)
    if not t:
        return False
    return (time.time() - t) < COOLDOWN_SECONDS


def load_model(model_name: str):
    """Load or retrieve a model from cache, and set as current model.

    Implements an LRU cache of model instances and a cooldown/backoff for failed
    model load attempts.
    """
    global current_model, current_model_name

    # If model is already current, nothing to do.
    if current_model_name == model_name and current_model is not None:
        return

    # Check cooldown for previously failed loads
    if _in_cooldown(model_name):
        raise RuntimeError(f"Recent failed attempt for model '{model_name}', in cooldown")

    # If model is cached, reuse it and mark as most-recently-used
    if model_name in model_cache:
        try:
            current_model = model_cache.pop(model_name)
            # re-insert as most-recently-used
            model_cache[model_name] = current_model
            current_model_name = model_name
            log.info("Reused cached model: %s", model_name)
            return
        except Exception:
            log.exception("Error reusing cached model %s", model_name)

    # Prefer vLLM backend if available and suitable
    backend_used = None

    # Probe model size from HF Hub when possible (used to decide quantization)
    size_bytes = None
    try:
        if hf_hub_available and model_info is not None:
            info = model_info(model_name)
            # Sum sizes of siblings (files) as approximation
            siblings = getattr(info, "siblings", []) or []
            size_bytes = sum([getattr(s, "size", 0) or 0 for s in siblings])
            model_meta[model_name] = {**model_meta.get(model_name, {}), "size_bytes": size_bytes}
    except Exception:
        log.exception("Failed to probe model size for %s", model_name)

    # Ensure model is present locally. We do not auto-download on load; models must be pulled via /pull.
    local_path = model_meta.get(model_name, {}).get("local_path")
    if local_path is None:
        if hf_hub_available and snapshot_download is not None:
            try:
                # Try to resolve a local snapshot without downloading (local_files_only=True)
                local_path = snapshot_download(repo_id=model_name, local_files_only=True)
                model_meta[model_name] = {**model_meta.get(model_name, {}), "local_path": local_path}
            except Exception:
                # Not present locally
                local_path = None

    if local_path is None:
        raise RuntimeError(f"Model '{model_name}' not available locally. Use the /pull endpoint to download it before loading.")

    # Helper to add to cache and meta
    def _cache_model(name: str, backend: str, instance: Any, extra_meta: Dict[str, Any] = {}):
        model_cache[name] = instance
        model_meta[name] = {"backend": backend, **extra_meta}
        _evict_lru_if_needed()
        # set current
        global current_model, current_model_name
        current_model = instance
        current_model_name = name
        log.info("✅ Model loaded and cached: %s (backend=%s)", name, backend)

    # Attempt vLLM first
    if LLM is not None:
        try:
            log.info("Loading model with vLLM: %s", model_name)
            tp_size = 1
            if torch is not None:
                try:
                    tp_size = max(1, torch.cuda.device_count())
                except Exception:
                    tp_size = 1
            dtype = "bfloat16" if torch is not None and torch.cuda.is_available() else "float32"
            start_ns = time.time_ns()
            # Prefer loading from local snapshot path so we don't trigger remote downloads
            mdl = LLM(
                model=local_path or model_name,
                tensor_parallel_size=tp_size,
                dtype=dtype,
                trust_remote_code=True,
                max_model_len=32768,
            )
            load_duration_ns = time.time_ns() - start_ns
            backend_used = "vllm"
            _cache_model(model_name, backend_used, mdl, extra_meta={"load_duration_ns": load_duration_ns})
            return
        except Exception:
            log.exception("vLLM failed to load model %s, will try transformers fallback if available", model_name)
            _record_failed_load(model_name)

    # Transformers fallback
    if not transformers_available:
        # No viable backend available
        details = {
            "vllm_available": LLM is not None,
            "vllm_error": vllm_import_error if 'vllm_import_error' in globals() else None,
            "transformers_available": transformers_available,
        }
        raise RuntimeError(f"No inference backend available for model {model_name}: {details}")

    # Use transformers pipeline (simple fallback). This will download model to HF cache.
    # Transformers fallback: prefer quantized 4-bit loads for large models when possible
    try:
        log.info("Loading model with transformers pipeline: %s", model_name)
        # choose device: 0 for first CUDA GPU, -1 for CPU
        device = 0 if (torch is not None and torch.cuda.is_available()) else -1

        start_ns = time.time_ns()
        quantized = False
        # If model is large and bitsandbytes is available and GPU present, try 4-bit load
        try:
            q4_threshold = int(os.getenv("MODEL_Q4_THRESHOLD_BYTES", str(30 * 10**9)))
        except Exception:
            q4_threshold = 30 * 10**9

        # Decide whether to quantize based on user preference stored at pull time, or size threshold
        preferred_quant = model_meta.get(model_name, {}).get("preferred_quantized")
        should_q4 = False
        if preferred_quant is not None:
            should_q4 = bool(preferred_quant)
        else:
            should_q4 = (bitsandbytes_available and device >= 0 and size_bytes is not None and size_bytes >= q4_threshold)

        if should_q4:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=(torch.float16 if torch is not None and torch.cuda.is_available() else "float16"),
                )
                log.info("Attempting 4-bit (q4) load for %s using bitsandbytes", model_name)
                # Load from local_path to avoid network. If local_path is a snapshot dir, pass it directly.
                model = AutoModelForCausalLM.from_pretrained(
                    local_path or model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(local_path or model_name, use_fast=False, trust_remote_code=True, local_files_only=True)
                pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
                quantized = True
            except Exception:
                log.exception("Quantized 4-bit load failed for %s, falling back to standard pipeline", model_name)

        # If not quantized, use standard pipeline which will place model on CPU or specified device
        if not quantized:
            # Use local files only to avoid remote downloads during load
            pipe = pipeline("text-generation", model=local_path or model_name, device=device, trust_remote_code=True)

        load_duration_ns = time.time_ns() - start_ns
        backend_used = "transformers_pipeline"
        extra = {"load_duration_ns": load_duration_ns, "quantized": quantized}
        if size_bytes is not None:
            extra["size_bytes"] = size_bytes
        _cache_model(model_name, backend_used, pipe, extra_meta=extra)
        return
    except Exception:
        log.exception("Transformers fallback failed to load model %s", model_name)
        _record_failed_load(model_name)
        raise


def _extract_text_from_pipeline_result(val: Any) -> str:
    """Normalize various pipeline return shapes into a single assistant text string.

    Handles common shapes:
    - [{'generated_text': '...'}]
    - [{'generated_text': [{'role':'user',...}, {'role':'assistant','content':'...'}]}]
    - {'generated_text': '...'}
    - nested dict/list structures returned by some chat-capable pipelines
    """
    try:
        # Strings are already fine
        if isinstance(val, str):
            return val

        # Lists: prefer the last assistant/content-like entry
        if isinstance(val, list):
            # Try to find assistant content from list of dicts
            for item in reversed(val):
                if isinstance(item, dict):
                    # common keys
                    for key in ("content", "generated_text", "text"):
                        if key in item:
                            res = _extract_text_from_pipeline_result(item[key])
                            if res:
                                return res
            # If list of strings or no obvious dicts, join stringified entries
            parts = [_extract_text_from_pipeline_result(x) if not isinstance(x, str) else x for x in val]
            return "\n".join([p for p in parts if p])

        # Dicts: look for common keys then scan values
        if isinstance(val, dict):
            for key in ("generated_text", "text", "content"):
                if key in val:
                    return _extract_text_from_pipeline_result(val[key])
            # Otherwise search values for content-like strings
            for v in val.values():
                res = _extract_text_from_pipeline_result(v)
                if res:
                    return res

        # Fallback: stringify
        return str(val)
    except Exception:
        try:
            return str(val)
        except Exception:
            return ""


@app.post("/chat")
async def chat(request: ChatRequest):
    model_name = request.model

    # Load the model if different
    if current_model_name != model_name:
        try:
            load_model(model_name)
        except Exception as e:
            log.exception("Failed to load model")
            raise HTTPException(status_code=500, detail=str(e))

    # Build prompt (simple role-based concatenation)
    prompt = ""
    for msg in request.messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    prompt += "Assistant: "

    if current_model is None:
        raise HTTPException(status_code=500, detail="No model loaded")

    backend = model_meta.get(current_model_name, {}).get("backend")

    # vLLM path
    if backend == "vllm":
        if SamplingParams is None:
            raise HTTPException(status_code=500, detail="vLLM SamplingParams not available")
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        gen_start = time.time_ns()
        outputs = current_model.generate([prompt], sampling_params)  # type: ignore
        gen_duration = time.time_ns() - gen_start
        try:
            text = outputs[0].outputs[0].text.strip()
        except Exception:
            text = ""

        created_at = datetime.utcnow().isoformat() + "Z"
        load_duration = model_meta.get(current_model_name, {}).get("load_duration_ns", 0)
        resp = {
            "model": model_name,
            "created_at": created_at,
            "message": {"role": "assistant", "content": text},
            "done": True,
            "done_reason": "stop",
            "total_duration": gen_duration + load_duration,
            "load_duration": load_duration,
            "prompt_eval_count": None,
            "prompt_eval_duration": None,
            "eval_count": None,
            "eval_duration": gen_duration,
        }
        return resp

    # transformers pipeline path
    if backend and backend.startswith("transformers"):
        if not transformers_available:
            raise HTTPException(status_code=500, detail="Transformers not available on this host")
        pipe = current_model
        # pipeline params: max_new_tokens, temperature, do_sample
        try:
            gen_start = time.time_ns()
            # First, try chat-style invocation: pass the messages list directly to the pipeline
            # Some remote (chat-capable) pipelines support being called with a messages list
            try:
                print(request.messages)
                result = pipe(request.messages, max_new_tokens=request.max_tokens, temperature=request.temperature, do_sample=(request.temperature > 0.0))
                print(result)
            except Exception:
                # Fallback to the legacy prompt string for pipelines that expect a single string
                result = pipe(prompt, max_new_tokens=request.max_tokens, temperature=request.temperature, do_sample=(request.temperature > 0.0))

            gen_duration = time.time_ns() - gen_start

            # Normalize possible return shapes from different pipelines into a plain text string
            try:
                text = _extract_text_from_pipeline_result(result).strip()
            except Exception:
                # Fallback to a safe string representation
                try:
                    text = str(result)
                except Exception:
                    text = ""
        except Exception:
            log.exception("Transformers pipeline generation failed for model %s", model_name)
            raise HTTPException(status_code=500, detail="Generation error on transformers backend")

        # Try to compute token counts using the pipeline tokenizer if available
        prompt_tokens = None
        gen_tokens = None
        try:
            tokenizer = getattr(pipe, "tokenizer", None)
            if tokenizer is not None:
                # Count prompt tokens
                enc = tokenizer(prompt)
                prompt_tokens = len(enc.get("input_ids", []))
                # Count generated tokens
                enc_out = tokenizer(text)
                gen_tokens = len(enc_out.get("input_ids", []))
        except Exception:
            prompt_tokens = None
            gen_tokens = None

        created_at = datetime.utcnow().isoformat() + "Z"
        load_duration = model_meta.get(current_model_name, {}).get("load_duration_ns", 0)
        resp = {
            "model": model_name,
            "created_at": created_at,
            "message": {"role": "assistant", "content": text},
            "done": True,
            "done_reason": "stop",
            "total_duration": gen_duration + load_duration,
            "load_duration": load_duration,
            "prompt_eval_count": prompt_tokens,
            "prompt_eval_duration": None,
            "eval_count": gen_tokens,
            "eval_duration": gen_duration,
        }
        return resp

    raise HTTPException(status_code=500, detail=f"Unknown backend for model {model_name}: {backend}")


@app.get("/models")
async def list_models():
    # Return models in an Ollama-compatible list structure. We include cached models plus example known models.
    known = ["Qwen/Qwen2-72B-Chat", "meta-llama/Meta-Llama-3-70B-Instruct"]
    models = []

    # include models tracked in model_meta (these may have been pulled or initialized)
    for name, meta in model_meta.items():
        models.append({
            "model": name,
            "description": meta.get("description"),
            "loaded": True if meta.get("backend") else False,
            "backend": meta.get("backend"),
            "size_bytes": meta.get("size_bytes"),
            "local_path": meta.get("local_path"),
            "load_duration": meta.get("load_duration_ns"),
        })

    # Discover locally present HF snapshots in the HF cache and include them if not already present
    try:
        hf_home = os.getenv("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        if os.path.isdir(hf_home):
            for entry in os.listdir(hf_home):
                # Hugging Face snapshot dirs often start with 'models--' and encode repo id: models--owner--repo
                if not entry.startswith("models--"):
                    continue
                repo_part = entry[len("models--"):]
                # Convert models--owner--repo to owner/repo
                repo_id = repo_part.replace("--", "/")
                if repo_id in model_meta:
                    continue
                full_path = os.path.join(hf_home, entry)
                # compute approximate size (sum of files)
                size_bytes = None
                try:
                    total = 0
                    for root, _, files in os.walk(full_path):
                        for f in files:
                            try:
                                total += os.path.getsize(os.path.join(root, f))
                            except Exception:
                                pass
                    size_bytes = total
                except Exception:
                    size_bytes = None

                models.append({
                    "model": repo_id,
                    "description": None,
                    "loaded": False,
                    "backend": None,
                    "size_bytes": size_bytes,
                    "local_path": full_path,
                    "load_duration": None,
                })
    except Exception:
        log.exception("Error while scanning HF cache for models")

    # include known models that are not present
    for k in known:
        if not any(m.get("model") == k for m in models):
            models.append({"model": k, "description": None, "loaded": False, "backend": None, "size_bytes": None, "load_duration": None})

    return {"models": models}


@app.post("/pull")
async def pull_model(payload: Dict[str, str]):
    """Request body: {"model":"<model-id>"}

    This endpoint forces download/initialization of the model and caches it.
    """
    model_name = payload.get("model")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing 'model' in payload")

    # Optional: user can request quantization preference: 'auto'|'q4'|'fp32' etc.
    quant = payload.get("quantize", "auto")
    init = bool(payload.get("init", False))

    if not hf_hub_available or snapshot_download is None:
        raise HTTPException(status_code=500, detail="HF Hub not available on this host; cannot pull models")

    # Create a background job to perform the pull so API remains responsive.
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "model": model_name,
        "quantize": quant,
        "init": init,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "started_at": None,
        "finished_at": None,
        "error": None,
        "local_path": None,
        "size_bytes": None,
        "preferred_quantized": None,
    }

    # Schedule the background pull task
    asyncio.create_task(_background_pull(job_id, model_name, quant, init))

    # Return accepted with job id so callers can poll /jobs/{id}
    return {"status": "accepted", "job_id": job_id}


@app.get("/status")
async def status():
    """Return server status including cache, failed loads, and backend availability."""
    return {
        "vllm_available": LLM is not None,
        "vllm_import_error": vllm_import_error if 'vllm_import_error' in globals() else None,
        "transformers_available": transformers_available,
        "cache_size": len(model_cache),
        "cached_models": list(model_meta.items()),
        "failed_loads": failed_loads,
        "max_cache_models": MAX_CACHE_MODELS,
        "cooldown_seconds": COOLDOWN_SECONDS,
    }


async def _background_pull(job_id: str, model_name: str, quant: str, init: bool):
    """Background task to perform HF snapshot download and optional initialization.

    This runs in the event loop but performs blocking HF calls inside threads
    via `asyncio.to_thread` and is concurrency-limited by `download_semaphore`.
    """
    job = jobs.get(job_id)
    if job is None:
        return
    job["status"] = "running"
    job["started_at"] = datetime.utcnow().isoformat() + "Z"

    try:
        log.info("Background pull start: %s (job=%s, quant=%s)", model_name, job_id, quant)

        # Acquire semaphore to limit concurrent downloads
        await download_semaphore.acquire()
        try:
            # snapshot_download can be blocking; run in thread
            local_path = await asyncio.to_thread(snapshot_download, repo_id=model_name)
        finally:
            download_semaphore.release()

        job["local_path"] = local_path

        # Probe size if possible
        size_bytes = None
        try:
            if model_info is not None:
                info = await asyncio.to_thread(model_info, model_name)
                siblings = getattr(info, "siblings", []) or []
                size_bytes = sum([getattr(s, "size", 0) or 0 for s in siblings])
        except Exception:
            log.exception("Failed to probe model size after download for %s", model_name)

        # Decide preferred quantization
        preferred_quant = None
        if quant == "q4":
            preferred_quant = True
        elif quant in ("fp32", "fp16", "no"):
            preferred_quant = False
        else:
            try:
                q4_threshold = int(os.getenv("MODEL_Q4_THRESHOLD_BYTES", str(14 * 10**9)))
            except Exception:
                q4_threshold = 14 * 10**9
            preferred_quant = (size_bytes is not None and size_bytes >= q4_threshold)

        # Record metadata so later load_model will use local_path and preferred quantization
        model_meta[model_name] = {**model_meta.get(model_name, {}), "local_path": local_path, "preferred_quantized": preferred_quant, "size_bytes": size_bytes}

        # Optionally initialize (load into cache) after pulling
        if init:
            try:
                await asyncio.to_thread(load_model, model_name)
            except Exception:
                # Record initialization error but keep model metadata
                log.exception("Pull succeeded but initialization failed for %s", model_name)
                # Note: do not mark job as failed because download succeeded
                job.setdefault("notes", []).append("init_failed")

        job["status"] = "succeeded"
        job["size_bytes"] = size_bytes
        job["preferred_quantized"] = preferred_quant
        job["finished_at"] = datetime.utcnow().isoformat() + "Z"
        log.info("Background pull succeeded: %s (job=%s)", model_name, job_id)

    except Exception as e:
        tb = traceback.format_exc()
        job["status"] = "failed"
        job["error"] = str(e)
        job["traceback"] = tb
        job["finished_at"] = datetime.utcnow().isoformat() + "Z"
        log.exception("Background pull failed: %s (job=%s): %s", model_name, job_id, e)


@app.get("/jobs")
async def list_jobs():
    # Return shallow summary of jobs
    return {"jobs": [{"id": j["id"], "model": j["model"], "status": j["status"], "created_at": j.get("created_at"), "started_at": j.get("started_at"), "finished_at": j.get("finished_at")} for j in jobs.values()]}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# Run: uvicorn app:app --host 0.0.0.0 --port 8005
