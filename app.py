import gc
import os
import logging
import time
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

app = FastAPI()

# Global state
current_model: Optional[object] = None
current_model_name: Optional[str] = None

# Model cache: OrderedDict for LRU (model_name -> model_instance)
model_cache: "OrderedDict[str, Any]" = OrderedDict()
# Track recent failed loads: model_name -> last_failed_timestamp
failed_loads: Dict[str, float] = {}

# Configuration: max number of models to keep cached, cooldown seconds after failure
MAX_CACHE_MODELS = int(os.getenv("MAX_CACHE_MODELS", "2"))
COOLDOWN_SECONDS = int(os.getenv("MODEL_LOAD_COOLDOWN", "300"))

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

    if LLM is None:
        raise RuntimeError("vLLM is not installed or failed to import")

    log.info(f"Loading model: {model_name}")

    tp_size = 1
    if torch is not None:
        try:
            tp_size = max(1, torch.cuda.device_count())
        except Exception:
            tp_size = 1

    dtype = "bfloat16" if torch is not None and torch.cuda.is_available() else "float32"

    try:
        mdl = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            dtype=dtype,
            trust_remote_code=True,
            max_model_len=32768,
        )
    except Exception as e:
        log.exception("Failed to instantiate model %s", model_name)
        _record_failed_load(model_name)
        raise

    # Add to cache as most-recently-used
    model_cache[model_name] = mdl
    # If cache too large, evict LRU
    _evict_lru_if_needed()

    # Set current model reference
    current_model = mdl
    current_model_name = model_name
    log.info(f"✅ Model loaded and cached: {model_name}")


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

    if SamplingParams is None or current_model is None:
        raise HTTPException(status_code=500, detail="vLLM not available on this host")

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    outputs = current_model.generate([prompt], sampling_params)

    try:
        text = outputs[0].outputs[0].text.strip()
    except Exception:
        text = ""

    return {"response": text, "model": model_name}


@app.get("/models")
async def list_models():
    # Example hardcoded list. Replace with dynamic discovery if desired.
    return {"models": ["Qwen/Qwen2-72B-Chat", "meta-llama/Meta-Llama-3-70B-Instruct"]}


# Run: uvicorn app:app --host 0.0.0.0 --port 8000
