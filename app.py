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

# Set PyTorch CUDA allocator config to reduce fragmentation and OOM errors
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

try:
    import torch
except Exception:
    torch = None

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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

# Track server start time for uptime calculation
server_start_time = time.time()

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
    stream: bool = False  # Enable streaming responses


def _evict_lru_if_needed():
    """Evict least-recently-used model if cache exceeds MAX_CACHE_MODELS."""
    while len(model_cache) > MAX_CACHE_MODELS:
        name, mdl = model_cache.popitem(last=False)
        try:
            del mdl
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass
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

    # Aggressive GPU memory cleanup before loading to reduce OOM errors
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        except Exception:
            log.debug("GPU memory cleanup failed (non-fatal)")

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

    # Log GPU / torch diagnostics to help determine device placement
    try:
        if torch is not None:
            try:
                torch_info = {
                    "torch_version": getattr(torch, "__version__", None),
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "cuda_device_name": None,
                }
                if torch.cuda.is_available():
                    try:
                        torch_info["cuda_device_name"] = torch.cuda.get_device_name(0)
                    except Exception:
                        torch_info["cuda_device_name"] = None
                log.info("Torch diagnostics: %s", torch_info)
            except Exception:
                log.exception("Error while collecting torch diagnostics")
    except Exception:
        pass

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
        log.info("Transformers selected device=%s (0=gpu, -1=cpu). torch.cuda.is_available=%s", device, (torch is not None and torch.cuda.is_available()))

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
            # Prefer to load the model onto GPU(s) when available by using device_map='auto'
            if device >= 0 and torch is not None and torch.cuda.is_available():
                try:
                    log.info("Attempting GPU-backed load via from_pretrained with device_map='auto' for %s", model_name)
                    torch_dtype = torch.float16
                    model = AutoModelForCausalLM.from_pretrained(
                        local_path or model_name,
                        device_map="auto",
                        torch_dtype=torch_dtype,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(local_path or model_name, use_fast=False, trust_remote_code=True, local_files_only=True)
                    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
                except ValueError as ve:
                    # Some model repos (Mixture-of-Experts / custom Qwen variants) expose
                    # a configuration class that does not map to transformers' AutoModelForCausalLM.
                    # In that case, fall back to using the pipeline with trust_remote_code=True
                    # which will import model-specific code from the repo. Log a clear warning
                    # because `trust_remote_code=True` runs repository code (security implication).
                    msg = str(ve)
                    if "Unrecognized configuration class" in msg or "MoeConfig" in msg or "Qwen3OmniMoeConfig" in msg:
                        log.warning(
                            "Model configuration %s not supported by AutoModelForCausalLM; falling back to pipeline with trust_remote_code=True. "
                            "This will execute model repo code; ensure you trust the model source.",
                            model_name,
                        )
                        # mark in metadata that this model required pipeline-only loading
                        model_meta[model_name] = {**model_meta.get(model_name, {}), "pipeline_only": True}
                        try:
                            # For MoE/custom models, try to apply quantization if should_q4 is True
                            model_kwargs = {"trust_remote_code": True, "local_files_only": True}
                            if should_q4:
                                log.info("Applying 4-bit quantization to MoE/custom model %s", model_name)
                                bnb_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16,
                                )
                                model_kwargs["quantization_config"] = bnb_config
                                model_kwargs["device_map"] = "auto"
                                quantized = True
                            
                            pipe = pipeline("text-generation", model=local_path or model_name, model_kwargs=model_kwargs, device=device if not should_q4 else None)
                        except Exception:
                            log.exception("Pipeline fallback (trust_remote_code) failed for %s", model_name)
                            raise
                    else:
                        # re-raise other ValueErrors so they're handled by outer except
                        raise
                except Exception:
                    log.exception("GPU-backed from_pretrained load failed for %s, falling back to pipeline with device param", model_name)
                    # Fallback: let pipeline manage device placement (device param)
                    pipe = pipeline("text-generation", model=local_path or model_name, device=device, trust_remote_code=True, local_files_only=True)
            else:
                pipe = pipeline("text-generation", model=local_path or model_name, device=device, trust_remote_code=True, local_files_only=True)

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


async def _stream_chat_response(model_name: str, messages: List[Dict[str, str]], max_tokens: int = 256, temperature: float = 0.7):
    """Generate streaming chat response chunks in SSE format.
    
    Supports both vLLM and transformers backends with TextIteratorStreamer.
    Yields Server-Sent Events (SSE) formatted chunks.
    """
    import json
    from typing import AsyncIterator
    
    # Load model if not already loaded
    if current_model_name != model_name:
        try:
            load_model(model_name)
        except Exception as e:
            log.exception("Failed to load model for streaming")
            error_chunk = {"error": str(e), "done": True}
            yield f"data: {json.dumps(error_chunk)}\n\n"
            return
    
    if current_model is None:
        error_chunk = {"error": "No model loaded", "done": True}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        return
    
    backend = model_meta.get(current_model_name, {}).get("backend")
    
    # Build prompt for backends that need it
    prompt = ""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    prompt += "Assistant: "
    
    # vLLM streaming path
    if backend == "vllm":
        try:
            if SamplingParams is None:
                error_chunk = {"error": "vLLM SamplingParams not available", "done": True}
                yield f"data: {json.dumps(error_chunk)}\n\n"
                return
            
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # vLLM supports streaming through its generate method
            for output in current_model.generate([prompt], sampling_params, use_tqdm=False):
                try:
                    text_delta = output.outputs[0].text
                    chunk = {"delta": {"content": text_delta}, "done": False}
                    yield f"data: {json.dumps(chunk)}\n\n"
                except Exception as e:
                    log.exception("Error in vLLM streaming")
                    error_chunk = {"error": str(e), "done": True}
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    return
            
            # Final chunk
            final_chunk = {"delta": {}, "done": True}
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
        except Exception as e:
            log.exception("vLLM streaming failed")
            error_chunk = {"error": str(e), "done": True}
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    # Transformers streaming path
    elif backend and backend.startswith("transformers"):
        try:
            # Import TextIteratorStreamer for transformers streaming
            try:
                from transformers import TextIteratorStreamer
            except ImportError:
                error_chunk = {"error": "TextIteratorStreamer not available. Install transformers with streaming support.", "done": True}
                yield f"data: {json.dumps(error_chunk)}\n\n"
                return
            
            from threading import Thread
            
            pipe = current_model
            tokenizer = getattr(pipe, "tokenizer", None)
            model_obj = getattr(pipe, "model", None)
            
            if tokenizer is None or model_obj is None:
                error_chunk = {"error": "Pipeline does not expose tokenizer/model for streaming", "done": True}
                yield f"data: {json.dumps(error_chunk)}\n\n"
                return
            
            # Create streamer
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # Prepare generation function to run in thread
            def generate():
                try:
                    # Try messages-based generation first
                    try:
                        pipe(messages, max_new_tokens=max_tokens, temperature=temperature, do_sample=(temperature > 0.0), streamer=streamer)
                    except Exception:
                        # Fallback to prompt-based generation
                        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
                        input_ids = inputs.get("input_ids")
                        
                        # Move to device
                        try:
                            next_param = next(model_obj.parameters())
                            model_device = next_param.device
                            input_ids = input_ids.to(model_device)
                        except Exception:
                            pass
                        
                        gen_kwargs = {"max_new_tokens": max_tokens, "streamer": streamer}
                        if temperature and temperature > 0.0:
                            gen_kwargs.update({"do_sample": True, "temperature": temperature})
                        else:
                            gen_kwargs.update({"do_sample": False})
                        
                        model_obj.generate(input_ids=input_ids, **gen_kwargs)
                except Exception as e:
                    log.exception("Error in generation thread")
            
            # Start generation in background thread
            thread = Thread(target=generate)
            thread.start()
            
            # Stream chunks as they become available
            try:
                for text in streamer:
                    if text:
                        chunk = {"delta": {"content": text}, "done": False}
                        yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                log.exception("Error streaming from TextIteratorStreamer")
                error_chunk = {"error": str(e), "done": True}
                yield f"data: {json.dumps(error_chunk)}\n\n"
                thread.join(timeout=5)
                return
            
            # Wait for thread to complete
            thread.join(timeout=30)
            
            # Final chunk
            final_chunk = {"delta": {}, "done": True}
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
        except Exception as e:
            log.exception("Transformers streaming failed")
            error_chunk = {"error": str(e), "done": True}
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    else:
        error_chunk = {"error": f"Streaming not supported for backend: {backend}", "done": True}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    model_name = request.model
    
    # If streaming is requested, return StreamingResponse
    if request.stream:
        return StreamingResponse(
            _stream_chat_response(
                model_name=model_name,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    # Non-streaming response (existing implementation)
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
            # Some chat-capable pipelines support being called with a messages list
            result = None
            try:
                result = pipe(request.messages, max_new_tokens=request.max_tokens, temperature=request.temperature, do_sample=(request.temperature > 0.0))
            except Exception as e_messages:
                log.debug("pipeline(messages) failed: %s", e_messages)
                # Fallback to the legacy prompt string for pipelines that expect a single string
                try:
                    result = pipe(prompt, max_new_tokens=request.max_tokens, temperature=request.temperature, do_sample=(request.temperature > 0.0))
                except Exception as e_prompt:
                    log.debug("pipeline(prompt) also failed: %s", e_prompt)
                    # As a last resort, attempt direct model.generate using tokenizer + model
                    try:
                        model_obj = getattr(pipe, "model", None)
                        tokenizer = getattr(pipe, "tokenizer", None)
                        if model_obj is None or tokenizer is None:
                            raise RuntimeError("Pipeline does not expose model/tokenizer for direct generate fallback")

                        # Tokenize prompt
                        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
                        input_ids = inputs.get("input_ids")
                        attention_mask = inputs.get("attention_mask")

                        # Move model and tensors to GPU if available
                        model_device = None
                        try:
                            next_param = next(model_obj.parameters())
                            model_device = next_param.device
                        except Exception:
                            model_device = None

                        if torch is not None and torch.cuda.is_available():
                            try:
                                model_obj.to("cuda")
                                model_device = torch.device("cuda")
                            except Exception:
                                log.debug("Could not move model to CUDA; continuing on CPU")

                        if input_ids is not None and model_device is not None:
                            try:
                                input_ids = input_ids.to(model_device)
                                if attention_mask is not None:
                                    attention_mask = attention_mask.to(model_device)
                            except Exception:
                                log.debug("Failed to move input tensors to model device; proceeding without explicit move")

                        gen_kwargs = {"max_new_tokens": request.max_tokens}
                        if request.temperature and request.temperature > 0.0:
                            gen_kwargs.update({"do_sample": True, "temperature": request.temperature})
                        else:
                            gen_kwargs.update({"do_sample": False})

                        # Call model.generate directly
                        outputs_ids = model_obj.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

                        # Extract generated portion (tokens after input length)
                        gen_text = ""
                        try:
                            gen_tokens = outputs_ids[:, input_ids.shape[1]:]
                            gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                            gen_text = gen_texts[0] if isinstance(gen_texts, list) and len(gen_texts) > 0 else ""
                        except Exception:
                            # Fallback: decode full output
                            gen_text = tokenizer.decode(outputs_ids[0], skip_special_tokens=True)

                        # Build a normalized pipeline-style result
                        result = [{"generated_text": gen_text}]
                    except Exception:
                        log.exception("Direct model.generate fallback failed for model %s", model_name)
                        raise

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
async def pull_model(payload: Dict[str, Any]):
    """Request body: {"model":"<model-id>"}

    This endpoint forces download/initialization of the model and caches it.
    """
    model_name = payload.get("model")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing 'model' in payload")

    # Optional: user can request quantization preference: 'auto'|'q4'|'fp32' etc.
    quant = payload.get("quantize", "auto")
    # Accept init as boolean or string (e.g. "true"), coerce to bool
    raw_init = payload.get("init", False)
    if isinstance(raw_init, str):
        init = raw_init.lower() in ("1", "true", "yes", "y")
    else:
        init = bool(raw_init)

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


@app.get("/diag")
async def diag():
    """Return torch / CUDA diagnostics to help with GPU troubleshooting."""
    info = {
        "torch_installed": torch is not None,
        "torch_version": None,
        "cuda_available": False,
        "cuda_count": 0,
        "cuda_devices": [],
    }
    try:
        if torch is not None:
            info["torch_version"] = getattr(torch, "__version__", None)
            try:
                info["cuda_available"] = torch.cuda.is_available()
            except Exception:
                info["cuda_available"] = False
            try:
                info["cuda_count"] = torch.cuda.device_count() if info["cuda_available"] else 0
            except Exception:
                info["cuda_count"] = 0
            try:
                devices = []
                for i in range(info["cuda_count"]):
                    try:
                        devices.append({"index": i, "name": torch.cuda.get_device_name(i)})
                    except Exception:
                        devices.append({"index": i, "name": None})
                info["cuda_devices"] = devices
            except Exception:
                info["cuda_devices"] = []
    except Exception:
        log.exception("Error collecting diag info")

    return info


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


@app.get("/health")
async def health():
    """Health check endpoint for load balancers and monitoring.
    
    Returns overall server health status, uptime, cache statistics,
    and GPU metrics when CUDA is available.
    """
    uptime = time.time() - server_start_time
    
    # GPU diagnostics
    gpu_status = "unavailable"
    gpu_memory_allocated_mb = None
    gpu_memory_reserved_mb = None
    
    if torch is not None and torch.cuda.is_available():
        try:
            gpu_status = "available"
            gpu_memory_allocated_mb = torch.cuda.memory_allocated(0) / (1024**2)
            gpu_memory_reserved_mb = torch.cuda.memory_reserved(0) / (1024**2)
        except Exception:
            gpu_status = "error"
            log.exception("Error collecting GPU memory stats")
    
    # Determine overall health status
    status = "healthy"
    if not transformers_available and LLM is None:
        status = "degraded"  # No inference backends available
    
    # Count active and queued downloads
    downloads_active = sum(1 for j in jobs.values() if j.get("status") == "running")
    downloads_queued = sum(1 for j in jobs.values() if j.get("status") == "queued")
    
    return {
        "status": status,
        "uptime_seconds": int(uptime),
        "models_cached": len(model_cache),
        "cache_limit": MAX_CACHE_MODELS,
        "downloads_active": downloads_active,
        "downloads_queued": downloads_queued,
        "torch_version": getattr(torch, "__version__", None) if torch else None,
        "cuda_available": torch.cuda.is_available() if torch else False,
        "gpu_status": gpu_status,
        "gpu_memory_allocated_mb": gpu_memory_allocated_mb,
        "gpu_memory_reserved_mb": gpu_memory_reserved_mb,
    }


# Run: uvicorn app:app --host 0.0.0.0 --port 8005
