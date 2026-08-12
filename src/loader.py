import json
import os
import re
import time
from typing import Any, Dict, Optional

import config
import state
import model_cache as mcache
import gguf as gguf_backend

MANIFEST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_manifest.json")

# ---------------------------------------------------------------------------
# Manifest — persists discovered model paths across restarts
# ---------------------------------------------------------------------------

def load_manifest() -> None:
    """Populate state.model_meta from the on-disk manifest (best-effort)."""
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data: Dict[str, Dict] = json.load(f)
        for model_name, meta in data.items():
            local_path = meta.get("local_path")
            # only register if path still exists on disk
            if local_path and os.path.exists(local_path):
                existing = state.model_meta.get(model_name, {})
                state.model_meta[model_name] = {**meta, **existing}
        config.log.info("Manifest loaded: %d entries from %s", len(data), MANIFEST_PATH)
    except FileNotFoundError:
        pass
    except Exception:
        config.log.exception("Failed to load models manifest")


def save_manifest() -> None:
    """Persist state.model_meta entries that have a local_path to disk."""
    try:
        # only persist discovery metadata, not runtime state like backend/load_duration
        keep_keys = {"local_path", "size_bytes", "backend", "preferred_quantized"}
        data = {
            name: {k: v for k, v in meta.items() if k in keep_keys}
            for name, meta in state.model_meta.items()
            if meta.get("local_path")
        }
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        config.log.exception("Failed to save models manifest")


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def resolve_model_cache_path(model_name: str) -> str:
    hf_home = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    return os.path.join(hf_home, "hub", f"models--{model_name.replace('/', '--')}")


def calculate_downloaded_bytes(cache_path: str) -> int:
    if not os.path.exists(cache_path):
        return 0
    total = 0
    for root, _, files in os.walk(cache_path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except Exception:
                pass
    return total


# ---------------------------------------------------------------------------
# Pipeline output normalisation
# ---------------------------------------------------------------------------

def extract_text_from_pipeline_result(val: Any) -> str:
    try:
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            for item in reversed(val):
                if isinstance(item, dict):
                    for key in ("content", "generated_text", "text"):
                        if key in item:
                            res = extract_text_from_pipeline_result(item[key])
                            if res:
                                return res
            parts = [extract_text_from_pipeline_result(x) if not isinstance(x, str) else x for x in val]
            return "\n".join(p for p in parts if p)
        if isinstance(val, dict):
            for key in ("generated_text", "text", "content"):
                if key in val:
                    return extract_text_from_pipeline_result(val[key])
            for v in val.values():
                res = extract_text_from_pipeline_result(v)
                if res:
                    return res
        return str(val)
    except Exception:
        try:
            return str(val)
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Private loading helpers
# ---------------------------------------------------------------------------

def _probe_model_size(model_name: str) -> Optional[int]:
    try:
        if config.hf_hub_available and config.model_info is not None:
            info = config.model_info(model_name)
            siblings = getattr(info, "siblings", []) or []
            size = sum(getattr(s, "size", 0) or 0 for s in siblings)
            state.model_meta[model_name] = {**state.model_meta.get(model_name, {}), "size_bytes": size}
            return size
    except Exception:
        config.log.exception("Failed to probe model size for %s", model_name)
    return None


def _resolve_local_path(model_name: str) -> Optional[str]:
    local_path = state.model_meta.get(model_name, {}).get("local_path")
    if local_path:
        return local_path

    # Try HF hub resolution first (works when HF_HOME matches the cache location)
    if config.hf_hub_available and config.snapshot_download is not None:
        try:
            local_path = config.snapshot_download(repo_id=model_name, local_files_only=True)
            state.model_meta[model_name] = {**state.model_meta.get(model_name, {}), "local_path": local_path}
            return local_path
        except Exception:
            pass

    # Fallback: scan HF_HOME directly for the cache directory and resolve its snapshot
    hf_home = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    cache_dir_name = f"models--{model_name.replace('/', '--')}"
    for root, dirs, _ in os.walk(hf_home):
        for entry in dirs:
            if entry == cache_dir_name:
                cache_root = os.path.join(root, entry)
                snapshots_dir = os.path.join(cache_root, "snapshots")
                if os.path.isdir(snapshots_dir):
                    try:
                        snaps = [e for e in os.listdir(snapshots_dir)
                                 if os.path.isdir(os.path.join(snapshots_dir, e))]
                        if snaps:
                            snaps.sort(key=lambda e: os.path.getmtime(os.path.join(snapshots_dir, e)),
                                       reverse=True)
                            local_path = os.path.join(snapshots_dir, snaps[0])
                            state.model_meta[model_name] = {
                                **state.model_meta.get(model_name, {}), "local_path": local_path}
                            config.log.info("Resolved local path via filesystem scan: %s -> %s",
                                            model_name, local_path)
                            return local_path
                    except Exception:
                        pass
        # don't recurse deeper than needed — the cache dirs sit directly under hub/
        dirs[:] = [d for d in dirs if not d.startswith("models--")]

    return None


def _try_vllm(model_name: str, local_path: str) -> bool:
    if config.LLM is None:
        return False
    try:
        tp_size = 1
        if config.torch is not None:
            try:
                tp_size = max(1, config.torch.cuda.device_count())
            except Exception:
                pass
        dtype = "bfloat16" if config.torch is not None and config.torch.cuda.is_available() else "float32"
        config.log.info("Loading with vLLM: %s", model_name)
        start_ns = time.time_ns()
        mdl = config.LLM(
            model=local_path,
            tensor_parallel_size=tp_size,
            dtype=dtype,
            trust_remote_code=True,
            max_model_len=32768,
        )
        mcache.cache_model(model_name, "vllm", mdl, extra_meta={"load_duration_ns": time.time_ns() - start_ns})
        return True
    except Exception:
        config.log.exception("vLLM failed for %s, trying next backend", model_name)
        mcache.record_failed_load(model_name)
        return False


def _build_max_memory() -> dict:
    """Build the max_memory dict for layer-splitting GPU/CPU offload."""
    mm: dict = {}
    if config.torch is not None and config.torch.cuda.is_available():
        try:
            mm[0] = config.MAX_GPU_MEMORY
        except Exception:
            pass
    mm["cpu"] = config.MAX_CPU_MEMORY
    return mm


def _load_transformers(model_name: str, local_path: str, size_bytes: Optional[int]) -> None:
    if not config.transformers_available:
        raise RuntimeError(
            f"No inference backend available for '{model_name}'. "
            f"Install: pip install transformers torch  (vllm={config.LLM is not None})"
        )

    device = 0 if (config.torch is not None and config.torch.cuda.is_available()) else -1
    config.log.info("Loading with transformers (device=%s, max_memory=%s): %s",
                    device, _build_max_memory(), model_name)
    start_ns = time.time_ns()

    preferred_quant = state.model_meta.get(model_name, {}).get("preferred_quantized")
    should_q4 = bool(preferred_quant) if preferred_quant is not None else (
        config.bitsandbytes_available and device >= 0
        and size_bytes is not None and size_bytes >= config.Q4_THRESHOLD_BYTES
    )

    pipe = None
    quantized = False

    if should_q4:
        try:
            bnb_cfg = config.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(config.torch.float16 if config.torch else "float16"),
            )
            config.log.info("4-bit (NF4) load for %s (layer-split GPU/CPU)", model_name)
            model = config.AutoModelForCausalLM.from_pretrained(
                local_path, quantization_config=bnb_cfg, device_map="auto",
                max_memory=_build_max_memory(),
                trust_remote_code=True, local_files_only=True,
            )
            tok = config.AutoTokenizer.from_pretrained(local_path, use_fast=False,
                                                        trust_remote_code=True, local_files_only=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            pipe = config.pipeline("text-generation", model=model, tokenizer=tok,
                                   device=0, token=os.environ.get("HUGGINGFACE_TOKEN"))
            quantized = True
        except Exception:
            config.log.exception("4-bit load failed for %s, falling back to fp16", model_name)

    if not quantized:
        if device >= 0 and config.torch is not None and config.torch.cuda.is_available():
            try:
                config.log.info("GPU fp16 load for %s (layer-split GPU/CPU)", model_name)
                model = config.AutoModelForCausalLM.from_pretrained(
                    local_path, device_map="auto", torch_dtype=config.torch.float16,
                    max_memory=_build_max_memory(),
                    trust_remote_code=True, local_files_only=True,
                )
                tok = config.AutoTokenizer.from_pretrained(local_path, use_fast=False,
                                                            trust_remote_code=True, local_files_only=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                pipe = config.pipeline("text-generation", model=model, tokenizer=tok,
                                       token=os.environ.get("HUGGINGFACE_TOKEN"))
            except ValueError as ve:
                msg = str(ve)
                # Custom architecture (e.g. muse_glimmer, MoE) — transformers doesn't
                # have a built-in AutoModelForCausalLM class for it. Load config and
                # model directly with trust_remote_code=True so the model's own code
                # in the repo is executed.
                if any(k in msg for k in (
                    "Unrecognized configuration class",
                    "MoeConfig",
                    "Qwen3OmniMoeConfig",
                    "Transformers does not recognize this architecture",
                )):
                    config.log.warning(
                        "Custom arch for %s — loading via AutoConfig/AutoModelForCausalLM "
                        "with trust_remote_code=True (executes model repo code; ensure "
                        "you trust the source)", model_name,
                    )
                    state.model_meta[model_name] = {**state.model_meta.get(model_name, {}), "pipeline_only": True}

                    # Load config first with trust_remote_code so the custom config
                    # class from the repo gets registered before model loading.
                    cfg = config.AutoConfig.from_pretrained(
                        local_path, trust_remote_code=True, local_files_only=True,
                    )
                    model_kwargs: dict = {"config": cfg, "trust_remote_code": True, "local_files_only": True}
                    if should_q4:
                        model_kwargs["quantization_config"] = config.BitsAndBytesConfig(
                            load_in_4bit=True, bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=config.torch.float16,
                        )
                        model_kwargs["device_map"] = "auto"
                        model_kwargs["max_memory"] = _build_max_memory()
                        quantized = True
                    else:
                        model_kwargs["torch_dtype"] = config.torch.float16
                        model_kwargs["device_map"] = "auto"
                        model_kwargs["max_memory"] = _build_max_memory()
                    model = config.AutoModelForCausalLM.from_pretrained(local_path, **model_kwargs)
                    tok = config.AutoTokenizer.from_pretrained(
                        local_path, use_fast=False, trust_remote_code=True, local_files_only=True,
                    )
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token
                    pipe = config.pipeline(
                        "text-generation", model=model, tokenizer=tok,
                        device=0 if not should_q4 else None,
                        token=os.environ.get("HUGGINGFACE_TOKEN"),
                    )
                else:
                    raise
            except Exception:
                config.log.exception("GPU load failed for %s, fallback to pipeline device param", model_name)
                pipe = config.pipeline("text-generation", model=local_path, device=device,
                                       trust_remote_code=True, local_files_only=True,
                                       token=os.environ.get("HUGGINGFACE_TOKEN"))
        else:
            pipe = config.pipeline("text-generation", model=local_path, device=device,
                                   trust_remote_code=True, local_files_only=True,
                                   token=os.environ.get("HUGGINGFACE_TOKEN"))

    extra: dict = {"load_duration_ns": time.time_ns() - start_ns, "quantized": quantized}
    if size_bytes is not None:
        extra["size_bytes"] = size_bytes
    mcache.cache_model(model_name, "transformers_pipeline", pipe, extra_meta=extra)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(model_name: str) -> None:
    mcache.gpu_cleanup()

    if state.current_model_name == model_name and state.current_model is not None:
        return

    if mcache.in_cooldown(model_name):
        raise RuntimeError(
            f"Model '{model_name}' is in cooldown ({config.COOLDOWN_SECONDS}s) after a recent failure. "
            f"Wait or restart the server to reset."
        )

    if model_name in state.model_cache:
        try:
            state.current_model = state.model_cache.pop(model_name)
            state.model_cache[model_name] = state.current_model  # re-insert as MRU
            state.current_model_name = model_name
            config.log.info("Reused cached model: %s", model_name)
            return
        except Exception:
            config.log.exception("Error reusing cached model %s", model_name)

    if config.torch is not None:
        try:
            config.log.info(
                "Torch: version=%s cuda=%s devices=%s",
                getattr(config.torch, "__version__", None),
                config.torch.cuda.is_available(),
                config.torch.cuda.device_count() if config.torch.cuda.is_available() else 0,
            )
        except Exception:
            pass

    size_bytes = _probe_model_size(model_name)
    local_path = _resolve_local_path(model_name)

    if local_path is None:
        raise RuntimeError(
            f"Model '{model_name}' not downloaded. POST /pull {{\"model\":\"{model_name}\"}}"
        )

    if _try_vllm(model_name, local_path):
        return

    is_gguf = (
        state.model_meta.get(model_name, {}).get("backend") == "gguf_llama_cpp"
        or local_path.lower().endswith(".gguf")
    )

    # Transformers is the primary backend. For HF repo ids, prefer it.
    # For local .gguf files, try llama-cpp first since Transformers can't read them,
    # but still surface a clear error if it fails (no Transformers fallback for GGUF).
    if is_gguf:
        try:
            gguf_backend.load_gguf(model_name, local_path)
            return
        except Exception:
            mcache.record_failed_load(model_name)
            raise

    try:
        _load_transformers(model_name, local_path, size_bytes)
    except Exception as e:
        msg = str(e)
        config.log.exception("Transformers load failed for %s", model_name)
        mcache.record_failed_load(model_name)
        m = re.search(r'pip install ([^`\n]+)', msg)
        if m and ("requires the following packages" in msg or "No module named" in msg):
            raise RuntimeError(f"Missing dependency for '{model_name}'. Install: pip install {m.group(1).strip()}")
        raise
