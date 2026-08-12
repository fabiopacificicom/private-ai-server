import gc
import os
import logging
import time
import asyncio
from typing import Optional

# Must be set before torch is imported
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def _load_local_env(path: str = ".env") -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and os.environ.get(key) is None:
                    os.environ[key] = val
    except FileNotFoundError:
        pass
    except Exception:
        pass


_load_local_env()

_default_hf_home = os.getenv("HF_HOME") or os.path.join(os.path.expanduser("~"), "ai-server-models")
os.environ.setdefault("HF_HOME", _default_hf_home)
# Force-disable symlinks: on Windows creating symlinks requires Developer Mode or
# Administrator, which raises WinError 1314. Hard-set (not setdefault) so it always
# applies regardless of what huggingface_hub may have cached.
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

try:
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    _perm = os.path.join(os.environ["HF_HOME"], ".perm_check")
    with open(_perm, "w") as _f:
        _f.write("ok")
    os.remove(_perm)
except Exception as _e:
    print(f"Warning: HF_HOME '{os.environ.get('HF_HOME')}' may not be writable: {_e}")

# ---------------------------------------------------------------------------
# Backend imports — all optional, availability tracked via flags
# ---------------------------------------------------------------------------

try:
    import torch
except Exception:
    torch = None

vllm_import_error: str = ""
try:
    from vllm import LLM, SamplingParams
except Exception as _e:
    LLM = None
    SamplingParams = None
    vllm_import_error = repr(_e)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig
    transformers_available = True
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoConfig = None
    transformers_available = False

try:
    from huggingface_hub import model_info, snapshot_download
    hf_hub_available = True
except Exception:
    model_info = None
    snapshot_download = None
    hf_hub_available = False

try:
    from transformers import BitsAndBytesConfig
    bitsandbytes_available = True
except Exception:
    BitsAndBytesConfig = None
    bitsandbytes_available = False

try:
    from llama_cpp import Llama as LlamaCpp
    llama_cpp_available = True
except Exception:
    LlamaCpp = None
    llama_cpp_available = False

# ---------------------------------------------------------------------------
# Constants — all overridable via environment variables
# ---------------------------------------------------------------------------

MAX_CACHE_MODELS: int = int(os.getenv("MAX_CACHE_MODELS", "2"))
COOLDOWN_SECONDS: int = int(os.getenv("MODEL_LOAD_COOLDOWN", "300"))
MAX_CONCURRENT_PULLS: int = int(os.getenv("MAX_CONCURRENT_PULLS", "2"))
STREAMING_THREAD_TIMEOUT: int = int(os.getenv("STREAMING_THREAD_TIMEOUT", "30"))
Q4_THRESHOLD_BYTES: int = int(os.getenv("MODEL_Q4_THRESHOLD_BYTES", str(20 * 10**9)))
# Layer-splitting: cap GPU memory to leave a safety buffer, allow CPU RAM offload
MAX_GPU_MEMORY: str = os.getenv("MAX_GPU_MEMORY", "12GiB")
MAX_CPU_MEMORY: str = os.getenv("MAX_CPU_MEMORY", "48GiB")
# Semicolon-separated extra dirs to scan for GGUF/HF models outside HF_HOME
MODELS_EXTRA_SCAN_DIRS: list = [
    d.strip() for d in os.getenv("MODELS_EXTRA_SCAN_DIRS", "").split(";")
    if d.strip() and os.path.isdir(d.strip())
]
# Root of an existing Ollama install's model store (manifests + sha256 blobs).
# When set, the server discovers Ollama models and can serve them as GGUF.
OLLAMA_MODELS_DIR: Optional[str] = os.getenv("OLLAMA_MODELS_DIR") or None

server_start_time: float = time.time()
download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PULLS)

log = logging.getLogger("ai-server")
# Configure root logger so uvicorn's access logs and our app logs share one format.
# Force=True re-applies handlers on reload; we keep it idempotent by clearing first.
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for h in list(_root.handlers):
    _root.removeHandler(h)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_root.addHandler(_handler)
# Uvicorn already has its own access log; don't double-log it.
logging.getLogger("uvicorn.access").propagate = True
