import os
import time
from typing import List, Dict

import config
import state
import model_cache as mcache


def build_llama_cpp_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
        else:  # user, tool, or any unknown role → treat as user
            parts.append(f"<|user|>\n{content}")
    parts.append("<|assistant|>")
    return "\n".join(parts)


def load_gguf(model_name: str, gguf_path: str) -> None:
    if not config.llama_cpp_available:
        raise RuntimeError("llama-cpp-python is not installed. Install: pip install llama-cpp-python")
    if not os.path.isfile(gguf_path):
        raise RuntimeError(f"GGUF file not found: {gguf_path}")

    n_gpu_layers = int(os.getenv("GGUF_N_GPU_LAYERS", "-1"))  # -1 offloads all layers to GPU
    n_ctx = int(os.getenv("GGUF_N_CTX", "4096"))
    config.log.info("Loading GGUF with llama-cpp-python: %s (n_gpu_layers=%s, n_ctx=%s)",
                    gguf_path, n_gpu_layers, n_ctx)

    start_ns = time.time_ns()
    try:
        mdl = config.LlamaCpp(model_path=gguf_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)
    except Exception as e:
        config.log.exception("llama-cpp-python failed for %s", gguf_path)
        msg = str(e)
        # Detect the most common cause: GGUF uses an architecture llama.cpp doesn't know yet
        if "unknown model architecture" in msg:
            arch = msg.split("'")[-2] if "'" in msg else "?"
            raise RuntimeError(
                f"GGUF uses architecture '{arch}' which your installed llama-cpp-python "
                f"does not recognize. Upgrade: pip install -U llama-cpp-python. "
                f"If the model is brand new, you may need a llama.cpp build that supports it "
                f"from source. The Transformers backend cannot load GGUF files directly — "
                f"you need the matching safetensors repo on Hugging Face."
            ) from e
        raise RuntimeError(
            f"llama-cpp-python failed to load GGUF: {type(e).__name__}: {msg}. "
            f"Common causes: (1) llama-cpp-python was built without CUDA/GPU support — "
            f"set GGUF_N_GPU_LAYERS=0 to run on CPU; (2) GGUF file is corrupt or incomplete; "
            f"(3) insufficient RAM for n_ctx={n_ctx} — try GGUF_N_CTX=512."
        ) from e
    mcache.cache_model(model_name, "gguf_llama_cpp", mdl, extra_meta={
        "load_duration_ns": time.time_ns() - start_ns,
        "local_path": gguf_path,
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
    })
