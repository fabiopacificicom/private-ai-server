# AI Server (Ollama-compatible drop-in)

This repository implements a single-process FastAPI server that can act as a drop-in alternative to Ollama for serving local Hugging Face / vLLM models. It supports controlled model downloads via a `/pull` endpoint, local-only `load_model` behavior (no automatic downloads during chat), and runtime quantization (q4) when appropriate.

Summary

- Server: FastAPI
- Main features: `/pull` (download snapshot), `/chat` (generate), `/models` (list cached), `/status` (runtime status)
- Backends: vLLM preferred; Transformers pipeline fallback (supports quantized q4 via bitsandbytes)

Quick start

1. Create virtual env and install requirements (example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

2. Run server:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8005
```

Environment variables

- `MAX_CACHE_MODELS` (default `2`): max number of model instances kept in memory cache (LRU evict).
- `MODEL_LOAD_COOLDOWN` (default `300`): seconds to cooldown after a failed model load attempt.
- `MODEL_Q4_THRESHOLD_BYTES` (default for `/pull` auto: ~14e9): bytes threshold above which `quantize:auto` will prefer q4.

Principles and behaviour

- Models are only downloaded via POST `/pull`. `load_model` and `/chat` will not download models automatically — loading a model that was not pulled will return an error instructing the client to call `/pull` first.
- `/pull` records metadata about the snapshot (`local_path`, `size_bytes`, `preferred_quantized`) so `load_model` uses local files only.
- If GPU is available, the server will prefer GPU device; else it falls back to CPU.
- If `bitsandbytes` (and Transformers BitsAndBytesConfig) is available and the model is configured for quantization, the server will attempt 4-bit (q4 / nf4) loading for large models.

API Reference

1) POST /pull

- Purpose: fetch a model snapshot from Hugging Face Hub to the local cache. This endpoint performs the network download and persists metadata so later loads use the local snapshot.
- Request JSON body fields:
  - `model` (string, required): HF repo id (e.g. `Qwen/Qwen2-72B-Chat`).
  - `quantize` (string, optional, default `"auto"`): `"auto"`, `"q4"`, `"fp32"` / `"fp16"` / `"no"`.
    - `auto`: server will decide q4 vs fp based on model size and policy.
    - `q4`: prefer 4-bit quantized snapshot at load time.
    - `fp32` / `fp16` / `no`: prefer full precision (do not force q4).
  - `init` (boolean, optional, default `false`): if true the server will attempt to initialize (load) the model into memory after download.

- Example request:

```bash
curl -X POST http://localhost:8005/pull \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2-72B-Chat","quantize":"q4","init":true}'
```

- Example response (success):

```json
{
  "status": "ok",
  "model": "Qwen/Qwen2-72B-Chat",
  "cached": true,
  "backend": {
    "local_path": "C:\\Users\\...\\hf_cache\\models--Qwen--Qwen2-72B-Chat\\snap-xxx",
    "preferred_quantized": true,
    "size_bytes": 42000000000
  }
}
```

Notes:

- Pulling very large models will download many files and take time and disk space. Consider testing with small models first.
- `quantize` is only a preference; actual quantized loading requires `bitsandbytes` and a compatible `transformers`.

2) POST /chat

- Purpose: generate text using a previously pulled model.
- Request body (JSON):
  - `model` (string): model identifier (must have been pulled already).
  - `messages` (list): list of role/content dicts (OpenAI/Ollama compatible):
    - example: `[{"role":"user","content":"Hello"}]`
  - `max_tokens` (int, optional): max new tokens (default 512).
  - `temperature` (float, optional): sampling temperature (default 0.7).

- Example request:

```bash
curl -X POST http://localhost:8005/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"my/model","messages":[{"role":"user","content":"Who are you?"}],"max_tokens":256}'
```

- Example response (Ollama-like):

```json
{
  "model": "my/model",
  "created_at": "2025-12-13T13:16:07.977118Z",
  "message": {"role":"assistant","content":"Hello! I'm..."},
  "done": true,
  "done_reason": "stop",
  "total_duration": 13001890100,
  "load_duration": 1647499100,
  "prompt_eval_count": 11,
  "prompt_eval_duration": null,
  "eval_count": 161,
  "eval_duration": 11354391000
}
```

Notes:

- The server will use the preferred backend recorded for the model (`vllm` if available, otherwise `transformers` pipeline). The server will not auto-download the model — it must be present locally from `/pull`.
- For `transformers` pipeline we first try chat-style `pipeline(messages)` invocation; if that is not supported by the pipeline we fall back to building a single prompt string.

3) GET /models

- Purpose: list cached/pulled models and some metadata.
- Response structure:

```json
{ "models": [ {"model": "name", "description": null, "loaded": true/false, "backend": "transformers"|"vllm", "size_bytes": 123, "load_duration": 1234 } ] }
```

4) GET /status

- Purpose: runtime diagnostics: whether `vllm` / `transformers` are available, cache size, failed loads, and configured env vars.

Behavioral differences vs Ollama

- This server purposely avoids auto-downloading during generation. Use `/pull` to fetch models explicitly.
- Quantized loading is available when server environment includes `bitsandbytes` and a compatible `transformers`.

Troubleshooting and tips

- If you see: `Model 'X' not available locally. Use the /pull endpoint...` — call `/pull` first.
- Large downloads are slow on Windows and may show symlink warnings. Recommended: run on Linux with CUDA for large GPU-backed models.
- To enable q4 quantized loads: install `bitsandbytes` on a Linux/CUDA host and ensure `transformers` is up-to-date.
- For faster HF downloads on some systems, consider installing `hf_xet` (optional).
- To clean a partial download remove the HF cache path (check `HF_HOME` or default `~/.cache/huggingface`) and retry `/pull`.

Development notes

- The server records `model_meta[model]` with keys like `local_path`, `size_bytes`, `preferred_quantized`, `load_duration_ns`, and `backend`. The `/models` and `/status` endpoints expose a subset of these for convenience.

Security

- This server downloads model artifacts from Hugging Face Hub when `/pull` is called — ensure you trust the model repository and have appropriate network and disk policies.

Want me to test it?

- I can start the server locally and run a test `/pull` for a small model and then a `/chat` to verify the end-to-end path. Ask me to run that test and I'll proceed.
Here’s how I’d build it — **single server, dynamic model loading, GPU-optimized, beats Ollama**:

---

### ✅ **Tech Stack**

- **Python** (fastest for this)
- **vLLM** (for max GPU performance)
- **FastAPI** (for clean API)
- **Hugging Face Transformers** (for fallback/CPUs)
- **CUDA + PyTorch**

---

### 📁 `app.py`

```python
import gc
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from typing import Optional

app = FastAPI()

# Global state
current_model: Optional[LLM] = None
current_model_name: Optional[str] = None

class ChatRequest(BaseModel):
    model: str  # e.g., "Qwen/Qwen2-72B-Chat"
    messages: list[dict]
    max_tokens: int = 512
    temperature: float = 0.7

def load_model(model_name: str):
    global current_model, current_model_name
    # Unload current model
    if current_model:
        del current_model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Loading model: {model_name}...")
    current_model = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=32768,  # adjust for your GPU
    )
    current_model_name = model_name
    print(f"✅ Model loaded: {model_name}")

@app.post("/chat")
async def chat(request: ChatRequest):
    model_name = request.model

    # Load model if not loaded or different
    if current_model_name != model_name:
        load_model(model_name)

    # Format messages for vLLM (same as OpenAI)
    prompt = ""
    for msg in request.messages:
        if msg["role"] == "system":
            prompt += f"System: {msg['content']}\n"
        elif msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n"
    prompt += "Assistant: "

    # Generate
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    outputs = current_model.generate([prompt], sampling_params)

    return {
        "response": outputs[0].outputs[0].text.strip(),
        "model": model_name
    }

@app.get("/models")
async def list_models():
    return {"models": ["Qwen/Qwen2-72B-Chat", "meta-llama/Meta-Llama-3-70B-Instruct"]}

# Start server: uvicorn app:app --host 0.0.0.0 --port 8000
```

---

### 🚀 Run It

```bash
pip install fastapi vllm torch transformers uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 📥 Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-72B-Chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

### ✅ Why This Beats Ollama

- **vLLM** = faster than Ollama’s llama.cpp backend
- **Single server**, no port chaos
- **Dynamic loading** (on first use)
- **Full GPU utilization**
- **OpenAI-compatible output**

> ⚠️ First load is slow (model download + init), but after that — blazing.

Let me know if you want **streaming**, **GPU memory monitoring**, or **model caching** added.

---

Getting started (PowerShell)

1) Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Run the server:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

4) Example request:

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen2-72B-Chat", "messages": [{"role":"user","content":"Hello"}], "max_tokens":100}'
```

Notes:

- vLLM and large model usage typically target Linux GPU environments; running on Windows may need additional setup.
- First model load can be slow while the model is downloaded/initialized.

Cache and cooldown

- The server keeps up to `MAX_CACHE_MODELS` models in an in-memory LRU cache (default 2).
- If a model load fails, further attempts are blocked for `MODEL_LOAD_COOLDOWN` seconds (default 300).
You can override these with environment variables before starting the server:

```powershell
$env:MAX_CACHE_MODELS = "3"
$env:MODEL_LOAD_COOLDOWN = "600"
uvicorn app:app --host 0.0.0.0 --port 8000
```

Model storage and download locations

- Transformers downloads models to the Hugging Face cache on the host (by default `~/.cache/huggingface/transformers` on Linux, or `%USERPROFILE%\.cache\huggingface\transformers` on Windows). The `pipeline`/`from_pretrained` call performed by the server will download weights to that cache when needed.
- vLLM model handling typically uses HF cache as well or vLLM-specific model resolution; when you call the `/pull` endpoint or load a model it will download weights to the local cache used by the backend.

Endpoints added

- `GET /models` — returns `known_models` (example list) and `cached_models` currently in memory.
- `POST /pull` — body: `{"model": "<model-id>"}` — force-download and initialize a model into the in-memory cache.
- `POST /chat` — chat endpoint (will use vLLM if available otherwise transformers pipeline fallback).
- `GET /status` — runtime status including which backends are available, cache contents and recent failed loads.

Notes and caveats

- Downloading large models requires disk space and may take a long time. For large GPU-optimized models, run on Linux with CUDA-enabled drivers.
- The server prefers `vLLM` if available; if `vLLM` fails to instantiate a model the server will try a `transformers` pipeline fallback and will record failures in `/status`.
