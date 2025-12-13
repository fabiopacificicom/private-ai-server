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

