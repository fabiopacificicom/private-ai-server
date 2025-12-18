# AI Inference Server

A high-performance, GPU-optimized FastAPI server for serving Hugging Face language models. Built as an Ollama-compatible alternative with dynamic model loading, automatic 4-bit quantization for large models, and explicit control over downloads.

## Overview

**What it does:** Dynamically loads and serves LLMs from Hugging Face Hub with GPU acceleration, automatic quantization, and intelligent caching.

**Why use it:**

- 🚀 **GPU-optimized**: Prefers vLLM backend, falls back to transformers with CUDA support
- 💾 **Smart memory management**: Automatic 4-bit quantization for models >14GB, LRU cache with configurable limits
- 🎯 **Explicit control**: Models must be explicitly pulled before use (no surprise downloads during inference)
- 🔧 **Production-ready**: Background job system for non-blocking downloads, CUDA OOM prevention, MoE/custom model support
- 📊 **Observable**: Comprehensive diagnostics endpoints (`/health`, `/status`, `/diag`, `/jobs`)
- 🔄 **Streaming**: Real-time response streaming with Server-Sent Events (SSE)

**Key Features:**

- **Backends**: vLLM (preferred for max GPU performance) → Transformers pipeline fallback
- **Auto-quantization**: 4-bit (q4/nf4) for models exceeding threshold using bitsandbytes
- **Streaming responses**: Server-Sent Events (SSE) for real-time text generation
- **Non-blocking downloads**: Background `/pull` jobs with status tracking
- **LRU caching**: Configurable model cache with automatic eviction
- **CUDA-optimized**: Memory fragmentation prevention, aggressive cleanup, RTX 4090 tested
- **MoE support**: Handles custom configs (Qwen3-Omni) with `trust_remote_code` fallback
- **Health monitoring**: `/health` endpoint for load balancers and production deployments

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.x (for GPU acceleration)
- 16GB+ VRAM recommended for large models with quantization

### Installation

1. **Clone and setup environment:**

```powershell
git clone <repo-url>
cd ai-server-py
python -m venv .venv .\.venv\Scripts\Activate.ps1
```

2. **Install dependencies:**

```powershell
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

3. **Enable CUDA (Windows):**
   - If you have an NVIDIA GPU but PyTorch shows CPU-only, run:

   ```powershell .\setup_cuda_pytorch.ps1
   ```

   - This installs CUDA-enabled PyTorch (CUDA 12.6) and bitsandbytes for quantization

4. **Start the server:**

```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8005
```

### First Steps

1. **Check diagnostics:**

```bash
curl http://localhost:8005/diag
# Should show: cuda_available: true, device: RTX 4090...
```

2. **Pull a model:**

```bash
curl -X POST http://localhost:8005/pull \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt2","quantize":"auto","init":true}'
```

3. **Chat with the model:**

```bash
curl -X POST http://localhost:8005/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt2","messages":[{"role":"user","content":"Hello!"}]}'
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CACHE_MODELS` | `2` | Maximum models kept in memory (LRU eviction) |
| `MODEL_LOAD_COOLDOWN` | `300` | Cooldown seconds after failed load before retry |
| `MODEL_Q4_THRESHOLD_BYTES` | `~14e9` | Size threshold (bytes) for auto q4 quantization |
| `MAX_CONCURRENT_PULLS` | `2` | Max simultaneous background downloads |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | Auto-set to reduce CUDA fragmentation |

**Example:**

```powershell
$env:MAX_CACHE_MODELS = "1"
$env:MODEL_Q4_THRESHOLD_BYTES = "30000000000"  # 30GB threshold
python -m uvicorn app:app --host 0.0.0.0 --port 8005
```

### Core Principles

1. **Explicit downloads only**: Models must be pulled via `/pull` before use. `/chat` will never auto-download.
2. **Local-only loading**: Once pulled, models load from local HF cache (`~/.cache/huggingface/hub`)
3. **GPU-first**: Automatically uses GPU when available, CPU fallback otherwise
4. **Smart quantization**: Models >14GB auto-use 4-bit when `bitsandbytes` available
5. **LRU caching**: Keeps `MAX_CACHE_MODELS` in memory, evicts oldest when limit exceeded
6. **Failed-load protection**: 300s cooldown after failed loads prevents retry storms

## API Reference

### POST /pull

**Download and optionally initialize a model**

**Request body:**

```json
{
  "model": "Qwen/Qwen3-0.6B",           // HF repo ID (required)
  "quantize": "auto",                    // auto|q4|fp16|no (optional, default: auto)
  "init": true                           // Load after download (optional, default: false)
}
```

**Quantization options:**

- `auto`: Server decides (q4 if model >14GB, otherwise fp16)
- `q4`: Force 4-bit quantization (requires bitsandbytes + CUDA)
- `fp16`/`no`: Full precision (no quantization)

**Response:**

```json
{
  "status": "accepted",
  "job_id": "uuid-here"                 // Poll via /jobs/{job_id}
}
```

**Example:**

```bash
# Pull and initialize small model
curl -X POST http://localhost:8005/pull \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt2","init":true}'

# Pull large model with forced q4
curl -X POST http://localhost:8005/pull \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Omni-30B-A3B-Instruct","quantize":"q4","init":true}'
```

---

### POST /chat

**Generate text from a loaded model**

**Request body:**

```json
{
  "model": "gpt2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 512,                    // optional, default: 512
  "temperature": 0.7                    // optional, default: 0.7
}
```

**Response (Ollama-compatible):**

```json
{
  "model": "gpt2",
  "created_at": "2025-12-13T18:38:39.023109Z",
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help you today?"
  },
  "done": true,
  "done_reason": "stop",
  "total_duration": 5656717800,         // nanoseconds
  "load_duration": 871150500,
  "prompt_eval_count": 19,
  "eval_count": 531,
  "eval_duration": 4785567300
}
```

**Streaming support:**

To enable streaming responses, set `"stream": true` in the request:

```bash
# Example: Streaming chat request
curl -N -X POST http://localhost:8005/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [{"role": "user", "content": "Write a short story"}],
    "stream": true,
    "max_tokens": 100
  }'
```

**Streaming response format (SSE):**

```
data: {"delta": {"content": "Once"}, "done": false}

data: {"delta": {"content": " upon"}, "done": false}

data: {"delta": {"content": " a time"}, "done": false}

...

data: {"delta": {}, "done": true}
```

- Chunks arrive as Server-Sent Events (SSE) with `text/event-stream` media type
- Each chunk has `delta.content` with incremental text
- Final chunk has `done: true` with empty delta
- Supports both vLLM and transformers backends

---

### GET /models

**List all available models**

**Response:**

```json
{
  "models": [
    {
      "model": "gpt2",
      "loaded": true,
      "backend": "transformers_pipeline",
      "size_bytes": 548118077,
      "local_path": "C:\\Users\\...\\models--gpt2\\snapshots\\...",
      "load_duration": 871150500
    },
    {
      "model": "Qwen/Qwen3-0.6B",
      "loaded": false,
      "size_bytes": 1519182405,
      "local_path": "C:\\Users\\...\\models--Qwen--Qwen3-0.6B"
    }
  ]
}
```

---

### GET /jobs

**List all pull jobs**

**Response:**

```json
{
  "jobs": [
    {
      "id": "uuid",
      "model": "gpt2",
      "status": "succeeded",              // running|succeeded|failed
      "created_at": "2025-12-13T18:38:31Z",
      "finished_at": "2025-12-13T18:38:33Z",
      "local_path": "C:\\Users\\...\\models--gpt2\\...",
      "preferred_quantized": false
    }
  ]
}
```

---

### GET /jobs/{job_id}

**Get specific job status**

Poll this endpoint to track download progress after calling `/pull`.

---

### GET /status

**Server runtime diagnostics**

**Response:**

```json
{
  "vllm_available": false,
  "transformers_available": true,
  "bitsandbytes_available": true,
  "hf_hub_available": true,
  "cache_size": 1,
  "max_cache": 2,
  "cached_models": ["gpt2"],
  "failed_loads": {},
  "config": {
    "max_cache_models": 2,
    "cooldown_seconds": 300,
    "q4_threshold_bytes": 14000000000
  }
}
```

---

### GET /health

**Health check endpoint for load balancers and monitoring**

Returns server health status, uptime, cache statistics, and GPU metrics.

**Response:**

```json
{
  "status": "healthy",                   // healthy|degraded
  "uptime_seconds": 3600,
  "models_cached": 2,
  "cache_limit": 2,
  "downloads_active": 0,
  "downloads_queued": 1,
  "torch_version": "2.9.1+cu126",
  "cuda_available": true,
  "gpu_status": "available",             // available|unavailable|error
  "gpu_memory_allocated_mb": 8192.5,
  "gpu_memory_reserved_mb": 10240.0
}
```

**Status values:**

- `healthy`: All inference backends available, server operational
- `degraded`: No inference backends (transformers/vLLM) available

**GPU status:**

- `available`: CUDA available and GPU accessible
- `unavailable`: CUDA not available or no GPU
- `error`: Error while collecting GPU stats

**Use cases:**

- Load balancer health checks
- Monitoring and alerting
- Capacity planning (via GPU memory stats)
- Uptime tracking

---

### GET /diag

**CUDA/PyTorch diagnostics**

**Response:**

```json
{
  "torch_installed": true,
  "torch_version": "2.9.1+cu126",
  "cuda_available": true,
  "cuda_count": 1,
  "cuda_devices": [
    {"index": 0, "name": "NVIDIA GeForce RTX 4090 Laptop GPU"}
  ]
}
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **"Model not available locally"** | Call `/pull` first to download the model |
| **CUDA OOM errors** | 1. Reduce `MAX_CACHE_MODELS` to `1`<br>2. Use q4 quantization<br>3. Restart server to clear GPU<br>4. Check `nvidia-smi` for other GPU processes |
| **"requires accelerate" error** | `pip install accelerate` |
| **CPU-only despite having GPU** | Run `.\setup_cuda_pytorch.ps1` to install CUDA PyTorch |
| **Slow inference** | Ensure CUDA is enabled (`/diag` should show `cuda_available: true`) |
| **Download fails/hangs** | Clear HF cache and retry: `rm -r ~/.cache/huggingface/hub/models--<model-name>` |

### CUDA OOM Deep Dive

The server automatically sets `PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation. If you still hit OOM:

1. **Check GPU memory:**

   ```powershell
   nvidia-smi
   ```

2. **Reduce cache size:**

   ```powershell
   $env:MAX_CACHE_MODELS = "1"
   ```

3. **Force quantization:**

   ```bash
   # Pull with explicit q4
   curl -X POST http://localhost:8005/pull \
     -H "Content-Type: application/json" \
     -d '{"model":"your/model","quantize":"q4","init":true}'
   ```

4. **Clear GPU memory completely:**

   ```powershell
   Get-Process python | Stop-Process -Force
   # Then restart server
   ```

### Model Compatibility

**Tested and working:**

- ✅ GPT-2 (548MB)
- ✅ Qwen/Qwen3-0.6B (1.5GB)
- ✅ facebook/opt-1.3b (5.3GB)
- ✅ Qwen/Qwen3-Omni-30B-A3B-Instruct (70GB with q4)

**Known limitations:**

- MoE models require `trust_remote_code=True` (security consideration)
- Very large models (>70GB) may need CPU offloading even with q4
- vLLM backend requires Linux + CUDA (Windows uses transformers fallback)

## Architecture

### Request Flow

```
1. Client → POST /pull → Background job starts

---

## Production Deployment

### Docker

```dockerfile
# See Dockerfile for complete setup
docker build -t ai-server .
docker run -p 8005:8005 --gpus all ai-server
```

### Performance Tuning

**For single large model (70B+):**

```powershell
$env:MAX_CACHE_MODELS = "1"           # Only one model in memory
$env:MODEL_Q4_THRESHOLD_BYTES = "0"   # Always use q4
python -m uvicorn app:app --host 0.0.0.0 --port 8005
```

**For multiple small models:**

```powershell
$env:MAX_CACHE_MODELS = "5"           # Keep 5 models cached
$env:MODEL_Q4_THRESHOLD_BYTES = "30000000000"  # Only q4 for >30GB
python -m uvicorn app:app --host 0.0.0.0 --port 8005
```

### Monitoring

- **Health check**: `GET /status`
- **GPU usage**: `nvidia-smi` or `nvidia-smi dmon`
- **Job tracking**: `GET /jobs`
- **Diagnostics**: `GET /diag`

---

## Security

 **Important**: This server downloads and executes model code from Hugging Face Hub.

- MoE/custom models use `trust_remote_code=True` (executes model repo code)
- Only pull models from trusted sources
- Review model repos before pulling
- Consider network isolation for production deployments
- HF cache location: `~/.cache/huggingface/hub` (or `HF_HOME`)

---

## Development

### Project Structure

```
ai-server-py/
 app.py                    # Main FastAPI server
 requirements.txt          # Python dependencies
 setup_cuda_pytorch.ps1    # CUDA setup helper (Windows)
 scripts/
    dai-cazzo.ps1        # Test script
 .specs/                   # Feature specs and plans
```

### Key Dependencies

- `fastapi` + `uvicorn`: API server
- `torch` (CUDA): GPU acceleration
- `transformers`: Model loading
- `bitsandbytes`: 4-bit quantization
- `accelerate`: Device mapping for large models
- `vllm` (optional): High-performance inference
- `huggingface_hub`: Model downloads

### Running Tests

```powershell
# Start server in one terminal
python -m uvicorn app:app --host 0.0.0.0 --port 8005

# In another terminal
.\scripts\dai-cazzo.ps1  # Runs diagnostic + pull + chat test
```

---

## Comparison: This Server vs Ollama

| Feature | This Server | Ollama |
|---------|-------------|--------|
| **Backend** | vLLM  transformers | llama.cpp |
| **GPU optimization** | Native CUDA + quantization | Metal/CUDA via llama.cpp |
| **Model format** | Native HF (safetensors) | GGUF conversion required |
| **Auto-download** | Explicit `/pull` only | Auto-downloads on first use |
| **Quantization** | Auto 4-bit for large models | GGUF quant levels |
| **MoE support** |  (trust_remote_code) | Limited |
| **Job tracking** |  Background jobs + status |  |
| **Multi-model cache** |  Configurable LRU |  |
| **API format** | Ollama-compatible | Native Ollama |

**Use this server if:**

- You want native HF model support (no GGUF conversion)
- You need explicit control over downloads
- You want automatic quantization for large models
- You prefer Python/FastAPI stack

**Use Ollama if:**

- You need cross-platform simplicity
- You prefer GGUF ecosystem
- You want one-command setup

---

## License

MIT

---

## Roadmap

**Current Version**: v0.9 (MVP Complete)

See [`.specs/ROADMAP.md`](.specs/ROADMAP.md) for detailed feature roadmap and timeline.

**Current Release (v1.0)**: ✅ Production-Ready - COMPLETED

- ✅ Streaming responses (SSE)
- ✅ Health check endpoint
- ✅ Request timeouts
- ✅ Download progress tracking
- ✅ Persistent job storage (SQLite)
- ✅ Improved error messages

**Next Release (v1.5)**: Advanced Features

- 🔄 Multi-turn conversation sessions
- 🎯 Custom sampling parameters  
- 🏷️ Model aliases and presets
- 🔄 Model preloading on startup
- ⚖️ Load balancing across models
- 📊 Advanced metrics and monitoring

**Future Releases**:

- v1.5: Advanced features (sessions, sampling params, model aliases)
- v2.0: Production hardening (auth, monitoring, scaling)
- v2.5: Ecosystem integration (OpenAI API, LangChain)
- v3.0: Performance optimizations (batching, speculative decoding)

---

## Documentation

### For Users

- [README.md](README.md) - Getting started, API reference, troubleshooting
- [.specs/ROADMAP.md](.specs/ROADMAP.md) - Feature roadmap and timeline

### For Developers

- [.github/instructions/coder-guidelines.instructions.md](.github/instructions/coder-guidelines.instructions.md) - Coding standards and patterns
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - AI agent onboarding guide
- [.specs/SERVER-REVIEW.md](.specs/SERVER-REVIEW.md) - Architecture review and recommendations
- [.specs/plans/phase1-implementation.md](.specs/plans/phase1-implementation.md) - Detailed v1.0 implementation plan

---

## Contributing

Issues and PRs welcome! See documentation above for development guidelines and roadmap.

**Quick Links**:

- [Roadmap](.specs/ROADMAP.md) - What's planned
- [Coder Guidelines](.github/instructions/coder-guidelines.instructions.md) - How to code
- [Phase 1 Plan](.specs/plans/phase1-implementation.md) - Next steps
