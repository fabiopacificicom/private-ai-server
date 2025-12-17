# AI Inference Server - Roadmap

## Project Status

**Current Version**: v0.9 (MVP Complete)

The server is fully functional with core features:
- ✅ Dynamic model loading with vLLM/transformers backends
- ✅ Non-blocking background downloads via `/pull` endpoint
- ✅ Automatic 4-bit quantization for large models (>14GB)
- ✅ LRU model caching with configurable limits
- ✅ CUDA memory management and OOM prevention
- ✅ Job tracking system for download status
- ✅ Ollama-compatible `/chat` endpoint

**Tested on**: RTX 4090 Laptop GPU, CUDA 12.6, Windows 11

**Validated models**: gpt2 (548MB), Qwen2-0.6B (1.5GB), opt-1.3b (5.3GB), Qwen3-Omni-30B (70GB with q4)

---

## Phase 1: Stability & Performance (v1.0)

**Goal**: Production-ready server with improved reliability, performance monitoring, and error handling.

**Timeline**: 2-3 weeks

### 1.1 Streaming Support

**Priority**: HIGH  
**Effort**: Medium (1-2 days)

**Rationale**: Streaming responses improve UX for long generations and enable real-time interaction.

**Implementation**:
- Add `POST /chat` with `stream=true` parameter
- Use Server-Sent Events (SSE) for streaming chunks
- Support both vLLM and transformers streaming APIs
- Return JSON chunks: `{"delta": {"content": "text"}, "done": false}`

**Files to modify**:
- `app.py`: Add streaming logic to `/chat` endpoint
- Update `load_model()` to store streaming-capable backend info

**Acceptance criteria**:
- `/chat` with `stream=true` returns SSE stream
- Client can process chunks in real-time
- Final chunk includes `done=true` and token counts

**Testing**:
```powershell
# Test with curl
curl -N http://localhost:8005/chat -H "Content-Type: application/json" -d '{"model":"gpt2", "messages":[...], "stream":true}'
```

**References**:
- [FastAPI SSE](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [vLLM streaming](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

---

### 1.2 Progress Tracking for Downloads

**Priority**: MEDIUM  
**Effort**: Small (0.5-1 day)

**Rationale**: Large model downloads (50-70GB) take significant time; users need progress feedback.

**Implementation**:
- Add `progress` field to job dict: `{"progress": 0.0-1.0}`
- Periodically check downloaded bytes during `snapshot_download()`
- Update `GET /jobs/{job_id}` to return `progress`, `downloaded_bytes`, `total_bytes`

**Technical approach**:
```python
# In _background_pull(), before snapshot_download:
# 1. Get total size from model_info()
# 2. Start download in thread
# 3. In async loop, poll local_path for size growth every 2s
# 4. Update jobs[job_id]["progress"] = current_bytes / total_bytes
```

**Challenges**:
- `snapshot_download()` is blocking; requires thread + polling
- Cache directory structure may be complex (multiple files)

**Acceptance criteria**:
- Job progress updates every 2-5 seconds
- Progress reaches 1.0 when download completes
- Works for models with known size (via `model_info()`)

---

### 1.3 Health Check Endpoint

**Priority**: HIGH  
**Effort**: Small (0.5 day)

**Rationale**: Production deployments need health checks for load balancers, monitoring, and orchestration.

**Implementation**:
- Add `GET /health` endpoint
- Return 200 OK with status: `{"status": "healthy", "uptime": seconds, "models_loaded": count}`
- Add `GET /metrics` for Prometheus-compatible metrics (optional)

**Files to modify**:
- `app.py`: Add health endpoint

**Metrics to expose**:
```python
{
    "status": "healthy",  # or "degraded" if CUDA unavailable
    "uptime_seconds": 12345,
    "models_cached": 2,
    "cache_limit": 2,
    "downloads_active": 1,
    "downloads_queued": 3,
    "torch_version": "2.9.1+cu126",
    "cuda_available": true,
    "gpu_memory_allocated_mb": 8192,
    "gpu_memory_reserved_mb": 10240,
}
```

**Acceptance criteria**:
- `/health` returns 200 when server operational
- Includes GPU memory stats if CUDA available
- Degraded status if critical backends missing

---

### 1.4 Request Timeout & Cancellation

**Priority**: MEDIUM  
**Effort**: Medium (1 day)

**Rationale**: Long-running inference requests can block resources; need configurable timeouts.

**Implementation**:
- Add `timeout` parameter to `/chat` (default 120s)
- Use `asyncio.wait_for()` to enforce timeout
- Return 408 Request Timeout on expiration
- Support request cancellation via client disconnect

**Files to modify**:
- `app.py`: Wrap inference in timeout context

**Example**:
```python
@app.post("/chat")
async def chat(payload: Dict[str, Any]):
    timeout = payload.get("timeout", 120)
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_generate_text, ...),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")
```

**Acceptance criteria**:
- Requests timeout after specified duration
- GPU resources released on timeout
- Client receives clear timeout error

---

### 1.5 Persistent Job Storage

**Priority**: MEDIUM  
**Effort**: Medium (1-2 days)

**Rationale**: In-memory `jobs` dict lost on restart; production needs durability.

**Implementation options**:

**Option A: SQLite** (simpler)
- Create `jobs.db` with schema: `id, model, status, created_at, finished_at, error, local_path, ...`
- Persist job updates immediately
- Load active jobs on startup

**Option B: Redis** (scalable)
- Store jobs as JSON in Redis hashes
- Enable pub/sub for job status updates
- Support distributed deployments

**Recommended**: Start with SQLite for simplicity.

**Files to create**:
- `database.py`: Job persistence layer

**Files to modify**:
- `app.py`: Replace `jobs` dict with database calls

**Acceptance criteria**:
- Jobs survive server restarts
- Job history queryable with filters
- Minimal performance impact (<10ms per job update)

---

### 1.6 Improved Error Messages

**Priority**: LOW  
**Effort**: Small (0.5 day)

**Rationale**: Current errors are technical; improve UX with actionable messages.

**Examples**:

| Current | Improved |
|---------|----------|
| `RuntimeError: Model not available locally` | `Model 'gpt2' not downloaded. Run: POST /pull {"model":"gpt2"}` |
| `HTTPException: No inference backend available` | `Server misconfigured: transformers library not installed. Check installation.` |
| `CUDA OOM` | `GPU out of memory. Try: 1) Reduce cache size (MAX_CACHE_MODELS=1), 2) Use smaller model, 3) Enable quantization` |

**Implementation**:
- Update exception messages with specific actions
- Add troubleshooting URLs to errors
- Include relevant diagnostics (GPU memory, cache size)

---

## Phase 2: Advanced Features (v1.5)

**Goal**: Extend functionality with advanced inference features and optimizations.

**Timeline**: 4-6 weeks

### 2.1 Multi-Turn Conversation Context

**Priority**: MEDIUM  
**Effort**: Medium (2 days)

**Rationale**: Current `/chat` is stateless; add session management for multi-turn conversations.

**Implementation**:
- Add session storage (in-memory or Redis)
- Accept `session_id` in `/chat` payload
- Store conversation history per session
- Auto-prune old sessions (TTL: 1 hour)

**API**:
```json
POST /chat
{
    "model": "gpt2",
    "session_id": "optional-uuid",
    "messages": [{"role": "user", "content": "..."}],
    "max_history": 10
}
```

**Files to modify**:
- `app.py`: Add session management logic

**Acceptance criteria**:
- Multi-turn conversations maintain context
- Sessions auto-expire after inactivity
- Session storage configurable (memory/Redis)

---

### 2.2 Model Preloading on Startup

**Priority**: LOW  
**Effort**: Small (0.5 day)

**Rationale**: Reduce cold-start latency by preloading frequently-used models.

**Implementation**:
- Add `PRELOAD_MODELS` environment variable (comma-separated list)
- Load models in background task on startup
- Respect cache limits

**Example**:
```bash
PRELOAD_MODELS=gpt2,meta-llama/Llama-2-7b-chat
```

**Files to modify**:
- `app.py`: Add startup event handler

**Acceptance criteria**:
- Specified models loaded on startup
- Server remains responsive during preload
- Logs indicate preload progress

---

### 2.3 Custom Sampling Parameters

**Priority**: MEDIUM  
**Effort**: Small (1 day)

**Rationale**: Power users need control over temperature, top_p, top_k, repetition penalty.

**Implementation**:
- Accept sampling parameters in `/chat` payload
- Pass to backend (vLLM or transformers)
- Validate ranges (temperature 0-2, top_p 0-1)

**API**:
```json
POST /chat
{
    "model": "gpt2",
    "messages": [...],
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 256,
    "repetition_penalty": 1.1
}
```

**Files to modify**:
- `app.py`: Extract sampling params, pass to backends

**Acceptance criteria**:
- Sampling params affect generation
- Invalid params return 400 error
- Works with both vLLM and transformers

---

### 2.4 Model Aliases

**Priority**: LOW  
**Effort**: Small (0.5 day)

**Rationale**: Long model names (e.g., `meta-llama/Meta-Llama-3-70B-Instruct`) are unwieldy.

**Implementation**:
- Add `aliases.json` mapping short names to full IDs
- Accept aliases in all endpoints
- Default aliases: `llama3-70b`, `qwen2-72b`, `gpt2`

**Example**:
```json
{
    "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "qwen2": "Qwen/Qwen2-72B-Chat"
}
```

**Files to create**:
- `aliases.json`

**Files to modify**:
- `app.py`: Resolve aliases before model loading

---

### 2.5 Quantization Profiles

**Priority**: MEDIUM  
**Effort**: Medium (1-2 days)

**Rationale**: Support more quantization options (8-bit, GPTQ, AWQ) for different performance/quality tradeoffs.

**Implementation**:
- Add quantization profiles: `fp16`, `int8`, `int4`, `gptq`, `awq`
- Auto-select profile based on model metadata
- Allow override via `/pull` payload

**Profiles**:
```python
{
    "fp16": {},  # No quantization, FP16 default
    "int8": {
        "load_in_8bit": True,
        "bnb_8bit_compute_dtype": torch.float16,
    },
    "int4": {  # Current default
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
    },
}
```

**Files to modify**:
- `app.py`: Add profile selection logic

**Acceptance criteria**:
- Multiple quantization modes available
- User can specify profile in `/pull`
- Performance/quality documented for each profile

---

## Phase 3: Production Hardening (v2.0)

**Goal**: Enterprise-ready deployment with security, observability, and scalability.

**Timeline**: 6-8 weeks

### 3.1 Authentication & Authorization

**Priority**: HIGH  
**Effort**: Medium (2-3 days)

**Rationale**: Production deployments need access control.

**Implementation**:
- API key authentication via `Authorization: Bearer <key>` header
- Support multiple keys with rate limits
- Optional: integrate with OAuth2/JWT

**Files to create**:
- `auth.py`: Authentication middleware

**Files to modify**:
- `app.py`: Add auth dependency to endpoints

**Acceptance criteria**:
- Endpoints reject requests without valid key
- Keys stored securely (environment or encrypted file)
- Rate limiting per key (e.g., 100 req/min)

---

### 3.2 Request Logging & Audit Trail

**Priority**: MEDIUM  
**Effort**: Small (1 day)

**Rationale**: Production needs request logging for debugging, analytics, and compliance.

**Implementation**:
- Log all requests: timestamp, endpoint, model, user, duration, status
- Store in structured format (JSON lines)
- Optional: ship to centralized logging (ELK, Splunk)

**Files to create**:
- `logs/requests.jsonl`

**Files to modify**:
- `app.py`: Add middleware to log requests

**Example log entry**:
```json
{
    "timestamp": "2025-12-17T10:30:00Z",
    "endpoint": "/chat",
    "model": "gpt2",
    "user": "api-key-123",
    "duration_ms": 1234,
    "status": 200,
    "tokens_generated": 50
}
```

---

### 3.3 GPU Memory Monitoring & Alerts

**Priority**: HIGH  
**Effort**: Medium (1-2 days)

**Rationale**: Prevent OOM errors with proactive monitoring.

**Implementation**:
- Add `/metrics/gpu` endpoint with memory stats
- Log warnings when memory >80% utilized
- Optional: integrate with Prometheus/Grafana

**Metrics**:
```python
{
    "gpu_index": 0,
    "name": "NVIDIA RTX 4090",
    "memory_allocated_mb": 12288,
    "memory_reserved_mb": 14336,
    "memory_total_mb": 16384,
    "utilization_percent": 75,
    "temperature_c": 68,
}
```

**Files to modify**:
- `app.py`: Add GPU monitoring endpoint

**Acceptance criteria**:
- Real-time GPU memory visibility
- Alerts when nearing capacity
- Historical metrics queryable

---

### 3.4 Horizontal Scaling Support

**Priority**: LOW  
**Effort**: Large (1 week)

**Rationale**: Single server limits throughput; enable load balancing across multiple instances.

**Implementation**:
- Externalize state (Redis for jobs, sessions)
- Support sticky sessions for multi-turn conversations
- Add load balancer configuration (Nginx, Traefik)

**Architecture**:
```
Client → Load Balancer → [Server 1, Server 2, Server 3]
                              ↓
                          Redis (shared state)
```

**Files to create**:
- `docker-compose.yml`: Multi-instance setup
- `nginx.conf`: Load balancer config

**Files to modify**:
- `app.py`: Use Redis for jobs/sessions

**Acceptance criteria**:
- Multiple instances serve requests concurrently
- Jobs/sessions shared across instances
- No request duplication

---

### 3.5 Model Hot-Swapping

**Priority**: LOW  
**Effort**: Medium (2 days)

**Rationale**: Update models without server restart.

**Implementation**:
- Add `POST /models/{model_name}/reload` endpoint
- Evict model from cache and reload from disk
- Support rolling updates (reload one model at a time)

**Files to modify**:
- `app.py`: Add reload endpoint

**Acceptance criteria**:
- Models reloaded without server restart
- Active requests complete before reload
- No downtime for other models

---

## Phase 4: Ecosystem Integration (v2.5)

**Goal**: Integrate with popular AI tools and frameworks.

**Timeline**: 4-6 weeks

### 4.1 OpenAI-Compatible API

**Priority**: HIGH  
**Effort**: Medium (2-3 days)

**Rationale**: Many tools expect OpenAI API format; maximize compatibility.

**Implementation**:
- Add `/v1/chat/completions` endpoint (OpenAI format)
- Add `/v1/models` endpoint
- Support streaming with `stream=true`

**Files to modify**:
- `app.py`: Add OpenAI-compatible endpoints

**Acceptance criteria**:
- Works with LangChain, LlamaIndex, OpenAI SDKs
- Streaming compatible with OpenAI format
- Model listing matches OpenAI schema

---

### 4.2 Function Calling Support

**Priority**: MEDIUM  
**Effort**: Large (1 week)

**Rationale**: Enable tool use for agents and chatbots.

**Implementation**:
- Parse function definitions from request
- Format prompts for function calling
- Extract function calls from responses
- Support multi-step tool use

**API**:
```json
POST /chat
{
    "model": "gpt2",
    "messages": [...],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {...}
            }
        }
    ]
}
```

**Challenges**:
- Not all models support function calling
- Requires prompt engineering per model

**Acceptance criteria**:
- Function calls extracted from responses
- Multi-step tool use supported
- Works with compatible models

---

### 4.3 LangChain Integration

**Priority**: MEDIUM  
**Effort**: Small (1 day)

**Rationale**: LangChain is popular for AI app development.

**Implementation**:
- Create LangChain wrapper class
- Publish as separate package (`langchain-ai-server`)
- Support both completion and chat interfaces

**Files to create**:
- `integrations/langchain/ai_server.py`

**Example**:
```python
from langchain_ai_server import AIServerLLM

llm = AIServerLLM(base_url="http://localhost:8005", model="gpt2")
response = llm("What is the capital of France?")
```

---

### 4.4 Docker Compose Stack

**Priority**: HIGH  
**Effort**: Small (1 day)

**Rationale**: Simplify deployment with containers.

**Implementation**:
- Create `Dockerfile` for server
- Add `docker-compose.yml` with server, Redis, Nginx
- Include GPU support (NVIDIA Container Toolkit)

**Files to create**:
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

**Acceptance criteria**:
- Single command deploys full stack
- GPU accessible inside container
- Persistent storage for models

---

## Phase 5: Advanced Optimizations (v3.0)

**Goal**: Squeeze maximum performance from hardware.

**Timeline**: 6-8 weeks

### 5.1 Continuous Batching (vLLM)

**Priority**: HIGH  
**Effort**: Medium (2-3 days)

**Rationale**: Process multiple requests concurrently for higher throughput.

**Implementation**:
- Configure vLLM with continuous batching
- Tune batch size, KV cache size
- Benchmark throughput improvements

**Files to modify**:
- `app.py`: Configure vLLM batching

**Expected improvement**: 2-4x throughput on vLLM backend

---

### 5.2 Model Quantization Caching

**Priority**: MEDIUM  
**Effort**: Medium (2 days)

**Rationale**: Quantizing large models on-the-fly is slow; cache quantized versions.

**Implementation**:
- Save quantized models to disk after first load
- Reload from quantized cache on subsequent loads
- Auto-clean old quantized models (LRU)

**Files to modify**:
- `app.py`: Add quantized model caching

**Expected improvement**: 5-10x faster cold starts for large models

---

### 5.3 Speculative Decoding

**Priority**: LOW  
**Effort**: Large (1 week)

**Rationale**: Use small draft model + large verification model for faster generation.

**Implementation**:
- Support draft model specification
- Coordinate draft + verification passes
- Tune acceptance rate

**Challenges**:
- Requires compatible draft/verification models
- Complex implementation

**Expected improvement**: 2-3x generation speed

---

### 5.4 INT4 Quantization (GPTQ/AWQ)

**Priority**: MEDIUM  
**Effort**: Medium (2-3 days)

**Rationale**: GPTQ/AWQ provide better quality than bitsandbytes at same bit width.

**Implementation**:
- Support loading GPTQ/AWQ models
- Auto-detect quantization format from model
- Benchmark quality vs bitsandbytes

**Dependencies**:
- `auto-gptq` or `autoawq` library

**Expected improvement**: Better quality at 4-bit, same memory usage

---

## Deprecation & Migration Plan

### Breaking Changes in v2.0

1. **Job storage schema**: Migration script provided for SQLite
2. **Authentication required**: Default API keys in `.env.example`
3. **Response format**: OpenAI-compatible format becomes default

### Migration Guide

**From v0.9 to v1.0**:
- No breaking changes
- Update `.env` with new variables (`PRELOAD_MODELS`, etc.)

**From v1.x to v2.0**:
- Add API keys to environment
- Migrate job data: `python migrate_jobs.py`
- Update client code to use `/v1/chat/completions`

---

## Success Metrics

### v1.0 Goals
- [ ] 99% uptime over 1 week
- [ ] <500ms latency for small models (gpt2)
- [ ] Support models up to 70B parameters
- [ ] Zero CUDA OOM errors with proper config

### v2.0 Goals
- [ ] Support 100+ concurrent requests
- [ ] <50ms p99 latency for health checks
- [ ] 100% OpenAI API compatibility

### v3.0 Goals
- [ ] 10x throughput vs v1.0 (via batching)
- [ ] 5x faster cold starts (via quantization cache)
- [ ] Support 100B+ parameter models

---

## Contributing

See `CONTRIBUTING.md` for development workflow.

**Quick Links**:
- [Coder Guidelines](.github/instructions/coder-guidelines.instructions.md)
- [Copilot Instructions](.github/copilot-instructions.md)
- [Implementation Plans](.specs/plans/)

---

## Changelog

### v0.9 (2025-12-17)
- Initial MVP release
- Core inference with vLLM/transformers
- Non-blocking downloads
- Job tracking system
- LRU caching
- CUDA memory management

### Upcoming

**v1.0** (Q1 2026): Streaming, health checks, persistent jobs  
**v1.5** (Q2 2026): Advanced features (sessions, sampling, aliases)  
**v2.0** (Q3 2026): Production hardening (auth, monitoring, scaling)  
**v2.5** (Q4 2026): Ecosystem integration (OpenAI API, LangChain)  
**v3.0** (2027): Performance optimizations (batching, speculative decoding)
