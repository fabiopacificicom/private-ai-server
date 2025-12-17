---
applyTo: '**'
---

# AI Inference Server - Coding Guidelines

## Project Philosophy

This is a **single-file FastAPI server** (`app.py`) designed for simplicity, portability, and explicit control. The entire inference logic lives in one file to make it easy to understand, debug, and deploy. Avoid fragmenting logic across multiple modules unless absolutely necessary.

**Core principles**:
1. **Explicit over implicit**: Models must be explicitly pulled before use (no auto-downloads)
2. **Fail fast**: Raise descriptive errors immediately rather than silently degrading
3. **Log everything important**: Model loads, failures, evictions, and background jobs
4. **Memory consciousness**: GPU memory is precious—cleanup aggressively and track usage

---

## Code Style and Structure

### 1. File Organization (app.py)

Follow this top-to-bottom structure:

```python
# 1. Module-level environment setup (BEFORE imports)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# 2. Standard library imports
import os, sys, logging, ...

# 3. Third-party imports (grouped by purpose)
# - FastAPI
# - PyTorch/transformers
# - HuggingFace Hub
# - Optional backends (vLLM)

# 4. Global configuration (environment variables, constants)
MAX_CACHE_MODELS = int(os.getenv("MAX_CACHE_MODELS", "2"))

# 5. Global state (dicts, caches, registries)
model_cache: OrderedDict[str, Any] = OrderedDict()
jobs: Dict[str, Dict[str, Any]] = {}

# 6. FastAPI app initialization
app = FastAPI()

# 7. Helper functions (prefixed with underscore)
def _evict_lru_if_needed(): ...
def _record_failed_load(): ...

# 8. Core model loading logic
def load_model(model_name: str): ...

# 9. API endpoints (grouped by functionality)
@app.post("/chat")
@app.post("/pull")
@app.get("/models")
...
```

**Rules**:
- Keep related functions together (e.g., all cache helpers near each other)
- Private/internal functions start with `_`
- Put environment variable setup **before** imports when it affects library behavior
- Document non-obvious ordering requirements with comments

### 2. Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Global variables | SCREAMING_SNAKE_CASE | `MAX_CACHE_MODELS`, `COOLDOWN_SECONDS` |
| Global state dicts | snake_case | `model_cache`, `failed_loads`, `jobs` |
| Functions (public) | snake_case | `load_model()`, `list_models()` |
| Functions (private) | _snake_case | `_evict_lru_if_needed()`, `_background_pull()` |
| Endpoint paths | lowercase with hyphens | `/models`, `/jobs/{job_id}` |
| Model backends | lowercase strings | `"vllm"`, `"transformers"`, `"pipeline"` |
| Job statuses | lowercase strings | `"queued"`, `"running"`, `"succeeded"`, `"failed"` |

**DO NOT** use:
- Camel case for functions/variables (PEP 8 violation)
- Generic names like `data`, `result`, `temp` without context
- Abbreviations that aren't obvious (`mdl` → use `model`)

### 3. Type Hints

Use type hints for function signatures (aids debugging and IDE autocomplete):

```python
def load_model(model_name: str) -> None:
    """Load model into cache. Raises RuntimeError if not available locally."""
    ...

def _cache_model(name: str, backend: str, instance: Any, extra_meta: Dict[str, Any] = {}) -> None:
    """Add model to cache and update metadata."""
    ...
```

**Required hints**:
- Function parameters
- Return types (`None`, `Dict`, `str`, etc.)
- Complex structures: `Dict[str, Any]`, `OrderedDict[str, Any]`

**Optional hints** (use sparingly):
- Local variables (only if type is ambiguous)

### 4. Docstrings

Use concise docstrings for:
- All public functions (non-underscore)
- Complex private functions
- API endpoints

**Format**:
```python
def load_model(model_name: str) -> None:
    """Load a model into the cache using available backends.
    
    Tries vLLM first, falls back to transformers with optional quantization.
    Raises RuntimeError if model not available locally or all backends fail.
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'gpt2', 'Qwen/Qwen2-72B')
    """
```

**Keep docstrings short** (1-3 sentences). For complex logic, add inline comments instead.

---

## Memory Management Patterns

### GPU Memory Cleanup (Critical)

**Always** perform aggressive cleanup before loading models:

```python
# Before every model load
if torch is not None and torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()
```

**Where to apply**:
1. Start of `load_model()` (before any backend attempt)
2. Inside `_evict_lru_if_needed()` after removing models from cache
3. After failed model loads (in exception handlers)

**Rationale**: Prevents CUDA memory fragmentation which causes OOM errors even when nominal free memory exists.

### LRU Cache Pattern

Use `OrderedDict` for LRU behavior:

```python
# Cache hit: move to end (most recent)
if model_name in model_cache:
    current_model = model_cache.pop(model_name)
    model_cache[model_name] = current_model  # Re-insert at end
    return

# Eviction: remove from front (least recent)
def _evict_lru_if_needed():
    while len(model_cache) > MAX_CACHE_MODELS:
        old_name, _ = model_cache.popitem(last=False)  # last=False = FIFO
        # GPU cleanup here
```

**DO NOT** use `dict` for cache (insertion order guaranteed since Python 3.7, but `popitem(last=False)` is OrderedDict-specific).

---

## Error Handling Conventions

### 1. FastAPI HTTPException

Raise `HTTPException` for client errors:

```python
from fastapi import HTTPException

# Bad request
if not model_name:
    raise HTTPException(status_code=400, detail="Missing 'model' in payload")

# Model not found
if model_name not in model_meta:
    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

# Server error
if not transformers_available:
    raise HTTPException(status_code=500, detail="No inference backend available")
```

**DO NOT** catch and suppress `HTTPException`—FastAPI handles serialization.

### 2. RuntimeError for Internal Failures

Use `RuntimeError` for internal logic failures (these become HTTP 500 automatically):

```python
# Model not pulled
if local_path is None:
    raise RuntimeError(f"Model '{model_name}' not available locally. Use /pull to download.")

# Backend failures
raise RuntimeError(f"No inference backend available: {details}")
```

### 3. Failed Load Tracking

Always record failed loads to prevent retry storms:

```python
def _record_failed_load(model_name: str) -> None:
    failed_loads[model_name] = time.time()

def _in_cooldown(model_name: str) -> bool:
    if model_name not in failed_loads:
        return False
    elapsed = time.time() - failed_loads[model_name]
    return elapsed < COOLDOWN_SECONDS
```

**Check cooldown** at start of `load_model()`:

```python
if _in_cooldown(model_name):
    raise RuntimeError(f"Model {model_name} failed recently, in cooldown ({COOLDOWN_SECONDS}s)")
```

---

## Logging Guidelines

Use module-level logger (NOT root logger):

```python
log = logging.getLogger("ai-server")
```

### What to Log

| Event | Level | Example |
|-------|-------|---------|
| Model load start | INFO | `log.info("Loading model with vLLM: %s", model_name)` |
| Model load success | INFO | `log.info("✅ Model loaded: %s (backend=%s)", name, backend)` |
| Model load failure | EXCEPTION | `log.exception("vLLM failed to load %s", model_name)` |
| LRU eviction | INFO | `log.info("Evicted %s from cache (LRU)", old_name)` |
| Background job start | INFO | `log.info("Background pull start: %s (job=%s)", model, job_id)` |
| Background job finish | INFO | `log.info("Background pull succeeded: %s (job=%s)", model, job_id)` |
| Configuration | INFO | `log.info("Torch diagnostics: version=%s, cuda=%s", version, cuda)` |

**Format strings** with `%s` (not f-strings) for lazy evaluation:
```python
# Good
log.info("Model loaded: %s", model_name)

# Bad (f-string evaluated even if log level filters it)
log.info(f"Model loaded: {model_name}")
```

**Use `.exception()`** in exception handlers (includes traceback):
```python
except Exception as e:
    log.exception("Failed to load model %s", model_name)
    raise RuntimeError(f"Load failed: {e}")
```

---

## Backend Selection Pattern

Follow this strict fallback chain:

1. **vLLM** (preferred for speed, tensor parallelism)
2. **Transformers with quantization** (for large models with GPU)
3. **Transformers pipeline with trust_remote_code** (for MoE/custom models)

**Implementation pattern**:

```python
# 1. Try vLLM
if LLM is not None:
    try:
        mdl = LLM(model=local_path, ...)
        _cache_model(model_name, "vllm", mdl, ...)
        return
    except Exception:
        log.exception("vLLM failed, trying transformers")
        _record_failed_load(model_name)

# 2. Check if transformers available
if not transformers_available:
    raise RuntimeError("No backend available")

# 3. Try transformers with quantization
try:
    if should_quantize:
        bnb_config = BitsAndBytesConfig(...)
        model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)
        _cache_model(model_name, "transformers", model, {"quantized": True})
        return
except ValueError as e:
    # MoE models raise "Unrecognized configuration class"
    if "Unrecognized configuration" in str(e):
        # Fall through to pipeline
        pass
    else:
        raise

# 4. Final fallback: pipeline with trust_remote_code
model_kwargs = {}
if should_quantize:
    # CRITICAL: Pass quantization to pipeline via model_kwargs
    model_kwargs["quantization_config"] = bnb_config
    model_kwargs["device_map"] = "auto"

pipe = pipeline("text-generation", model=local_path, model_kwargs=model_kwargs, ...)
_cache_model(model_name, "pipeline", pipe, ...)
```

**Critical rule for MoE models**: When using `pipeline()` fallback, pass `quantization_config` via `model_kwargs` (NOT as a direct parameter). This ensures large MoE models load quantized instead of full-precision.

---

## Quantization Decision Logic

### Threshold-Based Auto-Quantization

```python
# Get threshold from env (default 14GB)
q4_threshold = int(os.getenv("MODEL_Q4_THRESHOLD_BYTES", str(14 * 10**9)))

# Check user preference (set during /pull)
preferred_quant = model_meta.get(model_name, {}).get("preferred_quantized")

if preferred_quant is not None:
    should_q4 = bool(preferred_quant)
else:
    # Auto-decide based on size
    should_q4 = (
        bitsandbytes_available 
        and torch.cuda.is_available() 
        and size_bytes is not None 
        and size_bytes >= q4_threshold
    )
```

### Quantization Configuration

Use consistent `BitsAndBytesConfig`:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # or torch.bfloat16 if available
)
```

**DO NOT** change quantization settings unless you have a specific reason (e.g., testing 8-bit).

---

## Background Job Pattern

### Job Structure

Every job in `jobs` dict has this structure:

```python
jobs[job_id] = {
    "id": job_id,
    "model": model_name,
    "quantize": quant,
    "init": init,
    "status": "queued",  # → "running" → "succeeded" | "failed"
    "created_at": datetime.utcnow().isoformat() + "Z",
    "started_at": None,
    "finished_at": None,
    "error": None,
    "traceback": None,
    "local_path": None,
    "size_bytes": None,
    "preferred_quantized": None,
}
```

### Background Task Pattern

```python
async def _background_pull(job_id: str, model_name: str, quant: str, init: bool):
    job = jobs.get(job_id)
    if job is None:
        return
    
    job["status"] = "running"
    job["started_at"] = datetime.utcnow().isoformat() + "Z"
    
    try:
        # Acquire semaphore (limit concurrency)
        await download_semaphore.acquire()
        try:
            # Blocking IO in thread
            local_path = await asyncio.to_thread(snapshot_download, repo_id=model_name)
        finally:
            download_semaphore.release()
        
        # Update job and metadata
        job["local_path"] = local_path
        model_meta[model_name] = {..., "local_path": local_path}
        
        # Optional: initialize
        if init:
            await asyncio.to_thread(load_model, model_name)
        
        job["status"] = "succeeded"
        job["finished_at"] = datetime.utcnow().isoformat() + "Z"
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()
        job["finished_at"] = datetime.utcnow().isoformat() + "Z"
        log.exception("Background pull failed: %s", model_name)
```

**Key rules**:
1. Always use `await asyncio.to_thread()` for blocking calls (downloads, model loads)
2. Always update job status on completion (succeeded/failed)
3. Always use semaphore to limit concurrency
4. Always log exceptions with traceback

---

## API Response Formats

### Ollama-Compatible Chat Response

```python
{
    "model": "gpt2",
    "created_at": "2025-12-17T10:30:00Z",
    "message": {"role": "assistant", "content": "generated text"},
    "done": True,
    "done_reason": "stop",
    "total_duration": 1500000000,  # nanoseconds
    "load_duration": 500000000,
    "prompt_eval_count": 10,
    "prompt_eval_duration": None,
    "eval_count": 50,
    "eval_duration": 1000000000,
}
```

### Job Status Response

```python
{
    "id": "uuid-here",
    "model": "gpt2",
    "status": "succeeded",  # queued | running | succeeded | failed
    "created_at": "2025-12-17T10:30:00Z",
    "started_at": "2025-12-17T10:30:01Z",
    "finished_at": "2025-12-17T10:30:15Z",
    "error": None,
    "local_path": "/path/to/model",
    "size_bytes": 548000000,
    "preferred_quantized": False,
}
```

**Always use ISO 8601 timestamps** with Z suffix: `datetime.utcnow().isoformat() + "Z"`

---

## Testing Practices

### Manual Testing Workflow

Use `scripts/dai-cazzo.ps1` for end-to-end testing:

```powershell
# Terminal 1: Start server
python -m uvicorn app:app --host 0.0.0.0 --port 8005 --reload

# Terminal 2: Run test script
.\scripts\dai-cazzo.ps1
```

### Validation Checklist

Before committing changes, verify:
1. `/diag` returns CUDA info correctly
2. `/pull` with small model (gpt2) succeeds
3. `/jobs/{job_id}` shows `succeeded` status
4. `/chat` generates responses
5. `/models` lists cached models
6. No CUDA OOM errors in logs

### Testing New Features

When adding endpoints or modifying logic:

```powershell
# Test endpoint manually
$body = @{model='gpt2'; init=$true} | ConvertTo-Json
Invoke-RestMethod -Method POST -Uri http://localhost:8005/pull -ContentType 'application/json' -Body $body

# Check logs for errors
Get-Content app.log -Tail 50
```

---

## Common Pitfalls and Solutions

### 1. Quantization Not Applied in Pipeline Fallback

**Problem**: MoE models load full-precision in pipeline, causing OOM.

**Solution**: Pass `quantization_config` via `model_kwargs`:

```python
model_kwargs = {}
if should_q4:
    model_kwargs["quantization_config"] = bnb_config
    model_kwargs["device_map"] = "auto"

pipe = pipeline("text-generation", model=local_path, model_kwargs=model_kwargs, ...)
```

### 2. CUDA Fragmentation

**Problem**: OOM despite free memory showing in `nvidia-smi`.

**Solution**: Set `PYTORCH_ALLOC_CONF` **before** importing torch:

```python
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch
```

### 3. Job Dict Updates Lost on Restart

**Problem**: In-memory `jobs` dict cleared when server restarts.

**Solution**: Document this limitation. For production, recommend persistent storage (Redis/SQLite).

### 4. Windows PowerShell Heredoc Failures

**Problem**: PowerShell heredocs (`@" ... "@`) fail with syntax errors.

**Solution**: Use temp file pattern:

```powershell
$tempScript = [System.IO.Path]::GetTempFileName() + ".py"
@"
import torch
print(torch.cuda.is_available())
"@ | Out-File -FilePath $tempScript -Encoding utf8
python $tempScript
Remove-Item $tempScript
```

### 5. Pipeline Output Inconsistency

**Problem**: `pipeline()` returns different shapes: `[{'generated_text': str}]`, `[{'generated_text': [dict]}]`.

**Solution**: Use `_extract_text_from_pipeline_result()`:

```python
result = pipe(messages, max_new_tokens=100, ...)
content = _extract_text_from_pipeline_result(result[0])
```

---

## Environment Variables

Document all configurable settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CACHE_MODELS` | `2` | Maximum models kept in memory |
| `MODEL_LOAD_COOLDOWN` | `300` | Seconds to wait after failed load |
| `MODEL_Q4_THRESHOLD_BYTES` | `14000000000` | Size threshold for auto-quantization (14GB) |
| `MAX_CONCURRENT_PULLS` | `2` | Max simultaneous downloads |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | Prevent CUDA fragmentation |
| `HF_TOKEN` | None | HuggingFace API token for private models |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache directory |

**Always provide defaults** with `os.getenv("VAR", "default")`.

---

## Git Commit Conventions

Use conventional commits for clear history:

- `feat: add streaming support to /chat endpoint`
- `fix: apply quantization in pipeline fallback for MoE models`
- `docs: update README with job polling examples`
- `refactor: extract model loading logic into separate function`
- `perf: reduce GPU memory fragmentation with aggressive cleanup`
- `test: add integration test for background pull jobs`

**Keep commits atomic**: one logical change per commit.

---

## When to Modify This File

Update `coder-guidelines.instructions.md` when:
1. Adding new coding patterns (e.g., streaming responses)
2. Changing error handling conventions
3. Adding new environment variables
4. Discovering new pitfalls/workarounds
5. Updating API response formats

**Keep guidelines synchronized** with `.github/copilot-instructions.md` (AI onboarding doc).