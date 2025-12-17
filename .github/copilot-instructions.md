# AI Inference Server - Copilot Instructions

## Project Overview

Single-file FastAPI server (`app.py`) serving Hugging Face LLMs with GPU acceleration. Acts as an Ollama-compatible drop-in with explicit download control, automatic quantization, and intelligent caching. Target: RTX 4090 with CUDA 12.6, tested up to 70GB models (Qwen3-Omni-30B) with 4-bit quantization.

## Architecture Principles

**No auto-downloads during inference**: Models MUST be explicitly pulled via `/pull` endpoint. The `load_model()` function enforces `local_files_only=True` and raises errors if models aren't cached. This is intentional—never bypass it.

**Backend fallback chain**: vLLM (preferred) → transformers with quantization → transformers pipeline with `trust_remote_code=True` for MoE/custom models. Each backend attempt wrapped in try-except with `_record_failed_load()` on failures.

**Memory management strategy**:
- Module-level: `os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")` prevents CUDA fragmentation
- Pre-load: Aggressive `torch.cuda.empty_cache()` + `synchronize()` + `gc.collect()` in `load_model()`
- Post-evict: Same cleanup in `_evict_lru_if_needed()` when LRU cache exceeds `MAX_CACHE_MODELS`
- Cooldown: Failed loads trigger 300s (configurable) backoff via `failed_loads` dict to prevent retry storms

## Critical Code Patterns

### 1. Background Job System (Non-blocking `/pull`)

**Pattern**: `/pull` endpoint returns immediately with `job_id`, spawns async task `_background_pull()` that:
1. Acquires semaphore slot (max concurrent downloads configurable via `MAX_CONCURRENT_PULLS`)
2. Runs blocking `snapshot_download()` in thread executor
3. Probes model size via `model_info()` in thread
4. Optionally calls `load_model()` if `init=true` (also in thread)
5. Updates `jobs[job_id]` with status/error/metadata

**Key invariant**: Job metadata (`local_path`, `size_bytes`, `preferred_quantized`) written to `model_meta[model_name]` so `load_model()` can enforce local-only loading.

```python
# Jobs polled via GET /jobs/{job_id}
# Status: "running" | "succeeded" | "failed"
# Always check jobs[job_id].get("error") for exceptions
```

### 2. Quantization Decision Logic

**Auto-quantization threshold**: Models >14GB (configurable `MODEL_Q4_THRESHOLD_BYTES`) auto-use 4-bit when `bitsandbytes_available`.

**Critical fix for MoE models**: When `AutoModelForCausalLM.from_pretrained()` raises `ValueError: Unrecognized configuration class`, fallback path MUST pass `quantization_config` via `model_kwargs` to `pipeline()`:

```python
# Example from app.py lines 342-355
if should_q4:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model_kwargs["quantization_config"] = bnb_config
    model_kwargs["device_map"] = "auto"
    quantized = True

pipe = pipeline("text-generation", model=local_path or model_name, 
                model_kwargs=model_kwargs, device=device if not should_q4 else None)
```

**Rationale**: Original implementation only applied quantization in `from_pretrained()` path, causing 70GB models to load full-precision in fallback, leading to OOM.

### 3. Pipeline Output Normalization

**Problem**: Transformers pipelines return inconsistent shapes: `[{'generated_text': str}]`, `[{'generated_text': [dict]}]`, or nested message arrays.

**Solution**: `_extract_text_from_pipeline_result()` recursively searches for `content`/`generated_text`/`text` keys, prefers last assistant message, stringifies fallback.

**Usage**: Always wrap pipeline results:
```python
content = _extract_text_from_pipeline_result(result[0])
message = {"role": "assistant", "content": content}
```

### 4. LRU Cache with OrderedDict

**Implementation**: `model_cache` is `OrderedDict[str, Any]`. Cache hit re-inserts to make most-recent:
```python
current_model = model_cache.pop(model_name)
model_cache[model_name] = current_model  # Re-insert as MRU
```

**Eviction**: `_evict_lru_if_needed()` pops from front (`last=False`) until size ≤ `MAX_CACHE_MODELS`.

## Development Workflows

### Testing Changes

**Quick validation**: Use `scripts/dai-cazzo.ps1` (polls `/diag`, pulls gpt2, runs chat test)
```powershell
# Terminal 1
python -m uvicorn app:app --host 0.0.0.0 --port 8005

# Terminal 2
.\scripts\dai-cazzo.ps1
```

**Manual endpoint testing**:
```powershell
# Check CUDA availability
Invoke-RestMethod http://localhost:8005/diag

# Non-blocking pull
$body = @{model='gpt2'; init=$true} | ConvertTo-Json
$job = Invoke-RestMethod -Method POST -Uri http://localhost:8005/pull -ContentType 'application/json' -Body $body

# Poll job status
Invoke-RestMethod "http://localhost:8005/jobs/$($job.job_id)"
```

### CUDA Setup (Windows)

If `/diag` shows `cuda_available: false` despite GPU:
```powershell
.\setup_cuda_pytorch.ps1  # Installs torch+cu126, bitsandbytes
```

**Verify installation**:
```python
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Debugging OOM Errors

1. Check `nvidia-smi` for fragmentation (large "reserved" vs "allocated")
2. Reduce `MAX_CACHE_MODELS=1` (environment variable)
3. Force q4: `/pull` with `"quantize":"q4"`
4. Restart server to clear GPU memory completely

## Project-Specific Conventions

### Error Handling

**DO**: Use `_record_failed_load()` on model load failures, check `_in_cooldown()` before retry attempts.

**DON'T**: Catch and suppress `HTTPException`—let FastAPI serialize them properly.

### Logging

**Pattern**: Use module-level `log = logging.getLogger("ai-server")` (not root logger).

**Key events to log**:
- `load_model()`: Torch diagnostics, backend selection, quantization decision
- Background jobs: Start/finish with job_id, errors with full traceback
- LRU evictions: Which model evicted

### Global State Management

**Thread safety**: `jobs`, `model_cache`, `model_meta` are accessed from async tasks and main thread. Current implementation is single-process, relies on GIL. If adding multiprocessing, wrap in locks.

**Startup environment**: `PYTORCH_ALLOC_CONF` MUST be set before `import torch` (currently line 13).

## Integration Points

### Hugging Face Hub

**Authentication**: Respects `HF_TOKEN` env var for private models (via `huggingface_hub` library).

**Cache location**: `~/.cache/huggingface/hub` (Linux) or `%USERPROFILE%\.cache\huggingface\hub` (Windows). Set `HF_HOME` to override.

**Model metadata**: `snapshot_download()` returns path, `model_info()` provides `siblings` array for size estimation.

### Ollama Compatibility

**Response format**: `/chat` returns Ollama-like JSON with `message`, `done`, `total_duration`, etc.

**Breaking differences**:
- No `/api/generate` endpoint (use `/chat`)
- No streaming support (yet)
- No GGUF models (native HF only)

## Files to Reference

- **Architecture decisions**: `.specs/plans/background-pull-plan.md` (non-blocking job design)
- **CUDA setup**: `setup_cuda_pytorch.ps1` (PyInstaller pattern for temp .py files)
- **API examples**: `README.md` lines 100-280 (comprehensive endpoint docs)
- **Test workflow**: `scripts/dai-cazzo.ps1` (PowerShell integration test)

## Common Pitfalls

1. **Modifying quantization logic**: Always test with both small (gpt2) and large (>14GB) models. Verify `model_meta[name]["quantized"]` matches expected behavior.

2. **Adding new endpoints**: If endpoint loads models, MUST check cooldown and use `load_model()` (don't bypass cache).

3. **Changing memory management**: Test with `nvidia-smi dmon` running to catch fragmentation regressions.

4. **Background job changes**: Remember job dict updates are in-memory only (cleared on restart). Status must be idempotent (can poll same job_id multiple times).

5. **Windows PowerShell escaping**: Heredocs fail—use temp files pattern from `setup_cuda_pytorch.ps1` if embedding Python code in scripts.
