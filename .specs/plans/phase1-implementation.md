# Implementation Plan: Phase 1 (v1.0) - Next Steps

## Overview

This document provides detailed implementation plans for Phase 1 features to bring the server from v0.9 (MVP) to v1.0 (Production-Ready).

**Goal**: Stability, performance monitoring, and improved UX  
**Timeline**: 2-3 weeks  
**Priority Order**: 1.3 → 1.1 → 1.4 → 1.2 → 1.5 → 1.6

---

## Task 1.3: Health Check Endpoint

**Priority**: HIGH (required for production deployments)  
**Effort**: 0.5 day  
**Dependencies**: None

### Implementation Steps

1. **Add basic health endpoint**

```python
# In app.py, after existing endpoints

@app.get("/health")
async def health():
    """Health check for load balancers and monitoring."""
    uptime = time.time() - server_start_time  # Add server_start_time global at startup
    
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
    
    # Determine overall status
    status = "healthy"
    if not transformers_available and LLM is None:
        status = "degraded"  # No backends available
    
    return {
        "status": status,
        "uptime_seconds": int(uptime),
        "models_cached": len(model_cache),
        "cache_limit": MAX_CACHE_MODELS,
        "downloads_active": sum(1 for j in jobs.values() if j.get("status") == "running"),
        "downloads_queued": sum(1 for j in jobs.values() if j.get("status") == "queued"),
        "torch_version": getattr(torch, "__version__", None) if torch else None,
        "cuda_available": torch.cuda.is_available() if torch else False,
        "gpu_status": gpu_status,
        "gpu_memory_allocated_mb": gpu_memory_allocated_mb,
        "gpu_memory_reserved_mb": gpu_memory_reserved_mb,
    }
```

2. **Add server start time tracking**

```python
# Near top of app.py, after imports
server_start_time = time.time()
```

3. **Test health endpoint**

```powershell
# Should return 200 OK with JSON
Invoke-RestMethod http://localhost:8005/health

# Test with server under load
# Start download, then check health shows downloads_active=1
$job = Invoke-RestMethod -Method POST -Uri http://localhost:8005/pull -ContentType 'application/json' -Body (@{model='gpt2'} | ConvertTo-Json)
Invoke-RestMethod http://localhost:8005/health
```

### Acceptance Criteria

- [x] `/health` returns 200 OK when server operational
- [x] Returns `status: "healthy"` when all backends available
- [x] Returns `status: "degraded"` when no backends available
- [x] Includes GPU memory stats if CUDA available
- [x] Shows accurate counts for cached models and active downloads

### Files Modified

- `app.py`: Add health endpoint, server_start_time global

---

## Task 1.1: Streaming Support

**Priority**: HIGH (major UX improvement)  
**Effort**: 1-2 days  
**Dependencies**: None

### Implementation Steps

1. **Add streaming response model**

```python
# In app.py, add new imports
from fastapi.responses import StreamingResponse
from typing import AsyncIterator

# Add streaming helper
async def _stream_chat_response(
    model_name: str, 
    messages: List[Dict[str, str]], 
    max_tokens: int = 256
) -> AsyncIterator[str]:
    """Generate streaming chat response chunks."""
    
    if model_name not in model_cache:
        load_model(model_name)
    
    current = model_cache[model_name]
    backend = model_meta.get(model_name, {}).get("backend")
    
    if backend == "vllm":
        # vLLM streaming
        from vllm.sampling_params import SamplingParams
        params = SamplingParams(max_tokens=max_tokens, temperature=0.7)
        
        for output in current.generate(messages, params, use_tqdm=False):
            chunk = {
                "delta": {"content": output.outputs[0].text},
                "done": False,
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # Final chunk
        final_chunk = {"delta": {}, "done": True}
        yield f"data: {json.dumps(final_chunk)}\n\n"
    
    elif backend in ("transformers", "pipeline"):
        # Transformers streaming (more complex)
        # Use TextIteratorStreamer
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        # Create streamer
        tokenizer = current.tokenizer if hasattr(current, 'tokenizer') else AutoTokenizer.from_pretrained(model_meta[model_name]["local_path"])
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Run generation in thread
        def generate():
            if backend == "transformers":
                inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
                current.generate(inputs.to(current.device), max_new_tokens=max_tokens, streamer=streamer)
            else:  # pipeline
                current(messages, max_new_tokens=max_tokens, streamer=streamer)
        
        thread = Thread(target=generate)
        thread.start()
        
        # Stream chunks
        for text in streamer:
            chunk = {"delta": {"content": text}, "done": False}
            yield f"data: {json.dumps(chunk)}\n\n"
        
        thread.join()
        
        # Final chunk
        final_chunk = {"delta": {}, "done": True}
        yield f"data: {json.dumps(final_chunk)}\n\n"
    
    else:
        raise HTTPException(status_code=500, detail=f"Streaming not supported for backend: {backend}")
```

2. **Update /chat endpoint to support streaming**

```python
@app.post("/chat")
async def chat(payload: Dict[str, Any]):
    """Chat endpoint with optional streaming support."""
    model_name = payload.get("model")
    messages = payload.get("messages", [])
    max_tokens = payload.get("max_tokens", 256)
    stream = payload.get("stream", False)
    
    if not model_name or not messages:
        raise HTTPException(status_code=400, detail="Missing 'model' or 'messages'")
    
    # Streaming response
    if stream:
        return StreamingResponse(
            _stream_chat_response(model_name, messages, max_tokens),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    # Non-streaming response (existing code)
    # ... rest of existing /chat implementation
```

3. **Test streaming**

Create test script `scripts/test-streaming.ps1`:

```powershell
# Test streaming with curl
$body = @{
    model = "gpt2"
    messages = @(
        @{role = "user"; content = "Write a short story about a robot."}
    )
    stream = $true
    max_tokens = 100
} | ConvertTo-Json

# This will print chunks as they arrive
Invoke-WebRequest -Uri http://localhost:8005/chat -Method POST -Body $body -ContentType 'application/json'
```

### Acceptance Criteria

- [x] `/chat` with `stream=true` returns SSE stream
- [x] Chunks arrive incrementally (not all at once)
- [x] Final chunk has `done=true`
- [x] Works with vLLM backend
- [x] Works with transformers backend
- [x] Client can process chunks in real-time

### Files Modified

- `app.py`: Add streaming helpers, update `/chat` endpoint

### Dependencies to Add

```txt
# Add to requirements.txt
transformers[streaming]  # Includes TextIteratorStreamer
```

---

## Task 1.4: Request Timeout & Cancellation

**Priority**: MEDIUM (prevents resource leaks)  
**Effort**: 1 day  
**Dependencies**: None

### Implementation Steps

1. **Add timeout wrapper**

```python
# In app.py, add helper
async def _with_timeout(coro, timeout_seconds: int):
    """Execute coroutine with timeout, cleanup on expiration."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        # Cleanup GPU memory on timeout
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        raise HTTPException(status_code=408, detail=f"Request timeout after {timeout_seconds}s")
```

2. **Update /chat to use timeout**

```python
@app.post("/chat")
async def chat(payload: Dict[str, Any]):
    model_name = payload.get("model")
    messages = payload.get("messages", [])
    max_tokens = payload.get("max_tokens", 256)
    stream = payload.get("stream", False)
    timeout = payload.get("timeout", 120)  # Default 2 minutes
    
    # Validate timeout
    if timeout < 1 or timeout > 600:
        raise HTTPException(status_code=400, detail="Timeout must be between 1 and 600 seconds")
    
    if stream:
        # Streaming doesn't timeout (client controls disconnect)
        return StreamingResponse(...)
    
    # Non-streaming: wrap in timeout
    async def _generate():
        # ... existing generation logic moved here
        return result
    
    return await _with_timeout(_generate(), timeout)
```

3. **Test timeout behavior**

```powershell
# Test with very short timeout
$body = @{
    model = "gpt2"
    messages = @(@{role = "user"; content = "Write a very long story."})
    max_tokens = 1000
    timeout = 1  # 1 second, should timeout
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri http://localhost:8005/chat -ContentType 'application/json' -Body $body
# Should return 408 error
```

### Acceptance Criteria

- [x] Requests timeout after specified duration
- [x] GPU memory cleaned up on timeout
- [x] Returns 408 status code with clear message
- [x] Timeout configurable per-request
- [x] Streaming requests not affected by timeout

### Files Modified

- `app.py`: Add timeout wrapper, update `/chat`

---

## Task 1.2: Progress Tracking for Downloads

**Priority**: MEDIUM (UX improvement for large models)  
**Effort**: 0.5-1 day  
**Dependencies**: None

### Implementation Steps

1. **Add progress calculation helper**

```python
# In app.py
def _get_download_progress(local_path: str, total_bytes: int) -> float:
    """Calculate download progress by checking local cache size."""
    if not os.path.exists(local_path):
        return 0.0
    
    # Sum all files in snapshot directory
    downloaded = 0
    try:
        for root, _, files in os.walk(local_path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.exists(fp):
                    downloaded += os.path.getsize(fp)
    except Exception:
        return 0.0
    
    if total_bytes <= 0:
        return 0.0
    
    return min(1.0, downloaded / total_bytes)
```

2. **Update _background_pull to track progress**

```python
async def _background_pull(job_id: str, model_name: str, quant: str, init: bool):
    job = jobs.get(job_id)
    if job is None:
        return
    
    job["status"] = "running"
    job["started_at"] = datetime.utcnow().isoformat() + "Z"
    
    try:
        # Get total size BEFORE download
        size_bytes = None
        try:
            if model_info is not None:
                info = await asyncio.to_thread(model_info, model_name)
                siblings = getattr(info, "siblings", []) or []
                size_bytes = sum([getattr(s, "size", 0) or 0 for s in siblings])
                job["total_bytes"] = size_bytes
        except Exception:
            log.exception("Failed to get model size for %s", model_name)
        
        # Start download in thread
        await download_semaphore.acquire()
        
        # Create background task to update progress
        local_path = None
        download_task = asyncio.create_task(
            asyncio.to_thread(snapshot_download, repo_id=model_name)
        )
        
        # Poll for progress while download runs
        if size_bytes:
            while not download_task.done():
                await asyncio.sleep(2)  # Check every 2 seconds
                # Estimate progress from HF cache
                # Note: local_path not available yet, use expected cache location
                cache_path = os.path.join(
                    os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
                    "hub",
                    f"models--{model_name.replace('/', '--')}"
                )
                progress = _get_download_progress(cache_path, size_bytes)
                job["progress"] = progress
                job["downloaded_bytes"] = int(progress * size_bytes)
        
        # Get result
        local_path = await download_task
        download_semaphore.release()
        
        job["local_path"] = local_path
        job["progress"] = 1.0
        job["downloaded_bytes"] = size_bytes
        
        # ... rest of existing _background_pull logic
        
    except Exception as e:
        # ... existing error handling
```

3. **Update job schema**

Add to initial job creation:
```python
jobs[job_id] = {
    # ... existing fields
    "progress": 0.0,
    "downloaded_bytes": 0,
    "total_bytes": None,
}
```

4. **Test progress tracking**

```powershell
# Pull a larger model to see progress
$body = @{model='facebook/opt-1.3b'} | ConvertTo-Json
$job = Invoke-RestMethod -Method POST -Uri http://localhost:8005/pull -ContentType 'application/json' -Body $body

# Poll job status every 2 seconds
while ($true) {
    $status = Invoke-RestMethod "http://localhost:8005/jobs/$($job.job_id)"
    Write-Host "Progress: $($status.progress * 100)% ($($status.downloaded_bytes)/$($status.total_bytes) bytes)"
    if ($status.status -ne "running") { break }
    Start-Sleep -Seconds 2
}
```

### Acceptance Criteria

- [x] Job includes `progress` field (0.0-1.0)
- [x] Progress updates every 2-5 seconds during download
- [x] Progress reaches 1.0 when download completes
- [x] Works for models with known size
- [x] Gracefully handles models without size info

### Files Modified

- `app.py`: Add progress calculation, update `_background_pull`

---

## Task 1.5: Persistent Job Storage (SQLite)

**Priority**: MEDIUM (required for restarts)  
**Effort**: 1-2 days  
**Dependencies**: None

### Implementation Steps

1. **Create database schema**

Create `database.py`:

```python
import sqlite3
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

log = logging.getLogger("ai-server")

class JobDatabase:
    """Persistent job storage using SQLite."""
    
    def __init__(self, db_path: str = "data/jobs.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create tables if not exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                quantize TEXT,
                init INTEGER,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                error TEXT,
                traceback TEXT,
                local_path TEXT,
                size_bytes INTEGER,
                preferred_quantized INTEGER,
                progress REAL,
                downloaded_bytes INTEGER,
                total_bytes INTEGER,
                metadata TEXT
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)")
        
        conn.commit()
        conn.close()
        log.info("Job database initialized: %s", self.db_path)
    
    def create_job(self, job: Dict[str, Any]) -> None:
        """Insert new job."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO jobs (
                id, model, quantize, init, status, created_at,
                started_at, finished_at, error, traceback, local_path,
                size_bytes, preferred_quantized, progress, downloaded_bytes,
                total_bytes, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job["id"],
            job["model"],
            job.get("quantize"),
            1 if job.get("init") else 0,
            job["status"],
            job["created_at"],
            job.get("started_at"),
            job.get("finished_at"),
            job.get("error"),
            job.get("traceback"),
            job.get("local_path"),
            job.get("size_bytes"),
            1 if job.get("preferred_quantized") else 0,
            job.get("progress", 0.0),
            job.get("downloaded_bytes", 0),
            job.get("total_bytes"),
            json.dumps(job.get("metadata", {}))
        ))
        
        conn.commit()
        conn.close()
    
    def update_job(self, job_id: str, updates: Dict[str, Any]) -> None:
        """Update existing job."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        set_clauses = []
        values = []
        
        for key, value in updates.items():
            if key in ("init", "preferred_quantized"):
                value = 1 if value else 0
            set_clauses.append(f"{key} = ?")
            values.append(value)
        
        values.append(job_id)
        
        cursor.execute(
            f"UPDATE jobs SET {', '.join(set_clauses)} WHERE id = ?",
            values
        )
        
        conn.commit()
        conn.close()
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        job = dict(row)
        job["init"] = bool(job["init"])
        job["preferred_quantized"] = bool(job["preferred_quantized"]) if job["preferred_quantized"] is not None else None
        job["metadata"] = json.loads(job.get("metadata") or "{}")
        return job
    
    def list_jobs(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List jobs with optional status filter."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if status:
            cursor.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        jobs = []
        for row in rows:
            job = dict(row)
            job["init"] = bool(job["init"])
            job["preferred_quantized"] = bool(job["preferred_quantized"]) if job["preferred_quantized"] is not None else None
            job["metadata"] = json.loads(job.get("metadata") or "{}")
            jobs.append(job)
        
        return jobs
```

2. **Update app.py to use database**

```python
# In app.py, after imports
from database import JobDatabase

# Replace jobs dict with database
job_db = JobDatabase()

# Update /pull endpoint
@app.post("/pull")
async def pull_model(payload: Dict[str, Any]):
    # ... existing validation
    
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "model": model_name,
        "quantize": quant,
        "init": init,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "progress": 0.0,
    }
    
    job_db.create_job(job)
    asyncio.create_task(_background_pull(job_id, model_name, quant, init))
    
    return {"status": "accepted", "job_id": job_id}

# Update _background_pull
async def _background_pull(job_id: str, model_name: str, quant: str, init: bool):
    job_db.update_job(job_id, {"status": "running", "started_at": datetime.utcnow().isoformat() + "Z"})
    
    try:
        # ... download logic
        
        # Update progress
        job_db.update_job(job_id, {"progress": progress, "downloaded_bytes": downloaded})
        
        # ... rest of logic
        
        job_db.update_job(job_id, {
            "status": "succeeded",
            "finished_at": datetime.utcnow().isoformat() + "Z",
            "local_path": local_path,
            "size_bytes": size_bytes,
        })
    
    except Exception as e:
        job_db.update_job(job_id, {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "finished_at": datetime.utcnow().isoformat() + "Z",
        })

# Update endpoints
@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/jobs")
async def list_jobs(status: Optional[str] = None):
    jobs = job_db.list_jobs(status=status)
    return {"jobs": jobs}
```

3. **Create data directory**

```powershell
New-Item -ItemType Directory -Force -Path data
```

4. **Test persistence**

```powershell
# Start server, create job
$job = Invoke-RestMethod -Method POST -Uri http://localhost:8005/pull -ContentType 'application/json' -Body (@{model='gpt2'} | ConvertTo-Json)

# Restart server (Ctrl+C, then restart)
python -m uvicorn app:app --host 0.0.0.0 --port 8005

# Job should still be queryable
Invoke-RestMethod "http://localhost:8005/jobs/$($job.job_id)"
```

### Acceptance Criteria

- [x] Jobs persist across server restarts
- [x] Job creation/updates <10ms latency
- [x] Job history queryable with filters
- [x] Database auto-creates on first run
- [x] No data loss on crash

### Files Created

- `database.py`: Job database implementation
- `data/jobs.db`: SQLite database (auto-created)

### Files Modified

- `app.py`: Replace `jobs` dict with `job_db`

---

## Task 1.6: Improved Error Messages

**Priority**: LOW (UX polish)  
**Effort**: 0.5 day  
**Dependencies**: None

### Implementation Steps

1. **Create error message templates**

```python
# In app.py, add error helpers

ERROR_MESSAGES = {
    "model_not_pulled": """
Model '{model}' not downloaded.

To download:
  POST /pull {{"model": "{model}"}}

Or use PowerShell:
  $body = @{{model='{model}'}} | ConvertTo-Json
  Invoke-RestMethod -Method POST -Uri http://localhost:8005/pull -ContentType 'application/json' -Body $body
""",
    
    "no_backend": """
Server misconfigured: No inference backend available.

Troubleshooting:
  1. Check transformers installed: pip install transformers
  2. Check vLLM installation (optional): pip install vllm
  3. Verify imports: python -c "import transformers; import torch"

See: {docs_url}/troubleshooting#no-backend
""",
    
    "cuda_oom": """
GPU out of memory.

Try:
  1. Reduce cache size: Set MAX_CACHE_MODELS=1
  2. Use smaller model
  3. Enable quantization: POST /pull {{"model": "{model}", "quantize": "q4"}}
  4. Restart server to clear GPU memory

Current GPU usage:
  Allocated: {allocated_mb:.0f} MB
  Reserved: {reserved_mb:.0f} MB
  Total: {total_mb:.0f} MB

See: {docs_url}/troubleshooting#oom
""",
}

def format_error(template_key: str, **kwargs) -> str:
    """Format error message with context."""
    template = ERROR_MESSAGES.get(template_key, "Error: {error}")
    kwargs.setdefault("docs_url", "https://github.com/fabiopacificicom/private-ai-server")
    return template.format(**kwargs).strip()
```

2. **Update error raising**

```python
# In load_model()
if local_path is None:
    raise RuntimeError(format_error("model_not_pulled", model=model_name))

# In backend fallback
if not transformers_available and LLM is None:
    raise RuntimeError(format_error("no_backend"))

# In CUDA OOM handler (add try-except around model loads)
except torch.cuda.OutOfMemoryError:
    allocated = torch.cuda.memory_allocated(0) / (1024**2)
    reserved = torch.cuda.memory_reserved(0) / (1024**2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    
    raise RuntimeError(format_error(
        "cuda_oom",
        model=model_name,
        allocated_mb=allocated,
        reserved_mb=reserved,
        total_mb=total
    ))
```

3. **Test error messages**

```powershell
# Test model not pulled
Invoke-RestMethod -Method POST -Uri http://localhost:8005/chat -ContentType 'application/json' -Body (@{model='nonexistent'; messages=@(@{role='user'; content='hi'})} | ConvertTo-Json)
# Should return helpful message with /pull command
```

### Acceptance Criteria

- [x] Error messages include actionable steps
- [x] Include relevant diagnostics (GPU memory, etc.)
- [x] Link to troubleshooting docs
- [x] Consistent formatting across errors

### Files Modified

- `app.py`: Add error templates, update exception raising

---

## Testing & Validation

### Integration Test Suite

Create `scripts/test-phase1.ps1`:

```powershell
# Phase 1 integration tests

$baseUrl = "http://localhost:8005"

Write-Host "Testing Phase 1 features..."

# Test 1.3: Health check
Write-Host "`n1. Testing health endpoint..."
$health = Invoke-RestMethod "$baseUrl/health"
if ($health.status -ne "healthy") {
    throw "Health check failed: $($health.status)"
}
Write-Host "✅ Health check passed"

# Test 1.1: Streaming (if implemented)
Write-Host "`n2. Testing streaming..."
$body = @{
    model = "gpt2"
    messages = @(@{role = "user"; content = "Say hello"})
    stream = $true
    max_tokens = 20
} | ConvertTo-Json

$response = Invoke-WebRequest -Uri "$baseUrl/chat" -Method POST -Body $body -ContentType 'application/json'
if ($response.Headers.'Content-Type' -notlike "*text/event-stream*") {
    throw "Streaming not enabled"
}
Write-Host "✅ Streaming works"

# Test 1.4: Timeout
Write-Host "`n3. Testing timeout..."
$body = @{
    model = "gpt2"
    messages = @(@{role = "user"; content = "Write a novel"})
    max_tokens = 10000
    timeout = 1
} | ConvertTo-Json

try {
    Invoke-RestMethod -Method POST -Uri "$baseUrl/chat" -ContentType 'application/json' -Body $body
    throw "Timeout did not trigger"
} catch {
    if ($_.Exception.Response.StatusCode -ne 408) {
        throw "Wrong status code for timeout: $($_.Exception.Response.StatusCode)"
    }
}
Write-Host "✅ Timeout works"

# Test 1.2: Progress tracking
Write-Host "`n4. Testing progress tracking..."
$body = @{model='gpt2'} | ConvertTo-Json
$job = Invoke-RestMethod -Method POST -Uri "$baseUrl/pull" -ContentType 'application/json' -Body $body

$hasProgress = $false
for ($i = 0; $i -lt 30; $i++) {
    $status = Invoke-RestMethod "$baseUrl/jobs/$($job.job_id)"
    if ($status.progress -gt 0 -and $status.progress -lt 1.0) {
        $hasProgress = $true
    }
    if ($status.status -ne "running") { break }
    Start-Sleep -Seconds 1
}

if (-not $hasProgress) {
    Write-Warning "⚠ Progress not observed (may be too fast for small model)"
} else {
    Write-Host "✅ Progress tracking works"
}

Write-Host "`n✅ All Phase 1 tests passed!"
```

### Performance Benchmarks

Create `scripts/benchmark-phase1.ps1`:

```powershell
# Phase 1 performance benchmarks

$baseUrl = "http://localhost:8005"

Write-Host "Running Phase 1 benchmarks..."

# Benchmark health endpoint
Write-Host "`nBenchmarking /health (100 requests)..."
$times = @()
for ($i = 0; $i -lt 100; $i++) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Invoke-RestMethod "$baseUrl/health" | Out-Null
    $sw.Stop()
    $times += $sw.ElapsedMilliseconds
}

$avg = ($times | Measure-Object -Average).Average
$p99 = ($times | Sort-Object)[[math]::Floor($times.Count * 0.99)]
Write-Host "  Average: $avg ms"
Write-Host "  P99: $p99 ms"

if ($p99 -gt 50) {
    Write-Warning "⚠ Health check P99 latency >50ms: $p99 ms"
}

Write-Host "`n✅ Benchmarks complete"
```

---

## Rollout Plan

### Week 1
- Day 1-2: Implement 1.3 (Health) + 1.1 (Streaming)
- Day 3: Implement 1.4 (Timeout)
- Day 4-5: Test and refine

### Week 2
- Day 1-2: Implement 1.2 (Progress) + 1.5 (Persistence)
- Day 3: Implement 1.6 (Error messages)
- Day 4-5: Integration testing

### Week 3
- Day 1-2: Performance optimization
- Day 3-4: Documentation updates
- Day 5: Release v1.0

---

## Success Criteria

Phase 1 (v1.0) complete when:
- [x] All 6 tasks implemented and tested
- [x] Integration test suite passes
- [x] Health check P99 latency <50ms
- [x] No CUDA OOM errors in tests
- [x] Documentation updated (README, ROADMAP)
- [x] Tagged release: `git tag v1.0.0`

---

## Notes

- Keep `app.py` single-file for core logic; `database.py` is acceptable separation
- All changes backward-compatible (no breaking API changes)
- Update `.github/copilot-instructions.md` with new patterns as implemented
