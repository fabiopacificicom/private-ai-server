# Database Schema Documentation

## Overview

The AI Inference Server uses SQLite for persistent job storage. The database is automatically created on first run and stores all pull job metadata across server restarts.

**Database file**: `jobs.db` (git-ignored, local only)  
**Schema version**: v1.0  
**Module**: `database.py` (JobDatabase class)

---

## Tables

### jobs

Stores metadata for all model download/pull jobs initiated via the `/pull` endpoint.

#### Schema

```sql
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    quantize TEXT,
    init BOOLEAN,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    traceback TEXT,
    local_path TEXT,
    size_bytes INTEGER,
    downloaded_bytes INTEGER DEFAULT 0,
    progress REAL,
    total_bytes INTEGER,
    preferred_quantized BOOLEAN
)
```

#### Column Definitions

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | TEXT | NO | - | UUID primary key, generated at job creation |
| `model` | TEXT | NO | - | HuggingFace model identifier (e.g., "gpt2", "Qwen/Qwen3-0.6B") |
| `quantize` | TEXT | YES | NULL | Quantization preference: "auto", "q4", "fp16", "no" |
| `init` | BOOLEAN | YES | NULL | Whether to load model into memory after download |
| `status` | TEXT | NO | 'queued' | Job status: "queued", "running", "succeeded", "failed" |
| `created_at` | TEXT | NO | - | ISO 8601 timestamp with 'Z' suffix (UTC) |
| `started_at` | TEXT | YES | NULL | ISO 8601 timestamp when job started executing |
| `finished_at` | TEXT | YES | NULL | ISO 8601 timestamp when job completed (success or failure) |
| `error` | TEXT | YES | NULL | Error message if status="failed" |
| `traceback` | TEXT | YES | NULL | Full Python traceback for debugging if status="failed" |
| `local_path` | TEXT | YES | NULL | Absolute path to downloaded model snapshot |
| `size_bytes` | INTEGER | YES | NULL | Total model size in bytes (from HF Hub metadata) |
| `downloaded_bytes` | INTEGER | NO | 0 | Bytes downloaded so far (for progress tracking) |
| `progress` | REAL | YES | NULL | Download progress as decimal (0.0 to 1.0) |
| `total_bytes` | INTEGER | YES | NULL | Expected total bytes to download |
| `preferred_quantized` | BOOLEAN | YES | NULL | Whether model should be loaded quantized |

#### Status Lifecycle

```
queued → running → succeeded
                 ↘ failed
```

- **queued**: Job created, waiting for execution
- **running**: Background download/initialization in progress
- **succeeded**: Job completed successfully, model available
- **failed**: Job failed, check `error` and `traceback` fields

#### Example Rows

**Successful pull with initialization:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "gpt2",
  "quantize": "auto",
  "init": true,
  "status": "succeeded",
  "created_at": "2025-12-18T10:30:00Z",
  "started_at": "2025-12-18T10:30:01Z",
  "finished_at": "2025-12-18T10:30:15Z",
  "error": null,
  "traceback": null,
  "local_path": "E:\\private-ai-server\\models\\hub\\models--gpt2\\snapshots\\abc123",
  "size_bytes": 548118077,
  "downloaded_bytes": 548118077,
  "progress": 1.0,
  "total_bytes": 548118077,
  "preferred_quantized": false
}
```

**Failed pull with error:**

```json
{
  "id": "650e8400-e29b-41d4-a716-446655440001",
  "model": "invalid/model-name",
  "quantize": "auto",
  "init": false,
  "status": "failed",
  "created_at": "2025-12-18T11:00:00Z",
  "started_at": "2025-12-18T11:00:02Z",
  "finished_at": "2025-12-18T11:00:05Z",
  "error": "Repository Not Found for url: https://huggingface.co/invalid/model-name",
  "traceback": "Traceback (most recent call last):\n  File ...",
  "local_path": null,
  "size_bytes": null,
  "downloaded_bytes": 0,
  "progress": null,
  "total_bytes": null,
  "preferred_quantized": null
}
```

---

## Database Operations

### Initialization

The database is automatically created on first server start by `JobDatabase._init_schema()`. No manual setup required.

### CRUD Operations

#### Create Job
```python
from database import JobDatabase

db = JobDatabase()
db.create_job({
    "id": "uuid-here",
    "model": "gpt2",
    "quantize": "auto",
    "init": True,
    "status": "queued",
    "created_at": "2025-12-18T10:30:00Z"
})
```

#### Read Job
```python
job = db.get_job("uuid-here")
# Returns: Dict[str, Any] or None if not found
```

#### Update Job
```python
db.update_job("uuid-here", {
    "status": "running",
    "started_at": "2025-12-18T10:30:01Z"
})
# Returns: bool (True if updated, False if not found)
```

#### List All Jobs
```python
jobs = db.get_all_jobs()
# Returns: List[Dict[str, Any]]
```

#### Delete Old Jobs
```python
deleted_count = db.delete_old_jobs(days=30)
# Deletes jobs older than 30 days
```

---

## Migration History

### v1.0 (2025-12-18)
- Initial schema creation
- Added all core fields for job tracking
- Implemented progress tracking fields (`downloaded_bytes`, `progress`, `total_bytes`)
- Added quantization preference field

---

## Maintenance

### Cleanup Strategy

The database grows over time. Recommended cleanup approaches:

1. **Manual cleanup**:
   ```python
   db.delete_old_jobs(days=30)  # Delete jobs older than 30 days
   ```

2. **Auto-cleanup** (future enhancement):
   - Schedule background task to prune old jobs
   - Keep only last N jobs per model
   - Configurable retention period

### Backup

To backup job history:

```powershell
# Windows
Copy-Item jobs.db jobs.db.backup

# Linux/WSL
cp jobs.db jobs.db.backup
```

### Reset Database

To start fresh (loses all job history):

```powershell
# Stop server first
Remove-Item jobs.db
# Restart server - schema recreates automatically
```

---

## Security Considerations

1. **No sensitive data**: Database does not store API keys, credentials, or user data
2. **Local only**: `jobs.db` is git-ignored, never committed to version control
3. **Error messages**: May contain local file paths in `error` and `traceback` fields

---

## Future Enhancements

Potential schema additions for v1.5+:

- **User tracking**: Add `user_id` field for multi-user deployments
- **Job priority**: Add `priority` field for queue management
- **Retry tracking**: Add `retry_count` and `max_retries` fields
- **Job tags**: Add `tags` JSON field for categorization
- **Rate limiting**: Add `api_call_count` and `last_api_call` for throttling
- **Workspace context**: Add `workspace_id` for multi-workspace support

See [ROADMAP.md](../ROADMAP.md) for planned database enhancements.
