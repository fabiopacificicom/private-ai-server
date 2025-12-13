Background Pull (Option 1) — In-Process Background Downloads

Goal

Make `/pull` non-blocking by running model snapshot downloads in background tasks so the FastAPI server remains responsive to other endpoints while a download is in progress. Provide job tracking endpoints so clients can inspect progress/status and optionally initialize (load) the model when the download completes.

High-level approach

- Schedule downloads as background tasks inside the FastAPI process using `asyncio.create_task` and `asyncio.to_thread` for blocking IO.
- Maintain an in-memory job registry (dict) for job metadata and status (queued, running, succeeded, failed), and a concurrency limiter (asyncio.Semaphore) to limit simultaneous downloads.
- Expose `GET /jobs` and `GET /jobs/{job_id}` endpoints to inspect jobs.
- Update `/pull` to enqueue a job and return 202 + job id immediately.
- When a background job finishes successfully, update `model_meta` with `local_path`, `size_bytes`, and `preferred_quantized`. Optionally initialize (load_model) if `init=true`.
- Keep current `MODEL_LOAD_COOLDOWN` and failed-load behavior; background tasks must respect it.

Detailed steps (implementation plan)

1) Add global job registry and semaphore

- Files to edit: `app.py`
- Add near top-level globals:
  - `import uuid, asyncio`
  - `jobs: Dict[str, Dict[str, Any]] = {}`
  - `MAX_CONCURRENT_PULLS = int(os.getenv("MAX_CONCURRENT_PULLS", "2"))`
  - `download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PULLS)`
- Acceptance criteria: these variables exist and default to 2 concurrent pulls.

2) Implement background pull worker function

- Files to edit: `app.py`
- Add function `_background_pull(job_id: str, model_name: str, quant: str, init: bool)` that:
  - Sets `jobs[job_id]` fields: `status='running'`, `started_at` timestamp.
  - Uses `async with _semaphore_wrapper(download_semaphore)` to limit concurrency. (If a wrapper is not used, use `await asyncio.to_thread(download_semaphore.acquire)` / `download_semaphore.release()` pattern.)
  - Calls `await asyncio.to_thread(snapshot_download, repo_id=model_name)` (blocking call offloaded to thread).
  - After download, probes `model_info` in thread to compute `size_bytes` (or use stored `size_bytes`).
  - Computes preferred quantization (use already implemented `/pull` rules or `model_meta[model]['preferred_quantized']`).
  - Writes `model_meta[model_name]` with `local_path`, `size_bytes`, `preferred_quantized`.
  - Sets `jobs[job_id]['status']='succeeded'` and stores `local_path` and timestamps.
  - If `init` is true, calls `load_model(model_name)` inside a `try`/`except` and records `init_result` in job.
  - On exception, set `jobs[job_id]['status']='failed'` and record `error` and stack trace.
- Notes:
  - Use `asyncio.to_thread` for both `snapshot_download` and `model_info` to avoid blocking the event loop.
  - Keep logs for start/finish/fail events.
- Acceptance criteria: `_background_pull` exists and updates `jobs` accordingly on success/failure.

3) Update `/pull` endpoint to spawn job

- Files to edit: `app.py`
- `/pull` should:
  - Validate payload (`model`, optional `quantize`, optional `init`).
  - Create job id: `jid = str(uuid.uuid4())`.
  - Create initial job entry: `jobs[jid] = {"id": jid, "model": model, "status": "queued", "created_at": ts, "quantize": quant, "init": init}`.
  - Schedule task: `asyncio.create_task(_background_pull(jid, model, quant, init))`.
  - Return 202 response: `{"job_id": jid}`, HTTP status 202.
- Acceptance criteria: `/pull` returns quickly with job id and does not block the request while download proceeds.

4) Add jobs endpoints

- Files to edit: `app.py`
- Implement:
  - `GET /jobs` → returns list of jobs (optionally filter by status)
  - `GET /jobs/{job_id}` → returns job details (status, error, local_path, timestamps, init_result)
- Ensure these endpoints return 200 and stable JSON structures.
- Acceptance criteria: clients can poll `GET /jobs/{id}` to discover completion and local_path.

5) Concurrency and throttling

- Use `MAX_CONCURRENT_PULLS` env var (default 2). Ensure semaphore is used in `_background_pull`.
- If concurrent limit reached, queued jobs will wait until semaphore available — but `/pull` will still return 202 immediately.
- Acceptance criteria: running number of downloads never exceeds configured limit.

6) Optional: progress estimation

- Implement a simple progress estimator by comparing downloaded bytes in `local_path` against `size_bytes` (if `model_info` available). Implementation note:
  - In `_background_pull`, after snapshot_download is started in thread, we can periodically (every few seconds) check on-disk bytes written (sum of file sizes) and update `jobs[job_id]['progress']`.
  - Snapshot download itself is blocking in `to_thread`; to estimate progress concurrently, run `snapshot_download` in a thread and in the async function poll the cache directory for size growth. This is slightly more complex but feasible.
- Acceptance criteria (optional): job contains `progress` field between 0-100 while running.

7) Tests and validation

- Create minimal integration test (script or manual instructions):
  - Call `POST /pull` for a small model (e.g., `sshleifer/tiny-gpt2` or another small HF model) with `init=false`.
  - Poll `GET /jobs/{job_id}` until status `succeeded`.
  - Verify `GET /models` contains the model with `local_path` and `size_bytes` set.
  - If `init=true`, verify `GET /status` shows cache size increased / model loaded.
- Acceptance criteria: test passes locally on a small model.

8) Documentation update

- Files to edit: `README.md` (already added) — add job workflow examples and sample `curl`/Python calls showing `/pull` returning a job id and how to poll `GET /jobs/{id}`.
- Acceptance criteria: README contains examples and troubleshooting notes for background pulls.

9) Edge cases and failure handling

- If snapshot_download raises due to network failure or insufficient disk, record error and keep job status `failed`. Provide a retry mechanism: `POST /jobs/{id}/retry` (optional) or re-run `/pull`.
- If process crashes, in-memory `jobs` is lost — document this limitation and recommend a persisted job store (Redis/sqlite) for production.

Milestones and timeline (approx)

- 0.5h — Add globals, semaphore, job registry, skeleton `_background_pull` and update `/pull` to return 202.
- 0.5h — Implement _background_pull details and update model_meta when download completes.
- 0.5h — Add `GET /jobs` and `GET /jobs/{id}` and basic tests with a small model.
- 0.5h — Add progress estimation (optional) and update README with examples.

Risks and mitigations

- Risk: running downloads inside server process consumes CPU and memory and may affect inference. Mitigate: allow `MAX_CONCURRENT_PULLS` to be low by default (2) and recommend separate worker processes for heavy usage (future Option 2).
- Risk: process crash loses jobs. Mitigate: recommend redis/sqlite persistence in production and document it in plan.

Acceptance criteria (overall)

- `/pull` is non-blocking and returns a `job_id` immediately (HTTP 202).
- Downloads run in background respecting `MAX_CONCURRENT_PULLS`.
- `GET /jobs/{job_id}` reflects completion and provides `local_path` and `size_bytes` when succeeded.
- `model_meta` is updated after successful pull so `load_model` can load from the local snapshot without network.


References

- `asyncio.create_task`, `asyncio.to_thread`
- `huggingface_hub.snapshot_download`
- Existing `model_meta` and `load_model` behavior in `app.py`
