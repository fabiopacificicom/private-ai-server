# Building `inference_core_rb` for Windows (`.pyd`)

The Rust core here (`inference_core_rb/`) is a PyO3 extension. The **deploy
target is Windows-hosted Python**, so the final artifact must be a MSVC-built
`.pyd` — **not** the Linux `.so` that a WSL `cargo build` produces.

> **Why this matters:** `private-ai-server` ships inside a Windows PyInstaller
> bundle (`server.spec`) and runs the local Python runtime. A Linux `.so` will
> not import on Windows Python. The WSL box can only compile and unit-test the
> Rust logic; the final Windows `.pyd` must be produced on the Windows side
> (or cross-compiled to the MSVC target).

## What is committed here

- `Cargo.toml` — crate manifest (pyo3 0.20 + pyo3-asyncio 0.20 + tokio + numpy).
- `src/batcher.rs` — Issue #3 continuous-batching: queue, dynamic windowing
  trigger, batch compaction under a strict `max_context_window` cap.
- `src/acceptance.rs` — Issue #4 acceptance rule `min(1, target_prob/draft_prob)`,
  residual correction, `rollback_allocation_count`.
- `src/kv_cache.rs` — Issue #4 paged-KV block manager: `rollback_stream_blocks`
  + `defragment`.
- `src/speculative.rs` — Issue #4 stream state machine over the above.
- `src/lib.rs` — the pyo3 Python surface (`TokenBatchRequest`, `BatchCoordinator`,
  `SpeculativeStream`, `KVCacheBlockManager`).
- `smoke_test.py` — guards the Python integration (falls back cleanly when the
  binary is absent).
- `src/inference_core.py` (repo root `src/`) — guarded FastAPI-side import that
  returns `None` and preserves the existing Python sequential path when the
  `.pyd` is missing.

**26 Rust unit tests** cover the pure logic (no GPU/Windows backend needed) and
pass via `cargo test`.

## Build the Windows `.pyd` (remaining integration step)

On the **Windows host** (or a Windows CI runner) with the project's Windows
Python and the MSVC toolchain installed:

```powershell
# 1. Install the Rust toolchain with the MSVC target (if not already present)
rustup default stable-msvc

# 2. Install maturin into a Python environment that matches your target
#    (the same Python the PyInstaller bundle will use)
pip install maturin

# 3. From this directory, build the wheel
cd inference_core_rb
maturin build --release

# 4. Install the built wheel into the target Python
pip install target\wheels\inference_core_rb-0.1.0-cp313-cp313-win_amd64.whl
```

The wheel ships `inference_core_rb.pyd` for your target CPython version. **Pin
the wheel/`.pyd` to the exact CPython minor the server runs** (e.g. `cp313`) —
PyO3 extensions are per-interpreter-version unless built with `abi3`.

### Cross-compiling from WSL (optional / harder)

Building the MSVC `.pyd` directly from WSL requires:
- the MSVC toolchain reachable from WSL (via `vcvarsall.bat` + a `cmd.exe`
  bridge), **or** a GNU/MinGW target that Windows Python will not accept
  (Windows CPython is MSVC-built), **or** an `abi3` build via:
  ```bash
  PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release --target x86_64-pc-windows-msvc
  ```
  which still needs the MSVC linker. In practice, **the reliable path is
  `maturin build` on the Windows host**.

## Verification on Windows

```powershell
python -c "import inference_core_rb as rb; print(rb.healthcheck())"
python inference_core_rb\smoke_test.py
```

`smoke_test.py` exercises the async `BatchCoordinator` and the speculative/KV
logic through the Python boundary, and exits `0` with a clear message when the
core is absent so the server can fall back to the Python sequential pipeline.

## Guardrails honoured

- **Graceful fallback** — `src/inference_core.py` wraps the import in
  `try/except`; `make_coordinator()` returns `None` when the core fails to
  init/import, so the FastAPI server keeps running on the Python path.
- **Strict context cap** — the batcher defers (never drops) requests that would
  exceed `max_context_window`.
- **OOM safety** — `defragment()` + `rollback_stream_blocks()` free draft blocks
  and pack free space before the caller steps down to sequential decoding
  without dropping connections.
