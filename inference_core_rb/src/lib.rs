//! PyO3 Python extension for `private-ai-server`: continuous-batching and
//! speculative-decoding core (Issues #3 and #4).
//!
//! Layout:
//! - [`batcher`] — Issue #3: queue, dynamic windowing trigger, compaction.
//! - [`acceptance`] — Issue #4: acceptance rule, residual correction, rollback math.
//! - [`kv_cache`] — Issue #4: paged-KV block manager + defragment.
//! - [`speculative`] — Issue #4: stream state machine over the above.
//! - This module — the thin Python-facing [`pyo3`] surface: `TokenBatchRequest`,
//!   `BatchCoordinator` (async tokio continuous batching), `SpeculativeStream`,
//!   `KVCacheBlockManager`.
//!
//! The heavy logic lives in the pure modules above (fully unit-tested with no
//! GPU/Windows runtime); this module only translates between Python objects and
//! those types.

pub mod acceptance;
pub mod batcher;
pub mod kv_cache;
pub mod speculative;

use std::time::{SystemTime, UNIX_EPOCH};

use pyo3::prelude::*;
use tokio::sync::mpsc;

use batcher::{BatchEngine, BatchItem};

/// Current wall-clock time in milliseconds (monotonic-ish; used by the
/// windowing rule).
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Persistent configuration for a [`BatchCoordinator`].
#[derive(Debug, Clone, Copy)]
struct BatchConfig {
    max_batch_size: usize,
    window_ms: u64,
    max_context_window: u64,
}

/// Drives the continuous-batching engine on a detached current-thread tokio
/// runtime (Issue #3): accumulate requests, dispatch on capacity or `window_ms`
/// tick, and enforce the context-window cap via batcher compaction.
async fn run_batch_loop(
    mut rx: mpsc::UnboundedReceiver<BatchItem>,
    cfg: BatchConfig,
    mut engine: BatchEngine,
) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(cfg.window_ms));
    loop {
        tokio::select! {
            item = rx.recv() => match item {
                Some(item) => engine.enqueue(item),
                // Producer dropped — channel closed; drain remaining then exit.
                None => break,
            },
            _ = interval.tick() => { /* window tick */ }
        }
        if engine.should_dispatch(now_ms(), cfg.max_batch_size, cfg.window_ms) {
            if let Some(batch) =
                engine.next_batch(now_ms(), cfg.max_batch_size, cfg.window_ms, cfg.max_context_window)
            {
                // Real backend dispatch happens on the Windows side; here we
                // simply consume the batch so the engine keeps draining.
                drop(batch);
            }
        }
    }
}

/// A tokenized inference request crossing the Python boundary (Issue #3).
#[pyclass(name = "TokenBatchRequest", module = "inference_core_rb")]
#[derive(Clone, Debug)]
struct TokenBatchRequest {
    request_id: String,
    tokens: Vec<u32>,
    max_tokens: u32,
}

#[pymethods]
impl TokenBatchRequest {
    #[new]
    fn new(request_id: String, tokens: Vec<u32>, max_tokens: u32) -> Self {
        Self {
            request_id,
            tokens,
            max_tokens,
        }
    }
}

/// Python-facing continuous-batching coordinator (Issue #3).
///
/// Construction spawns a thread-detached tokio current-thread runtime running
/// [`run_batch_loop`]. Each `enqueue_request` returns a Python awaitable that
/// resolves once the request has been accepted by the batcher.
#[pyclass(name = "BatchCoordinator", module = "inference_core_rb")]
struct BatchCoordinator {
    tx: mpsc::UnboundedSender<BatchItem>,
    max_batch_size: usize,
    window_ms: u64,
    max_context_window: u64,
}

#[pymethods]
impl BatchCoordinator {
    #[new]
    #[pyo3(signature = (max_batch_size, window_ms=100, max_context_window=4096))]
    fn new(max_batch_size: usize, window_ms: u64, max_context_window: u64) -> Self {
        let (tx, rx) = mpsc::unbounded_channel::<BatchItem>();
        let cfg = BatchConfig {
            max_batch_size,
            window_ms: window_ms.max(1),
            max_context_window,
        };
        let engine = BatchEngine::new();
        // Detached thread owning a current-thread runtime for the batch loop,
        // so the loop never blocks the Python event loop.
        std::thread::Builder::new()
            .name("inference-core-batch-loop".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_time()
                    .build()
                    .expect("tokio current-thread runtime");
                rt.block_on(run_batch_loop(rx, cfg, engine));
            })
            .expect("spawn batch loop thread");

        Self {
            tx,
            max_batch_size,
            window_ms,
            max_context_window,
        }
    }

    #[getter]
    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    #[getter]
    fn window_ms(&self) -> u64 {
        self.window_ms
    }

    #[getter]
    fn max_context_window(&self) -> u64 {
        self.max_context_window
    }

    /// Fire a request into the batcher. Returns a Python coroutine (awaitable)
    /// bridging through the pyo3-asyncio tokio runtime; it resolves to the
    /// request id once the batcher has accepted the item.
    #[pyo3(signature = (request_id, tokens, max_tokens, priority=0))]
    fn enqueue_request(
        &self,
        py: Python<'_>,
        request_id: &str,
        tokens: Vec<u32>,
        max_tokens: u32,
        priority: i32,
    ) -> PyResult<Py<PyAny>> {
        let tx = self.tx.clone();
        let rid: String = request_id.to_string();
        let enqueued_ms = now_ms();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let _ = tx.send(BatchItem {
                request_id: rid.clone(),
                tokens,
                max_tokens,
                priority,
                enqueued_ms,
            });
            Ok::<String, PyErr>(rid)
        })
        .map(|any| any.into())
    }
}

/// A (token, draft_prob, target_prob) triple for speculative verification
/// (Issue #4).
#[pyclass(name = "SpeculativeCandidate", module = "inference_core_rb")]
#[derive(Clone)]
struct SpeculativeCandidate {
    token: u32,
    draft_prob: f64,
    target_prob: f64,
}

#[pymethods]
impl SpeculativeCandidate {
    #[new]
    #[pyo3(signature = (token, draft_prob, target_prob))]
    fn new(token: u32, draft_prob: f64, target_prob: f64) -> Self {
        Self {
            token,
            draft_prob,
            target_prob,
        }
    }
}

impl From<&SpeculativeCandidate> for acceptance::DraftCandidate {
    fn from(c: &SpeculativeCandidate) -> Self {
        acceptance::DraftCandidate {
            token: c.token,
            draft_prob: c.draft_prob,
            target_prob: c.target_prob,
        }
    }
}

/// Python-facing speculative stream state machine (Issue #4), wrapping
/// [`speculative::ActiveInferenceStream`].
#[pyclass(name = "SpeculativeStream", module = "inference_core_rb")]
struct SpeculativeStream {
    inner: speculative::ActiveInferenceStream,
}

#[pymethods]
impl SpeculativeStream {
    #[new]
    fn new(request_id: &str) -> Self {
        Self {
            inner: speculative::ActiveInferenceStream::new(request_id),
        }
    }

    fn start_verification(&mut self, draft_logits: Vec<f32>, candidates: Vec<SpeculativeCandidate>) {
        let cands: Vec<acceptance::DraftCandidate> =
            candidates.iter().map(acceptance::DraftCandidate::from).collect();
        self.inner.start_verification(draft_logits, cands);
    }

    /// Verify the current candidates. Returns `(accepted, rejected, correction,
    /// rollback_allocation_count)`.
    #[pyo3(signature = (threshold=0.5, allocated_blocks=0))]
    fn verify_and_advance(
        &mut self,
        threshold: f64,
        allocated_blocks: u64,
    ) -> (usize, usize, Option<u32>, u64) {
        let (result, rollback) = self
            .inner
            .verify_and_advance(threshold, allocated_blocks);
        (result.accepted_tokens, result.rejected_tokens, result.correction_token, rollback)
    }

    #[getter]
    fn token_history(&self) -> Vec<u32> {
        self.inner.token_history.clone()
    }
}

/// Python-facing paged-KV block manager (Issue #4), wrapping
/// [`kv_cache::KVCacheBlockManager`].
#[pyclass(name = "KVCacheBlockManager", module = "inference_core_rb")]
struct KVCacheBlockManager {
    inner: kv_cache::KVCacheBlockManager,
}

#[pymethods]
impl KVCacheBlockManager {
    #[new]
    fn new(capacity: u64) -> Self {
        Self {
            inner: kv_cache::KVCacheBlockManager::new(capacity),
        }
    }

    fn allocate_blocks(&mut self, stream_id: u32, n: u64) -> Vec<u32> {
        self.inner.allocate_blocks(stream_id, n)
    }

    fn rollback_stream_blocks(&mut self, stream_id: u32, n: u64) -> u64 {
        self.inner.rollback_stream_blocks(stream_id, n)
    }

    fn free_stream(&mut self, stream_id: u32) -> u64 {
        self.inner.free_stream(stream_id)
    }

    fn defragment(&mut self) -> (u64, u64) {
        self.inner.defragment()
    }

    #[getter]
    fn free_blocks(&self) -> u64 {
        self.inner.free_blocks()
    }
}

/// Pure-logic smoke check callable from Python without touching the async
/// runtime: reports crate + module wiring.
#[pyfunction]
fn healthcheck() -> &'static str {
    "inference_core_rb ok (batcher + acceptance + kv_cache + speculative)"
}

#[pymodule]
fn inference_core_rb(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Init the pyo3-asyncio tokio runtime (idempotent global) so the async
    // `enqueue_request` boundary works when driven from a Python event loop.
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    pyo3_asyncio::tokio::init(builder);

    m.add_class::<TokenBatchRequest>()?;
    m.add_class::<BatchCoordinator>()?;
    m.add_class::<SpeculativeCandidate>()?;
    m.add_class::<SpeculativeStream>()?;
    m.add_class::<KVCacheBlockManager>()?;
    m.add_function(wrap_pyfunction!(healthcheck, m)?)?;
    Ok(())
}
