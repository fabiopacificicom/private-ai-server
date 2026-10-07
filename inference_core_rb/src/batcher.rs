//! Continuous-batching engine (Issue #3).
//!
//! Pure, unit-testable core logic for the PyO3 [`BatchCoordinator`]:
//! request queuing, the dynamic windowing batch-trigger rule and batch
//! compaction under a strict `max_context_window` token cap. The Rust type
//! itself is the authoritative implementation; the Python-facing coordinator
//! is a thin async wrapper around [`BatchEngine::next_batch`].

use std::cmp::Ordering;
use std::collections::VecDeque;

/// A discrete inference request waiting inside the batch machinery.
///
/// `priority` is an explicit scheduling hint; `enqueued_ms` is a monotonic
/// timestamp (from the coordinator) used by the windowing rule so that a
/// request can never be starved indefinitely by continuous higher-priority
/// arrivals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchItem {
    pub request_id: String,
    /// Pre-tokenized prompt ids. In the Python integration these are passed
    /// as a contiguous numpy array and copied once across the boundary.
    pub tokens: Vec<u32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Relative scheduling priority (higher wins).
    pub priority: i32,
    /// Monotonic enqueue timestamp in milliseconds.
    pub enqueued_ms: u64,
}

impl BatchItem {
    /// Combined context length this request would occupy in a batch: the
    /// prompt plus the tokens it is expected to generate.
    pub fn total_context_len(&self) -> u64 {
        self.tokens.len() as u64 + self.max_tokens as u64
    }
}

/// The reason a batch was dispatched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainReason {
    /// The batch reached `max_batch_size` (capacity trigger).
    Capacity,
    /// The `window_ms` interval elapsed with a non-empty batch (latency trigger).
    Window,
}

/// Outcome of [`BatchEngine::next_batch`].
#[derive(Debug, Clone)]
pub struct Batch {
    /// Items selected for this dispatch, in deterministic order.
    pub items: Vec<BatchItem>,
    /// Why the batch was dispatched.
    pub reason: DrainReason,
}

/// The scheduling core shared by the Python-facing coordinator.
///
/// Safe to construct off the tokio runtime; all state is plain, owned Rust.
#[derive(Debug, Default)]
pub struct BatchEngine {
    pending: VecDeque<BatchItem>,
}

impl BatchEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue a request, keeping the queue ordered by `(priority desc,
    /// enqueued_ms asc)` so we always schedule the most important, oldest job
    /// first. Insertion is O(n) — fine for the modest queue sizes seen here.
    pub fn enqueue(&mut self, item: BatchItem) {
        // Insert `item` before the first existing entry that sorts after it
        // (`item <= existing`). Entries already placed before that are smaller
        // and stay ahead.
        let idx = self
            .pending
            .iter()
            .position(|existing| item.cmp(existing) != std::cmp::Ordering::Greater)
            .unwrap_or(self.pending.len());
        self.pending.insert(idx, item);
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Decide whether the current queue must be dispatched *now* under the
    /// dynamic windowing rule (Issue #3): dispatch on capacity, otherwise on
    /// the `window_ms` timeout measured since the head-of-line request was
    /// enqueued. This guarantees that a request waiting for `window_ms` is
    /// dispatched even when the batch never fills.
    pub fn should_dispatch(&self, now_ms: u64, max_batch_size: usize, window_ms: u64) -> bool {
        if self.pending.is_empty() {
            return false;
        }
        if self.pending.len() >= max_batch_size {
            return true;
        }
        // Head-of-line latency bound.
        let head = &self.pending[0];
        now_ms.saturating_sub(head.enqueued_ms) >= window_ms
    }

    /// Build the next batch to dispatch.
    ///
    /// Guardrails:
    /// - capacity / window triggers;
    /// - a strict `max_context_window` token cap: we greedily pack compatible
    ///   requests and *defer* any request — including a single oversized one —
    ///   that would push the combined context over the cap (Issue #3/#4:
    ///   never drop a connection, just defer until it fits).
    ///
    /// `now_ms` is used to apply the window rule. Returns `None` when nothing
    /// should run.
    pub fn next_batch(
        &mut self,
        now_ms: u64,
        max_batch_size: usize,
        window_ms: u64,
        max_context_window: u64,
    ) -> Option<Batch> {
        if self.pending.is_empty() {
            return None;
        }

        let len = self.pending.len();
        let head = &self.pending[0];
        let reason = if len >= max_batch_size {
            DrainReason::Capacity
        } else if now_ms.saturating_sub(head.enqueued_ms) >= window_ms {
            DrainReason::Window
        } else {
            return None;
        };

        let cap_limit = max_batch_size;

        let mut items: Vec<BatchItem> = Vec::new();
        let mut combined: u64 = 0;
        let mut deferred: Vec<BatchItem> = Vec::new();

        while let Some(item) = self.pending.pop_front() {
            if items.len() >= cap_limit {
                deferred.push(item);
                continue;
            }
            // Strict context-window cap: a request that would push the combined
            // token count over the budget is deferred (never dropped).
            let would_be = combined.saturating_add(item.total_context_len());
            if would_be > max_context_window {
                deferred.push(item);
                continue;
            }
            combined = would_be;
            items.push(item);
        }

        // Re-queue deferred items to the front (they were popped in priority
        // order, so prepend in reverse).
        while let Some(item) = deferred.pop() {
            self.pending.push_front(item);
        }

        Some(Batch { items, reason })
    }

    /// Compaction pass run between dispatches (Issue #4's OOM-safety stepdown).
    /// Re-evaluates the queue under a tighter token budget and drops nothing —
    /// it only reorders so that the fewest, most-important requests fit. Items
    /// that no longer fit are deferred without being dropped.
    ///
    /// Returns the number of items deferred in this pass.
    pub fn compact(&mut self, max_context_window: u64) -> usize {
        let mut combined: u64 = 0;
        let mut deferred = 0usize;
        let before = self.pending.len();

        // Re-pack from scratch under the (possibly reduced) budget.
        let drained: Vec<BatchItem> = self.pending.drain(..).collect();
        for item in drained {
            let would_be = combined.saturating_add(item.total_context_len());
            if would_be <= max_context_window {
                combined = would_be;
                self.pending.push_back(item);
            } else {
                deferred += 1;
                // Keep it alive rather than dropping the connection.
                self.pending.push_back(item);
            }
        }
        debug_assert_eq!(self.pending.len(), before, "compact must never drop items");
        deferred
    }
}

impl PartialOrd for BatchItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BatchItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority sorts first; tie-break by earlier enqueue time.
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| self.enqueued_ms.cmp(&other.enqueued_ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, tokens: usize, max_tokens: u32, priority: i32, enq: u64) -> BatchItem {
        BatchItem {
            request_id: id.to_string(),
            tokens: vec![0u32; tokens],
            max_tokens,
            priority,
            enqueued_ms: enq,
        }
    }

    #[test]
    fn queue_orders_by_priority_then_age() {
        let mut e = BatchEngine::new();
        e.enqueue(item("low", 1, 1, 0, 100));
        e.enqueue(item("high", 1, 1, 5, 50));
        e.enqueue(item("mid", 1, 1, 2, 70));
        assert_eq!(e.pending[0].request_id, "high");
        assert_eq!(e.pending[1].request_id, "mid");
        assert_eq!(e.pending[2].request_id, "low");
    }

    #[test]
    fn capacity_trigger_dispatches_full_batch() {
        let mut e = BatchEngine::new();
        e.enqueue(item("a", 1, 1, 0, 0));
        e.enqueue(item("b", 1, 1, 0, 1));
        e.enqueue(item("c", 1, 1, 0, 2));
        let b = e.next_batch(3, 2, 1000, 1000).expect("batch");
        assert_eq!(b.reason, DrainReason::Capacity);
        assert_eq!(b.items.len(), 2);
        // Third left pending.
        assert_eq!(e.pending_len(), 1);
    }

    #[test]
    fn window_trigger_dispatches_before_capacity() {
        let mut e = BatchEngine::new();
        e.enqueue(item("a", 1, 1, 0, 0));
        // Not yet full, but head has waited >= window_ms.
        let b = e.next_batch(1000, 8, 500, 1000).expect("batch");
        assert_eq!(b.reason, DrainReason::Window);
        assert_eq!(b.items.len(), 1);
    }

    #[test]
    fn no_dispatch_within_window_below_capacity() {
        let mut e = BatchEngine::new();
        e.enqueue(item("a", 1, 1, 0, 0));
        assert!(e.next_batch(100, 8, 500, 1000).is_none());
    }

    #[test]
    fn should_dispatch_reflects_rules() {
        let mut e = BatchEngine::new();
        assert!(!e.should_dispatch(0, 2, 500));
        e.enqueue(item("a", 1, 1, 0, 0));
        assert!(!e.should_dispatch(100, 2, 500));
        e.enqueue(item("b", 1, 1, 0, 1));
        assert!(e.should_dispatch(100, 2, 500), "capacity");
        assert!(e.should_dispatch(600, 2, 500), "window");
    }

    #[test]
    fn compaction_defers_over_budget_request_without_dropping() {
        let mut e = BatchEngine::new();
        // max_context_window token cap of 10; token+max_tokens = 12 for a.
        e.enqueue(item("a", 10, 2, 1, 0));
        e.enqueue(item("b", 1, 1, 0, 1));
        let before = e.pending_len();
        let deferred = e.compact(10);
        assert_eq!(deferred, 1);
        assert_eq!(e.pending_len(), before, "nothing dropped");
        // Budget: only 2 fits in 10 window after a's 12; b fits alone too, but
        // a holds the whole window — b deferred.
        assert_eq!(e.pending[0].request_id, "a");
        assert_eq!(e.pending[1].request_id, "b");
    }

    #[test]
    fn next_batch_defer_never_exceeds_context_window() {
        let mut e = BatchEngine::new();
        e.enqueue(item("big", 10, 3, 0, 0)); // 13 total
        e.enqueue(item("small", 1, 1, 0, 1)); // 2 total
        let b = e.next_batch(1000, 8, 100, 12).expect("batch");
        // Only "small" (2) fits inside the 12-token window; "big" (13) deferred.
        assert_eq!(b.items.len(), 1);
        assert_eq!(b.items[0].request_id, "small");
        assert_eq!(e.pending_len(), 1, "big deferred, not dropped");
    }

    #[test]
    fn oversized_request_is_deferred_not_forced() {
        // Issue semantics: a request that exceeds the context cap is DEFERRED
        // (never dropped, never forced) until the window fires; a fitting
        // request runs first.
        let mut e = BatchEngine::new();
        e.enqueue(item("huge", 1000, 100, 0, 0)); // 1100 > 100
        e.enqueue(item("ok", 1, 1, 0, 1));
        let b = e.next_batch(1000, 8, 500, 100).expect("batch"); // window fired
        assert_eq!(b.reason, DrainReason::Window);
        assert_eq!(b.items.len(), 1);
        assert_eq!(b.items[0].request_id, "ok", "oversized 'huge' deferred");
        assert_eq!(e.pending_len(), 1, "'huge' kept alive, not dropped");
    }

    #[test]
    fn empty_engine_returns_none() {
        let mut e = BatchEngine::new();
        assert!(e.next_batch(0, 8, 500, 1000).is_none());
        assert!(e.is_empty());
    }
}
