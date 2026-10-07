//! Speculative stream state machine (Issue #4).
//!
//! Models the lifecycle of one active inference stream as it alternates
//! between:
//!   Drafting  ->  Verifying  ->  Accepted  ->  (emit tokens)  ->  Drafting ...
//!
//! It is the real, unit-tested orchestration layer over [`crate::acceptance`]
//! and the KV rollback in [`crate::kv_cache`]. The actual draft/target backend
//! execution is provided by the caller through a trait; the machine itself is
//! backend-agnostic.

use crate::acceptance::{DraftCandidate, VerificationResult, verify_speculative_sequence};
use crate::kv_cache::KVCacheBlockManager;

/// Current phase of a speculative stream.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeculativeState {
    /// Draft model is producing candidate tokens.
    Drafting,
    /// Candidates are being verified against the target model.
    Verifying {
        candidates: Vec<DraftCandidate>,
        draft_logits: Vec<f32>,
    },
    /// A batch of candidates was accepted (or corrected) and is being emitted.
    Accepted { validated_tokens: Vec<u32> },
}

/// One live speculative inference stream.
#[derive(Debug, Clone)]
pub struct ActiveInferenceStream {
    pub request_id: String,
    /// Emitted/promoted token history (grows monotonically).
    pub token_history: Vec<u32>,
    pub state: SpeculativeState,
    /// Block ids this stream holds in the paged KV cache.
    pub paged_cache_indices: Vec<u32>,
}

impl ActiveInferenceStream {
    pub fn new(request_id: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            token_history: Vec::new(),
            state: SpeculativeState::Drafting,
            paged_cache_indices: Vec::new(),
        }
    }

    /// Feed a draft produced by the draft backend into the stream, entering the
    /// Verifying phase.
    pub fn start_verification(
        &mut self,
        draft_logits: Vec<f32>,
        candidates: Vec<DraftCandidate>,
    ) {
        self.state = SpeculativeState::Verifying {
            candidates,
            draft_logits,
        };
    }

    /// Run the verification pass on this stream: apply the acceptance rule
    /// over the current candidates (target probabilities must already be merged
    /// into each candidate), compute validated tokens, then transition to
    /// Accepted and return the result plus the KV rollback decision.
    ///
    /// `allocated_blocks` is how many KV blocks the draft phase provisioned for
    /// this stream.
    pub fn verify_and_advance(
        &mut self,
        threshold: f64,
        allocated_blocks: u64,
    ) -> (VerificationResult, u64) {
        let candidates = match &self.state {
            SpeculativeState::Verifying { candidates, .. } => candidates.clone(),
            _ => {
                // Not verifying — nothing to do.
                return (
                    VerificationResult {
                        accepted_tokens: 0,
                        rejected_tokens: 0,
                        validated_tokens: Vec::new(),
                        correction_token: None,
                    },
                    0,
                );
            }
        };

        let result = verify_speculative_sequence(&candidates, threshold);

        // KV rollback: surplus draft blocks are freed; validated tokens keep
        // theirs.
        let rollback = crate::acceptance::rollback_allocation_count(
            allocated_blocks,
            result.accepted_tokens.max(1) as u64,
        );

        // Promote accepted (and correction) tokens into history.
        self.token_history.extend(result.validated_tokens.iter().copied());
        if let Some(correction) = result.correction_token {
            self.token_history.push(correction);
        }

        // Transition to Accepted for emission, then back to Drafting for the
        // next speculative round.
        self.state = SpeculativeState::Accepted {
            validated_tokens: result.validated_tokens.clone(),
        };
        self.state = SpeculativeState::Drafting;

        (result, rollback)
    }

    /// Apply a finished verification's KV rollback to a live block manager,
    /// then promote history. Convenience used by the coordinator.
    pub fn commit_with_kv(
        &mut self,
        kv: &mut KVCacheBlockManager,
        threshold: f64,
        allocated_blocks: u64,
    ) -> VerificationResult {
        let (result, rollback) = self.verify_and_advance(threshold, allocated_blocks);
        let freed = kv.rollback_stream_blocks(self.stream_id(), rollback);
        // Keep only blocks that correspond to validated prefix length.
        let keep = result.accepted_tokens as usize;
        if self.paged_cache_indices.len() > keep {
            self.paged_cache_indices.truncate(keep);
        }
        debug_assert_eq!(freed, rollback, "expected rollback to free exactly n blocks");
        result
    }

    fn stream_id(&self) -> u32 {
        // Stable per-stream id derived from request id (for the KV manager).
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.request_id.hash(&mut h);
        (h.finish() & 0xFFFF_FFFF) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::KVCacheBlockManager;

    fn cand(token: u32, draft: f64, target: f64) -> DraftCandidate {
        DraftCandidate {
            token,
            draft_prob: draft,
            target_prob: target,
        }
    }

    #[test]
    fn stream_lifecycle_drafting_to_accepted() {
        let mut stream = ActiveInferenceStream::new("s1");
        assert_eq!(stream.state, SpeculativeState::Drafting);

        stream.start_verification(
            vec![0.5, 0.5],
            vec![cand(1, 0.9, 0.8), cand(2, 0.8, 0.9)],
        );
        assert!(matches!(stream.state, SpeculativeState::Verifying { .. }));

        let (result, rollback) = stream.verify_and_advance(0.5, 4);
        assert_eq!(result.accepted_tokens, 2);
        assert_eq!(rollback, 2, "allocated 4, validated 2 -> rollback 2");
        assert_eq!(stream.token_history, vec![1, 2]);
        assert_eq!(stream.state, SpeculativeState::Drafting);
    }

    #[test]
    fn rejection_appends_correction_token() {
        let mut stream = ActiveInferenceStream::new("s2");
        stream.start_verification(
            vec![0.5, 0.5],
            vec![cand(10, 0.9, 0.8), cand(11, 0.9, 0.1)],
        );
        let (result, _) = stream.verify_and_advance(0.5, 3);
        assert_eq!(result.accepted_tokens, 1);
        assert_eq!(result.rejected_tokens, 1);
        assert!(result.correction_token.is_some());
        // History includes validated token + correction.
        assert_eq!(stream.token_history.len(), 2);
    }

    #[test]
    fn commit_with_kv_rolls_back_blocks() {
        let mut kv = KVCacheBlockManager::new(16);
        let mut stream = ActiveInferenceStream::new("s3");
        let allocated = kv.allocate_blocks(stream.stream_id(), 5);
        stream.paged_cache_indices = allocated;
        stream.start_verification(
            vec![0.5, 0.5],
            vec![cand(1, 0.9, 0.8), cand(2, 0.9, 0.8)],
        );
        let result = stream.commit_with_kv(&mut kv, 0.5, 5);
        assert_eq!(result.accepted_tokens, 2);
        // Validated 2, kept 2 blocks; rollback freed 3.
        assert_eq!(stream.paged_cache_indices.len(), 2);
        assert_eq!(kv.total_used(), 2);
    }
}
