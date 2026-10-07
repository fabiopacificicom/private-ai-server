//! Paged-KV block management (Issue #4 OOM-safety).
//!
//! A real, unit-testable block manager backing the speculative streams. It
//! owns a fixed pool of KV blocks, tracks per-stream allocations and supports:
//! - [`KVCacheBlockManager::rollback_stream_blocks`] — free the surplus draft
//!   blocks a speculative stream allocated but did not validate;
//! - [`KVCacheBlockManager::defragment`] — reclaim free blocks by packing
//!   stream allocations together, the OOM-safety stepdown before falling back
//!   to sequential decoding.
//!
//! It is deliberately generic over the actual KV tensor storage (the GPU/Windows
//! backend cannot be exercised on this box); only the bookkeeping is here.

/// One page (block) of KV memory. `token_ids` are illustrative of what a block
/// would hold; the real tensors live in the backend.
#[derive(Debug, Clone, Default)]
pub struct KvBlock {
    pub token_ids: Vec<u32>,
    pub occupied: bool,
}

/// Tracks how much KV memory a single inference stream holds.
#[derive(Debug, Clone, Default)]
pub struct StreamAllocation {
    pub stream_id: u32,
    pub block_ids: Vec<u32>,
}

pub struct KVCacheBlockManager {
    /// Total block capacity of the cache.
    capacity: u64,
    /// Fixed pool of blocks.
    blocks: Vec<KvBlock>,
    /// Map stream_id -> its allocation.
    streams: std::collections::HashMap<u32, StreamAllocation>,
}

impl KVCacheBlockManager {
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            blocks: vec![KvBlock::default(); capacity as usize],
            streams: std::collections::HashMap::new(),
        }
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn total_used(&self) -> u64 {
        self.blocks.iter().filter(|b| b.occupied).count() as u64
    }

    pub fn free_blocks(&self) -> u64 {
        self.capacity - self.total_used()
    }

    /// Allocate `n` free blocks to `stream_id`, recording them in the stream's
    /// allocation. Returns the allocated block ids.
    pub fn allocate_blocks(&mut self, stream_id: u32, n: u64) -> Vec<u32> {
        let mut ids = Vec::new();
        for (idx, block) in self.blocks.iter_mut().enumerate() {
            if ids.len() as u64 >= n {
                break;
            }
            if !block.occupied {
                block.occupied = true;
                ids.push(idx as u32);
            }
        }
        // Record allocation even if partial (stream knows what it got).
        self.streams.entry(stream_id).or_default().stream_id = stream_id;
        if let Some(a) = self.streams.get_mut(&stream_id) {
            a.block_ids.extend(ids.iter().copied());
        }
        ids
    }

    /// Free `n` blocks owned by `stream_id` — the rollback for rejected
    /// speculative draft tokens. Blocks are freed from the *tail* of the
    /// stream's allocation (the most recently allocated, i.e. draft-speculative)
    /// so validated prefix blocks are never touched.
    pub fn rollback_stream_blocks(&mut self, stream_id: u32, n: u64) -> u64 {
        let Some(allocation) = self.streams.get_mut(&stream_id) else {
            return 0;
        };
        let mut freed = 0u64;
        while freed < n {
            let Some(block_id) = allocation.block_ids.pop() else {
                break;
            };
            if let Some(block) = self.blocks.get_mut(block_id as usize) {
                block.occupied = false;
                block.token_ids.clear();
                freed += 1;
            }
        }
        freed
    }

    /// Release *all* blocks of a stream (called when a stream finishes).
    pub fn free_stream(&mut self, stream_id: u32) -> u64 {
        let freed = if let Some(allocation) = self.streams.get(&stream_id) {
            let ids = allocation.block_ids.clone();
            let count = ids.len() as u64;
            for id in ids {
                if let Some(block) = self.blocks.get_mut(id as usize) {
                    block.occupied = false;
                    block.token_ids.clear();
                }
            }
            count
        } else {
            0
        };
        self.streams.remove(&stream_id);
        freed
    }

    /// Compaction: returns `(defragmented, blocks_reclaimed)`.
    /// Defragment packs all live blocks to the front of the pool so free blocks
    /// form a contiguous region. This is the pre-fallback step for OOM safety:
    /// after defragmenting, if there is still no room, the caller drops to
    /// sequential decoding — but the connection (stream) stays alive.
    pub fn defragment(&mut self) -> (u64, u64) {
        let live: Vec<KvBlock> = self
            .blocks
            .iter()
            .filter(|b| b.occupied)
            .cloned()
            .collect();
        let reclaimed = (self.capacity - live.len() as u64).max(0);
        // Rebuild pool: live first, then fresh empty blocks.
        self.blocks = live;
        while (self.blocks.len() as u64) < self.capacity {
            self.blocks.push(KvBlock::default());
        }
        // Rewrite stream block-ids compactly to the new front positions.
        let mut cursor = 0usize;
        for allocation in self.streams.values_mut() {
            let mut new_ids = Vec::with_capacity(allocation.block_ids.len());
            for _ in allocation.block_ids.iter() {
                new_ids.push(cursor as u32);
                cursor += 1;
            }
            allocation.block_ids = new_ids;
        }
        (self.blocks.iter().filter(|b| b.occupied).count() as u64, reclaimed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_and_free_blocks_tracked() {
        let mut mgr = KVCacheBlockManager::new(8);
        assert_eq!(mgr.free_blocks(), 8);
        let ids = mgr.allocate_blocks(1, 3);
        assert_eq!(ids.len(), 3);
        assert_eq!(mgr.free_blocks(), 5);
    }

    #[test]
    fn rollback_frees_tail_blocks() {
        let mut mgr = KVCacheBlockManager::new(8);
        mgr.allocate_blocks(7, 4);
        assert_eq!(mgr.total_used(), 4);
        // Validated 3 of 4 -> roll back 1.
        let freed = mgr.rollback_stream_blocks(7, 1);
        assert_eq!(freed, 1);
        assert_eq!(mgr.total_used(), 3);
        // Rollback count matches the Issue #4 formula.
        assert_eq!(crate::acceptance::rollback_allocation_count(4, 3), 1);
    }

    #[test]
    fn rollback_preserves_validated_prefix() {
        let mut mgr = KVCacheBlockManager::new(8);
        mgr.allocate_blocks(3, 5);
        // Roll back 3 of 5 -> only the first 2 (validated) remain.
        mgr.rollback_stream_blocks(3, 3);
        let alloc = mgr.streams.get(&3).unwrap();
        assert_eq!(alloc.block_ids.len(), 2);
        assert_eq!(mgr.total_used(), 2);
    }

    #[test]
    fn free_stream_releases_everything() {
        let mut mgr = KVCacheBlockManager::new(8);
        mgr.allocate_blocks(9, 2);
        mgr.allocate_blocks(10, 3);
        assert_eq!(mgr.total_used(), 5);
        let freed = mgr.free_stream(9);
        assert_eq!(freed, 2);
        assert_eq!(mgr.total_used(), 3);
        assert!(!mgr.streams.contains_key(&9));
    }

    #[test]
    fn defragment_packs_free_space_contiguous() {
        let mut mgr = KVCacheBlockManager::new(8);
        // Stream 1 takes 0,1; stream 2 takes 2,3; free the middle stream.
        mgr.allocate_blocks(1, 2);
        mgr.allocate_blocks(2, 2);
        mgr.free_stream(1); // frees blocks 0,1
        // Now live blocks are 2,3 (non-contiguous with free at front).
        let (live, reclaimed) = mgr.defragment();
        assert_eq!(live, 2);
        assert_eq!(reclaimed, 6);
        // After defrag, free blocks are contiguous at the tail.
        let first_free = mgr
            .blocks
            .iter()
            .position(|b| !b.occupied)
            .unwrap();
        assert_eq!(first_free, 2, "free space contiguous from idx 2");
    }

    #[test]
    fn defragment_noop_when_full() {
        let mut mgr = KVCacheBlockManager::new(4);
        mgr.allocate_blocks(1, 4);
        let (live, reclaimed) = mgr.defragment();
        assert_eq!(live, 4);
        assert_eq!(reclaimed, 0);
    }

    #[test]
    fn allocation_respects_capacity() {
        let mut mgr = KVCacheBlockManager::new(4);
        mgr.allocate_blocks(1, 4);
        let extra = mgr.allocate_blocks(2, 2);
        assert_eq!(extra.len(), 0, "no blocks left");
        assert_eq!(mgr.total_used(), 4);
    }
}
