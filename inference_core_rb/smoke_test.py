"""Smoke test / launcher for the Rust PyO3 inference core (Issues #3 & #4).

Runs the synchronous speculative + KV logic through the Python boundary and
exercises the async continuous-batching coordinator, and reports availability
clearly. It does NOT require a live model server or GPU: if the Rust core is
not importable on this platform (no Windows ``.pyd``), it logs the reason and
exits 0 with a clear message — exactly the graceful fallback behaviour.

Run with the project venv:
    <venv>/bin/python inference_core_rb/smoke_test.py
"""

import asyncio
import os
import sys

# Make the repo's src/ importable regardless of CWD (src is a namespace package).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import inference_core as core  # noqa: E402


def _run_sync_checks() -> int:
    failures = 0

    spec = core.make_speculative_stream("smoke-s1")
    if spec is None:
        print("[skip] SpeculativeStream unavailable (core not loaded)")
        return 1
    spec.start_verification(
        [0.5, 0.5],
        [
            core._rb.SpeculativeCandidate(1, 0.9, 0.8),
            core._rb.SpeculativeCandidate(2, 0.9, 0.8),
        ],
    )
    accepted, rejected, correction, rollback = spec.verify_and_advance(0.5, 4)
    print(f"[ok] verify_and_advance -> accepted={accepted} rejected={rejected} "
          f"correction={correction} rollback={rollback} history={spec.token_history}")
    if accepted != 2 or rejected != 0 or rollback != 2:
        print("[FAIL] unexpected verification result")
        failures += 1

    kv = core.make_kv_manager(16)
    if kv is None:
        print("[skip] KVCacheBlockManager unavailable")
        return 1
    kv.allocate_blocks(7, 4)
    freed = kv.rollback_stream_blocks(7, 1)
    print(f"[ok] kv rollback freed={freed} free_blocks={kv.free_blocks} defrag={kv.defragment()}")
    if freed != 1:
        print("[FAIL] unexpected rollback count")
        failures += 1

    return failures


def _run_async_coordinator() -> int:
    async def main() -> bool:
        coord = core.make_coordinator(max_batch_size=4, window_ms=100, max_context_window=1000)
        if coord is None:
            print("[skip] BatchCoordinator unavailable (Python fallback active)")
            return True
        awaitables = [
            coord.enqueue_request(f"rid-{i}", list(range(i + 1)), 50, priority=i)
            for i in range(6)
        ]
        results = await asyncio.gather(*awaitables)
        await asyncio.sleep(0.3)
        print(f"[ok] enqueued {len(results)} requests through Rust coordinator")
        return len(results) == 6

    return 0 if asyncio.run(main()) else 1


def main() -> int:
    print(f"Rust core available: {core.CORE_AVAILABLE}")
    err = core.core_load_error()
    if err:
        print(f"core load error: {err}")

    if not core.CORE_AVAILABLE:
        print("Rust core not available on this platform — Python sequential fallback preserved.")
        return 0

    failures = _run_sync_checks()
    failures += _run_async_coordinator()
    if failures:
        print(f"SMOKE TEST FAILED ({failures} failures)")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
