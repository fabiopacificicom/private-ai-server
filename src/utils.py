import asyncio
import gc

from fastapi import HTTPException

import config


async def with_timeout(coro, timeout_seconds: int, cleanup_func=None):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        config.log.warning("Request timed out after %s seconds", timeout_seconds)
        if cleanup_func is not None:
            try:
                cleanup_func()
            except Exception:
                config.log.exception("Error during timeout cleanup")
        raise HTTPException(status_code=408, detail=f"Request timed out after {timeout_seconds} seconds")


def gpu_cleanup_fn():
    if config.torch is not None and config.torch.cuda.is_available():
        try:
            config.torch.cuda.empty_cache()
            config.torch.cuda.synchronize()
        except Exception:
            pass
    gc.collect()
