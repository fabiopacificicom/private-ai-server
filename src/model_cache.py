import gc
import time
from typing import Any, Dict

import config
import state


def gpu_cleanup() -> None:
    if config.torch is not None and config.torch.cuda.is_available():
        try:
            config.torch.cuda.empty_cache()
            config.torch.cuda.synchronize()
            gc.collect()
        except Exception:
            config.log.debug("GPU cleanup failed (non-fatal)")


def evict_lru_if_needed() -> None:
    while len(state.model_cache) > config.MAX_CACHE_MODELS:
        name, mdl = state.model_cache.popitem(last=False)
        try:
            del mdl
            gc.collect()
            if config.torch is not None and config.torch.cuda.is_available():
                config.torch.cuda.empty_cache()
                config.torch.cuda.synchronize()
        except Exception:
            config.log.exception("Error evicting model %s", name)
        config.log.info("Evicted LRU model: %s", name)


def record_failed_load(model_name: str) -> None:
    state.failed_loads[model_name] = time.time()


def in_cooldown(model_name: str) -> bool:
    t = state.failed_loads.get(model_name)
    return bool(t and (time.time() - t) < config.COOLDOWN_SECONDS)


def cache_model(name: str, backend: str, instance: Any, extra_meta: Dict[str, Any] = {}) -> None:
    # Delete first so re-inserting an existing key moves it to MRU position
    state.model_cache.pop(name, None)
    state.model_cache[name] = instance
    state.model_meta[name] = {"backend": backend, **extra_meta}
    evict_lru_if_needed()
    state.current_model = instance
    state.current_model_name = name
    config.log.info("✅ Loaded and cached: %s (backend=%s)", name, backend)
