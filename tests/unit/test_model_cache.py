"""
Unit tests for model_cache.py — LRU eviction and cooldown tracking.
"""

import pytest

import config
import state
import model_cache as mcache


@pytest.fixture(autouse=True)
def reset_state():
    """Wipe global state between tests."""
    state.model_cache.clear()
    state.model_meta.clear()
    state.failed_loads.clear()
    state.current_model = None
    state.current_model_name = None
    yield


class TestCooldown:
    def test_not_in_cooldown_initially(self):
        assert mcache.in_cooldown("foo") is False

    def test_in_cooldown_after_failure(self):
        mcache.record_failed_load("foo")
        assert mcache.in_cooldown("foo") is True

    def test_outside_cooldown_window(self, monkeypatch):
        mcache.record_failed_load("foo")
        # Simulate time passing beyond COOLDOWN_SECONDS
        monkeypatch.setattr(config, "COOLDOWN_SECONDS", 0)
        assert mcache.in_cooldown("foo") is False


class TestCacheModel:
    def test_caches_instance_and_metadata(self):
        mcache.cache_model("m1", "vllm", {"obj": 1}, extra_meta={"size_bytes": 100})
        assert "m1" in state.model_cache
        assert state.model_cache["m1"] == {"obj": 1}
        assert state.model_meta["m1"]["backend"] == "vllm"
        assert state.model_meta["m1"]["size_bytes"] == 100
        assert state.current_model_name == "m1"
        assert state.current_model == {"obj": 1}

    def test_lru_eviction(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CACHE_MODELS", 2)
        mcache.cache_model("a", "vllm", "objA")
        mcache.cache_model("b", "vllm", "objB")
        mcache.cache_model("c", "vllm", "objC")  # should evict "a"
        assert "a" not in state.model_cache
        assert "b" in state.model_cache
        assert "c" in state.model_cache
        assert len(state.model_cache) == 2

    def test_lru_promotion_on_access(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CACHE_MODELS", 2)
        mcache.cache_model("a", "vllm", "objA")
        mcache.cache_model("b", "vllm", "objB")
        # Touch "a" — re-insert to mark as MRU
        mcache.cache_model("a", "vllm", "objA")
        # Now add "c" — should evict "b" (oldest), not "a"
        mcache.cache_model("c", "vllm", "objC")
        assert "a" in state.model_cache
        assert "b" not in state.model_cache
        assert "c" in state.model_cache
