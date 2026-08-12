"""
Unit tests for the model scanner in routes/models.py.
"""

import json
import os
from pathlib import Path

import pytest

import config
import state
from routes.models import get_model_scan_dirs, _resolve_snapshot_path, _discover_ollama_models


@pytest.fixture(autouse=True)
def reset_state():
    state.model_meta.clear()
    state.model_cache.clear()
    yield


class TestGetModelScanDirs:
    def test_returns_base_path(self, tmp_path):
        dirs = get_model_scan_dirs(str(tmp_path))
        assert os.path.normpath(str(tmp_path)) in dirs

    def test_includes_hub_subdir(self, tmp_path):
        (tmp_path / "hub").mkdir()
        dirs = get_model_scan_dirs(str(tmp_path))
        assert os.path.normpath(str(tmp_path / "hub")) in dirs

    def test_dedupes_overlapping_paths(self, tmp_path):
        (tmp_path / "hub").mkdir()
        (tmp_path / "huggingface").mkdir()
        dirs = get_model_scan_dirs(str(tmp_path))
        # All entries should be unique
        assert len(dirs) == len(set(dirs))


class TestResolveSnapshotPath:
    def test_returns_snapshot_subdir_when_present(self, tmp_path):
        # Layout: cache_root/snapshots/<hash>/...
        cache_root = tmp_path / "models--owner--repo"
        snap_a = cache_root / "snapshots" / "aaaa"
        snap_b = cache_root / "snapshots" / "bbbb"
        snap_a.mkdir(parents=True)
        snap_b.mkdir(parents=True)
        result = _resolve_snapshot_path(str(cache_root))
        # Should pick one of the snapshot dirs (most recent)
        assert "snapshots" in result

    def test_falls_back_to_root_when_no_snapshots(self, tmp_path):
        cache_root = tmp_path / "models--owner--repo"
        cache_root.mkdir()
        result = _resolve_snapshot_path(str(cache_root))
        assert result == str(cache_root)


class TestDiscoverOllamaModels:
    def _make_ollama_store(self, tmp_path):
        """Create a minimal Ollama store with one model manifest + blob."""
        store = tmp_path / "ollama"
        manifests_dir = store / "manifests" / "registry.ollama.ai" / "library" / "gemma2"
        blobs_dir = store / "blobs"
        manifests_dir.mkdir(parents=True)
        blobs_dir.mkdir()

        # A fake GGUF blob
        blob_name = "sha256-abc123"
        (blobs_dir / blob_name).write_bytes(b"fake-gguf-data")

        # Manifest referencing the blob as the model layer
        manifest = {
            "schemaVersion": 2,
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": "sha256:abc123",
                    "size": 12345,
                }
            ],
        }
        (manifests_dir / "latest").write_text(json.dumps(manifest))
        return store

    def test_discovers_ollama_model(self, tmp_path, monkeypatch):
        store = self._make_ollama_store(tmp_path)
        monkeypatch.setattr(config, "OLLAMA_MODELS_DIR", str(store))
        found = _discover_ollama_models()
        assert len(found) == 1
        m = found[0]
        assert m["model"] == "gemma2:latest"
        assert m["backend"] == "gguf_llama_cpp"
        assert m["size_bytes"] == 12345
        assert m["local_path"].endswith("sha256-abc123")

    def test_returns_empty_when_no_store(self, monkeypatch):
        monkeypatch.setattr(config, "OLLAMA_MODELS_DIR", None)
        assert _discover_ollama_models() == []

    def test_returns_empty_when_store_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OLLAMA_MODELS_DIR", str(tmp_path / "nope"))
        assert _discover_ollama_models() == []

    def test_skips_manifest_without_model_layer(self, tmp_path, monkeypatch):
        store = tmp_path / "ollama"
        manifests_dir = store / "manifests" / "registry.ollama.ai" / "library" / "foo"
        manifests_dir.mkdir(parents=True)
        # Manifest with no model layer
        (manifests_dir / "latest").write_text(json.dumps({"layers": []}))
        monkeypatch.setattr(config, "OLLAMA_MODELS_DIR", str(store))
        assert _discover_ollama_models() == []
