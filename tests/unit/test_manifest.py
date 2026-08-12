"""
Unit tests for the models_manifest.json load/save cycle.
"""

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def fresh_loader(monkeypatch, tmp_path):
    """Load a fresh copy of `loader` with MANIFEST_PATH pointing at a temp file,
    so the real on-disk manifest doesn't pollute the test."""
    # Clear state before each test
    import state
    state.model_meta.clear()
    state.model_cache.clear()

    # Reload loader with patched MANIFEST_PATH
    manifest = tmp_path / "models_manifest.json"
    # Remove cached module so we can re-import with the patch in place
    sys.modules.pop("loader", None)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    import loader as fresh_loader
    monkeypatch.setattr(fresh_loader, "MANIFEST_PATH", str(manifest))
    yield fresh_loader, state, manifest


def test_save_then_load_roundtrip(fresh_loader):
    loader, state, manifest = fresh_loader
    state.model_meta["meta-models/Muse-Glimmer-30B"] = {
        "local_path": str(Path(manifest).parent / "snapshots" / "abc"),
        "size_bytes": 60_000_000_000,
        "backend": "transformers_pipeline",
    }
    # Create the directory so the path exists for load_manifest
    (Path(manifest).parent / "snapshots" / "abc").mkdir(parents=True)

    state.model_meta["skipped"] = {"description": "no local_path"}

    loader.save_manifest()

    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert "meta-models/Muse-Glimmer-30B" in data
    assert "skipped" not in data
    assert data["meta-models/Muse-Glimmer-30B"]["size_bytes"] == 60_000_000_000

    # Wipe state and reload
    state.model_meta.clear()
    loader.load_manifest()
    assert "meta-models/Muse-Glimmer-30B" in state.model_meta
    assert state.model_meta["meta-models/Muse-Glimmer-30B"]["local_path"].endswith("abc")


def test_load_skips_missing_paths(fresh_loader, tmp_path):
    loader, state, manifest = fresh_loader
    manifest.write_text(json.dumps({
        "real": {"local_path": str(tmp_path / "real_dir")},
        "ghost": {"local_path": str(tmp_path / "does_not_exist")},
    }))
    (tmp_path / "real_dir").mkdir()

    loader.load_manifest()

    assert "real" in state.model_meta
    assert "ghost" not in state.model_meta


def test_load_handles_missing_file(fresh_loader):
    loader, state, _ = fresh_loader
    loader.load_manifest()  # manifest file doesn't exist — should not raise
    assert state.model_meta == {}

