"""
Unit tests for the model scanner in routes/models.py.
"""

import os
from pathlib import Path

import pytest

import config
import state
from routes.models import get_model_scan_dirs, _resolve_snapshot_path


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
