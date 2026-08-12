"""
Shared pytest fixtures and path setup for the entire test suite.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure src/ is on sys.path so tests can `import app`, `import config`, etc.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


@pytest.fixture
def tmp_env(monkeypatch, tmp_path):
    """Provide an isolated env-var namespace backed by a temp directory."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    monkeypatch.setenv("HF_HUB_DISABLE_SYMLINKS", "1")
    return tmp_path


@pytest.fixture
def tmp_db_path():
    """Path to a temporary SQLite database file. Cleaned up after the test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass
