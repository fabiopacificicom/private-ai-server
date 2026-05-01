"""
Unit tests for the JobDatabase class in database.py.
Uses a temp file for isolation — no side effects on production jobs.db.
"""

import os
import tempfile
import pytest
from datetime import datetime, timezone

# Ensure we can import database from the parent directory
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import JobDatabase


def make_job(job_id: str = "test-job-1", model: str = "test/model", status: str = "queued") -> dict:
    """Helper: build a minimal valid job dict."""
    return {
        "id": job_id,
        "model": model,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quantize": None,
        "init": False,
        "started_at": None,
        "finished_at": None,
        "error": None,
        "traceback": None,
        "local_path": None,
        "size_bytes": None,
        "downloaded_bytes": 0,
        "progress": None,
        "total_bytes": None,
        "preferred_quantized": None,
    }


@pytest.fixture
def db():
    """Create a fresh JobDatabase backed by a temp file, cleaned up after the test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        yield JobDatabase(db_path=db_path)
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# create_job
# ---------------------------------------------------------------------------

class TestCreateJob:
    def test_create_job_succeeds(self, db):
        job = make_job()
        db.create_job(job)
        result = db.get_job(job["id"])
        assert result is not None
        assert result["id"] == job["id"]
        assert result["model"] == job["model"]
        assert result["status"] == "queued"

    def test_create_job_defaults_downloaded_bytes(self, db):
        job = make_job()
        db.create_job(job)
        result = db.get_job(job["id"])
        assert result["downloaded_bytes"] == 0

    def test_create_multiple_jobs(self, db):
        for i in range(3):
            db.create_job(make_job(job_id=f"job-{i}", model=f"model/{i}"))
        jobs = db.list_jobs()
        assert len(jobs) == 3


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------

class TestGetJob:
    def test_get_existing_job(self, db):
        job = make_job("get-test")
        db.create_job(job)
        result = db.get_job("get-test")
        assert result is not None
        assert result["id"] == "get-test"

    def test_get_nonexistent_job_returns_none(self, db):
        result = db.get_job("does-not-exist")
        assert result is None

    def test_get_job_preserves_fields(self, db):
        job = make_job("field-test", model="nvidia/test-model")
        job["quantize"] = "q4"
        job["init"] = True
        job["size_bytes"] = 1024
        db.create_job(job)
        result = db.get_job("field-test")
        assert result["model"] == "nvidia/test-model"
        assert result["quantize"] == "q4"
        assert result["init"] == 1  # SQLite stores booleans as integers
        assert result["size_bytes"] == 1024


# ---------------------------------------------------------------------------
# update_job
# ---------------------------------------------------------------------------

class TestUpdateJob:
    def test_update_status(self, db):
        db.create_job(make_job("upd-test"))
        result = db.update_job("upd-test", {"status": "running"})
        assert result is True
        assert db.get_job("upd-test")["status"] == "running"

    def test_update_multiple_fields(self, db):
        db.create_job(make_job("multi-upd"))
        db.update_job("multi-upd", {"status": "succeeded", "progress": 100.0, "local_path": "/models/test"})
        result = db.get_job("multi-upd")
        assert result["status"] == "succeeded"
        assert result["progress"] == 100.0
        assert result["local_path"] == "/models/test"

    def test_update_nonexistent_job_returns_false(self, db):
        result = db.update_job("ghost-job", {"status": "running"})
        assert result is False

    def test_update_error_fields(self, db):
        db.create_job(make_job("err-job"))
        db.update_job("err-job", {
            "status": "failed",
            "error": "OOM",
            "traceback": "Traceback...",
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        result = db.get_job("err-job")
        assert result["status"] == "failed"
        assert result["error"] == "OOM"
        assert result["traceback"] == "Traceback..."


# ---------------------------------------------------------------------------
# list_jobs
# ---------------------------------------------------------------------------

class TestListJobs:
    def test_list_all_jobs(self, db):
        for i in range(5):
            db.create_job(make_job(f"list-{i}"))
        jobs = db.list_jobs()
        assert len(jobs) == 5

    def test_list_jobs_with_status_filter(self, db):
        db.create_job(make_job("running-1", status="running"))
        db.create_job(make_job("running-2", status="running"))
        db.create_job(make_job("queued-1", status="queued"))
        running = db.list_jobs(status_filter="running")
        assert len(running) == 2
        assert all(j["status"] == "running" for j in running)

    def test_list_jobs_empty_db(self, db):
        jobs = db.list_jobs()
        assert jobs == []

    def test_list_jobs_respects_limit(self, db):
        for i in range(10):
            db.create_job(make_job(f"lim-{i}"))
        jobs = db.list_jobs(limit=3)
        assert len(jobs) == 3

    def test_list_jobs_ordered_by_created_at_desc(self, db):
        import time
        for i in range(3):
            j = make_job(f"ord-{i}")
            j["created_at"] = f"2024-01-0{i+1}T00:00:00Z"
            db.create_job(j)
        jobs = db.list_jobs()
        dates = [j["created_at"] for j in jobs]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# get_active_jobs
# ---------------------------------------------------------------------------

class TestGetActiveJobs:
    def test_returns_queued_and_running(self, db):
        db.create_job(make_job("q1", status="queued"))
        db.create_job(make_job("r1", status="running"))
        db.create_job(make_job("d1", status="succeeded"))
        db.create_job(make_job("f1", status="failed"))
        active = db.get_active_jobs()
        assert len(active) == 2
        statuses = {j["status"] for j in active}
        assert statuses == {"queued", "running"}

    def test_empty_when_no_active_jobs(self, db):
        db.create_job(make_job("done-1", status="succeeded"))
        assert db.get_active_jobs() == []

    def test_empty_db(self, db):
        assert db.get_active_jobs() == []


# ---------------------------------------------------------------------------
# cleanup_old_jobs
# ---------------------------------------------------------------------------

class TestCleanupOldJobs:
    def test_removes_old_completed_jobs(self, db):
        old_job = make_job("old-succeeded")
        old_job["status"] = "succeeded"
        old_job["created_at"] = "2020-01-01T00:00:00Z"
        db.create_job(old_job)

        old_failed = make_job("old-failed")
        old_failed["status"] = "failed"
        old_failed["created_at"] = "2020-01-01T00:00:00Z"
        db.create_job(old_failed)

        deleted = db.cleanup_old_jobs(days=7)
        assert deleted == 2
        assert db.get_job("old-succeeded") is None
        assert db.get_job("old-failed") is None

    def test_keeps_recent_jobs(self, db):
        recent = make_job("recent-ok")
        recent["status"] = "succeeded"
        recent["created_at"] = datetime.now(timezone.utc).isoformat()
        db.create_job(recent)

        deleted = db.cleanup_old_jobs(days=7)
        assert deleted == 0
        assert db.get_job("recent-ok") is not None

    def test_does_not_delete_active_old_jobs(self, db):
        """Active jobs (queued/running) should never be cleaned up."""
        old_active = make_job("old-running")
        old_active["status"] = "running"
        old_active["created_at"] = "2020-01-01T00:00:00Z"
        db.create_job(old_active)

        deleted = db.cleanup_old_jobs(days=7)
        assert deleted == 0
        assert db.get_job("old-running") is not None

    def test_empty_db_returns_zero(self, db):
        assert db.cleanup_old_jobs(days=1) == 0
