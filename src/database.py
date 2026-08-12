"""
Database module for persistent job storage using SQLite.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List


class JobDatabase:
    """SQLite-based job storage for persistence across server restarts."""
    
    def __init__(self, db_path: str = "jobs.db"):
        """Initialize database connection and create schema if needed."""
        self.db_path = db_path
        self._init_schema()
    
    def _init_schema(self):
        """Create jobs table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    quantize TEXT,
                    init BOOLEAN,
                    status TEXT NOT NULL DEFAULT 'queued',
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    error TEXT,
                    traceback TEXT,
                    local_path TEXT,
                    size_bytes INTEGER,
                    downloaded_bytes INTEGER DEFAULT 0,
                    progress REAL,
                    total_bytes INTEGER,
                    preferred_quantized BOOLEAN
                )
            ''')
            conn.commit()
    
    def create_job(self, job_data: Dict[str, Any]) -> None:
        """Insert a new job into the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO jobs (
                    id, model, quantize, init, status, created_at, started_at, 
                    finished_at, error, traceback, local_path, size_bytes,
                    downloaded_bytes, progress, total_bytes, preferred_quantized
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_data["id"],
                job_data["model"],
                job_data.get("quantize"),
                job_data.get("init", False),
                job_data.get("status", "queued"),
                job_data.get("created_at"),
                job_data.get("started_at"),
                job_data.get("finished_at"),
                job_data.get("error"),
                job_data.get("traceback"),
                job_data.get("local_path"),
                job_data.get("size_bytes"),
                job_data.get("downloaded_bytes", 0),
                job_data.get("progress"),
                job_data.get("total_bytes"),
                job_data.get("preferred_quantized")
            ))
            conn.commit()
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a job by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT * FROM jobs WHERE id = ?', (job_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
                
            return dict(row)
    
    def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job fields. Returns True if job was found and updated."""
        if not updates:
            return False
        
        # Build dynamic UPDATE query
        set_clauses = []
        values = []
        
        for field, value in updates.items():
            set_clauses.append(f"{field} = ?")
            values.append(value)
        
        values.append(job_id)  # For WHERE clause
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f'''
                UPDATE jobs SET {", ".join(set_clauses)} WHERE id = ?
            ''', values)
            conn.commit()
            
            return cursor.rowcount > 0
    
    def list_jobs(self, limit: int = 50, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List recent jobs, optionally filtered by status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if status_filter:
                cursor = conn.execute('''
                    SELECT * FROM jobs WHERE status = ? 
                    ORDER BY created_at DESC LIMIT ?
                ''', (status_filter, limit))
            else:
                cursor = conn.execute('''
                    SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?
                ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get all jobs with status 'queued' or 'running'."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM jobs WHERE status IN ('queued', 'running')
                ORDER BY created_at ASC
            ''')
            
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_jobs(self, days: int = 7) -> int:
        """Remove jobs older than specified days. Returns count of deleted jobs."""
        cutoff_date = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat() + "Z"
        
        # Calculate cutoff timestamp (days ago)
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        cutoff_str = cutoff.isoformat() + "Z"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                DELETE FROM jobs WHERE created_at < ? AND status IN ('succeeded', 'failed')
            ''', (cutoff_str,))
            conn.commit()
            
            return cursor.rowcount
    
    def close(self):
        """Close database connection (if needed for cleanup)."""
        # SQLite connections are managed per-transaction, no persistent connection
        pass


# Global database instance
job_db: Optional[JobDatabase] = None


def init_job_database(db_path: str = "jobs.db") -> JobDatabase:
    """Initialize the global job database instance."""
    global job_db
    job_db = JobDatabase(db_path)
    return job_db


def get_job_db() -> JobDatabase:
    """Get the global job database instance."""
    if job_db is None:
        raise RuntimeError("Job database not initialized. Call init_job_database() first.")
    return job_db