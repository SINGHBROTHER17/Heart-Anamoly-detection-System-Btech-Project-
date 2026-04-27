"""
Report storage — SQLite backend with a Firestore-compatible interface.

We use SQLite here because:
  - Zero external dependencies for local dev and Cloud Run testing
  - One file ships with the container; easy to swap for Firestore in prod
  - Handles the read-mostly workload (store once, read on GET /report/{id})

To migrate to Firestore/Postgres, implement the same 4 methods on a new
class and wire it in via the `get_storage()` dependency.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class SQLiteStorage:
    """Thread-safe SQLite-backed report storage."""

    def __init__(self, db_path: str = "./data/reports.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS reports (
                    report_id    TEXT PRIMARY KEY,
                    patient_id   TEXT,
                    timestamp    TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS reports_timestamp_idx
                    ON reports(timestamp DESC);

                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id   TEXT PRIMARY KEY,
                    report_id     TEXT NOT NULL,
                    clinician_id  TEXT,
                    timestamp     TEXT NOT NULL,
                    payload_json  TEXT NOT NULL,
                    FOREIGN KEY (report_id) REFERENCES reports(report_id)
                );
                CREATE INDEX IF NOT EXISTS feedback_report_idx
                    ON feedback(report_id);
            """)

    # ---- Reports -------------------------------------------------------

    def save_report(self, report_id: str, report: dict[str, Any],
                    patient_id: Optional[str] = None) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reports(report_id, patient_id, timestamp, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    report_id,
                    patient_id,
                    report.get("timestamp", datetime.utcnow().isoformat()),
                    json.dumps(report, default=str),
                ),
            )

    def get_report(self, report_id: str) -> Optional[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM reports WHERE report_id = ?",
                (report_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])

    def list_recent_reports(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM reports ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    # ---- Feedback ------------------------------------------------------

    def save_feedback(self, feedback_id: str, feedback: dict[str, Any]) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO feedback(feedback_id, report_id, clinician_id, timestamp, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    feedback_id,
                    feedback["report_id"],
                    feedback.get("clinician_id"),
                    feedback.get("timestamp", datetime.utcnow().isoformat()),
                    json.dumps(feedback, default=str),
                ),
            )


_storage: Optional[SQLiteStorage] = None


def get_storage() -> SQLiteStorage:
    """FastAPI dependency — returns the process-wide storage instance."""
    global _storage
    if _storage is None:
        _storage = SQLiteStorage(
            db_path=os.environ.get("ECG_DB_PATH", "./data/reports.db")
        )
    return _storage
