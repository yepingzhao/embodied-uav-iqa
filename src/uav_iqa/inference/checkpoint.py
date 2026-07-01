"""SQLite-backed checkpoint store for task state.

The checkpoint database has a single table ``tasks``:

    task_id      TEXT PRIMARY KEY
    file_path    TEXT
    chunk_id     INTEGER
    start        INTEGER
    end          INTEGER
    status       TEXT   (pending | running | success | failed)
    retry_count  INTEGER
    updated_at   TEXT   (ISO 8601)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import Task, TaskStatus

_log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id      TEXT PRIMARY KEY,
    file_path    TEXT NOT NULL,
    chunk_id     INTEGER NOT NULL,
    start        INTEGER NOT NULL,
    end          INTEGER NOT NULL,
    status       TEXT NOT NULL,
    retry_count  INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_file   ON tasks(file_path);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointStore:
    """Low-level SQLite accessor for the ``tasks`` table."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- write operations ---------------------------------------------------

    def insert_task(self, task: Task) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO tasks
                    (task_id, file_path, chunk_id, start, end, status, retry_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    task.task_id,
                    str(task.file_path),
                    task.chunk_id,
                    task.start,
                    task.end,
                    TaskStatus.PENDING.value,
                    _now_iso(),
                ),
            )
            conn.commit()

    def set_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        increment_retry: bool = False,
    ) -> None:
        fields = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status.value, _now_iso()]
        if increment_retry:
            fields.append("retry_count = retry_count + 1")
        with self._connect() as conn:
            conn.execute(
                f"UPDATE tasks SET {', '.join(fields)} WHERE task_id = ?",
                (*params, task_id),
            )
            conn.commit()

    # -- read operations ----------------------------------------------------

    def get_pending_tasks(self) -> list[Task]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT task_id, file_path, chunk_id, start, end
                FROM tasks
                WHERE status IN (?, ?)
                ORDER BY file_path, chunk_id
                """,
                (TaskStatus.PENDING.value, TaskStatus.FAILED.value),
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def get_pending_or_running_tasks(self) -> list[Task]:
        """Return tasks that are PENDING, FAILED, or RUNNING.

        RUNNING is included because a crash may leave tasks in that state;
        on resume they should be re-processed.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT task_id, file_path, chunk_id, start, end
                FROM tasks
                WHERE status IN (?, ?, ?)
                ORDER BY file_path, chunk_id
                """,
                (
                    TaskStatus.PENDING.value,
                    TaskStatus.FAILED.value,
                    TaskStatus.RUNNING.value,
                ),
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def get_tasks_for_file(self, file_path: str) -> list[Task]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT task_id, file_path, chunk_id, start, end
                FROM tasks
                WHERE file_path = ?
                ORDER BY chunk_id
                """,
                (file_path,),
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def count_by_status(self, status: TaskStatus) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM tasks WHERE status = ?",
                (status.value,),
            ).fetchone()
        return int(row["c"])

    def total_tasks(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()
        return int(row["c"])

    # -- maintenance --------------------------------------------------------

    def clear(self) -> None:
        """Drop all rows. Mainly for tests."""
        with self._connect() as conn:
            conn.execute("DELETE FROM tasks")
            conn.commit()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Task:
        return Task(
            task_id=row["task_id"],
            file_path=Path(row["file_path"]),
            chunk_id=row["chunk_id"],
            start=row["start"],
            end=row["end"],
        )
