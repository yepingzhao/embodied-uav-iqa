"""Task repository: the only module that touches :class:`CheckpointStore`.

The repository exposes high-level operations needed by the Scheduler:

- ``create_tasks``  — generate and persist tasks for a list of files
- ``pending``       — return tasks ready to be processed
- ``mark_running``  — claim a task for execution
- ``mark_success``  — record successful completion
- ``mark_failed``   — record failure (with optional retry increment)
- ``file_complete`` — check whether all chunks of a file are done
- ``file_chunks``   — return all tasks belonging to a file (for validation)
"""

from __future__ import annotations

import logging
from pathlib import Path

from .checkpoint import CheckpointStore
from .types import FileRecord, Task, TaskStatus

_log = logging.getLogger(__name__)


class TaskRepository:
    """High-level task lifecycle manager backed by SQLite."""

    def __init__(self, store: CheckpointStore) -> None:
        self._store = store

    # -- task creation ------------------------------------------------------

    def create_tasks(
        self,
        files: list[FileRecord],
        chunk_size: int,
        *,
        max_entries: int = 0,
    ) -> int:
        """Generate and persist tasks for *files*.

        Args:
            files: Scanned file records.
            chunk_size: Number of records per chunk.
            max_entries: If > 0, cap the total records across all files
                (for testing). Records beyond the cap are dropped.

        Returns:
            Number of tasks created.
        """
        tasks_created = 0
        budget = max_entries if max_entries > 0 else None

        for frec in files:
            n = frec.record_count
            if budget is not None:
                n = min(n, budget)
                budget -= n
            if n <= 0:
                continue
            file_stem = Path(frec.name).stem
            for chunk_id, start in enumerate(range(0, n, chunk_size)):
                end = min(start + chunk_size, n)
                task = Task(
                    task_id=f"{file_stem}__{chunk_id:04d}",
                    file_path=frec.path,
                    chunk_id=chunk_id,
                    start=start,
                    end=end,
                )
                self._store.insert_task(task)
                tasks_created += 1
            if budget is not None and budget <= 0:
                break
        _log.info("Created %d tasks across %d files", tasks_created, len(files))
        return tasks_created

    # -- task queries -------------------------------------------------------

    def pending(self) -> list[Task]:
        """Return tasks that are PENDING or FAILED (resumable)."""
        return self._store.get_pending_tasks()

    def pending_or_running(self) -> list[Task]:
        """Return tasks that are PENDING, FAILED, or RUNNING.

        Used on resume to reclaim tasks left RUNNING by a crash.
        """
        return self._store.get_pending_or_running_tasks()

    def file_chunks(self, file_path: str) -> list[Task]:
        """Return all tasks belonging to *file_path*, ordered by chunk_id."""
        return self._store.get_tasks_for_file(file_path)

    def count_success(self) -> int:
        return self._store.count_by_status(TaskStatus.SUCCESS)

    def count_failed(self) -> int:
        return self._store.count_by_status(TaskStatus.FAILED)

    def total(self) -> int:
        return self._store.total_tasks()

    # -- status transitions -------------------------------------------------

    def mark_running(self, task_id: str) -> None:
        self._store.set_status(task_id, TaskStatus.RUNNING)

    def mark_success(self, task_id: str) -> None:
        self._store.set_status(task_id, TaskStatus.SUCCESS)

    def mark_failed(self, task_id: str, *, increment_retry: bool = True) -> None:
        self._store.set_status(
            task_id, TaskStatus.FAILED, increment_retry=increment_retry
        )

    # -- file-level queries -------------------------------------------------

    def file_complete(self, file_path: str) -> bool:
        """True iff every chunk of *file_path* has status SUCCESS."""
        chunks = self.file_chunks(file_path)
        if not chunks:
            return False
        return self._count_success_for_file(file_path) == len(chunks)

    def _count_success_for_file(self, file_path: str) -> int:
        """Direct SQLite query for per-file success count."""
        import sqlite3

        conn = sqlite3.connect(self._store.db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM tasks WHERE file_path = ? AND status = ?",
                (file_path, TaskStatus.SUCCESS.value),
            ).fetchone()
            return int(row[0])
        finally:
            conn.close()
