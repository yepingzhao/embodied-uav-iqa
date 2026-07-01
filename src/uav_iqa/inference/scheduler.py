"""Scheduler: scans input files, creates tasks, and feeds the task queue.

The scheduler is the **producer** side of the pipeline. It:

1. Scans the input directory for JSON files (via :class:`BaseStorage`).
2. Creates tasks in the :class:`TaskRepository` (chunked by ``chunk_size``).
3. On a fresh run, enqueues all pending tasks.
4. On resume, enqueues all pending/running/failed tasks.
5. Closes the task queue when done (sends sentinel to workers).

The scheduler does **not** perform inference or write output.
"""

from __future__ import annotations

import logging

from .queue import BaseQueue
from .repository import TaskRepository
from .storage import BaseStorage

_log = logging.getLogger(__name__)


class Scheduler:
    """Produces tasks and feeds them into the task queue."""

    def __init__(
        self,
        storage: BaseStorage,
        repo: TaskRepository,
        task_queue: BaseQueue,
        chunk_size: int = 256,
        max_entries: int = 0,
    ) -> None:
        self.storage = storage
        self.repo = repo
        self.task_queue = task_queue
        self.chunk_size = chunk_size
        self.max_entries = max_entries

    def run(self, *, resume: bool = False) -> int:
        """Scan files, create tasks, and enqueue them.

        Args:
            resume: If True, only enqueue tasks that are pending/failed/running
                (i.e. not yet successfully completed). If False, create fresh
                tasks (existing tasks in the DB are ignored via INSERT OR IGNORE).

        Returns:
            Number of tasks enqueued.
        """
        if not resume:
            files = self.storage.scan_files()
            _log.info("Scanned %d files from input dir", len(files))
            created = self.repo.create_tasks(files, self.chunk_size, max_entries=self.max_entries)
            _log.info("Created %d tasks (chunk_size=%d)", created, self.chunk_size)

        tasks = self.repo.pending() if not resume else self.repo.pending_or_running()
        _log.info("Enqueuing %d tasks (resume=%s)", len(tasks), resume)

        for task in tasks:
            self.task_queue.put(task)

        self.task_queue.close()
        _log.info("Task queue closed, scheduler done")
        return len(tasks)
