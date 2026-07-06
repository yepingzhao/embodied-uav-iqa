"""Collector: consumes results from the result queue and triggers file writes.

The collector is the **consumer** side of the pipeline. It:

1. Receives :class:`Result` items from the result queue.
2. Immediately persists each chunk via :class:`Writer.write_chunk`.
3. Marks tasks as SUCCESS in the repository.
4. When all chunks of a file are complete, triggers final merge via
   :class:`Writer.write_file`.
5. Releases memory immediately after each chunk is persisted.

The collector does **not** perform inference, manage workers, or parse
file paths beyond grouping by the ``file_path`` field carried in results.

It exits when it has received one sentinel (``None``) per worker from the
result queue.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .queue import BaseQueue
from .repository import TaskRepository
from .storage import BaseStorage
from .types import Result
from .writer import Writer

_log = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration string."""
    if seconds <= 0:
        return "0s"
    h, r = divmod(round(seconds), 3600)
    m, s = divmod(r, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s or not parts:
        parts.append(f"{s}s")
    return "".join(parts)


class Collector:
    """Consumes results, persists chunks incrementally, and triggers final writes."""

    def __init__(
        self,
        result_queue: BaseQueue,
        repo: TaskRepository,
        storage: BaseStorage,
        writer: Writer,
        num_workers: int = 1,
    ) -> None:
        self.result_queue = result_queue
        self.repo = repo
        self.storage = storage
        self.writer = writer
        self.num_workers = num_workers

    def run(self) -> dict[str, int]:
        """Consume results until all workers have sent sentinels.

        Returns:
            Dict mapping file name → number of records written.
        """
        # file_key → total expected chunks (synced from repo on first encounter)
        file_total_chunks: dict[str, int] = {}
        # file_key → completed chunk count
        file_done: dict[str, int] = defaultdict(int)
        sentinels_received = 0
        tasks_done = 0
        samples_done = 0
        total_tasks = self.repo.total()
        try:
            scan_results = self.storage.scan_files()
            total_files = len(scan_results)
            total_samples = sum(
                self.storage.count_records(fr.path)
                for fr in scan_results
            )
        except Exception:
            total_files = 0
            total_samples = 0
        files_written: dict[str, int] = {}
        last_file: str = "-"
        t_start = time.time()
        next_log_samples = 100

        while sentinels_received < self.num_workers:
            result = self.result_queue.get()
            if result is None:
                sentinels_received += 1
                continue

            self._handle_result(result, file_total_chunks, file_done)
            tasks_done += 1
            samples_done += len(result.outputs)
            if len(file_total_chunks) > total_files:
                total_files = len(file_total_chunks)

            # Check if the file is now complete
            file_key = str(result.file_path)
            total = file_total_chunks.get(file_key, 0)
            if total > 0 and file_done[file_key] >= total:
                written = self._write_file(file_key)
                if written is not None:
                    files_written[result.file_path.name] = written
                    last_file = result.file_path.stem
                # Reset tracking for this file
                file_total_chunks.pop(file_key, None)
                file_done.pop(file_key, None)

            if samples_done >= next_log_samples or tasks_done == total_tasks:
                elapsed = time.time() - t_start
                rate = samples_done / elapsed if elapsed > 0 else 0
                eta = elapsed / tasks_done * (total_tasks - tasks_done) if tasks_done > 0 else 0
                file_progress = ", ".join(
                    f"{Path(k).stem}: {v}/{file_total_chunks.get(k, '?')}"
                    for k, v in sorted(file_done.items())
                )
                _log.info(
                    "Progress: %d/%d samples (%.1f/s), %d/%d tasks | "
                    "files: %d/%d [%s] | last: %s | %s | ETA %s",
                    samples_done, total_samples, rate,
                    tasks_done, total_tasks,
                    len(files_written), total_files, file_progress,
                    last_file,
                    _fmt_duration(elapsed), _fmt_duration(eta),
                )
                next_log_samples = samples_done + 100

        # Write any remaining files (in case sentinels arrived before last chunks)
        for file_key in list(file_done.keys()):
            written = self._write_file(file_key)
            if written is not None:
                files_written[Path(file_key).name] = written

        _log.info("Collector done: wrote %d files", len(files_written))
        return files_written

    def _handle_result(
        self,
        result: Result,
        file_total_chunks: dict[str, int],
        file_done: dict[str, int],
    ) -> None:
        """Persist chunk output immediately and mark the task as successful."""
        file_key = str(result.file_path)

        # Persist chunk to disk
        self.writer.write_chunk(result.file_path, result.chunk_id, result.outputs)

        # Track total chunk count on first encounter
        if file_key not in file_total_chunks:
            try:
                n_chunks = self.repo.file_chunks(file_key)
                file_total_chunks[file_key] = len(n_chunks) if n_chunks else 0
            except Exception:
                _log.warning("Could not count chunks for %s", file_key, exc_info=True)
                file_total_chunks[file_key] = 0

        file_done[file_key] += 1
        self.repo.mark_success(result.task_id)

    def _write_file(self, file_key: str) -> int | None:
        """Merge persisted chunks into final output. Returns record count or None."""
        try:
            output_path = self.writer.write_file(Path(file_key))
            return self.storage.count_records(output_path)
        except Exception:
            _log.error("Failed to write %s", file_key, exc_info=True)
            return None
