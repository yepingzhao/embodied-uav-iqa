"""Collector: consumes results from the result queue and triggers file writes.

The collector is the **consumer** side of the pipeline. It:

1. Receives :class:`Result` items from the result queue.
2. Groups results by file path.
3. Marks tasks as SUCCESS in the repository.
4. When all chunks of a file are complete, hands them to the :class:`Writer`.
5. Releases memory immediately after writing.

The collector does **not** perform inference, manage workers, or parse
file paths beyond grouping by the ``file_path`` field carried in results.

It exits when it has received one sentinel (``None``) per worker from the
result queue.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from .queue import BaseQueue
from .repository import TaskRepository
from .storage import BaseStorage
from .types import Result
from .writer import Writer

_log = logging.getLogger(__name__)


class Collector:
    """Consumes results, groups by file, and triggers writes."""

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
        # file_path -> {chunk_id: outputs}
        file_chunks: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
        # file_path -> total expected records
        file_totals: dict[str, int] = {}
        sentinels_received = 0
        files_written: dict[str, int] = {}

        while sentinels_received < self.num_workers:
            result = self.result_queue.get()
            if result is None:
                sentinels_received += 1
                continue

            self._handle_result(result, file_chunks, file_totals)

            # Check if the file is now complete
            file_key = str(result.file_path)
            if self.repo.file_complete(file_key):
                written = self._write_file(file_key, file_chunks[file_key], file_totals)
                if written is not None:
                    files_written[result.file_path.name] = written
                # Release memory
                file_chunks.pop(file_key, None)
                file_totals.pop(file_key, None)

        # Write any remaining files (in case sentinels arrived before last chunks)
        for file_key, chunks in file_chunks.items():
            written = self._write_file(file_key, chunks, file_totals)
            if written is not None:
                files_written[Path(file_key).name] = written

        _log.info("Collector done: wrote %d files", len(files_written))
        return files_written

    def _handle_result(
        self,
        result: Result,
        file_chunks: dict[str, dict[int, list[dict[str, Any]]]],
        file_totals: dict[str, int],
    ) -> None:
        """Store result outputs and mark the task as successful."""
        file_key = str(result.file_path)
        file_chunks[file_key][result.chunk_id] = result.outputs

        # Cache the total record count on first encounter
        if file_key not in file_totals:
            try:
                file_totals[file_key] = self.storage.count_records(result.file_path)
            except Exception:
                _log.warning("Could not count records for %s", result.file_path, exc_info=True)
                file_totals[file_key] = sum(
                    len(o) for o in file_chunks[file_key].values()
                )

        self.repo.mark_success(result.task_id)

    def _write_file(
        self,
        file_key: str,
        chunks: dict[int, list[dict[str, Any]]],
        file_totals: dict[str, int],
    ) -> int | None:
        """Write a completed file. Returns record count or None on failure."""
        if not chunks:
            return None

        chunk_list = sorted(chunks.items())  # sort by chunk_id
        total = file_totals.get(file_key, sum(len(o) for _, o in chunk_list))

        try:
            self.writer.write_file(Path(file_key), chunk_list, total)
            return total
        except Exception:
            _log.error("Failed to write %s", file_key, exc_info=True)
            return None
