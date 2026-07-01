"""Core type definitions for the inference framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(str, Enum):
    """Lifecycle status of a Task."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(slots=True)
class Task:
    """A unit of work: a contiguous chunk of records from one file.

    Attributes:
        task_id: Unique identifier (e.g. ``"{file_stem}__{chunk_id:04d}"``).
        file_path: Absolute path to the source JSON file.
        chunk_id: Sequential chunk index within the file (0-based).
        start: Inclusive start index of this chunk within the file's record list.
        end: Exclusive end index of this chunk within the file's record list.
    """

    task_id: str
    file_path: Path
    chunk_id: int
    start: int
    end: int

    @property
    def size(self) -> int:
        """Number of records in this chunk."""
        return self.end - self.start


@dataclass(slots=True)
class Result:
    """Output produced by a Worker for one Task.

    Attributes:
        task_id: Identifier matching the originating :class:`Task`.
        file_path: Path of the source file (for collector grouping).
        chunk_id: Chunk index within the file.
        outputs: List of output dicts, length == ``Task.size``.
        failed: List of original indices (relative to chunk start) that failed inference.
    """

    task_id: str
    file_path: Path
    chunk_id: int
    outputs: list[dict[str, Any]]
    failed: list[int] = field(default_factory=list)


@dataclass(slots=True)
class FileRecord:
    """Metadata for a scanned input file.

    Attributes:
        path: Absolute path to the JSON file.
        name: Filename (e.g. ``"Sim3_VQA_train.json"``).
        record_count: Total number of records in the file.
    """

    path: Path
    name: str
    record_count: int
