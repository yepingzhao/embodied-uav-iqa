"""Storage abstraction for reading input and writing output JSON files.

Phase 1 implements :class:`JsonStorage` which reads/writes lists of JSON
objects. The interface is designed so that future implementations
(JSONL, Parquet, Arrow) can be dropped in without changing callers.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .types import FileRecord

_log = logging.getLogger(__name__)


class BaseStorage(ABC):
    """Abstract storage interface."""

    @abstractmethod
    def scan_files(self) -> list[FileRecord]:
        """Scan the input directory and return metadata for each file."""

    @abstractmethod
    def count_records(self, file_path: Path) -> int:
        """Return the number of records in *file_path*."""

    @abstractmethod
    def load_chunk(
        self, file_path: Path, start: int, end: int
    ) -> list[dict[str, Any]]:
        """Load records ``[start, end)`` from *file_path*."""

    @abstractmethod
    def write_output(
        self, file_path: Path, records: list[dict[str, Any]]
    ) -> None:
        """Atomically write *records* to *file_path*."""


class JsonStorage(BaseStorage):
    """JSON-list storage. Each file is a top-level JSON array of objects."""

    def __init__(self, input_dir: Path, glob_pattern: str = "*_VQA_*.json") -> None:
        self.input_dir = Path(input_dir)
        self.glob_pattern = glob_pattern

    # -- public API ---------------------------------------------------------

    def scan_files(self) -> list[FileRecord]:
        files = sorted(self.input_dir.glob(self.glob_pattern))
        records: list[FileRecord] = []
        for fpath in files:
            records.append(
                FileRecord(
                    path=fpath.resolve(),
                    name=fpath.name,
                    record_count=self.count_records(fpath),
                )
            )
        return records

    def count_records(self, file_path: Path) -> int:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array in {file_path}, got {type(data).__name__}"
            )
        return len(data)

    def load_chunk(
        self, file_path: Path, start: int, end: int
    ) -> list[dict[str, Any]]:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array in {file_path}, got {type(data).__name__}"
            )
        if start < 0 or end > len(data) or start > end:
            raise IndexError(
                f"Chunk [{start}, {end}) out of range for {file_path} "
                f"with {len(data)} records"
            )
        return data[start:end]

    def write_output(
        self, file_path: Path, records: list[dict[str, Any]]
    ) -> None:
        """Atomically write *records* to *file_path*.

        Uses a temporary file in the same directory then ``os.replace`` to
        swap, guaranteeing that a crash never leaves a half-written file.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        dir_ = file_path.parent
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".tmp",
            dir=dir_,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            json.dump(records, tmp, ensure_ascii=False, indent=2)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, file_path)
        _log.debug("Wrote %d records to %s", len(records), file_path)
