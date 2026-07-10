"""Writer: merges chunks and atomically writes output JSON files.

The writer is the **only** module that creates output files. It receives
completed chunks from the collector and writes them in the correct order.

Guarantees:
- Output file has the same number of records as the input file.
- Record order matches the original input order (restored via chunk_id/start/end).
- Atomic write (temp file + os.replace) prevents corruption on crash.
- Incremental: each chunk is persisted to .chunks/<stem>/ immediately so
  completed work survives a crash.  Final merge reads from disk.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from .storage import BaseStorage

_log = logging.getLogger(__name__)


class Writer:
    """Merges chunks and writes the final output file."""

    def __init__(self, storage: BaseStorage, output_dir: Path, *, model_name: str = "") -> None:
        self.storage = storage
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- chunk staging -------------------------------------------------------

    @property
    def _chunks_dir(self) -> Path:
        if self.model_name:
            return self.output_dir / "vlm" / self.model_name / ".chunks"
        return self.output_dir / ".chunks"

    def _chunk_subdir(self, source_file_path: Path) -> Path:
        return self._chunks_dir / source_file_path.stem

    def write_chunk(
        self,
        source_file_path: Path,
        chunk_id: int,
        outputs: list[dict[str, Any]],
    ) -> None:
        """Persist a single chunk to the staging directory.

        Called by the collector as soon as a chunk is complete.
        """
        subdir = self._chunk_subdir(source_file_path)
        subdir.mkdir(parents=True, exist_ok=True)
        chunk_path = subdir / f"chunk_{chunk_id:04d}.json"
        self.storage.write_output(chunk_path, outputs)

    def clean_chunks(self) -> None:
        """Remove all stale chunk files (called on fresh start)."""
        if self._chunks_dir.exists():
            shutil.rmtree(self._chunks_dir)
            _log.info("Cleaned stale chunks at %s", self._chunks_dir)

    # -- final output --------------------------------------------------------

    def write_file(
        self,
        source_file_path: Path,
        chunks: list[tuple[int, list[dict[str, Any]]]] | None = None,
        total_records: int = 0,
    ) -> Path:
        """Read chunks from disk and write the final output file.

        The *chunks* and *total_records* parameters are kept for
        backward-compatible testing; when None, chunks are read from the
        ``.chunks/`` staging directory.
        """
        if chunks is not None:
            # Legacy code path (tests, manual calls)
            all_records = self._flatten_chunks(chunks, total_records)
        else:
            all_records = self._read_chunks(source_file_path)

        # Preserve relative directory structure from input_dir
        try:
            relative_path = source_file_path.relative_to(self.storage.input_dir)
        except ValueError:
            # Paths with different roots (e.g. symlinks, mounts) fall back to filename
            relative_path = source_file_path.name
        output_path = self.output_dir / relative_path

        # Extract _model_details before writing main output
        model_details: dict[str, list[dict[str, Any]]] = {}
        for record in all_records:
            details = record.pop("_model_details", None)
            if details:
                for model_name, detail in details.items():
                    # Sidecar carries only sample_id for joining + model detail fields
                    detail_entry = {"sample_id": record.get("sample_id"), **detail}
                    model_details.setdefault(model_name, []).append(detail_entry)

        # Write model detail sidecar files
        for model_name, detail_records in model_details.items():
            detail_path = self.output_dir / "vlm" / model_name / relative_path
            self.storage.write_output(detail_path, detail_records)
            _log.info(
                "Wrote %s: %d model detail records",
                detail_path.name, len(detail_records),
            )

        self.storage.write_output(output_path, all_records)
        _log.info(
            "Wrote %s: %d records", output_path.name, len(all_records),
        )

        # Clean up the per-file chunk staging directory
        subdir = self._chunk_subdir(source_file_path)
        if subdir.exists():
            shutil.rmtree(subdir)

        return output_path

    def _flatten_chunks(
        self,
        chunks: list[tuple[int, list[dict[str, Any]]]],
        total_records: int,
    ) -> list[dict[str, Any]]:
        """Sort chunks by chunk_id and flatten into a single list."""
        chunks.sort(key=lambda c: c[0])
        all_records: list[dict[str, Any]] = []
        for _chunk_id, outputs in chunks:
            all_records.extend(outputs)
        if total_records and len(all_records) != total_records:
            raise ValueError(
                f"Record count mismatch: expected {total_records}, got {len(all_records)}"
            )
        return all_records

    def _read_chunks(self, source_file_path: Path) -> list[dict[str, Any]]:
        """Read all persisted chunk files from disk in order."""
        subdir = self._chunk_subdir(source_file_path)
        if not subdir.exists():
            raise FileNotFoundError(f"Chunk staging dir not found: {subdir}")
        chunk_files = sorted(subdir.glob("chunk_*.json"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {subdir}")
        all_records: list[dict[str, Any]] = []
        for cf in chunk_files:
            all_records.extend(json.loads(cf.read_text()))
        return all_records
