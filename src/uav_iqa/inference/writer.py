"""Writer: merges chunks and atomically writes output JSON files.

The writer is the **only** module that creates output files. It receives
completed chunks from the collector and writes them in the correct order.

Guarantees:
- Output file has the same number of records as the input file.
- Record order matches the original input order (restored via chunk_id/start/end).
- Atomic write (temp file + os.replace) prevents corruption on crash.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .storage import BaseStorage

_log = logging.getLogger(__name__)


class Writer:
    """Merges chunks and writes the final output file."""

    def __init__(self, storage: BaseStorage, output_dir: Path) -> None:
        self.storage = storage
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_file(
        self,
        source_file_path: Path,
        chunks: list[tuple[int, list[dict[str, Any]]]],
        total_records: int,
    ) -> Path:
        """Assemble chunks and write the output file.

        Args:
            source_file_path: Path to the original input file (for naming).
            chunks: List of ``(chunk_id, outputs)`` tuples.
            total_records: Expected number of records (for validation).

        Returns:
            Path to the written output file.

        Raises:
            ValueError: If the assembled record count doesn't match.
        """
        # Sort chunks by chunk_id to restore original order
        chunks.sort(key=lambda c: c[0])

        all_records: list[dict[str, Any]] = []
        for _chunk_id, outputs in chunks:
            all_records.extend(outputs)

        if len(all_records) != total_records:
            raise ValueError(
                f"Record count mismatch for {source_file_path.name}: "
                f"expected {total_records}, got {len(all_records)}"
            )

        output_path = self.output_dir / source_file_path.name
        self.storage.write_output(output_path, all_records)
        _log.info(
            "Wrote %s: %d records from %d chunks",
            output_path.name, len(all_records), len(chunks),
        )
        return output_path
