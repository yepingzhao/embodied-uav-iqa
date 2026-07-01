"""Validator: checks input/output consistency after a run.

Validates:
1. File count: output has the same number of files as input.
2. File names: output filenames match input filenames.
3. Record count: each output file has the same number of records as its input.
4. Record order: the ``sample_id`` field (if present) matches between
   input and output at each index.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

_log = logging.getLogger(__name__)


class Validator:
    """Post-run consistency checker."""

    def __init__(self, storage) -> None:  # BaseStorage
        self.storage = storage

    def validate(self, input_dir: Path, output_dir: Path) -> dict[str, object]:
        """Validate output against input.

        Returns:
            Dict with keys ``valid`` (bool), ``errors`` (list[str]),
            ``file_count`` (int), ``total_records`` (int).
        """
        errors: list[str] = []
        input_files = sorted(f.name for f in input_dir.glob("*.json"))
        output_files = sorted(f.name for f in output_dir.glob("*.json"))

        # 1. File count
        if len(input_files) != len(output_files):
            errors.append(
                f"File count mismatch: input={len(input_files)}, output={len(output_files)}"
            )

        # 2. File names
        missing = set(input_files) - set(output_files)
        extra = set(output_files) - set(input_files)
        if missing:
            errors.append(f"Missing output files: {sorted(missing)}")
        if extra:
            errors.append(f"Extra output files: {sorted(extra)}")

        # 3 & 4. Record count and order
        total_records = 0
        for name in input_files:
            if name not in output_files:
                continue
            in_path = input_dir / name
            out_path = output_dir / name
            in_data = json.loads(in_path.read_text())
            out_data = json.loads(out_path.read_text())
            total_records += len(out_data)

            if len(in_data) != len(out_data):
                errors.append(
                    f"{name}: record count mismatch (input={len(in_data)}, "
                    f"output={len(out_data)})"
                )
                continue

            # Check order via sample_id if present
            for i, (in_entry, out_entry) in enumerate(zip(in_data, out_data)):
                in_sid = in_entry.get("sample_id")
                out_sid = out_entry.get("sample_id")
                if in_sid is not None and out_sid is not None and in_sid != out_sid:
                    errors.append(
                        f"{name}: order mismatch at index {i} "
                        f"(input sample_id={in_sid}, output={out_sid})"
                    )
                    break

        valid = len(errors) == 0
        result = {
            "valid": valid,
            "errors": errors,
            "file_count": len(output_files),
            "total_records": total_records,
        }
        if valid:
            _log.info("Validation passed: %d files, %d records", len(output_files), total_records)
        else:
            _log.error("Validation failed: %d errors", len(errors))
            for e in errors:
                _log.error("  - %s", e)
        return result
