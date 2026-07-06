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
from typing import Any

_log = logging.getLogger(__name__)


class Validator:
    """Post-run consistency checker."""

    # Directories under output_dir to skip during validation (sidecar data)
    _SIDECAR_EXCLUDE: set[str] = {"vlm"}

    def __init__(self, storage) -> None:  # BaseStorage
        self.storage = storage

    def validate(self, input_dir: Path, output_dir: Path) -> dict[str, Any]:
        """Validate output against input.

        Returns:
            Dict with keys ``valid`` (bool), ``errors`` (list[str]),
            ``file_count`` (int), ``total_records`` (int).
        """
        errors: list[str] = []

        # Scan input files relative to input_dir (use same glob as storage)
        glob_pattern = getattr(self.storage, "glob_pattern", "*_VQA_*.json")
        input_rel = sorted(
            f.relative_to(input_dir)
            for f in input_dir.rglob(glob_pattern)
        )

        # Scan output files relative to output_dir, skipping vlm/ sidecar dir
        output_rel = sorted(
            f.relative_to(output_dir)
            for f in output_dir.rglob(glob_pattern)
            if not any(part in self._SIDECAR_EXCLUDE for part in f.relative_to(output_dir).parts)
        )

        input_names = {str(r) for r in input_rel}
        output_names = {str(r) for r in output_rel}

        # 1. File count
        if len(input_names) != len(output_names):
            errors.append(
                f"File count mismatch: input={len(input_names)}, output={len(output_names)}"
            )

        # 2. File names
        missing = input_names - output_names
        extra = output_names - input_names
        if missing:
            errors.append(f"Missing output files: {sorted(missing)}")
        if extra:
            errors.append(f"Extra output files: {sorted(extra)}")

        # 3 & 4. Record count and order
        total_records = 0
        for rel in sorted(input_rel):
            rel_str = str(rel)
            if rel_str not in output_names:
                continue
            in_data = json.loads((input_dir / rel).read_text())
            out_data = json.loads((output_dir / rel).read_text())
            total_records += len(out_data)

            if len(in_data) != len(out_data):
                errors.append(
                    f"{rel_str}: record count mismatch (input={len(in_data)}, "
                    f"output={len(out_data)})"
                )
                continue

            # Check order via sample_id if present
            for i, (in_entry, out_entry) in enumerate(zip(in_data, out_data)):
                in_sid = in_entry.get("sample_id")
                out_sid = out_entry.get("sample_id")
                if in_sid is not None and out_sid is not None and in_sid != out_sid:
                    errors.append(
                        f"{rel_str}: order mismatch at index {i} "
                        f"(input sample_id={in_sid}, output={out_sid})"
                    )
                    break

        valid = len(errors) == 0
        result = {
            "valid": valid,
            "errors": errors,
            "file_count": len(output_names),
            "total_records": total_records,
        }
        if valid:
            _log.info("Validation passed: %d files, %d records", len(output_names), total_records)
        else:
            _log.error("Validation failed: %d errors", len(errors))
            for e in errors:
                _log.error("  - %s", e)
        return result
