"""The shared flat processed-sample contract."""

import json
from pathlib import Path


def validate_manifest(manifest_path: Path) -> dict:
    """Validate that a processed manifest is a readable JSON array."""
    try:
        with open(manifest_path) as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        return {"valid": False, "count": 0, "warnings": [f"Cannot read: {exc}"]}
    if not isinstance(data, list):
        return {"valid": False, "count": 0, "warnings": ["Not a JSON array"]}
    return {"valid": True, "count": len(data), "warnings": []}


def load_processed_entries(data_root: Path | str, split: str) -> list[dict]:
    """Load all flat VQA entries for ``split`` with schema validation."""
    split_dir = Path(data_root) / split
    entries: list[dict] = []
    for path in sorted(split_dir.glob("*_VQA_*.json")):
        with open(path) as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Processed sample file is not a JSON array: {path}")
        entries.extend(payload)
    return entries


def load_benchmark_samples(data_root: Path | str, split: str) -> list[dict]:
    """Project processed entries into the baseline benchmark sample schema."""
    samples: list[dict] = []
    for entry in load_processed_entries(data_root, split):
        uav_paths = entry.get("uav_paths", {})
        if not uav_paths:
            continue
        uav_keys = sorted(uav_paths)
        if not uav_keys:
            continue
        distorted_uav_paths = entry.get("distorted_uav_paths", {})
        distortion_info = entry.get("distortion_info", {})
        first_key = uav_keys[0]
        samples.append({
            "path": str(distorted_uav_paths.get(first_key, "")),
            "ref_path": str(uav_paths.get(first_key, "")),
            "uav_paths": uav_paths,
            "uav_keys": uav_keys,
            "distorted_uav_paths": distorted_uav_paths,
            "task": entry.get("subtask_type", "unknown"),
            "distortion": distortion_info.get("type", ""),
            "category": distortion_info.get("category", "unknown"),
            "intensity_level": distortion_info.get("intensity", 0.5),
            "score": entry.get("cognitive_score") or 0.0,
            "ref_id": str(Path(str(uav_paths.get(first_key, ""))).stem),
            "sample_id": entry.get("sample_id", ""),
        })
    return samples
