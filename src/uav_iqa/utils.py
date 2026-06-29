import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


_THIRD_PARTY_LOGGERS = (
    "transformers",
    "huggingface_hub",
    "datasets",
    "diffusers",
    "accelerate",
    "peft",
    "safetensors",
    "filelock",
    "torch.distributed",
)


def setup_logging(name: str = None, level: int = logging.INFO):
    """Configure stdlib logging with uniform format for scripts.

    Suppresses verbose INFO logs from third-party libraries (transformers,
    huggingface_hub, etc.) while keeping application logs at the requested
    level.  Returns a logger instance.  Call once at module top-level::

        _log = setup_logging(__name__)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Silence noisy third-party loggers below WARNING so they don't flood
    # the terminal during model loading / dataset streaming.
    for name_ in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name_).setLevel(logging.WARNING)

    return logging.getLogger(name)


def find_images(
    image_dir: Path,
    exts: Optional[set] = None,
    exclude_dirs: Optional[set] = None,
) -> list:
    """Find all image files in a directory tree, optionally excluding subdirs.

    Args:
        image_dir: Root directory to scan recursively.
        exts: Allowed file extensions (default: .jpg, .jpeg, .png).
        exclude_dirs: Subdirectory names to skip (e.g. {'loss', 'noise'}).
    """
    if exts is None:
        exts = IMAGE_EXTS
    images = []
    for p in sorted(image_dir.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        if exclude_dirs:
            parts = set(p.relative_to(image_dir).parts)
            if parts & exclude_dirs:
                continue
        images.append(p)
    return images


def load_image_tensor(path: Path, image_size: int = 256) -> torch.Tensor:
    """Load an image and convert to (C, H, W) float tensor in [0, 1].

    Used by both UAVIQADataset and benchmark scripts.
    """
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def count_parameters(model) -> tuple:
    """Return (total_params, trainable_params) for the given model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_task_map(path: Optional[str]) -> Optional[dict]:
    """Load a JSON task-map file, returning the parsed dict or None."""
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def split_samples(
    samples: list,
    ratios: list = None,
    seed: int = 42,
) -> dict:
    """Shuffle and split a list into train/test.

    Uses np.random.RandomState for reproducible permutation.
    Default ratios: [0.8, 0.2].
    """
    if ratios is None:
        ratios = [0.8, 0.2]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(samples))
    n = len(samples)
    if len(ratios) != 2:
        raise ValueError(f"Expected 2 ratios (train, test), got {len(ratios)}")
    train_end = int(n * ratios[0])
    return {
        "train": [samples[i] for i in indices[:train_end]],
        "test": [samples[i] for i in indices[train_end:]],
    }


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load manifest entries from a JSON file.

    Returns the parsed list of entries. The caller is responsible for
    checking that the file exists before calling.
    """
    with open(manifest_path) as f:
        return json.load(f)


def write_manifest(entries: list, manifest_path: Path, _log=None) -> int:
    """Write manifest JSON, validate schema, and return entry count.

    Creates parent directories as needed. Logs summary via `_log` if provided.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(entries, f, indent=2)

    from uav_iqa.dataset import validate_manifest

    diag = validate_manifest(manifest_path)
    if not diag.get("valid", True) and _log:
        _log.warning("  Manifest validation warnings: %s", diag)

    return len(entries)


def load_flat_samples(output_dir: Path, split: str) -> list[dict]:
    """Load grouped processed JSONs and flatten to old manifest-compatible format.

    Each returned dict has: path, ref_path, task (subtask_name), distortion,
    intensity_level, score (cognitive_score), ref_id.

    This provides backward compatibility for benchmarking/finetuning scripts
    that expected the old manifest.json format.
    """
    import json as _json

    split_dir = output_dir / split
    samples: list[dict] = []

    for fpath in sorted(split_dir.glob("*_VQA_*.json")):
        with open(fpath) as f:
            groups = _json.load(f)

        for group in groups:
            uav_paths = group.get("uav_paths", {})
            uav_keys = group.get("uav_keys", [])
            distortions = group.get("distortions", {})
            vqa_entries = group.get("vqa_entries", [])

            if not uav_paths or not uav_keys:
                continue

            for dist_key, dist_info in distortions.items():
                for entry in vqa_entries:
                    score = entry.get("cognitive_score", 0.0)
                    subtask_name = entry.get("subtask_name", "unknown")

                    samples.append({
                        "path": str(dist_info.get("distorted_uav_paths", {}).get(
                            uav_keys[0], ""
                        )),
                        "ref_path": str(uav_paths.get(uav_keys[0], "")),
                        "uav_paths": uav_paths,
                        "uav_keys": uav_keys,
                        "distorted_uav_paths": dist_info.get("distorted_uav_paths", {}),
                        "task": subtask_name,
                        "distortion": dist_info.get("type", ""),
                        "category": dist_info.get("category", "unknown"),
                        "intensity_level": dist_info.get("intensity", 0.5),
                        "score": score,
                        "ref_id": str(Path(str(uav_paths.get(uav_keys[0], ""))).stem),
                        "sample_id": dist_info.get("sample_id", ""),
                    })

    return samples
