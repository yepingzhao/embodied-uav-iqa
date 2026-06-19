import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def setup_logging(name: str = None, level: int = logging.INFO):
    """Configure stdlib logging with uniform format for scripts.

    Returns a logger instance. Call once at module top-level:
        _log = setup_logging(__name__)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
    """Shuffle and split a list into {'train': [...], 'val': [...], 'test': [...]}.

    Uses numpy.RandomState for reproducible permutation.
    """
    if ratios is None:
        ratios = [0.8, 0.1, 0.1]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(samples))
    n = len(samples)
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))
    return {
        "train": [samples[i] for i in indices[:train_end]],
        "val": [samples[i] for i in indices[train_end:val_end]],
        "test": [samples[i] for i in indices[val_end:]],
    }


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
