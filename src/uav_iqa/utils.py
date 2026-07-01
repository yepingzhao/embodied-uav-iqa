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


def load_flat_samples(output_dir: Path, split: str) -> list[dict]:
    """Load processed flat JSON entries and convert to manifest-compatible format.

    The data synthesis pipeline writes flat entries (one per question × distortion)
    with keys: sample_id, uav_paths, distorted_uav_paths, distortion_info,
    subtask_type, cognitive_score, etc.

    Each returned dict has: path, ref_path, uav_paths, uav_keys,
    distorted_uav_paths, task, distortion, category, intensity_level,
    score, ref_id, sample_id.

    This provides backward compatibility for benchmarking/finetuning scripts
    that expected the old manifest.json format.
    """
    split_dir = output_dir / split
    samples: list[dict] = []

    for fpath in sorted(split_dir.glob("*_VQA_*.json")):
        with open(fpath) as f:
            entries = json.load(f)

        if not isinstance(entries, list):
            continue

        for entry in entries:
            uav_paths = entry.get("uav_paths", {})
            if not uav_paths:
                continue

            uav_keys = sorted(uav_paths.keys())
            if not uav_keys:
                continue

            distorted_uav_paths = entry.get("distorted_uav_paths", {})
            distortion_info = entry.get("distortion_info", {})

            score = entry.get("cognitive_score")
            if score is None:
                score = 0.0

            samples.append({
                "path": str(distorted_uav_paths.get(uav_keys[0], "")),
                "ref_path": str(uav_paths.get(uav_keys[0], "")),
                "uav_paths": uav_paths,
                "uav_keys": uav_keys,
                "distorted_uav_paths": distorted_uav_paths,
                "task": entry.get("subtask_type", "unknown"),
                "distortion": distortion_info.get("type", ""),
                "category": distortion_info.get("category", "unknown"),
                "intensity_level": distortion_info.get("intensity", 0.5),
                "score": score,
                "ref_id": str(Path(str(uav_paths.get(uav_keys[0], ""))).stem),
                "sample_id": entry.get("sample_id", ""),
            })

    return samples
